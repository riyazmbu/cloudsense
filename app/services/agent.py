from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


AGENT_SYSTEM_PROMPT = """
You are CloudSense AI, an enterprise cloud cost intelligence agent.

You have access to read-only CloudSense tools. Use tools to obtain evidence before
answering questions about cost, resources, utilization, anomalies, policies, or savings.

Rules:
- Never invent AWS costs, resources, utilization, savings, or policy requirements.
- Prefer actual tool output over assumptions.
- Resource-level monetary savings are valid only when a resource has an actual monthly
  cost supplied by CloudSense. If it is unavailable, explicitly say so.
- CloudSense savings percentages are deterministic estimates, not AWS price quotes.
- Never claim an AWS action was executed. The current agent is recommendation-only.
- If a policy source is returned, cite the exact evidence ID such as [rightsizing_policy.txt#0]. Never cite a source that was not returned by a tool.
- Explain findings in simple business language.
- Give a concise answer with: finding, evidence, recommendation, and savings when available. When policy evidence is used, include an Evidence section with the exact [filename#chunk] citations returned by search_policies.
- If the user asks a broad question, call the minimum useful set of tools needed to answer it.
"""


class CloudSenseAgent:
    """CloudSense tool-calling agent with an LLM planner and safe local fallback.

    The previous MVP used keyword matching. This version lets the LLM select and sequence
    read-only CloudSense tools using the OpenAI Responses API. If no API key/SDK is
    available, the deterministic planner remains available so the backend still works.
    """

    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self.tools = tools

    # ------------------------------------------------------------------
    # Deterministic fallback planner
    # ------------------------------------------------------------------

    def plan(self, question: str) -> List[str]:
        q = question.lower()
        plan: List[str] = []

        if any(x in q for x in ["bill", "cost", "spend", "expensive"]):
            plan.append("get_cost_summary")

        if any(x in q for x in [
            "resource", "server", "instance", "cpu", "memory",
            "utilization", "underutilized"
        ]):
            plan.append("get_resource_metrics")

        if any(x in q for x in [
            "save", "saving", "optimize", "optimization", "reduce"
        ]):
            plan.append("calculate_savings")

        if any(x in q for x in [
            "policy", "production", "approval", "rightsiz", "schedule", "safe"
        ]):
            plan.append("search_policies")

        if any(x in q for x in [
            "anomaly", "unusual", "spike", "forecast", "predict"
        ]):
            plan.append("analyze_ml")

        if any(x in q for x in [
            "aws resource", "ec2", "ebs", "rds", "s3", "lambda",
            "nat gateway", "cloudfront", "rightsizing", "underutilized",
            "unused", "idle", "resource intelligence"
        ]):
            plan.append("get_aws_resource_intelligence")

        if not plan:
            plan = ["get_cost_summary", "get_resource_metrics"]

        return list(dict.fromkeys(plan))

    def fallback_answer(self, question: str, outputs: Dict[str, Any]) -> str:
        cost = outputs.get("get_cost_summary") or {}
        resources = outputs.get("get_resource_metrics") or {}
        savings = outputs.get("calculate_savings") or {}
        policy = outputs.get("search_policies") or {}
        ml = outputs.get("analyze_ml") or {}
        aws = outputs.get("get_aws_resource_intelligence") or {}
        parts = []

        if "get_cost_summary" in outputs:
            parts.append(f"Finding: Current monthly cost is ₹{float(cost.get('current_monthly_cost', 0) or 0):,.0f}.")

        candidates = resources.get("optimization_candidates")
        if candidates is not None:
            parts.append(f"Evidence: CloudSense identified {candidates} optimization candidate(s) from the available resource data.")

        if aws.get("recommendations"):
            parts.append(f"Evidence: AWS resource intelligence returned {len(aws['recommendations'])} recommendation(s).")

        if savings.get("estimated_monthly_saving") is not None:
            parts.append(f"Savings: Estimated monthly savings are ₹{float(savings.get('estimated_monthly_saving', 0) or 0):,.0f} under CloudSense rules.")

        if policy.get("results"):
            cites = [f"[{x.get('source')}#{x.get('chunk_id')}]" for x in policy["results"] if x.get("source") is not None and x.get("chunk_id") is not None]
            if cites:
                parts.append("Evidence: Relevant policy guidance was found in " + ", ".join(cites) + ".")

        if ml.get("anomaly_count") is not None:
            parts.append(f"ML check: {ml.get('anomaly_count', 0)} cost anomaly/anomalies were detected in the available history.")

        if not parts:
            parts.append("I could not find enough CloudSense data to answer that yet. Upload billing data or connect AWS resources first.")
        parts.append("Recommendation: Review the evidence before making any cloud changes; CloudSense does not execute changes automatically.")
        return "\n\n".join(parts)

    def run_fallback(self, question: str) -> dict:
        selected = self.plan(question)
        outputs: Dict[str, Any] = {}

        calls = []
        for name in selected:
            tool = self.tools.get(name)
            started = time.perf_counter()
            if tool:
                try:
                    outputs[name] = tool(question) if name == "search_policies" else tool()
                    calls.append({"name": name, "status": "completed", "duration_ms": round((time.perf_counter()-started)*1000, 1)})
                except Exception as exc:
                    outputs[name] = {"error": str(exc)}
                    calls.append({"name": name, "status": "error", "duration_ms": round((time.perf_counter()-started)*1000, 1), "error": str(exc)})

        return {
            "question": question,
            "agent_mode": "deterministic_fallback",
            "tool_plan": selected,
            "tool_calls": calls,
            "tool_outputs": outputs,
            "answer": self.fallback_answer(question, outputs),
            "observability": {"total_duration_ms": round(sum(c.get("duration_ms", 0) for c in calls), 1), "rounds": 1},
        }

    # ------------------------------------------------------------------
    # OpenAI function tool definitions
    # ------------------------------------------------------------------

    @staticmethod
    def tool_definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "get_cost_summary",
                "description": "Get the current CloudSense billing summary and run-rate.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_resource_metrics",
                "description": "Get normalized resource metrics and optimization candidates from CloudSense billing records.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "calculate_savings",
                "description": "Calculate deterministic CloudSense savings from the normalized optimization candidates.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "search_policies",
                "description": "Semantically search CloudSense governance and FinOps policies relevant to the user's question and return evidence IDs, scores, and source text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"}
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "analyze_ml",
                "description": "Run CloudSense ML analysis for cost anomalies and a short forecast.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_aws_resource_intelligence",
                "description": "Analyze collected AWS resources using state, utilization, resource-level cost when available, and deterministic optimization rules.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "strict": True,
            },
        ]

    # ------------------------------------------------------------------
    # LLM agent loop
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or OpenAI is None:
            return self.run_fallback(question)

        model = os.getenv("CLOUDSENSE_AI_MODEL", "gpt-5.6")
        max_rounds = int(os.getenv("CLOUDSENSE_AGENT_MAX_ROUNDS", "6"))
        client = OpenAI(api_key=api_key)
        tools = self.tool_definitions()
        input_items: list[Any] = [{
            "role": "user",
            "content": [{"type": "input_text", "text": question}],
        }]
        called_tools: list[dict[str, Any]] = []
        tool_outputs: dict[str, Any] = {}
        agent_started = time.perf_counter()
        rounds_used = 0

        for round_no in range(max_rounds):
            rounds_used = round_no + 1
            response = client.responses.create(
                model=model,
                instructions=AGENT_SYSTEM_PROMPT,
                tools=tools,
                tool_choice="auto",
                input=input_items,
            )
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]

            if not function_calls:
                return {
                    "question": question,
                    "agent_mode": "llm_tool_calling",
                    "model": model,
                    "tool_plan": [x["name"] for x in called_tools],
                    "tool_calls": called_tools,
                    "tool_outputs": tool_outputs,
                    "answer": response.output_text,
                    "observability": {
                        "total_duration_ms": round((time.perf_counter() - agent_started) * 1000, 1),
                        "rounds": rounds_used,
                    },
                }

            input_items.extend(response.output)
            for call in function_calls:
                name = getattr(call, "name", "")
                call_id = getattr(call, "call_id", "")
                raw_args = getattr(call, "arguments", "{}") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                started = time.perf_counter()
                tool = self.tools.get(name)
                if tool is None:
                    result: Any = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = tool(args.get("question", question)) if name == "search_policies" else tool()
                    except Exception as exc:
                        result = {"error": str(exc)}

                status = "error" if isinstance(result, dict) and "error" in result else "completed"
                duration_ms = round((time.perf_counter() - started) * 1000, 1)
                called_tools.append({
                    "name": name,
                    "arguments": args,
                    "status": status,
                    "duration_ms": duration_ms,
                })
                tool_outputs[name] = result
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result, default=str),
                })

        return {
            "question": question,
            "agent_mode": "llm_tool_calling",
            "model": model,
            "tool_plan": [x["name"] for x in called_tools],
            "tool_calls": called_tools,
            "tool_outputs": tool_outputs,
            "answer": "I gathered the available CloudSense evidence, but the agent reached its tool-call limit before producing a final response.",
            "observability": {
                "total_duration_ms": round((time.perf_counter() - agent_started) * 1000, 1),
                "rounds": rounds_used,
            },
        }

