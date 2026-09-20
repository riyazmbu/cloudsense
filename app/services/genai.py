import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    Groq = None


SYSTEM_PROMPT = """
You are CloudSense AI, a cloud cost optimization copilot.

Explain CloudSense findings in simple business language. Never dump raw JSON or a long technical report.

Rules:
- Use only supplied CloudSense data and supplied RAG knowledge.
- Never invent resources, costs, metrics or savings.
- Use deterministic savings values supplied by CloudSense.
- For resource-specific questions, prioritize the supplied resource data.
- Use RAG knowledge to explain the AWS/service-specific reasoning.
- Do not claim that any AWS action has been executed.
- Recommendations require validation and human approval.

For a resource question, explain:

1. What is happening
2. Evidence
3. Why it matters
4. Recommended action
5. Estimated savings, if available
6. Validation required

Be concise and actionable. Use a clear structure with a short headline, key numbers, plain-English explanation, recommended next steps, and a brief validation note. Bold important values using Markdown **bold**. Prefer bullets over dense paragraphs.
"""


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_resource_id(question: str) -> Optional[str]:

    patterns = [
        r"\bi-[a-zA-Z0-9-]+\b",
        r"\bvol-[a-zA-Z0-9-]+\b",
        r"\bdb-[a-zA-Z0-9-]+\b",
        r"\bnat-[a-zA-Z0-9-]+\b",
        r"\bfn-[a-zA-Z0-9-]+\b",
        r"\bbucket-[a-zA-Z0-9-]+\b",
        r"\bdist-[a-zA-Z0-9-]+\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)

        if match:
            return match.group(0)

    return None


def find_resource(resource_id, records, pricing):

    target = resource_id.lower()

    for record in records:

        rid = str(
            record.get("resource_id") or ""
        ).lower()

        if rid == target:
            return record

    for candidate in pricing:

        rid = str(
            candidate.get("resource_id") or ""
        ).lower()

        if rid == target:
            return candidate

    return None


def build_resource_context(resource_id, records, pricing):

    if not resource_id:
        return None

    resource = find_resource(
        resource_id,
        records,
        pricing
    )

    if not resource:
        return None

    result = {
        "resource_id": resource.get("resource_id"),
        "service": resource.get("service"),
        "resource_name": resource.get("resource_name"),
        "region": resource.get("region"),
        "environment": resource.get("environment"),
        "team": resource.get("team"),
        "state": resource.get("state"),

        "cpu_utilization": safe_float(
            resource.get("cpu_utilization")
        ),

        "memory_utilization": safe_float(
            resource.get("memory_utilization")
        ),

        "monthly_cost": safe_float(
            resource.get("monthly_cost")
        ),

        "cost_currency": resource.get(
            "cost_currency"
        ),
    }

    # Find recommendation
    for candidate in pricing:

        rid = str(
            candidate.get("resource_id") or ""
        ).lower()

        if rid == resource_id.lower():

            result["optimization_candidate"] = (
                candidate.get(
                    "optimization_candidate"
                )
            )

            result["recommendation_type"] = (
                candidate.get(
                    "recommendation_type"
                )
            )

            result["estimated_monthly_saving"] = (
                safe_float(
                    candidate.get(
                        "estimated_monthly_saving"
                    )
                )
            )

            result["recommendation_reason"] = (
                candidate.get("reason")
                or candidate.get(
                    "recommendation_reason"
                )
            )

            result["evidence"] = (
                candidate.get("evidence")
            )

            break

    return result


# ---------------------------------------------------------
# RAG-aware deterministic fallback
# ---------------------------------------------------------

def deterministic_resource_answer(
    question,
    resource,
    rag_context=""
):
    """Readable fallback used when no LLM quota/key is available."""
    if not resource:
        return None

    rid = resource.get("resource_id", "this resource")
    service = resource.get("service", "unknown service")
    cpu = resource.get("cpu_utilization")
    memory = resource.get("memory_utilization")
    cost = resource.get("monthly_cost")
    currency = resource.get("cost_currency") or "USD"
    saving = resource.get("estimated_monthly_saving")
    recommendation = resource.get("recommendation_type")

    lines = [f"### 💡 {rid} — Optimization opportunity", ""]
    lines.append(f"**Service:** {service}")
    if resource.get("region"): lines.append(f"**Region:** {resource['region']}")
    if resource.get("environment"): lines.append(f"**Environment:** {resource['environment']}")
    if resource.get("team"): lines.append(f"**Owner / team:** {resource['team']}")
    if cost is not None: lines.append(f"**Current cost:** {currency} {cost:,.2f} / month")
    lines.append("")
    lines.append("### What the data shows")
    if cpu is not None: lines.append(f"- CPU utilization: **{cpu:.1f}%**")
    if memory is not None: lines.append(f"- Memory utilization: **{memory:.1f}%**")
    if resource.get("state"): lines.append(f"- State: **{resource['state']}**")
    lines.append(f"- CloudSense status: **optimization candidate**")

    service_lower=str(service).lower()
    lines.append("")
    lines.append("### Why CloudSense flagged it")
    if service_lower == "ec2" and recommendation == "right_sizing":
        lines.append("Average CPU and memory usage are relatively low, so the instance may have more capacity than the workload currently needs.")
        lines.append("This makes it a **right-sizing candidate**, but peak workload must be checked before changing the instance type.")
    elif service_lower == "rds" and recommendation == "right_sizing":
        lines.append("Observed utilization suggests the database class may be larger than required. Validate DB load, memory pressure, IOPS and peak workload before resizing.")
    elif service_lower == "rds" and recommendation == "scheduling":
        lines.append("The database is stopped or stopping. Confirm that this is intentional and review whether its operating schedule can be optimized safely.")
    elif service_lower == "ebs":
        lines.append("The volume appears potentially unattached. Confirm dependencies, backups, snapshots, retention and compliance before deletion.")
    elif "nat" in service_lower:
        lines.append("NAT Gateway optimization should be based on traffic, data-processing charges, cross-AZ routing and VPC endpoint opportunities — not EC2-style CPU or memory right-sizing.")
    elif service_lower == "s3":
        lines.append("S3 optimization should focus on storage class, object age, access patterns and lifecycle rules.")
    elif service_lower == "lambda":
        lines.append("Lambda optimization should focus on memory configuration, duration and invocation behavior.")
    elif service_lower == "cloudfront":
        lines.append("CloudFront optimization should focus on cache behavior, cache-hit ratio and origin traffic.")
    else:
        lines.append("CloudSense found evidence that this resource should be reviewed against its workload and operating requirements.")

    lines.append("")
    lines.append("### Recommended next steps")
    if service_lower == "ec2" and recommendation == "right_sizing":
        lines.extend([
            "1. Review recent **peak** CPU, memory and traffic usage.",
            "2. Identify a compatible smaller instance type.",
            "3. Load-test or validate the proposed size before production.",
            "4. Confirm monitoring, backup and security dependencies, then obtain DevOps approval.",
        ])
    elif "nat" in service_lower:
        lines.extend([
            "1. Review NAT bytes processed and traffic patterns.",
            "2. Check cross-AZ routing and VPC endpoint opportunities.",
            "3. Validate application connectivity before changing routing.",
        ])
    else:
        lines.append("Validate the workload, dependencies, business impact and rollback plan before approval.")

    if saving is not None:
        lines.extend(["", "### 💰 Estimated savings", f"**{currency} {saving:,.2f} / month**", f"**{currency} {saving*12:,.2f} / year**"])
    else:
        lines.extend(["", "### 💰 Estimated savings", "A resource-level savings estimate is not available from the current evidence."])

    lines.extend(["", "> ⚠️ This is a recommendation, not an executed AWS change. Validate the evidence and obtain human approval before implementation."])
    return "\n".join(lines)


# ---------------------------------------------------------
# General context
# ---------------------------------------------------------

def build_context(
    records,
    pricing,
    overview,
    ml_insights
):

    top = sorted(
        records,
        key=lambda x: safe_float(
            x.get("monthly_cost"),
            0
        ),
        reverse=True
    )[:10]

    candidates = [
        x for x in pricing
        if x.get("optimization_candidate")
    ]

    candidates = sorted(
        candidates,
        key=lambda x: safe_float(
            x.get(
                "estimated_monthly_saving"
            ),
            0
        ),
        reverse=True
    )[:10]

    return {
        "overview": overview,
        "ml_insights": ml_insights,
        "top_cost_resources": top,
        "optimization_candidates": candidates,
    }


# ---------------------------------------------------------
# General fallback
# ---------------------------------------------------------

def deterministic_answer(
    question,
    context
):

    q = question.lower()

    overview = context[
        "overview"
    ]

    candidates = context[
        "optimization_candidates"
    ]

    top = context[
        "top_cost_resources"
    ]

    if any(
        x in q
        for x in [
            "saving",
            "save",
            "optimization",
            "optimize"
        ]
    ):

        if not candidates:

            return (
                "I don't see an optimization "
                "candidate in the current data."
            )

        total = sum(
            safe_float(
                x.get(
                    "estimated_monthly_saving"
                ),
                0
            )
            for x in candidates
        )
        currency = next((str(x.get("cost_currency") or "").strip() for x in candidates if x.get("cost_currency")), "USD")
        symbol = "₹" if currency.upper() == "INR" else "$" if currency.upper() == "USD" else currency + " "

        lines = [
            f"I found {len(candidates)} "
            f"optimization candidate(s).",

            f"Estimated savings from the detected "
            f"candidates: {symbol}{total:,.0f} per month.",

            "",
            "Highest-impact opportunities:"
        ]

        for candidate in candidates[:5]:

            saving = safe_float(
                candidate.get(
                    "estimated_monthly_saving"
                ),
                0
            )

            lines.append(
                f"- {candidate.get('resource_id')}: "
                f"{candidate.get('recommendation_type')} "
                f"— about {symbol}{saving:,.0f}/month."
            )

        return "\n".join(lines)

    if any(
        x in q
        for x in [
            "bill",
            "cost",
            "expensive",
            "spend"
        ]
    ):

        current = overview.get(
            "current_monthly_cost"
        )

        if current is not None:

            return (
                f"The current uploaded cloud "
                f"run-rate is approximately "
                f"₹{current:,.0f} per month."
            )

    return (
        "I can explain cloud spend, resource "
        "utilization, optimization candidates "
        "and savings. You can also ask about "
        "a specific resource ID."
    )


# ---------------------------------------------------------
# Main answer function
# ---------------------------------------------------------

def answer_question(
    question: str,
    context: dict,
    rag_context: str = "",
    records=None,
    pricing=None
):

    records = records or []
    pricing = pricing or []

    # Detect resource
    resource_id = extract_resource_id(
        question
    )

    resource_context = None

    if resource_id:

        resource_context = build_resource_context(
            resource_id,
            records,
            pricing
        )

    # -----------------------------------------------------
    # Groq configuration
    # -----------------------------------------------------

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    model = os.getenv(
        "CLOUDSENSE_AI_MODEL",
        "openai/gpt-oss-120b"
    )

    # -----------------------------------------------------
    # If Groq unavailable → resource-aware fallback
    # -----------------------------------------------------

    if not api_key or Groq is None:

        if resource_context:

            return {
                "answer":
                    deterministic_resource_answer(
                        question,
                        resource_context,
                        rag_context
                    ),

                "provider":
                    "local_fallback_resource_aware",

                "model": None,
                "used_llm": False,
                "resource_id": resource_id,
                "rag_used": bool(rag_context),
            }

        return {
            "answer":
                deterministic_answer(
                    question,
                    context
                ),

            "provider":
                "local_fallback",

            "model": None,
            "used_llm": False,
            "resource_id": None,
            "rag_used": bool(rag_context),
        }

    # -----------------------------------------------------
    # Groq client
    # -----------------------------------------------------

    client = Groq(
        api_key=api_key
    )

    focused_context = {
        "general_cloudsense_data": context,
        "resource_id_requested": resource_id,
        "resource_specific_data": resource_context,
        "rag_knowledge": rag_context,
    }

    user_prompt = f"""
CloudSense data:

{focused_context}

Relevant RAG knowledge:

{rag_context}

Customer question:

{question}

Answer using only the supplied information.

If this is a resource-specific question,
prioritize resource_specific_data.

Use the RAG knowledge to explain the
service-specific reasoning.

Do not invent missing metrics or savings.
"""

    # -----------------------------------------------------
    # Groq call
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            max_tokens=700
        )

        answer = response.choices[
            0
        ].message.content

        return {

            "answer": answer,

            "provider": "groq",

            "model": model,

            "used_llm": True,

            "resource_id": resource_id,

            "rag_used": bool(
                rag_context
            ),
        }

    except Exception as exc:

        # -------------------------------------------------
        # If Groq quota/rate limit/error occurs,
        # still answer using CloudSense data.
        # -------------------------------------------------

        if resource_context:

            answer = (
                deterministic_resource_answer(
                    question,
                    resource_context,
                    rag_context
                )
            )

            return {

                "answer": answer,

                "provider":
                    "local_fallback_after_groq_error",

                "model": model,

                "used_llm": False,

                "resource_id": resource_id,

                "rag_used": bool(
                    rag_context
                ),

                "error": str(exc),
            }

        return {

            "answer":
                deterministic_answer(
                    question,
                    context
                ),

            "provider":
                "local_fallback_after_groq_error",

            "model": model,

            "used_llm": False,

            "resource_id": None,

            "rag_used": bool(
                rag_context
            ),

            "error": str(exc),
        }