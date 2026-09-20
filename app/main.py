from statistics import mean
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.services.aws_collector import AWSCloudSenseCollector

from .db import Base, engine, SessionLocal, ensure_schema
from .models import BillingRecord, BillingDocument, AWSResource, CostSnapshot
from .schemas import BillingRecordResponse, BillingDocumentResponse
from .services.file_reader import read_csv_bytes, read_excel_bytes
from .services.normalizer import normalize_records
from .services.extractor import extract_pdf, extract_image
from .services.metrics import analyze_record
from .services.pricing import pricing_for_metric
from .services.ml_engine import detect_cost_anomalies, service_cost_anomalies, forecast_monthly_cost
from .services.genai import answer_question, build_context
from .services.rag import retrieve, format_context, retrieval_status
from .services.agent import CloudSenseAgent
from .services.evaluation import evaluate_result, evaluate_suite
from .services.resource_intelligence import AWSResourceIntelligence
from .services.aws_connector import test_connection, get_connection_state

ensure_schema()

app = FastAPI(
    title="CloudSense AI Backend",
    version="1.0.0",
    description="CloudSense AI cloud billing ingestion, analysis, optimization and AI APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CLOUDSENSE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ALLOWED = {".pdf", ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}
MAX_SIZE = 15 * 1024 * 1024

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CloudSense AI Backend", "steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "aws_resource_intelligence": True, "resource_level_billing": True, "resource_intelligence_savings": True}

def save_records(db, records, source_file):
    # Keep repeated demo uploads idempotent by replacing records from the same source.
    db.query(BillingRecord).filter(BillingRecord.source_file == source_file).delete()
    for record in records:
        db.add(BillingRecord(**record, source_file=source_file))
    db.commit()
    return len(records)


def capture_snapshot(db, source="demo", account_id=None, region=None):
    records = db.query(BillingRecord).all()
    monthly = round(sum(float(r.monthly_cost or 0) for r in records), 2)
    anomalies = [x for x in detect_cost_anomalies(records) if x.get("anomaly")]
    priced = [pricing_for_metric(analyze_record(r)) for r in records]
    saving = round(sum(float(x.get("estimated_monthly_saving") or 0) for x in priced), 2)
    snap = CostSnapshot(source=source, account_id=account_id, region=region,
                        monthly_cost=monthly, resource_count=len(records),
                        anomaly_count=len(anomalies), potential_monthly_saving=saving)
    db.add(snap); db.commit()
    return snap

@app.post("/api/billing/upload")
async def upload_billing(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 15 MB limit.")

    result = {
        "filename": filename,
        "file_type": ext.lstrip("."),
        "size_bytes": len(data),
        "records_saved": 0,
        "extraction_method": None,
        "ocr_required": False,
    }

    if ext == ".csv":
        rows = read_csv_bytes(data)
        normalized = normalize_records(rows)
        result["records_saved"] = save_records(db, normalized, filename)
        capture_snapshot(db, source="demo")
        result["extraction_method"] = "csv_parser"

    elif ext in {".xlsx", ".xls"}:
        rows = read_excel_bytes(data)
        normalized = normalize_records(rows)
        result["records_saved"] = save_records(db, normalized, filename)
        capture_snapshot(db, source="demo")
        result["extraction_method"] = "excel_parser"

    elif ext == ".pdf":
        extracted = extract_pdf(data)
        result["extraction_method"] = extracted["method"]
        result["ocr_required"] = extracted["ocr_required"]
        doc = BillingDocument(
            filename=filename,
            file_type="pdf",
            extracted_text=extracted["text"],
            extraction_method=extracted["method"],
            ocr_required=int(extracted["ocr_required"]),
        )
        db.add(doc)
        db.commit()
        result["extracted_text_preview"] = extracted["text"][:1500]

    else:
        extracted = extract_image(data)
        result["extraction_method"] = extracted["method"]
        doc = BillingDocument(
            filename=filename,
            file_type=ext.lstrip("."),
            extracted_text=extracted["text"],
            extraction_method=extracted["method"],
            ocr_required=0,
        )
        db.add(doc)
        db.commit()
        result["extracted_text_preview"] = extracted["text"][:1500]

    return {"success": True, **result}

@app.post("/api/billing/ingest")
async def ingest_structured(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(
            status_code=400,
            detail="Structured ingestion currently accepts CSV/XLSX/XLS. Use /api/billing/upload for PDF/image extraction."
        )

    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 15 MB limit.")

    rows = read_csv_bytes(data) if ext == ".csv" else read_excel_bytes(data)
    normalized = normalize_records(rows)
    count = save_records(db, normalized, filename)
    capture_snapshot(db, source="demo")

    return {"success": True, "filename": filename, "records_saved": count}

@app.get("/api/billing/records", response_model=list[BillingRecordResponse])
def get_records(limit: int = 100000, db: Session = Depends(get_db)):
    safe_limit = min(max(limit, 1), 100000)
    return db.query(BillingRecord).order_by(BillingRecord.id.desc()).limit(safe_limit).all()

@app.get("/api/billing/summary")
def billing_summary(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    total_monthly = sum(r.monthly_cost or 0 for r in records)
    total_daily = sum(r.daily_cost or 0 for r in records)
    services = {}

    for r in records:
        key = r.service or "Unknown"
        services[key] = services.get(key, 0) + (r.monthly_cost or 0)

    return {
        "record_count": len(records),
        "total_daily_cost": round(total_daily, 2),
        "total_monthly_cost": round(total_monthly, 2),
        "service_breakdown": {
            k: round(v, 2) for k, v in sorted(services.items(), key=lambda x: x[1], reverse=True)
        },
    }

@app.get("/api/billing/documents", response_model=list[BillingDocumentResponse])
def get_documents(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(BillingDocument).order_by(BillingDocument.id.desc()).limit(limit).all()


@app.get("/api/metrics/resources")
def resource_metrics(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).order_by(BillingRecord.id.asc()).all()
    return {
        "success": True,
        "record_count": len(records),
        "resources": [analyze_record(r) for r in records],
    }

@app.get("/api/metrics/optimization-candidates")
def optimization_candidates(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    analyzed = [analyze_record(r) for r in records]
    candidates = [r for r in analyzed if r["optimization_candidate"]]

    total_monthly_cost = sum(r["monthly_cost"] for r in candidates)

    return {
        "success": True,
        "candidate_count": len(candidates),
        "candidate_monthly_cost": round(total_monthly_cost, 2),
        "candidates": candidates,
    }

@app.get("/api/metrics/overview")
def metrics_overview(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    analyzed = [analyze_record(r) for r in records]

    total = len(analyzed)
    underutilized = sum(
        1 for r in analyzed
        if r["utilization_status"] in {"underutilized", "severely_underutilized"}
    )
    candidates = sum(1 for r in analyzed if r["optimization_candidate"])

    cpu_values = [r["cpu_utilization"] for r in analyzed if r["cpu_utilization"] is not None]
    memory_values = [r["memory_utilization"] for r in analyzed if r["memory_utilization"] is not None]
    uptime_values = [r["hours_running"] for r in analyzed if r["hours_running"] is not None]

    return {
        "success": True,
        "resource_count": total,
        "underutilized_resources": underutilized,
        "optimization_candidates": candidates,
        "average_cpu_utilization": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else None,
        "average_memory_utilization": round(sum(memory_values) / len(memory_values), 2) if memory_values else None,
        "average_hours_running": round(sum(uptime_values) / len(uptime_values), 2) if uptime_values else None,
    }


@app.get("/api/pricing/resources")
def pricing_resources(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    analyzed = [analyze_record(r) for r in records]
    return {
        "success": True,
        "resource_count": len(analyzed),
        "resources": [pricing_for_metric(m) for m in analyzed],
    }

@app.get("/api/pricing/savings")
def pricing_savings(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    analyzed = [pricing_for_metric(analyze_record(r)) for r in records]

    current_monthly = round(sum(r["monthly_cost"] for r in analyzed), 2)
    monthly_saving = round(sum(r["estimated_monthly_saving"] for r in analyzed), 2)
    projected_monthly = round(current_monthly - monthly_saving, 2)

    return {
        "success": True,
        "current_monthly_cost": current_monthly,
        "estimated_monthly_saving": monthly_saving,
        "estimated_yearly_saving": round(monthly_saving * 12, 2),
        "projected_monthly_cost": projected_monthly,
        "projected_yearly_cost": round(projected_monthly * 12, 2),
        "saving_percentage": round((monthly_saving / current_monthly) * 100, 2) if current_monthly else 0,
    }

@app.get("/api/pricing/optimization-plan")
def optimization_plan(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    priced = [pricing_for_metric(analyze_record(r)) for r in records]
    candidates = [r for r in priced if r["optimization_candidate"]]

    candidates.sort(key=lambda x: x["estimated_monthly_saving"], reverse=True)
    total_saving = round(sum(r["estimated_monthly_saving"] for r in candidates), 2)

    # Keep the API payload/UI responsive for large inventories. The full
    # inventory remains available through Resource Inventory pagination.
    top_recommendations = candidates

    return {
        "success": True,
        "recommendation_count": len(candidates),
        "returned_recommendation_count": len(top_recommendations),
        "estimated_monthly_saving": total_saving,
        "estimated_yearly_saving": round(total_saving * 12, 2),
        "recommendations": top_recommendations,
    }


@app.get("/api/ml/anomalies")
def ml_anomalies(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    anomalies = detect_cost_anomalies(records)
    detected = [a for a in anomalies if a["anomaly"]]

    return {
        "success": True,
        "resource_count": len(records),
        "anomaly_count": len(detected),
        "anomalies": detected,
    }

@app.get("/api/ml/service-analysis")
def ml_service_analysis(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()

    return {
        "success": True,
        "services": service_cost_anomalies(records),
    }

@app.get("/api/ml/forecast")
def ml_forecast(months: int = 3, db: Session = Depends(get_db)):
    if months < 1 or months > 12:
        raise HTTPException(status_code=400, detail="months must be between 1 and 12")

    records = db.query(BillingRecord).all()

    return {
        "success": True,
        **forecast_monthly_cost(records, months),
    }

@app.get("/api/ml/insights")
def ml_insights(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    anomalies = detect_cost_anomalies(records)
    candidates = [a for a in anomalies if a["anomaly"]]

    total = sum(float(r.monthly_cost or 0) for r in records)
    avg_cpu_values = [r.cpu_utilization for r in records if r.cpu_utilization is not None]
    avg_memory_values = [r.memory_utilization for r in records if r.memory_utilization is not None]

    insights = []

    if candidates:
        highest = sorted(candidates, key=lambda x: x["monthly_cost"], reverse=True)[0]
        insights.append(
            f"{len(candidates)} resource(s) have unusually high cost relative to the uploaded dataset. "
            f"The highest-cost anomaly is {highest['resource_id']}."
        )

    if avg_cpu_values:
        avg_cpu = mean(avg_cpu_values)
        if avg_cpu < 25:
            insights.append(
                f"Average CPU utilization is {avg_cpu:.1f}%, indicating broad underutilization across resources."
            )

    if avg_memory_values:
        avg_memory = mean(avg_memory_values)
        if avg_memory < 35:
            insights.append(
                f"Average memory utilization is {avg_memory:.1f}%, suggesting right-sizing opportunities."
            )

    insights.append(
        f"Current uploaded monthly run-rate is ₹{total:,.0f}."
    )

    return {
        "success": True,
        "insights": insights,
    }


@app.get("/api/analytics/executive")
def executive_analytics(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    anomalies = detect_cost_anomalies(records)
    detected = [a for a in anomalies if a.get("anomaly")]
    services = service_cost_anomalies(records)
    forecast = forecast_monthly_cost(records, 3)
    total = round(sum(float(r.monthly_cost or 0) for r in records), 2)
    return {
        "success": True,
        "monthly_cost": total,
        "record_count": len(records),
        "anomaly_count": len(detected),
        "top_service": services[0] if services else None,
        "forecast": forecast,
        "anomalies": detected[:20],
        "services": services[:20],
    }

from pydantic import BaseModel

class GenAIChatRequest(BaseModel):
    question: str

@app.post("/api/genai/chat")
def genai_chat(payload: GenAIChatRequest, db: Session = Depends(get_db)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    records_db = db.query(BillingRecord).all()

    metrics = [analyze_record(r) for r in records_db]
    pricing = [pricing_for_metric(m) for m in metrics]

    current_monthly = round(sum(x["monthly_cost"] for x in pricing), 2)
    monthly_saving = round(sum(x["estimated_monthly_saving"] for x in pricing), 2)

    overview = {
        "resource_count": len(pricing),
        "current_monthly_cost": current_monthly,
        "estimated_monthly_saving": monthly_saving,
        "estimated_yearly_saving": round(monthly_saving * 12, 2),
        "projected_monthly_cost": round(current_monthly - monthly_saving, 2),
    }

    ml = detect_cost_anomalies(records_db)
    ml_insights = []
    anomalies = [x for x in ml if x["anomaly"]]
    if anomalies:
        ml_insights.append(
            f"{len(anomalies)} resource(s) were flagged by the cost anomaly detector."
        )

    context = build_context(pricing, pricing, overview, ml_insights)
    rag_chunks = retrieve(payload.question)
    rag_context = format_context(rag_chunks)
    result = answer_question(payload.question, context, rag_context, records=pricing, pricing=pricing)

    return {
        "success": True,
        "question": payload.question,
        **result,
        "context_summary": {
            "resource_count": len(pricing),
            "current_monthly_cost": current_monthly,
            "estimated_monthly_saving": monthly_saving,
        },
    }


@app.get("/api/rag/search")
def rag_search(question: str, top_k: int = 4):
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    chunks = retrieve(question, top_k)

    return {
        "success": True,
        "question": question,
        "result_count": len(chunks),
        "results": [
            {
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "score": c.get("score"),
                "retrieval": c.get("retrieval"),
                "text": c["text"],
            }
            for c in chunks
        ],
    }

@app.get("/api/rag/status")
def rag_status():
    return {"success": True, **retrieval_status()}


@app.get("/api/rag/documents")
def rag_documents():
    from .services.rag import load_documents
    docs = load_documents()
    return {
        "success": True,
        "document_count": len(docs),
        "documents": [d["source"] for d in docs],
    }


def _agent_cost_summary(db):
    records = db.query(BillingRecord).all()
    current = round(sum(float(r.monthly_cost or 0) for r in records), 2)
    return {
        "resource_count": len(records),
        "current_monthly_cost": current,
        "current_yearly_run_rate": round(current * 12, 2),
    }

def _agent_resource_metrics(db):
    records = db.query(BillingRecord).all()
    analyzed = [analyze_record(r) for r in records]
    candidates = [x for x in analyzed if x["optimization_candidate"]]
    return {
        "resource_count": len(analyzed),
        "optimization_candidate_count": len(candidates),
        "resources": analyzed[:20],
    }

def _agent_savings(db):
    records = db.query(BillingRecord).all()
    priced = [pricing_for_metric(analyze_record(r)) for r in records]
    saving = round(sum(x["estimated_monthly_saving"] for x in priced), 2)
    current = round(sum(x["monthly_cost"] for x in priced), 2)
    return {
        "current_monthly_cost": current,
        "estimated_monthly_saving": saving,
        "estimated_yearly_saving": round(saving * 12, 2),
        "projected_monthly_cost": round(current - saving, 2),
    }

def _agent_policy_search(question):
    chunks = retrieve(question, 4)
    return {
        "matches": [
            {
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "evidence_id": f"{c["source"]}#{c["chunk_id"]}",
                "score": c.get("score"),
                "retrieval": c.get("retrieval"),
                "text": c["text"],
            }
            for c in chunks
        ]
    }

def _agent_ml(db):
    records = db.query(BillingRecord).all()
    anomalies = detect_cost_anomalies(records)
    return {
        "anomaly_count": sum(1 for x in anomalies if x["anomaly"]),
        "anomalies": [x for x in anomalies if x["anomaly"]],
        "forecast": forecast_monthly_cost(records, 3),
    }

@app.post("/api/agent/run")
def run_agent(payload: GenAIChatRequest, db: Session = Depends(get_db)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    tools = {
        "get_cost_summary": lambda: _agent_cost_summary(db),
        "get_resource_metrics": lambda: _agent_resource_metrics(db),
        "calculate_savings": lambda: _agent_savings(db),
        "search_policies": lambda: _agent_policy_search(payload.question),
        "analyze_ml": lambda: _agent_ml(db),
        "get_aws_resource_intelligence": lambda: _agent_aws_resource_intelligence(db),
    }

    agent = CloudSenseAgent(tools)
    result = agent.run(payload.question)
    result["evaluation"] = evaluate_result(result, [])

    return {
        "success": True,
        "question": result.get("question"),
        "answer": result.get("answer"),
        "agent_mode": result.get("agent_mode"),
        "model": result.get("model"),
        "evaluation": result.get("evaluation"),
    }


@app.get("/api/evaluation/cases")
def evaluation_cases():
    from .services.evaluation import EVAL_CASES
    return {"success": True, "cases": EVAL_CASES}


@app.post("/api/evaluation/run")
def run_evaluation(db: Session = Depends(get_db)):
    def run_case(question: str):
        tools = {
            "get_cost_summary": lambda: _agent_cost_summary(db),
            "get_resource_metrics": lambda: _agent_resource_metrics(db),
            "calculate_savings": lambda: _agent_savings(db),
            "search_policies": lambda: _agent_policy_search(question),
            "analyze_ml": lambda: _agent_ml(db),
            "get_aws_resource_intelligence": lambda: _agent_aws_resource_intelligence(db),
        }
        return CloudSenseAgent(tools).run(question)

    suite = evaluate_suite(run_case)
    return {"success": True, "evaluation_mode": "live_agent_or_deterministic_fallback", **suite}


class AWSConnectionRequest(BaseModel):
    account_id: str
    access_key_id: str
    secret_access_key: str
    region: str = "ap-south-1"

@app.get("/api/system/capabilities")
def system_capabilities():
    return {
        "success": True,
        "live_aws": True,
        "demo_upload": True,
        "ml": True,
        "rag": True,
        "tool_calling_agent": True,
        "evaluation": True,
        "simulation_only": True,
        "read_only_aws": True,
        "production_auth_recommendation": "Use IAM roles/OIDC and temporary credentials in production.",
    }

@app.get("/api/analytics/history")
def analytics_history(limit: int = 30, db: Session = Depends(get_db)):
    rows = db.query(CostSnapshot).order_by(CostSnapshot.captured_at.desc()).limit(min(max(limit, 1), 100)).all()
    return {"success": True, "snapshots": [{
        "id": r.id, "source": r.source, "account_id": r.account_id, "region": r.region,
        "monthly_cost": r.monthly_cost, "resource_count": r.resource_count,
        "anomaly_count": r.anomaly_count, "potential_monthly_saving": r.potential_monthly_saving,
        "captured_at": r.captured_at.isoformat() if r.captured_at else None
    } for r in reversed(rows)]}

@app.get("/api/analytics/root-cause")
def analytics_root_cause(db: Session = Depends(get_db)):
    records = db.query(BillingRecord).all()
    services = service_cost_anomalies(records)
    anomalies = [x for x in detect_cost_anomalies(records) if x.get("anomaly")]
    total = sum(float(r.monthly_cost or 0) for r in records)
    top = services[0] if services else None
    reasons = []
    if top and total:
        reasons.append(f"{top['service']} represents {top['total_monthly_cost']/total*100:.1f}% of the current dataset cost.")
    if anomalies:
        reasons.append(f"{len(anomalies)} resource-level cost anomalies were detected by the explainable detector.")
    low_cpu = [r for r in records if r.cpu_utilization is not None and r.cpu_utilization < 20]
    if low_cpu:
        reasons.append(f"{len(low_cpu)} records have CPU utilization below 20%, which may indicate right-sizing opportunities.")
    return {"success": True, "drivers": services[:5], "anomalies": anomalies[:10], "reasons": reasons, "note": "These are contributing signals, not a causal proof."}

@app.post("/api/workspace/clear")
def clear_workspace(db: Session = Depends(get_db)):
    """Clear all locally stored CloudSense data and remove saved AWS connection state.

    This does not call AWS and does not change any AWS infrastructure.
    """
    counts = {
        "billing_records": db.query(BillingRecord).count(),
        "billing_documents": db.query(BillingDocument).count(),
        "aws_resources": db.query(AWSResource).count(),
        "cost_snapshots": db.query(CostSnapshot).count(),
    }

    db.query(BillingRecord).delete(synchronize_session=False)
    db.query(BillingDocument).delete(synchronize_session=False)
    db.query(AWSResource).delete(synchronize_session=False)
    db.query(CostSnapshot).delete(synchronize_session=False)
    db.commit()

    # Remove the persisted AWS connection status. Credentials themselves are not
    # stored by CloudSense, but this prevents the UI from showing a stale connection.
    from app.services.aws_connector import CONNECTION_FILE
    try:
        if CONNECTION_FILE.exists():
            CONNECTION_FILE.unlink()
    except OSError:
        pass

    return {
        "success": True,
        "message": "CloudSense workspace cleared. Demo data, collected AWS data, snapshots and saved AWS connection status were removed.",
        "deleted": counts,
        "aws_infrastructure_changed": False,
    }


@app.get("/api/aws/status")
def aws_status():
    return {"success": True, **get_connection_state()}

@app.post("/api/aws/test-connection")
def aws_test_connection(payload: AWSConnectionRequest):
    if not (payload.account_id.isdigit() and len(payload.account_id) == 12):
        raise HTTPException(status_code=400, detail="AWS Account ID must be 12 digits.")
    result = test_connection(payload.account_id, payload.access_key_id, payload.secret_access_key, payload.region)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "AWS connection failed."))
    return result

class AWSCollectRequest(BaseModel):
    access_key_id: str
    secret_access_key: str
    region: str = "ap-south-1"
    days: int = 30

@app.post("/api/aws/collect")
def aws_collect(payload: AWSCollectRequest, db: Session = Depends(get_db)):
    if not 1 <= payload.days <= 30:
        return {"success": False, "message": "days must be between 1 and 30"}
    try:
        collector = AWSCloudSenseCollector(
            payload.access_key_id, payload.secret_access_key, payload.region
        )
        data = collector.collect_all(payload.days)

        account_id = data.get("account_id") or "unknown"
        source_name = f"AWS Live / {account_id} / {payload.region}"

        billing_rows = []
        for row in data.get("billing", []):
            billing_rows.append({
                "provider": "AWS",
                "billing_month": row.get("billing_month"),
                "service": row.get("service"),
                "region": row.get("region"),
                "resource_id": row.get("resource_id"),
                "environment": row.get("environment"),
                "team": row.get("team"),
                "daily_cost": row.get("daily_cost", 0),
                "monthly_cost": row.get("monthly_cost", 0),
                "cpu_utilization": None,
                "memory_utilization": None,
                "hours_running": None,
            })
        records_saved = save_records(db, billing_rows, source_name)

        db.query(AWSResource).filter(
            AWSResource.account_id == account_id,
            AWSResource.region == payload.region,
        ).delete(synchronize_session=False)

        cpu_map = {m.get("resource_id"): m for m in data.get("cloudwatch", [])}
        resource_cost_map = {
            row.get("resource_id"): row
            for row in data.get("resource_level_billing", [])
            if row.get("resource_id")
        }
        resource_rows = []
        for resource in data.get("resources", []):
            rid = resource.get("resource_id")
            metric = cpu_map.get(rid, {})
            excluded = {
                "provider", "service", "resource_id", "resource_type",
                "resource_name", "region", "state", "environment", "team",
                "cpu_utilization", "memory_utilization", "monthly_cost", "cost_currency"
            }
            metadata = {k: v for k, v in resource.items() if k not in excluded}
            resource_rows.append(AWSResource(
                account_id=account_id,
                service=resource.get("service") or "Unknown",
                resource_id=rid,
                resource_type=resource.get("resource_type") or resource.get("instance_type"),
                resource_name=resource.get("resource_name"),
                region=resource.get("region") or payload.region,
                state=resource.get("state"),
                environment=resource.get("environment"),
                team=resource.get("team"),
                cpu_utilization=resource.get("cpu_utilization") if resource.get("cpu_utilization") is not None else metric.get("average"),
                memory_utilization=resource.get("memory_utilization"),
                monthly_cost=(resource_cost_map.get(rid) or {}).get("monthly_cost"),
                cost_currency=(resource_cost_map.get(rid) or {}).get("currency"),
                metadata_json=metadata,
            ))

        if resource_rows:
            db.add_all(resource_rows)
        db.commit()
        capture_snapshot(db, source="aws", account_id=account_id, region=payload.region)

        return {
            "success": True,
            "message": "AWS data collected and billing/resources loaded into CloudSense.",
            "records_saved": records_saved,
            "resource_records_saved": len(resource_rows),
            "resource_level_billing_records": len(data.get("resource_level_billing", [])),
            **data,
        }
    except Exception as exc:
        return {"success": False, "message": f"AWS data collection failed: {exc}"}


def _aws_resource_intelligence(db: Session):
    resources = db.query(AWSResource).all()
    billing = db.query(BillingRecord).all()
    engine = AWSResourceIntelligence(resources, billing)
    return engine.analyze()

@app.get("/api/aws/resource-intelligence")
def aws_resource_intelligence(db: Session = Depends(get_db)):
    return {"success": True, **_aws_resource_intelligence(db)}


@app.get("/api/aws/resource-billing")
def aws_resource_billing(db: Session = Depends(get_db)):
    resources = db.query(AWSResource).filter(AWSResource.monthly_cost.isnot(None)).all()
    total = round(sum(float(r.monthly_cost or 0) for r in resources), 2)
    return {
        "success": True,
        "resource_level_billing_available": bool(resources),
        "resource_count_with_cost": len(resources),
        "monthly_cost_total": total,
        "currency": next((r.cost_currency for r in resources if r.cost_currency), "USD"),
        "resources": [
            {
                "resource_id": r.resource_id,
                "service": r.service,
                "resource_name": r.resource_name,
                "monthly_cost": r.monthly_cost,
                "currency": r.cost_currency,
                "cpu_utilization": r.cpu_utilization,
            }
            for r in resources
        ],
        "note": "Resource-level Cost Explorer is limited to supported resource types and recent periods; unavailable rows are not fabricated.",
    }


@app.get("/api/aws/resources")
def aws_resources(
    service: str | None = None,
    limit: int = 100000,
    db: Session = Depends(get_db),
):
    query = db.query(AWSResource).order_by(AWSResource.id.desc())
    if service:
        query = query.filter(AWSResource.service == service)
    resources = query.limit(min(max(limit, 1), 100000)).all()
    return {
        "success": True,
        "count": len(resources),
        "resources": [
            {
                "id": r.id,
                "account_id": r.account_id,
                "service": r.service,
                "resource_id": r.resource_id,
                "resource_type": r.resource_type,
                "resource_name": r.resource_name,
                "region": r.region,
                "state": r.state,
                "environment": r.environment,
                "team": r.team,
                "cpu_utilization": r.cpu_utilization,
                "memory_utilization": r.memory_utilization,
                "monthly_cost": r.monthly_cost,
                "cost_currency": r.cost_currency,
                "metadata": r.metadata_json or {},
                "collected_at": r.collected_at.isoformat() if r.collected_at else None,
            }
            for r in resources
        ],
    }


@app.get("/api/aws/resource-overview")
def aws_resource_overview(db: Session = Depends(get_db)):
    resources = db.query(AWSResource).all()
    by_service = {}
    by_state = {}
    underutilized_ec2 = 0

    for r in resources:
        by_service[r.service] = by_service.get(r.service, 0) + 1
        if r.state:
            by_state[r.state] = by_state.get(r.state, 0) + 1
        if r.service == "EC2" and r.cpu_utilization is not None and r.cpu_utilization < 20:
            underutilized_ec2 += 1

    return {
        "success": True,
        "resource_count": len(resources),
        "by_service": dict(sorted(by_service.items(), key=lambda x: x[1], reverse=True)),
        "by_state": dict(sorted(by_state.items(), key=lambda x: x[1], reverse=True)),
        "potentially_underutilized_ec2": underutilized_ec2,
    }

