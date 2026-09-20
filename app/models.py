from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from .db import Base


class BillingRecord(Base):
    __tablename__ = "billing_records"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), default="AWS")
    billing_month = Column(String(30), nullable=True)
    service = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    resource_id = Column(String(150), nullable=True)
    environment = Column(String(50), nullable=True)
    team = Column(String(100), nullable=True)
    daily_cost = Column(Float, default=0)
    monthly_cost = Column(Float, default=0)
    cpu_utilization = Column(Float, nullable=True)
    memory_utilization = Column(Float, nullable=True)
    cost_currency = Column(String(10), nullable=True)
    hours_running = Column(Float, nullable=True)
    source_file = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BillingDocument(Base):
    __tablename__ = "billing_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(30), nullable=False)
    extracted_text = Column(Text, nullable=True)
    extraction_method = Column(String(50), nullable=True)
    ocr_required = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AWSResource(Base):
    __tablename__ = "aws_resources"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(20), nullable=False, index=True)
    service = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(300), nullable=True, index=True)
    resource_type = Column(String(150), nullable=True)
    resource_name = Column(String(300), nullable=True)
    region = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    environment = Column(String(100), nullable=True)
    team = Column(String(150), nullable=True)
    cpu_utilization = Column(Float, nullable=True)
    memory_utilization = Column(Float, nullable=True)
    monthly_cost = Column(Float, nullable=True)
    cost_currency = Column(String(10), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    collected_at = Column(DateTime, default=datetime.utcnow, index=True)


class CostSnapshot(Base):
    __tablename__ = "cost_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(30), nullable=False, index=True)
    account_id = Column(String(20), nullable=True, index=True)
    region = Column(String(100), nullable=True)
    monthly_cost = Column(Float, default=0)
    resource_count = Column(Integer, default=0)
    anomaly_count = Column(Integer, default=0)
    potential_monthly_saving = Column(Float, default=0)
    captured_at = Column(DateTime, default=datetime.utcnow, index=True)
