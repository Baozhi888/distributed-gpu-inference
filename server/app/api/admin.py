"""
管理后台API
提供：算力监控、使用量统计、企业管理、Worker管理、账单管理
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, timedelta
from uuid import UUID
from pydantic import BaseModel, Field

from app.db.database import get_db
from app.models.models import Worker, Job, JobStatus
from app.models.usage import Enterprise, EnterpriseAPIKey, UsageRecord, Bill, PricePlan, WorkerUsageSummary
from app.services.usage import UsageService
from app.services.privacy import EnterprisePrivacyService, DataRetentionService, PrivacyAuditService

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ============ Schemas ============

class EnterpriseCreate(BaseModel):
    name: str
    code: str
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    billing_email: Optional[str] = None
    billing_period: str = "monthly"
    monthly_budget: Optional[float] = None
    data_retention_days: int = 30
    allow_logging: bool = True
    anonymize_data: bool = False


class EnterpriseUpdate(BaseModel):
    name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    monthly_budget: Optional[float] = None
    credit_balance: Optional[float] = None
    is_active: Optional[bool] = None
    allow_logging: Optional[bool] = None
    anonymize_data: Optional[bool] = None


class APIKeyCreate(BaseModel):
    name: str
    allowed_types: List[str] = []
    allowed_models: List[str] = []
    rate_limit_per_minute: int = 60
    daily_limit: Optional[int] = None
    ip_whitelist: List[str] = []
    expires_days: Optional[int] = None


class DashboardStats(BaseModel):
    workers: dict
    enterprises: dict
    today: dict
    this_month: dict
    timestamp: str


class UsageQueryParams(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    group_by: str = "day"


# ============ 仪表盘 ============

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """获取管理仪表盘统计数据"""
    return UsageService.get_platform_stats(db)


@router.get("/dashboard/realtime")
async def get_realtime_stats(db: Session = Depends(get_db)):
    """获取实时监控数据"""
    # 当前在线Worker
    online_workers = db.query(Worker).filter(
        Worker.status.in_(['online', 'busy'])
    ).all()

    # 正在执行的任务
    running_jobs = db.query(Job).filter(
        Job.status == JobStatus.RUNNING.value
    ).all()

    # 队列中的任务
    queued_count = db.query(Job).filter(
        Job.status == JobStatus.QUEUED.value
    ).count()

    workers_data = []
    for w in online_workers:
        workers_data.append({
            'id': str(w.id),
            'name': w.name,
            'machine_id': w.machine_id,
            'status': w.status,
            'region': w.region,
            'gpu_model': w.gpu_model,
            'gpu_memory_gb': w.gpu_memory_gb,
            'gpu_memory_used_gb': w.gpu_memory_used_gb,
            'current_job': str(w.current_job_id) if w.current_job_id else None,
            'reliability_score': w.reliability_score,
            'last_heartbeat': w.last_heartbeat.isoformat() if w.last_heartbeat else None
        })

    jobs_data = []
    for j in running_jobs:
        jobs_data.append({
            'id': str(j.id),
            'type': j.type,
            'worker_id': str(j.worker_id) if j.worker_id else None,
            'started_at': j.started_at.isoformat() if j.started_at else None,
            'duration_seconds': (datetime.utcnow() - j.started_at).total_seconds() if j.started_at else 0
        })

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'workers': {
            'online': len([w for w in online_workers if w.status == 'online']),
            'busy': len([w for w in online_workers if w.status == 'busy']),
            'details': workers_data
        },
        'jobs': {
            'running': len(running_jobs),
            'queued': queued_count,
            'details': jobs_data
        }
    }


# ============ Worker管理 ============

@router.get("/workers")
async def list_workers(
    status: Optional[str] = None,
    region: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """获取Worker列表"""
    query = db.query(Worker)

    if status:
        query = query.filter(Worker.status == status)
    if region:
        query = query.filter(Worker.region == region)

    total = query.count()
    workers = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': [{
            'id': str(w.id),
            'name': w.name,
            'machine_id': w.machine_id,
            'status': w.status,
            'region': w.region,
            'country': w.country,
            'city': w.city,
            'gpu_model': w.gpu_model,
            'gpu_memory_gb': w.gpu_memory_gb,
            'gpu_count': w.gpu_count,
            'supported_types': w.supported_types,
            'reliability_score': w.reliability_score,
            'success_rate': w.success_rate,
            'total_jobs': w.total_jobs,
            'completed_jobs': w.completed_jobs,
            'last_heartbeat': w.last_heartbeat.isoformat() if w.last_heartbeat else None,
            'registered_at': w.registered_at.isoformat() if w.registered_at else None
        } for w in workers]
    }


@router.get("/workers/{worker_id}")
async def get_worker_detail(
    worker_id: UUID,
    db: Session = Depends(get_db)
):
    """获取Worker详情"""
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # 获取最近使用统计
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    usage_stats = UsageService.get_worker_usage(db, worker_id, day_ago, now)

    return {
        'worker': {
            'id': str(worker.id),
            'name': worker.name,
            'machine_id': worker.machine_id,
            'hardware_hash': worker.hardware_hash,
            'hardware_details': worker.hardware_details,
            'status': worker.status,
            'region': worker.region,
            'country': worker.country,
            'city': worker.city,
            'timezone': worker.timezone,
            'gpu_model': worker.gpu_model,
            'gpu_memory_gb': worker.gpu_memory_gb,
            'gpu_count': worker.gpu_count,
            'cpu_cores': worker.cpu_cores,
            'ram_gb': worker.ram_gb,
            'supported_types': worker.supported_types,
            'loaded_models': worker.loaded_models,
            'reliability_score': worker.reliability_score,
            'success_rate': worker.success_rate,
            'avg_latency_ms': worker.avg_latency_ms,
            'total_online_seconds': worker.total_online_seconds,
            'total_jobs': worker.total_jobs,
            'completed_jobs': worker.completed_jobs,
            'failed_jobs': worker.failed_jobs,
            'config_override': worker.config_override,
            'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
            'registered_at': worker.registered_at.isoformat() if worker.registered_at else None
        },
        'usage_24h': usage_stats
    }


@router.get("/workers/{worker_id}/usage")
async def get_worker_usage(
    worker_id: UUID,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """获取Worker使用量统计"""
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=7)

    return UsageService.get_worker_usage(db, worker_id, start_time, end_time)


@router.put("/workers/{worker_id}/config")
async def update_worker_config(
    worker_id: UUID,
    config: dict,
    db: Session = Depends(get_db)
):
    """更新Worker配置覆盖"""
    worker = db.query(Worker).filter(Worker.id == worker_id).first()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # 合并配置
    current_config = worker.config_override or {}
    current_config.update(config)
    worker.config_override = current_config
    worker.config_version = (worker.config_version or 0) + 1

    db.commit()

    return {'status': 'success', 'config_version': worker.config_version}


# ============ 企业管理 ============

@router.get("/enterprises")
async def list_enterprises(
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """获取企业列表"""
    query = db.query(Enterprise)

    if is_active is not None:
        query = query.filter(Enterprise.is_active == is_active)

    total = query.count()
    enterprises = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': [{
            'id': str(e.id),
            'name': e.name,
            'code': e.code,
            'contact_name': e.contact_name,
            'contact_email': e.contact_email,
            'credit_balance': e.credit_balance,
            'monthly_budget': e.monthly_budget,
            'is_active': e.is_active,
            'is_verified': e.is_verified,
            'created_at': e.created_at.isoformat() if e.created_at else None
        } for e in enterprises]
    }


@router.post("/enterprises")
async def create_enterprise(
    data: EnterpriseCreate,
    db: Session = Depends(get_db)
):
    """创建企业"""
    # 检查code是否已存在
    existing = db.query(Enterprise).filter(Enterprise.code == data.code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Enterprise code already exists")

    enterprise = Enterprise(**data.dict())
    db.add(enterprise)
    db.commit()
    db.refresh(enterprise)

    return {'id': str(enterprise.id), 'code': enterprise.code}


@router.get("/enterprises/{enterprise_id}")
async def get_enterprise_detail(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """获取企业详情"""
    enterprise = db.query(Enterprise).filter(Enterprise.id == enterprise_id).first()
    if not enterprise:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    # 获取API密钥数量
    api_key_count = db.query(EnterpriseAPIKey).filter(
        EnterpriseAPIKey.enterprise_id == enterprise_id,
        EnterpriseAPIKey.is_active == True
    ).count()

    # 获取本月使用量
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage_stats = UsageService.get_enterprise_usage(db, enterprise_id, month_start, now)

    return {
        'enterprise': {
            'id': str(enterprise.id),
            'name': enterprise.name,
            'code': enterprise.code,
            'contact_name': enterprise.contact_name,
            'contact_email': enterprise.contact_email,
            'contact_phone': enterprise.contact_phone,
            'billing_email': enterprise.billing_email,
            'billing_period': enterprise.billing_period,
            'currency': enterprise.currency,
            'credit_balance': enterprise.credit_balance,
            'credit_limit': enterprise.credit_limit,
            'monthly_budget': enterprise.monthly_budget,
            'data_retention_days': enterprise.data_retention_days,
            'allow_logging': enterprise.allow_logging,
            'anonymize_data': enterprise.anonymize_data,
            'private_deployment': enterprise.private_deployment,
            'is_active': enterprise.is_active,
            'is_verified': enterprise.is_verified,
            'created_at': enterprise.created_at.isoformat() if enterprise.created_at else None
        },
        'api_key_count': api_key_count,
        'usage_this_month': usage_stats
    }


@router.put("/enterprises/{enterprise_id}")
async def update_enterprise(
    enterprise_id: UUID,
    data: EnterpriseUpdate,
    db: Session = Depends(get_db)
):
    """更新企业信息"""
    enterprise = db.query(Enterprise).filter(Enterprise.id == enterprise_id).first()
    if not enterprise:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    for key, value in data.dict(exclude_unset=True).items():
        setattr(enterprise, key, value)

    db.commit()
    return {'status': 'success'}


@router.get("/enterprises/{enterprise_id}/usage")
async def get_enterprise_usage(
    enterprise_id: UUID,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    group_by: str = Query("day", regex="^(hour|day|week|month)$"),
    db: Session = Depends(get_db)
):
    """获取企业使用量统计"""
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=30)

    return UsageService.get_enterprise_usage(db, enterprise_id, start_time, end_time, group_by)


@router.post("/enterprises/{enterprise_id}/api-keys")
async def create_api_key(
    enterprise_id: UUID,
    data: APIKeyCreate,
    db: Session = Depends(get_db)
):
    """为企业创建API密钥"""
    import secrets
    import hashlib

    enterprise = db.query(Enterprise).filter(Enterprise.id == enterprise_id).first()
    if not enterprise:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    # 生成密钥
    raw_key = f"ent_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    expires_at = None
    if data.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=data.expires_days)

    api_key = EnterpriseAPIKey(
        enterprise_id=enterprise_id,
        name=data.name,
        key_hash=key_hash,
        key_prefix=raw_key[:10],
        allowed_types=data.allowed_types,
        allowed_models=data.allowed_models,
        rate_limit_per_minute=data.rate_limit_per_minute,
        daily_limit=data.daily_limit,
        ip_whitelist=data.ip_whitelist,
        expires_at=expires_at
    )
    db.add(api_key)
    db.commit()

    # 只在创建时返回完整密钥
    return {
        'id': str(api_key.id),
        'key': raw_key,  # 仅此一次显示
        'prefix': raw_key[:10],
        'name': data.name,
        'expires_at': expires_at.isoformat() if expires_at else None
    }


@router.get("/enterprises/{enterprise_id}/api-keys")
async def list_api_keys(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """获取企业API密钥列表"""
    keys = db.query(EnterpriseAPIKey).filter(
        EnterpriseAPIKey.enterprise_id == enterprise_id
    ).all()

    return {
        'items': [{
            'id': str(k.id),
            'name': k.name,
            'prefix': k.key_prefix,
            'is_active': k.is_active,
            'allowed_types': k.allowed_types,
            'rate_limit_per_minute': k.rate_limit_per_minute,
            'total_requests': k.total_requests,
            'last_used_at': k.last_used_at.isoformat() if k.last_used_at else None,
            'expires_at': k.expires_at.isoformat() if k.expires_at else None,
            'created_at': k.created_at.isoformat() if k.created_at else None
        } for k in keys]
    }


# ============ 使用记录 ============

@router.get("/usage/records")
async def list_usage_records(
    enterprise_id: Optional[UUID] = None,
    worker_id: Optional[UUID] = None,
    machine_id: Optional[str] = None,
    usage_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """查询使用记录"""
    query = db.query(UsageRecord)

    if enterprise_id:
        query = query.filter(UsageRecord.enterprise_id == enterprise_id)
    if worker_id:
        query = query.filter(UsageRecord.worker_id == worker_id)
    if machine_id:
        query = query.filter(UsageRecord.machine_id == machine_id)
    if usage_type:
        query = query.filter(UsageRecord.usage_type == usage_type)
    if start_time:
        query = query.filter(UsageRecord.created_at >= start_time)
    if end_time:
        query = query.filter(UsageRecord.created_at <= end_time)

    total = query.count()
    records = query.order_by(UsageRecord.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': [{
            'id': str(r.id),
            'enterprise_id': str(r.enterprise_id),
            'worker_id': str(r.worker_id),
            'machine_id': r.machine_id,
            'job_id': str(r.job_id),
            'usage_type': r.usage_type,
            'job_type': r.job_type,
            'model_id': r.model_id,
            'quantity': r.quantity,
            'unit': r.unit,
            'unit_price': r.unit_price,
            'total_cost': r.total_cost,
            'gpu_seconds': r.gpu_seconds,
            'duration_ms': r.duration_ms,
            'worker_region': r.worker_region,
            'created_at': r.created_at.isoformat() if r.created_at else None
        } for r in records]
    }


@router.get("/usage/summary")
async def get_usage_summary(
    group_by: str = Query("worker", regex="^(worker|enterprise|region|type)$"),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """获取使用量汇总"""
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=7)

    from sqlalchemy import func as sqlfunc

    if group_by == "worker":
        results = db.query(
            UsageRecord.worker_id,
            UsageRecord.machine_id,
            sqlfunc.count(UsageRecord.id).label('total_jobs'),
            sqlfunc.sum(UsageRecord.total_cost).label('total_cost'),
            sqlfunc.sum(UsageRecord.gpu_seconds).label('total_gpu_seconds')
        ).filter(
            UsageRecord.created_at >= start_time,
            UsageRecord.created_at <= end_time
        ).group_by(UsageRecord.worker_id, UsageRecord.machine_id).all()

        return {
            'group_by': 'worker',
            'period': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'items': [{
                'worker_id': str(r.worker_id),
                'machine_id': r.machine_id,
                'total_jobs': r.total_jobs,
                'total_cost': round(r.total_cost or 0, 4),
                'total_gpu_seconds': round(r.total_gpu_seconds or 0, 2)
            } for r in results]
        }

    elif group_by == "enterprise":
        results = db.query(
            UsageRecord.enterprise_id,
            sqlfunc.count(UsageRecord.id).label('total_jobs'),
            sqlfunc.sum(UsageRecord.total_cost).label('total_cost'),
            sqlfunc.sum(UsageRecord.gpu_seconds).label('total_gpu_seconds')
        ).filter(
            UsageRecord.created_at >= start_time,
            UsageRecord.created_at <= end_time
        ).group_by(UsageRecord.enterprise_id).all()

        return {
            'group_by': 'enterprise',
            'period': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'items': [{
                'enterprise_id': str(r.enterprise_id),
                'total_jobs': r.total_jobs,
                'total_cost': round(r.total_cost or 0, 4),
                'total_gpu_seconds': round(r.total_gpu_seconds or 0, 2)
            } for r in results]
        }

    elif group_by == "region":
        results = db.query(
            UsageRecord.worker_region,
            sqlfunc.count(UsageRecord.id).label('total_jobs'),
            sqlfunc.sum(UsageRecord.total_cost).label('total_cost'),
            sqlfunc.sum(UsageRecord.gpu_seconds).label('total_gpu_seconds')
        ).filter(
            UsageRecord.created_at >= start_time,
            UsageRecord.created_at <= end_time
        ).group_by(UsageRecord.worker_region).all()

        return {
            'group_by': 'region',
            'period': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'items': [{
                'region': r.worker_region,
                'total_jobs': r.total_jobs,
                'total_cost': round(r.total_cost or 0, 4),
                'total_gpu_seconds': round(r.total_gpu_seconds or 0, 2)
            } for r in results]
        }

    else:  # type
        results = db.query(
            UsageRecord.usage_type,
            sqlfunc.count(UsageRecord.id).label('total_jobs'),
            sqlfunc.sum(UsageRecord.quantity).label('total_quantity'),
            sqlfunc.sum(UsageRecord.total_cost).label('total_cost')
        ).filter(
            UsageRecord.created_at >= start_time,
            UsageRecord.created_at <= end_time
        ).group_by(UsageRecord.usage_type).all()

        return {
            'group_by': 'type',
            'period': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'items': [{
                'usage_type': r.usage_type,
                'total_jobs': r.total_jobs,
                'total_quantity': round(r.total_quantity or 0, 4),
                'total_cost': round(r.total_cost or 0, 4)
            } for r in results]
        }


# ============ 账单管理 ============

@router.get("/bills")
async def list_bills(
    enterprise_id: Optional[UUID] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """获取账单列表"""
    query = db.query(Bill)

    if enterprise_id:
        query = query.filter(Bill.enterprise_id == enterprise_id)
    if status:
        query = query.filter(Bill.status == status)

    total = query.count()
    bills = query.order_by(Bill.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': [{
            'id': str(b.id),
            'enterprise_id': str(b.enterprise_id),
            'billing_period': b.billing_period,
            'period_start': b.period_start.isoformat() if b.period_start else None,
            'period_end': b.period_end.isoformat() if b.period_end else None,
            'subtotal': b.subtotal,
            'discount': b.discount,
            'tax': b.tax,
            'total': b.total,
            'currency': b.currency,
            'status': b.status,
            'invoice_number': b.invoice_number,
            'created_at': b.created_at.isoformat() if b.created_at else None,
            'due_at': b.due_at.isoformat() if b.due_at else None
        } for b in bills]
    }


@router.get("/bills/{bill_id}")
async def get_bill_detail(
    bill_id: UUID,
    db: Session = Depends(get_db)
):
    """获取账单详情"""
    bill = db.query(Bill).filter(Bill.id == bill_id).first()
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")

    return {
        'id': str(bill.id),
        'enterprise_id': str(bill.enterprise_id),
        'billing_period': bill.billing_period,
        'period_start': bill.period_start.isoformat() if bill.period_start else None,
        'period_end': bill.period_end.isoformat() if bill.period_end else None,
        'subtotal': bill.subtotal,
        'discount': bill.discount,
        'tax': bill.tax,
        'total': bill.total,
        'currency': bill.currency,
        'usage_summary': bill.usage_summary,
        'status': bill.status,
        'paid_at': bill.paid_at.isoformat() if bill.paid_at else None,
        'payment_method': bill.payment_method,
        'invoice_number': bill.invoice_number,
        'invoice_url': bill.invoice_url,
        'created_at': bill.created_at.isoformat() if bill.created_at else None,
        'due_at': bill.due_at.isoformat() if bill.due_at else None
    }


# ============ 隐私保护管理 ============

class PrivacySettingsUpdate(BaseModel):
    data_retention_days: Optional[int] = Field(None, ge=7, le=365)
    allow_logging: Optional[bool] = None
    anonymize_data: Optional[bool] = None
    private_deployment: Optional[bool] = None


@router.get("/enterprises/{enterprise_id}/privacy")
async def get_enterprise_privacy_settings(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """获取企业隐私设置"""
    privacy_service = EnterprisePrivacyService(db)
    settings = privacy_service.get_enterprise_privacy_settings(enterprise_id)

    if not settings:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    return settings


@router.put("/enterprises/{enterprise_id}/privacy")
async def update_enterprise_privacy_settings(
    enterprise_id: UUID,
    data: PrivacySettingsUpdate,
    changed_by: str = Query(..., description="操作者ID"),
    db: Session = Depends(get_db)
):
    """更新企业隐私设置"""
    privacy_service = EnterprisePrivacyService(db)

    settings_dict = data.dict(exclude_unset=True)
    if not settings_dict:
        raise HTTPException(status_code=400, detail="No settings to update")

    success = privacy_service.update_privacy_settings(
        enterprise_id, changed_by, settings_dict
    )

    if not success:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    return {'status': 'success', 'updated_fields': list(settings_dict.keys())}


@router.get("/enterprises/{enterprise_id}/privacy/compliance")
async def get_privacy_compliance_report(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """获取企业隐私合规报告"""
    audit_service = PrivacyAuditService(db)
    report = audit_service.generate_compliance_report(enterprise_id)

    if not report:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    return report


@router.get("/enterprises/{enterprise_id}/privacy/retention-status")
async def get_data_retention_status(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """获取数据保留状态"""
    retention_service = DataRetentionService(db)
    status = retention_service.get_retention_status(enterprise_id)

    if not status:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    return status


@router.post("/enterprises/{enterprise_id}/privacy/cleanup")
async def run_data_cleanup(
    enterprise_id: UUID,
    db: Session = Depends(get_db)
):
    """执行数据清理（删除过期数据）"""
    retention_service = DataRetentionService(db)
    stats = retention_service.cleanup_expired_data(enterprise_id)

    return {
        'status': 'success',
        'cleanup_stats': stats
    }


@router.post("/enterprises/{enterprise_id}/privacy/export")
async def export_enterprise_data(
    enterprise_id: UUID,
    exporter_id: str = Query(..., description="导出者ID"),
    include_sensitive: bool = Query(False, description="是否包含敏感数据"),
    db: Session = Depends(get_db)
):
    """导出企业数据（数据可携带权）"""
    privacy_service = EnterprisePrivacyService(db)

    export_json, export_data = privacy_service.export_enterprise_data(
        enterprise_id, exporter_id, 'json', include_sensitive
    )

    if not export_data:
        raise HTTPException(status_code=404, detail="Enterprise not found")

    return {
        'status': 'success',
        'data': export_data,
        'export_metadata': {
            'enterprise_id': str(enterprise_id),
            'exported_at': datetime.utcnow().isoformat(),
            'include_sensitive': include_sensitive
        }
    }


@router.delete("/enterprises/{enterprise_id}/privacy/data")
async def delete_enterprise_data(
    enterprise_id: UUID,
    requester_id: str = Query(..., description="请求者ID"),
    confirm: bool = Query(False, description="确认删除"),
    db: Session = Depends(get_db)
):
    """删除企业数据（被遗忘权）"""
    privacy_service = EnterprisePrivacyService(db)

    result = privacy_service.delete_enterprise_data(
        enterprise_id, requester_id, confirm
    )

    return result


@router.post("/privacy/scheduled-cleanup")
async def run_scheduled_cleanup(
    db: Session = Depends(get_db)
):
    """运行全平台定时清理任务"""
    privacy_service = EnterprisePrivacyService(db)
    result = privacy_service.run_scheduled_cleanup()

    return result
