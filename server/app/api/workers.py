"""
Worker管理API - 增强版
支持：可靠性追踪、优雅下线、直连配置、Token刷新、远程配置
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid
import hashlib
import secrets
import hmac

from app.db.database import get_db
from app.models.models import Worker, Job, JobStatus, WorkerStatus
from app.services.scheduler import SmartScheduler
from app.services.reliability import ReliabilityService
from app.services.task_guarantee import TaskGuaranteeService

router = APIRouter(prefix="/api/v1/workers", tags=["workers"])


# ==================== 安全配置 ====================

TOKEN_VALIDITY_HOURS = 24
REFRESH_TOKEN_LENGTH = 64


# ==================== 辅助函数 ====================

def generate_token() -> str:
    return secrets.token_urlsafe(32)


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(REFRESH_TOKEN_LENGTH)


def generate_signing_secret() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    salt = "distributed-gpu-inference-v1"
    return hashlib.sha256(f"{salt}:{token}".encode()).hexdigest()


def verify_token_hash(token: str, token_hash: str) -> bool:
    """常量时间比较防止时序攻击"""
    computed_hash = hash_token(token)
    return hmac.compare_digest(computed_hash, token_hash)


async def verify_worker_token(
    worker_id: str,
    token: str,
    db: AsyncSession
) -> Worker:
    """验证Worker Token"""
    result = await db.execute(
        select(Worker).where(Worker.id == uuid.UUID(worker_id))
    )
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(404, "Worker not found")

    # 检查是否被锁定
    if worker.locked_until and worker.locked_until > datetime.utcnow():
        raise HTTPException(403, "Account locked")

    if not verify_token_hash(token, worker.auth_token_hash):
        # 记录失败的认证尝试
        worker.failed_auth_attempts = (worker.failed_auth_attempts or 0) + 1
        worker.last_failed_auth = datetime.utcnow()

        if worker.failed_auth_attempts >= 5:
            worker.locked_until = datetime.utcnow() + timedelta(minutes=15)

        await db.commit()
        raise HTTPException(401, "Invalid token")

    # 检查Token是否过期
    if worker.token_expires_at and worker.token_expires_at < datetime.utcnow():
        raise HTTPException(401, "Token expired")

    # 重置失败计数
    if worker.failed_auth_attempts and worker.failed_auth_attempts > 0:
        worker.failed_auth_attempts = 0
        await db.commit()

    return worker


# ==================== 请求/响应模型 ====================

class WorkerRegisterRequest(BaseModel):
    name: Optional[str] = None

    # 地理信息
    region: str = Field(..., description="区域代码，如 asia-east")
    country: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None

    # 硬件信息
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_count: int = 1
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None

    # 能力
    supported_types: List[str] = []

    # 直连配置
    direct_url: Optional[str] = Field(None, description="Worker的公网地址")
    supports_direct: bool = Field(False, description="是否支持直连")


class WorkerRegisterResponse(BaseModel):
    worker_id: str
    token: str
    refresh_token: str
    signing_secret: str
    token_expires_at: datetime
    message: str = "Worker registered successfully"


class HeartbeatRequest(BaseModel):
    status: str = Field(..., description="状态: online, busy, going_offline")
    current_job_id: Optional[str] = None

    # GPU状态
    gpu_memory_used_gb: Optional[float] = None

    # 可选更新
    supported_types: Optional[List[str]] = None
    loaded_models: Optional[List[str]] = None
    direct_url: Optional[str] = None

    # 配置版本
    config_version: int = 0


class HeartbeatResponse(BaseModel):
    status: str = "ok"
    action: Optional[str] = None  # continue, pause, shutdown, reload_config
    message: Optional[str] = None
    config_changed: bool = False


class JobAssignment(BaseModel):
    job_id: str
    type: str
    params: dict
    timeout_seconds: int = 300
    priority: int = 0


class JobCompleteRequest(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


class WorkerInfo(BaseModel):
    id: str
    name: Optional[str]
    status: str
    region: str
    gpu_model: Optional[str]
    gpu_memory_gb: Optional[float]
    supported_types: List[str]
    reliability_score: float
    success_rate: float
    total_jobs: int
    supports_direct: bool
    direct_url: Optional[str]
    last_heartbeat: Optional[datetime]


# ==================== API端点 ====================

@router.post("/register", response_model=WorkerRegisterResponse)
async def register_worker(
    payload: WorkerRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """注册新Worker"""
    token = generate_token()
    refresh_token = generate_refresh_token()
    signing_secret = generate_signing_secret()
    token_expires_at = datetime.utcnow() + timedelta(hours=TOKEN_VALIDITY_HOURS)

    worker = Worker(
        name=payload.name,
        region=payload.region,
        country=payload.country,
        city=payload.city,
        timezone=payload.timezone,
        gpu_model=payload.gpu_model,
        gpu_memory_gb=payload.gpu_memory_gb,
        gpu_count=payload.gpu_count,
        cpu_cores=payload.cpu_cores,
        ram_gb=payload.ram_gb,
        supported_types=payload.supported_types,
        direct_url=payload.direct_url,
        supports_direct=payload.supports_direct,
        auth_token_hash=hash_token(token),
        refresh_token_hash=hash_token(refresh_token),
        signing_secret=signing_secret,
        token_expires_at=token_expires_at,
        status=WorkerStatus.ONLINE.value,
        current_session_start=datetime.utcnow()
    )

    db.add(worker)
    await db.commit()
    await db.refresh(worker)

    # 开始会话追踪
    reliability = ReliabilityService(db)
    await reliability.start_session(worker)

    return WorkerRegisterResponse(
        worker_id=str(worker.id),
        token=token,
        refresh_token=refresh_token,
        signing_secret=signing_secret,
        token_expires_at=token_expires_at
    )


@router.post("/{worker_id}/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    worker_id: str,
    payload: HeartbeatRequest,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """Worker心跳"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    # 更新状态
    worker.status = payload.status
    worker.last_heartbeat = datetime.utcnow()

    if payload.current_job_id:
        worker.current_job_id = uuid.UUID(payload.current_job_id)
    else:
        worker.current_job_id = None

    if payload.gpu_memory_used_gb is not None:
        worker.gpu_memory_used_gb = payload.gpu_memory_used_gb

    if payload.supported_types:
        worker.supported_types = payload.supported_types

    if payload.loaded_models:
        worker.loaded_models = payload.loaded_models

    if payload.direct_url:
        worker.direct_url = payload.direct_url

    # 更新可靠性和在线模式
    reliability = ReliabilityService(db)
    await reliability.update_score(worker, "heartbeat")

    await db.commit()

    # 检查配置是否有更新
    config_changed = (worker.config_version or 0) > payload.config_version

    # 返回控制指令
    action = None
    message = None

    # 这里可以添加服务器端控制逻辑
    # 例如负载过高时暂停接收任务

    return HeartbeatResponse(
        status="ok",
        action=action,
        message=message,
        config_changed=config_changed
    )


@router.get("/{worker_id}/next-job", response_model=Optional[JobAssignment])
async def get_next_job(
    worker_id: str,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """获取下一个任务（原子性分配）"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    # 更新心跳
    worker.last_heartbeat = datetime.utcnow()

    # 如果Worker正在下线中，不分配新任务
    if worker.status == WorkerStatus.GOING_OFFLINE.value:
        await db.commit()
        return None

    # 原子性分配任务
    scheduler = SmartScheduler(db)
    job = await scheduler.atomic_assign_job(
        str(worker.id),
        worker.supported_types
    )

    if not job:
        await db.commit()
        return None

    # 更新Worker状态
    worker.status = WorkerStatus.BUSY.value
    worker.current_job_id = job.id

    # 记录实际执行区域
    job.actual_region = worker.region

    await db.commit()

    return JobAssignment(
        job_id=str(job.id),
        type=job.type,
        params=job.params,
        timeout_seconds=job.timeout_seconds,
        priority=job.priority
    )


@router.post("/{worker_id}/jobs/{job_id}/complete")
async def complete_job(
    worker_id: str,
    job_id: str,
    payload: JobCompleteRequest,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """完成任务"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    # 查找任务
    result = await db.execute(
        select(Job).where(Job.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job not found")

    if str(job.worker_id) != worker_id:
        raise HTTPException(403, "Not authorized to complete this job")

    # 更新任务
    job.status = JobStatus.COMPLETED.value if payload.success else JobStatus.FAILED.value
    job.result = payload.result
    job.error = payload.error
    job.completed_at = datetime.utcnow()

    if payload.processing_time_ms:
        job.actual_duration_ms = payload.processing_time_ms

    # 更新Worker
    if worker.status != WorkerStatus.GOING_OFFLINE.value:
        worker.status = WorkerStatus.ONLINE.value
    worker.current_job_id = None

    # 更新可靠性
    reliability = ReliabilityService(db)
    event = "job_completed" if payload.success else "job_failed"
    await reliability.update_score(
        worker, event,
        latency_ms=payload.processing_time_ms
    )

    await db.commit()

    return {"status": "ok", "job_id": job_id}


@router.post("/{worker_id}/going-offline")
async def notify_going_offline(
    worker_id: str,
    x_worker_token: str = Header(...),
    finish_current: bool = Query(True, description="是否完成当前任务"),
    db: AsyncSession = Depends(get_db)
):
    """通知即将下线（优雅下线）"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    if finish_current and worker.current_job_id:
        # 标记为即将下线，完成当前任务后离线
        worker.status = WorkerStatus.GOING_OFFLINE.value
    else:
        # 立即下线
        guarantee = TaskGuaranteeService(db)
        await guarantee.handle_worker_offline(worker, graceful=True)

    await db.commit()

    return {
        "status": "ok",
        "message": "Worker marked as going offline",
        "will_finish_current": finish_current and worker.current_job_id is not None
    }


@router.post("/{worker_id}/offline")
async def notify_offline(
    worker_id: str,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """通知已下线"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    guarantee = TaskGuaranteeService(db)
    await guarantee.handle_worker_offline(worker, graceful=True)

    return {"status": "ok", "message": "Worker offline recorded"}


# ==================== 安全端点 ====================

@router.post("/{worker_id}/verify")
async def verify_credentials(
    worker_id: str,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """验证凭据是否有效"""
    try:
        worker = await verify_worker_token(worker_id, x_worker_token, db)
        return {"valid": True, "worker_id": str(worker.id)}
    except HTTPException:
        return {"valid": False}


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@router.post("/{worker_id}/refresh-token")
async def refresh_token(
    worker_id: str,
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """刷新Token"""
    result = await db.execute(
        select(Worker).where(Worker.id == uuid.UUID(worker_id))
    )
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(404, "Worker not found")

    # 验证refresh_token
    if not worker.refresh_token_hash:
        raise HTTPException(400, "No refresh token set")

    if not verify_token_hash(payload.refresh_token, worker.refresh_token_hash):
        raise HTTPException(401, "Invalid refresh token")

    # 生成新Token
    new_token = generate_token()
    new_refresh_token = generate_refresh_token()
    new_expires_at = datetime.utcnow() + timedelta(hours=TOKEN_VALIDITY_HOURS)

    worker.auth_token_hash = hash_token(new_token)
    worker.refresh_token_hash = hash_token(new_refresh_token)
    worker.token_expires_at = new_expires_at

    await db.commit()

    return {
        "token": new_token,
        "refresh_token": new_refresh_token,
        "token_expires_at": new_expires_at.isoformat()
    }


@router.get("/{worker_id}/config")
async def get_worker_config(
    worker_id: str,
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """获取Worker远程配置"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    # 更新最后配置同步时间
    worker.last_config_sync = datetime.utcnow()
    await db.commit()

    # 构建配置响应
    config_override = worker.config_override or {}

    return {
        "version": worker.config_version or 0,
        "load_control": {
            "acceptance_rate": config_override.get("acceptance_rate", 1.0),
            "max_concurrent_jobs": config_override.get("max_concurrent_jobs", 1),
            "max_jobs_per_hour": config_override.get("max_jobs_per_hour", 0),
            "max_gpu_memory_percent": config_override.get("max_gpu_memory_percent", 90.0),
            "working_hours_start": config_override.get("working_hours_start"),
            "working_hours_end": config_override.get("working_hours_end"),
            "cooldown_seconds": config_override.get("cooldown_seconds", 0),
        },
        "model_configs": config_override.get("model_configs", {}),
        "security": {
            "require_signature": config_override.get("require_signature", False),
        }
    }


@router.put("/{worker_id}/config")
async def update_worker_config(
    worker_id: str,
    config: Dict[str, Any],
    x_worker_token: str = Header(...),
    db: AsyncSession = Depends(get_db)
):
    """更新Worker配置（从Worker端设置自己的负载参数）"""
    worker = await verify_worker_token(worker_id, x_worker_token, db)

    # 合并配置
    current_config = worker.config_override or {}
    current_config.update(config)
    worker.config_override = current_config
    worker.config_version = (worker.config_version or 0) + 1

    await db.commit()

    return {
        "status": "ok",
        "config_version": worker.config_version
    }


# ==================== 管理端点 ====================

@router.get("", response_model=List[WorkerInfo])
async def list_workers(
    region: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """列出所有Worker"""
    query = select(Worker)

    if region:
        query = query.where(Worker.region == region)

    if status:
        query = query.where(Worker.status == status)

    query = query.order_by(Worker.reliability_score.desc())

    result = await db.execute(query)
    workers = result.scalars().all()

    return [
        WorkerInfo(
            id=str(w.id),
            name=w.name,
            status=w.status,
            region=w.region,
            gpu_model=w.gpu_model,
            gpu_memory_gb=w.gpu_memory_gb,
            supported_types=w.supported_types or [],
            reliability_score=w.reliability_score,
            success_rate=w.success_rate,
            total_jobs=w.total_jobs,
            supports_direct=w.supports_direct,
            direct_url=w.direct_url,
            last_heartbeat=w.last_heartbeat
        )
        for w in workers
    ]


@router.get("/{worker_id}")
async def get_worker(
    worker_id: str,
    db: AsyncSession = Depends(get_db)
):
    """获取Worker详情"""
    result = await db.execute(
        select(Worker).where(Worker.id == uuid.UUID(worker_id))
    )
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(404, "Worker not found")

    # 获取可靠性预测
    reliability = ReliabilityService(db)

    return {
        "id": str(worker.id),
        "name": worker.name,
        "status": worker.status,
        "region": worker.region,
        "country": worker.country,
        "city": worker.city,
        "gpu_model": worker.gpu_model,
        "gpu_memory_gb": worker.gpu_memory_gb,
        "supported_types": worker.supported_types,
        "loaded_models": worker.loaded_models,

        # 可靠性
        "reliability_score": worker.reliability_score,
        "success_rate": worker.success_rate,
        "unexpected_offline_count": worker.unexpected_offline_count,

        # 在线统计
        "total_online_hours": worker.total_online_seconds / 3600 if worker.total_online_seconds else 0,
        "avg_session_minutes": worker.avg_session_minutes,
        "total_sessions": worker.total_sessions,
        "online_pattern": worker.online_pattern,

        # 任务统计
        "total_jobs": worker.total_jobs,
        "completed_jobs": worker.completed_jobs,
        "failed_jobs": worker.failed_jobs,
        "avg_latency_ms": worker.avg_latency_ms,

        # 直连
        "supports_direct": worker.supports_direct,
        "direct_url": worker.direct_url,

        # 预测
        "predicted_online_1h": reliability.predict_online_probability(worker, 1),
        "predicted_online_4h": reliability.predict_online_probability(worker, 4),
        "predicted_remaining_minutes": reliability.predict_remaining_online_time(worker),

        "last_heartbeat": worker.last_heartbeat,
        "registered_at": worker.registered_at
    }
