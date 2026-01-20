"""
任务API - 增强版
支持：多区域、直连模式、智能路由
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime
import uuid
import asyncio

from app.db.database import get_db
from app.models.models import Job, JobStatus, Worker, WorkerStatus
from app.services.scheduler import SmartScheduler
from app.services.task_guarantee import TaskGuaranteeService
from app.services.geo import detect_client_region

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


# ==================== 请求/响应模型 ====================

class JobCreateRequest(BaseModel):
    type: str = Field(..., description="任务类型: llm, image_gen, whisper")
    params: dict = Field(..., description="任务参数")
    priority: int = Field(default=0, description="优先级（越高越优先）")

    # 区域设置
    region: Optional[str] = Field(None, description="首选区域")
    allow_cross_region: bool = Field(True, description="是否允许跨区域执行")

    # 直连模式
    prefer_direct: bool = Field(False, description="是否优先使用直连模式")

    # 超时
    timeout_seconds: int = Field(300, description="任务超时时间")


class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

    # 执行信息
    region: Optional[str] = None
    worker_id: Optional[str] = None
    direct_url: Optional[str] = None  # 直连地址

    # 时间
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 队列信息
    queue_position: Optional[int] = None
    estimated_wait_seconds: Optional[int] = None

    class Config:
        from_attributes = True


class DirectConnectionInfo(BaseModel):
    """直连信息"""
    worker_id: str
    direct_url: str
    region: str
    gpu_model: Optional[str]
    reliability_score: float


# ==================== API端点 ====================

@router.post("", response_model=JobResponse)
async def create_job(
    request: Request,
    payload: JobCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """创建推理任务（异步）"""

    # 检测客户端区域
    client_ip = request.client.host if request.client else None
    client_region = await detect_client_region(client_ip)

    job = Job(
        type=payload.type,
        params=payload.params,
        priority=payload.priority,
        preferred_region=payload.region,
        allow_cross_region=payload.allow_cross_region,
        timeout_seconds=payload.timeout_seconds,
        client_ip=client_ip,
        client_region=client_region
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # 获取队列信息
    scheduler = SmartScheduler(db)
    queue_stats = await scheduler.get_queue_stats(region=payload.region)

    return JobResponse(
        job_id=str(job.id),
        status=job.status,
        created_at=job.created_at,
        queue_position=queue_stats["total_queued"],
        estimated_wait_seconds=queue_stats["estimated_wait_seconds"]
    )


@router.post("/sync", response_model=JobResponse)
async def create_job_sync(
    request: Request,
    payload: JobCreateRequest,
    timeout: int = Query(60, description="等待超时时间"),
    wait_for_worker: bool = Query(True, description="无Worker时是否等待"),
    db: AsyncSession = Depends(get_db)
):
    """创建推理任务（同步等待结果）"""

    # 检测客户端区域
    client_ip = request.client.host if request.client else None
    client_region = await detect_client_region(client_ip)

    # 检查是否有可用Worker
    scheduler = SmartScheduler(db)
    queue_stats = await scheduler.get_queue_stats(region=payload.region)

    if queue_stats["available_workers"] == 0:
        if not wait_for_worker:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "no_workers_available",
                    "message": "当前无可用GPU节点",
                    "suggestion": "请稍后重试，或设置 wait_for_worker=true",
                    "client_region": client_region
                }
            )

    # 创建任务（同步任务提高优先级）
    job = Job(
        type=payload.type,
        params=payload.params,
        priority=payload.priority + 10,
        preferred_region=payload.region or client_region,
        allow_cross_region=payload.allow_cross_region,
        timeout_seconds=min(payload.timeout_seconds, timeout),
        client_ip=client_ip,
        client_region=client_region
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # 等待结果
    guarantee = TaskGuaranteeService(db)
    try:
        completed_job = await guarantee.get_job_with_fallback(
            str(job.id),
            wait_if_running=True,
            max_wait_seconds=timeout
        )

        return JobResponse(
            job_id=str(completed_job.id),
            status=completed_job.status,
            result=completed_job.result,
            error=completed_job.error,
            region=completed_job.actual_region,
            worker_id=str(completed_job.worker_id) if completed_job.worker_id else None,
            created_at=completed_job.created_at,
            started_at=completed_job.started_at,
            completed_at=completed_job.completed_at
        )

    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail={
                "error": "timeout",
                "message": f"任务在 {timeout} 秒内未完成",
                "job_id": str(job.id)
            }
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """查询任务状态"""
    result = await db.execute(
        select(Job).where(Job.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job not found")

    return JobResponse(
        job_id=str(job.id),
        status=job.status,
        result=job.result,
        error=job.error,
        region=job.actual_region,
        worker_id=str(job.worker_id) if job.worker_id else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.delete("/{job_id}")
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """取消任务"""
    result = await db.execute(
        select(Job).where(Job.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job not found")

    if job.status not in [JobStatus.QUEUED.value]:
        raise HTTPException(400, "只能取消排队中的任务")

    job.status = JobStatus.CANCELLED.value
    await db.commit()

    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/{job_id}/direct", response_model=DirectConnectionInfo)
async def get_direct_connection(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    获取任务的直连信息

    用于P2P直连模式，客户端可以直接与Worker通信
    """
    result = await db.execute(
        select(Job).where(Job.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job not found")

    if not job.worker_id:
        raise HTTPException(400, "任务尚未分配Worker")

    # 获取Worker信息
    worker_result = await db.execute(
        select(Worker).where(Worker.id == job.worker_id)
    )
    worker = worker_result.scalar_one_or_none()

    if not worker:
        raise HTTPException(404, "Worker not found")

    if not worker.supports_direct or not worker.direct_url:
        raise HTTPException(400, "该Worker不支持直连模式")

    return DirectConnectionInfo(
        worker_id=str(worker.id),
        direct_url=worker.direct_url,
        region=worker.region,
        gpu_model=worker.gpu_model,
        reliability_score=worker.reliability_score
    )


# ==================== 直连路由 ====================

@router.get("/direct/nearest")
async def get_nearest_worker(
    request: Request,
    job_type: str = Query(..., description="任务类型"),
    db: AsyncSession = Depends(get_db)
):
    """
    获取最近的支持直连的Worker

    用于客户端直接与Worker建立连接，跳过服务器中转
    """
    client_ip = request.client.host if request.client else None
    client_region = await detect_client_region(client_ip)

    # 查找支持直连的在线Worker
    result = await db.execute(
        select(Worker)
        .where(
            Worker.status == WorkerStatus.ONLINE.value,
            Worker.supports_direct == True,
            Worker.direct_url != None,
            Worker.supported_types.contains([job_type])
        )
        .order_by(Worker.reliability_score.desc())
    )
    workers = list(result.scalars().all())

    if not workers:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "no_direct_workers",
                "message": "当前无支持直连的Worker"
            }
        )

    # 按区域距离排序
    from app.services.scheduler import get_region_distance

    sorted_workers = sorted(
        workers,
        key=lambda w: (
            get_region_distance(w.region, client_region),
            -w.reliability_score
        )
    )

    best = sorted_workers[0]

    return {
        "worker_id": str(best.id),
        "direct_url": best.direct_url,
        "region": best.region,
        "client_region": client_region,
        "gpu_model": best.gpu_model,
        "reliability_score": best.reliability_score
    }


# ==================== 队列状态 ====================

@router.get("/stats/queue")
async def get_queue_stats(
    region: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """获取队列状态统计"""
    scheduler = SmartScheduler(db)
    stats = await scheduler.get_queue_stats(region=region)

    return stats
