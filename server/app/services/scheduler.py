"""
智能调度器 - 多区域、可靠性感知的任务分配
"""
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import logging
import math

from app.models.models import Job, Worker, JobStatus, WorkerStatus, QueueStats
from app.services.reliability import ReliabilityService

logger = logging.getLogger(__name__)


# 区域距离矩阵（用于跨区域调度）
REGION_DISTANCES = {
    ("asia-east", "asia-south"): 1,
    ("asia-east", "europe-west"): 3,
    ("asia-east", "europe-east"): 2,
    ("asia-east", "america-north"): 3,
    ("asia-east", "america-south"): 4,
    ("asia-east", "oceania"): 2,
    ("asia-south", "europe-west"): 3,
    ("asia-south", "oceania"): 2,
    ("europe-west", "europe-east"): 1,
    ("europe-west", "america-north"): 2,
    ("europe-east", "america-north"): 3,
    ("america-north", "america-south"): 2,
}


def get_region_distance(region1: str, region2: str) -> int:
    """获取两个区域之间的距离（1-5，越小越近）"""
    if region1 == region2:
        return 0

    key = tuple(sorted([region1, region2]))
    return REGION_DISTANCES.get(key, 4)


class SmartScheduler:
    """智能调度器"""

    # 权重配置
    WEIGHT_RELIABILITY = 35      # 可靠性权重
    WEIGHT_REGION = 25           # 区域匹配权重
    WEIGHT_PREDICTED_ONLINE = 20 # 预测在线时长权重
    WEIGHT_PERFORMANCE = 15      # 性能权重
    WEIGHT_LOAD = 5              # 负载权重

    def __init__(self, db: AsyncSession):
        self.db = db
        self.reliability_service = ReliabilityService(db)

    async def assign_job(self, job: Job) -> Optional[Worker]:
        """
        为任务分配最佳Worker

        考虑因素：
        1. 区域匹配（延迟）
        2. Worker可靠性
        3. 预测的在线时长
        4. GPU性能
        5. 当前负载
        """
        # 获取可用Worker
        available_workers = await self._get_available_workers(job)

        if not available_workers:
            logger.info(f"No available workers for job {job.id}")
            return None

        # 计算每个Worker的得分
        scored_workers = []
        for worker in available_workers:
            score = await self._calculate_worker_score(worker, job)
            scored_workers.append((worker, score))

        # 按得分排序
        scored_workers.sort(key=lambda x: x[1], reverse=True)

        best_worker = scored_workers[0][0]
        best_score = scored_workers[0][1]

        logger.info(
            f"Job {job.id} assigned to worker {best_worker.id} "
            f"(score: {best_score:.1f}, region: {best_worker.region})"
        )

        return best_worker

    async def _get_available_workers(self, job: Job) -> List[Worker]:
        """获取可用的Worker列表"""
        # 基础条件：在线且支持该任务类型
        conditions = [
            Worker.status == WorkerStatus.ONLINE.value,
            Worker.supported_types.contains([job.type])
        ]

        # 区域条件
        if job.preferred_region and not job.allow_cross_region:
            conditions.append(Worker.region == job.preferred_region)

        query = select(Worker).where(and_(*conditions))
        result = await self.db.execute(query)

        return list(result.scalars().all())

    async def _calculate_worker_score(self, worker: Worker, job: Job) -> float:
        """计算Worker得分"""
        score = 0.0

        # 1. 可靠性得分 (0-35分)
        reliability_score = worker.reliability_score * self.WEIGHT_RELIABILITY
        score += reliability_score

        # 2. 区域匹配得分 (0-25分)
        if job.preferred_region:
            distance = get_region_distance(worker.region, job.preferred_region)
            region_score = (5 - distance) / 5 * self.WEIGHT_REGION
        else:
            # 如果有客户端区域，使用它
            if job.client_region:
                distance = get_region_distance(worker.region, job.client_region)
                region_score = (5 - distance) / 5 * self.WEIGHT_REGION
            else:
                region_score = self.WEIGHT_REGION * 0.5  # 无区域偏好给一半分

        score += region_score

        # 3. 预测在线时长得分 (0-20分)
        estimated_duration = self._estimate_job_duration(job)
        predicted_online = self.reliability_service.predict_remaining_online_time(worker)

        if predicted_online > estimated_duration * 2:
            online_score = self.WEIGHT_PREDICTED_ONLINE
        elif predicted_online > estimated_duration:
            online_score = self.WEIGHT_PREDICTED_ONLINE * 0.7
        elif predicted_online > estimated_duration * 0.5:
            online_score = self.WEIGHT_PREDICTED_ONLINE * 0.3
        else:
            online_score = 0  # 可能完不成任务

        score += online_score

        # 4. 性能得分 (0-15分)
        if worker.gpu_memory_gb:
            # 显存越大越好（最高24GB给满分）
            perf_score = min(worker.gpu_memory_gb / 24, 1.0) * self.WEIGHT_PERFORMANCE
        else:
            perf_score = self.WEIGHT_PERFORMANCE * 0.3

        score += perf_score

        # 5. 负载得分 (0-5分)
        # 空闲的Worker得分更高
        if worker.status == WorkerStatus.ONLINE.value:
            score += self.WEIGHT_LOAD
        elif worker.status == WorkerStatus.BUSY.value:
            score += self.WEIGHT_LOAD * 0.2

        return score

    def _estimate_job_duration(self, job: Job) -> float:
        """预估任务执行时间（分钟）"""
        base_estimates = {
            "llm": 0.5,
            "image_gen": 2.0,
            "whisper": 3.0,
            "embedding": 0.2,
            "custom": 1.0,
        }

        base = base_estimates.get(job.type, 1.0)

        # 根据参数调整
        params = job.params or {}

        if job.type == "llm":
            max_tokens = params.get("max_tokens", 1000)
            base *= max_tokens / 1000

        elif job.type == "image_gen":
            steps = params.get("steps", 20)
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            pixels = width * height
            base *= (steps / 20) * (pixels / (1024 * 1024))

        return base

    async def atomic_assign_job(
        self,
        worker_id: str,
        supported_types: List[str]
    ) -> Optional[Job]:
        """
        原子性任务分配（Worker主动拉取时使用）
        使用 SELECT FOR UPDATE SKIP LOCKED 防止并发冲突
        """
        from sqlalchemy import text

        # 构建查询
        query = (
            select(Job)
            .where(
                Job.status == JobStatus.QUEUED.value,
                Job.type.in_(supported_types)
            )
            .order_by(
                Job.priority.desc(),
                Job.created_at.asc()
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        result = await self.db.execute(query)
        job = result.scalar_one_or_none()

        if not job:
            return None

        # 分配任务
        job.status = JobStatus.RUNNING.value
        job.worker_id = worker_id
        job.started_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(job)

        return job

    async def get_queue_stats(self, region: str = None) -> dict:
        """获取队列统计信息"""
        query = select(Job).where(Job.status == JobStatus.QUEUED.value)

        if region:
            query = query.where(
                or_(Job.preferred_region == region, Job.preferred_region == None)
            )

        result = await self.db.execute(query)
        queued_jobs = result.scalars().all()

        # 按类型统计
        by_type = {}
        for job in queued_jobs:
            if job.type not in by_type:
                by_type[job.type] = 0
            by_type[job.type] += 1

        # 获取可用Worker数
        worker_query = select(Worker).where(
            Worker.status == WorkerStatus.ONLINE.value
        )
        if region:
            worker_query = worker_query.where(Worker.region == region)

        worker_result = await self.db.execute(worker_query)
        available_workers = len(list(worker_result.scalars().all()))

        return {
            "total_queued": len(queued_jobs),
            "by_type": by_type,
            "available_workers": available_workers,
            "estimated_wait_seconds": self._estimate_wait_time(
                len(queued_jobs), available_workers
            )
        }

    def _estimate_wait_time(self, queued: int, workers: int) -> int:
        """预估等待时间（秒）"""
        if workers == 0:
            return -1  # 无可用Worker

        avg_job_time = 30  # 假设平均30秒
        return int(queued / max(1, workers) * avg_job_time)
