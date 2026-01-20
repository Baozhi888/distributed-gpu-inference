"""
任务保障服务 - 处理失败、重试、降级
"""
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
import logging
import asyncio

from app.models.models import Job, Worker, JobStatus, WorkerStatus
from app.services.reliability import ReliabilityService

logger = logging.getLogger(__name__)


class TaskGuaranteeService:
    """任务保障服务"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.reliability_service = ReliabilityService(db)

    async def handle_worker_offline(
        self,
        worker: Worker,
        graceful: bool = False
    ):
        """
        处理Worker下线

        Args:
            worker: 下线的Worker
            graceful: 是否是优雅下线
        """
        logger.info(
            f"Worker {worker.id} going offline "
            f"({'graceful' if graceful else 'unexpected'})"
        )

        # 查找正在执行的任务
        result = await self.db.execute(
            select(Job)
            .where(Job.worker_id == worker.id)
            .where(Job.status == JobStatus.RUNNING.value)
        )
        running_jobs = list(result.scalars().all())

        if running_jobs:
            for job in running_jobs:
                await self._handle_interrupted_job(job, worker, graceful)

        # 更新Worker状态和可靠性
        worker.status = WorkerStatus.OFFLINE.value
        worker.current_job_id = None

        event = "graceful_offline" if graceful else "unexpected_offline"
        await self.reliability_service.update_score(worker, event)

    async def _handle_interrupted_job(
        self,
        job: Job,
        worker: Worker,
        graceful: bool
    ):
        """处理被中断的任务"""
        logger.warning(f"Job {job.id} interrupted by worker {worker.id} offline")

        if graceful:
            # 优雅下线，任务可能已接近完成
            # 等待一小段时间看是否能收到结果
            logger.info(f"Waiting for job {job.id} final result...")
            # 实际实现中可能需要不同的处理
            pass

        # 检查重试次数
        if job.retry_count < job.max_retries:
            # 重新放回队列
            job.status = JobStatus.QUEUED.value
            job.worker_id = None
            job.retry_count += 1
            job.started_at = None

            logger.info(
                f"Job {job.id} returned to queue "
                f"(retry {job.retry_count}/{job.max_retries})"
            )
        else:
            # 超过重试次数，标记失败
            job.status = JobStatus.FAILED.value
            job.error = f"Exceeded max retries ({job.max_retries}) due to worker failures"
            job.completed_at = datetime.utcnow()

            logger.error(f"Job {job.id} failed after {job.retry_count} retries")

        await self.db.commit()

    async def check_stale_jobs(self, timeout_minutes: int = 30) -> int:
        """
        检查超时任务，将其重新放回队列

        Returns:
            重新排队的任务数
        """
        timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        # 查找超时的运行中任务
        result = await self.db.execute(
            select(Job)
            .where(
                Job.status == JobStatus.RUNNING.value,
                Job.started_at < timeout_threshold
            )
        )
        stale_jobs = list(result.scalars().all())

        requeued_count = 0

        for job in stale_jobs:
            # 检查Worker是否还在线
            if job.worker_id:
                worker_result = await self.db.execute(
                    select(Worker).where(Worker.id == job.worker_id)
                )
                worker = worker_result.scalar_one_or_none()

                if worker and worker.status in [WorkerStatus.ONLINE.value, WorkerStatus.BUSY.value]:
                    # Worker还在线，可能只是任务很慢
                    # 检查是否超过任务自身的超时设置
                    job_timeout = job.timeout_seconds or 300
                    elapsed = (datetime.utcnow() - job.started_at).total_seconds()

                    if elapsed < job_timeout:
                        continue  # 未超时，跳过

                    # 超时了，标记Worker可靠性下降
                    await self.reliability_service.update_score(worker, "job_failed")

            # 重新排队或标记失败
            if job.retry_count < job.max_retries:
                job.status = JobStatus.QUEUED.value
                job.worker_id = None
                job.retry_count += 1
                job.started_at = None
                requeued_count += 1
                logger.info(f"Stale job {job.id} requeued")
            else:
                job.status = JobStatus.FAILED.value
                job.error = "Task timeout"
                job.completed_at = datetime.utcnow()
                logger.warning(f"Stale job {job.id} marked as failed")

        await self.db.commit()

        if requeued_count > 0:
            logger.info(f"Requeued {requeued_count} stale jobs")

        return requeued_count

    async def check_dead_workers(self, timeout_seconds: int = 90) -> List[Worker]:
        """
        检查心跳超时的Worker

        Returns:
            被标记为离线的Worker列表
        """
        timeout_threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)

        result = await self.db.execute(
            select(Worker)
            .where(
                Worker.status.in_([WorkerStatus.ONLINE.value, WorkerStatus.BUSY.value]),
                Worker.last_heartbeat < timeout_threshold
            )
        )
        dead_workers = list(result.scalars().all())

        for worker in dead_workers:
            logger.warning(
                f"Worker {worker.id} heartbeat timeout "
                f"(last: {worker.last_heartbeat})"
            )
            await self.handle_worker_offline(worker, graceful=False)

        return dead_workers

    async def get_job_with_fallback(
        self,
        job_id: str,
        wait_if_running: bool = True,
        max_wait_seconds: int = 60
    ) -> Job:
        """
        获取任务结果，支持等待和降级

        Args:
            job_id: 任务ID
            wait_if_running: 如果任务正在运行，是否等待
            max_wait_seconds: 最大等待时间
        """
        from uuid import UUID

        start_time = datetime.utcnow()

        while True:
            result = await self.db.execute(
                select(Job).where(Job.id == UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                raise ValueError(f"Job {job_id} not found")

            # 如果已完成，直接返回
            if job.status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                return job

            # 如果不等待，直接返回当前状态
            if not wait_if_running:
                return job

            # 检查是否超时
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed >= max_wait_seconds:
                raise TimeoutError(f"Waiting for job {job_id} timed out")

            # 等待一段时间后重试
            await asyncio.sleep(0.5)


class TaskGuaranteeBackgroundWorker:
    """后台任务保障工作线程"""

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.running = False

    async def start(self):
        """启动后台检查"""
        self.running = True
        logger.info("Task guarantee background worker started")

        while self.running:
            try:
                async with self.db_session_factory() as db:
                    service = TaskGuaranteeService(db)

                    # 检查死掉的Worker
                    await service.check_dead_workers(timeout_seconds=90)

                    # 检查超时任务
                    await service.check_stale_jobs(timeout_minutes=30)

            except Exception as e:
                logger.error(f"Background check error: {e}")

            # 每30秒检查一次
            await asyncio.sleep(30)

    def stop(self):
        """停止后台检查"""
        self.running = False
        logger.info("Task guarantee background worker stopped")
