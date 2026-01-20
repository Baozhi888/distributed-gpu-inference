"""
可靠性服务 - 评估和追踪Worker可靠性
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import logging

from app.models.models import Worker, WorkerStatus

logger = logging.getLogger(__name__)


class ReliabilityService:
    """Worker可靠性评估服务"""

    # 评分调整参数
    SCORE_JOB_COMPLETED = 0.02      # 完成任务加分
    SCORE_JOB_FAILED = -0.05        # 任务失败扣分
    SCORE_UNEXPECTED_OFFLINE = -0.15 # 意外掉线扣分
    SCORE_GRACEFUL_OFFLINE = -0.02  # 正常下线轻微扣分
    SCORE_LONG_SESSION = 0.05       # 长时间在线加分
    SCORE_QUICK_RESPONSE = 0.01     # 快速响应加分

    LONG_SESSION_THRESHOLD = 4 * 3600  # 4小时算长会话

    def __init__(self, db: AsyncSession):
        self.db = db

    async def update_score(self, worker: Worker, event: str, **kwargs):
        """
        根据事件更新可靠性评分

        Events:
            - job_completed: 任务完成
            - job_failed: 任务失败
            - unexpected_offline: 意外掉线
            - graceful_offline: 正常下线
            - long_session: 长时间在线
            - heartbeat: 心跳（用于更新在线模式）
        """
        old_score = worker.reliability_score

        if event == "job_completed":
            worker.reliability_score = min(1.0, worker.reliability_score + self.SCORE_JOB_COMPLETED)
            worker.completed_jobs += 1
            worker.total_jobs += 1
            self._update_success_rate(worker)

        elif event == "job_failed":
            worker.reliability_score = max(0.1, worker.reliability_score + self.SCORE_JOB_FAILED)
            worker.failed_jobs += 1
            worker.total_jobs += 1
            self._update_success_rate(worker)

        elif event == "unexpected_offline":
            worker.reliability_score = max(0.1, worker.reliability_score + self.SCORE_UNEXPECTED_OFFLINE)
            worker.unexpected_offline_count += 1
            logger.warning(f"Worker {worker.id} unexpected offline, score: {old_score:.2f} -> {worker.reliability_score:.2f}")

        elif event == "graceful_offline":
            worker.reliability_score = max(0.2, worker.reliability_score + self.SCORE_GRACEFUL_OFFLINE)
            await self._end_session(worker)

        elif event == "long_session":
            worker.reliability_score = min(1.0, worker.reliability_score + self.SCORE_LONG_SESSION)

        elif event == "heartbeat":
            await self._update_online_pattern(worker)
            # 检查是否达到长会话
            if worker.current_session_start:
                session_duration = (datetime.utcnow() - worker.current_session_start).total_seconds()
                if session_duration > self.LONG_SESSION_THRESHOLD:
                    # 每4小时只加一次分
                    hours_online = int(session_duration / 3600)
                    if hours_online > 0 and hours_online % 4 == 0:
                        await self.update_score(worker, "long_session")

        # 响应时间影响
        if "latency_ms" in kwargs:
            latency = kwargs["latency_ms"]
            if latency < 100:  # 快速响应
                worker.reliability_score = min(1.0, worker.reliability_score + self.SCORE_QUICK_RESPONSE)
            # 更新平均延迟
            if worker.avg_latency_ms:
                worker.avg_latency_ms = int(worker.avg_latency_ms * 0.9 + latency * 0.1)
            else:
                worker.avg_latency_ms = latency

        await self.db.commit()

    def _update_success_rate(self, worker: Worker):
        """更新成功率"""
        if worker.total_jobs > 0:
            worker.success_rate = worker.completed_jobs / worker.total_jobs

    async def _update_online_pattern(self, worker: Worker):
        """更新在线时段模式"""
        current_hour = datetime.utcnow().hour
        pattern = worker.online_pattern or {str(i): 0.0 for i in range(24)}

        # 指数移动平均更新
        alpha = 0.1
        hour_key = str(current_hour)
        pattern[hour_key] = alpha * 1.0 + (1 - alpha) * pattern.get(hour_key, 0.0)

        worker.online_pattern = pattern

    async def _end_session(self, worker: Worker):
        """结束会话，更新统计"""
        if worker.current_session_start:
            session_duration = (datetime.utcnow() - worker.current_session_start).total_seconds()
            worker.total_online_seconds += int(session_duration)
            worker.total_sessions += 1

            # 更新平均会话时长
            worker.avg_session_minutes = (
                worker.total_online_seconds / 60 / max(1, worker.total_sessions)
            )

            worker.current_session_start = None

    async def start_session(self, worker: Worker):
        """开始新会话"""
        worker.current_session_start = datetime.utcnow()
        worker.total_sessions += 1
        await self.db.commit()

    def predict_online_probability(self, worker: Worker, hours_ahead: int = 1) -> float:
        """预测Worker在未来N小时是否在线的概率"""
        if not worker.online_pattern:
            return 0.5

        future_hour = (datetime.utcnow().hour + hours_ahead) % 24
        base_probability = worker.online_pattern.get(str(future_hour), 0.5)

        # 结合可靠性分数调整
        adjusted = base_probability * (0.5 + 0.5 * worker.reliability_score)

        return min(1.0, adjusted)

    def predict_remaining_online_time(self, worker: Worker) -> float:
        """预测Worker还能在线多久（分钟）"""
        if not worker.current_session_start:
            return 0

        session_duration = (datetime.utcnow() - worker.current_session_start).total_seconds() / 60
        avg_session = worker.avg_session_minutes or 60

        # 简单预测：平均时长 - 已在线时长
        remaining = max(5, avg_session - session_duration)

        # 可靠性高的Worker，预测时间更长
        remaining *= (0.5 + 0.5 * worker.reliability_score)

        return remaining

    async def get_reliable_workers(
        self,
        min_score: float = 0.5,
        region: str = None,
        job_type: str = None
    ):
        """获取可靠的Worker列表"""
        query = select(Worker).where(
            Worker.status.in_([WorkerStatus.ONLINE.value, WorkerStatus.BUSY.value]),
            Worker.reliability_score >= min_score
        )

        if region:
            query = query.where(Worker.region == region)

        if job_type:
            query = query.where(Worker.supported_types.contains([job_type]))

        query = query.order_by(Worker.reliability_score.desc())

        result = await self.db.execute(query)
        return result.scalars().all()
