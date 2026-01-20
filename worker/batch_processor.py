"""
连续批处理器 (Continuous Batcher)

实现动态批处理，将多个请求合并为一个批次执行，
提升 GPU 利用率和整体吞吐量。

支持：
- 动态批处理大小调整
- 请求优先级队列
- 超时控制
- 前缀共享优化
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """请求优先级"""
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True)
class PendingRequest:
    """待处理请求"""
    priority: int
    timestamp: float
    job_id: str = field(compare=False)
    params: Dict[str, Any] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    prefix_hash: str = field(compare=False, default="")

    @classmethod
    def create(
        cls,
        job_id: str,
        params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        prefix_hash: str = ""
    ) -> "PendingRequest":
        return cls(
            priority=priority.value,
            timestamp=time.time(),
            job_id=job_id,
            params=params,
            future=asyncio.Future(),
            prefix_hash=prefix_hash
        )


class ContinuousBatcher:
    """
    连续批处理器

    将多个推理请求动态合并为批次执行，支持：
    - 最大批处理大小限制
    - 最大等待时间控制
    - 优先级队列
    - 前缀共享批处理
    """

    def __init__(
        self,
        engine,
        max_batch_size: int = 32,
        max_wait_ms: float = 50,
        enable_prefix_grouping: bool = True,
        max_queue_size: int = 1000,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.enable_prefix_grouping = enable_prefix_grouping
        self.max_queue_size = max_queue_size

        # 请求队列（优先级堆）
        self._pending: List[PendingRequest] = []
        self._pending_by_prefix: Dict[str, List[PendingRequest]] = defaultdict(list)

        # 批处理任务
        self._batch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # 统计信息
        self._stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time_ms": 0.0,
        }

        # 运行状态
        self._running = False

    async def start(self) -> None:
        """启动批处理器"""
        self._running = True
        logger.info("ContinuousBatcher started")

    async def stop(self) -> None:
        """停止批处理器"""
        self._running = False

        # 取消所有待处理请求
        async with self._lock:
            for req in self._pending:
                if not req.future.done():
                    req.future.cancel()
            self._pending.clear()
            self._pending_by_prefix.clear()

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        logger.info("ContinuousBatcher stopped")

    async def submit(
        self,
        job_id: str,
        params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        提交推理请求

        Args:
            job_id: 任务ID
            params: 推理参数
            priority: 请求优先级
            timeout: 超时时间（秒）

        Returns:
            推理结果
        """
        if not self._running:
            raise RuntimeError("Batcher is not running")

        if len(self._pending) >= self.max_queue_size:
            raise RuntimeError(f"Queue full (max={self.max_queue_size})")

        # 计算前缀哈希（用于分组）
        prefix_hash = ""
        if self.enable_prefix_grouping:
            prefix_hash = self._compute_prefix_hash(params)

        # 创建请求
        request = PendingRequest.create(
            job_id=job_id,
            params=params,
            priority=priority,
            prefix_hash=prefix_hash
        )

        async with self._lock:
            heapq.heappush(self._pending, request)

            if self.enable_prefix_grouping and prefix_hash:
                self._pending_by_prefix[prefix_hash].append(request)

            self._stats["total_requests"] += 1

            # 检查是否应该触发批处理
            if len(self._pending) >= self.max_batch_size:
                # 立即处理满批次
                asyncio.create_task(self._process_batch())
            elif self._batch_task is None or self._batch_task.done():
                # 启动等待定时器
                self._batch_task = asyncio.create_task(self._wait_and_process())

        # 等待结果
        try:
            return await asyncio.wait_for(request.future, timeout=timeout)
        except asyncio.TimeoutError:
            # 超时，尝试从队列中移除
            async with self._lock:
                try:
                    self._pending.remove(request)
                    heapq.heapify(self._pending)
                except ValueError:
                    pass  # 可能已被处理
            raise

    async def _wait_and_process(self) -> None:
        """等待指定时间后处理批次"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._process_batch()

    async def _process_batch(self) -> None:
        """处理一个批次"""
        async with self._lock:
            if not self._pending:
                return

            batch_start_time = time.time()

            # 选择要处理的请求
            if self.enable_prefix_grouping:
                batch = self._select_batch_with_prefix_grouping()
            else:
                batch = self._select_batch_simple()

            if not batch:
                return

            # 从队列中移除
            for req in batch:
                try:
                    self._pending.remove(req)
                except ValueError:
                    pass
                if req.prefix_hash:
                    try:
                        self._pending_by_prefix[req.prefix_hash].remove(req)
                    except ValueError:
                        pass
            heapq.heapify(self._pending)

        # 执行批量推理（在锁外执行）
        try:
            results = await self._execute_batch(batch)

            # 设置结果
            for req, result in zip(batch, results):
                if not req.future.done():
                    if isinstance(result, Exception):
                        req.future.set_exception(result)
                    else:
                        req.future.set_result(result)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # 设置所有请求失败
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

        # 更新统计
        batch_time = (time.time() - batch_start_time) * 1000
        self._stats["total_batches"] += 1
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * (self._stats["total_batches"] - 1) + len(batch))
            / self._stats["total_batches"]
        )
        self._stats["avg_wait_time_ms"] = (
            (self._stats["avg_wait_time_ms"] * (self._stats["total_batches"] - 1) + batch_time)
            / self._stats["total_batches"]
        )

    def _select_batch_simple(self) -> List[PendingRequest]:
        """简单的批次选择（按优先级）"""
        return [heapq.heappop(self._pending) for _ in range(min(len(self._pending), self.max_batch_size))]

    def _select_batch_with_prefix_grouping(self) -> List[PendingRequest]:
        """带前缀分组的批次选择"""
        batch = []

        # 首先尝试找到最大的前缀组
        if self._pending_by_prefix:
            # 按组大小排序
            sorted_groups = sorted(
                self._pending_by_prefix.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            for prefix_hash, group in sorted_groups:
                if not group:
                    continue

                # 取该组的请求
                take_count = min(len(group), self.max_batch_size - len(batch))
                batch.extend(group[:take_count])

                if len(batch) >= self.max_batch_size:
                    break

        # 如果还有空间，添加没有前缀的请求
        remaining = self.max_batch_size - len(batch)
        if remaining > 0:
            no_prefix_requests = [
                req for req in self._pending
                if not req.prefix_hash and req not in batch
            ]
            batch.extend(no_prefix_requests[:remaining])

        return batch

    async def _execute_batch(
        self,
        batch: List[PendingRequest]
    ) -> List[Any]:
        """执行批量推理"""
        params_list = [req.params for req in batch]

        if hasattr(self.engine, "batch_inference_async"):
            return await self.engine.batch_inference_async(params_list)
        elif hasattr(self.engine, "batch_inference"):
            # 在线程池中执行同步方法
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.engine.batch_inference,
                params_list
            )
        else:
            # 回退到串行执行
            results = []
            for params in params_list:
                try:
                    if hasattr(self.engine, "inference_async"):
                        result = await self.engine.inference_async(params)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            self.engine.inference,
                            params
                        )
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results

    def _compute_prefix_hash(self, params: Dict[str, Any]) -> str:
        """计算请求的前缀哈希"""
        import hashlib

        messages = params.get("messages", [])
        if not messages:
            return ""

        # 使用系统消息作为前缀
        system_messages = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system"
        ]

        if not system_messages:
            return ""

        prefix_str = "".join(system_messages)
        return hashlib.sha256(prefix_str.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "queue_size": len(self._pending),
            "prefix_groups": len(self._pending_by_prefix),
        }


class AdaptiveBatcher(ContinuousBatcher):
    """
    自适应批处理器

    根据负载和延迟要求动态调整批处理参数
    """

    def __init__(
        self,
        engine,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        target_latency_ms: float = 100,
        **kwargs
    ):
        super().__init__(engine, max_batch_size=max_batch_size, **kwargs)
        self.min_batch_size = min_batch_size
        self.target_latency_ms = target_latency_ms

        # 自适应参数
        self._current_batch_size = max_batch_size // 2
        self._latency_history: List[float] = []
        self._max_history = 100

    async def _process_batch(self) -> None:
        """处理批次并自适应调整参数"""
        start_time = time.time()

        # 使用当前自适应的批次大小
        original_max = self.max_batch_size
        self.max_batch_size = self._current_batch_size

        await super()._process_batch()

        self.max_batch_size = original_max

        # 记录延迟
        latency_ms = (time.time() - start_time) * 1000
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._max_history:
            self._latency_history.pop(0)

        # 自适应调整
        self._adapt_batch_size()

    def _adapt_batch_size(self) -> None:
        """根据延迟历史调整批处理大小"""
        if len(self._latency_history) < 10:
            return

        avg_latency = sum(self._latency_history[-10:]) / 10

        if avg_latency > self.target_latency_ms * 1.2:
            # 延迟过高，减小批次大小
            self._current_batch_size = max(
                self.min_batch_size,
                int(self._current_batch_size * 0.8)
            )
        elif avg_latency < self.target_latency_ms * 0.8:
            # 延迟较低，增大批次大小
            self._current_batch_size = min(
                self.max_batch_size,
                int(self._current_batch_size * 1.2)
            )

        logger.debug(
            f"Adaptive batch size: {self._current_batch_size} "
            f"(avg latency: {avg_latency:.1f}ms)"
        )
