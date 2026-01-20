"""
Prefill/Decode 分离调度器

实现 DistServe 风格的 Prefill/Decode 分离架构：
- Prefill 阶段：计算密集，适合高算力 GPU
- Decode 阶段：内存密集，适合高带宽 GPU

核心优化：
- 分离调度，各阶段独立优化
- KV-Cache 迁移优化
- 动态负载均衡
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class JobPhase(Enum):
    """任务阶段"""
    PREFILL = "prefill"
    DECODE = "decode"


class WorkerRole(Enum):
    """Worker 角色"""
    PREFILL = "prefill"      # 专门处理 Prefill
    DECODE = "decode"        # 专门处理 Decode
    HYBRID = "hybrid"        # 混合模式


@dataclass
class WorkerCapability:
    """Worker 能力描述"""
    worker_id: str
    role: WorkerRole = WorkerRole.HYBRID

    # 硬件能力
    compute_flops: float = 0.0      # TFLOPS（用于 Prefill）
    memory_bandwidth_gbps: float = 0.0  # GB/s（用于 Decode）
    gpu_memory_gb: float = 0.0

    # 当前状态
    active_prefill_jobs: int = 0
    active_decode_jobs: int = 0
    kv_cache_tokens_used: int = 0
    kv_cache_tokens_total: int = 0

    # 性能指标
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    reliability_score: float = 1.0

    @property
    def prefill_capacity(self) -> float:
        """Prefill 容量评分"""
        if self.role == WorkerRole.DECODE:
            return 0.0
        return self.compute_flops * self.reliability_score

    @property
    def decode_capacity(self) -> float:
        """Decode 容量评分"""
        if self.role == WorkerRole.PREFILL:
            return 0.0
        return self.memory_bandwidth_gbps * self.reliability_score

    @property
    def kv_cache_utilization(self) -> float:
        """KV-Cache 利用率"""
        if self.kv_cache_tokens_total == 0:
            return 0.0
        return self.kv_cache_tokens_used / self.kv_cache_tokens_total


@dataclass(order=True)
class PendingJob:
    """待处理任务"""
    priority: float
    created_at: float
    job_id: str = field(compare=False)
    phase: JobPhase = field(compare=False)
    prompt_tokens: int = field(compare=False, default=0)
    max_tokens: int = field(compare=False, default=512)
    kv_cache_key: str = field(compare=False, default="")
    kv_cache_worker: str = field(compare=False, default="")  # 持有 KV-Cache 的 Worker
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class WorkerAssignment:
    """Worker 分配结果"""
    worker_id: str
    phase: JobPhase
    estimated_latency_ms: float = 0.0
    kv_migration_needed: bool = False
    migration_source: str = ""


class PrefillDecodeScheduler:
    """
    Prefill/Decode 分离调度器

    核心思想（参考 DistServe）：
    1. Prefill 阶段由计算能力强的 Worker 处理
    2. Decode 阶段由内存带宽高的 Worker 处理
    3. 两个阶段可以在不同的 Worker 上执行
    4. 需要在阶段切换时迁移 KV-Cache
    """

    def __init__(
        self,
        enable_migration: bool = True,
        migration_threshold_ms: float = 50.0,  # 迁移延迟阈值
        prefill_batch_timeout_ms: float = 20.0,
        decode_batch_timeout_ms: float = 5.0,
    ):
        self.enable_migration = enable_migration
        self.migration_threshold_ms = migration_threshold_ms
        self.prefill_batch_timeout_ms = prefill_batch_timeout_ms
        self.decode_batch_timeout_ms = decode_batch_timeout_ms

        # Worker 信息
        self._workers: Dict[str, WorkerCapability] = {}

        # 任务队列（按阶段分开）
        self._prefill_queue: List[PendingJob] = []
        self._decode_queue: List[PendingJob] = []

        # KV-Cache 位置索引
        self._kv_cache_locations: Dict[str, str] = {}  # kv_key -> worker_id

        # 统计信息
        self._stats = {
            "prefill_jobs": 0,
            "decode_jobs": 0,
            "migrations": 0,
            "migration_bytes": 0,
            "avg_prefill_latency_ms": 0.0,
            "avg_decode_latency_ms": 0.0,
        }

    def register_worker(
        self,
        worker_id: str,
        capability: WorkerCapability
    ) -> None:
        """注册 Worker"""
        self._workers[worker_id] = capability
        logger.info(f"Registered worker {worker_id} with role {capability.role}")

    def unregister_worker(self, worker_id: str) -> None:
        """注销 Worker"""
        self._workers.pop(worker_id, None)
        logger.info(f"Unregistered worker {worker_id}")

    def update_worker_stats(
        self,
        worker_id: str,
        stats: Dict[str, Any]
    ) -> None:
        """更新 Worker 状态"""
        if worker_id not in self._workers:
            return

        worker = self._workers[worker_id]
        worker.active_prefill_jobs = stats.get("active_prefill_jobs", worker.active_prefill_jobs)
        worker.active_decode_jobs = stats.get("active_decode_jobs", worker.active_decode_jobs)
        worker.kv_cache_tokens_used = stats.get("kv_cache_tokens_used", worker.kv_cache_tokens_used)
        worker.prefill_latency_ms = stats.get("prefill_latency_ms", worker.prefill_latency_ms)
        worker.decode_latency_ms = stats.get("decode_latency_ms", worker.decode_latency_ms)

    async def submit_job(
        self,
        job_id: str,
        prompt_tokens: int,
        max_tokens: int = 512,
        priority: float = 1.0,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        提交新任务

        新任务从 Prefill 阶段开始
        """
        job = PendingJob(
            priority=-priority,  # 负数以实现最大堆
            created_at=time.time(),
            job_id=job_id,
            phase=JobPhase.PREFILL,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            metadata=metadata or {},
        )

        heapq.heappush(self._prefill_queue, job)
        self._stats["prefill_jobs"] += 1

        return job_id

    async def transition_to_decode(
        self,
        job_id: str,
        kv_cache_key: str,
        kv_cache_worker: str,
    ) -> None:
        """
        将任务从 Prefill 阶段转换到 Decode 阶段

        Args:
            job_id: 任务 ID
            kv_cache_key: KV-Cache 键
            kv_cache_worker: 持有 KV-Cache 的 Worker
        """
        job = PendingJob(
            priority=0,  # Decode 按 FIFO 顺序
            created_at=time.time(),
            job_id=job_id,
            phase=JobPhase.DECODE,
            kv_cache_key=kv_cache_key,
            kv_cache_worker=kv_cache_worker,
        )

        heapq.heappush(self._decode_queue, job)
        self._kv_cache_locations[kv_cache_key] = kv_cache_worker
        self._stats["decode_jobs"] += 1

    async def assign_job(self, job: PendingJob) -> WorkerAssignment:
        """
        分配任务到 Worker

        根据任务阶段选择最合适的 Worker
        """
        if job.phase == JobPhase.PREFILL:
            return await self._assign_prefill(job)
        else:
            return await self._assign_decode(job)

    async def _assign_prefill(self, job: PendingJob) -> WorkerAssignment:
        """分配 Prefill 任务"""
        # 选择计算能力最强的可用 Worker
        candidates = [
            (w.prefill_capacity / (1 + w.active_prefill_jobs), w)
            for w in self._workers.values()
            if w.role in (WorkerRole.PREFILL, WorkerRole.HYBRID)
            and w.prefill_capacity > 0
        ]

        if not candidates:
            raise RuntimeError("No available workers for prefill")

        # 选择评分最高的
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, selected = candidates[0]

        # 估算延迟
        estimated_latency = self._estimate_prefill_latency(
            selected,
            job.prompt_tokens
        )

        return WorkerAssignment(
            worker_id=selected.worker_id,
            phase=JobPhase.PREFILL,
            estimated_latency_ms=estimated_latency,
        )

    async def _assign_decode(self, job: PendingJob) -> WorkerAssignment:
        """分配 Decode 任务"""
        # 首选：持有 KV-Cache 的 Worker
        kv_holder = job.kv_cache_worker or self._kv_cache_locations.get(job.kv_cache_key)

        if kv_holder and kv_holder in self._workers:
            holder_worker = self._workers[kv_holder]

            # 检查是否适合做 Decode
            if holder_worker.role in (WorkerRole.DECODE, WorkerRole.HYBRID):
                return WorkerAssignment(
                    worker_id=kv_holder,
                    phase=JobPhase.DECODE,
                    estimated_latency_ms=self._estimate_decode_latency(holder_worker),
                    kv_migration_needed=False,
                )

        # 需要选择其他 Worker
        candidates = [
            (w.decode_capacity / (1 + w.active_decode_jobs), w)
            for w in self._workers.values()
            if w.role in (WorkerRole.DECODE, WorkerRole.HYBRID)
            and w.decode_capacity > 0
        ]

        if not candidates:
            raise RuntimeError("No available workers for decode")

        candidates.sort(reverse=True, key=lambda x: x[0])
        _, selected = candidates[0]

        # 计算是否需要迁移
        need_migration = (
            self.enable_migration
            and kv_holder
            and kv_holder != selected.worker_id
        )

        estimated_latency = self._estimate_decode_latency(selected)
        if need_migration:
            # 加上迁移延迟
            estimated_latency += self.migration_threshold_ms

        return WorkerAssignment(
            worker_id=selected.worker_id,
            phase=JobPhase.DECODE,
            estimated_latency_ms=estimated_latency,
            kv_migration_needed=need_migration,
            migration_source=kv_holder if need_migration else "",
        )

    def _estimate_prefill_latency(
        self,
        worker: WorkerCapability,
        prompt_tokens: int,
    ) -> float:
        """估算 Prefill 延迟"""
        if worker.prefill_latency_ms > 0:
            # 使用历史数据
            return worker.prefill_latency_ms * (prompt_tokens / 512)  # 归一化到 512 tokens

        # 基于算力估算
        # Prefill 主要是计算密集，延迟 ≈ tokens * hidden_size^2 / FLOPS
        base_latency = 100  # ms for 512 tokens on reference GPU
        return base_latency * (prompt_tokens / 512) * (10 / max(1, worker.compute_flops))

    def _estimate_decode_latency(self, worker: WorkerCapability) -> float:
        """估算单步 Decode 延迟"""
        if worker.decode_latency_ms > 0:
            return worker.decode_latency_ms

        # 基于内存带宽估算
        # Decode 主要是内存带宽密集
        base_latency = 10  # ms per token on reference GPU
        return base_latency * (1000 / max(1, worker.memory_bandwidth_gbps))

    async def get_batch(
        self,
        phase: JobPhase,
        max_batch_size: int = 32,
    ) -> List[Tuple[PendingJob, WorkerAssignment]]:
        """
        获取一批任务及其分配

        用于批量调度
        """
        queue = self._prefill_queue if phase == JobPhase.PREFILL else self._decode_queue
        timeout = self.prefill_batch_timeout_ms if phase == JobPhase.PREFILL else self.decode_batch_timeout_ms

        batch = []
        deadline = time.time() + timeout / 1000

        while len(batch) < max_batch_size and queue:
            if time.time() > deadline and batch:
                break

            job = heapq.heappop(queue)

            try:
                assignment = await self.assign_job(job)
                batch.append((job, assignment))
            except Exception as e:
                logger.warning(f"Failed to assign job {job.job_id}: {e}")
                # 重新入队
                heapq.heappush(queue, job)

        return batch

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        prefill_workers = sum(
            1 for w in self._workers.values()
            if w.role in (WorkerRole.PREFILL, WorkerRole.HYBRID)
        )
        decode_workers = sum(
            1 for w in self._workers.values()
            if w.role in (WorkerRole.DECODE, WorkerRole.HYBRID)
        )

        return {
            **self._stats,
            "prefill_queue_size": len(self._prefill_queue),
            "decode_queue_size": len(self._decode_queue),
            "total_workers": len(self._workers),
            "prefill_workers": prefill_workers,
            "decode_workers": decode_workers,
            "kv_cache_entries": len(self._kv_cache_locations),
        }


class KVCacheMigrator:
    """
    KV-Cache 迁移器

    处理 KV-Cache 在 Worker 间的迁移
    """

    def __init__(self, scheduler: PrefillDecodeScheduler):
        self.scheduler = scheduler
        self._pending_migrations: Dict[str, asyncio.Task] = {}

    async def migrate(
        self,
        kv_cache_key: str,
        source_worker: str,
        target_worker: str,
    ) -> bool:
        """
        迁移 KV-Cache

        Args:
            kv_cache_key: KV-Cache 键
            source_worker: 源 Worker
            target_worker: 目标 Worker

        Returns:
            是否成功
        """
        migration_id = f"{kv_cache_key}:{source_worker}:{target_worker}"

        # 检查是否已有迁移进行中
        if migration_id in self._pending_migrations:
            # 等待现有迁移完成
            await self._pending_migrations[migration_id]
            return True

        # 启动迁移任务
        task = asyncio.create_task(
            self._do_migrate(kv_cache_key, source_worker, target_worker)
        )
        self._pending_migrations[migration_id] = task

        try:
            result = await task
            return result
        finally:
            self._pending_migrations.pop(migration_id, None)

    async def _do_migrate(
        self,
        kv_cache_key: str,
        source_worker: str,
        target_worker: str,
    ) -> bool:
        """执行迁移"""
        logger.info(f"Migrating KV-Cache {kv_cache_key}: {source_worker} -> {target_worker}")

        try:
            # TODO: 实现实际的迁移逻辑
            # 1. 从源 Worker 读取 KV-Cache
            # 2. 传输到目标 Worker
            # 3. 更新位置索引

            # 模拟迁移延迟
            await asyncio.sleep(0.05)  # 50ms

            # 更新索引
            self.scheduler._kv_cache_locations[kv_cache_key] = target_worker
            self.scheduler._stats["migrations"] += 1

            logger.info(f"KV-Cache migration completed: {kv_cache_key}")
            return True

        except Exception as e:
            logger.error(f"KV-Cache migration failed: {e}")
            return False
