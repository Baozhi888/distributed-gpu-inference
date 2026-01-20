"""
测试 server/app/services/pd_scheduler.py 模块

覆盖：
- JobPhase / WorkerRole 枚举
- WorkerCapability 数据类
- PendingJob 数据类
- WorkerAssignment 数据类
- PrefillDecodeScheduler 调度器
- KVCacheMigrator 迁移器
"""
import pytest
import asyncio
import time
import importlib.util
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_pd_scheduler_module():
    """使用 importlib 直接加载 pd_scheduler.py"""
    module_path = REPO_ROOT / "server" / "app" / "services" / "pd_scheduler.py"
    spec = importlib.util.spec_from_file_location("pd_scheduler", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pd_scheduler"] = module
    sys.modules["server.app.services.pd_scheduler"] = module
    spec.loader.exec_module(module)
    return module


pd_scheduler = _load_pd_scheduler_module()
JobPhase = pd_scheduler.JobPhase
WorkerRole = pd_scheduler.WorkerRole
WorkerCapability = pd_scheduler.WorkerCapability
PendingJob = pd_scheduler.PendingJob
WorkerAssignment = pd_scheduler.WorkerAssignment
PrefillDecodeScheduler = pd_scheduler.PrefillDecodeScheduler
KVCacheMigrator = pd_scheduler.KVCacheMigrator


# ============== JobPhase / WorkerRole 枚举测试 ==============

class TestEnums:
    """枚举测试"""

    def test_job_phase_values(self):
        """测试 JobPhase 枚举值"""
        assert JobPhase.PREFILL.value == "prefill"
        assert JobPhase.DECODE.value == "decode"
        assert len(JobPhase) == 2

    def test_worker_role_values(self):
        """测试 WorkerRole 枚举值"""
        assert WorkerRole.PREFILL.value == "prefill"
        assert WorkerRole.DECODE.value == "decode"
        assert WorkerRole.HYBRID.value == "hybrid"
        assert len(WorkerRole) == 3


# ============== WorkerCapability 测试 ==============

class TestWorkerCapability:
    """WorkerCapability 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        cap = WorkerCapability(worker_id="worker-1")

        assert cap.worker_id == "worker-1"
        assert cap.role == WorkerRole.HYBRID
        assert cap.compute_flops == 0.0
        assert cap.memory_bandwidth_gbps == 0.0
        assert cap.active_prefill_jobs == 0
        assert cap.active_decode_jobs == 0

    def test_custom_values(self):
        """测试自定义值"""
        cap = WorkerCapability(
            worker_id="a100-worker",
            role=WorkerRole.PREFILL,
            compute_flops=312.0,  # A100 FP16
            memory_bandwidth_gbps=2000.0,
            gpu_memory_gb=80.0,
            reliability_score=0.95,
        )

        assert cap.compute_flops == 312.0
        assert cap.memory_bandwidth_gbps == 2000.0
        assert cap.gpu_memory_gb == 80.0
        assert cap.reliability_score == 0.95

    def test_prefill_capacity_hybrid(self):
        """测试混合 Worker 的 Prefill 容量"""
        cap = WorkerCapability(
            worker_id="hybrid-worker",
            role=WorkerRole.HYBRID,
            compute_flops=100.0,
            reliability_score=0.9,
        )

        assert cap.prefill_capacity == 100.0 * 0.9

    def test_prefill_capacity_decode_only(self):
        """测试仅 Decode Worker 的 Prefill 容量"""
        cap = WorkerCapability(
            worker_id="decode-worker",
            role=WorkerRole.DECODE,
            compute_flops=100.0,
        )

        assert cap.prefill_capacity == 0.0

    def test_decode_capacity_hybrid(self):
        """测试混合 Worker 的 Decode 容量"""
        cap = WorkerCapability(
            worker_id="hybrid-worker",
            role=WorkerRole.HYBRID,
            memory_bandwidth_gbps=1000.0,
            reliability_score=0.8,
        )

        assert cap.decode_capacity == 1000.0 * 0.8

    def test_decode_capacity_prefill_only(self):
        """测试仅 Prefill Worker 的 Decode 容量"""
        cap = WorkerCapability(
            worker_id="prefill-worker",
            role=WorkerRole.PREFILL,
            memory_bandwidth_gbps=1000.0,
        )

        assert cap.decode_capacity == 0.0

    def test_kv_cache_utilization(self):
        """测试 KV-Cache 利用率"""
        cap = WorkerCapability(
            worker_id="worker",
            kv_cache_tokens_used=500,
            kv_cache_tokens_total=1000,
        )

        assert cap.kv_cache_utilization == 0.5

    def test_kv_cache_utilization_zero_total(self):
        """测试 KV-Cache 总量为零时的利用率"""
        cap = WorkerCapability(
            worker_id="worker",
            kv_cache_tokens_used=0,
            kv_cache_tokens_total=0,
        )

        assert cap.kv_cache_utilization == 0.0


# ============== PendingJob 测试 ==============

class TestPendingJob:
    """PendingJob 数据类测试"""

    def test_creation(self):
        """测试创建"""
        job = PendingJob(
            priority=1.0,
            created_at=time.time(),
            job_id="job-123",
            phase=JobPhase.PREFILL,
            prompt_tokens=512,
            max_tokens=256,
        )

        assert job.job_id == "job-123"
        assert job.phase == JobPhase.PREFILL
        assert job.prompt_tokens == 512
        assert job.max_tokens == 256

    def test_ordering_by_priority(self):
        """测试按优先级排序"""
        job1 = PendingJob(priority=-1.0, created_at=1.0, job_id="j1", phase=JobPhase.PREFILL)
        job2 = PendingJob(priority=-2.0, created_at=2.0, job_id="j2", phase=JobPhase.PREFILL)
        job3 = PendingJob(priority=-0.5, created_at=3.0, job_id="j3", phase=JobPhase.PREFILL)

        # 优先级使用负数，所以 -2.0 < -1.0 < -0.5
        jobs = [job1, job2, job3]
        jobs.sort()

        assert jobs[0].job_id == "j2"  # -2.0 最小（最高优先级）
        assert jobs[1].job_id == "j1"  # -1.0
        assert jobs[2].job_id == "j3"  # -0.5

    def test_kv_cache_fields(self):
        """测试 KV-Cache 相关字段"""
        job = PendingJob(
            priority=0,
            created_at=time.time(),
            job_id="job-decode",
            phase=JobPhase.DECODE,
            kv_cache_key="cache-key-abc",
            kv_cache_worker="worker-1",
        )

        assert job.kv_cache_key == "cache-key-abc"
        assert job.kv_cache_worker == "worker-1"


# ============== WorkerAssignment 测试 ==============

class TestWorkerAssignment:
    """WorkerAssignment 数据类测试"""

    def test_creation(self):
        """测试创建"""
        assignment = WorkerAssignment(
            worker_id="worker-1",
            phase=JobPhase.PREFILL,
            estimated_latency_ms=100.0,
        )

        assert assignment.worker_id == "worker-1"
        assert assignment.phase == JobPhase.PREFILL
        assert assignment.estimated_latency_ms == 100.0
        assert assignment.kv_migration_needed is False

    def test_with_migration(self):
        """测试带迁移的分配"""
        assignment = WorkerAssignment(
            worker_id="worker-2",
            phase=JobPhase.DECODE,
            estimated_latency_ms=60.0,
            kv_migration_needed=True,
            migration_source="worker-1",
        )

        assert assignment.kv_migration_needed is True
        assert assignment.migration_source == "worker-1"


# ============== PrefillDecodeScheduler 测试 ==============

class TestPrefillDecodeScheduler:
    """PrefillDecodeScheduler 测试"""

    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        return PrefillDecodeScheduler(
            enable_migration=True,
            migration_threshold_ms=50.0,
        )

    @pytest.fixture
    def workers(self):
        """创建测试用 Worker"""
        return {
            "prefill-a100": WorkerCapability(
                worker_id="prefill-a100",
                role=WorkerRole.PREFILL,
                compute_flops=312.0,
                memory_bandwidth_gbps=2000.0,
                reliability_score=0.95,
            ),
            "decode-4090": WorkerCapability(
                worker_id="decode-4090",
                role=WorkerRole.DECODE,
                compute_flops=82.0,
                memory_bandwidth_gbps=1000.0,
                reliability_score=0.9,
            ),
            "hybrid-3090": WorkerCapability(
                worker_id="hybrid-3090",
                role=WorkerRole.HYBRID,
                compute_flops=71.0,
                memory_bandwidth_gbps=936.0,
                reliability_score=0.85,
            ),
        }

    def test_init(self, scheduler):
        """测试初始化"""
        assert scheduler.enable_migration is True
        assert scheduler.migration_threshold_ms == 50.0
        assert len(scheduler._workers) == 0
        assert len(scheduler._prefill_queue) == 0
        assert len(scheduler._decode_queue) == 0

    def test_register_worker(self, scheduler, workers):
        """测试注册 Worker"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])

        assert "prefill-a100" in scheduler._workers
        assert scheduler._workers["prefill-a100"].compute_flops == 312.0

    def test_unregister_worker(self, scheduler, workers):
        """测试注销 Worker"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])
        scheduler.unregister_worker("prefill-a100")

        assert "prefill-a100" not in scheduler._workers

    def test_update_worker_stats(self, scheduler, workers):
        """测试更新 Worker 状态"""
        scheduler.register_worker("hybrid-3090", workers["hybrid-3090"])

        scheduler.update_worker_stats("hybrid-3090", {
            "active_prefill_jobs": 2,
            "active_decode_jobs": 5,
            "prefill_latency_ms": 150.0,
        })

        worker = scheduler._workers["hybrid-3090"]
        assert worker.active_prefill_jobs == 2
        assert worker.active_decode_jobs == 5
        assert worker.prefill_latency_ms == 150.0

    @pytest.mark.asyncio
    async def test_submit_job(self, scheduler, workers):
        """测试提交任务"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])

        job_id = await scheduler.submit_job(
            job_id="job-001",
            prompt_tokens=1024,
            max_tokens=512,
            priority=2.0,
        )

        assert job_id == "job-001"
        assert len(scheduler._prefill_queue) == 1
        assert scheduler._stats["prefill_jobs"] == 1

    @pytest.mark.asyncio
    async def test_transition_to_decode(self, scheduler, workers):
        """测试转换到 Decode 阶段"""
        scheduler.register_worker("decode-4090", workers["decode-4090"])

        await scheduler.transition_to_decode(
            job_id="job-001",
            kv_cache_key="kv-001",
            kv_cache_worker="prefill-a100",
        )

        assert len(scheduler._decode_queue) == 1
        assert scheduler._kv_cache_locations["kv-001"] == "prefill-a100"
        assert scheduler._stats["decode_jobs"] == 1

    @pytest.mark.asyncio
    async def test_assign_prefill_job(self, scheduler, workers):
        """测试分配 Prefill 任务"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])
        scheduler.register_worker("hybrid-3090", workers["hybrid-3090"])

        job = PendingJob(
            priority=-1.0,
            created_at=time.time(),
            job_id="job-prefill",
            phase=JobPhase.PREFILL,
            prompt_tokens=512,
        )

        assignment = await scheduler.assign_job(job)

        # 应该选择计算能力最强的 Worker (A100)
        assert assignment.worker_id == "prefill-a100"
        assert assignment.phase == JobPhase.PREFILL

    @pytest.mark.asyncio
    async def test_assign_decode_job_same_worker(self, scheduler, workers):
        """测试分配 Decode 任务到持有 KV-Cache 的 Worker"""
        scheduler.register_worker("hybrid-3090", workers["hybrid-3090"])

        job = PendingJob(
            priority=0,
            created_at=time.time(),
            job_id="job-decode",
            phase=JobPhase.DECODE,
            kv_cache_key="kv-001",
            kv_cache_worker="hybrid-3090",
        )

        assignment = await scheduler.assign_job(job)

        assert assignment.worker_id == "hybrid-3090"
        assert assignment.kv_migration_needed is False

    @pytest.mark.asyncio
    async def test_assign_decode_job_migration_needed(self, scheduler, workers):
        """测试分配 Decode 任务需要迁移"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])
        scheduler.register_worker("decode-4090", workers["decode-4090"])
        scheduler._kv_cache_locations["kv-001"] = "prefill-a100"

        job = PendingJob(
            priority=0,
            created_at=time.time(),
            job_id="job-decode",
            phase=JobPhase.DECODE,
            kv_cache_key="kv-001",
            kv_cache_worker="prefill-a100",
        )

        assignment = await scheduler.assign_job(job)

        # Prefill Worker 不适合做 Decode，应该选择 Decode Worker
        assert assignment.worker_id == "decode-4090"
        assert assignment.kv_migration_needed is True
        assert assignment.migration_source == "prefill-a100"

    @pytest.mark.asyncio
    async def test_assign_no_workers(self, scheduler):
        """测试无可用 Worker"""
        job = PendingJob(
            priority=-1.0,
            created_at=time.time(),
            job_id="job-fail",
            phase=JobPhase.PREFILL,
        )

        with pytest.raises(RuntimeError, match="No available workers"):
            await scheduler.assign_job(job)

    @pytest.mark.asyncio
    async def test_get_batch(self, scheduler, workers):
        """测试获取批量任务"""
        scheduler.register_worker("hybrid-3090", workers["hybrid-3090"])

        # 提交多个任务
        for i in range(5):
            await scheduler.submit_job(
                job_id=f"job-{i}",
                prompt_tokens=256,
                priority=float(i),
            )

        # 获取批量
        batch = await scheduler.get_batch(JobPhase.PREFILL, max_batch_size=3)

        assert len(batch) == 3
        # 检查每个都有分配
        for job, assignment in batch:
            assert assignment.worker_id == "hybrid-3090"

    def test_estimate_prefill_latency_with_history(self, scheduler, workers):
        """测试基于历史数据的 Prefill 延迟估算"""
        worker = workers["prefill-a100"]
        worker.prefill_latency_ms = 200.0

        latency = scheduler._estimate_prefill_latency(worker, 1024)

        # 1024 tokens = 2x 512 tokens
        assert latency == pytest.approx(400.0, rel=0.01)

    def test_estimate_prefill_latency_no_history(self, scheduler, workers):
        """测试无历史数据的 Prefill 延迟估算"""
        worker = workers["prefill-a100"]
        worker.prefill_latency_ms = 0.0

        latency = scheduler._estimate_prefill_latency(worker, 512)

        assert latency > 0

    def test_estimate_decode_latency(self, scheduler, workers):
        """测试 Decode 延迟估算"""
        worker = workers["decode-4090"]
        worker.decode_latency_ms = 15.0

        latency = scheduler._estimate_decode_latency(worker)

        assert latency == 15.0

    def test_get_stats(self, scheduler, workers):
        """测试获取统计"""
        scheduler.register_worker("prefill-a100", workers["prefill-a100"])
        scheduler.register_worker("decode-4090", workers["decode-4090"])
        scheduler.register_worker("hybrid-3090", workers["hybrid-3090"])

        stats = scheduler.get_stats()

        assert stats["total_workers"] == 3
        assert stats["prefill_workers"] == 2  # prefill + hybrid
        assert stats["decode_workers"] == 2  # decode + hybrid


# ============== KVCacheMigrator 测试 ==============

class TestKVCacheMigrator:
    """KVCacheMigrator 测试"""

    @pytest.fixture
    def migrator(self):
        """创建迁移器实例"""
        scheduler = PrefillDecodeScheduler()
        return KVCacheMigrator(scheduler)

    @pytest.mark.asyncio
    async def test_migrate_success(self, migrator):
        """测试成功迁移"""
        result = await migrator.migrate(
            kv_cache_key="kv-001",
            source_worker="worker-1",
            target_worker="worker-2",
        )

        assert result is True
        assert migrator.scheduler._kv_cache_locations["kv-001"] == "worker-2"
        assert migrator.scheduler._stats["migrations"] == 1

    @pytest.mark.asyncio
    async def test_migrate_concurrent(self, migrator):
        """测试并发迁移"""
        # 启动两个相同的迁移
        task1 = asyncio.create_task(migrator.migrate(
            kv_cache_key="kv-001",
            source_worker="worker-1",
            target_worker="worker-2",
        ))
        task2 = asyncio.create_task(migrator.migrate(
            kv_cache_key="kv-001",
            source_worker="worker-1",
            target_worker="worker-2",
        ))

        results = await asyncio.gather(task1, task2)

        # 两个都应该成功
        assert all(results)
        # 但只应该执行一次迁移
        assert migrator.scheduler._stats["migrations"] == 1

    @pytest.mark.asyncio
    async def test_migrate_different_keys(self, migrator):
        """测试不同 key 的迁移"""
        await asyncio.gather(
            migrator.migrate("kv-001", "w1", "w2"),
            migrator.migrate("kv-002", "w1", "w3"),
        )

        assert migrator.scheduler._kv_cache_locations["kv-001"] == "w2"
        assert migrator.scheduler._kv_cache_locations["kv-002"] == "w3"
        assert migrator.scheduler._stats["migrations"] == 2


# ============== 集成测试 ==============

class TestPDSchedulerIntegration:
    """P/D 调度器集成测试"""

    @pytest.mark.asyncio
    async def test_full_job_lifecycle(self):
        """测试完整任务生命周期"""
        scheduler = PrefillDecodeScheduler()

        # 注册 Worker
        prefill_worker = WorkerCapability(
            worker_id="prefill-w1",
            role=WorkerRole.PREFILL,
            compute_flops=200.0,
        )
        decode_worker = WorkerCapability(
            worker_id="decode-w1",
            role=WorkerRole.DECODE,
            memory_bandwidth_gbps=800.0,
        )

        scheduler.register_worker("prefill-w1", prefill_worker)
        scheduler.register_worker("decode-w1", decode_worker)

        # 1. 提交任务
        await scheduler.submit_job(
            job_id="job-lifecycle",
            prompt_tokens=1024,
            max_tokens=256,
        )

        # 2. 获取 Prefill 批次
        prefill_batch = await scheduler.get_batch(JobPhase.PREFILL, max_batch_size=1)
        assert len(prefill_batch) == 1
        job, assignment = prefill_batch[0]
        assert assignment.worker_id == "prefill-w1"

        # 3. 转换到 Decode 阶段
        await scheduler.transition_to_decode(
            job_id=job.job_id,
            kv_cache_key="kv-lifecycle",
            kv_cache_worker="prefill-w1",
        )

        # 4. 获取 Decode 批次
        decode_batch = await scheduler.get_batch(JobPhase.DECODE, max_batch_size=1)
        assert len(decode_batch) == 1
        job, assignment = decode_batch[0]
        assert assignment.worker_id == "decode-w1"
        assert assignment.kv_migration_needed is True

    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """测试负载均衡"""
        scheduler = PrefillDecodeScheduler()

        # 注册多个 Worker
        for i in range(3):
            worker = WorkerCapability(
                worker_id=f"worker-{i}",
                role=WorkerRole.HYBRID,
                compute_flops=100.0,
                memory_bandwidth_gbps=500.0,
            )
            scheduler.register_worker(f"worker-{i}", worker)

        # 提交多个任务
        for i in range(6):
            await scheduler.submit_job(
                job_id=f"job-{i}",
                prompt_tokens=256,
            )

        # 获取所有任务的分配
        batch = await scheduler.get_batch(JobPhase.PREFILL, max_batch_size=6)

        # 统计分配
        worker_counts = {}
        for job, assignment in batch:
            worker_counts[assignment.worker_id] = worker_counts.get(assignment.worker_id, 0) + 1

        # 由于负载均衡，应该分布到多个 Worker
        assert len(worker_counts) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
