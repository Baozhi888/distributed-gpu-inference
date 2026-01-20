import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from batch_processor import ContinuousBatcher, AdaptiveBatcher, PendingRequest, RequestPriority  # noqa: E402


def test_compute_prefix_hash_only_uses_system_messages() -> None:
    b = ContinuousBatcher(engine=object(), enable_prefix_grouping=True)
    assert b._compute_prefix_hash({"messages": []}) == ""
    assert b._compute_prefix_hash({"messages": [{"role": "user", "content": "x"}]}) == ""

    h1 = b._compute_prefix_hash({"messages": [{"role": "system", "content": "a"}]})
    h2 = b._compute_prefix_hash({"messages": [{"role": "system", "content": "a"}]})
    assert h1 == h2
    assert len(h1) == 16


def test_select_batch_with_prefix_grouping_prefers_largest_group() -> None:
    b = ContinuousBatcher(engine=object(), max_batch_size=4, enable_prefix_grouping=True)

    def add(job_id: str, prefix: str) -> PendingRequest:
        req = PendingRequest(
            priority=RequestPriority.NORMAL.value,
            timestamp=time.time(),
            job_id=job_id,
            params={"messages": []},
            future=MagicMock(),
            prefix_hash=prefix,
        )
        b._pending.append(req)  # 此测试只验证选择逻辑，不依赖 heapq 或 Future 绑定到事件循环
        if prefix:
            b._pending_by_prefix[prefix].append(req)
        return req

    a1 = add("a1", "p1")
    a2 = add("a2", "p1")
    b1 = add("b1", "p2")
    c1 = add("c1", "")

    batch = b._select_batch_with_prefix_grouping()
    assert a1 in batch and a2 in batch
    assert len(batch) == 4 or len(batch) == 3
    assert c1 in batch
    assert b1 in batch


def test_submit_requires_running() -> None:
    b = ContinuousBatcher(engine=object())

    async def run():
        try:
            await b.submit("j1", {})
        except RuntimeError as e:
            return str(e)
        return "no-error"

    msg = asyncio.run(run())
    assert "not running" in msg.lower()


def test_adaptive_batcher_reduces_and_increases_batch_size() -> None:
    b = AdaptiveBatcher(engine=object(), min_batch_size=1, max_batch_size=10, target_latency_ms=100)
    b._current_batch_size = 10
    b._latency_history = [200.0] * 10
    b._adapt_batch_size()
    assert b._current_batch_size < 10

    b._current_batch_size = 5
    b._latency_history = [50.0] * 10
    b._adapt_batch_size()
    assert b._current_batch_size > 5
