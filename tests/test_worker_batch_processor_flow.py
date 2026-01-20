import asyncio
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from batch_processor import ContinuousBatcher  # noqa: E402


class _AsyncBatchEngine:
    async def batch_inference_async(self, params_list):
        return [{"value": p.get("value")} for p in params_list]


class _AsyncBatchEngineWithError:
    async def batch_inference_async(self, params_list):
        return [{"ok": True}, RuntimeError("boom")]


class _SyncBatchEngine:
    def batch_inference(self, params_list):
        return [{"value": p.get("value")} for p in params_list]


def test_submit_processes_batch_with_async_engine() -> None:
    async def runner():
        b = ContinuousBatcher(_AsyncBatchEngine(), max_batch_size=2, max_wait_ms=1, enable_prefix_grouping=False)
        await b.start()
        t1 = asyncio.create_task(b.submit("j1", {"value": 1}, timeout=1))
        t2 = asyncio.create_task(b.submit("j2", {"value": 2}, timeout=1))
        r1, r2 = await asyncio.gather(t1, t2)
        await b.stop()
        return r1, r2

    r1, r2 = asyncio.run(runner())
    assert r1["value"] == 1
    assert r2["value"] == 2


def test_submit_processes_batch_with_sync_batch_engine() -> None:
    async def runner():
        b = ContinuousBatcher(_SyncBatchEngine(), max_batch_size=2, max_wait_ms=1, enable_prefix_grouping=False)
        await b.start()
        t1 = asyncio.create_task(b.submit("j1", {"value": 1}, timeout=1))
        t2 = asyncio.create_task(b.submit("j2", {"value": 2}, timeout=1))
        r1, r2 = await asyncio.gather(t1, t2)
        await b.stop()
        return r1, r2

    r1, r2 = asyncio.run(runner())
    assert r1["value"] == 1
    assert r2["value"] == 2


def test_submit_sets_exception_when_engine_returns_exception() -> None:
    async def runner():
        b = ContinuousBatcher(_AsyncBatchEngineWithError(), max_batch_size=2, max_wait_ms=1, enable_prefix_grouping=False)
        await b.start()
        ok_task = asyncio.create_task(b.submit("j1", {"value": 1}, timeout=1))
        err_task = asyncio.create_task(b.submit("j2", {"value": 2}, timeout=1))
        ok = await ok_task
        with pytest.raises(RuntimeError):
            await err_task
        await b.stop()
        return ok

    ok = asyncio.run(runner())
    assert ok["ok"] is True


def test_submit_timeout_removes_request_from_queue() -> None:
    class _SlowEngine:
        async def batch_inference_async(self, params_list):
            await asyncio.sleep(0.2)
            return [{"ok": True} for _ in params_list]

    async def runner():
        b = ContinuousBatcher(_SlowEngine(), max_batch_size=10, max_wait_ms=1000, enable_prefix_grouping=False)
        await b.start()
        with pytest.raises(asyncio.TimeoutError):
            await b.submit("j1", {"value": 1}, timeout=0.01)
        stats = b.get_stats()
        await b.stop()
        return stats

    stats = asyncio.run(runner())
    assert stats["queue_size"] == 0


def test_submit_queue_full_raises() -> None:
    async def runner():
        b = ContinuousBatcher(_AsyncBatchEngine(), max_queue_size=1, max_batch_size=10, max_wait_ms=1000)
        await b.start()
        t1 = asyncio.create_task(b.submit("j1", {"value": 1}, timeout=1))
        await asyncio.sleep(0)  # 确保请求已入队
        with pytest.raises(RuntimeError):
            await b.submit("j2", {"value": 2}, timeout=1)
        t1.cancel()
        await b.stop()

    asyncio.run(runner())

