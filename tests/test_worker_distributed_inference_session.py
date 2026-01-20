import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from common.data_structures import WorkerInfo, WorkerState, SessionConfig  # noqa: E402
from distributed.session import DistributedInferenceSession, SessionManager, SessionState  # noqa: E402


class _FakeWorkerSession:
    def __init__(self, worker_info: WorkerInfo):
        self.worker_info = worker_info
        self.session_id = "s"
        self.state = SessionState.INITIALIZING
        self.position = 0
        self.next_session = None
        self._forward_calls = 0

    async def connect(self, timeout: float = 30.0) -> None:
        self.state = SessionState.READY

    async def forward(self, hidden_states, position: int, kv_cache_keys=None):
        self._forward_calls += 1
        return hidden_states, (kv_cache_keys or [])

    async def close(self) -> None:
        self.state = SessionState.CLOSED


class _FlakyWorkerSession(_FakeWorkerSession):
    async def forward(self, hidden_states, position: int, kv_cache_keys=None):
        self._forward_calls += 1
        if self._forward_calls == 1:
            raise RuntimeError("transient")
        return hidden_states, (kv_cache_keys or [])


class _AlwaysFailWorkerSession(_FakeWorkerSession):
    async def forward(self, hidden_states, position: int, kv_cache_keys=None):
        self._forward_calls += 1
        raise RuntimeError("permanent")


def test_distributed_session_setup_and_step_updates_stats() -> None:
    async def runner():
        config = SessionConfig(model_name="m", max_length=10, max_retries=1, connect_timeout=1.0)
        route = [WorkerInfo(worker_id="w1", state=WorkerState.ONLINE, api_endpoint="http://w")]

        with patch("distributed.session.WorkerSession", _FakeWorkerSession):
            s = DistributedInferenceSession(config, route)
            await s.setup()
            assert s.state == SessionState.READY
            out = await s.step(np.zeros((1, 2), dtype=np.float32))
            assert out.shape == (1, 2)
            assert s.position == 2
            stats = s.get_stats()
            assert stats["total_steps"] == 1
            await s.close()
            assert s.state == SessionState.CLOSED

    asyncio.run(runner())


def test_distributed_session_step_retries_on_transient_failure() -> None:
    async def runner():
        config = SessionConfig(model_name="m", max_length=10, max_retries=2, connect_timeout=1.0)
        route = [WorkerInfo(worker_id="w1", state=WorkerState.ONLINE, api_endpoint="http://w")]

        with patch("distributed.session.WorkerSession", _FlakyWorkerSession), patch("distributed.session.asyncio.sleep", return_value=None):
            s = DistributedInferenceSession(config, route)
            await s.setup()
            await s.step(np.zeros((1, 1), dtype=np.float32))
            stats = s.get_stats()
            assert stats["retries"] >= 1
            await s.close()

    asyncio.run(runner())


def test_distributed_session_step_length_exceeded_raises() -> None:
    async def runner():
        config = SessionConfig(model_name="m", max_length=1, max_retries=1, connect_timeout=1.0)
        route = [WorkerInfo(worker_id="w1", state=WorkerState.ONLINE, api_endpoint="http://w")]

        with patch("distributed.session.WorkerSession", _FakeWorkerSession):
            s = DistributedInferenceSession(config, route)
            await s.setup()
            with pytest.raises(ValueError):
                await s.step(np.zeros((1, 2), dtype=np.float32))

    asyncio.run(runner())


def test_distributed_session_step_failure_calls_handle_failure() -> None:
    async def runner():
        config = SessionConfig(model_name="m", max_length=10, max_retries=1, connect_timeout=1.0)
        route = [WorkerInfo(worker_id="w1", state=WorkerState.ONLINE, api_endpoint="http://w")]

        with patch("distributed.session.WorkerSession", _AlwaysFailWorkerSession):
            s = DistributedInferenceSession(config, route)
            await s.setup()
            with pytest.raises(RuntimeError):
                await s.step(np.zeros((1, 1), dtype=np.float32))
            await s.close()

    asyncio.run(runner())


def test_session_manager_cleans_up_closed_sessions() -> None:
    async def runner():
        config = SessionConfig(model_name="m", max_length=10, max_retries=1, connect_timeout=1.0)
        route = [WorkerInfo(worker_id="w1", state=WorkerState.ONLINE, api_endpoint="http://w")]

        with patch("distributed.session.WorkerSession", _FakeWorkerSession):
            mgr = SessionManager(max_sessions=1)
            s1 = await mgr.create_session(config, route)
            assert s1.state == SessionState.READY

            s1.state = SessionState.CLOSED
            s2 = await mgr.create_session(config, route)
            assert s2 is not None
            await mgr.close_all()

    asyncio.run(runner())

