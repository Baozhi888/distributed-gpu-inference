import asyncio
import sys
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


from common.data_structures import WorkerInfo, WorkerState  # noqa: E402
from common.serialization import serialize_tensor  # noqa: E402
from distributed.session import WorkerSession, SessionState  # noqa: E402


class _FakeResponse:
    def __init__(self, status: int, json_data=None, text_data: str = ""):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClientSession:
    def __init__(self, *, health_status: int = 200, forward_status: int = 200, forward_result=None):
        self._health_status = health_status
        self._forward_status = forward_status
        self._forward_result = forward_result
        self.closed = False
        self.last_post = None

    def get(self, url: str):
        return _FakeResponse(self._health_status)

    def post(self, url: str, json=None):
        self.last_post = (url, json)
        if url.endswith("/inference/forward"):
            if self._forward_status != 200:
                return _FakeResponse(self._forward_status, text_data="bad")
            return _FakeResponse(200, json_data=self._forward_result or {})
        if url.endswith("/inference/close"):
            return _FakeResponse(200, json_data={})
        return _FakeResponse(404, text_data="not found")

    async def close(self):
        self.closed = True


def test_worker_session_connect_success_sets_ready() -> None:
    async def runner():
        fake = _FakeClientSession(health_status=200)
        with patch("distributed.session.aiohttp.ClientSession", return_value=fake):
            ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE, api_endpoint="http://w"))
            await ws.connect()
            assert ws.state == SessionState.READY

    asyncio.run(runner())


def test_worker_session_connect_failure_sets_error() -> None:
    async def runner():
        fake = _FakeClientSession(health_status=500)
        with patch("distributed.session.aiohttp.ClientSession", return_value=fake):
            ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE, api_endpoint="http://w"))
            with pytest.raises(ConnectionError):
                await ws.connect()
            assert ws.state == SessionState.ERROR

    asyncio.run(runner())


def test_worker_session_forward_roundtrip_and_updates_position() -> None:
    async def runner():
        hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
        forward_result = {
            "output": serialize_tensor(hidden),
            "kv_cache_keys": ["k1"],
        }
        fake = _FakeClientSession(health_status=200, forward_status=200, forward_result=forward_result)
        with patch("distributed.session.aiohttp.ClientSession", return_value=fake):
            ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE, api_endpoint="http://w"))
            await ws.connect()
            out, keys = await ws.forward(hidden_states=hidden, position=5, kv_cache_keys=["x"])
            assert keys == ["k1"]
            # 默认 deserialize_tensor 会返回 torch.Tensor（如果 torch 可用）
            if hasattr(out, "cpu"):
                assert np.array_equal(out.cpu().numpy(), hidden)
            else:
                assert np.array_equal(out, hidden)
            assert ws.position == 5 + hidden.shape[1]

            await ws.close()
            assert ws.state == SessionState.CLOSED
            assert fake.closed is True

    asyncio.run(runner())


def test_worker_session_forward_non_200_sets_error() -> None:
    async def runner():
        fake = _FakeClientSession(health_status=200, forward_status=500, forward_result=None)
        with patch("distributed.session.aiohttp.ClientSession", return_value=fake):
            ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE, api_endpoint="http://w"))
            await ws.connect()
            with pytest.raises(RuntimeError):
                await ws.forward(hidden_states=np.zeros((1, 1), dtype=np.float32), position=0)
            assert ws.state == SessionState.ERROR

    asyncio.run(runner())

