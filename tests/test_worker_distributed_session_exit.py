import asyncio
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from common.data_structures import WorkerInfo, WorkerState  # noqa: E402
from distributed.session import WorkerSession  # noqa: E402


def test_worker_session_exit_inside_running_loop_does_not_raise() -> None:
    called = {"n": 0}

    async def fake_close(self) -> None:
        called["n"] += 1

    ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE))
    ws.close = types.MethodType(fake_close, ws)

    async def runner() -> None:
        ws.__exit__(None, None, None)

    asyncio.run(runner())
    assert called["n"] == 1


def test_worker_session_exit_without_running_loop_does_not_raise() -> None:
    called = {"n": 0}

    async def fake_close(self) -> None:
        called["n"] += 1

    ws = WorkerSession(worker_info=WorkerInfo(worker_id="w", state=WorkerState.ONLINE))
    ws.close = types.MethodType(fake_close, ws)

    ws.__exit__(None, None, None)
    assert called["n"] == 1

