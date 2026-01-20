import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common.data_structures import (  # noqa: E402
    BlockRange,
    WorkerInfo,
    WorkerState,
    InferenceState,
    KVCacheBlock,
)


def test_block_range_basic() -> None:
    br = BlockRange(start=2, end=5)
    assert br.length == 3
    assert 2 in br
    assert 4 in br
    assert 5 not in br
    assert BlockRange.from_dict(br.to_dict()) == br


def test_worker_info_utilizations_and_health() -> None:
    now = time.time()
    w = WorkerInfo(
        worker_id="w1",
        state=WorkerState.ONLINE,
        gpu_memory_gb=10.0,
        gpu_memory_used_gb=2.5,
        cache_tokens_available=100,
        cache_tokens_used=25,
        last_heartbeat=now,
    )
    assert w.cache_utilization == 0.25
    assert w.gpu_utilization == 0.25
    assert w.is_healthy(timeout_seconds=60.0) is True

    w.last_heartbeat = now - 120
    assert w.is_healthy(timeout_seconds=60.0) is False


def test_inference_state_update_position_updates_timestamp() -> None:
    state = InferenceState(session_id="s1", position=0)
    old_updated = state.updated_at
    state.update_position(new_tokens=3)
    assert state.position == 3
    assert state.updated_at >= old_updated


def test_kv_cache_block_ref_count() -> None:
    block = KVCacheBlock(block_id="b1", layer_idx=0, ref_count=1)
    assert block.is_shared is False
    block.increment_ref()
    assert block.ref_count == 2
    assert block.is_shared is True
    assert block.decrement_ref() == 1
    assert block.decrement_ref() == 0
    assert block.ref_count == 0

