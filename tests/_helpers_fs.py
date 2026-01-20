import os
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / "manual_tmp" / "tests"


def make_temp_dir(prefix: str) -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"{prefix}_{os.getpid()}_{time.time_ns()}"
    path.mkdir(parents=True, exist_ok=False)
    return path

