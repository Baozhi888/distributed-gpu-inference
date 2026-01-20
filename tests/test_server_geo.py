import asyncio
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = (REPO_ROOT / "server").resolve()
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


from app.services.geo import detect_client_region, get_region_name  # noqa: E402


def test_detect_client_region_defaults_and_private_ip() -> None:
    assert asyncio.run(detect_client_region(None)) == "asia-east"
    assert asyncio.run(detect_client_region("10.0.0.1")) == "asia-east"
    assert asyncio.run(detect_client_region("localhost")) == "asia-east"


def test_detect_client_region_prefix_match() -> None:
    assert asyncio.run(detect_client_region("2.1.1.1")) == "europe-west"
    assert asyncio.run(detect_client_region("3.9.9.9")) == "america-north"


def test_get_region_name_fallback() -> None:
    assert "东亚" in get_region_name("asia-east")
    assert get_region_name("unknown") == "unknown"

