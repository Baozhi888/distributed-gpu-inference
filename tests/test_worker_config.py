import os
import sys
from pathlib import Path

from tests._helpers_fs import make_temp_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from config import get_env, load_dotenv, load_config  # noqa: E402


def test_get_env_casting(monkeypatch) -> None:
    monkeypatch.setenv("X_BOOL", "true")
    monkeypatch.setenv("X_INT", "10")
    monkeypatch.setenv("X_LIST", "a, b,,c")

    assert get_env("X_BOOL", False, bool) is True
    assert get_env("X_INT", 0, int) == 10
    assert get_env("X_LIST", [], list) == ["a", "b", "c"]


def test_load_dotenv_sets_only_missing_vars(monkeypatch) -> None:
    tmp_dir = make_temp_dir("dotenv")
    env_file = tmp_dir / ".env"
    env_file.write_text("A=1\nB=2\n", encoding="utf-8")

    monkeypatch.setenv("B", "existing")
    load_dotenv(str(env_file))

    assert os.environ["A"] == "1"
    assert os.environ["B"] == "existing"


def test_load_config_reads_yaml_and_env_engine_overrides(monkeypatch) -> None:
    tmp_dir = make_temp_dir("config")
    cfg = tmp_dir / "config.yaml"
    cfg.write_text(
        "region: europe-west\n"
        "server:\n"
        "  url: http://example\n"
        "engines:\n"
        "  llm:\n"
        "    model_id: base\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GPU_LLM_MODEL", "env-model")
    loaded = load_config(str(cfg))
    assert loaded.region == "europe-west"
    assert loaded.server.url == "http://example"
    assert loaded.engines["llm"]["model_id"] == "env-model"
