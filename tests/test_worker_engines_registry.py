import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from engines import (  # noqa: E402
    get_engine,
    create_llm_engine,
    list_engines,
    get_recommended_backend,
    ENGINE_REGISTRY,
    LLMEngine,
)


def test_get_engine_handles_aliases_and_unknown() -> None:
    assert get_engine("native") is ENGINE_REGISTRY["llm"]
    assert get_engine("transformers") is ENGINE_REGISTRY["llm"]
    assert get_engine("llm") is LLMEngine

    with pytest.raises(ValueError):
        get_engine("does_not_exist")


def test_create_llm_engine_validates_backend_prefix() -> None:
    with pytest.raises(ValueError):
        create_llm_engine({"backend": "image_gen"})


def test_list_engines_includes_lazy_entries() -> None:
    engines = list_engines()
    assert "llm" in engines
    assert engines["llm"]["available"] is True
    assert "llm_sglang" in engines


def test_get_recommended_backend_returns_string() -> None:
    backend = get_recommended_backend()
    assert backend in {"native", "sglang", "vllm", "vllm_async"}

