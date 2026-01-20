import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from engines.base import BaseEngine  # noqa: E402
from engines.llm_base import LLMBaseEngine, GenerationConfig, GenerationResult  # noqa: E402


class _DummyBaseEngine(BaseEngine):
    def load_model(self) -> None:
        self.loaded = True

    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def unload_model(self) -> None:
        self.loaded = False


class _DummyLLMEngine(LLMBaseEngine):
    def load_model(self) -> None:
        self.loaded = True

    def unload_model(self) -> None:
        self.loaded = False

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        await asyncio.sleep(0)
        return GenerationResult(text="ok", prompt_tokens=1, completion_tokens=1, total_tokens=2)

    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
    ) -> List[GenerationResult]:
        return [await self.generate_async(m, config) for m in batch_messages]


def test_base_engine_get_status_has_expected_keys() -> None:
    e = _DummyBaseEngine(config={})
    status = e.get_status()
    assert "loaded" in status
    assert "device" in status


def test_llm_base_inference_outside_event_loop_works() -> None:
    e = _DummyLLMEngine(config={})
    out = e.inference({"messages": [{"role": "user", "content": "hi"}]})
    assert out["response"] == "ok"


def test_llm_base_stream_generate_yields_text() -> None:
    async def runner():
        e = _DummyLLMEngine(config={})
        chunks = []
        async for chunk in e.stream_generate([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(runner())
    assert chunks == ["ok"]


def test_llm_base_backend_info_defaults() -> None:
    e = _DummyLLMEngine(config={})
    info = e.get_backend_info()
    assert info["backend"] == e.backend_type.value
    assert info["supports_streaming"] is False
