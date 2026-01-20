import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from engines.llm_base import (  # noqa: E402
    GenerationConfig,
    GenerationResult,
    LLMBaseEngine,
)


class _DummyEngine(LLMBaseEngine):
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
        return GenerationResult(
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
    ) -> List[GenerationResult]:
        results: List[GenerationResult] = []
        for messages in batch_messages:
            results.append(await self.generate_async(messages, config))
        return results


def test_inference_inside_running_event_loop_does_not_raise() -> None:
    engine = _DummyEngine(config={})

    async def _call_inference() -> Dict[str, Any]:
        return engine.inference({"messages": [{"role": "user", "content": "hi"}]})

    result = asyncio.run(_call_inference())
    assert result["response"] == "ok"
