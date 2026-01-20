"""LLM引擎扩展基类 - 支持高性能推理后端"""
from abc import abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import Future
import logging

from .base import BaseEngine

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """LLM 推理后端类型"""
    NATIVE = "native"       # 原生 Transformers
    SGLANG = "sglang"       # SGLang (推荐)
    VLLM = "vllm"           # vLLM


@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str = "stop"
    cached_tokens: int = 0  # 前缀缓存命中的 token 数


class LLMBaseEngine(BaseEngine):
    """
    LLM 引擎扩展基类

    提供高性能推理后端的统一接口，支持：
    - 异步生成
    - 批量处理
    - 流式输出
    - 前缀缓存
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type: LLMBackend = LLMBackend.NATIVE
        self.tokenizer = None
        self._batch_processor = None

    @abstractmethod
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        异步生成接口

        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            config: 生成配置

        Returns:
            生成结果
        """
        pass

    @abstractmethod
    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        批量生成接口

        Args:
            batch_messages: 批量对话消息
            config: 生成配置

        Returns:
            批量生成结果
        """
        pass

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        流式生成接口（默认实现：单次返回完整结果）

        Args:
            messages: 对话消息列表
            config: 生成配置

        Yields:
            生成的文本片段
        """
        result = await self.generate_async(messages, config)
        yield result.text

    @staticmethod
    def _run_coroutine_in_new_thread(coro):
        future: Future = Future()

        def runner() -> None:
            try:
                future.set_result(asyncio.run(coro))
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(target=runner, daemon=True).start()
        return future.result()

    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步推理接口（兼容 BaseEngine）

        包装异步接口，供同步调用
        """
        messages = params.get("messages", [])
        config = GenerationConfig(
            max_tokens=params.get("max_tokens", 2048),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            stop_sequences=params.get("stop", None),
            stream=params.get("stream", False)
        )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self.generate_async(messages, config))
        else:
            result = self._run_coroutine_in_new_thread(self.generate_async(messages, config))

        return {
            "response": result.text,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "cached_tokens": result.cached_tokens
            },
            "finish_reason": result.finish_reason
        }

    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        return False

    def supports_prefix_caching(self) -> bool:
        """是否支持前缀缓存"""
        return False

    def supports_batch_inference(self) -> bool:
        """是否支持批量推理"""
        return False

    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return {
            "backend": self.backend_type.value,
            "supports_streaming": self.supports_streaming(),
            "supports_prefix_caching": self.supports_prefix_caching(),
            "supports_batch_inference": self.supports_batch_inference()
        }

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态（扩展父类）"""
        status = super().get_status()
        status["backend_info"] = self.get_backend_info()
        return status


def create_llm_engine(config: Dict[str, Any]) -> LLMBaseEngine:
    """
    LLM 引擎工厂函数

    根据配置创建对应的 LLM 引擎实例

    Args:
        config: 引擎配置

    Returns:
        LLM 引擎实例
    """
    backend = config.get("backend", "native").lower()

    if backend == "sglang":
        from .llm_sglang import SGLangEngine
        return SGLangEngine(config)

    elif backend == "vllm":
        from .llm_vllm import VLLMEngine
        return VLLMEngine(config)

    else:
        # 默认使用原生 Transformers
        from .llm import LLMEngine
        return LLMEngine(config)
