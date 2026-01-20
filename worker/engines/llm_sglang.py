"""
SGLang 高性能推理引擎

特点：
- RadixAttention 前缀缓存（自动复用相同前缀的 KV-Cache，适合 RAG/Agent）
- PagedAttention 内存管理（高效显存利用率 80%+）
- 连续批处理 (Continuous Batching)
- 分块预填充 (Chunked Prefill)
- 3.1x 吞吐量提升（相比 vLLM 2024 基准）

参考：https://github.com/sgl-project/sglang
"""
from typing import Dict, Any, List, Optional, AsyncIterator
import logging
import asyncio
import time

from .llm_base import LLMBaseEngine, LLMBackend, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class SGLangEngine(LLMBaseEngine):
    """
    基于 SGLang 的高性能 LLM 推理引擎

    SGLang 核心优势：
    1. RadixAttention - 自动前缀缓存，共享系统提示/对话历史的 KV-Cache
    2. 纯 Python 实现 (<4K 核心代码)，易于调试和定制
    3. 原生支持结构化输出和约束解码
    4. 支持主流模型架构（Llama, Qwen, Mistral, Gemma 等）
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = LLMBackend.SGLANG
        self.runtime = None
        self._server_process = None
        self._sglang_config = config.get("sglang", {})

        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0

    def load_model(self) -> None:
        """加载模型到 SGLang Runtime"""
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "SGLang not installed. Please install with:\n"
                "  pip install sglang[all]\n"
                "Or for minimal install:\n"
                "  pip install sglang"
            )

        model_id = self.config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"Loading model with SGLang: {model_id}")

        # 合并配置
        tp_size = self._sglang_config.get("tp_size", 1)
        mem_fraction = self._sglang_config.get("mem_fraction_static", 0.85)
        chunked_prefill = self._sglang_config.get("chunked_prefill_size", 8192)
        enable_prefix_caching = self._sglang_config.get("enable_prefix_caching", True)
        max_batch_size = self._sglang_config.get("max_running_requests", 32)
        context_length = self._sglang_config.get("context_length", 8192)

        # SGLang Runtime 配置
        runtime_config = {
            "model_path": model_id,
            "tp_size": tp_size,
            "mem_fraction_static": mem_fraction,
            "chunked_prefill_size": chunked_prefill,
            "max_running_requests": max_batch_size,
            "context_length": context_length,
            "disable_radix_cache": not enable_prefix_caching,
        }

        # 量化配置
        quantization = self.config.get("quantization")
        if quantization:
            if quantization in ["int8", "fp8", "awq", "gptq"]:
                runtime_config["quantization"] = quantization
            logger.info(f"Using quantization: {quantization}")

        # 启动模式选择
        if self._sglang_config.get("use_server_mode", False):
            # 服务器模式 - 连接到已运行的 SGLang 服务器
            self._connect_to_server()
        else:
            # 嵌入式模式 - 直接启动 Runtime
            self._start_embedded_runtime(model_id, runtime_config)

        # 加载 tokenizer（用于消息格式化）
        self._load_tokenizer(model_id)

        self.loaded = True
        logger.info("SGLang engine loaded successfully")
        logger.info(f"  - Tensor Parallel: {tp_size}")
        logger.info(f"  - Memory Fraction: {mem_fraction}")
        logger.info(f"  - Prefix Caching: {enable_prefix_caching}")
        logger.info(f"  - Max Batch Size: {max_batch_size}")

    def _connect_to_server(self) -> None:
        """连接到已运行的 SGLang 服务器"""
        from sglang import RuntimeEndpoint

        server_url = self._sglang_config.get("server_url", "http://localhost:30000")
        logger.info(f"Connecting to SGLang server at {server_url}")

        self.runtime = RuntimeEndpoint(server_url)

    def _start_embedded_runtime(self, model_id: str, config: Dict[str, Any]) -> None:
        """启动嵌入式 Runtime"""
        try:
            import sglang as sgl

            self.runtime = sgl.Runtime(**config)
            sgl.set_default_backend(self.runtime)
            logger.info("Started embedded SGLang runtime")

        except Exception as e:
            logger.warning(f"Failed to start embedded runtime: {e}")
            logger.info("Falling back to server mode...")
            self._start_sglang_server(model_id, config)

    def _start_sglang_server(self, model_id: str, config: Dict[str, Any]) -> None:
        """启动 SGLang 服务器进程"""
        import subprocess

        port = self._sglang_config.get("port", 30000)

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_id,
            "--port", str(port),
            "--mem-fraction-static", str(config.get("mem_fraction_static", 0.85)),
        ]

        if config.get("tp_size", 1) > 1:
            cmd.extend(["--tp", str(config["tp_size"])])

        if config.get("context_length"):
            cmd.extend(["--context-length", str(config["context_length"])])

        if not config.get("disable_radix_cache", False):
            cmd.append("--enable-radix-cache")

        logger.info(f"Starting SGLang server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 等待服务器启动
        self._wait_for_server(port)

        from sglang import RuntimeEndpoint
        self.runtime = RuntimeEndpoint(f"http://localhost:{port}")

    def _wait_for_server(self, port: int, timeout: int = 120) -> None:
        """等待服务器就绪"""
        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                urllib.request.urlopen(url, timeout=1)
                logger.info("SGLang server is ready")
                return
            except (urllib.error.URLError, urllib.error.HTTPError):
                time.sleep(2)

        raise RuntimeError(f"SGLang server failed to start within {timeout}s")

    def _load_tokenizer(self, model_id: str) -> None:
        """加载 tokenizer"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """异步生成"""
        if config is None:
            config = GenerationConfig()

        start_time = time.time()

        # 尝试使用 SGLang function API
        try:
            result = await self._generate_with_sglang_api(messages, config)
        except Exception as e:
            logger.warning(f"SGLang function API failed: {e}, falling back to HTTP API")
            result = await self._generate_with_http_api(messages, config)

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Generation completed in {latency_ms:.2f}ms")

        return result

    async def _generate_with_sglang_api(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> GenerationResult:
        """使用 SGLang native API 生成"""
        import sglang as sgl

        @sgl.function
        def chat_completion(s, messages_list, max_tokens, temperature, top_p, top_k):
            for msg in messages_list:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    s += sgl.system(content)
                elif role == "user":
                    s += sgl.user(content)
                elif role == "assistant":
                    s += sgl.assistant(content)

            s += sgl.assistant(sgl.gen(
                "response",
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ))

        # 在线程池中运行
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
            None,
            lambda: chat_completion.run(
                messages_list=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
        )

        response_text = state["response"]

        # 获取 meta 信息
        meta = {}
        if hasattr(state, 'get_meta_info'):
            meta = {
                "prompt_tokens": state.get_meta_info("prompt_tokens", 0),
                "completion_tokens": state.get_meta_info("completion_tokens", 0),
                "cached_tokens": state.get_meta_info("cached_tokens", 0),
            }

        return GenerationResult(
            text=response_text,
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            total_tokens=meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0),
            cached_tokens=meta.get("cached_tokens", 0),
            finish_reason="stop"
        )

    async def _generate_with_http_api(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> GenerationResult:
        """使用 HTTP API 生成（OpenAI 兼容接口）"""
        import aiohttp

        server_url = self._sglang_config.get("server_url", "http://localhost:30000")

        payload = {
            "model": self.config.get("model_id"),
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"SGLang API error: {error_text}")

                result = await response.json()

        if "error" in result:
            raise RuntimeError(f"SGLang API error: {result['error']}")

        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = result.get("usage", {})

        return GenerationResult(
            text=message.get("content", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            cached_tokens=usage.get("cached_tokens", 0)
        )

    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """批量生成 - 利用 SGLang 的连续批处理能力"""
        if config is None:
            config = GenerationConfig()

        # 并发执行所有请求，SGLang 会自动进行批处理优化
        tasks = [
            self.generate_async(messages, config)
            for messages in batch_messages
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch generation error: {result}")
                outputs.append(GenerationResult(
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    finish_reason="error"
                ))
            else:
                outputs.append(result)

        return outputs

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成"""
        if config is None:
            config = GenerationConfig()

        import aiohttp

        server_url = self._sglang_config.get("server_url", "http://localhost:30000")

        payload = {
            "model": self.config.get("model_id"),
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"SGLang stream API error: {error_text}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        import json
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            # Fallback to non-streaming
            result = await self.generate_async(messages, config)
            yield result.text

    def supports_streaming(self) -> bool:
        """支持流式输出"""
        return True

    def supports_prefix_caching(self) -> bool:
        """支持前缀缓存（RadixAttention）"""
        return self._sglang_config.get("enable_prefix_caching", True)

    def supports_batch_inference(self) -> bool:
        """支持批量推理"""
        return True

    def unload_model(self) -> None:
        """卸载模型"""
        if self.runtime:
            try:
                if hasattr(self.runtime, 'shutdown'):
                    self.runtime.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down runtime: {e}")
            self.runtime = None

        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded = False
        logger.info("SGLang engine unloaded")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取前缀缓存统计"""
        stats = {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }

        # 尝试从 runtime 获取更详细的统计
        if self.runtime and hasattr(self.runtime, 'get_server_info'):
            try:
                server_info = self.runtime.get_server_info()
                if "cache_stats" in server_info:
                    stats.update(server_info["cache_stats"])
            except Exception:
                pass

        return stats

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        status = super().get_status()
        status["cache_stats"] = self.get_cache_stats()
        status["features"] = [
            "paged_attention",
            "radix_attention",
            "continuous_batching",
            "chunked_prefill",
            "streaming",
        ]
        return status
