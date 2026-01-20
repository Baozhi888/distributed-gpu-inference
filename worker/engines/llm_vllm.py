"""
vLLM 高性能推理引擎

特点：
- PagedAttention 高效内存管理（PyTorch Foundation 项目）
- 连续批处理 (Continuous Batching)
- 张量并行 (Tensor Parallelism)
- 前缀缓存 (Prefix Caching)
- 分块预填充 (Chunked Prefill)
- 多种量化支持 (AWQ, GPTQ, INT8, FP8)

参考：https://github.com/vllm-project/vllm
"""
from typing import Dict, Any, List, Optional, AsyncIterator
import logging
import asyncio
import time

from .llm_base import LLMBaseEngine, LLMBackend, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class VLLMEngine(LLMBaseEngine):
    """
    基于 vLLM 的高性能 LLM 推理引擎

    vLLM 核心优势：
    1. PagedAttention - 高效显存管理，提升利用率到 85%+
    2. 成熟的生态系统和社区支持
    3. 支持多种量化方案 (AWQ, GPTQ, FP8, INT8)
    4. 原生支持张量并行，适合多卡部署
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = LLMBackend.VLLM
        self.llm = None
        self._vllm_config = config.get("vllm", {})
        self._default_sampling_params = None

    def load_model(self) -> None:
        """加载模型到 vLLM"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Please install with:\n"
                "  pip install vllm"
            )

        model_id = self.config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"Loading model with vLLM: {model_id}")

        # 合并配置
        tp_size = self._vllm_config.get("tensor_parallel_size", 1)
        gpu_util = self._vllm_config.get("gpu_memory_utilization", 0.85)
        max_model_len = self._vllm_config.get("max_model_len", 8192)
        max_num_seqs = self._vllm_config.get("max_num_seqs", 256)
        enable_prefix_caching = self._vllm_config.get("enable_prefix_caching", True)
        enable_chunked_prefill = self._vllm_config.get("enable_chunked_prefill", True)

        # vLLM 配置
        llm_config = {
            "model": model_id,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": gpu_util,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "trust_remote_code": True,
            "enforce_eager": self._vllm_config.get("enforce_eager", False),
        }

        # 前缀缓存
        if enable_prefix_caching:
            llm_config["enable_prefix_caching"] = True

        # 分块预填充
        if enable_chunked_prefill:
            llm_config["enable_chunked_prefill"] = True

        # 量化配置
        quantization = self.config.get("quantization")
        if quantization:
            if quantization in ["awq", "gptq", "squeezellm", "fp8", "int8"]:
                llm_config["quantization"] = quantization
            logger.info(f"Using quantization: {quantization}")

        # dtype 配置
        dtype = self._vllm_config.get("dtype", "auto")
        llm_config["dtype"] = dtype

        # 创建 LLM 实例
        logger.info(f"Creating vLLM instance with config: {llm_config}")
        self.llm = LLM(**llm_config)

        # 获取 tokenizer
        self.tokenizer = self.llm.get_tokenizer()

        # 默认采样参数
        self._default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )

        self.loaded = True
        logger.info("vLLM engine loaded successfully")
        logger.info(f"  - Tensor Parallel: {tp_size}")
        logger.info(f"  - GPU Memory Utilization: {gpu_util}")
        logger.info(f"  - Prefix Caching: {enable_prefix_caching}")
        logger.info(f"  - Max Sequences: {max_num_seqs}")

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """异步生成"""
        if config is None:
            config = GenerationConfig()

        # vLLM 的 generate 是同步的，在线程池中执行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_sync(messages, config)
        )

    def _generate_sync(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> GenerationResult:
        """同步生成"""
        from vllm import SamplingParams

        start_time = time.time()

        # 格式化输入
        prompt = self._format_messages(messages)

        # 采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 1.0,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_tokens,
            use_beam_search=config.temperature == 0,
            stop=config.stop_sequences,
        )

        # 执行生成
        outputs = self.llm.generate([prompt], sampling_params)

        # 解析输出
        output = outputs[0]
        response_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason or "stop"

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Generation completed in {latency_ms:.2f}ms")

        return GenerationResult(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason
        )

    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """批量生成 - 利用 vLLM 的连续批处理"""
        if config is None:
            config = GenerationConfig()

        # 在线程池中执行批量生成
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._batch_generate_sync(batch_messages, config)
        )

    def _batch_generate_sync(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerationConfig
    ) -> List[GenerationResult]:
        """同步批量生成"""
        from vllm import SamplingParams

        # 格式化所有输入
        prompts = [self._format_messages(msgs) for msgs in batch_messages]

        # 采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 1.0,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
        )

        # 批量生成（vLLM 自动优化批处理）
        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            response_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            finish_reason = output.outputs[0].finish_reason or "stop"

            results.append(GenerationResult(
                text=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=finish_reason
            ))

        return results

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息为 prompt"""
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Fallback: 简单拼接
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        formatted.append("assistant: ")
        return "\n".join(formatted)

    def supports_streaming(self) -> bool:
        """vLLM 同步模式不支持流式"""
        return False

    def supports_prefix_caching(self) -> bool:
        """支持前缀缓存"""
        return self._vllm_config.get("enable_prefix_caching", True)

    def supports_batch_inference(self) -> bool:
        """支持批量推理"""
        return True

    def unload_model(self) -> None:
        """卸载模型"""
        if self.llm:
            del self.llm
            self.llm = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        self._default_sampling_params = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded = False
        logger.info("vLLM engine unloaded")

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        status = super().get_status()
        status["features"] = [
            "paged_attention",
            "continuous_batching",
            "tensor_parallelism",
            "prefix_caching",
            "chunked_prefill",
        ]
        return status


class VLLMAsyncEngine(LLMBaseEngine):
    """
    基于 vLLM AsyncLLMEngine 的异步推理引擎

    适用于需要流式输出和高并发的场景
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend_type = LLMBackend.VLLM
        self.engine = None
        self._vllm_config = config.get("vllm", {})

    def load_model(self) -> None:
        """加载模型到 AsyncLLMEngine"""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise ImportError(
                "vLLM not installed. Please install with:\n"
                "  pip install vllm"
            )

        model_id = self.config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"Loading model with vLLM AsyncEngine: {model_id}")

        # 合并配置
        tp_size = self._vllm_config.get("tensor_parallel_size", 1)
        gpu_util = self._vllm_config.get("gpu_memory_utilization", 0.85)
        max_model_len = self._vllm_config.get("max_model_len", 8192)
        enable_prefix_caching = self._vllm_config.get("enable_prefix_caching", True)
        enable_chunked_prefill = self._vllm_config.get("enable_chunked_prefill", True)

        # 引擎参数
        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        # 量化配置
        quantization = self.config.get("quantization")
        if quantization:
            engine_args.quantization = quantization

        # 创建异步引擎
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 获取 tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.loaded = True
        logger.info("vLLM AsyncEngine loaded successfully")

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """异步生成"""
        from vllm import SamplingParams
        import uuid

        if config is None:
            config = GenerationConfig()

        start_time = time.time()

        # 格式化输入
        prompt = self._format_messages(messages)
        request_id = str(uuid.uuid4())

        # 采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 1.0,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
        )

        # 异步生成
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise RuntimeError("No output generated")

        output = final_output.outputs[0]
        response_text = output.text
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(output.token_ids)
        finish_reason = output.finish_reason or "stop"

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Generation completed in {latency_ms:.2f}ms")

        return GenerationResult(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason
        )

    async def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """批量生成"""
        if config is None:
            config = GenerationConfig()

        # 并发执行所有请求
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
        from vllm import SamplingParams
        import uuid

        if config is None:
            config = GenerationConfig()

        # 格式化输入
        prompt = self._format_messages(messages)
        request_id = str(uuid.uuid4())

        # 采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 1.0,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
        )

        # 流式生成
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        prev_text = ""
        async for request_output in results_generator:
            output = request_output.outputs[0]
            new_text = output.text[len(prev_text):]
            prev_text = output.text

            if new_text:
                yield new_text

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息为 prompt"""
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Fallback
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        formatted.append("assistant: ")
        return "\n".join(formatted)

    def supports_streaming(self) -> bool:
        """支持流式输出"""
        return True

    def supports_prefix_caching(self) -> bool:
        """支持前缀缓存"""
        return self._vllm_config.get("enable_prefix_caching", True)

    def supports_batch_inference(self) -> bool:
        """支持批量推理"""
        return True

    def unload_model(self) -> None:
        """卸载模型"""
        if self.engine:
            del self.engine
            self.engine = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded = False
        logger.info("vLLM AsyncEngine unloaded")

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        status = super().get_status()
        status["async_mode"] = True
        status["features"] = [
            "paged_attention",
            "continuous_batching",
            "tensor_parallelism",
            "prefix_caching",
            "async_inference",
            "streaming",
        ]
        return status
