"""
测试 worker/engines/llm_vllm.py 模块

使用 mock 测试 VLLMEngine 和 VLLMAsyncEngine：
- 引擎初始化和配置
- 模型加载流程
- 生成方法 (generate_async, batch_generate, stream_generate)
- 消息格式化
- 特性支持检查
- 引擎卸载
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Dict, Any, List

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.engines.llm_base import (
    LLMBackend,
    GenerationConfig,
    GenerationResult,
)


# ============== VLLMEngine 初始化测试 ==============

class TestVLLMEngineInit:
    """VLLMEngine 初始化测试"""

    def test_init_default_config(self):
        """测试默认配置初始化"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {"model_id": "Qwen/Qwen2.5-7B-Instruct"}
            engine = VLLMEngine(config)

            assert engine.backend_type == LLMBackend.VLLM
            assert engine.llm is None
            assert engine._default_sampling_params is None

    def test_init_with_vllm_config(self):
        """测试带 vLLM 配置的初始化"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {
                "model_id": "test-model",
                "vllm": {
                    "tensor_parallel_size": 2,
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                    "max_num_seqs": 128,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                }
            }
            engine = VLLMEngine(config)

            assert engine._vllm_config["tensor_parallel_size"] == 2
            assert engine._vllm_config["gpu_memory_utilization"] == 0.9
            assert engine._vllm_config["max_model_len"] == 4096


# ============== VLLMEngine 消息格式化测试 ==============

class TestVLLMEngineFormatMessages:
    """VLLMEngine 消息格式化测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {"model_id": "test"}
            engine = VLLMEngine(config)
            engine.tokenizer = None  # 无 tokenizer 时使用 fallback
            return engine

    def test_format_messages_fallback(self, engine):
        """测试 fallback 消息格式化"""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = engine._format_messages(messages)

        assert "system: You are helpful." in result
        assert "user: Hello" in result
        assert "assistant:" in result

    def test_format_messages_with_tokenizer(self, engine):
        """测试使用 tokenizer 的消息格式化"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        engine.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        result = engine._format_messages(messages)

        mock_tokenizer.apply_chat_template.assert_called_once()
        assert "<|im_start|>" in result


# ============== VLLMEngine 生成测试 ==============

class TestVLLMEngineGeneration:
    """VLLMEngine 生成方法测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟的 VLLMEngine"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {"model_id": "test"}
            engine = VLLMEngine(config)
            engine.loaded = True
            engine.tokenizer = None

            # Mock LLM
            mock_llm = MagicMock()
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock(
                text="Generated text",
                token_ids=[1, 2, 3, 4, 5],
                finish_reason="stop"
            )]
            mock_output.prompt_token_ids = [1, 2, 3]
            mock_llm.generate.return_value = [mock_output]

            engine.llm = mock_llm

            return engine

    @pytest.mark.asyncio
    async def test_generate_async(self, mock_engine):
        """测试异步生成"""
        messages = [{"role": "user", "content": "Hello"}]
        config = GenerationConfig(max_tokens=100)

        # Mock _generate_sync
        mock_result = GenerationResult(
            text="Hello!",
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
            finish_reason="stop"
        )
        mock_engine._generate_sync = MagicMock(return_value=mock_result)

        result = await mock_engine.generate_async(messages, config)

        assert isinstance(result, GenerationResult)
        assert result.text == "Hello!"

    def test_generate_sync(self, mock_engine):
        """测试同步生成"""
        mock_vllm = MagicMock()
        mock_sampling_params = MagicMock()
        mock_vllm.SamplingParams = mock_sampling_params

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            messages = [{"role": "user", "content": "Hello"}]
            config = GenerationConfig(max_tokens=100, temperature=0.8)

            result = mock_engine._generate_sync(messages, config)

            assert isinstance(result, GenerationResult)
            assert result.text == "Generated text"
            assert result.prompt_tokens == 3
            assert result.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_engine):
        """测试批量生成"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            # Mock batch output
            mock_outputs = []
            for i in range(3):
                mock_output = MagicMock()
                mock_output.outputs = [MagicMock(
                    text=f"Response {i}",
                    token_ids=[1, 2, 3],
                    finish_reason="stop"
                )]
                mock_output.prompt_token_ids = [1, 2]
                mock_outputs.append(mock_output)

            mock_engine.llm.generate.return_value = mock_outputs

            batch_messages = [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Hi"}],
                [{"role": "user", "content": "Hey"}],
            ]

            results = await mock_engine.batch_generate(batch_messages)

            assert len(results) == 3
            assert all(isinstance(r, GenerationResult) for r in results)


# ============== VLLMEngine 特性支持测试 ==============

class TestVLLMEngineFeatures:
    """VLLMEngine 特性支持测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {
                "model_id": "test",
                "vllm": {"enable_prefix_caching": True}
            }
            return VLLMEngine(config)

    def test_supports_streaming(self, engine):
        """测试流式输出支持（同步引擎不支持）"""
        assert engine.supports_streaming() is False

    def test_supports_prefix_caching(self, engine):
        """测试前缀缓存支持"""
        assert engine.supports_prefix_caching() is True

    def test_supports_batch_inference(self, engine):
        """测试批量推理支持"""
        assert engine.supports_batch_inference() is True

    def test_get_status(self, engine):
        """测试获取状态"""
        status = engine.get_status()

        assert "features" in status
        assert "paged_attention" in status["features"]
        assert "continuous_batching" in status["features"]
        assert "tensor_parallelism" in status["features"]


# ============== VLLMEngine 卸载测试 ==============

class TestVLLMEngineUnload:
    """VLLMEngine 卸载测试"""

    def test_unload_model(self):
        """测试卸载模型"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMEngine

            config = {"model_id": "test"}
            engine = VLLMEngine(config)
            engine.llm = MagicMock()
            engine.tokenizer = MagicMock()
            engine._default_sampling_params = MagicMock()
            engine.loaded = True

            with patch('torch.cuda.is_available', return_value=False):
                engine.unload_model()

            assert engine.llm is None
            assert engine.tokenizer is None
            assert engine._default_sampling_params is None
            assert engine.loaded is False


# ============== VLLMAsyncEngine 初始化测试 ==============

class TestVLLMAsyncEngineInit:
    """VLLMAsyncEngine 初始化测试"""

    def test_init_default_config(self):
        """测试默认配置初始化"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {"model_id": "test"}
            engine = VLLMAsyncEngine(config)

            assert engine.backend_type == LLMBackend.VLLM
            assert engine.engine is None

    def test_init_with_config(self):
        """测试带配置的初始化"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {
                "model_id": "test",
                "vllm": {
                    "tensor_parallel_size": 4,
                    "enable_prefix_caching": False,
                }
            }
            engine = VLLMAsyncEngine(config)

            assert engine._vllm_config["tensor_parallel_size"] == 4


# ============== VLLMAsyncEngine 生成测试 ==============

class TestVLLMAsyncEngineGeneration:
    """VLLMAsyncEngine 生成方法测试"""

    @pytest.fixture
    def mock_async_engine(self):
        """创建模拟的 VLLMAsyncEngine"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {"model_id": "test"}
            engine = VLLMAsyncEngine(config)
            engine.loaded = True
            engine.tokenizer = None

            return engine

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_async_engine):
        """测试批量生成"""
        # Mock generate_async
        mock_result = GenerationResult(
            text="Response",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            finish_reason="stop"
        )
        mock_async_engine.generate_async = AsyncMock(return_value=mock_result)

        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]

        results = await mock_async_engine.batch_generate(batch_messages)

        assert len(results) == 2
        assert all(r.text == "Response" for r in results)

    @pytest.mark.asyncio
    async def test_batch_generate_with_error(self, mock_async_engine):
        """测试批量生成中的错误处理"""
        call_count = 0

        async def mock_generate(messages, config=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Test error")
            return GenerationResult(
                text="OK",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                finish_reason="stop"
            )

        mock_async_engine.generate_async = mock_generate

        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Error trigger"}],
            [{"role": "user", "content": "Hi"}],
        ]

        results = await mock_async_engine.batch_generate(batch_messages)

        assert len(results) == 3
        assert results[0].text == "OK"
        assert results[1].finish_reason == "error"
        assert results[2].text == "OK"


# ============== VLLMAsyncEngine 特性支持测试 ==============

class TestVLLMAsyncEngineFeatures:
    """VLLMAsyncEngine 特性支持测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {
                "model_id": "test",
                "vllm": {"enable_prefix_caching": True}
            }
            return VLLMAsyncEngine(config)

    def test_supports_streaming(self, engine):
        """测试流式输出支持（异步引擎支持）"""
        assert engine.supports_streaming() is True

    def test_supports_prefix_caching(self, engine):
        """测试前缀缓存支持"""
        assert engine.supports_prefix_caching() is True

    def test_supports_batch_inference(self, engine):
        """测试批量推理支持"""
        assert engine.supports_batch_inference() is True

    def test_get_status(self, engine):
        """测试获取状态"""
        status = engine.get_status()

        assert status["async_mode"] is True
        assert "features" in status
        assert "async_inference" in status["features"]
        assert "streaming" in status["features"]


# ============== VLLMAsyncEngine 卸载测试 ==============

class TestVLLMAsyncEngineUnload:
    """VLLMAsyncEngine 卸载测试"""

    def test_unload_model(self):
        """测试卸载模型"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {"model_id": "test"}
            engine = VLLMAsyncEngine(config)
            engine.engine = MagicMock()
            engine.tokenizer = MagicMock()
            engine.loaded = True

            with patch('torch.cuda.is_available', return_value=False):
                engine.unload_model()

            assert engine.engine is None
            assert engine.tokenizer is None
            assert engine.loaded is False


# ============== 消息格式化测试 (VLLMAsyncEngine) ==============

class TestVLLMAsyncEngineFormatMessages:
    """VLLMAsyncEngine 消息格式化测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_vllm = MagicMock()

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            from worker.engines.llm_vllm import VLLMAsyncEngine

            config = {"model_id": "test"}
            engine = VLLMAsyncEngine(config)
            engine.tokenizer = None
            return engine

    def test_format_messages_fallback(self, engine):
        """测试 fallback 消息格式化"""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
            {"role": "user", "content": "Tell me more"},
        ]

        result = engine._format_messages(messages)

        assert "user: What is AI?" in result
        assert "assistant: AI is..." in result
        assert "user: Tell me more" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
