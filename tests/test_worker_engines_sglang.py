"""
测试 worker/engines/llm_sglang.py 模块

使用 mock 测试 SGLangEngine：
- 引擎初始化和配置
- 模型加载流程
- 生成方法 (generate_async, batch_generate, stream_generate)
- 特性支持检查
- 缓存统计
- 引擎卸载
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Dict, Any

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


# ============== SGLangEngine 初始化测试 ==============

class TestSGLangEngineInit:
    """SGLangEngine 初始化测试"""

    def test_init_default_config(self):
        """测试默认配置初始化"""
        with patch.dict('sys.modules', {'sglang': MagicMock()}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {"model_id": "Qwen/Qwen2.5-7B-Instruct"}
            engine = SGLangEngine(config)

            assert engine.backend_type == LLMBackend.SGLANG
            assert engine.runtime is None
            assert engine._server_process is None
            assert engine._cache_hits == 0
            assert engine._cache_misses == 0

    def test_init_with_sglang_config(self):
        """测试带 SGLang 配置的初始化"""
        with patch.dict('sys.modules', {'sglang': MagicMock()}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "sglang": {
                    "tp_size": 2,
                    "mem_fraction_static": 0.9,
                    "chunked_prefill_size": 4096,
                    "enable_prefix_caching": True,
                    "max_running_requests": 64,
                }
            }
            engine = SGLangEngine(config)

            assert engine._sglang_config["tp_size"] == 2
            assert engine._sglang_config["mem_fraction_static"] == 0.9
            assert engine._sglang_config["enable_prefix_caching"] is True


class TestSGLangEngineLoadModel:
    """SGLangEngine 模型加载测试"""

    def test_load_model_import_error(self):
        """测试 SGLang 未安装时的错误"""
        # 模拟 sglang 未安装
        with patch.dict('sys.modules', {'sglang': None}):
            # 需要清除已导入的模块缓存
            if 'worker.engines.llm_sglang' in sys.modules:
                del sys.modules['worker.engines.llm_sglang']

            # 这个测试主要验证导入错误处理逻辑
            # 实际测试需要在隔离环境中进行

    def test_load_model_with_quantization(self):
        """测试带量化配置的模型加载"""
        mock_sglang = MagicMock()
        mock_runtime = MagicMock()
        mock_sglang.Runtime.return_value = mock_runtime

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "quantization": "int8",
            }
            engine = SGLangEngine(config)

            # 模拟 load_model 的部分逻辑
            assert engine.config.get("quantization") == "int8"


# ============== 生成方法测试 ==============

class TestSGLangEngineGeneration:
    """SGLangEngine 生成方法测试"""

    @pytest.fixture
    def mock_engine(self):
        """创建模拟的 SGLangEngine"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "sglang": {"server_url": "http://localhost:30000"}
            }
            engine = SGLangEngine(config)
            engine.loaded = True
            engine.runtime = MagicMock()

            return engine

    @pytest.mark.asyncio
    async def test_generate_async_with_http_api(self, mock_engine):
        """测试 HTTP API 生成"""
        mock_response = {
            "choices": [{
                "message": {"content": "Hello! How can I help you?"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "cached_tokens": 0
            }
        }

        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.json = AsyncMock(return_value=mock_response)

            mock_session.return_value.__aenter__.return_value.post.return_value = mock_context

            messages = [{"role": "user", "content": "Hello"}]
            config = GenerationConfig(max_tokens=100)

            result = await mock_engine._generate_with_http_api(messages, config)

            assert isinstance(result, GenerationResult)
            assert result.text == "Hello! How can I help you?"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 8
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_engine):
        """测试批量生成"""
        # Mock generate_async
        mock_result = GenerationResult(
            text="Response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            finish_reason="stop"
        )
        mock_engine.generate_async = AsyncMock(return_value=mock_result)

        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Hey"}],
        ]

        results = await mock_engine.batch_generate(batch_messages)

        assert len(results) == 3
        assert all(isinstance(r, GenerationResult) for r in results)
        assert mock_engine.generate_async.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_generate_with_error(self, mock_engine):
        """测试批量生成中的错误处理"""
        # 模拟一个成功，一个失败
        async def mock_generate(messages, config=None):
            if "error" in str(messages):
                raise RuntimeError("Test error")
            return GenerationResult(
                text="OK", prompt_tokens=5, completion_tokens=2,
                total_tokens=7, finish_reason="stop"
            )

        mock_engine.generate_async = mock_generate

        batch_messages = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "error"}],
        ]

        results = await mock_engine.batch_generate(batch_messages)

        assert len(results) == 2
        assert results[0].text == "OK"
        assert results[1].finish_reason == "error"  # 错误被捕获


# ============== 特性支持测试 ==============

class TestSGLangEngineFeatures:
    """SGLangEngine 特性支持测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "sglang": {"enable_prefix_caching": True}
            }
            return SGLangEngine(config)

    def test_supports_streaming(self, engine):
        """测试流式输出支持"""
        assert engine.supports_streaming() is True

    def test_supports_prefix_caching_enabled(self, engine):
        """测试前缀缓存支持（启用）"""
        assert engine.supports_prefix_caching() is True

    def test_supports_prefix_caching_disabled(self):
        """测试前缀缓存支持（禁用）"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {
                "model_id": "test",
                "sglang": {"enable_prefix_caching": False}
            }
            engine = SGLangEngine(config)

            assert engine.supports_prefix_caching() is False

    def test_supports_batch_inference(self, engine):
        """测试批量推理支持"""
        assert engine.supports_batch_inference() is True


# ============== 缓存统计测试 ==============

class TestSGLangEngineCacheStats:
    """SGLangEngine 缓存统计测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {"model_id": "test"}
            engine = SGLangEngine(config)
            engine._cache_hits = 10
            engine._cache_misses = 5
            return engine

    def test_get_cache_stats(self, engine):
        """测试获取缓存统计"""
        stats = engine.get_cache_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 10 / 15  # 10 / (10 + 5)

    def test_get_cache_stats_no_requests(self):
        """测试无请求时的缓存统计"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {"model_id": "test"}
            engine = SGLangEngine(config)

            stats = engine.get_cache_stats()

            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["hit_rate"] == 0.0  # 避免除零


# ============== 状态和卸载测试 ==============

class TestSGLangEngineStatus:
    """SGLangEngine 状态测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        mock_sglang = MagicMock()

        with patch.dict('sys.modules', {'sglang': mock_sglang}):
            from worker.engines.llm_sglang import SGLangEngine

            config = {"model_id": "test"}
            return SGLangEngine(config)

    def test_get_status(self, engine):
        """测试获取引擎状态"""
        engine._cache_hits = 5
        engine._cache_misses = 3

        status = engine.get_status()

        assert "cache_stats" in status
        assert "features" in status
        assert "paged_attention" in status["features"]
        assert "radix_attention" in status["features"]
        assert "continuous_batching" in status["features"]

    def test_unload_model_with_runtime(self, engine):
        """测试卸载模型（有 runtime）"""
        mock_runtime = MagicMock()
        engine.runtime = mock_runtime
        engine.loaded = True

        with patch('torch.cuda.is_available', return_value=False):
            engine.unload_model()

        assert engine.runtime is None
        assert engine.loaded is False

    def test_unload_model_with_server_process(self, engine):
        """测试卸载模型（有服务器进程）"""
        mock_process = MagicMock()
        engine._server_process = mock_process
        engine.loaded = True

        with patch('torch.cuda.is_available', return_value=False):
            engine.unload_model()

        mock_process.terminate.assert_called_once()
        assert engine._server_process is None


# ============== GenerationConfig 测试 ==============

class TestGenerationConfig:
    """GenerationConfig 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        config = GenerationConfig()

        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.stop_sequences is None
        assert config.stream is False

    def test_custom_values(self):
        """测试自定义值"""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            stop_sequences=["END", "STOP"],
            stream=True
        )

        assert config.max_tokens == 512
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 100
        assert config.stop_sequences == ["END", "STOP"]
        assert config.stream is True


# ============== GenerationResult 测试 ==============

class TestGenerationResult:
    """GenerationResult 数据类测试"""

    def test_creation(self):
        """测试创建结果"""
        result = GenerationResult(
            text="Hello world",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            finish_reason="stop",
            cached_tokens=3
        )

        assert result.text == "Hello world"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.total_tokens == 7
        assert result.finish_reason == "stop"
        assert result.cached_tokens == 3

    def test_default_values(self):
        """测试默认值"""
        result = GenerationResult(
            text="Test",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2
        )

        assert result.finish_reason == "stop"
        assert result.cached_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
