"""
测试 worker/distributed/model_shard.py 模块

覆盖：
- LayerInfo 数据类
- ModelShard 模型分片
- ShardedModelLoader 分片加载器
- get_layer_range_for_worker 工具函数
- 辅助函数 (_get_layer_module, _get_embedding_module 等)
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.distributed.model_shard import (
    LayerInfo,
    ModelShard,
    ShardedModelLoader,
    get_layer_range_for_worker,
    _get_layer_module,
    _get_embedding_module,
    _get_norm_module,
    _create_device_map_for_layers,
)


# ============== LayerInfo 测试 ==============

class TestLayerInfo:
    """LayerInfo 数据类测试"""

    def test_creation(self):
        """测试创建"""
        info = LayerInfo(
            layer_idx=5,
            layer_name="model.layers.5",
            param_count=1000000,
            memory_bytes=2000000,
        )

        assert info.layer_idx == 5
        assert info.layer_name == "model.layers.5"
        assert info.param_count == 1000000
        assert info.memory_bytes == 2000000


# ============== ModelShard 测试 ==============

class TestModelShard:
    """ModelShard 测试"""

    @pytest.fixture
    def shard(self):
        """创建 ModelShard 实例"""
        return ModelShard(
            model_id="test-model",
            start_layer=0,
            end_layer=10,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, shard):
        """测试初始化"""
        assert shard.model_id == "test-model"
        assert shard.start_layer == 0
        assert shard.end_layer == 10
        assert shard.device == "cpu"
        assert shard.dtype == torch.float32
        assert len(shard.layers) == 0
        assert shard.config is None
        assert shard.is_first_shard is False
        assert shard.is_last_shard is False

    def test_get_memory_usage_empty(self, shard):
        """测试空分片内存使用"""
        memory = shard.get_memory_usage()
        assert memory == 0.0

    def test_get_memory_usage_with_layers(self, shard):
        """测试有层时的内存使用"""
        # 添加一个简单的层
        layer = nn.Linear(512, 512)
        shard.layers.append(layer)

        memory = shard.get_memory_usage()
        assert memory > 0

    def test_get_layer_count(self, shard):
        """测试获取层数"""
        assert shard.get_layer_count() == 0

        shard.layers.append(nn.Linear(10, 10))
        shard.layers.append(nn.Linear(10, 10))

        assert shard.get_layer_count() == 2

    def test_get_logits_not_last_shard(self, shard):
        """测试非最后分片调用 get_logits"""
        shard.is_last_shard = False
        hidden_states = torch.randn(1, 10, 512)

        with pytest.raises(RuntimeError, match="only be called on the last shard"):
            shard.get_logits(hidden_states)

    def test_get_logits_last_shard_no_head(self, shard):
        """测试最后分片但没有 lm_head"""
        shard.is_last_shard = True
        hidden_states = torch.randn(1, 10, 512)

        with pytest.raises(RuntimeError, match="No lm_head available"):
            shard.get_logits(hidden_states)

    def test_get_logits_last_shard_with_head(self, shard):
        """测试最后分片有 lm_head"""
        shard.is_last_shard = True
        shard.lm_head = nn.Linear(512, 1000)

        hidden_states = torch.randn(1, 10, 512)
        logits = shard.get_logits(hidden_states)

        assert logits.shape == (1, 10, 1000)


class TestModelShardForward:
    """ModelShard forward 测试"""

    @pytest.fixture
    def mock_layer(self):
        """创建模拟层"""
        layer = MagicMock()
        layer.return_value = (torch.randn(1, 10, 256), (torch.randn(1, 8, 10, 32), torch.randn(1, 8, 10, 32)))
        return layer

    def test_forward_middle_shard(self, mock_layer):
        """测试中间分片的 forward"""
        shard = ModelShard(
            model_id="test",
            start_layer=5,
            end_layer=10,
            device="cpu",
        )
        shard.is_first_shard = False
        shard.is_last_shard = False
        shard.layers.append(mock_layer)

        hidden_states = torch.randn(1, 10, 256)
        output, kv_cache = shard.forward(hidden_states, use_cache=True)

        assert mock_layer.called
        assert output is not None
        assert kv_cache is not None

    def test_forward_first_shard_with_embedding(self, mock_layer):
        """测试第一个分片带 embedding"""
        shard = ModelShard(
            model_id="test",
            start_layer=0,
            end_layer=5,
            device="cpu",
        )
        shard.is_first_shard = True
        shard.is_last_shard = False
        shard.embed_tokens = nn.Embedding(1000, 256)
        shard.layers.append(mock_layer)

        # 输入是 token ids
        input_ids = torch.randint(0, 1000, (1, 10))
        output, kv_cache = shard.forward(input_ids, use_cache=True)

        assert output is not None

    def test_forward_last_shard_with_norm(self, mock_layer):
        """测试最后分片带 norm"""
        shard = ModelShard(
            model_id="test",
            start_layer=25,
            end_layer=32,
            device="cpu",
        )
        shard.is_first_shard = False
        shard.is_last_shard = True
        shard.norm = nn.LayerNorm(256)
        shard.layers.append(mock_layer)

        hidden_states = torch.randn(1, 10, 256)
        output, kv_cache = shard.forward(hidden_states, use_cache=True)

        assert output is not None

    def test_forward_without_cache(self, mock_layer):
        """测试不使用缓存的 forward"""
        mock_layer.return_value = (torch.randn(1, 10, 256), None)

        shard = ModelShard(
            model_id="test",
            start_layer=0,
            end_layer=5,
            device="cpu",
        )
        shard.layers.append(mock_layer)

        hidden_states = torch.randn(1, 10, 256)
        output, kv_cache = shard.forward(hidden_states, use_cache=False)

        assert output is not None
        assert kv_cache is None


# ============== ShardedModelLoader 测试 ==============

class TestShardedModelLoader:
    """ShardedModelLoader 测试"""

    @pytest.fixture
    def loader(self):
        """创建加载器实例"""
        return ShardedModelLoader(model_id="test-model")

    def test_init(self, loader):
        """测试初始化"""
        assert loader.model_id == "test-model"
        assert loader.config is None
        assert loader.total_layers == 0

    @patch('worker.distributed.model_shard.AutoConfig')
    def test_analyze_model(self, mock_auto_config, loader):
        """测试模型分析"""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.intermediate_size = 14336
        mock_auto_config.from_pretrained.return_value = mock_config

        info = loader.analyze_model()

        assert info["model_id"] == "test-model"
        assert info["total_layers"] == 32
        assert info["hidden_size"] == 4096
        assert info["num_attention_heads"] == 32
        assert info["memory_per_layer_gb"] > 0

    @patch('worker.distributed.model_shard.AutoConfig')
    def test_create_shard_plan(self, mock_auto_config, loader):
        """测试创建分片计划"""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.intermediate_size = 14336
        mock_auto_config.from_pretrained.return_value = mock_config

        # 3 个 Worker，各 24GB 显存
        worker_memory = [24.0, 24.0, 24.0]

        plan = loader.create_shard_plan(worker_memory, reserve_ratio=0.2)

        # 应该返回分片计划
        assert len(plan) > 0
        # 覆盖所有层
        assert plan[0][0] == 0
        assert plan[-1][1] == 32

    @patch('worker.distributed.model_shard.AutoConfig')
    def test_create_shard_plan_insufficient_memory(self, mock_auto_config, loader):
        """测试内存不足的分片计划"""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 80  # 80 层
        mock_config.hidden_size = 8192
        mock_config.num_attention_heads = 64
        mock_config.num_key_value_heads = 8
        mock_config.intermediate_size = 28672  # 大模型
        mock_auto_config.from_pretrained.return_value = mock_config

        # 显存不足
        worker_memory = [8.0]  # 只有 8GB

        with pytest.raises(ValueError, match="Insufficient memory"):
            loader.create_shard_plan(worker_memory, reserve_ratio=0.2)


# ============== get_layer_range_for_worker 测试 ==============

class TestGetLayerRangeForWorker:
    """get_layer_range_for_worker 函数测试"""

    def test_even_distribution(self):
        """测试均匀分配"""
        # 32 层，4 个 Worker
        ranges = [
            get_layer_range_for_worker(32, 4, i)
            for i in range(4)
        ]

        assert ranges[0] == (0, 8)
        assert ranges[1] == (8, 16)
        assert ranges[2] == (16, 24)
        assert ranges[3] == (24, 32)

    def test_uneven_distribution(self):
        """测试不均匀分配"""
        # 10 层，3 个 Worker
        ranges = [
            get_layer_range_for_worker(10, 3, i)
            for i in range(3)
        ]

        # 10 / 3 = 3 余 1，前 1 个 Worker 多 1 层
        assert ranges[0] == (0, 4)   # 4 层
        assert ranges[1] == (4, 7)   # 3 层
        assert ranges[2] == (7, 10)  # 3 层

    def test_single_worker(self):
        """测试单个 Worker"""
        start, end = get_layer_range_for_worker(32, 1, 0)

        assert start == 0
        assert end == 32

    def test_more_workers_than_layers(self):
        """测试 Worker 多于层数"""
        # 5 层，10 个 Worker
        ranges = [
            get_layer_range_for_worker(5, 10, i)
            for i in range(10)
        ]

        # 前 5 个 Worker 各 1 层，后 5 个 Worker 0 层
        assert ranges[0] == (0, 1)
        assert ranges[4] == (4, 5)
        assert ranges[5] == (5, 5)  # 空范围
        assert ranges[9] == (5, 5)  # 空范围


# ============== 辅助函数测试 ==============

class TestHelperFunctions:
    """辅助函数测试"""

    def test_get_layer_module_llama_style(self):
        """测试获取 Llama 风格模型的层"""
        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

        layers = _get_layer_module(model, None)

        assert layers is model.model.layers

    def test_get_layer_module_gpt_style(self):
        """测试获取 GPT 风格模型的层"""
        model = MagicMock()
        model.model = None  # 没有 model 属性
        model.transformer = MagicMock()
        model.transformer.h = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

        # 需要确保 hasattr 正确工作
        del model.model
        layers = _get_layer_module(model, None)

        assert layers is model.transformer.h

    def test_get_layer_module_none(self):
        """测试无法获取层"""
        model = MagicMock()
        model.model = None
        model.transformer = None

        # Mock hasattr to return False
        del model.model
        del model.transformer

        layers = _get_layer_module(model, None)

        assert layers is None

    def test_get_embedding_module_llama_style(self):
        """测试获取 Llama 风格的 embedding"""
        model = MagicMock()
        model.model = MagicMock()
        model.model.embed_tokens = nn.Embedding(1000, 256)

        embedding = _get_embedding_module(model, None)

        assert embedding is model.model.embed_tokens

    def test_get_embedding_module_gpt_style(self):
        """测试获取 GPT 风格的 embedding"""
        model = MagicMock()
        model.model = None
        del model.model
        model.transformer = MagicMock()
        model.transformer.wte = nn.Embedding(1000, 256)

        embedding = _get_embedding_module(model, None)

        assert embedding is model.transformer.wte

    def test_get_norm_module_llama_style(self):
        """测试获取 Llama 风格的 norm"""
        model = MagicMock()
        model.model = MagicMock()
        model.model.norm = nn.LayerNorm(256)

        norm = _get_norm_module(model, None)

        assert norm is model.model.norm

    def test_get_norm_module_gpt_style(self):
        """测试获取 GPT 风格的 norm"""
        model = MagicMock()
        model.model = None
        del model.model
        model.transformer = MagicMock()
        model.transformer.ln_f = nn.LayerNorm(256)

        norm = _get_norm_module(model, None)

        assert norm is model.transformer.ln_f


# ============== _create_device_map_for_layers 测试 ==============

class TestCreateDeviceMapForLayers:
    """_create_device_map_for_layers 函数测试"""

    def test_middle_layers_only(self):
        """测试只加载中间层"""
        device_map = _create_device_map_for_layers(
            config=None,
            start_layer=10,
            end_layer=20,
            device="cuda:0",
            include_embeddings=False,
            include_lm_head=False,
        )

        assert "model.embed_tokens" not in device_map
        assert "lm_head" not in device_map
        assert device_map["model.layers.10"] == "cuda:0"
        assert device_map["model.layers.19"] == "cuda:0"
        assert "model.layers.20" not in device_map

    def test_first_shard_with_embeddings(self):
        """测试第一个分片包含 embedding"""
        device_map = _create_device_map_for_layers(
            config=None,
            start_layer=0,
            end_layer=10,
            device="cuda:0",
            include_embeddings=True,
            include_lm_head=False,
        )

        assert device_map["model.embed_tokens"] == "cuda:0"
        assert device_map["model.layers.0"] == "cuda:0"
        assert "lm_head" not in device_map

    def test_last_shard_with_lm_head(self):
        """测试最后分片包含 lm_head"""
        device_map = _create_device_map_for_layers(
            config=None,
            start_layer=25,
            end_layer=32,
            device="cuda:0",
            include_embeddings=False,
            include_lm_head=True,
        )

        assert "model.embed_tokens" not in device_map
        assert device_map["model.layers.25"] == "cuda:0"
        assert device_map["model.norm"] == "cuda:0"
        assert device_map["lm_head"] == "cuda:0"

    def test_full_model_single_device(self):
        """测试完整模型单设备"""
        device_map = _create_device_map_for_layers(
            config=None,
            start_layer=0,
            end_layer=32,
            device="cuda:0",
            include_embeddings=True,
            include_lm_head=True,
        )

        assert device_map["model.embed_tokens"] == "cuda:0"
        assert device_map["model.layers.0"] == "cuda:0"
        assert device_map["model.layers.31"] == "cuda:0"
        assert device_map["model.norm"] == "cuda:0"
        assert device_map["lm_head"] == "cuda:0"


# ============== 集成测试 ==============

class TestModelShardIntegration:
    """模型分片集成测试"""

    def test_layer_range_coverage(self):
        """测试层范围完整覆盖"""
        total_layers = 80
        num_workers = 5

        ranges = [
            get_layer_range_for_worker(total_layers, num_workers, i)
            for i in range(num_workers)
        ]

        # 检查覆盖完整性
        covered = set()
        for start, end in ranges:
            for layer in range(start, end):
                assert layer not in covered, f"Layer {layer} covered multiple times"
                covered.add(layer)

        assert len(covered) == total_layers, "Not all layers covered"

    def test_device_map_consistency(self):
        """测试 device_map 一致性"""
        total_layers = 32

        # 创建 3 个分片的 device_map
        device_maps = []
        for i in range(3):
            start, end = get_layer_range_for_worker(total_layers, 3, i)
            dm = _create_device_map_for_layers(
                config=None,
                start_layer=start,
                end_layer=end,
                device=f"cuda:{i}",
                include_embeddings=(i == 0),
                include_lm_head=(i == 2),
            )
            device_maps.append(dm)

        # 检查所有层都被分配
        all_layers = set()
        for dm in device_maps:
            for key in dm:
                if key.startswith("model.layers."):
                    layer_idx = int(key.split(".")[-1])
                    all_layers.add(layer_idx)

        assert len(all_layers) == total_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
