"""
测试 worker/engines/speculative.py 模块

覆盖：
- SpeculativeConfig / SpeculativeOutput 数据类
- DraftHead 神经网络
- TreeDraftBuffer 树缓冲区
- SpeculativeDecoder 推测解码器
- MedusaHead 多头预测
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.engines.speculative import (
    SpeculativeConfig,
    SpeculativeOutput,
    DraftHead,
    TreeDraftBuffer,
    SpeculativeDecoder,
    MedusaHead,
)


# ============== SpeculativeConfig 测试 ==============

class TestSpeculativeConfig:
    """SpeculativeConfig 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        config = SpeculativeConfig()

        assert config.draft_model_id is None
        assert config.use_self_draft is True
        assert config.draft_head_hidden_size == 1024
        assert config.num_speculative_tokens == 5
        assert config.tree_width == 3
        assert config.tree_depth == 5
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.min_accept_rate == 0.3
        assert config.adaptive_depth is True

    def test_custom_values(self):
        """测试自定义值"""
        config = SpeculativeConfig(
            draft_model_id="small-model",
            use_self_draft=False,
            draft_head_hidden_size=512,
            num_speculative_tokens=10,
            tree_width=5,
            tree_depth=8,
            temperature=0.5,
            min_accept_rate=0.5,
            adaptive_depth=False,
        )

        assert config.draft_model_id == "small-model"
        assert config.use_self_draft is False
        assert config.draft_head_hidden_size == 512
        assert config.num_speculative_tokens == 10
        assert config.tree_width == 5
        assert config.tree_depth == 8
        assert config.adaptive_depth is False


# ============== SpeculativeOutput 测试 ==============

class TestSpeculativeOutput:
    """SpeculativeOutput 数据类测试"""

    def test_creation(self):
        """测试创建输出"""
        output = SpeculativeOutput(
            tokens=[1, 2, 3, 4],
            accept_rate=0.8,
            draft_tokens=5,
            accepted_tokens=4,
            latency_ms=10.5,
        )

        assert output.tokens == [1, 2, 3, 4]
        assert output.accept_rate == 0.8
        assert output.draft_tokens == 5
        assert output.accepted_tokens == 4
        assert output.latency_ms == 10.5


# ============== DraftHead 测试 ==============

class TestDraftHead:
    """DraftHead 神经网络测试"""

    @pytest.fixture
    def draft_head(self):
        """创建 DraftHead 实例"""
        return DraftHead(
            hidden_size=256,
            vocab_size=1000,
            num_layers=2,
            hidden_dim=128,
        )

    def test_init(self, draft_head):
        """测试初始化"""
        assert draft_head.hidden_size == 256
        assert draft_head.vocab_size == 1000
        assert draft_head.token_embedding is None
        assert draft_head.feature_predictor is not None

    def test_set_token_embedding(self, draft_head):
        """测试设置 token embedding"""
        mock_embedding = nn.Embedding(1000, 256)
        draft_head.set_token_embedding(mock_embedding)

        assert draft_head.token_embedding is mock_embedding

    def test_forward_without_embedding_raises(self, draft_head):
        """测试没有 embedding 时 forward 抛出异常"""
        hidden_states = torch.randn(1, 10, 256)
        token_ids = torch.randint(0, 1000, (1, 10))

        with pytest.raises(RuntimeError, match="Token embedding not set"):
            draft_head(hidden_states, token_ids)

    def test_forward_with_embedding(self, draft_head):
        """测试正常 forward"""
        mock_embedding = nn.Embedding(1000, 256)
        draft_head.set_token_embedding(mock_embedding)

        batch_size, seq_len, hidden_size = 2, 5, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = draft_head(hidden_states, token_ids)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_forward_different_seq_lengths(self, draft_head):
        """测试不同序列长度"""
        mock_embedding = nn.Embedding(1000, 256)
        draft_head.set_token_embedding(mock_embedding)

        for seq_len in [1, 5, 10, 100]:
            hidden_states = torch.randn(1, seq_len, 256)
            token_ids = torch.randint(0, 1000, (1, seq_len))

            output = draft_head(hidden_states, token_ids)

            assert output.shape == (1, seq_len, 256)


# ============== TreeDraftBuffer 测试 ==============

class TestTreeDraftBuffer:
    """TreeDraftBuffer 测试"""

    @pytest.fixture
    def buffer(self):
        """创建缓冲区实例"""
        return TreeDraftBuffer(
            tree_width=3,
            tree_depth=5,
            device="cpu",
        )

    def test_init(self, buffer):
        """测试初始化"""
        assert buffer.tree_width == 3
        assert buffer.tree_depth == 5
        assert buffer.device == "cpu"
        assert len(buffer.nodes) == 0
        assert len(buffer.layer_offsets) == 0

    def test_reset(self, buffer):
        """测试重置"""
        # 添加一些数据
        buffer.nodes.append((1, 0.5, -1))
        buffer.layer_offsets.append(0)

        buffer.reset()

        assert len(buffer.nodes) == 0
        assert len(buffer.layer_offsets) == 0

    def test_add_candidates(self, buffer):
        """测试添加候选"""
        token_ids = torch.tensor([1, 2, 3])
        log_probs = torch.tensor([-0.1, -0.2, -0.3])
        parent_indices = torch.tensor([0, 0, 0])

        buffer.add_candidates(token_ids, log_probs, parent_indices)

        assert len(buffer.nodes) == 3
        assert buffer.nodes[0] == (1, pytest.approx(-0.1, abs=1e-5), 0)
        assert buffer.nodes[1] == (2, pytest.approx(-0.2, abs=1e-5), 0)
        assert buffer.nodes[2] == (3, pytest.approx(-0.3, abs=1e-5), 0)
        assert len(buffer.layer_offsets) == 1

    def test_add_candidates_multiple_layers(self, buffer):
        """测试添加多层候选"""
        # 第一层
        buffer.add_candidates(
            torch.tensor([1, 2]),
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-1, -1]),
        )

        # 第二层
        buffer.add_candidates(
            torch.tensor([3, 4]),
            torch.tensor([-0.3, -0.4]),
            torch.tensor([0, 1]),
        )

        assert len(buffer.nodes) == 4
        assert len(buffer.layer_offsets) == 2
        assert buffer.layer_offsets == [0, 2]

    def test_get_tree_tokens(self, buffer):
        """测试获取树中所有 tokens"""
        buffer.add_candidates(
            torch.tensor([10, 20, 30]),
            torch.tensor([-0.1, -0.2, -0.3]),
            torch.tensor([0, 0, 0]),
        )

        tokens = buffer.get_tree_tokens()

        assert tokens.tolist() == [10, 20, 30]

    def test_get_tree_attention_mask(self, buffer):
        """测试生成树形 attention mask"""
        # 添加节点
        buffer.add_candidates(
            torch.tensor([1, 2]),
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-1, -1]),  # 根节点
        )

        seq_len = 5
        mask = buffer.get_tree_attention_mask(seq_len)

        assert mask.shape == (2, seq_len + 2)
        # 每个节点都能看到前缀
        assert mask[:, :seq_len].sum() == 2 * seq_len

    def test_trace_accepted_path_all_accepted(self, buffer):
        """测试追踪全部接受的路径"""
        buffer.nodes = [
            (1, -0.1, -1),  # 根
            (2, -0.2, 0),   # 第一层，父节点 0
            (3, -0.3, 1),   # 第二层，父节点 1
        ]

        accepted_mask = torch.tensor([True, True, True])
        path = buffer.trace_accepted_path(accepted_mask)

        assert path == [1, 2, 3]

    def test_trace_accepted_path_partial(self, buffer):
        """测试追踪部分接受的路径"""
        buffer.nodes = [
            (1, -0.1, -1),
            (2, -0.2, 0),
            (3, -0.3, 1),
            (4, -0.4, 2),
        ]

        # 只有前两个被接受
        accepted_mask = torch.tensor([True, True, False, False])
        path = buffer.trace_accepted_path(accepted_mask)

        assert path == [1, 2]

    def test_trace_accepted_path_none_accepted(self, buffer):
        """测试追踪无接受的路径"""
        buffer.nodes = [
            (1, -0.1, -1),
            (2, -0.2, 0),
        ]

        accepted_mask = torch.tensor([False, False])
        path = buffer.trace_accepted_path(accepted_mask)

        assert path == []


# ============== SpeculativeDecoder 测试 ==============

class TestSpeculativeDecoder:
    """SpeculativeDecoder 测试"""

    @pytest.fixture
    def mock_target_model(self):
        """创建模拟的目标模型"""
        model = MagicMock()
        model.lm_head = nn.Linear(256, 1000)
        model.model = MagicMock()
        model.model.embed_tokens = nn.Embedding(1000, 256)
        return model

    @pytest.fixture
    def config(self):
        """创建配置"""
        return SpeculativeConfig(
            tree_width=3,
            tree_depth=3,
            adaptive_depth=True,
        )

    @pytest.fixture
    def decoder(self, mock_target_model, config):
        """创建解码器"""
        return SpeculativeDecoder(
            target_model=mock_target_model,
            config=config,
            device="cpu",
        )

    def test_init(self, decoder, config):
        """测试初始化"""
        assert decoder.config == config
        assert decoder.draft_head is None
        assert decoder.draft_model is None
        assert decoder._current_depth == config.tree_depth

    def test_setup_draft_head(self, decoder):
        """测试设置 draft head"""
        decoder.setup_draft_head(hidden_size=256, vocab_size=1000)

        assert decoder.draft_head is not None
        assert isinstance(decoder.draft_head, DraftHead)
        assert decoder.draft_head.hidden_size == 256

    def test_adapt_depth_decrease(self, decoder, config):
        """测试自适应减少深度"""
        decoder._current_depth = 5

        # 低接受率应该减少深度
        decoder._adapt_depth(accept_rate=0.1)

        assert decoder._current_depth == 4

    def test_adapt_depth_increase(self, decoder, config):
        """测试自适应增加深度"""
        decoder._current_depth = 2
        decoder.config.tree_depth = 5

        # 高接受率应该增加深度
        decoder._adapt_depth(accept_rate=0.8)

        assert decoder._current_depth == 3

    def test_adapt_depth_min_bound(self, decoder):
        """测试深度最小边界"""
        decoder._current_depth = 1

        # 应该不会低于 1
        decoder._adapt_depth(accept_rate=0.05)

        assert decoder._current_depth == 1

    def test_adapt_depth_max_bound(self, decoder, config):
        """测试深度最大边界"""
        decoder._current_depth = config.tree_depth

        # 应该不会超过配置的最大深度
        decoder._adapt_depth(accept_rate=0.95)

        assert decoder._current_depth == config.tree_depth

    def test_get_stats_initial(self, decoder):
        """测试获取初始统计"""
        stats = decoder.get_stats()

        assert stats["total_steps"] == 0
        assert stats["total_draft_tokens"] == 0
        assert stats["total_accepted_tokens"] == 0
        assert stats["avg_accept_rate"] == 0.0
        assert stats["speedup_estimate"] >= 1.0

    def test_get_stats_after_updates(self, decoder):
        """测试更新后的统计"""
        decoder._stats["total_steps"] = 10
        decoder._stats["total_draft_tokens"] = 50
        decoder._stats["total_accepted_tokens"] = 35
        decoder._stats["avg_accept_rate"] = 0.7
        decoder._current_depth = 4

        stats = decoder.get_stats()

        assert stats["total_steps"] == 10
        assert stats["current_depth"] == 4
        assert stats["speedup_estimate"] == pytest.approx(0.7 * 4, abs=0.01)


# ============== MedusaHead 测试 ==============

class TestMedusaHead:
    """MedusaHead 多头预测测试"""

    @pytest.fixture
    def medusa_head(self):
        """创建 MedusaHead 实例"""
        return MedusaHead(
            hidden_size=256,
            vocab_size=1000,
            num_heads=4,
            hidden_dim=128,
        )

    def test_init(self, medusa_head):
        """测试初始化"""
        assert medusa_head.num_heads == 4
        assert len(medusa_head.heads) == 4

    def test_forward_output_count(self, medusa_head):
        """测试 forward 输出数量"""
        hidden_states = torch.randn(2, 10, 256)

        outputs = medusa_head(hidden_states)

        assert len(outputs) == 4  # num_heads

    def test_forward_output_shapes(self, medusa_head):
        """测试 forward 输出形状"""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 256)

        outputs = medusa_head(hidden_states)

        for output in outputs:
            assert output.shape == (batch_size, seq_len, 1000)  # vocab_size

    def test_forward_different_batch_sizes(self, medusa_head):
        """测试不同批大小"""
        for batch_size in [1, 4, 8]:
            hidden_states = torch.randn(batch_size, 5, 256)
            outputs = medusa_head(hidden_states)

            assert all(out.shape[0] == batch_size for out in outputs)

    def test_forward_different_num_heads(self):
        """测试不同头数"""
        for num_heads in [1, 2, 8]:
            head = MedusaHead(
                hidden_size=256,
                vocab_size=1000,
                num_heads=num_heads,
                hidden_dim=64,
            )

            hidden_states = torch.randn(1, 5, 256)
            outputs = head(hidden_states)

            assert len(outputs) == num_heads


# ============== 集成测试 ==============

class TestSpeculativeIntegration:
    """推测解码集成测试"""

    def test_draft_head_with_tree_buffer(self):
        """测试 DraftHead 与 TreeDraftBuffer 的集成"""
        # 创建组件
        hidden_size = 256
        vocab_size = 1000

        draft_head = DraftHead(hidden_size, vocab_size)
        embedding = nn.Embedding(vocab_size, hidden_size)
        draft_head.set_token_embedding(embedding)

        buffer = TreeDraftBuffer(tree_width=3, tree_depth=3, device="cpu")

        # 模拟推测解码流程
        batch_size = 1
        seq_len = 5

        # 初始状态
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 生成 draft
        predicted_hidden = draft_head(hidden_states, token_ids)
        assert predicted_hidden.shape == hidden_states.shape

        # 模拟 top-k 选择
        logits = torch.randn(batch_size, vocab_size)
        top_log_probs, top_indices = torch.topk(logits, k=3)

        # 添加到树
        buffer.add_candidates(
            top_indices.squeeze(),
            top_log_probs.squeeze(),
            torch.zeros(3, dtype=torch.long),
        )

        # 验证树结构
        assert len(buffer.nodes) == 3
        tokens = buffer.get_tree_tokens()
        assert tokens.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
