"""
推测解码引擎

实现 EAGLE-3 风格的推测解码，提升单请求解码速度 2-3x

核心思想：
1. 使用轻量级 Draft 模型预测多个候选 token
2. 目标模型并行验证候选序列
3. 接受最长的正确前缀

参考：
- EAGLE-3: https://arxiv.org/abs/2503.01840
- Medusa: https://arxiv.org/abs/2401.10774
- SpecInfer: https://arxiv.org/abs/2305.09781
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """推测解码配置"""
    # Draft 模型配置
    draft_model_id: Optional[str] = None  # 独立的 draft 模型
    use_self_draft: bool = True           # 使用自身作为 draft（EAGLE 风格）
    draft_head_hidden_size: int = 1024    # Draft head 隐藏层大小

    # 推测参数
    num_speculative_tokens: int = 5       # 每步推测的 token 数
    tree_width: int = 3                   # 树宽度（每个位置的候选数）
    tree_depth: int = 5                   # 树深度

    # 验证参数
    temperature: float = 0.0              # 验证时的温度
    top_p: float = 1.0                    # 验证时的 top_p

    # 性能参数
    min_accept_rate: float = 0.3          # 最小接受率阈值
    adaptive_depth: bool = True           # 自适应调整推测深度


@dataclass
class SpeculativeOutput:
    """推测解码输出"""
    tokens: List[int]                     # 接受的 tokens
    accept_rate: float                    # 接受率
    draft_tokens: int                     # 生成的 draft tokens
    accepted_tokens: int                  # 接受的 tokens
    latency_ms: float                     # 延迟


class DraftHead(nn.Module):
    """
    EAGLE 风格的 Draft Head

    在 feature level 进行自回归预测，而不是 token level
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 2,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Feature 预测网络
        layers = []
        input_dim = hidden_size * 2  # 当前 hidden + token embedding
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else hidden_size
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.SiLU() if i < num_layers - 1 else nn.Identity(),
            ])
            input_dim = output_dim

        self.feature_predictor = nn.Sequential(*layers)

        # Token embedding（共享目标模型的 embedding）
        self.token_embedding = None

    def set_token_embedding(self, embedding: nn.Embedding) -> None:
        """设置 token embedding（通常共享目标模型的 embedding）"""
        self.token_embedding = embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测下一步的 hidden states

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len]

        Returns:
            predicted_hidden: [batch, seq_len, hidden_size]
        """
        if self.token_embedding is None:
            raise RuntimeError("Token embedding not set")

        # 获取 token embeddings
        token_embeds = self.token_embedding(token_ids)  # [batch, seq_len, hidden_size]

        # 拼接 hidden states 和 token embeddings
        combined = torch.cat([hidden_states, token_embeds], dim=-1)

        # 预测下一步的 hidden states
        predicted_hidden = self.feature_predictor(combined)

        return predicted_hidden


class TreeDraftBuffer:
    """
    Token 树缓冲区

    管理推测解码中的候选 token 树
    """

    def __init__(
        self,
        tree_width: int = 3,
        tree_depth: int = 5,
        device: str = "cuda",
    ):
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.device = device

        # 树节点：(token_id, log_prob, parent_idx)
        self.nodes: List[Tuple[int, float, int]] = []

        # 层级索引
        self.layer_offsets: List[int] = []

    def reset(self) -> None:
        """重置缓冲区"""
        self.nodes.clear()
        self.layer_offsets.clear()

    def add_candidates(
        self,
        token_ids: torch.Tensor,
        log_probs: torch.Tensor,
        parent_indices: torch.Tensor,
    ) -> None:
        """
        添加候选 tokens

        Args:
            token_ids: [num_candidates]
            log_probs: [num_candidates]
            parent_indices: [num_candidates] 每个候选的父节点索引
        """
        self.layer_offsets.append(len(self.nodes))

        for tid, lp, pid in zip(
            token_ids.cpu().tolist(),
            log_probs.cpu().tolist(),
            parent_indices.cpu().tolist()
        ):
            self.nodes.append((tid, lp, pid))

    def get_tree_tokens(self) -> torch.Tensor:
        """获取树中所有 tokens（用于验证）"""
        tokens = [node[0] for node in self.nodes]
        return torch.tensor(tokens, device=self.device)

    def get_tree_attention_mask(self, seq_len: int) -> torch.Tensor:
        """
        生成树形 attention mask

        Returns:
            mask: [num_nodes, seq_len + num_nodes]
        """
        num_nodes = len(self.nodes)
        total_len = seq_len + num_nodes

        # 初始化 mask
        mask = torch.zeros(num_nodes, total_len, device=self.device)

        # 每个节点可以看到：
        # 1. 所有前缀 tokens
        # 2. 自己的祖先节点
        for i, (_, _, parent_idx) in enumerate(self.nodes):
            # 可以看到所有前缀
            mask[i, :seq_len] = 1

            # 可以看到自己和祖先
            current = i
            while current >= 0:
                mask[i, seq_len + current] = 1
                if current < len(self.nodes):
                    current = self.nodes[current][2]
                else:
                    break

        return mask

    def trace_accepted_path(
        self,
        accepted_mask: torch.Tensor
    ) -> List[int]:
        """
        追踪被接受的路径

        Args:
            accepted_mask: [num_nodes] 布尔掩码

        Returns:
            accepted_tokens: 被接受的 token 列表
        """
        accepted = accepted_mask.cpu().tolist()

        # 找到最长的被接受路径
        best_path = []

        for i in range(len(self.nodes) - 1, -1, -1):
            if accepted[i]:
                path = []
                current = i
                while current >= 0 and accepted[current]:
                    path.append(self.nodes[current][0])
                    current = self.nodes[current][2]
                path.reverse()

                if len(path) > len(best_path):
                    best_path = path

        return best_path


class SpeculativeDecoder:
    """
    推测解码器

    实现完整的推测解码流程
    """

    def __init__(
        self,
        target_model: nn.Module,
        config: SpeculativeConfig,
        device: str = "cuda",
    ):
        self.target = target_model
        self.config = config
        self.device = device

        # Draft 组件
        self.draft_head: Optional[DraftHead] = None
        self.draft_model: Optional[nn.Module] = None

        # 树缓冲区
        self.tree_buffer = TreeDraftBuffer(
            tree_width=config.tree_width,
            tree_depth=config.tree_depth,
            device=device,
        )

        # 统计
        self._stats = {
            "total_steps": 0,
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "avg_accept_rate": 0.0,
        }

        # 自适应深度
        self._current_depth = config.tree_depth

    def setup_draft_head(
        self,
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        """设置 Draft Head"""
        self.draft_head = DraftHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=self.config.draft_head_hidden_size,
        ).to(self.device)

        # 共享 embedding
        if hasattr(self.target, "model") and hasattr(self.target.model, "embed_tokens"):
            self.draft_head.set_token_embedding(self.target.model.embed_tokens)
        elif hasattr(self.target, "transformer") and hasattr(self.target.transformer, "wte"):
            self.draft_head.set_token_embedding(self.target.transformer.wte)

    async def decode_step(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], int]:
        """
        执行一步推测解码

        Args:
            hidden_states: 当前隐藏状态
            past_key_values: KV-Cache
            input_ids: 当前输入 tokens

        Returns:
            (accepted_tokens, num_accepted)
        """
        start_time = time.time()

        # 1. Draft 阶段：生成候选 token 树
        self.tree_buffer.reset()
        draft_tokens = await self._generate_draft_tree(
            hidden_states,
            input_ids,
            past_key_values,
        )

        # 2. Verify 阶段：目标模型验证
        accepted_mask = await self._verify_candidates(
            draft_tokens,
            hidden_states,
            past_key_values,
        )

        # 3. Accept 阶段：确定接受的 tokens
        accepted_tokens = self.tree_buffer.trace_accepted_path(accepted_mask)

        # 更新统计
        num_draft = len(self.tree_buffer.nodes)
        num_accepted = len(accepted_tokens)
        accept_rate = num_accepted / max(1, num_draft)

        self._stats["total_steps"] += 1
        self._stats["total_draft_tokens"] += num_draft
        self._stats["total_accepted_tokens"] += num_accepted
        self._stats["avg_accept_rate"] = (
            self._stats["total_accepted_tokens"] /
            max(1, self._stats["total_draft_tokens"])
        )

        # 自适应调整深度
        if self.config.adaptive_depth:
            self._adapt_depth(accept_rate)

        latency_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Speculative step: {num_accepted}/{num_draft} accepted "
            f"({accept_rate:.1%}) in {latency_ms:.1f}ms"
        )

        return accepted_tokens, num_accepted

    async def _generate_draft_tree(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """生成 draft token 树"""
        if self.draft_head is None:
            raise RuntimeError("Draft head not initialized")

        current_hidden = hidden_states
        current_tokens = input_ids[:, -1:]

        for depth in range(self._current_depth):
            # 预测下一步的 hidden states
            predicted_hidden = self.draft_head(current_hidden, current_tokens)

            # 使用目标模型的 LM head 获取 logits
            if hasattr(self.target, "lm_head"):
                logits = self.target.lm_head(predicted_hidden)
            else:
                logits = predicted_hidden @ self.target.model.embed_tokens.weight.T

            # 获取 top-k candidates
            log_probs = torch.log_softmax(logits[:, -1], dim=-1)
            top_log_probs, top_indices = torch.topk(
                log_probs,
                k=min(self.config.tree_width, logits.size(-1)),
                dim=-1
            )

            # 添加到树
            parent_indices = torch.zeros(top_indices.size(-1), device=self.device, dtype=torch.long)
            if depth > 0:
                # 连接到上一层的最佳节点
                parent_indices = torch.full_like(
                    top_indices.squeeze(),
                    len(self.tree_buffer.nodes) - 1
                )

            self.tree_buffer.add_candidates(
                top_indices.squeeze(),
                top_log_probs.squeeze(),
                parent_indices,
            )

            # 更新状态（使用最佳候选）
            current_tokens = top_indices[:, :1]
            current_hidden = predicted_hidden

        return self.tree_buffer.get_tree_tokens()

    async def _verify_candidates(
        self,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """验证候选 tokens"""
        # 获取树形 attention mask
        seq_len = hidden_states.size(1)
        tree_mask = self.tree_buffer.get_tree_attention_mask(seq_len)

        # 准备输入
        verify_input = draft_tokens.unsqueeze(0)

        # 目标模型前向传播
        with torch.no_grad():
            outputs = self.target(
                input_ids=verify_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # 获取 logits 并验证
        logits = outputs.logits  # [1, num_tokens, vocab_size]

        # 采样验证
        if self.config.temperature > 0:
            probs = torch.softmax(logits / self.config.temperature, dim=-1)
            sampled_tokens = torch.multinomial(probs.squeeze(), num_samples=1).squeeze()
        else:
            sampled_tokens = logits.argmax(dim=-1).squeeze()

        # 检查匹配
        accepted_mask = (sampled_tokens == draft_tokens)

        return accepted_mask

    def _adapt_depth(self, accept_rate: float) -> None:
        """自适应调整推测深度"""
        if accept_rate < self.config.min_accept_rate:
            # 接受率过低，减少深度
            self._current_depth = max(1, self._current_depth - 1)
        elif accept_rate > 0.7 and self._current_depth < self.config.tree_depth:
            # 接受率高，增加深度
            self._current_depth = min(self.config.tree_depth, self._current_depth + 1)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "current_depth": self._current_depth,
            "speedup_estimate": max(1.0, self._stats["avg_accept_rate"] * self._current_depth),
        }


class MedusaHead(nn.Module):
    """
    Medusa 风格的多头预测

    使用多个独立的预测头，每个头预测不同位置的 token
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size),
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        预测多个位置的 logits

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            logits_list: List of [batch, seq_len, vocab_size]
        """
        return [head(hidden_states) for head in self.heads]
