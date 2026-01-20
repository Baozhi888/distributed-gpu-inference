"""
模型分片加载器

支持按 Transformer Block 范围加载模型的部分层，
用于分布式推理场景。

参考 Petals 的 from_pretrained 实现。
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """层信息"""
    layer_idx: int
    layer_name: str
    param_count: int
    memory_bytes: int


class ModelShard(nn.Module):
    """
    模型分片

    只加载模型的指定层范围，用于分布式推理
    """

    def __init__(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype

        # 模型组件（加载后填充）
        self.layers: nn.ModuleList = nn.ModuleList()
        self.config = None
        self.is_first_shard = False  # 是否包含 embedding
        self.is_last_shard = False   # 是否包含 lm_head

        # 元数据
        self.total_layers = 0
        self.hidden_size = 0
        self.num_heads = 0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        start_layer: int,
        end_layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs
    ) -> "ModelShard":
        """
        加载模型分片

        Args:
            model_id: HuggingFace 模型 ID
            start_layer: 起始层（包含）
            end_layer: 结束层（不包含）
            device: 目标设备
            dtype: 数据类型

        Returns:
            ModelShard 实例
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        logger.info(f"Loading model shard: {model_id} layers [{start_layer}, {end_layer})")

        # 加载配置
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # 获取总层数
        total_layers = getattr(config, "num_hidden_layers", None)
        if total_layers is None:
            total_layers = getattr(config, "n_layer", 32)  # 默认值

        if end_layer > total_layers:
            raise ValueError(f"end_layer ({end_layer}) > total_layers ({total_layers})")

        # 创建分片实例
        shard = cls(model_id, start_layer, end_layer, device, dtype)
        shard.config = config
        shard.total_layers = total_layers
        shard.hidden_size = getattr(config, "hidden_size", 4096)
        shard.num_heads = getattr(config, "num_attention_heads", 32)
        shard.is_first_shard = (start_layer == 0)
        shard.is_last_shard = (end_layer == total_layers)

        # 加载完整模型然后提取需要的层
        # 注意：对于非常大的模型，应该使用更高效的方法
        logger.info("Loading full model for layer extraction...")

        with torch.device("meta"):
            # 先在 meta 设备上创建模型结构
            full_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # 获取层的名称模式
        layer_module = _get_layer_module(full_model, config)

        # 使用 from_pretrained 的 device_map 功能只加载需要的层
        device_map = _create_device_map_for_layers(
            config,
            start_layer,
            end_layer,
            device,
            include_embeddings=shard.is_first_shard,
            include_lm_head=shard.is_last_shard,
        )

        logger.info(f"Loading layers with device_map: {list(device_map.keys())[:5]}...")

        full_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs
        )

        # 提取需要的层
        shard._extract_layers(full_model, config)

        # 清理不需要的组件
        del full_model
        torch.cuda.empty_cache()

        logger.info(f"Model shard loaded: {shard.get_memory_usage():.2f} GB")

        return shard

    def _extract_layers(self, full_model: nn.Module, config) -> None:
        """从完整模型中提取需要的层"""
        # 获取层模块
        layers = _get_layer_module(full_model, config)

        if layers is None:
            raise RuntimeError(f"Cannot find layers in model architecture")

        # 提取指定范围的层
        for i in range(self.start_layer, self.end_layer):
            layer = layers[i]
            self.layers.append(layer)

        # 如果是第一个分片，保存 embedding
        if self.is_first_shard:
            self.embed_tokens = _get_embedding_module(full_model, config)
            self.embed_positions = getattr(full_model.model, "embed_positions", None)

        # 如果是最后一个分片，保存 lm_head 和 norm
        if self.is_last_shard:
            self.lm_head = full_model.lm_head if hasattr(full_model, "lm_head") else None
            self.norm = _get_norm_module(full_model, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_values: 过去的 KV-Cache
            use_cache: 是否返回新的 KV-Cache

        Returns:
            (hidden_states, new_past_key_values)
        """
        # 如果是第一个分片，需要处理 embedding
        if self.is_first_shard and hasattr(self, 'embed_tokens'):
            # hidden_states 此时是 input_ids
            hidden_states = self.embed_tokens(hidden_states)
            if hasattr(self, 'embed_positions') and self.embed_positions is not None:
                hidden_states = hidden_states + self.embed_positions(position_ids)

        # 依次通过每一层
        new_past_key_values = [] if use_cache else None
        past_idx = 0

        for layer in self.layers:
            layer_past = past_key_values[past_idx] if past_key_values else None

            # 调用层的 forward
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                new_past_key_values.append(layer_outputs[1])

            past_idx += 1

        # 如果是最后一个分片，应用 norm
        if self.is_last_shard and hasattr(self, 'norm') and self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算 logits（仅最后一个分片）

        Args:
            hidden_states: 隐藏状态

        Returns:
            logits
        """
        if not self.is_last_shard:
            raise RuntimeError("get_logits() can only be called on the last shard")

        if hasattr(self, 'lm_head') and self.lm_head is not None:
            return self.lm_head(hidden_states)

        raise RuntimeError("No lm_head available")

    def get_memory_usage(self) -> float:
        """获取显存使用量（GB）"""
        total_bytes = sum(
            p.numel() * p.element_size()
            for p in self.parameters()
        )
        return total_bytes / (1024 ** 3)

    def get_layer_count(self) -> int:
        """获取层数"""
        return len(self.layers)


class ShardedModelLoader:
    """
    分片模型加载器

    管理模型分片的加载和分发
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.config = None
        self.total_layers = 0

    def analyze_model(self) -> Dict[str, Any]:
        """
        分析模型结构

        Returns:
            模型信息字典
        """
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.total_layers = getattr(self.config, "num_hidden_layers", 32)
        hidden_size = getattr(self.config, "hidden_size", 4096)
        num_heads = getattr(self.config, "num_attention_heads", 32)
        num_kv_heads = getattr(self.config, "num_key_value_heads", num_heads)
        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)

        # 估算每层内存（粗略）
        head_dim = hidden_size // num_heads
        attention_params = hidden_size * (num_heads + 2 * num_kv_heads) * head_dim
        mlp_params = hidden_size * intermediate_size * 3  # gate, up, down
        layer_params = attention_params + mlp_params
        bytes_per_param = 2  # float16

        memory_per_layer_gb = layer_params * bytes_per_param / (1024 ** 3)

        return {
            "model_id": self.model_id,
            "total_layers": self.total_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "memory_per_layer_gb": memory_per_layer_gb,
            "total_memory_gb": memory_per_layer_gb * self.total_layers,
        }

    def create_shard_plan(
        self,
        worker_memory_gb: List[float],
        reserve_ratio: float = 0.2,
    ) -> List[Tuple[int, int]]:
        """
        创建分片计划

        Args:
            worker_memory_gb: 每个 Worker 的可用显存
            reserve_ratio: 预留显存比例（用于 KV-Cache）

        Returns:
            [(start_layer, end_layer), ...] 每个 Worker 的层范围
        """
        if not self.config:
            self.analyze_model()

        model_info = self.analyze_model()
        memory_per_layer = model_info["memory_per_layer_gb"]

        # 计算每个 Worker 可以承载的层数
        available_memory = [
            mem * (1 - reserve_ratio) for mem in worker_memory_gb
        ]
        layers_per_worker = [
            int(mem / memory_per_layer) for mem in available_memory
        ]

        # 确保总层数足够
        total_capacity = sum(layers_per_worker)
        if total_capacity < self.total_layers:
            raise ValueError(
                f"Insufficient memory: can only fit {total_capacity} layers, "
                f"but model has {self.total_layers} layers"
            )

        # 分配层
        shard_plan = []
        current_layer = 0

        for i, capacity in enumerate(layers_per_worker):
            # 按比例分配
            if i == len(layers_per_worker) - 1:
                # 最后一个 Worker 获取剩余所有层
                end_layer = self.total_layers
            else:
                # 按容量比例分配
                ratio = capacity / total_capacity
                num_layers = max(1, int(self.total_layers * ratio))
                end_layer = min(current_layer + num_layers, self.total_layers)

            if current_layer < self.total_layers:
                shard_plan.append((current_layer, end_layer))
                current_layer = end_layer

        return shard_plan


def get_layer_range_for_worker(
    total_layers: int,
    num_workers: int,
    worker_idx: int,
) -> Tuple[int, int]:
    """
    计算 Worker 的层范围（均匀分配）

    Args:
        total_layers: 总层数
        num_workers: Worker 数量
        worker_idx: 当前 Worker 索引

    Returns:
        (start_layer, end_layer)
    """
    layers_per_worker = total_layers // num_workers
    remainder = total_layers % num_workers

    start = worker_idx * layers_per_worker + min(worker_idx, remainder)
    end = start + layers_per_worker + (1 if worker_idx < remainder else 0)

    return start, end


# 辅助函数

def _get_layer_module(model: nn.Module, config) -> Optional[nn.ModuleList]:
    """获取模型的层模块"""
    # 尝试不同的模型架构
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return model.model.decoder.layers

    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model.transformer, "layers"):
            return model.transformer.layers

    return None


def _get_embedding_module(model: nn.Module, config) -> Optional[nn.Module]:
    """获取 embedding 模块"""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    return None


def _get_norm_module(model: nn.Module, config) -> Optional[nn.Module]:
    """获取最终 norm 模块"""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def _create_device_map_for_layers(
    config,
    start_layer: int,
    end_layer: int,
    device: str,
    include_embeddings: bool = False,
    include_lm_head: bool = False,
) -> Dict[str, str]:
    """
    创建 device_map 用于选择性加载层
    """
    device_map = {}

    # 模型前缀（不同模型可能不同）
    model_prefix = "model"

    if include_embeddings:
        device_map[f"{model_prefix}.embed_tokens"] = device

    # 层
    for i in range(start_layer, end_layer):
        device_map[f"{model_prefix}.layers.{i}"] = device

    if include_lm_head:
        device_map[f"{model_prefix}.norm"] = device
        device_map["lm_head"] = device

    # 其他层放到 CPU 或丢弃
    # 注意：这里简化处理，实际应该更细致

    return device_map
