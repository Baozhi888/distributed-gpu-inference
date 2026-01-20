"""
分布式推理核心数据结构

基于 Petals 项目设计，适配中心化调度 + P2P 直连混合架构
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import time


class WorkerRole(Enum):
    """Worker 角色类型"""
    PREFILL = "prefill"      # 专门处理 Prefill 阶段（计算密集）
    DECODE = "decode"        # 专门处理 Decode 阶段（内存密集）
    HYBRID = "hybrid"        # 混合模式，可处理两种阶段


class WorkerState(Enum):
    """Worker 状态"""
    OFFLINE = 0
    JOINING = 1
    ONLINE = 2
    BUSY = 3
    ERROR = 4


@dataclass
class BlockRange:
    """模型层范围 - 用于分布式模型分割"""
    start: int  # 起始层（包含）
    end: int    # 结束层（不包含）

    @property
    def length(self) -> int:
        return self.end - self.start

    def __contains__(self, layer_idx: int) -> bool:
        return self.start <= layer_idx < self.end

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "BlockRange":
        return cls(start=data["start"], end=data["end"])


@dataclass
class WorkerInfo:
    """Worker 节点信息"""
    worker_id: str
    blocks: Optional[BlockRange] = None  # 负责的模型层范围
    role: WorkerRole = WorkerRole.HYBRID
    state: WorkerState = WorkerState.OFFLINE

    # 硬件信息
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0

    # 性能指标
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    reliability_score: float = 1.0

    # 网络信息
    peer_address: str = ""          # P2P 直连地址 (gRPC)
    api_endpoint: str = ""          # HTTP API 端点

    # 缓存信息
    cache_tokens_available: int = 0  # 可用的 KV-Cache 容量（token 数）
    cache_tokens_used: int = 0

    # 元数据
    model_id: str = ""
    supported_models: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def cache_utilization(self) -> float:
        """缓存利用率"""
        if self.cache_tokens_available == 0:
            return 0.0
        return self.cache_tokens_used / self.cache_tokens_available

    @property
    def gpu_utilization(self) -> float:
        """GPU 内存利用率"""
        if self.gpu_memory_gb == 0:
            return 0.0
        return self.gpu_memory_used_gb / self.gpu_memory_gb

    def is_healthy(self, timeout_seconds: float = 60.0) -> bool:
        """检查 Worker 是否健康"""
        if self.state in (WorkerState.OFFLINE, WorkerState.ERROR):
            return False
        return (time.time() - self.last_heartbeat) < timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "blocks": self.blocks.to_dict() if self.blocks else None,
            "role": self.role.value,
            "state": self.state.value,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "latency_ms": self.latency_ms,
            "reliability_score": self.reliability_score,
            "peer_address": self.peer_address,
            "api_endpoint": self.api_endpoint,
            "cache_tokens_available": self.cache_tokens_available,
            "cache_tokens_used": self.cache_tokens_used,
            "model_id": self.model_id,
            "supported_models": self.supported_models,
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class InferenceState:
    """推理状态 - 跨 Worker 传递"""
    session_id: str
    position: int = 0                        # 当前序列位置
    kv_cache_keys: List[str] = field(default_factory=list)  # KV-Cache 在各 Worker 的索引

    # 用于序列化传输的隐藏状态 (实际使用时为 bytes)
    hidden_states_data: Optional[bytes] = None
    hidden_states_shape: Optional[Tuple[int, ...]] = None
    hidden_states_dtype: str = "float16"

    # 元数据
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_position(self, new_tokens: int) -> None:
        """更新位置"""
        self.position += new_tokens
        self.updated_at = time.time()


@dataclass
class KVCacheBlock:
    """KV-Cache 分页块 - PagedAttention 风格"""
    block_id: str
    layer_idx: int
    block_size: int = 16              # 每块 token 数

    # 缓存数据 (实际使用时为 bytes)
    keys_data: Optional[bytes] = None
    values_data: Optional[bytes] = None

    # 形状信息
    num_heads: int = 0
    head_dim: int = 0

    # 引用计数 (Copy-on-Write)
    ref_count: int = 1

    # 前缀哈希 (用于共享)
    prefix_hash: str = ""

    # 存储位置
    location: str = "gpu"  # "gpu", "cpu", "redis", "remote"

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    def increment_ref(self) -> None:
        self.ref_count += 1

    def decrement_ref(self) -> int:
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    session_id: str

    # 输入数据
    input_data: Optional[bytes] = None     # 序列化的输入 tensor
    input_shape: Optional[Tuple[int, ...]] = None
    input_dtype: str = "float16"

    # 推理参数
    position: int = 0
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 路由信息
    kv_cache_keys: List[str] = field(default_factory=list)
    next_worker_address: str = ""      # 下一跳 Worker（用于 server-to-server）

    # 元数据
    step_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    session_id: str

    # 输出数据
    output_data: Optional[bytes] = None    # 序列化的输出 tensor
    output_shape: Optional[Tuple[int, ...]] = None
    output_dtype: str = "float16"

    # KV-Cache 更新
    updated_kv_keys: List[str] = field(default_factory=list)

    # 性能指标
    latency_ms: float = 0.0
    tokens_generated: int = 0

    # 状态
    success: bool = True
    error_message: str = ""


@dataclass
class SessionConfig:
    """推理会话配置"""
    model_name: str
    max_length: int = 4096

    # 采样参数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # 性能配置
    use_cache: bool = True
    stream: bool = False

    # 超时设置
    connect_timeout: float = 30.0
    request_timeout: float = 120.0
    max_retries: int = 3

    # 高级选项
    use_speculative_decoding: bool = False
    speculative_depth: int = 5


@dataclass
class ModelShardConfig:
    """模型分片配置"""
    model_id: str
    total_layers: int

    # 分片映射: worker_id -> BlockRange
    shard_mapping: Dict[str, BlockRange] = field(default_factory=dict)

    # 模型元数据
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0

    # 内存估算
    memory_per_layer_gb: float = 0.0
    kv_cache_per_token_bytes: int = 0

    def get_worker_for_layer(self, layer_idx: int) -> Optional[str]:
        """获取负责指定层的 Worker"""
        for worker_id, block_range in self.shard_mapping.items():
            if layer_idx in block_range:
                return worker_id
        return None

    def get_inference_route(self) -> List[Tuple[str, BlockRange]]:
        """获取推理路由顺序"""
        route = sorted(
            self.shard_mapping.items(),
            key=lambda x: x[1].start
        )
        return [(worker_id, blocks) for worker_id, blocks in route]


def compute_prefix_hash(token_ids: List[int]) -> str:
    """计算前缀哈希（用于 KV-Cache 共享）"""
    data = bytes(token_ids)
    return hashlib.sha256(data).hexdigest()[:16]


def estimate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # float16
) -> int:
    """估算 KV-Cache 大小（字节）"""
    # KV-Cache: 2 (K+V) * num_layers * batch * seq_len * num_heads * head_dim * dtype_bytes
    return 2 * num_layers * batch_size * seq_length * num_heads * head_dim * dtype_bytes
