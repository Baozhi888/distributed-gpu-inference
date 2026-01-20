"""分布式推理组件

实现跨 Worker 的模型分片推理，参考 Petals 项目设计：
- DistributedInferenceSession: 分布式推理会话管理
- WorkerSession: Worker 级别会话
- ModelShard: 模型分片加载器
- GRPCServer: Worker 间 P2P 通信
"""
from .session import (
    DistributedInferenceSession,
    WorkerSession,
    SessionManager,
)
from .model_shard import (
    ModelShard,
    ShardedModelLoader,
    get_layer_range_for_worker,
)
from .kv_cache import (
    DistributedKVCacheManager,
    PagedKVCache,
    KVCachePool,
)

__all__ = [
    "DistributedInferenceSession",
    "WorkerSession",
    "SessionManager",
    "ModelShard",
    "ShardedModelLoader",
    "get_layer_range_for_worker",
    "DistributedKVCacheManager",
    "PagedKVCache",
    "KVCachePool",
]
