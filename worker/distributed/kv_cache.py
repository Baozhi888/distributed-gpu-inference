"""
分布式 KV-Cache 管理器

实现 PagedAttention 风格的 KV-Cache 管理，支持：
- GPU 内存分页管理
- 多级缓存（GPU → CPU → Redis）
- 前缀共享（RadixAttention 思想）
- 跨 Worker KV-Cache 传输

参考：vLLM PagedAttention, LMCache, Mooncake
"""
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class CacheLocation(Enum):
    """缓存位置"""
    GPU = "gpu"
    CPU = "cpu"
    REDIS = "redis"
    REMOTE = "remote"


@dataclass
class CacheBlock:
    """
    KV-Cache 分页块

    采用 PagedAttention 的设计，将 KV-Cache 分成固定大小的块
    """
    block_id: str
    block_size: int = 16  # 每块 token 数

    # 缓存数据
    keys: Optional[torch.Tensor] = None      # [num_heads, block_size, head_dim]
    values: Optional[torch.Tensor] = None    # [num_heads, block_size, head_dim]

    # 元数据
    layer_idx: int = 0
    num_tokens: int = 0  # 实际使用的 token 数
    ref_count: int = 1   # 引用计数（Copy-on-Write）
    prefix_hash: str = ""
    location: CacheLocation = CacheLocation.GPU

    # 时间戳（用于 LRU 淘汰）
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1

    def add_ref(self) -> None:
        self.ref_count += 1

    def remove_ref(self) -> int:
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count

    def touch(self) -> None:
        """更新访问时间"""
        self.last_access = time.time()


class PagedKVCache:
    """
    分页 KV-Cache 管理器

    实现 GPU 内存的分页管理，支持：
    - 动态分配和释放
    - 引用计数（Copy-on-Write）
    - LRU 淘汰策略
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        self.dtype = dtype

        # 块存储
        self._blocks: Dict[str, CacheBlock] = {}
        self._free_blocks: List[str] = []

        # 预分配内存池
        self._key_pool: Optional[torch.Tensor] = None
        self._value_pool: Optional[torch.Tensor] = None
        self._block_to_slot: Dict[str, int] = {}

        # LRU 队列
        self._lru_queue: OrderedDict[str, float] = OrderedDict()

        # 统计
        self._stats = {
            "allocations": 0,
            "evictions": 0,
            "hits": 0,
            "misses": 0,
        }

        # 初始化内存池
        self._init_memory_pool()

    def _init_memory_pool(self) -> None:
        """预分配内存池"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            # [max_blocks, num_heads, block_size, head_dim]
            pool_shape = (self.max_blocks, self.num_heads, self.block_size, self.head_dim)
            self._key_pool = torch.zeros(pool_shape, dtype=self.dtype, device=self.device)
            self._value_pool = torch.zeros(pool_shape, dtype=self.dtype, device=self.device)

            # 初始化空闲块列表
            self._free_blocks = [f"block_{i}" for i in range(self.max_blocks)]

            logger.info(
                f"Initialized KV-Cache pool: {self.max_blocks} blocks, "
                f"{self._get_pool_memory_gb():.2f} GB"
            )

    def _get_pool_memory_gb(self) -> float:
        """获取内存池大小（GB）"""
        if self._key_pool is not None:
            bytes_per_tensor = self._key_pool.numel() * self._key_pool.element_size()
            return 2 * bytes_per_tensor / (1024 ** 3)  # keys + values
        return 0.0

    def allocate_block(
        self,
        layer_idx: int,
        prefix_hash: str = "",
    ) -> Optional[CacheBlock]:
        """
        分配一个新块

        Args:
            layer_idx: 层索引
            prefix_hash: 前缀哈希（用于共享）

        Returns:
            CacheBlock 或 None（如果无可用块）
        """
        # 检查是否有空闲块
        if not self._free_blocks:
            # 尝试淘汰
            if not self._evict_lru():
                logger.warning("No free blocks and eviction failed")
                return None

        block_id = self._free_blocks.pop()
        slot_idx = int(block_id.split("_")[1])

        # 创建块
        block = CacheBlock(
            block_id=block_id,
            block_size=self.block_size,
            layer_idx=layer_idx,
            prefix_hash=prefix_hash,
            keys=self._key_pool[slot_idx] if self._key_pool is not None else None,
            values=self._value_pool[slot_idx] if self._value_pool is not None else None,
        )

        self._blocks[block_id] = block
        self._block_to_slot[block_id] = slot_idx
        self._lru_queue[block_id] = time.time()

        self._stats["allocations"] += 1

        return block

    def free_block(self, block_id: str) -> None:
        """释放块"""
        if block_id not in self._blocks:
            return

        block = self._blocks[block_id]
        block.remove_ref()

        if block.ref_count == 0:
            # 清除数据
            slot_idx = self._block_to_slot.get(block_id)
            if slot_idx is not None and self._key_pool is not None:
                self._key_pool[slot_idx].zero_()
                self._value_pool[slot_idx].zero_()

            del self._blocks[block_id]
            del self._block_to_slot[block_id]
            if block_id in self._lru_queue:
                del self._lru_queue[block_id]

            self._free_blocks.append(block_id)

    def get_block(self, block_id: str) -> Optional[CacheBlock]:
        """获取块"""
        block = self._blocks.get(block_id)
        if block:
            block.touch()
            self._lru_queue.move_to_end(block_id)
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return block

    def _evict_lru(self) -> bool:
        """淘汰最久未使用的块"""
        # 找到可淘汰的块（ref_count == 1）
        for block_id in self._lru_queue:
            block = self._blocks.get(block_id)
            if block and block.ref_count == 1:
                self.free_block(block_id)
                self._stats["evictions"] += 1
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "total_blocks": len(self._blocks),
            "free_blocks": len(self._free_blocks),
            "memory_gb": self._get_pool_memory_gb(),
        }


class KVCachePool:
    """
    多层 KV-Cache 池

    管理模型所有层的 KV-Cache
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks_per_layer: int = 100,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 每层一个 PagedKVCache
        self._layer_caches: List[PagedKVCache] = [
            PagedKVCache(
                num_layers=1,  # 每个缓存只管理一层
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                max_blocks=max_blocks_per_layer,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def allocate_sequence(
        self,
        seq_len: int,
        prefix_hash: str = "",
    ) -> List[List[CacheBlock]]:
        """
        为序列分配 KV-Cache

        Args:
            seq_len: 序列长度
            prefix_hash: 前缀哈希

        Returns:
            [[layer0_blocks], [layer1_blocks], ...]
        """
        block_size = self._layer_caches[0].block_size
        num_blocks = (seq_len + block_size - 1) // block_size

        all_blocks = []
        for layer_idx, cache in enumerate(self._layer_caches):
            layer_blocks = []
            for _ in range(num_blocks):
                block = cache.allocate_block(layer_idx, prefix_hash)
                if block is None:
                    # 回滚已分配的块
                    self._free_sequence_blocks(all_blocks)
                    raise RuntimeError(f"Failed to allocate KV-Cache for layer {layer_idx}")
                layer_blocks.append(block)
            all_blocks.append(layer_blocks)

        return all_blocks

    def _free_sequence_blocks(self, blocks: List[List[CacheBlock]]) -> None:
        """释放序列的所有块"""
        for layer_idx, layer_blocks in enumerate(blocks):
            for block in layer_blocks:
                self._layer_caches[layer_idx].free_block(block.block_id)

    def get_total_memory_gb(self) -> float:
        """获取总内存使用"""
        return sum(cache._get_pool_memory_gb() for cache in self._layer_caches)


class DistributedKVCacheManager:
    """
    分布式 KV-Cache 管理器

    实现多级缓存架构：
    L1: GPU HBM（最热数据，<1ms）
    L2: CPU RAM（温数据，~5ms）
    L3: Redis（冷数据，~10ms）
    L4: 远程 Worker（共享前缀，~50ms）
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        gpu_cache_blocks: int = 500,
        cpu_cache_gb: float = 16.0,
        redis_client = None,
        block_size: int = 16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size

        # L1: GPU 缓存
        self.gpu_cache = KVCachePool(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_blocks_per_layer=gpu_cache_blocks,
            device=device,
        )

        # L2: CPU 缓存（简单 LRU）
        self.cpu_cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.cpu_cache_max_items = int(cpu_cache_gb * 1024 * 1024 * 1024 / (
            2 * num_heads * block_size * head_dim * 2  # float16
        ))

        # L3: Redis 缓存
        self.redis = redis_client

        # 前缀索引
        self._prefix_index: Dict[str, str] = {}  # prefix_hash -> block_id

        # 统计
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l4_hits": 0,
            "misses": 0,
        }

    def compute_prefix_hash(self, token_ids: List[int]) -> str:
        """计算前缀哈希"""
        data = bytes(token_ids)
        return hashlib.sha256(data).hexdigest()[:16]

    async def get_or_compute(
        self,
        prefix_hash: str,
        layer_idx: int,
        compute_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取或计算 KV-Cache

        Args:
            prefix_hash: 前缀哈希
            layer_idx: 层索引
            compute_fn: 计算函数（如果缓存未命中）

        Returns:
            (keys, values)
        """
        cache_key = f"{prefix_hash}:{layer_idx}"

        # L1: GPU 缓存
        block = self.gpu_cache._layer_caches[layer_idx].get_block(
            self._prefix_index.get(cache_key)
        )
        if block and block.keys is not None:
            self._stats["l1_hits"] += 1
            return block.keys, block.values

        # L2: CPU 缓存
        if cache_key in self.cpu_cache:
            keys, values = self.cpu_cache[cache_key]
            self.cpu_cache.move_to_end(cache_key)
            self._stats["l2_hits"] += 1

            # 提升到 L1
            await self._promote_to_gpu(cache_key, keys, values, layer_idx, prefix_hash)
            return keys.to(self.gpu_cache._layer_caches[0].device), values.to(self.gpu_cache._layer_caches[0].device)

        # L3: Redis 缓存
        if self.redis:
            kv_bytes = await self._get_from_redis(cache_key)
            if kv_bytes:
                keys, values = self._deserialize_kv(kv_bytes)
                self._stats["l3_hits"] += 1

                # 提升到 L2 和 L1
                self._add_to_cpu_cache(cache_key, keys, values)
                await self._promote_to_gpu(cache_key, keys, values, layer_idx, prefix_hash)
                return keys, values

        # 缓存未命中，计算
        self._stats["misses"] += 1
        keys, values = await compute_fn()

        # 存储到各级缓存
        await self._store_kv(cache_key, keys, values, layer_idx, prefix_hash)

        return keys, values

    async def _promote_to_gpu(
        self,
        cache_key: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        prefix_hash: str,
    ) -> None:
        """提升到 GPU 缓存"""
        block = self.gpu_cache._layer_caches[layer_idx].allocate_block(
            layer_idx, prefix_hash
        )
        if block and block.keys is not None:
            block.keys.copy_(keys)
            block.values.copy_(values)
            self._prefix_index[cache_key] = block.block_id

    def _add_to_cpu_cache(
        self,
        cache_key: str,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """添加到 CPU 缓存"""
        # LRU 淘汰
        while len(self.cpu_cache) >= self.cpu_cache_max_items:
            self.cpu_cache.popitem(last=False)

        self.cpu_cache[cache_key] = (keys.cpu(), values.cpu())

    async def _get_from_redis(self, cache_key: str) -> Optional[bytes]:
        """从 Redis 获取"""
        if self.redis is None:
            return None
        try:
            return await self.redis.get(f"kv:{cache_key}")
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    async def _store_kv(
        self,
        cache_key: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        prefix_hash: str,
    ) -> None:
        """存储 KV-Cache 到各级缓存"""
        # L1: GPU
        await self._promote_to_gpu(cache_key, keys, values, layer_idx, prefix_hash)

        # L2: CPU
        self._add_to_cpu_cache(cache_key, keys, values)

        # L3: Redis（异步写入）
        if self.redis:
            asyncio.create_task(self._write_to_redis(cache_key, keys, values))

    async def _write_to_redis(
        self,
        cache_key: str,
        keys: torch.Tensor,
        values: torch.Tensor,
        ttl: int = 3600,
    ) -> None:
        """写入 Redis"""
        if self.redis is None:
            return
        try:
            kv_bytes = self._serialize_kv(keys, values)
            await self.redis.setex(f"kv:{cache_key}", ttl, kv_bytes)
        except Exception as e:
            logger.warning(f"Redis write error: {e}")

    def _serialize_kv(self, keys: torch.Tensor, values: torch.Tensor) -> bytes:
        """序列化 KV tensors"""
        import io
        import pickle
        buffer = io.BytesIO()
        pickle.dump({
            "keys": keys.cpu().numpy(),
            "values": values.cpu().numpy(),
        }, buffer)
        return buffer.getvalue()

    def _deserialize_kv(self, data: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """反序列化 KV tensors"""
        import io
        import pickle
        buffer = io.BytesIO(data)
        kv_dict = pickle.load(buffer)
        return (
            torch.from_numpy(kv_dict["keys"]),
            torch.from_numpy(kv_dict["values"]),
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = sum(self._stats.values())
        return {
            **self._stats,
            "total_requests": total_requests,
            "l1_hit_rate": self._stats["l1_hits"] / max(1, total_requests),
            "l2_hit_rate": self._stats["l2_hits"] / max(1, total_requests),
            "l3_hit_rate": self._stats["l3_hits"] / max(1, total_requests),
            "gpu_memory_gb": self.gpu_cache.get_total_memory_gb(),
            "cpu_cache_items": len(self.cpu_cache),
        }
