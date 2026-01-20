"""
测试 worker/distributed/kv_cache.py 模块

覆盖：
- CacheLocation 枚举
- CacheBlock 数据类
- PagedKVCache 分页缓存管理
- KVCachePool 多层缓存池
- DistributedKVCacheManager 分布式缓存管理器
"""
import pytest
import torch
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from collections import OrderedDict

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.distributed.kv_cache import (
    CacheLocation,
    CacheBlock,
    PagedKVCache,
    KVCachePool,
    DistributedKVCacheManager,
)


# ============== CacheLocation 测试 ==============

class TestCacheLocation:
    """CacheLocation 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert CacheLocation.GPU.value == "gpu"
        assert CacheLocation.CPU.value == "cpu"
        assert CacheLocation.REDIS.value == "redis"
        assert CacheLocation.REMOTE.value == "remote"

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(CacheLocation) == 4
        assert CacheLocation.GPU in CacheLocation
        assert CacheLocation.CPU in CacheLocation


# ============== CacheBlock 测试 ==============

class TestCacheBlock:
    """CacheBlock 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        block = CacheBlock(block_id="test_block")

        assert block.block_id == "test_block"
        assert block.block_size == 16
        assert block.keys is None
        assert block.values is None
        assert block.layer_idx == 0
        assert block.num_tokens == 0
        assert block.ref_count == 1
        assert block.prefix_hash == ""
        assert block.location == CacheLocation.GPU

    def test_custom_values(self):
        """测试自定义值"""
        keys = torch.randn(8, 16, 64)
        values = torch.randn(8, 16, 64)

        block = CacheBlock(
            block_id="custom_block",
            block_size=32,
            keys=keys,
            values=values,
            layer_idx=5,
            num_tokens=10,
            ref_count=2,
            prefix_hash="abc123",
            location=CacheLocation.CPU,
        )

        assert block.block_id == "custom_block"
        assert block.block_size == 32
        assert block.keys is keys
        assert block.values is values
        assert block.layer_idx == 5
        assert block.num_tokens == 10
        assert block.ref_count == 2
        assert block.prefix_hash == "abc123"
        assert block.location == CacheLocation.CPU

    def test_is_full_property(self):
        """测试 is_full 属性"""
        block = CacheBlock(block_id="test", block_size=16, num_tokens=10)
        assert block.is_full is False

        block.num_tokens = 16
        assert block.is_full is True

        block.num_tokens = 20
        assert block.is_full is True

    def test_is_shared_property(self):
        """测试 is_shared 属性"""
        block = CacheBlock(block_id="test", ref_count=1)
        assert block.is_shared is False

        block.ref_count = 2
        assert block.is_shared is True

    def test_add_ref(self):
        """测试增加引用计数"""
        block = CacheBlock(block_id="test", ref_count=1)

        block.add_ref()
        assert block.ref_count == 2

        block.add_ref()
        assert block.ref_count == 3

    def test_remove_ref(self):
        """测试减少引用计数"""
        block = CacheBlock(block_id="test", ref_count=3)

        result = block.remove_ref()
        assert result == 2
        assert block.ref_count == 2

        result = block.remove_ref()
        assert result == 1

        # 不会低于 0
        block.ref_count = 0
        result = block.remove_ref()
        assert result == 0
        assert block.ref_count == 0

    def test_touch(self):
        """测试更新访问时间"""
        block = CacheBlock(block_id="test")
        old_time = block.last_access

        time.sleep(0.01)
        block.touch()

        assert block.last_access > old_time


# ============== PagedKVCache 测试 ==============

class TestPagedKVCache:
    """PagedKVCache 测试"""

    @pytest.fixture
    def cache(self):
        """创建 PagedKVCache 实例（CPU 模式）"""
        return PagedKVCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_blocks=10,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, cache):
        """测试初始化"""
        assert cache.num_layers == 4
        assert cache.num_heads == 8
        assert cache.head_dim == 64
        assert cache.block_size == 16
        assert cache.max_blocks == 10
        assert cache.device == "cpu"

    def test_init_stats(self, cache):
        """测试初始统计"""
        stats = cache.get_stats()

        assert stats["allocations"] == 0
        assert stats["evictions"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_allocate_block(self, cache):
        """测试分配块"""
        block = cache.allocate_block(layer_idx=0, prefix_hash="test_prefix")

        assert block is not None
        assert block.layer_idx == 0
        assert block.prefix_hash == "test_prefix"
        assert block.block_id in cache._blocks

        stats = cache.get_stats()
        assert stats["allocations"] == 1

    def test_allocate_multiple_blocks(self, cache):
        """测试分配多个块"""
        blocks = []
        for i in range(5):
            block = cache.allocate_block(layer_idx=i % 4)
            blocks.append(block)

        assert len(blocks) == 5
        assert all(b is not None for b in blocks)

        stats = cache.get_stats()
        assert stats["allocations"] == 5
        assert stats["total_blocks"] == 5

    def test_free_block(self, cache):
        """测试释放块"""
        block = cache.allocate_block(layer_idx=0)
        block_id = block.block_id

        assert block_id in cache._blocks

        cache.free_block(block_id)

        assert block_id not in cache._blocks
        assert block_id in cache._free_blocks

    def test_free_shared_block(self, cache):
        """测试释放共享块"""
        block = cache.allocate_block(layer_idx=0)
        block.add_ref()  # ref_count = 2
        block_id = block.block_id

        # 第一次释放，只减少引用计数
        cache.free_block(block_id)
        assert block_id in cache._blocks
        assert block.ref_count == 1

        # 第二次释放，真正删除
        cache.free_block(block_id)
        assert block_id not in cache._blocks

    def test_get_block_hit(self, cache):
        """测试获取块 - 命中"""
        block = cache.allocate_block(layer_idx=0)
        block_id = block.block_id

        retrieved = cache.get_block(block_id)

        assert retrieved is block

        stats = cache.get_stats()
        assert stats["hits"] == 1

    def test_get_block_miss(self, cache):
        """测试获取块 - 未命中"""
        retrieved = cache.get_block("nonexistent_block")

        assert retrieved is None

        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        """测试 LRU 淘汰"""
        cache = PagedKVCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_blocks=3,  # 小容量
            device="cpu",
        )

        # 分配所有块
        blocks = []
        for i in range(3):
            block = cache.allocate_block(layer_idx=0)
            blocks.append(block)

        assert len(cache._free_blocks) == 0

        # 尝试分配新块，应触发淘汰
        new_block = cache.allocate_block(layer_idx=0)

        assert new_block is not None
        stats = cache.get_stats()
        assert stats["evictions"] == 1

    def test_get_stats(self, cache):
        """测试获取统计"""
        cache.allocate_block(layer_idx=0)
        cache.allocate_block(layer_idx=1)
        cache.get_block("nonexistent")

        stats = cache.get_stats()

        assert stats["allocations"] == 2
        assert stats["total_blocks"] == 2
        assert stats["misses"] == 1


# ============== KVCachePool 测试 ==============

class TestKVCachePool:
    """KVCachePool 测试"""

    @pytest.fixture
    def pool(self):
        """创建 KVCachePool 实例"""
        return KVCachePool(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_blocks_per_layer=10,
            device="cpu",
        )

    def test_init(self, pool):
        """测试初始化"""
        assert pool.num_layers == 4
        assert pool.num_heads == 8
        assert pool.head_dim == 64
        assert len(pool._layer_caches) == 4

    def test_allocate_sequence(self, pool):
        """测试分配序列"""
        # 32 tokens，block_size=16，需要 2 个块/层
        blocks = pool.allocate_sequence(seq_len=32, prefix_hash="test")

        assert len(blocks) == 4  # 4 层
        assert all(len(layer_blocks) == 2 for layer_blocks in blocks)

    def test_allocate_sequence_partial(self, pool):
        """测试分配不完整序列"""
        # 20 tokens，block_size=16，需要 2 个块/层（向上取整）
        blocks = pool.allocate_sequence(seq_len=20)

        assert len(blocks) == 4
        assert all(len(layer_blocks) == 2 for layer_blocks in blocks)

    def test_allocate_sequence_failure(self):
        """测试分配失败"""
        pool = KVCachePool(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            block_size=16,
            max_blocks_per_layer=2,  # 非常小的容量
            device="cpu",
        )

        # 分配一些块先占用
        pool.allocate_sequence(seq_len=32)

        # 再次分配应该失败
        with pytest.raises(RuntimeError, match="Failed to allocate"):
            pool.allocate_sequence(seq_len=64)

    def test_get_total_memory_gb(self, pool):
        """测试获取总内存"""
        memory = pool.get_total_memory_gb()

        # CPU 模式下应该是 0（因为没有初始化内存池）
        assert memory >= 0


# ============== DistributedKVCacheManager 测试 ==============

class TestDistributedKVCacheManager:
    """DistributedKVCacheManager 测试"""

    @pytest.fixture
    def manager(self):
        """创建 DistributedKVCacheManager 实例"""
        return DistributedKVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            gpu_cache_blocks=10,
            cpu_cache_gb=0.001,  # 小容量便于测试
            redis_client=None,
            block_size=16,
            device="cpu",
        )

    def test_init(self, manager):
        """测试初始化"""
        assert manager.num_layers == 4
        assert manager.num_heads == 8
        assert manager.head_dim == 64
        assert manager.redis is None

    def test_compute_prefix_hash(self, manager):
        """测试计算前缀哈希"""
        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 5]
        tokens3 = [1, 2, 3, 4, 6]

        hash1 = manager.compute_prefix_hash(tokens1)
        hash2 = manager.compute_prefix_hash(tokens2)
        hash3 = manager.compute_prefix_hash(tokens3)

        # 相同 tokens 应该得到相同哈希
        assert hash1 == hash2
        # 不同 tokens 应该得到不同哈希
        assert hash1 != hash3
        # 哈希长度
        assert len(hash1) == 16

    def test_initial_stats(self, manager):
        """测试初始统计"""
        stats = manager.get_stats()

        assert stats["l1_hits"] == 0
        assert stats["l2_hits"] == 0
        assert stats["l3_hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self, manager):
        """测试缓存未命中"""
        keys = torch.randn(8, 16, 64)
        values = torch.randn(8, 16, 64)

        async def compute_fn():
            return keys, values

        result_keys, result_values = await manager.get_or_compute(
            prefix_hash="test_prefix",
            layer_idx=0,
            compute_fn=compute_fn,
        )

        # 应该调用计算函数
        assert result_keys.shape == keys.shape
        assert result_values.shape == values.shape

        stats = manager.get_stats()
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_add_to_cpu_cache(self, manager):
        """测试添加到 CPU 缓存"""
        keys = torch.randn(8, 16, 64)
        values = torch.randn(8, 16, 64)

        manager._add_to_cpu_cache("test_key", keys, values)

        assert "test_key" in manager.cpu_cache
        cached_keys, cached_values = manager.cpu_cache["test_key"]
        assert cached_keys.shape == keys.shape
        assert cached_keys.device.type == "cpu"

    @pytest.mark.asyncio
    async def test_cpu_cache_lru(self):
        """测试 CPU 缓存 LRU"""
        manager = DistributedKVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            gpu_cache_blocks=10,
            cpu_cache_gb=0.0001,  # 极小容量
            redis_client=None,
            device="cpu",
        )

        # 强制设置小容量
        manager.cpu_cache_max_items = 3

        # 添加多个条目
        for i in range(5):
            keys = torch.randn(8, 16, 64)
            values = torch.randn(8, 16, 64)
            manager._add_to_cpu_cache(f"key_{i}", keys, values)

        # 应该只保留最新的 3 个
        assert len(manager.cpu_cache) == 3
        assert "key_2" in manager.cpu_cache
        assert "key_3" in manager.cpu_cache
        assert "key_4" in manager.cpu_cache

    def test_serialize_deserialize_kv(self, manager):
        """测试 KV 序列化/反序列化"""
        keys = torch.randn(8, 16, 64)
        values = torch.randn(8, 16, 64)

        serialized = manager._serialize_kv(keys, values)
        deserialized_keys, deserialized_values = manager._deserialize_kv(serialized)

        assert torch.allclose(keys, deserialized_keys, atol=1e-5)
        assert torch.allclose(values, deserialized_values, atol=1e-5)

    @pytest.mark.asyncio
    async def test_get_from_redis_no_client(self, manager):
        """测试无 Redis 客户端时获取"""
        result = await manager._get_from_redis("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_from_redis_with_client(self):
        """测试有 Redis 客户端时获取"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        manager = DistributedKVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            redis_client=mock_redis,
            device="cpu",
        )

        result = await manager._get_from_redis("test_key")

        mock_redis.get.assert_called_once_with("kv:test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_write_to_redis(self):
        """测试写入 Redis"""
        mock_redis = AsyncMock()

        manager = DistributedKVCacheManager(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            redis_client=mock_redis,
            device="cpu",
        )

        keys = torch.randn(8, 16, 64)
        values = torch.randn(8, 16, 64)

        await manager._write_to_redis("test_key", keys, values, ttl=60)

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args
        assert args[0][0] == "kv:test_key"
        assert args[0][1] == 60

    def test_get_stats(self, manager):
        """测试获取统计"""
        # 模拟一些访问
        manager._stats["l1_hits"] = 10
        manager._stats["l2_hits"] = 5
        manager._stats["misses"] = 3

        stats = manager.get_stats()

        assert stats["l1_hits"] == 10
        assert stats["l2_hits"] == 5
        assert stats["misses"] == 3
        assert stats["total_requests"] == 18
        assert stats["l1_hit_rate"] == pytest.approx(10 / 18, abs=0.01)


# ============== 集成测试 ==============

class TestKVCacheIntegration:
    """KV-Cache 集成测试"""

    @pytest.mark.asyncio
    async def test_multi_level_cache_flow(self):
        """测试多级缓存流程"""
        manager = DistributedKVCacheManager(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            gpu_cache_blocks=5,
            cpu_cache_gb=0.001,
            device="cpu",
        )

        call_count = 0

        async def compute_fn():
            nonlocal call_count
            call_count += 1
            return torch.randn(4, 16, 32), torch.randn(4, 16, 32)

        # 第一次访问 - 缓存未命中
        await manager.get_or_compute("prefix1", 0, compute_fn)
        assert call_count == 1
        assert manager._stats["misses"] == 1

        # 第二次访问相同前缀 - 应该命中 L2（CPU）缓存
        # 因为 L1 GPU 缓存在 CPU 模式下可能没有正确初始化
        await manager.get_or_compute("prefix1", 0, compute_fn)
        # compute_fn 不应该被再次调用
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_prefixes(self):
        """测试不同前缀"""
        manager = DistributedKVCacheManager(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            device="cpu",
        )

        call_count = 0

        async def compute_fn():
            nonlocal call_count
            call_count += 1
            return torch.randn(4, 16, 32), torch.randn(4, 16, 32)

        # 访问不同前缀
        await manager.get_or_compute("prefix1", 0, compute_fn)
        await manager.get_or_compute("prefix2", 0, compute_fn)
        await manager.get_or_compute("prefix3", 0, compute_fn)

        # 每个前缀都应该调用一次计算函数
        assert call_count == 3
        assert manager._stats["misses"] == 3

    def test_pool_with_multiple_sequences(self):
        """测试缓存池处理多个序列"""
        pool = KVCachePool(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            block_size=8,
            max_blocks_per_layer=20,
            device="cpu",
        )

        # 分配多个序列
        seq1_blocks = pool.allocate_sequence(16, "seq1")
        seq2_blocks = pool.allocate_sequence(24, "seq2")

        # seq1: 16 tokens = 2 blocks/layer
        assert all(len(lb) == 2 for lb in seq1_blocks)

        # seq2: 24 tokens = 3 blocks/layer
        assert all(len(lb) == 3 for lb in seq2_blocks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
