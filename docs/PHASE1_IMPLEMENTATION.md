# Phase 1: 推理后端升级实施指南

## 目标

将单 Worker 推理效率提升 5-10x，通过集成现代高性能推理后端。

---

## 1. 当前状态分析

### 现有架构

```
worker/engines/
├── base.py          # 引擎基类 (BaseEngine)
├── llm.py           # 原生 Transformers 推理
├── image_gen.py     # 图像生成引擎
└── vision.py        # 视觉引擎
```

### 现有 LLM 引擎问题

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| 使用原生 `model.generate()` | 无 PagedAttention，显存利用率低 | 集成 SGLang/vLLM |
| 无批处理优化 | 每次只处理一个请求 | 添加连续批处理 |
| 无前缀缓存 | 重复计算相同前缀 | 启用 RadixAttention |
| 同步推理 | 阻塞等待结果 | 异步批处理队列 |

---

## 2. 技术选型

### 推理后端对比

| 框架 | 吞吐量 | 易用性 | 特色功能 | 推荐场景 |
|------|--------|--------|----------|----------|
| **SGLang** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | RadixAttention、纯 Python | RAG、Agent、高 KV 复用 |
| **vLLM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PagedAttention、社区大 | 通用生产环境 |
| **TensorRT-LLM** | ⭐⭐⭐⭐⭐ | ⭐⭐ | NVIDIA 深度优化 | 纯 NVIDIA 高端集群 |

**决策**：优先实现 SGLang，保留 vLLM 兼容性

---

## 3. 实施步骤

### Step 1: 创建优化引擎基类

**文件**: `worker/engines/llm_base.py`

```python
from abc import abstractmethod
from typing import Dict, Any, List, Optional
from .base import BaseEngine

class LLMBaseEngine(BaseEngine):
    """LLM 引擎扩展基类"""

    @abstractmethod
    async def generate_async(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步生成接口"""
        pass

    @abstractmethod
    async def batch_generate(
        self,
        batch_params: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量生成接口"""
        pass

    def supports_streaming(self) -> bool:
        """是否支持流式输出"""
        return False

    def supports_prefix_caching(self) -> bool:
        """是否支持前缀缓存"""
        return False
```

### Step 2: 实现 SGLang 引擎

**文件**: `worker/engines/llm_sglang.py`

功能：
- PagedAttention 内存管理
- RadixAttention 前缀缓存
- 连续批处理
- 异步生成

### Step 3: 实现 vLLM 引擎

**文件**: `worker/engines/llm_vllm.py`

功能：
- PagedAttention
- 连续批处理
- 多种量化支持

### Step 4: 实现连续批处理器

**文件**: `worker/batch_processor.py`

功能：
- 请求队列管理
- 动态批大小调整
- 超时处理
- 优先级支持

### Step 5: 更新引擎注册表

**文件**: `worker/engines/__init__.py`

添加新引擎到注册表，支持通过配置选择后端。

### Step 6: 更新配置系统

**文件**: `worker/config.py`

添加推理后端配置选项。

---

## 4. 文件清单

| 文件 | 类型 | 描述 |
|------|------|------|
| `worker/engines/llm_base.py` | 新建 | LLM 引擎扩展基类 |
| `worker/engines/llm_sglang.py` | 新建 | SGLang 后端引擎 |
| `worker/engines/llm_vllm.py` | 新建 | vLLM 后端引擎 |
| `worker/batch_processor.py` | 新建 | 连续批处理器 |
| `worker/engines/__init__.py` | 修改 | 更新引擎注册表 |
| `worker/config.py` | 修改 | 添加后端配置 |
| `worker/requirements.txt` | 修改 | 添加依赖 |
| `benchmarks/single_worker.py` | 新建 | 性能基准测试 |

---

## 5. 依赖要求

### SGLang 依赖

```txt
sglang>=0.4.0
flashinfer>=0.1.0  # 可选，用于加速
```

### vLLM 依赖

```txt
vllm>=0.6.0
```

### 通用依赖

```txt
torch>=2.3.0
transformers>=4.43.0
```

---

## 6. 配置示例

### config.yaml 新增配置

```yaml
# LLM 引擎配置
engines:
  llm:
    # 后端选择: native | sglang | vllm
    backend: sglang

    # 模型配置
    model_id: Qwen/Qwen2.5-7B-Instruct

    # SGLang 特定配置
    sglang:
      tp_size: 1                    # 张量并行度
      mem_fraction_static: 0.85     # GPU 内存占用比例
      chunked_prefill_size: 8192    # 分块预填充大小
      enable_prefix_caching: true   # 启用前缀缓存

    # vLLM 特定配置
    vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.85
      max_num_seqs: 256
      enable_prefix_caching: true

    # 批处理配置
    batch:
      enabled: true
      max_batch_size: 32
      max_wait_ms: 50
```

---

## 7. 验证测试

### 功能测试

1. 单请求推理正确性
2. 批量请求处理
3. 流式输出（如支持）
4. 错误处理和恢复

### 性能测试

1. 单请求延迟对比
2. 吞吐量对比（QPS）
3. 显存利用率
4. 前缀缓存命中率

---

## 8. 回滚方案

保留原生 `llm.py`，通过配置 `backend: native` 可随时切换回原实现。

---

## 9. 预期收益

| 指标 | 原生 Transformers | SGLang | 提升 |
|------|-------------------|--------|------|
| 吞吐量 (tokens/s) | ~50 | ~200+ | 4x+ |
| 显存利用率 | 20-30% | 80%+ | 3x |
| 首 Token 延迟 | ~500ms | ~100ms | 5x |
| 前缀缓存 | 无 | 有 | - |

---

*文档版本: 1.0*
*最后更新: 2025-12-30*
