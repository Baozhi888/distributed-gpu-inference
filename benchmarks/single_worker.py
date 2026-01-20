"""
单 Worker 推理性能基准测试

比较不同推理后端的性能：
- Native (原生 Transformers)
- SGLang (RadixAttention)
- vLLM (PagedAttention)

测试指标：
- 吞吐量 (tokens/s)
- 首 Token 延迟 (TTFT)
- 端到端延迟 (E2E Latency)
- 显存利用率
- 前缀缓存命中率（如支持）

使用方法：
    python benchmarks/single_worker.py --backend sglang --model Qwen/Qwen2.5-7B-Instruct
    python benchmarks/single_worker.py --backend all --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    backend: str
    model_id: str

    # 吞吐量指标
    total_tokens: int
    total_time_s: float
    tokens_per_second: float

    # 延迟指标 (ms)
    avg_ttft_ms: float  # 首 Token 延迟
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    avg_e2e_ms: float  # 端到端延迟
    p50_e2e_ms: float
    p95_e2e_ms: float
    p99_e2e_ms: float

    # 资源指标
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_utilization_pct: float

    # 批处理指标
    avg_batch_size: float
    total_requests: int

    # 缓存指标（可选）
    prefix_cache_hit_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    model_id: str
    backend: str

    # 请求配置
    num_requests: int = 100
    concurrent_requests: int = 8
    max_tokens: int = 256
    temperature: float = 0.7

    # 测试 prompt 配置
    prompt_lengths: List[int] = None  # 不同长度的 prompt
    use_shared_prefix: bool = True  # 是否使用共享前缀（测试前缀缓存）

    # 预热配置
    warmup_requests: int = 5

    def __post_init__(self):
        if self.prompt_lengths is None:
            self.prompt_lengths = [128, 256, 512, 1024]


# 测试用 Prompt 模板
SYSTEM_PROMPT = """你是一个专业的AI助手，能够回答各种问题并提供有用的信息。请用简洁、准确的语言回答用户的问题。"""

TEST_PROMPTS = [
    "请解释什么是机器学习，并举例说明它的应用场景。",
    "Python 中的装饰器是什么？请给出一个实际的使用示例。",
    "请比较 RESTful API 和 GraphQL 的优缺点。",
    "什么是微服务架构？它有哪些优势和挑战？",
    "请解释 Docker 容器和虚拟机的区别。",
    "如何优化 SQL 查询性能？请列举一些常见的方法。",
    "请解释分布式系统中的 CAP 定理。",
    "什么是 Kubernetes？它解决了什么问题？",
]


def get_gpu_info() -> Dict[str, float]:
    """获取 GPU 信息"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            used = torch.cuda.memory_allocated(device) / (1024**3)

            # 尝试获取 GPU 利用率
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                pynvml.nvmlShutdown()
            except:
                gpu_util = 0.0

            return {
                "total_gb": total,
                "used_gb": used,
                "utilization_pct": gpu_util
            }
    except Exception as e:
        logger.warning(f"无法获取 GPU 信息: {e}")

    return {"total_gb": 0, "used_gb": 0, "utilization_pct": 0}


def percentile(data: List[float], p: float) -> float:
    """计算百分位数"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engine = None
        self._ttft_times: List[float] = []
        self._e2e_times: List[float] = []
        self._total_tokens: int = 0
        self._batch_sizes: List[int] = []

    def _create_engine(self):
        """创建推理引擎"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))

        from engines import create_llm_engine, get_engine

        engine_config = {
            "model_id": self.config.model_id,
            "backend": self.config.backend,
            # SGLang 配置
            "sglang": {
                "tp_size": 1,
                "mem_fraction_static": 0.85,
                "enable_prefix_caching": True,
            },
            # vLLM 配置
            "vllm": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.85,
                "enable_prefix_caching": True,
            }
        }

        if self.config.backend == "native":
            # 使用原生引擎
            EngineClass = get_engine("llm")
            self.engine = EngineClass(engine_config)
        else:
            # 使用高性能引擎
            self.engine = create_llm_engine(engine_config)

        logger.info(f"已加载引擎: {self.config.backend}")

    def _generate_prompts(self) -> List[Dict[str, Any]]:
        """生成测试用的 prompt"""
        prompts = []

        for i in range(self.config.num_requests):
            # 选择一个测试 prompt
            user_prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]

            # 构造 messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            prompts.append({
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            })

        return prompts

    async def _run_single_request(
        self,
        params: Dict[str, Any],
        request_id: int
    ) -> Dict[str, Any]:
        """运行单个请求并记录指标"""
        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0

        try:
            # 检查是否支持流式生成（用于测量 TTFT）
            if hasattr(self.engine, "stream_generate"):
                async for chunk in self.engine.stream_generate(
                    params["messages"],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"]
                ):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_generated += 1
            elif hasattr(self.engine, "generate_async"):
                # 异步生成
                from engines.llm_base import GenerationConfig
                config = GenerationConfig(
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"]
                )
                result = await self.engine.generate_async(params["messages"], config)
                first_token_time = time.perf_counter()  # 非流式无法测量真实 TTFT
                tokens_generated = result.completion_tokens
            else:
                # 同步生成
                result = self.engine.inference(params)
                first_token_time = time.perf_counter()
                # 估算 tokens
                tokens_generated = len(result.get("response", "").split()) * 1.5

        except Exception as e:
            logger.error(f"请求 {request_id} 失败: {e}")
            return {"success": False, "error": str(e)}

        end_time = time.perf_counter()

        # 计算延迟
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        e2e_ms = (end_time - start_time) * 1000

        return {
            "success": True,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "tokens": int(tokens_generated),
        }

    async def _run_concurrent_requests(
        self,
        prompts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """并发运行多个请求"""
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def limited_request(params, request_id):
            async with semaphore:
                return await self._run_single_request(params, request_id)

        tasks = [
            limited_request(params, i)
            for i, params in enumerate(prompts)
        ]

        return await asyncio.gather(*tasks)

    async def run(self) -> BenchmarkResult:
        """运行完整的基准测试"""
        logger.info(f"开始基准测试: backend={self.config.backend}, model={self.config.model_id}")

        # 1. 加载引擎
        logger.info("加载推理引擎...")
        self._create_engine()

        # 2. 生成测试 prompts
        prompts = self._generate_prompts()
        logger.info(f"生成了 {len(prompts)} 个测试请求")

        # 3. 预热
        logger.info(f"预热中 ({self.config.warmup_requests} 个请求)...")
        warmup_prompts = prompts[:self.config.warmup_requests]
        await self._run_concurrent_requests(warmup_prompts)

        # 4. 正式测试
        logger.info("开始正式测试...")
        test_prompts = prompts[self.config.warmup_requests:]

        start_time = time.perf_counter()
        results = await self._run_concurrent_requests(test_prompts)
        total_time = time.perf_counter() - start_time

        # 5. 统计结果
        successful_results = [r for r in results if r.get("success")]

        ttft_times = [r["ttft_ms"] for r in successful_results]
        e2e_times = [r["e2e_ms"] for r in successful_results]
        total_tokens = sum(r["tokens"] for r in successful_results)

        # 获取 GPU 信息
        gpu_info = get_gpu_info()

        # 获取缓存命中率（如果支持）
        cache_hit_rate = None
        if hasattr(self.engine, "get_cache_stats"):
            stats = self.engine.get_cache_stats()
            cache_hit_rate = stats.get("hit_rate")

        result = BenchmarkResult(
            backend=self.config.backend,
            model_id=self.config.model_id,
            total_tokens=total_tokens,
            total_time_s=total_time,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            avg_ttft_ms=statistics.mean(ttft_times) if ttft_times else 0,
            p50_ttft_ms=percentile(ttft_times, 50),
            p95_ttft_ms=percentile(ttft_times, 95),
            p99_ttft_ms=percentile(ttft_times, 99),
            avg_e2e_ms=statistics.mean(e2e_times) if e2e_times else 0,
            p50_e2e_ms=percentile(e2e_times, 50),
            p95_e2e_ms=percentile(e2e_times, 95),
            p99_e2e_ms=percentile(e2e_times, 99),
            gpu_memory_used_gb=gpu_info["used_gb"],
            gpu_memory_total_gb=gpu_info["total_gb"],
            gpu_utilization_pct=gpu_info["utilization_pct"],
            avg_batch_size=len(successful_results) / (total_time / (statistics.mean(e2e_times) / 1000)) if e2e_times else 0,
            total_requests=len(successful_results),
            prefix_cache_hit_rate=cache_hit_rate,
        )

        logger.info(f"测试完成: {result.tokens_per_second:.1f} tokens/s")

        return result


def print_results(results: List[BenchmarkResult]):
    """打印测试结果对比"""
    print("\n" + "=" * 80)
    print("基准测试结果对比")
    print("=" * 80)

    # 表头
    headers = ["指标", *[r.backend for r in results]]
    col_width = 15

    print(f"{'指标':<20}", end="")
    for r in results:
        print(f"{r.backend:>{col_width}}", end="")
    print()
    print("-" * (20 + col_width * len(results)))

    # 数据行
    metrics = [
        ("吞吐量 (tokens/s)", "tokens_per_second", ".1f"),
        ("平均 TTFT (ms)", "avg_ttft_ms", ".1f"),
        ("P95 TTFT (ms)", "p95_ttft_ms", ".1f"),
        ("平均延迟 (ms)", "avg_e2e_ms", ".1f"),
        ("P95 延迟 (ms)", "p95_e2e_ms", ".1f"),
        ("显存使用 (GB)", "gpu_memory_used_gb", ".2f"),
        ("GPU 利用率 (%)", "gpu_utilization_pct", ".1f"),
        ("总 Tokens", "total_tokens", "d"),
        ("成功请求数", "total_requests", "d"),
    ]

    for label, attr, fmt in metrics:
        print(f"{label:<20}", end="")
        for r in results:
            value = getattr(r, attr)
            if isinstance(value, float):
                print(f"{value:>{col_width}{fmt}}", end="")
            else:
                print(f"{value:>{col_width}}", end="")
        print()

    # 缓存命中率（如果有）
    has_cache = any(r.prefix_cache_hit_rate is not None for r in results)
    if has_cache:
        print(f"{'缓存命中率 (%)':<20}", end="")
        for r in results:
            if r.prefix_cache_hit_rate is not None:
                print(f"{r.prefix_cache_hit_rate * 100:>{col_width}.1f}", end="")
            else:
                print(f"{'N/A':>{col_width}}", end="")
        print()

    print("=" * (20 + col_width * len(results)))


def save_results(results: List[BenchmarkResult], output_path: str):
    """保存测试结果到 JSON 文件"""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="单 Worker 推理性能基准测试")

    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        choices=["native", "sglang", "vllm", "all"],
        help="要测试的后端 (默认: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型 ID (默认: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="测试请求数 (默认: 100)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="并发请求数 (默认: 8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="每个请求的最大 token 数 (默认: 256)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="输出文件路径 (默认: benchmark_results.json)"
    )

    args = parser.parse_args()

    # 确定要测试的后端
    backends = ["native", "sglang", "vllm"] if args.backend == "all" else [args.backend]

    results = []

    for backend in backends:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试后端: {backend}")
        logger.info(f"{'='*60}")

        config = BenchmarkConfig(
            model_id=args.model,
            backend=backend,
            num_requests=args.num_requests,
            concurrent_requests=args.concurrent,
            max_tokens=args.max_tokens,
        )

        try:
            runner = BenchmarkRunner(config)
            result = await runner.run()
            results.append(result)
        except ImportError as e:
            logger.warning(f"跳过 {backend}: 依赖未安装 - {e}")
        except Exception as e:
            logger.error(f"测试 {backend} 失败: {e}")

    if results:
        print_results(results)
        save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
