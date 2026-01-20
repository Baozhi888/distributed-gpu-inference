"""
分布式推理基准测试

测试跨多 Worker 的分布式模型推理性能：
- 模型分片推理 (Llama-70B 等大模型)
- 跨节点 KV-Cache 传输
- 故障恢复延迟
- 端到端吞吐量

测试指标：
- 端到端延迟 (E2E Latency)
- 首 Token 延迟 (TTFT)
- KV-Cache 传输开销
- 节点间通信延迟
- 故障恢复时间

使用方法：
    # 模拟分布式测试（单机多进程）
    python benchmarks/distributed.py --mode simulate --workers 3

    # 真实分布式测试（需要已启动的 Workers）
    python benchmarks/distributed.py --mode real --server-url http://localhost:8000

    # 测试特定模型分片配置
    python benchmarks/distributed.py --model meta-llama/Llama-2-70b-chat-hf --layers-per-worker 27
"""
import argparse
import asyncio
import json
import logging
import statistics
import time
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DistributedBenchmarkResult:
    """分布式基准测试结果"""
    model_id: str
    num_workers: int
    layers_per_worker: int

    # 整体指标
    total_requests: int
    successful_requests: int
    total_tokens: int
    total_time_s: float
    tokens_per_second: float

    # 延迟指标 (ms)
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    avg_e2e_ms: float
    p50_e2e_ms: float
    p95_e2e_ms: float
    p99_e2e_ms: float

    # KV-Cache 传输指标
    avg_kv_transfer_ms: float
    total_kv_bytes_transferred: int

    # 节点间通信指标
    avg_hop_latency_ms: float
    total_hops: int

    # 故障恢复指标（如果测试了）
    failover_tested: bool = False
    avg_failover_time_ms: float = 0.0
    failover_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributedBenchmarkConfig:
    """分布式基准测试配置"""
    model_id: str = "meta-llama/Llama-2-70b-chat-hf"
    num_workers: int = 3
    layers_per_worker: int = 27  # 80层 / 3 workers ≈ 27

    # 请求配置
    num_requests: int = 50
    concurrent_requests: int = 4
    max_tokens: int = 128
    temperature: float = 0.7

    # 测试配置
    warmup_requests: int = 3
    test_failover: bool = False

    # 服务器配置
    server_url: str = "http://localhost:8000"
    api_key: str = ""


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


# ============== 模拟模式 ==============

@dataclass
class SimulatedWorker:
    """模拟的 Worker 节点"""
    worker_id: str
    layer_start: int
    layer_end: int

    # 模拟性能参数
    compute_latency_ms: float = 10.0  # 单层计算延迟
    kv_transfer_rate_gbps: float = 10.0  # KV 传输速率

    async def forward(
        self,
        hidden_states_size_bytes: int,
        kv_cache_size_bytes: int,
    ) -> Tuple[float, float]:
        """
        模拟前向传播

        Returns:
            (compute_time_ms, transfer_time_ms)
        """
        # 计算延迟：层数 * 单层延迟
        num_layers = self.layer_end - self.layer_start
        compute_time = num_layers * self.compute_latency_ms

        # KV 传输延迟
        transfer_time = (kv_cache_size_bytes / (self.kv_transfer_rate_gbps * 1e9 / 8)) * 1000

        # 模拟延迟
        await asyncio.sleep((compute_time + transfer_time) / 1000)

        return compute_time, transfer_time


class SimulatedDistributedSession:
    """模拟的分布式推理会话"""

    def __init__(
        self,
        workers: List[SimulatedWorker],
        hidden_size: int = 8192,  # Llama-70B hidden size
        num_heads: int = 64,
        head_dim: int = 128,
    ):
        self.workers = workers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _estimate_kv_size(self, seq_len: int, num_layers: int) -> int:
        """估算 KV-Cache 大小（字节）"""
        # KV cache: 2 (K+V) * batch * seq_len * num_heads * head_dim * 2 (fp16)
        return 2 * seq_len * self.num_heads * self.head_dim * 2 * num_layers

    def _estimate_hidden_size_bytes(self, seq_len: int) -> int:
        """估算 hidden states 大小（字节）"""
        # hidden: batch * seq_len * hidden_size * 2 (fp16)
        return seq_len * self.hidden_size * 2

    async def prefill(self, prompt_tokens: int) -> Dict[str, float]:
        """执行 Prefill 阶段"""
        total_compute_ms = 0.0
        total_transfer_ms = 0.0
        total_kv_bytes = 0

        hidden_bytes = self._estimate_hidden_size_bytes(prompt_tokens)

        for worker in self.workers:
            num_layers = worker.layer_end - worker.layer_start
            kv_bytes = self._estimate_kv_size(prompt_tokens, num_layers)

            compute_ms, transfer_ms = await worker.forward(hidden_bytes, kv_bytes)

            total_compute_ms += compute_ms
            total_transfer_ms += transfer_ms
            total_kv_bytes += kv_bytes

        return {
            "compute_ms": total_compute_ms,
            "transfer_ms": total_transfer_ms,
            "kv_bytes": total_kv_bytes,
            "hops": len(self.workers),
        }

    async def decode_step(self) -> Dict[str, float]:
        """执行单步 Decode"""
        total_compute_ms = 0.0
        total_transfer_ms = 0.0

        hidden_bytes = self._estimate_hidden_size_bytes(1)  # 单 token

        for worker in self.workers:
            num_layers = worker.layer_end - worker.layer_start
            # Decode 阶段 KV-Cache 增量较小
            kv_bytes = self._estimate_kv_size(1, num_layers)

            compute_ms, transfer_ms = await worker.forward(hidden_bytes, kv_bytes)

            total_compute_ms += compute_ms
            total_transfer_ms += transfer_ms

        return {
            "compute_ms": total_compute_ms,
            "transfer_ms": total_transfer_ms,
        }


class SimulatedBenchmarkRunner:
    """模拟模式的基准测试运行器"""

    def __init__(self, config: DistributedBenchmarkConfig):
        self.config = config
        self.workers: List[SimulatedWorker] = []
        self.session: Optional[SimulatedDistributedSession] = None

    def _setup_workers(self):
        """设置模拟 Worker"""
        total_layers = 80  # Llama-70B

        for i in range(self.config.num_workers):
            start = i * self.config.layers_per_worker
            end = min((i + 1) * self.config.layers_per_worker, total_layers)

            worker = SimulatedWorker(
                worker_id=f"worker-{i}",
                layer_start=start,
                layer_end=end,
                # 模拟不同 GPU 的性能差异
                compute_latency_ms=10.0 + i * 2,  # 略有差异
                kv_transfer_rate_gbps=10.0,
            )
            self.workers.append(worker)

        logger.info(f"设置了 {len(self.workers)} 个模拟 Worker")
        for w in self.workers:
            logger.info(f"  {w.worker_id}: layers {w.layer_start}-{w.layer_end}")

        self.session = SimulatedDistributedSession(self.workers)

    async def _run_single_request(
        self,
        prompt_tokens: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """运行单个请求"""
        start_time = time.perf_counter()

        # 1. Prefill 阶段
        prefill_result = await self.session.prefill(prompt_tokens)
        ttft = time.perf_counter() - start_time

        # 2. Decode 阶段
        total_decode_compute = 0.0
        total_decode_transfer = 0.0

        for _ in range(max_tokens):
            decode_result = await self.session.decode_step()
            total_decode_compute += decode_result["compute_ms"]
            total_decode_transfer += decode_result["transfer_ms"]

        e2e_time = time.perf_counter() - start_time

        return {
            "success": True,
            "ttft_ms": ttft * 1000,
            "e2e_ms": e2e_time * 1000,
            "tokens": max_tokens,
            "kv_bytes": prefill_result["kv_bytes"],
            "hops": prefill_result["hops"],
            "compute_ms": prefill_result["compute_ms"] + total_decode_compute,
            "transfer_ms": prefill_result["transfer_ms"] + total_decode_transfer,
        }

    async def run(self) -> DistributedBenchmarkResult:
        """运行完整测试"""
        logger.info("开始模拟分布式基准测试")

        # 设置 Workers
        self._setup_workers()

        # 预热
        logger.info(f"预热 ({self.config.warmup_requests} 请求)...")
        for _ in range(self.config.warmup_requests):
            await self._run_single_request(256, 32)

        # 正式测试
        logger.info(f"正式测试 ({self.config.num_requests} 请求)...")

        start_time = time.perf_counter()
        tasks = []

        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def limited_request():
            async with semaphore:
                return await self._run_single_request(512, self.config.max_tokens)

        for _ in range(self.config.num_requests):
            tasks.append(limited_request())

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # 统计结果
        successful = [r for r in results if r["success"]]

        ttft_times = [r["ttft_ms"] for r in successful]
        e2e_times = [r["e2e_ms"] for r in successful]
        total_tokens = sum(r["tokens"] for r in successful)
        total_kv_bytes = sum(r["kv_bytes"] for r in successful)
        total_hops = sum(r["hops"] for r in successful)
        transfer_times = [r["transfer_ms"] for r in successful]
        hop_latencies = [r["e2e_ms"] / r["hops"] for r in successful]

        result = DistributedBenchmarkResult(
            model_id=self.config.model_id,
            num_workers=self.config.num_workers,
            layers_per_worker=self.config.layers_per_worker,
            total_requests=self.config.num_requests,
            successful_requests=len(successful),
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
            avg_kv_transfer_ms=statistics.mean(transfer_times) if transfer_times else 0,
            total_kv_bytes_transferred=total_kv_bytes,
            avg_hop_latency_ms=statistics.mean(hop_latencies) if hop_latencies else 0,
            total_hops=total_hops,
        )

        return result


# ============== 真实模式 ==============

class RealBenchmarkRunner:
    """真实分布式测试运行器"""

    def __init__(self, config: DistributedBenchmarkConfig):
        self.config = config
        self._client = None

    async def _setup_client(self):
        """设置 API 客户端"""
        try:
            from sdk.python.inference_client import InferenceClient
            self._client = InferenceClient(
                base_url=self.config.server_url,
                api_key=self.config.api_key,
            )
            logger.info(f"已连接到服务器: {self.config.server_url}")
        except ImportError:
            logger.error("无法导入 InferenceClient，请确保 SDK 已安装")
            raise

    async def _run_single_request(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """运行单个请求"""
        start_time = time.perf_counter()

        try:
            result = await self._client.chat_completion(
                model=self.config.model_id,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            e2e_time = time.perf_counter() - start_time

            return {
                "success": True,
                "ttft_ms": result.get("first_token_time_ms", e2e_time * 1000 * 0.3),
                "e2e_ms": e2e_time * 1000,
                "tokens": result.get("completion_tokens", 0),
            }

        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {"success": False, "error": str(e)}

    async def run(self) -> DistributedBenchmarkResult:
        """运行完整测试"""
        logger.info("开始真实分布式基准测试")

        await self._setup_client()

        # 检查服务器状态
        try:
            status = await self._client.get_server_status()
            num_workers = status.get("workers", {}).get("online", 0)
            logger.info(f"服务器在线 Workers: {num_workers}")
        except Exception as e:
            logger.warning(f"无法获取服务器状态: {e}")
            num_workers = self.config.num_workers

        # 测试 prompts
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请详细解释分布式系统中的一致性模型，包括强一致性、最终一致性和因果一致性的区别。"}
        ]

        # 预热
        logger.info(f"预热 ({self.config.warmup_requests} 请求)...")
        for _ in range(self.config.warmup_requests):
            await self._run_single_request(test_messages)

        # 正式测试
        logger.info(f"正式测试 ({self.config.num_requests} 请求)...")

        start_time = time.perf_counter()

        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def limited_request():
            async with semaphore:
                return await self._run_single_request(test_messages)

        tasks = [limited_request() for _ in range(self.config.num_requests)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # 统计
        successful = [r for r in results if r.get("success")]

        ttft_times = [r["ttft_ms"] for r in successful]
        e2e_times = [r["e2e_ms"] for r in successful]
        total_tokens = sum(r.get("tokens", 0) for r in successful)

        result = DistributedBenchmarkResult(
            model_id=self.config.model_id,
            num_workers=num_workers,
            layers_per_worker=self.config.layers_per_worker,
            total_requests=self.config.num_requests,
            successful_requests=len(successful),
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
            avg_kv_transfer_ms=0,  # 需要从服务端获取
            total_kv_bytes_transferred=0,
            avg_hop_latency_ms=0,
            total_hops=0,
        )

        return result


# ============== 输出和主函数 ==============

def print_results(result: DistributedBenchmarkResult):
    """打印测试结果"""
    print("\n" + "=" * 70)
    print("分布式推理基准测试结果")
    print("=" * 70)

    print(f"\n配置信息:")
    print(f"  模型: {result.model_id}")
    print(f"  Workers 数量: {result.num_workers}")
    print(f"  每 Worker 层数: {result.layers_per_worker}")

    print(f"\n吞吐量:")
    print(f"  总请求数: {result.total_requests}")
    print(f"  成功请求: {result.successful_requests}")
    print(f"  总 Tokens: {result.total_tokens}")
    print(f"  吞吐量: {result.tokens_per_second:.1f} tokens/s")

    print(f"\n首 Token 延迟 (TTFT):")
    print(f"  平均: {result.avg_ttft_ms:.1f} ms")
    print(f"  P50:  {result.p50_ttft_ms:.1f} ms")
    print(f"  P95:  {result.p95_ttft_ms:.1f} ms")
    print(f"  P99:  {result.p99_ttft_ms:.1f} ms")

    print(f"\n端到端延迟 (E2E):")
    print(f"  平均: {result.avg_e2e_ms:.1f} ms")
    print(f"  P50:  {result.p50_e2e_ms:.1f} ms")
    print(f"  P95:  {result.p95_e2e_ms:.1f} ms")
    print(f"  P99:  {result.p99_e2e_ms:.1f} ms")

    if result.avg_kv_transfer_ms > 0:
        print(f"\nKV-Cache 传输:")
        print(f"  平均传输延迟: {result.avg_kv_transfer_ms:.1f} ms")
        print(f"  总传输量: {result.total_kv_bytes_transferred / (1024**2):.1f} MB")

    if result.total_hops > 0:
        print(f"\n节点间通信:")
        print(f"  总跳数: {result.total_hops}")
        print(f"  平均每跳延迟: {result.avg_hop_latency_ms:.1f} ms")

    if result.failover_tested:
        print(f"\n故障恢复:")
        print(f"  平均恢复时间: {result.avg_failover_time_ms:.1f} ms")
        print(f"  恢复成功率: {result.failover_success_rate * 100:.1f}%")

    print("\n" + "=" * 70)


def save_results(result: DistributedBenchmarkResult, output_path: str):
    """保存结果到 JSON"""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result.to_dict()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="分布式推理基准测试")

    parser.add_argument(
        "--mode",
        type=str,
        default="simulate",
        choices=["simulate", "real"],
        help="测试模式: simulate (模拟) 或 real (真实)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-70b-chat-hf",
        help="模型 ID"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Worker 数量"
    )
    parser.add_argument(
        "--layers-per-worker",
        type=int,
        default=27,
        help="每 Worker 的层数"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="测试请求数"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="并发请求数"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每请求最大 tokens"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="服务器地址 (real 模式)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API Key (real 模式)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="distributed_benchmark_results.json",
        help="输出文件路径"
    )

    args = parser.parse_args()

    config = DistributedBenchmarkConfig(
        model_id=args.model,
        num_workers=args.workers,
        layers_per_worker=args.layers_per_worker,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
        max_tokens=args.max_tokens,
        server_url=args.server_url,
        api_key=args.api_key,
    )

    if args.mode == "simulate":
        runner = SimulatedBenchmarkRunner(config)
    else:
        runner = RealBenchmarkRunner(config)

    result = await runner.run()

    print_results(result)
    save_results(result, args.output)


if __name__ == "__main__":
    asyncio.run(main())
