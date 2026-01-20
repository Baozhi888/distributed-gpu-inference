"""
Prefill/Decode 分离基准测试

测试 DistServe 风格的 Prefill/Decode 分离架构：
- 对比分离模式 vs 混合模式
- 测量 TTFT (首 Token 时间) 提升
- 测量吞吐量 (tokens/s) 提升
- 分析 KV-Cache 迁移开销

测试指标：
- TTFT (首 Token 延迟)
- TPOT (每 Token 平均延迟)
- 吞吐量 (tokens/s)
- KV-Cache 迁移延迟
- Prefill/Decode 队列长度

使用方法：
    # 对比测试分离 vs 混合模式
    python benchmarks/pd_separation.py --compare

    # 测试特定配置
    python benchmarks/pd_separation.py --mode separated --prefill-workers 2 --decode-workers 4

    # 压力测试
    python benchmarks/pd_separation.py --stress --num-requests 500 --concurrent 32
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
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchedulingMode(Enum):
    """调度模式"""
    HYBRID = "hybrid"       # 混合模式：同一 Worker 处理 Prefill 和 Decode
    SEPARATED = "separated"  # 分离模式：专门的 Prefill 和 Decode Worker


@dataclass
class PDSeparationResult:
    """P/D 分离测试结果"""
    mode: str
    model_id: str

    # Worker 配置
    prefill_workers: int
    decode_workers: int

    # 整体指标
    total_requests: int
    successful_requests: int
    total_tokens: int
    total_time_s: float
    tokens_per_second: float

    # TTFT 指标 (ms) - 核心指标
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    # TPOT 指标 (ms) - 每 Token 平均延迟
    avg_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float

    # E2E 延迟
    avg_e2e_ms: float
    p50_e2e_ms: float
    p95_e2e_ms: float

    # KV-Cache 迁移指标（分离模式）
    avg_migration_ms: float = 0.0
    migration_count: int = 0
    migration_bytes: int = 0

    # 队列指标
    avg_prefill_queue_time_ms: float = 0.0
    avg_decode_queue_time_ms: float = 0.0
    max_prefill_queue_size: int = 0
    max_decode_queue_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PDSeparationConfig:
    """P/D 分离测试配置"""
    mode: SchedulingMode = SchedulingMode.SEPARATED
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Worker 配置
    prefill_workers: int = 2
    decode_workers: int = 4

    # 请求配置
    num_requests: int = 100
    concurrent_requests: int = 16
    prompt_length: int = 512
    max_tokens: int = 128
    temperature: float = 0.7

    # 测试配置
    warmup_requests: int = 5

    # 模拟参数
    prefill_flops_tflops: float = 312.0  # A100 FP16
    decode_bandwidth_gbps: float = 2039.0  # A100 HBM


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


# ============== 模拟器 ==============

@dataclass
class SimulatedRequest:
    """模拟请求"""
    request_id: int
    prompt_length: int
    max_tokens: int
    submit_time: float
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0
    decode_start_time: float = 0.0
    decode_end_time: float = 0.0
    migration_time_ms: float = 0.0
    tokens_generated: int = 0


class PDSimulator:
    """P/D 分离模拟器"""

    def __init__(self, config: PDSeparationConfig):
        self.config = config

        # 模型参数 (以 Qwen2.5-7B 为例)
        self.hidden_size = 4096
        self.num_layers = 32
        self.num_heads = 32
        self.head_dim = 128
        self.vocab_size = 152064

        # 队列
        self._prefill_queue: asyncio.Queue = asyncio.Queue()
        self._decode_queue: asyncio.Queue = asyncio.Queue()

        # 统计
        self._max_prefill_queue = 0
        self._max_decode_queue = 0

        # Worker 锁（模拟并发限制）
        self._prefill_semaphore = asyncio.Semaphore(config.prefill_workers)
        self._decode_semaphore = asyncio.Semaphore(config.decode_workers)

    def _estimate_prefill_time_ms(self, prompt_length: int) -> float:
        """
        估算 Prefill 延迟

        Prefill 是计算密集型：
        - 主要时间 = 2 * layers * (attention + ffn) / FLOPS
        - attention: 4 * seq^2 * hidden
        - ffn: 8 * seq * hidden^2
        """
        # 简化估算：FLOPS / 算力
        attention_flops = 4 * prompt_length * prompt_length * self.hidden_size
        ffn_flops = 8 * prompt_length * self.hidden_size * self.hidden_size
        total_flops = self.num_layers * 2 * (attention_flops + ffn_flops)

        time_s = total_flops / (self.config.prefill_flops_tflops * 1e12)
        return time_s * 1000  # ms

    def _estimate_decode_step_time_ms(self, kv_cache_length: int) -> float:
        """
        估算单步 Decode 延迟

        Decode 是内存带宽密集型：
        - 主要时间 = 模型权重 + KV-Cache 读取 / 带宽
        """
        # 模型权重大小 (7B * 2 bytes)
        model_bytes = 7e9 * 2

        # KV-Cache 大小
        kv_bytes = 2 * self.num_layers * kv_cache_length * self.num_heads * self.head_dim * 2

        total_bytes = model_bytes + kv_bytes
        time_s = total_bytes / (self.config.decode_bandwidth_gbps * 1e9)

        # Decode 通常受限于内存带宽，实际延迟约 20-50ms
        return max(time_s * 1000, 20.0)

    def _estimate_kv_migration_time_ms(self, prompt_length: int) -> float:
        """估算 KV-Cache 迁移延迟"""
        # KV-Cache 大小
        kv_bytes = 2 * self.num_layers * prompt_length * self.num_heads * self.head_dim * 2

        # 网络传输 (假设 10 Gbps 网络)
        network_gbps = 10.0
        return (kv_bytes / (network_gbps * 1e9 / 8)) * 1000  # ms

    async def _process_request_hybrid(self, request: SimulatedRequest) -> SimulatedRequest:
        """混合模式处理请求"""
        # 同一个 Worker 处理 Prefill 和 Decode
        async with self._prefill_semaphore:
            # Prefill
            request.prefill_start_time = time.perf_counter()
            prefill_time = self._estimate_prefill_time_ms(request.prompt_length)
            await asyncio.sleep(prefill_time / 1000)
            request.prefill_end_time = time.perf_counter()

            # Decode (同一 Worker，无迁移)
            request.decode_start_time = request.prefill_end_time
            kv_length = request.prompt_length

            for i in range(request.max_tokens):
                decode_time = self._estimate_decode_step_time_ms(kv_length)
                await asyncio.sleep(decode_time / 1000)
                kv_length += 1
                request.tokens_generated = i + 1

            request.decode_end_time = time.perf_counter()

        return request

    async def _process_request_separated(self, request: SimulatedRequest) -> SimulatedRequest:
        """分离模式处理请求"""
        # Prefill Worker
        async with self._prefill_semaphore:
            request.prefill_start_time = time.perf_counter()
            prefill_time = self._estimate_prefill_time_ms(request.prompt_length)
            await asyncio.sleep(prefill_time / 1000)
            request.prefill_end_time = time.perf_counter()

        # KV-Cache 迁移 (Prefill -> Decode)
        migration_time = self._estimate_kv_migration_time_ms(request.prompt_length)
        request.migration_time_ms = migration_time
        await asyncio.sleep(migration_time / 1000)

        # Decode Worker
        async with self._decode_semaphore:
            request.decode_start_time = time.perf_counter()
            kv_length = request.prompt_length

            for i in range(request.max_tokens):
                decode_time = self._estimate_decode_step_time_ms(kv_length)
                await asyncio.sleep(decode_time / 1000)
                kv_length += 1
                request.tokens_generated = i + 1

            request.decode_end_time = time.perf_counter()

        return request

    async def process_request(self, request: SimulatedRequest) -> SimulatedRequest:
        """处理请求"""
        if self.config.mode == SchedulingMode.HYBRID:
            return await self._process_request_hybrid(request)
        else:
            return await self._process_request_separated(request)


class PDSeparationBenchmarkRunner:
    """P/D 分离基准测试运行器"""

    def __init__(self, config: PDSeparationConfig):
        self.config = config
        self.simulator = PDSimulator(config)

    async def _run_single_request(
        self,
        request_id: int,
    ) -> SimulatedRequest:
        """运行单个请求"""
        request = SimulatedRequest(
            request_id=request_id,
            prompt_length=self.config.prompt_length,
            max_tokens=self.config.max_tokens,
            submit_time=time.perf_counter(),
        )

        return await self.simulator.process_request(request)

    async def run(self) -> PDSeparationResult:
        """运行完整测试"""
        mode_name = self.config.mode.value
        logger.info(f"开始 P/D 分离测试 - 模式: {mode_name}")
        logger.info(f"  Prefill Workers: {self.config.prefill_workers}")
        logger.info(f"  Decode Workers: {self.config.decode_workers}")

        # 预热
        logger.info(f"预热 ({self.config.warmup_requests} 请求)...")
        warmup_tasks = [
            self._run_single_request(i)
            for i in range(self.config.warmup_requests)
        ]
        await asyncio.gather(*warmup_tasks)

        # 正式测试
        logger.info(f"正式测试 ({self.config.num_requests} 请求, 并发 {self.config.concurrent_requests})...")

        start_time = time.perf_counter()

        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def limited_request(req_id):
            async with semaphore:
                return await self._run_single_request(req_id)

        tasks = [
            limited_request(i)
            for i in range(self.config.num_requests)
        ]

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # 统计结果
        ttft_times = []  # 首 Token 延迟
        tpot_times = []  # 每 Token 平均延迟
        e2e_times = []   # 端到端延迟
        migration_times = []
        total_tokens = 0

        for req in results:
            # TTFT = Prefill 结束时间 - 提交时间
            ttft = (req.prefill_end_time - req.submit_time) * 1000
            ttft_times.append(ttft)

            # E2E = Decode 结束 - 提交时间
            e2e = (req.decode_end_time - req.submit_time) * 1000
            e2e_times.append(e2e)

            # TPOT = Decode 时间 / tokens
            if req.tokens_generated > 0:
                decode_time = (req.decode_end_time - req.decode_start_time) * 1000
                tpot = decode_time / req.tokens_generated
                tpot_times.append(tpot)

            total_tokens += req.tokens_generated

            if req.migration_time_ms > 0:
                migration_times.append(req.migration_time_ms)

        result = PDSeparationResult(
            mode=mode_name,
            model_id=self.config.model_id,
            prefill_workers=self.config.prefill_workers,
            decode_workers=self.config.decode_workers,
            total_requests=self.config.num_requests,
            successful_requests=len(results),
            total_tokens=total_tokens,
            total_time_s=total_time,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            avg_ttft_ms=statistics.mean(ttft_times) if ttft_times else 0,
            p50_ttft_ms=percentile(ttft_times, 50),
            p95_ttft_ms=percentile(ttft_times, 95),
            p99_ttft_ms=percentile(ttft_times, 99),
            avg_tpot_ms=statistics.mean(tpot_times) if tpot_times else 0,
            p50_tpot_ms=percentile(tpot_times, 50),
            p95_tpot_ms=percentile(tpot_times, 95),
            avg_e2e_ms=statistics.mean(e2e_times) if e2e_times else 0,
            p50_e2e_ms=percentile(e2e_times, 50),
            p95_e2e_ms=percentile(e2e_times, 95),
            avg_migration_ms=statistics.mean(migration_times) if migration_times else 0,
            migration_count=len(migration_times),
        )

        return result


# ============== 输出函数 ==============

def print_single_result(result: PDSeparationResult):
    """打印单个结果"""
    print(f"\n{'='*60}")
    print(f"模式: {result.mode.upper()}")
    print(f"{'='*60}")

    print(f"\n配置:")
    print(f"  Prefill Workers: {result.prefill_workers}")
    print(f"  Decode Workers: {result.decode_workers}")

    print(f"\n吞吐量:")
    print(f"  成功请求: {result.successful_requests}/{result.total_requests}")
    print(f"  总 Tokens: {result.total_tokens}")
    print(f"  吞吐量: {result.tokens_per_second:.1f} tokens/s")

    print(f"\n首 Token 延迟 (TTFT) - 关键指标:")
    print(f"  平均: {result.avg_ttft_ms:.1f} ms")
    print(f"  P50:  {result.p50_ttft_ms:.1f} ms")
    print(f"  P95:  {result.p95_ttft_ms:.1f} ms")
    print(f"  P99:  {result.p99_ttft_ms:.1f} ms")

    print(f"\n每 Token 延迟 (TPOT):")
    print(f"  平均: {result.avg_tpot_ms:.1f} ms")
    print(f"  P50:  {result.p50_tpot_ms:.1f} ms")
    print(f"  P95:  {result.p95_tpot_ms:.1f} ms")

    print(f"\n端到端延迟 (E2E):")
    print(f"  平均: {result.avg_e2e_ms:.1f} ms")
    print(f"  P50:  {result.p50_e2e_ms:.1f} ms")
    print(f"  P95:  {result.p95_e2e_ms:.1f} ms")

    if result.migration_count > 0:
        print(f"\nKV-Cache 迁移:")
        print(f"  迁移次数: {result.migration_count}")
        print(f"  平均延迟: {result.avg_migration_ms:.1f} ms")


def print_comparison(hybrid: PDSeparationResult, separated: PDSeparationResult):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("Prefill/Decode 分离 vs 混合模式 对比")
    print("=" * 70)

    # 表头
    print(f"\n{'指标':<25} {'混合模式':>15} {'分离模式':>15} {'提升':>12}")
    print("-" * 70)

    # 数据行
    def print_row(label, hybrid_val, sep_val, higher_is_better=True):
        if higher_is_better:
            improvement = ((sep_val - hybrid_val) / hybrid_val * 100) if hybrid_val > 0 else 0
            sign = "+" if improvement > 0 else ""
        else:
            improvement = ((hybrid_val - sep_val) / hybrid_val * 100) if hybrid_val > 0 else 0
            sign = "+" if improvement > 0 else ""

        print(f"{label:<25} {hybrid_val:>15.1f} {sep_val:>15.1f} {sign}{improvement:>10.1f}%")

    # TTFT（越低越好）
    print_row("平均 TTFT (ms)", hybrid.avg_ttft_ms, separated.avg_ttft_ms, higher_is_better=False)
    print_row("P95 TTFT (ms)", hybrid.p95_ttft_ms, separated.p95_ttft_ms, higher_is_better=False)

    # 吞吐量（越高越好）
    print_row("吞吐量 (tokens/s)", hybrid.tokens_per_second, separated.tokens_per_second, higher_is_better=True)

    # TPOT（越低越好）
    print_row("平均 TPOT (ms)", hybrid.avg_tpot_ms, separated.avg_tpot_ms, higher_is_better=False)

    # E2E（越低越好）
    print_row("平均 E2E (ms)", hybrid.avg_e2e_ms, separated.avg_e2e_ms, higher_is_better=False)

    print("-" * 70)

    # 总结
    ttft_improvement = ((hybrid.avg_ttft_ms - separated.avg_ttft_ms) / hybrid.avg_ttft_ms * 100) if hybrid.avg_ttft_ms > 0 else 0
    tps_improvement = ((separated.tokens_per_second - hybrid.tokens_per_second) / hybrid.tokens_per_second * 100) if hybrid.tokens_per_second > 0 else 0

    print(f"\n总结:")
    print(f"  TTFT 提升: {ttft_improvement:+.1f}%")
    print(f"  吞吐量提升: {tps_improvement:+.1f}%")

    if separated.migration_count > 0:
        print(f"  KV-Cache 迁移开销: {separated.avg_migration_ms:.1f} ms/请求")

    print("=" * 70)


def save_results(results: List[PDSeparationResult], output_path: str):
    """保存结果"""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Prefill/Decode 分离基准测试")

    parser.add_argument(
        "--mode",
        type=str,
        default="separated",
        choices=["hybrid", "separated"],
        help="调度模式"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比混合模式和分离模式"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型 ID"
    )
    parser.add_argument(
        "--prefill-workers",
        type=int,
        default=2,
        help="Prefill Worker 数量"
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=4,
        help="Decode Worker 数量"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="测试请求数"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=16,
        help="并发请求数"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Prompt 长度"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="最大生成 tokens"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="压力测试模式（高并发）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pd_separation_results.json",
        help="输出文件"
    )

    args = parser.parse_args()

    # 压力测试配置
    if args.stress:
        args.num_requests = 500
        args.concurrent = 32

    results = []

    if args.compare:
        # 对比测试
        for mode in [SchedulingMode.HYBRID, SchedulingMode.SEPARATED]:
            config = PDSeparationConfig(
                mode=mode,
                model_id=args.model,
                prefill_workers=args.prefill_workers,
                decode_workers=args.decode_workers,
                num_requests=args.num_requests,
                concurrent_requests=args.concurrent,
                prompt_length=args.prompt_length,
                max_tokens=args.max_tokens,
            )

            runner = PDSeparationBenchmarkRunner(config)
            result = await runner.run()
            results.append(result)

            print_single_result(result)

        # 打印对比
        print_comparison(results[0], results[1])

    else:
        # 单一模式测试
        mode = SchedulingMode.HYBRID if args.mode == "hybrid" else SchedulingMode.SEPARATED

        config = PDSeparationConfig(
            mode=mode,
            model_id=args.model,
            prefill_workers=args.prefill_workers,
            decode_workers=args.decode_workers,
            num_requests=args.num_requests,
            concurrent_requests=args.concurrent,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
        )

        runner = PDSeparationBenchmarkRunner(config)
        result = await runner.run()
        results.append(result)

        print_single_result(result)

    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
