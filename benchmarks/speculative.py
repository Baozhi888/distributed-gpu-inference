"""
推测解码基准测试

测试 EAGLE-3 风格推测解码的性能提升：
- 对比启用/禁用推测解码
- 测量单请求延迟降低
- 分析接受率与加速比
- 自适应深度调整效果

测试指标：
- 单请求延迟 (Latency)
- 加速比 (Speedup)
- 接受率 (Accept Rate)
- Tokens/Step (每步生成的 tokens)
- Draft 生成开销

使用方法：
    # 对比测试
    python benchmarks/speculative.py --compare

    # 测试特定配置
    python benchmarks/speculative.py --tree-depth 5 --tree-width 3

    # 测试不同 prompt 长度
    python benchmarks/speculative.py --sweep-prompt-length
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

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SpeculativeResult:
    """推测解码测试结果"""
    enabled: bool
    model_id: str

    # 配置
    tree_depth: int
    tree_width: int
    num_speculative_tokens: int

    # 整体指标
    total_requests: int
    total_tokens: int
    total_time_s: float

    # 延迟指标 (ms)
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # 推测解码指标
    avg_accept_rate: float
    avg_tokens_per_step: float
    avg_speedup: float

    # Draft 开销
    avg_draft_time_ms: float
    avg_verify_time_ms: float
    draft_overhead_pct: float

    # 自适应深度
    avg_effective_depth: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeculativeConfig:
    """推测解码测试配置"""
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # 推测参数
    tree_depth: int = 5
    tree_width: int = 3
    num_speculative_tokens: int = 5

    # 请求配置
    num_requests: int = 50
    prompt_length: int = 256
    max_tokens: int = 128
    temperature: float = 0.0  # 确定性采样便于测试

    # 自适应
    adaptive_depth: bool = True
    min_accept_rate: float = 0.3


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

class SpeculativeDecodingSimulator:
    """推测解码模拟器"""

    def __init__(self, config: SpeculativeConfig):
        self.config = config

        # 模型参数
        self.hidden_size = 4096
        self.num_layers = 32

        # 性能参数（模拟）
        self.base_decode_time_ms = 25.0  # 标准解码单步时间
        self.draft_time_ratio = 0.1     # Draft 相对于 Target 的时间比例
        self.verify_overhead_ratio = 0.2  # 验证开销比例

        # 接受率模型（模拟真实分布）
        # 实际接受率与 temperature、模型质量等相关
        self.base_accept_rate = 0.65  # 基础接受率

        # 自适应状态
        self._current_depth = config.tree_depth
        self._accept_history: List[float] = []

    def _simulate_accept_rate(self, depth: int) -> float:
        """模拟接受率"""
        # 接受率随深度递减
        # rate = base_rate * decay^depth
        decay = 0.9
        rate = self.base_accept_rate * (decay ** (depth - 1))

        # 添加一些随机性
        import random
        rate *= random.uniform(0.9, 1.1)

        return max(0.1, min(0.95, rate))

    def _simulate_decode_step_no_spec(self) -> Tuple[int, float]:
        """模拟标准解码（无推测）"""
        # 单 token 生成
        latency = self.base_decode_time_ms

        # 添加一些随机性
        import random
        latency *= random.uniform(0.95, 1.05)

        return 1, latency

    def _simulate_decode_step_with_spec(self) -> Tuple[int, float, Dict[str, float]]:
        """模拟推测解码"""
        import random

        # 1. Draft 阶段
        draft_time = self.base_decode_time_ms * self.draft_time_ratio * self._current_depth

        # 2. 计算每个深度的接受情况
        accepted_tokens = 0
        total_candidates = 0

        for d in range(self._current_depth):
            width = self.config.tree_width if d == 0 else 1  # 简化：只在第一层有分支
            total_candidates += width

            accept_rate = self._simulate_accept_rate(d + 1)

            if random.random() < accept_rate:
                accepted_tokens += 1
            else:
                break  # 一旦拒绝，后续都不接受

        # 至少接受 1 个 token（验证后的新 token）
        accepted_tokens = max(1, accepted_tokens)

        # 3. Verify 阶段
        # 验证时间与候选数相关
        verify_time = self.base_decode_time_ms * (1 + self.verify_overhead_ratio)

        # 总时间
        total_time = draft_time + verify_time

        # 计算实际接受率
        actual_accept_rate = accepted_tokens / max(1, self._current_depth)

        # 自适应调整深度
        if self.config.adaptive_depth:
            self._adapt_depth(actual_accept_rate)

        stats = {
            "draft_time_ms": draft_time,
            "verify_time_ms": verify_time,
            "accept_rate": actual_accept_rate,
            "candidates": total_candidates,
        }

        return accepted_tokens, total_time, stats

    def _adapt_depth(self, accept_rate: float):
        """自适应调整深度"""
        self._accept_history.append(accept_rate)

        # 使用最近 10 次的平均接受率
        recent = self._accept_history[-10:]
        avg_rate = statistics.mean(recent)

        if avg_rate < self.config.min_accept_rate:
            self._current_depth = max(1, self._current_depth - 1)
        elif avg_rate > 0.7 and self._current_depth < self.config.tree_depth:
            self._current_depth = min(self.config.tree_depth, self._current_depth + 1)

    async def generate(self, max_tokens: int) -> Dict[str, Any]:
        """生成指定数量的 tokens"""
        generated = 0
        total_time = 0.0
        steps = 0

        draft_times = []
        verify_times = []
        accept_rates = []
        tokens_per_step = []
        depths = []

        while generated < max_tokens:
            if self.config.enabled:
                tokens, latency, stats = self._simulate_decode_step_with_spec()
                draft_times.append(stats["draft_time_ms"])
                verify_times.append(stats["verify_time_ms"])
                accept_rates.append(stats["accept_rate"])
                depths.append(self._current_depth)
            else:
                tokens, latency = self._simulate_decode_step_no_spec()

            tokens = min(tokens, max_tokens - generated)
            generated += tokens
            total_time += latency
            steps += 1
            tokens_per_step.append(tokens)

            # 模拟异步
            await asyncio.sleep(0.001)

        return {
            "tokens": generated,
            "latency_ms": total_time,
            "steps": steps,
            "avg_tokens_per_step": statistics.mean(tokens_per_step) if tokens_per_step else 1,
            "avg_draft_time_ms": statistics.mean(draft_times) if draft_times else 0,
            "avg_verify_time_ms": statistics.mean(verify_times) if verify_times else 0,
            "avg_accept_rate": statistics.mean(accept_rates) if accept_rates else 0,
            "avg_depth": statistics.mean(depths) if depths else self.config.tree_depth,
        }


class SpeculativeBenchmarkRunner:
    """推测解码基准测试运行器"""

    def __init__(self, config: SpeculativeConfig):
        self.config = config

    async def _run_single_request(self) -> Dict[str, Any]:
        """运行单个请求"""
        simulator = SpeculativeDecodingSimulator(self.config)

        start_time = time.perf_counter()
        result = await simulator.generate(self.config.max_tokens)
        wall_time = time.perf_counter() - start_time

        return {
            **result,
            "wall_time_ms": wall_time * 1000,
        }

    async def run(self) -> SpeculativeResult:
        """运行完整测试"""
        mode = "启用" if self.config.enabled else "禁用"
        logger.info(f"开始推测解码测试 - {mode}")
        if self.config.enabled:
            logger.info(f"  Tree Depth: {self.config.tree_depth}")
            logger.info(f"  Tree Width: {self.config.tree_width}")

        # 预热
        logger.info("预热...")
        for _ in range(3):
            await self._run_single_request()

        # 正式测试
        logger.info(f"正式测试 ({self.config.num_requests} 请求)...")

        start_time = time.perf_counter()

        tasks = [self._run_single_request() for _ in range(self.config.num_requests)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # 统计
        latencies = [r["latency_ms"] for r in results]
        total_tokens = sum(r["tokens"] for r in results)

        # 推测解码指标
        if self.config.enabled:
            accept_rates = [r["avg_accept_rate"] for r in results]
            tokens_per_step = [r["avg_tokens_per_step"] for r in results]
            draft_times = [r["avg_draft_time_ms"] for r in results]
            verify_times = [r["avg_verify_time_ms"] for r in results]
            depths = [r["avg_depth"] for r in results]

            avg_accept_rate = statistics.mean(accept_rates)
            avg_tokens_per_step = statistics.mean(tokens_per_step)
            avg_draft_time = statistics.mean(draft_times)
            avg_verify_time = statistics.mean(verify_times)
            avg_depth = statistics.mean(depths)

            # 计算 draft 开销占比
            total_step_time = avg_draft_time + avg_verify_time
            draft_overhead = avg_draft_time / total_step_time * 100 if total_step_time > 0 else 0
        else:
            avg_accept_rate = 0
            avg_tokens_per_step = 1.0
            avg_draft_time = 0
            avg_verify_time = 0
            draft_overhead = 0
            avg_depth = 0

        # 计算加速比（相对于基准单 token 生成）
        base_latency = self.config.max_tokens * 25.0  # 基准延迟
        avg_latency = statistics.mean(latencies)
        speedup = base_latency / avg_latency if avg_latency > 0 else 1.0

        result = SpeculativeResult(
            enabled=self.config.enabled,
            model_id=self.config.model_id,
            tree_depth=self.config.tree_depth,
            tree_width=self.config.tree_width,
            num_speculative_tokens=self.config.num_speculative_tokens,
            total_requests=self.config.num_requests,
            total_tokens=total_tokens,
            total_time_s=total_time,
            avg_latency_ms=avg_latency,
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            avg_accept_rate=avg_accept_rate,
            avg_tokens_per_step=avg_tokens_per_step,
            avg_speedup=speedup,
            avg_draft_time_ms=avg_draft_time,
            avg_verify_time_ms=avg_verify_time,
            draft_overhead_pct=draft_overhead,
            avg_effective_depth=avg_depth,
        )

        return result


# ============== 输出函数 ==============

def print_single_result(result: SpeculativeResult):
    """打印单个结果"""
    mode = "启用推测解码" if result.enabled else "标准解码"
    print(f"\n{'='*60}")
    print(f"{mode}")
    print(f"{'='*60}")

    if result.enabled:
        print(f"\n配置:")
        print(f"  Tree Depth: {result.tree_depth}")
        print(f"  Tree Width: {result.tree_width}")

    print(f"\n延迟:")
    print(f"  平均: {result.avg_latency_ms:.1f} ms")
    print(f"  P50:  {result.p50_latency_ms:.1f} ms")
    print(f"  P95:  {result.p95_latency_ms:.1f} ms")
    print(f"  P99:  {result.p99_latency_ms:.1f} ms")

    print(f"\n性能:")
    print(f"  总 Tokens: {result.total_tokens}")
    print(f"  加速比: {result.avg_speedup:.2f}x")

    if result.enabled:
        print(f"\n推测解码指标:")
        print(f"  平均接受率: {result.avg_accept_rate * 100:.1f}%")
        print(f"  平均 Tokens/Step: {result.avg_tokens_per_step:.2f}")
        print(f"  平均有效深度: {result.avg_effective_depth:.1f}")
        print(f"\nDraft 开销:")
        print(f"  Draft 时间: {result.avg_draft_time_ms:.1f} ms")
        print(f"  Verify 时间: {result.avg_verify_time_ms:.1f} ms")
        print(f"  Draft 开销占比: {result.draft_overhead_pct:.1f}%")


def print_comparison(baseline: SpeculativeResult, speculative: SpeculativeResult):
    """打印对比"""
    print("\n" + "=" * 70)
    print("推测解码 vs 标准解码 对比")
    print("=" * 70)

    print(f"\n{'指标':<25} {'标准解码':>15} {'推测解码':>15} {'提升':>12}")
    print("-" * 70)

    def print_row(label, base_val, spec_val, higher_is_better=True):
        if higher_is_better:
            improvement = ((spec_val - base_val) / base_val * 100) if base_val > 0 else 0
            sign = "+" if improvement > 0 else ""
        else:
            improvement = ((base_val - spec_val) / base_val * 100) if base_val > 0 else 0
            sign = "+" if improvement > 0 else ""

        print(f"{label:<25} {base_val:>15.1f} {spec_val:>15.1f} {sign}{improvement:>10.1f}%")

    # 延迟（越低越好）
    print_row("平均延迟 (ms)", baseline.avg_latency_ms, speculative.avg_latency_ms, higher_is_better=False)
    print_row("P95 延迟 (ms)", baseline.p95_latency_ms, speculative.p95_latency_ms, higher_is_better=False)

    # 加速比（越高越好）
    print_row("加速比", baseline.avg_speedup, speculative.avg_speedup, higher_is_better=True)

    print("-" * 70)

    # 总结
    latency_reduction = ((baseline.avg_latency_ms - speculative.avg_latency_ms) / baseline.avg_latency_ms * 100) if baseline.avg_latency_ms > 0 else 0

    print(f"\n总结:")
    print(f"  延迟降低: {latency_reduction:.1f}%")
    print(f"  实际加速: {speculative.avg_speedup:.2f}x")
    print(f"  平均接受率: {speculative.avg_accept_rate * 100:.1f}%")
    print(f"  平均每步生成: {speculative.avg_tokens_per_step:.2f} tokens")

    # 理论 vs 实际加速
    theoretical_speedup = speculative.avg_tokens_per_step
    efficiency = (speculative.avg_speedup / theoretical_speedup * 100) if theoretical_speedup > 0 else 0
    print(f"  理论加速: {theoretical_speedup:.2f}x")
    print(f"  效率: {efficiency:.1f}%")

    print("=" * 70)


def save_results(results: List[SpeculativeResult], output_path: str):
    """保存结果"""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="推测解码基准测试")

    parser.add_argument(
        "--enabled",
        type=str,
        default="true",
        choices=["true", "false"],
        help="是否启用推测解码"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比启用/禁用推测解码"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型 ID"
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=5,
        help="Tree 深度"
    )
    parser.add_argument(
        "--tree-width",
        type=int,
        default=3,
        help="Tree 宽度"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="测试请求数"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每请求最大 tokens"
    )
    parser.add_argument(
        "--sweep-depth",
        action="store_true",
        help="扫描不同深度"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="speculative_results.json",
        help="输出文件"
    )

    args = parser.parse_args()

    results = []

    if args.compare:
        # 对比测试
        for enabled in [False, True]:
            config = SpeculativeConfig(
                enabled=enabled,
                model_id=args.model,
                tree_depth=args.tree_depth,
                tree_width=args.tree_width,
                num_requests=args.num_requests,
                max_tokens=args.max_tokens,
            )

            runner = SpeculativeBenchmarkRunner(config)
            result = await runner.run()
            results.append(result)

            print_single_result(result)

        # 打印对比
        print_comparison(results[0], results[1])

    elif args.sweep_depth:
        # 扫描不同深度
        print("\n扫描不同 Tree Depth...")
        for depth in [1, 2, 3, 4, 5, 6, 7, 8]:
            config = SpeculativeConfig(
                enabled=True,
                model_id=args.model,
                tree_depth=depth,
                tree_width=args.tree_width,
                num_requests=args.num_requests // 2,  # 减少请求数
                max_tokens=args.max_tokens,
            )

            runner = SpeculativeBenchmarkRunner(config)
            result = await runner.run()
            results.append(result)

            logger.info(f"Depth {depth}: 加速 {result.avg_speedup:.2f}x, 接受率 {result.avg_accept_rate*100:.1f}%")

        # 找到最佳深度
        best = max(results, key=lambda r: r.avg_speedup)
        print(f"\n最佳深度: {best.tree_depth}, 加速: {best.avg_speedup:.2f}x")

    else:
        # 单一配置测试
        config = SpeculativeConfig(
            enabled=args.enabled == "true",
            model_id=args.model,
            tree_depth=args.tree_depth,
            tree_width=args.tree_width,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
        )

        runner = SpeculativeBenchmarkRunner(config)
        result = await runner.run()
        results.append(result)

        print_single_result(result)

    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
