"""
可观测性模块

实现分布式推理平台的监控和追踪：
- Prometheus 指标导出
- OpenTelemetry 分布式追踪
- 结构化日志
- 性能分析
"""
import logging
import time
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============== Prometheus 指标 ==============

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")


if HAS_PROMETHEUS:
    # 推理指标
    INFERENCE_REQUESTS_TOTAL = Counter(
        "inference_requests_total",
        "Total number of inference requests",
        ["model", "worker_id", "status"]
    )

    INFERENCE_LATENCY = Histogram(
        "inference_latency_seconds",
        "Inference latency in seconds",
        ["model", "phase", "worker_role"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    TOKENS_GENERATED = Counter(
        "tokens_generated_total",
        "Total tokens generated",
        ["model", "worker_id"]
    )

    TOKENS_PER_SECOND = Gauge(
        "tokens_per_second",
        "Current tokens per second",
        ["model", "worker_id"]
    )

    # KV-Cache 指标
    KV_CACHE_HIT_RATE = Gauge(
        "kv_cache_hit_rate",
        "KV cache hit rate",
        ["level"]  # gpu, cpu, redis, remote
    )

    KV_CACHE_SIZE_BYTES = Gauge(
        "kv_cache_size_bytes",
        "KV cache size in bytes",
        ["level", "worker_id"]
    )

    KV_CACHE_EVICTIONS = Counter(
        "kv_cache_evictions_total",
        "Total KV cache evictions",
        ["level", "worker_id"]
    )

    # Worker 指标
    WORKER_STATUS = Gauge(
        "worker_status",
        "Worker status (1=online, 0=offline)",
        ["worker_id", "role"]
    )

    GPU_MEMORY_USED = Gauge(
        "gpu_memory_used_bytes",
        "GPU memory used in bytes",
        ["worker_id", "gpu_id"]
    )

    GPU_MEMORY_TOTAL = Gauge(
        "gpu_memory_total_bytes",
        "GPU memory total in bytes",
        ["worker_id", "gpu_id"]
    )

    GPU_UTILIZATION = Gauge(
        "gpu_utilization_percent",
        "GPU utilization percentage",
        ["worker_id", "gpu_id"]
    )

    # 分布式推理指标
    DISTRIBUTED_HOPS = Histogram(
        "distributed_inference_hops",
        "Number of hops in distributed inference",
        ["model"],
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    KV_MIGRATION_LATENCY = Histogram(
        "kv_migration_latency_seconds",
        "KV cache migration latency",
        ["source_worker", "target_worker"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    )

    # 批处理指标
    BATCH_SIZE = Histogram(
        "batch_size",
        "Batch size distribution",
        ["phase"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128]
    )

    QUEUE_SIZE = Gauge(
        "queue_size",
        "Current queue size",
        ["phase"]  # prefill, decode
    )

    # 推测解码指标
    SPECULATIVE_ACCEPT_RATE = Gauge(
        "speculative_accept_rate",
        "Speculative decoding accept rate",
        ["worker_id"]
    )

    SPECULATIVE_SPEEDUP = Gauge(
        "speculative_speedup",
        "Speculative decoding speedup",
        ["worker_id"]
    )


# ============== OpenTelemetry 追踪 ==============

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger.warning("opentelemetry not installed. Tracing will be disabled.")


class TracingManager:
    """追踪管理器"""

    def __init__(self, service_name: str = "distributed-inference"):
        self.service_name = service_name
        self._tracer = None
        self._enabled = False

    def setup(
        self,
        exporter=None,
        sample_rate: float = 1.0,
    ) -> None:
        """
        设置追踪

        Args:
            exporter: OTLP 导出器（可选）
            sample_rate: 采样率
        """
        if not HAS_OTEL:
            logger.warning("OpenTelemetry not available")
            return

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.semconv.resource import ResourceAttributes

        resource = Resource(attributes={
            ResourceAttributes.SERVICE_NAME: self.service_name,
        })

        provider = TracerProvider(resource=resource)

        if exporter:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self.service_name)
        self._enabled = True

        logger.info(f"Tracing enabled for service: {self.service_name}")

    @contextmanager
    def span(
        self,
        name: str,
        kind: "SpanKind" = None,
        attributes: Dict[str, Any] = None,
    ):
        """创建追踪 span"""
        if not self._enabled or self._tracer is None:
            yield None
            return

        if kind is None:
            kind = SpanKind.INTERNAL

        with self._tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def trace_inference(self, func: Callable) -> Callable:
        """推理追踪装饰器"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attributes = {
                "inference.model": kwargs.get("model", "unknown"),
                "inference.batch_size": kwargs.get("batch_size", 1),
            }

            with self.span("inference", attributes=attributes) as span:
                start_time = time.time()
                result = await func(*args, **kwargs)
                latency = time.time() - start_time

                if span:
                    span.set_attribute("inference.latency_ms", latency * 1000)
                    if hasattr(result, "tokens"):
                        span.set_attribute("inference.tokens", len(result.tokens))

                return result

        return wrapper


# 全局追踪管理器
tracer = TracingManager()


# ============== 指标收集器 ==============

@dataclass
class MetricsCollector:
    """
    指标收集器

    收集和导出各种运行时指标
    """
    worker_id: str
    model_name: str = ""
    worker_role: str = "hybrid"

    # 内部计数器
    _request_count: int = 0
    _token_count: int = 0
    _error_count: int = 0
    _latency_sum: float = 0.0
    _last_update: float = field(default_factory=time.time)

    def record_request(
        self,
        phase: str,
        latency_seconds: float,
        tokens: int = 0,
        success: bool = True,
    ) -> None:
        """记录请求指标"""
        self._request_count += 1
        self._token_count += tokens
        self._latency_sum += latency_seconds

        if not success:
            self._error_count += 1

        if HAS_PROMETHEUS:
            status = "success" if success else "error"
            INFERENCE_REQUESTS_TOTAL.labels(
                model=self.model_name,
                worker_id=self.worker_id,
                status=status
            ).inc()

            INFERENCE_LATENCY.labels(
                model=self.model_name,
                phase=phase,
                worker_role=self.worker_role
            ).observe(latency_seconds)

            if tokens > 0:
                TOKENS_GENERATED.labels(
                    model=self.model_name,
                    worker_id=self.worker_id
                ).inc(tokens)

    def record_batch(self, phase: str, batch_size: int) -> None:
        """记录批处理指标"""
        if HAS_PROMETHEUS:
            BATCH_SIZE.labels(phase=phase).observe(batch_size)

    def record_kv_cache_stats(
        self,
        level: str,
        hit_rate: float,
        size_bytes: int = 0,
        evictions: int = 0,
    ) -> None:
        """记录 KV-Cache 指标"""
        if HAS_PROMETHEUS:
            KV_CACHE_HIT_RATE.labels(level=level).set(hit_rate)

            if size_bytes > 0:
                KV_CACHE_SIZE_BYTES.labels(
                    level=level,
                    worker_id=self.worker_id
                ).set(size_bytes)

            if evictions > 0:
                KV_CACHE_EVICTIONS.labels(
                    level=level,
                    worker_id=self.worker_id
                ).inc(evictions)

    def record_gpu_stats(
        self,
        gpu_id: int,
        memory_used: int,
        memory_total: int,
        utilization: float = 0.0,
    ) -> None:
        """记录 GPU 指标"""
        if HAS_PROMETHEUS:
            GPU_MEMORY_USED.labels(
                worker_id=self.worker_id,
                gpu_id=str(gpu_id)
            ).set(memory_used)

            GPU_MEMORY_TOTAL.labels(
                worker_id=self.worker_id,
                gpu_id=str(gpu_id)
            ).set(memory_total)

            GPU_UTILIZATION.labels(
                worker_id=self.worker_id,
                gpu_id=str(gpu_id)
            ).set(utilization)

    def record_speculative_stats(
        self,
        accept_rate: float,
        speedup: float,
    ) -> None:
        """记录推测解码指标"""
        if HAS_PROMETHEUS:
            SPECULATIVE_ACCEPT_RATE.labels(
                worker_id=self.worker_id
            ).set(accept_rate)

            SPECULATIVE_SPEEDUP.labels(
                worker_id=self.worker_id
            ).set(speedup)

    def update_tokens_per_second(self) -> None:
        """更新 tokens/s 指标"""
        now = time.time()
        elapsed = now - self._last_update

        if elapsed > 0:
            tps = self._token_count / elapsed
            if HAS_PROMETHEUS:
                TOKENS_PER_SECOND.labels(
                    model=self.model_name,
                    worker_id=self.worker_id
                ).set(tps)

        # 重置计数器
        self._token_count = 0
        self._last_update = now

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        avg_latency = self._latency_sum / max(1, self._request_count)
        error_rate = self._error_count / max(1, self._request_count)

        return {
            "worker_id": self.worker_id,
            "model_name": self.model_name,
            "total_requests": self._request_count,
            "total_tokens": self._token_count,
            "error_count": self._error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency * 1000,
        }


# ============== HTTP 端点 ==============

def create_metrics_endpoint():
    """创建 Prometheus 指标端点（用于 FastAPI）"""
    if not HAS_PROMETHEUS:
        async def metrics_disabled():
            return {"error": "Prometheus not available"}
        return metrics_disabled

    async def metrics():
        from fastapi.responses import Response
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    return metrics


def setup_metrics_routes(app):
    """设置 FastAPI 指标路由"""
    from fastapi import APIRouter

    router = APIRouter()

    if HAS_PROMETHEUS:
        @router.get("/metrics")
        async def metrics():
            from fastapi.responses import Response
            return Response(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )

    @router.get("/health")
    async def health():
        return {"status": "healthy"}

    @router.get("/ready")
    async def ready():
        return {"status": "ready"}

    app.include_router(router)


# ============== 日志增强 ==============

class StructuredLogger:
    """结构化日志器"""

    def __init__(self, name: str = "distributed-inference"):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """设置日志上下文"""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """清除日志上下文"""
        self._context.clear()

    def _format_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """格式化额外字段"""
        return {**self._context, **extra}

    def info(self, message: str, **extra) -> None:
        self.logger.info(message, extra=self._format_extra(extra))

    def warning(self, message: str, **extra) -> None:
        self.logger.warning(message, extra=self._format_extra(extra))

    def error(self, message: str, **extra) -> None:
        self.logger.error(message, extra=self._format_extra(extra))

    def debug(self, message: str, **extra) -> None:
        self.logger.debug(message, extra=self._format_extra(extra))


# 全局结构化日志器
structured_logger = StructuredLogger()
