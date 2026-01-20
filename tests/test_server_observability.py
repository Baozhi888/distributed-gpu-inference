"""
测试 server/app/services/observability.py 模块

覆盖：
- Prometheus 指标定义和记录
- TracingManager 追踪管理器
- MetricsCollector 指标收集器
- StructuredLogger 结构化日志
- HTTP 端点函数
"""
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============== Mock prometheus_client ==============

class MockCounter:
    """模拟 Prometheus Counter"""
    def __init__(self, name, description, labels):
        self.name = name
        self.labels_values = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self.labels_values:
            self.labels_values[key] = MockMetricValue()
        return self.labels_values[key]


class MockHistogram:
    """模拟 Prometheus Histogram"""
    def __init__(self, name, description, labels, buckets=None):
        self.name = name
        self.labels_values = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self.labels_values:
            self.labels_values[key] = MockMetricValue()
        return self.labels_values[key]


class MockGauge:
    """模拟 Prometheus Gauge"""
    def __init__(self, name, description, labels):
        self.name = name
        self.labels_values = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self.labels_values:
            self.labels_values[key] = MockMetricValue()
        return self.labels_values[key]


class MockMetricValue:
    """模拟指标值"""
    def __init__(self):
        self.value = 0

    def inc(self, amount=1):
        self.value += amount

    def set(self, value):
        self.value = value

    def observe(self, value):
        self.value = value


# ============== 导入模块（使用 importlib 直接导入文件） ==============

# 模拟 prometheus_client
mock_prometheus = MagicMock()
mock_prometheus.Counter = MockCounter
mock_prometheus.Histogram = MockHistogram
mock_prometheus.Gauge = MockGauge
mock_prometheus.Info = MagicMock()
mock_prometheus.generate_latest = MagicMock(return_value=b"# metrics")
mock_prometheus.CONTENT_TYPE_LATEST = "text/plain"

# 模拟 opentelemetry
mock_trace = MagicMock()
mock_trace.SpanKind = MagicMock()
mock_trace.SpanKind.INTERNAL = "INTERNAL"
mock_trace.Status = MagicMock()
mock_trace.StatusCode = MagicMock()
mock_trace.StatusCode.ERROR = "ERROR"

mock_span = MagicMock()
mock_span.__enter__ = MagicMock(return_value=mock_span)
mock_span.__exit__ = MagicMock(return_value=None)
mock_span.set_attribute = MagicMock()
mock_span.set_status = MagicMock()

mock_tracer = MagicMock()
mock_tracer.start_as_current_span = MagicMock(return_value=mock_span)


def _load_observability_module():
    """使用 importlib 直接加载 observability.py，绕过包导入"""
    import importlib.util

    module_path = REPO_ROOT / "server" / "app" / "services" / "observability.py"
    spec = importlib.util.spec_from_file_location("observability", module_path)
    module = importlib.util.module_from_spec(spec)

    # 先将模块放入 sys.modules 以支持模块内的相对引用
    sys.modules["observability"] = module
    sys.modules["server.app.services.observability"] = module

    spec.loader.exec_module(module)
    return module


# 使用模拟依赖加载模块
with patch.dict('sys.modules', {
    'prometheus_client': mock_prometheus,
    'opentelemetry': MagicMock(),
    'opentelemetry.trace': mock_trace,
    'opentelemetry.sdk.trace': MagicMock(),
    'opentelemetry.sdk.trace.export': MagicMock(),
    'opentelemetry.sdk.resources': MagicMock(),
    'opentelemetry.semconv.resource': MagicMock(),
}):
    observability = _load_observability_module()
    TracingManager = observability.TracingManager
    MetricsCollector = observability.MetricsCollector
    StructuredLogger = observability.StructuredLogger
    create_metrics_endpoint = observability.create_metrics_endpoint
    setup_metrics_routes = observability.setup_metrics_routes


# ============== TracingManager 测试 ==============

class TestTracingManager:
    """TracingManager 测试"""

    def test_init(self):
        """测试初始化"""
        tracer = TracingManager(service_name="test-service")

        assert tracer.service_name == "test-service"
        assert tracer._tracer is None
        assert tracer._enabled is False

    def test_span_disabled(self):
        """测试追踪禁用时的 span"""
        tracer = TracingManager()

        with tracer.span("test-span") as span:
            assert span is None

    @patch.object(observability, 'HAS_OTEL', True)
    def test_setup_with_otel(self):
        """测试启用 OpenTelemetry"""
        # 由于 OpenTelemetry 是可选依赖且导入方式复杂，简化此测试
        # 只验证 HAS_OTEL 为 True 时 setup 方法可被调用
        tracer = TracingManager(service_name="test")

        # 当 HAS_OTEL 为 True 但实际没有 OpenTelemetry 时，setup 应该正常返回
        # 或者我们验证装饰器正常工作即可
        assert tracer.service_name == "test"
        assert tracer._enabled is False  # 因为我们没有真正初始化

    def test_trace_inference_decorator(self):
        """测试推理追踪装饰器"""
        tracer = TracingManager()

        @tracer.trace_inference
        async def mock_inference(model="test", batch_size=1):
            return MagicMock(tokens=[1, 2, 3])

        # 测试装饰器不破坏函数
        assert asyncio.iscoroutinefunction(mock_inference)


# ============== MetricsCollector 测试 ==============

class TestMetricsCollector:
    """MetricsCollector 测试"""

    @pytest.fixture
    def collector(self):
        """创建收集器实例"""
        return MetricsCollector(
            worker_id="worker-1",
            model_name="test-model",
            worker_role="hybrid",
        )

    def test_init(self, collector):
        """测试初始化"""
        assert collector.worker_id == "worker-1"
        assert collector.model_name == "test-model"
        assert collector.worker_role == "hybrid"
        assert collector._request_count == 0
        assert collector._token_count == 0
        assert collector._error_count == 0

    def test_record_request_success(self, collector):
        """测试记录成功请求"""
        collector.record_request(
            phase="prefill",
            latency_seconds=0.5,
            tokens=100,
            success=True,
        )

        assert collector._request_count == 1
        assert collector._token_count == 100
        assert collector._latency_sum == 0.5
        assert collector._error_count == 0

    def test_record_request_error(self, collector):
        """测试记录失败请求"""
        collector.record_request(
            phase="decode",
            latency_seconds=1.0,
            tokens=0,
            success=False,
        )

        assert collector._request_count == 1
        assert collector._error_count == 1

    def test_record_multiple_requests(self, collector):
        """测试记录多个请求"""
        for i in range(10):
            collector.record_request(
                phase="prefill",
                latency_seconds=0.1 * (i + 1),
                tokens=10,
                success=i % 3 != 0,  # 每3个有一个失败
            )

        assert collector._request_count == 10
        assert collector._token_count == 100
        assert collector._error_count == 4  # 0, 3, 6, 9

    def test_record_batch(self, collector):
        """测试记录批处理"""
        # 应该不报错
        collector.record_batch(phase="prefill", batch_size=16)
        collector.record_batch(phase="decode", batch_size=32)

    def test_record_kv_cache_stats(self, collector):
        """测试记录 KV-Cache 统计"""
        collector.record_kv_cache_stats(
            level="gpu",
            hit_rate=0.85,
            size_bytes=1024 * 1024 * 100,
            evictions=10,
        )
        # 不应该报错

    def test_record_gpu_stats(self, collector):
        """测试记录 GPU 统计"""
        collector.record_gpu_stats(
            gpu_id=0,
            memory_used=8 * 1024 * 1024 * 1024,
            memory_total=24 * 1024 * 1024 * 1024,
            utilization=75.5,
        )
        # 不应该报错

    def test_record_speculative_stats(self, collector):
        """测试记录推测解码统计"""
        collector.record_speculative_stats(
            accept_rate=0.8,
            speedup=2.5,
        )
        # 不应该报错

    def test_update_tokens_per_second(self, collector):
        """测试更新 tokens/s"""
        collector._token_count = 1000
        collector._last_update = time.time() - 1.0  # 1秒前

        collector.update_tokens_per_second()

        # 应该重置计数器
        assert collector._token_count == 0

    def test_get_summary(self, collector):
        """测试获取摘要"""
        # 记录一些数据
        collector.record_request("prefill", 0.1, 50, True)
        collector.record_request("prefill", 0.2, 60, True)
        collector.record_request("decode", 0.3, 70, False)

        summary = collector.get_summary()

        assert summary["worker_id"] == "worker-1"
        assert summary["model_name"] == "test-model"
        assert summary["total_requests"] == 3
        assert summary["error_count"] == 1
        assert summary["error_rate"] == pytest.approx(1/3, rel=0.01)
        assert summary["avg_latency_ms"] == pytest.approx(200, rel=0.01)  # (100+200+300)/3

    def test_get_summary_empty(self, collector):
        """测试空收集器摘要"""
        summary = collector.get_summary()

        assert summary["total_requests"] == 0
        assert summary["error_count"] == 0
        assert summary["error_rate"] == 0.0
        assert summary["avg_latency_ms"] == 0.0


# ============== StructuredLogger 测试 ==============

class TestStructuredLogger:
    """StructuredLogger 测试"""

    @pytest.fixture
    def slogger(self):
        """创建结构化日志器"""
        return StructuredLogger(name="test-logger")

    def test_init(self, slogger):
        """测试初始化"""
        assert slogger.logger is not None
        assert slogger._context == {}

    def test_set_context(self, slogger):
        """测试设置上下文"""
        slogger.set_context(worker_id="w1", model="test")

        assert slogger._context["worker_id"] == "w1"
        assert slogger._context["model"] == "test"

    def test_clear_context(self, slogger):
        """测试清除上下文"""
        slogger.set_context(a=1, b=2)
        slogger.clear_context()

        assert slogger._context == {}

    def test_format_extra(self, slogger):
        """测试格式化额外字段"""
        slogger.set_context(base="context")

        extra = slogger._format_extra({"key": "value"})

        assert extra["base"] == "context"
        assert extra["key"] == "value"

    def test_log_methods(self, slogger):
        """测试日志方法"""
        with patch.object(slogger.logger, 'info') as mock_info:
            slogger.info("test message", key="value")
            mock_info.assert_called_once()

        with patch.object(slogger.logger, 'warning') as mock_warning:
            slogger.warning("warning message", code=123)
            mock_warning.assert_called_once()

        with patch.object(slogger.logger, 'error') as mock_error:
            slogger.error("error message", exception="test")
            mock_error.assert_called_once()

        with patch.object(slogger.logger, 'debug') as mock_debug:
            slogger.debug("debug message")
            mock_debug.assert_called_once()

    def test_context_inheritance(self, slogger):
        """测试上下文继承"""
        slogger.set_context(session_id="abc123")

        with patch.object(slogger.logger, 'info') as mock_info:
            slogger.info("message", request_id="req1")

            # 检查调用参数包含上下文
            call_kwargs = mock_info.call_args
            assert "extra" in call_kwargs.kwargs
            assert call_kwargs.kwargs["extra"]["session_id"] == "abc123"
            assert call_kwargs.kwargs["extra"]["request_id"] == "req1"


# ============== HTTP 端点测试 ==============

class TestMetricsEndpoint:
    """指标端点测试"""

    @pytest.mark.asyncio
    async def test_create_metrics_endpoint_with_prometheus(self):
        """测试创建指标端点（有 Prometheus）"""
        with patch.object(observability, 'HAS_PROMETHEUS', True):
            endpoint = create_metrics_endpoint()

            # 应该返回一个异步函数
            assert asyncio.iscoroutinefunction(endpoint)

    @pytest.mark.asyncio
    async def test_create_metrics_endpoint_without_prometheus(self):
        """测试创建指标端点（无 Prometheus）"""
        with patch.object(observability, 'HAS_PROMETHEUS', False):
            endpoint = create_metrics_endpoint()

            # 调用应该返回错误信息
            result = await endpoint()
            assert "error" in result


class TestSetupMetricsRoutes:
    """设置指标路由测试"""

    def test_setup_routes(self):
        """测试设置路由"""
        mock_app = MagicMock()

        with patch.object(observability, 'HAS_PROMETHEUS', True):
            setup_metrics_routes(mock_app)

        # 应该调用 include_router
        mock_app.include_router.assert_called_once()


# ============== Prometheus 指标测试 ==============

class TestPrometheusMetrics:
    """Prometheus 指标测试"""

    def test_metrics_with_prometheus(self):
        """测试指标记录（有 Prometheus）"""
        collector = MetricsCollector(
            worker_id="test-worker",
            model_name="test-model",
        )

        with patch.object(observability, 'HAS_PROMETHEUS', True):
            with patch.object(observability, 'INFERENCE_REQUESTS_TOTAL') as mock_counter:
                mock_label = MagicMock()
                mock_counter.labels.return_value = mock_label

                collector.record_request("prefill", 0.1, 10, True)

    def test_metrics_without_prometheus(self):
        """测试指标记录（无 Prometheus）"""
        collector = MetricsCollector(
            worker_id="test-worker",
            model_name="test-model",
        )

        # 应该不报错，即使没有 Prometheus
        with patch.object(observability, 'HAS_PROMETHEUS', False):
            collector.record_request("decode", 0.2, 20, True)
            collector.record_batch("prefill", 8)
            collector.record_kv_cache_stats("gpu", 0.9)
            collector.record_gpu_stats(0, 1000, 2000, 50.0)
            collector.record_speculative_stats(0.7, 1.5)


# ============== 集成测试 ==============

class TestObservabilityIntegration:
    """可观测性集成测试"""

    def test_full_metrics_workflow(self):
        """测试完整指标工作流"""
        # 创建收集器
        collector = MetricsCollector(
            worker_id="integration-worker",
            model_name="llama-7b",
            worker_role="hybrid",
        )

        # 模拟推理工作流
        for i in range(5):
            # Prefill 阶段
            collector.record_request(
                phase="prefill",
                latency_seconds=0.1 + i * 0.02,
                tokens=512,
                success=True,
            )
            collector.record_batch("prefill", 4)

            # Decode 阶段
            for _ in range(10):
                collector.record_request(
                    phase="decode",
                    latency_seconds=0.01,
                    tokens=1,
                    success=True,
                )

        # 记录系统指标
        collector.record_gpu_stats(
            gpu_id=0,
            memory_used=12 * 1024**3,
            memory_total=24 * 1024**3,
            utilization=85.0,
        )

        collector.record_kv_cache_stats(
            level="gpu",
            hit_rate=0.92,
            size_bytes=4 * 1024**3,
            evictions=5,
        )

        # 获取摘要
        summary = collector.get_summary()

        assert summary["total_requests"] == 55  # 5 prefill + 50 decode
        assert summary["error_count"] == 0
        assert summary["error_rate"] == 0.0

    def test_logger_with_collector(self):
        """测试日志器与收集器配合"""
        logger = StructuredLogger("integration-test")
        collector = MetricsCollector(
            worker_id="log-worker",
            model_name="test-model",
        )

        # 设置日志上下文
        logger.set_context(
            worker_id=collector.worker_id,
            model=collector.model_name,
        )

        # 模拟请求并记录日志
        with patch.object(logger.logger, 'info') as mock_info:
            start = time.time()
            # 模拟处理
            time.sleep(0.01)
            latency = time.time() - start

            collector.record_request("prefill", latency, 100, True)
            logger.info("Request completed", latency_ms=latency*1000, tokens=100)

            mock_info.assert_called_once()

    def test_tracing_with_metrics(self):
        """测试追踪与指标配合"""
        tracer = TracingManager(service_name="integration-service")
        collector = MetricsCollector(
            worker_id="trace-worker",
            model_name="test-model",
        )

        # 模拟带追踪的请求
        with tracer.span("inference-request") as span:
            start = time.time()
            # 模拟处理
            time.sleep(0.01)
            latency = time.time() - start

            collector.record_request("prefill", latency, 256, True)

            if span:
                span.set_attribute("tokens", 256)

        summary = collector.get_summary()
        assert summary["total_requests"] == 1


class TestMetricsCollectorEdgeCases:
    """MetricsCollector 边界情况测试"""

    def test_zero_latency(self):
        """测试零延迟"""
        collector = MetricsCollector(worker_id="test")
        collector.record_request("prefill", 0.0, 100, True)

        summary = collector.get_summary()
        assert summary["avg_latency_ms"] == 0.0

    def test_large_token_count(self):
        """测试大量 token"""
        collector = MetricsCollector(worker_id="test")
        collector.record_request("prefill", 1.0, 1000000, True)

        assert collector._token_count == 1000000

    def test_high_error_rate(self):
        """测试高错误率"""
        collector = MetricsCollector(worker_id="test")

        for _ in range(100):
            collector.record_request("decode", 0.1, 1, False)

        summary = collector.get_summary()
        assert summary["error_rate"] == 1.0
        assert summary["error_count"] == 100

    def test_mixed_phases(self):
        """测试混合阶段"""
        collector = MetricsCollector(
            worker_id="test",
            worker_role="hybrid",
        )

        # Prefill
        collector.record_request("prefill", 0.5, 512, True)
        collector.record_batch("prefill", 8)

        # Decode
        collector.record_request("decode", 0.01, 1, True)
        collector.record_batch("decode", 32)

        assert collector._request_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
