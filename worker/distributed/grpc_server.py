"""
gRPC 服务实现

用于 Worker 间的 P2P 通信，支持：
- 流式推理
- KV-Cache 传输
- 会话管理
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, AsyncIterator
from concurrent import futures

try:
    import grpc
    from grpc import aio as grpc_aio
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

import torch

from .session import WorkerSession, SessionState
from .kv_cache import DistributedKVCacheManager

# 本地导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common.serialization import TensorSerializer

logger = logging.getLogger(__name__)


class InferenceServicer:
    """
    分布式推理 gRPC 服务实现

    处理来自其他 Worker 或客户端的推理请求
    """

    def __init__(
        self,
        worker_id: str,
        model_shard,
        kv_cache_manager: DistributedKVCacheManager,
        max_sessions: int = 100,
    ):
        self.worker_id = worker_id
        self.model_shard = model_shard
        self.kv_cache = kv_cache_manager
        self.max_sessions = max_sessions

        # 会话管理
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # 统计
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "errors": 0,
        }

    async def StreamInference(
        self,
        request_iterator: AsyncIterator,
        context,
    ) -> AsyncIterator:
        """
        流式推理

        支持连续的推理步骤，维护会话状态
        """
        session_id = None

        async for request in request_iterator:
            try:
                start_time = time.time()
                session_id = request.session_id

                # 获取或创建会话
                session = self._get_or_create_session(session_id, request)

                # 反序列化输入
                hidden_states = TensorSerializer.deserialize(
                    request.hidden_states,
                    tuple(request.shape),
                    request.dtype,
                    device=str(self.model_shard.device) if hasattr(self.model_shard, 'device') else "cuda"
                )

                # 执行前向传播
                output, kv_keys = await self._forward(
                    session,
                    hidden_states,
                    request.position,
                    list(request.kv_cache_keys),
                )

                # 序列化输出
                output_bytes, output_shape, output_dtype = TensorSerializer.serialize(output)

                # 如果有下一跳，转发
                if request.next_worker_address:
                    await self._forward_to_next(
                        request.next_worker_address,
                        request.next_session_id,
                        output,
                        request.position,
                        kv_keys,
                    )

                latency_ms = (time.time() - start_time) * 1000
                self._stats["total_requests"] += 1
                self._stats["total_latency_ms"] += latency_ms

                # 构造响应（模拟 protobuf 消息）
                yield {
                    "session_id": session_id,
                    "step_id": request.step_id,
                    "hidden_states": output_bytes,
                    "shape": list(output_shape),
                    "dtype": output_dtype,
                    "updated_kv_keys": kv_keys,
                    "latency_ms": int(latency_ms),
                    "tokens_processed": hidden_states.shape[1] if hasattr(hidden_states, 'shape') else 1,
                    "success": True,
                    "error_message": "",
                }

            except Exception as e:
                logger.error(f"StreamInference error: {e}")
                self._stats["errors"] += 1
                yield {
                    "session_id": session_id or "",
                    "success": False,
                    "error_message": str(e),
                }

    async def Forward(self, request, context) -> Dict[str, Any]:
        """单次前向传播"""
        try:
            start_time = time.time()

            # 获取会话
            session = self._sessions.get(request.session_id)
            if not session:
                session = self._create_session(request.session_id)

            # 反序列化输入
            input_tensor = TensorSerializer.deserialize(
                request.input,
                tuple(request.shape),
                request.dtype,
                device=str(self.model_shard.device) if hasattr(self.model_shard, 'device') else "cuda"
            )

            # 执行前向传播
            output, kv_keys = await self._forward(
                session,
                input_tensor,
                request.position,
                list(request.kv_cache_keys),
            )

            # 序列化输出
            output_bytes, output_shape, output_dtype = TensorSerializer.serialize(output)

            latency_ms = (time.time() - start_time) * 1000

            return {
                "output": output_bytes,
                "shape": list(output_shape),
                "dtype": output_dtype,
                "updated_kv_keys": kv_keys,
                "success": True,
                "error_message": "",
                "latency_ms": int(latency_ms),
            }

        except Exception as e:
            logger.error(f"Forward error: {e}")
            return {
                "success": False,
                "error_message": str(e),
            }

    async def TransferKVCache(self, request, context) -> Dict[str, Any]:
        """接收 KV-Cache 传输"""
        try:
            start_time = time.time()
            total_bytes = 0

            for layer_data in request.layers:
                # 反序列化 KV
                keys = TensorSerializer.deserialize(
                    layer_data.keys,
                    tuple(layer_data.shape),
                    layer_data.dtype,
                )
                values = TensorSerializer.deserialize(
                    layer_data.values,
                    tuple(layer_data.shape),
                    layer_data.dtype,
                )

                # 存储到缓存
                cache_key = f"{request.prefix_key}:{layer_data.layer_idx}"
                await self.kv_cache._store_kv(
                    cache_key,
                    keys,
                    values,
                    layer_data.layer_idx,
                    request.prefix_key,
                )

                total_bytes += len(layer_data.keys) + len(layer_data.values)

            latency_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "error_message": "",
                "bytes_transferred": total_bytes,
                "latency_ms": int(latency_ms),
            }

        except Exception as e:
            logger.error(f"TransferKVCache error: {e}")
            return {
                "success": False,
                "error_message": str(e),
            }

    async def CreateSession(self, request, context) -> Dict[str, Any]:
        """创建会话"""
        try:
            if len(self._sessions) >= self.max_sessions:
                return {
                    "success": False,
                    "error_message": f"Max sessions reached: {self.max_sessions}",
                }

            import uuid
            session_id = str(uuid.uuid4())

            self._sessions[session_id] = {
                "session_id": session_id,
                "model_name": request.model_name,
                "max_length": request.max_length,
                "start_layer": request.start_layer,
                "end_layer": request.end_layer,
                "position": 0,
                "created_at": time.time(),
            }

            return {
                "session_id": session_id,
                "success": True,
                "error_message": "",
                "cache_tokens_available": self._get_cache_capacity(),
            }

        except Exception as e:
            logger.error(f"CreateSession error: {e}")
            return {
                "success": False,
                "error_message": str(e),
            }

    async def CloseSession(self, request, context) -> Dict[str, Any]:
        """关闭会话"""
        try:
            session = self._sessions.pop(request.session_id, None)
            if session:
                # 清理会话相关的 KV-Cache
                # TODO: 实现 KV-Cache 清理
                pass

            return {
                "success": True,
                "error_message": "",
            }

        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
            }

    async def HealthCheck(self, request, context) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = {
                "healthy": True,
                "worker_id": self.worker_id,
                "status": "online",
                "active_sessions": len(self._sessions),
            }

            if request.include_stats:
                # GPU 信息
                if torch.cuda.is_available():
                    response["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                    response["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                # KV-Cache 信息
                cache_stats = self.kv_cache.get_stats()
                response["cache_tokens_used"] = cache_stats.get("total_blocks", 0) * 16  # block_size
                response["cache_tokens_available"] = self._get_cache_capacity()

                # 性能指标
                if self._stats["total_requests"] > 0:
                    response["avg_latency_ms"] = self._stats["total_latency_ms"] / self._stats["total_requests"]
                    response["throughput_tokens_per_sec"] = (
                        self._stats["total_tokens"] / (self._stats["total_latency_ms"] / 1000)
                        if self._stats["total_latency_ms"] > 0 else 0
                    )

            return response

        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "error_message": str(e),
            }

    def _get_or_create_session(self, session_id: str, request) -> Dict[str, Any]:
        """获取或创建会话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "position": 0,
                "created_at": time.time(),
            }
        return self._sessions[session_id]

    def _create_session(self, session_id: str) -> Dict[str, Any]:
        """创建新会话"""
        session = {
            "session_id": session_id,
            "position": 0,
            "created_at": time.time(),
        }
        self._sessions[session_id] = session
        return session

    async def _forward(
        self,
        session: Dict[str, Any],
        hidden_states: torch.Tensor,
        position: int,
        kv_cache_keys: list,
    ) -> tuple:
        """执行前向传播"""
        # 这里应该调用 model_shard 的 forward 方法
        # 简化实现，实际需要处理 KV-Cache
        if self.model_shard is None:
            # 模拟输出
            return hidden_states, kv_cache_keys

        output, new_kv = self.model_shard.forward(
            hidden_states,
            position_ids=torch.arange(position, position + hidden_states.shape[1], device=hidden_states.device).unsqueeze(0),
            use_cache=True,
        )

        # 更新会话位置
        session["position"] = position + hidden_states.shape[1]

        return output, kv_cache_keys

    async def _forward_to_next(
        self,
        next_address: str,
        next_session_id: str,
        hidden_states: torch.Tensor,
        position: int,
        kv_cache_keys: list,
    ) -> None:
        """转发到下一个 Worker"""
        # TODO: 实现 server-to-server 转发
        # 这需要建立到下一个 Worker 的 gRPC 连接
        logger.debug(f"Forwarding to {next_address} (session: {next_session_id})")
        pass

    def _get_cache_capacity(self) -> int:
        """获取缓存容量"""
        stats = self.kv_cache.get_stats()
        free_blocks = stats.get("free_blocks", 0)
        return free_blocks * 16  # block_size


class GRPCServer:
    """
    gRPC 服务器

    启动和管理 gRPC 服务
    """

    def __init__(
        self,
        servicer: InferenceServicer,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ):
        if not HAS_GRPC:
            raise ImportError("grpcio not installed. Please install with: pip install grpcio")

        self.servicer = servicer
        self.host = host
        self.port = port
        self.max_workers = max_workers

        self._server: Optional[grpc_aio.Server] = None

    async def start(self) -> None:
        """启动服务器"""
        self._server = grpc_aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )

        # 注册服务
        # 注意：需要从 proto 生成的代码注册
        # 这里使用简化的 HTTP 风格服务

        listen_addr = f"{self.host}:{self.port}"
        self._server.add_insecure_port(listen_addr)

        await self._server.start()
        logger.info(f"gRPC server started on {listen_addr}")

    async def stop(self, grace: float = 5.0) -> None:
        """停止服务器"""
        if self._server:
            await self._server.stop(grace)
            logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """等待服务器终止"""
        if self._server:
            await self._server.wait_for_termination()


# HTTP 风格的 P2P API（作为 gRPC 的替代方案）
class HTTPInferenceServer:
    """
    HTTP 风格的推理服务器

    使用 FastAPI/aiohttp 实现，作为 gRPC 的简单替代
    """

    def __init__(
        self,
        servicer: InferenceServicer,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.servicer = servicer
        self.host = host
        self.port = port
        self._app = None
        self._runner = None

    async def start(self) -> None:
        """启动 HTTP 服务器"""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError("aiohttp not installed. Please install with: pip install aiohttp")

        self._app = web.Application()

        # 注册路由
        self._app.router.add_post("/inference/forward", self._handle_forward)
        self._app.router.add_post("/inference/close", self._handle_close_session)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"HTTP inference server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """停止服务器"""
        if self._runner:
            await self._runner.cleanup()
            logger.info("HTTP inference server stopped")

    async def _handle_forward(self, request) -> "web.Response":
        """处理前向传播请求"""
        from aiohttp import web
        import json

        try:
            data = await request.json()

            # 创建请求对象
            class MockRequest:
                def __init__(self, d):
                    self.session_id = d.get("session_id", "")
                    self.input = bytes.fromhex(d.get("input", {}).get("data", ""))
                    self.shape = tuple(d.get("input", {}).get("shape", []))
                    self.dtype = d.get("input", {}).get("dtype", "float16")
                    self.position = d.get("position", 0)
                    self.kv_cache_keys = d.get("kv_cache_keys", [])

            result = await self.servicer.Forward(MockRequest(data), None)

            return web.json_response(result)

        except Exception as e:
            logger.error(f"Forward error: {e}")
            return web.json_response(
                {"success": False, "error_message": str(e)},
                status=500
            )

    async def _handle_close_session(self, request) -> "web.Response":
        """处理关闭会话请求"""
        from aiohttp import web

        try:
            data = await request.json()

            class MockRequest:
                def __init__(self, d):
                    self.session_id = d.get("session_id", "")

            result = await self.servicer.CloseSession(MockRequest(data), None)
            return web.json_response(result)

        except Exception as e:
            return web.json_response(
                {"success": False, "error_message": str(e)},
                status=500
            )

    async def _handle_health(self, request) -> "web.Response":
        """处理健康检查请求"""
        from aiohttp import web

        try:
            class MockRequest:
                include_stats = True

            result = await self.servicer.HealthCheck(MockRequest(), None)
            return web.json_response(result)

        except Exception as e:
            return web.json_response(
                {"healthy": False, "error_message": str(e)},
                status=500
            )
