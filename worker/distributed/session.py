"""
分布式推理会话

参考 Petals InferenceSession 设计，实现：
- 跨 Worker 的推理会话管理
- 故障检测与自动恢复
- Server-to-Server 直连传输
"""
import asyncio
import threading
from concurrent.futures import Future
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

# 本地导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common.data_structures import (
    BlockRange,
    WorkerInfo,
    InferenceState,
    SessionConfig,
    WorkerState,
)
from common.serialization import serialize_tensor, deserialize_tensor

logger = logging.getLogger(__name__)

def _run_coroutine_in_new_thread(coro):
    future: Future = Future()

    def runner() -> None:
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=runner, daemon=True).start()
    return future.result()


class SessionState(Enum):
    """会话状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class WorkerSession:
    """
    单个 Worker 的推理会话

    管理与特定 Worker 的连接和状态
    """
    worker_info: WorkerInfo
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZING

    # 会话状态
    position: int = 0
    history: Optional[Any] = None  # 用于故障恢复的输入历史

    # 下一跳会话（用于 server-to-server）
    next_session: Optional["WorkerSession"] = None

    # 连接
    _http_session: Optional[aiohttp.ClientSession] = None

    async def connect(self, timeout: float = 30.0) -> None:
        """建立与 Worker 的连接"""
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            )

        # 验证 Worker 可用性
        try:
            async with self._http_session.get(
                f"{self.worker_info.api_endpoint}/health"
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"Worker health check failed: {response.status}"
                    )
        except Exception as e:
            self.state = SessionState.ERROR
            raise ConnectionError(f"Failed to connect to worker: {e}")

        self.state = SessionState.READY
        logger.info(f"Connected to worker {self.worker_info.worker_id}")

    async def forward(
        self,
        hidden_states: Any,
        position: int,
        kv_cache_keys: List[str] = None,
    ) -> Tuple[Any, List[str]]:
        """
        执行前向传播

        Args:
            hidden_states: 输入隐藏状态 (tensor)
            position: 当前位置
            kv_cache_keys: KV-Cache 键列表

        Returns:
            (output_hidden_states, updated_kv_keys)
        """
        if self.state not in (SessionState.READY, SessionState.ACTIVE):
            raise RuntimeError(f"Session not ready: {self.state}")

        self.state = SessionState.ACTIVE

        # 序列化输入
        serialized_input = serialize_tensor(hidden_states)

        # 构建请求
        payload = {
            "session_id": self.session_id,
            "input": serialized_input,
            "position": position,
            "kv_cache_keys": kv_cache_keys or [],
            "blocks": self.worker_info.blocks.to_dict() if self.worker_info.blocks else None,
        }

        # 如果有下一跳，添加路由信息
        if self.next_session:
            payload["next_worker"] = {
                "address": self.next_session.worker_info.api_endpoint,
                "session_id": self.next_session.session_id,
            }

        # 发送请求
        try:
            async with self._http_session.post(
                f"{self.worker_info.api_endpoint}/inference/forward",
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"Forward failed: {error}")

                result = await response.json()

        except Exception as e:
            self.state = SessionState.ERROR
            raise RuntimeError(f"Forward error: {e}")

        # 反序列化输出
        output_hidden_states = deserialize_tensor(result["output"])
        updated_kv_keys = result.get("kv_cache_keys", [])

        # 更新位置
        self.position = position + hidden_states.shape[1] if hasattr(hidden_states, 'shape') else position + 1

        return output_hidden_states, updated_kv_keys

    async def close(self) -> None:
        """关闭会话"""
        if self._http_session:
            # 通知 Worker 关闭会话
            try:
                async with self._http_session.post(
                    f"{self.worker_info.api_endpoint}/inference/close",
                    json={"session_id": self.session_id}
                ) as response:
                    pass
            except Exception as e:
                logger.warning(f"Error closing worker session: {e}")

            await self._http_session.close()
            self._http_session = None

        self.state = SessionState.CLOSED

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
        else:
            _run_coroutine_in_new_thread(self.close())


class DistributedInferenceSession:
    """
    分布式推理会话

    管理跨多个 Worker 的推理会话，参考 Petals InferenceSession
    """

    def __init__(
        self,
        config: SessionConfig,
        route: List[WorkerInfo],
    ):
        """
        Args:
            config: 会话配置
            route: 推理路由（按顺序的 Worker 列表）
        """
        self.config = config
        self.route = route

        self.session_id = str(uuid.uuid4())
        self.state = SessionState.INITIALIZING

        # Worker 会话
        self._worker_sessions: List[WorkerSession] = []

        # 推理状态
        self._position = 0
        self._max_length = config.max_length

        # 统计信息
        self._stats = {
            "total_tokens": 0,
            "total_steps": 0,
            "total_latency_ms": 0,
            "retries": 0,
        }

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        self._position = value
        for session in self._worker_sessions:
            session.position = value

    async def setup(self) -> None:
        """建立与所有 Worker 的连接"""
        logger.info(f"Setting up distributed session with {len(self.route)} workers")

        try:
            for worker_info in self.route:
                session = WorkerSession(worker_info=worker_info)
                await session.connect(timeout=self.config.connect_timeout)
                self._worker_sessions.append(session)

            # 链接会话（用于 server-to-server）
            for i in range(len(self._worker_sessions) - 1):
                self._worker_sessions[i].next_session = self._worker_sessions[i + 1]

            self.state = SessionState.READY
            logger.info("Distributed session setup complete")

        except Exception as e:
            self.state = SessionState.ERROR
            # 清理已创建的会话
            for session in self._worker_sessions:
                await session.close()
            self._worker_sessions.clear()
            raise

    async def step(
        self,
        inputs: Any,
        kv_cache_keys: List[str] = None,
    ) -> Any:
        """
        执行一步推理

        Args:
            inputs: 输入 tensor
            kv_cache_keys: KV-Cache 键列表

        Returns:
            输出 tensor
        """
        if self.state not in (SessionState.READY, SessionState.ACTIVE):
            raise RuntimeError(f"Session not ready: {self.state}")

        self.state = SessionState.ACTIVE
        step_start = time.time()

        # 检查长度限制
        n_input_tokens = inputs.shape[1] if hasattr(inputs, 'shape') else 1
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: {self._position} + {n_input_tokens} > {self._max_length}"
            )

        hidden_states = inputs
        current_kv_keys = kv_cache_keys or []

        # 依次通过每个 Worker
        for i, session in enumerate(self._worker_sessions):
            for attempt in range(self.config.max_retries):
                try:
                    hidden_states, current_kv_keys = await session.forward(
                        hidden_states,
                        position=self._position,
                        kv_cache_keys=current_kv_keys,
                    )
                    break

                except Exception as e:
                    logger.warning(
                        f"Worker {session.worker_info.worker_id} failed "
                        f"(attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    self._stats["retries"] += 1

                    if attempt + 1 == self.config.max_retries:
                        # 尝试故障恢复
                        await self._handle_failure(i, e)
                        hidden_states, current_kv_keys = await session.forward(
                            hidden_states,
                            position=self._position,
                            kv_cache_keys=current_kv_keys,
                        )
                    else:
                        await asyncio.sleep(0.5 * (attempt + 1))  # 指数退避

        # 更新状态
        self._position += n_input_tokens
        self._stats["total_tokens"] += n_input_tokens
        self._stats["total_steps"] += 1
        self._stats["total_latency_ms"] += (time.time() - step_start) * 1000

        return hidden_states

    async def _handle_failure(
        self,
        failed_idx: int,
        error: Exception
    ) -> None:
        """
        处理 Worker 故障

        Args:
            failed_idx: 故障 Worker 的索引
            error: 错误信息
        """
        failed_session = self._worker_sessions[failed_idx]
        logger.error(
            f"Worker {failed_session.worker_info.worker_id} failed: {error}. "
            f"Attempting recovery..."
        )

        # 关闭故障会话
        await failed_session.close()

        # TODO: 从调度器获取替代 Worker
        # 这里需要集成调度器服务
        raise RuntimeError(
            f"Worker failure recovery not implemented. "
            f"Failed worker: {failed_session.worker_info.worker_id}"
        )

    async def close(self) -> None:
        """关闭会话"""
        for session in self._worker_sessions:
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

        self._worker_sessions.clear()
        self.state = SessionState.CLOSED
        logger.info(f"Distributed session closed. Stats: {self._stats}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        if stats["total_steps"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_steps"]
            stats["tokens_per_second"] = (
                stats["total_tokens"] / (stats["total_latency_ms"] / 1000)
                if stats["total_latency_ms"] > 0 else 0
            )
        return stats

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, *exc):
        await self.close()


class SessionManager:
    """
    会话管理器

    管理多个分布式推理会话的生命周期
    """

    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self._sessions: Dict[str, DistributedInferenceSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        config: SessionConfig,
        route: List[WorkerInfo],
    ) -> DistributedInferenceSession:
        """创建新会话"""
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                # 清理过期会话
                await self._cleanup_expired_sessions()

            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(f"Maximum sessions reached: {self.max_sessions}")

            session = DistributedInferenceSession(config, route)
            await session.setup()
            self._sessions[session.session_id] = session

            return session

    async def get_session(self, session_id: str) -> Optional[DistributedInferenceSession]:
        """获取会话"""
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """关闭会话"""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.close()

    async def _cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.state in (SessionState.CLOSED, SessionState.ERROR)
        ]
        for sid in expired:
            await self.close_session(sid)

    async def close_all(self) -> None:
        """关闭所有会话"""
        async with self._lock:
            for session in list(self._sessions.values()):
                await session.close()
            self._sessions.clear()
