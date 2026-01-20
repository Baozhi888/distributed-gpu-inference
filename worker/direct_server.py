"""
直连服务器 - 允许客户端直接与Worker通信
跳过中央服务器，降低延迟
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
from threading import Thread

logger = logging.getLogger(__name__)


class DirectInferenceRequest(BaseModel):
    """直连推理请求"""
    type: str
    params: Dict[str, Any]
    timeout_seconds: int = 300


class DirectInferenceResponse(BaseModel):
    """直连推理响应"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int = 0


class DirectServer:
    """
    直连服务器

    允许客户端绕过中央服务器，直接与Worker通信
    适用于低延迟场景
    """

    def __init__(self, worker, host: str = "0.0.0.0", port: int = 8080):
        self.worker = worker
        self.host = host
        self.port = port
        self.app = FastAPI(title="Worker Direct API")
        self._setup_routes()
        self.server = None

    def _setup_routes(self):
        """设置路由"""

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "worker_id": self.worker.worker_id,
                "worker_status": self.worker.status,
                "supported_types": list(self.worker.engines.keys())
            }

        @self.app.get("/status")
        async def status():
            gpu_info = self.worker._get_gpu_info()
            return {
                "worker_id": self.worker.worker_id,
                "status": self.worker.status,
                "current_job": self.worker.current_job_id,
                "supported_types": list(self.worker.engines.keys()),
                "gpu_info": gpu_info,
                "accepting_jobs": self.worker.accepting_jobs
            }

        @self.app.post("/inference", response_model=DirectInferenceResponse)
        async def direct_inference(request: DirectInferenceRequest):
            """
            直连推理接口

            客户端可以直接调用此接口进行推理，跳过中央服务器
            """
            import time

            # 检查是否接受任务
            if not self.worker.accepting_jobs:
                raise HTTPException(503, "Worker is going offline")

            # 检查是否空闲
            if self.worker.status != "idle":
                raise HTTPException(503, "Worker is busy")

            # 检查引擎
            engine = self.worker.engines.get(request.type)
            if not engine:
                raise HTTPException(
                    400,
                    f"Unsupported type: {request.type}. "
                    f"Supported: {list(self.worker.engines.keys())}"
                )

            # 标记为忙碌
            self.worker.status = "busy"

            try:
                start_time = time.time()
                result = engine.inference(request.params)
                processing_time_ms = int((time.time() - start_time) * 1000)

                return DirectInferenceResponse(
                    success=True,
                    result=result,
                    processing_time_ms=processing_time_ms
                )

            except Exception as e:
                logger.error(f"Direct inference error: {e}")
                return DirectInferenceResponse(
                    success=False,
                    error=str(e)
                )

            finally:
                self.worker.status = "idle"

    def start(self):
        """启动服务器（阻塞）"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def start_background(self):
        """后台启动服务器"""
        thread = Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.should_exit = True
