"""
Worker主程序 - 轻量版
核心逻辑在服务端，Worker仅负责：
- 注册和心跳
- 执行推理任务
- 汇报状态
"""
import time
import signal
import logging
from threading import Thread, Event
from typing import Optional, Dict, Any, List
from datetime import datetime
import torch
import sys

from config import WorkerConfig, load_config
from api_client import APIClient
from engines import ENGINE_REGISTRY, BaseEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Worker:
    """
    分布式GPU推理Worker - 轻量版

    核心原则：
    - Worker是"哑终端"，服务端做决策
    - 配置从服务端获取
    - 负载控制由服务端判断
    """

    def __init__(self, config: WorkerConfig = None):
        self.config = config or load_config()
        self.api_client: Optional[APIClient] = None
        self.engines: Dict[str, BaseEngine] = {}

        # 认证信息
        self.worker_id: Optional[str] = self.config.worker_id
        self.token: Optional[str] = self.config.token
        self.refresh_token: Optional[str] = None
        self.signing_secret: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

        # 状态
        self.status = "initializing"
        self.current_job_id: Optional[str] = None
        self.running = False
        self.accepting_jobs = True

        # 服务端配置（从服务器获取）
        self.remote_config: Dict[str, Any] = {}
        self.config_version = 0

        # 用于优雅关闭
        self.shutdown_event = Event()

        # 直连服务器（可选）
        self.direct_server = None

    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "model": torch.cuda.get_device_name(0),
                "memory_total_gb": round(props.total_memory / 1024**3, 2),
                "memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "gpu_count": torch.cuda.device_count()
            }
        return {
            "model": "CPU Only",
            "memory_total_gb": 0,
            "memory_used_gb": 0,
            "gpu_count": 0
        }

    def _register(self):
        """注册Worker到服务器"""
        if self.worker_id and self.token:
            logger.info(f"Using existing worker ID: {self.worker_id}")
            self.api_client.set_credentials(self.token, self.signing_secret)

            # 验证凭据是否有效
            if not self._verify_credentials():
                logger.warning("Existing credentials invalid, re-registering...")
                self.worker_id = None
                self.token = None

        if not self.worker_id:
            self._do_register()

        # 获取远程配置
        self._fetch_remote_config()

    def _do_register(self):
        """执行注册"""
        gpu_info = self._get_gpu_info()

        # 构建直连URL
        direct_url = None
        if self.config.direct.enabled and self.config.direct.public_url:
            direct_url = self.config.direct.public_url

        result = self.api_client.register(
            name=self.config.name or f"Worker-{gpu_info.get('model', 'Unknown')[:20]}",
            region=self.config.region,
            country=self.config.country,
            city=self.config.city,
            timezone=self.config.timezone,
            gpu_model=gpu_info.get("model"),
            gpu_memory_gb=gpu_info.get("memory_total_gb"),
            gpu_count=gpu_info.get("gpu_count", 1),
            supported_types=self.config.supported_types,
            direct_url=direct_url,
            supports_direct=self.config.direct.enabled
        )

        # 保存认证信息
        self.worker_id = result["worker_id"]
        self.token = result["token"]
        self.refresh_token = result.get("refresh_token")
        self.signing_secret = result.get("signing_secret")

        if result.get("token_expires_at"):
            self.token_expires_at = datetime.fromisoformat(result["token_expires_at"])

        # 保存到配置文件
        self.config.worker_id = self.worker_id
        self.config.token = self.token
        self.config.save()

        # 更新API客户端凭据
        self.api_client.set_credentials(self.token, self.signing_secret)

        logger.info(f"Registered as worker: {self.worker_id}")

    def _verify_credentials(self) -> bool:
        """验证当前凭据是否有效"""
        try:
            return self.api_client.verify_credentials(self.worker_id, self.token)
        except Exception as e:
            logger.error(f"Credential verification failed: {e}")
            return False

    def _fetch_remote_config(self):
        """从服务端获取配置"""
        try:
            config = self.api_client.get_config(self.worker_id)

            if config and config.get("version", 0) > self.config_version:
                self.remote_config = config
                self.config_version = config.get("version", 0)
                logger.info(f"Remote config updated to version {self.config_version}")

                # 应用负载控制配置
                self._apply_load_control(config.get("load_control", {}))

        except Exception as e:
            logger.warning(f"Failed to fetch remote config: {e}")

    def _apply_load_control(self, load_control: Dict[str, Any]):
        """应用负载控制配置"""
        # 检查工作时间
        if "working_hours_start" in load_control and "working_hours_end" in load_control:
            current_hour = datetime.now().hour
            start = load_control["working_hours_start"]
            end = load_control["working_hours_end"]

            if start <= end:
                in_working_hours = start <= current_hour < end
            else:
                in_working_hours = current_hour >= start or current_hour < end

            if not in_working_hours:
                self.accepting_jobs = False
                logger.info("Outside working hours, not accepting jobs")

    def _should_accept_job(self) -> bool:
        """检查是否应该接受任务（服务端决策为主）"""
        if not self.accepting_jobs:
            return False

        # 服务端的负载控制配置
        load_control = self.remote_config.get("load_control", {})

        # 检查工作时间
        if "working_hours_start" in load_control:
            current_hour = datetime.now().hour
            start = load_control.get("working_hours_start", 0)
            end = load_control.get("working_hours_end", 24)

            if start <= end:
                if not (start <= current_hour < end):
                    return False
            else:
                if not (current_hour >= start or current_hour < end):
                    return False

        return True

    def _refresh_token_if_needed(self):
        """检查并刷新Token"""
        if not self.token_expires_at or not self.refresh_token:
            return

        # 提前4小时刷新
        refresh_threshold = datetime.utcnow()
        hours_until_expiry = (self.token_expires_at - refresh_threshold).total_seconds() / 3600

        if hours_until_expiry < 4:
            try:
                result = self.api_client.refresh_token(
                    self.worker_id, self.refresh_token
                )

                if result:
                    self.token = result["token"]
                    self.refresh_token = result.get("refresh_token")
                    if result.get("token_expires_at"):
                        self.token_expires_at = datetime.fromisoformat(result["token_expires_at"])

                    self.api_client.set_credentials(self.token, self.signing_secret)
                    logger.info("Token refreshed successfully")

            except Exception as e:
                logger.error(f"Token refresh failed: {e}")

    def _load_engines(self):
        """加载推理引擎"""
        # 优先使用服务端配置的模型
        model_configs = self.remote_config.get("model_configs", {})

        for engine_type in self.config.supported_types:
            if engine_type not in ENGINE_REGISTRY:
                logger.warning(f"Unknown engine type: {engine_type}, skipping")
                continue

            # 合并本地和远程配置
            engine_config = self.config.engines.get(engine_type, {})
            if engine_type in model_configs:
                engine_config.update(model_configs[engine_type])

            engine_config["enable_cpu_offload"] = self.config.gpu.enable_cpu_offload

            logger.info(f"Loading engine: {engine_type}")

            try:
                engine = ENGINE_REGISTRY[engine_type](engine_config)
                engine.load_model()
                self.engines[engine_type] = engine
                logger.info(f"Engine {engine_type} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load engine {engine_type}: {e}")
                if engine_type in self.config.supported_types:
                    self.config.supported_types.remove(engine_type)

    def _heartbeat_loop(self):
        """心跳循环"""
        heartbeat_count = 0

        while self.running and not self.shutdown_event.is_set():
            try:
                gpu_info = self._get_gpu_info()

                # 确定状态
                if not self.accepting_jobs:
                    status = "going_offline"
                elif self.current_job_id:
                    status = "busy"
                else:
                    status = "online"

                response = self.api_client.heartbeat(
                    worker_id=self.worker_id,
                    status=status,
                    current_job_id=self.current_job_id,
                    gpu_memory_used_gb=gpu_info.get("memory_used_gb"),
                    supported_types=list(self.engines.keys()),
                    loaded_models=self._get_loaded_models(),
                    config_version=self.config_version
                )

                # 处理服务器响应
                if response:
                    # 配置更新
                    if response.get("config_changed"):
                        self._fetch_remote_config()

                    # 服务器指令
                    action = response.get("action")
                    if action == "shutdown":
                        logger.info("Received shutdown command from server")
                        self.request_shutdown()
                    elif action == "reload_config":
                        self._fetch_remote_config()

                # 定期刷新Token
                heartbeat_count += 1
                if heartbeat_count % 10 == 0:
                    self._refresh_token_if_needed()

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            self.shutdown_event.wait(timeout=self.config.heartbeat_interval)

    def _main_loop(self):
        """任务处理主循环"""
        while self.running:
            if not self.accepting_jobs and not self.current_job_id:
                logger.info("No more jobs to process, shutting down")
                break

            # 检查是否应该接受任务
            if self._should_accept_job() and self.status == "idle":
                try:
                    job = self.api_client.fetch_next_job(self.worker_id)

                    if job:
                        self._process_job(job)

                except Exception as e:
                    logger.error(f"Error fetching job: {e}")

            if self.shutdown_event.wait(timeout=self.config.poll_interval):
                if not self.current_job_id:
                    break

    def _process_job(self, job: Dict[str, Any]):
        """处理单个任务"""
        job_id = job["job_id"]
        job_type = job["type"]
        params = job["params"]

        logger.info(f"Processing job {job_id} (type: {job_type})")

        self.status = "busy"
        self.current_job_id = job_id

        try:
            engine = self.engines.get(job_type)
            if not engine:
                raise ValueError(f"No engine for type: {job_type}")

            start_time = time.time()
            result = engine.inference(params)
            processing_time_ms = int((time.time() - start_time) * 1000)

            self.api_client.complete_job(
                worker_id=self.worker_id,
                job_id=job_id,
                success=True,
                result=result,
                processing_time_ms=processing_time_ms
            )

            logger.info(f"Job {job_id} completed in {processing_time_ms}ms")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            self.api_client.complete_job(
                worker_id=self.worker_id,
                job_id=job_id,
                success=False,
                error=str(e)
            )

        finally:
            self.status = "idle"
            self.current_job_id = None

    def _get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        models = []
        for engine_type, engine in self.engines.items():
            if hasattr(engine, 'config') and 'model_id' in engine.config:
                models.append(engine.config['model_id'])
        return models

    def _start_direct_server(self):
        """启动直连服务器（可选）"""
        if not self.config.direct.enabled:
            return

        from direct_server import DirectServer

        self.direct_server = DirectServer(
            worker=self,
            host=self.config.direct.host,
            port=self.config.direct.port
        )

        direct_thread = Thread(target=self.direct_server.start, daemon=True)
        direct_thread.start()

        logger.info(f"Direct server started on {self.config.direct.host}:{self.config.direct.port}")

    def start(self):
        """启动Worker"""
        logger.info("=" * 50)
        logger.info("Starting GPU Worker (Lightweight)")
        logger.info("=" * 50)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.api_client = APIClient(
            base_url=self.config.server.url,
            token=self.token,
            timeout=self.config.server.timeout
        )

        self._register()
        self._load_engines()

        if not self.engines:
            logger.error("No engines loaded, exiting")
            return

        self.running = True
        self.status = "idle"

        self._start_direct_server()

        heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        logger.info(f"Worker {self.worker_id} started")
        logger.info(f"Supported types: {list(self.engines.keys())}")

        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            self.shutdown()

    def request_shutdown(self, graceful: bool = True):
        """请求关闭Worker"""
        logger.info(f"Shutdown requested (graceful={graceful})")

        if graceful:
            self.accepting_jobs = False

            try:
                self.api_client.notify_going_offline(
                    self.worker_id,
                    finish_current=True
                )
            except Exception as e:
                logger.error(f"Failed to notify server: {e}")

            if not self.current_job_id:
                self.shutdown_event.set()
        else:
            self.shutdown_event.set()
            self.running = False

    def shutdown(self):
        """关闭Worker"""
        logger.info("Shutting down worker...")
        self.running = False
        self.shutdown_event.set()

        try:
            if self.api_client and self.worker_id:
                self.api_client.notify_offline(self.worker_id)
        except Exception as e:
            logger.error(f"Failed to notify offline: {e}")

        if self.direct_server:
            self.direct_server.stop()

        for name, engine in self.engines.items():
            try:
                engine.unload_model()
                logger.info(f"Engine {name} unloaded")
            except Exception as e:
                logger.error(f"Error unloading engine {name}: {e}")

        if self.api_client:
            self.api_client.close()

        logger.info("Worker shutdown complete")

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"Received signal {signum}")
        self.request_shutdown(graceful=True)


def main():
    """主入口 - 由CLI调用"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Worker")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--region", "-r", help="Override region")
    parser.add_argument("--server", "-s", help="Override server URL")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.region:
        config.region = args.region
    if args.server:
        config.server.url = args.server

    worker = Worker(config)
    worker.start()


if __name__ == "__main__":
    main()
