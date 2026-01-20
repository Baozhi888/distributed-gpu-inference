"""
API客户端 - 轻量版
支持Token刷新、请求签名、远程配置获取
"""
import httpx
from typing import Optional, List, Dict, Any
import logging
import time
import hashlib
import hmac

logger = logging.getLogger(__name__)


class APIClient:
    """与中央服务器通信的客户端 - 轻量版"""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.signing_secret: Optional[str] = None
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout)

    def set_credentials(self, token: str, signing_secret: Optional[str] = None):
        """设置认证凭据"""
        self.token = token
        self.signing_secret = signing_secret

    def _headers(self, body: Optional[str] = None, path: str = "") -> dict:
        """生成请求头（含签名）"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Worker-Token"] = self.token

        # 如果有签名密钥，添加请求签名
        if self.signing_secret and body:
            timestamp = int(time.time())
            signature = self._sign_request("POST", path, body, timestamp)
            headers["X-Signature"] = signature
            headers["X-Timestamp"] = str(timestamp)

        return headers

    def _sign_request(
        self,
        method: str,
        path: str,
        body: Optional[str],
        timestamp: int
    ) -> str:
        """生成请求签名"""
        body_hash = hashlib.sha256((body or "").encode()).hexdigest()
        sign_content = f"{method.upper()}:{path}:{body_hash}:{timestamp}"

        signature = hmac.new(
            self.signing_secret.encode(),
            sign_content.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """带重试的请求"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                # 4xx错误不重试
                if 400 <= e.response.status_code < 500:
                    raise
                last_error = e
            except httpx.RequestError as e:
                last_error = e

            # 指数退避
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise last_error

    def register(
        self,
        name: str,
        region: str,
        country: Optional[str] = None,
        city: Optional[str] = None,
        timezone: Optional[str] = None,
        gpu_model: Optional[str] = None,
        gpu_memory_gb: Optional[float] = None,
        gpu_count: int = 1,
        supported_types: List[str] = None,
        direct_url: Optional[str] = None,
        supports_direct: bool = False
    ) -> dict:
        """注册Worker"""
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/api/v1/workers/register",
            json={
                "name": name,
                "region": region,
                "country": country,
                "city": city,
                "timezone": timezone,
                "gpu_model": gpu_model,
                "gpu_memory_gb": gpu_memory_gb,
                "gpu_count": gpu_count,
                "supported_types": supported_types or [],
                "direct_url": direct_url,
                "supports_direct": supports_direct
            }
        )
        return response.json()

    def heartbeat(
        self,
        worker_id: str,
        status: str,
        current_job_id: Optional[str] = None,
        gpu_memory_used_gb: Optional[float] = None,
        supported_types: Optional[List[str]] = None,
        loaded_models: Optional[List[str]] = None,
        config_version: int = 0
    ) -> dict:
        """发送心跳"""
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/api/v1/workers/{worker_id}/heartbeat",
            headers=self._headers(),
            json={
                "status": status,
                "current_job_id": current_job_id,
                "gpu_memory_used_gb": gpu_memory_used_gb,
                "supported_types": supported_types,
                "loaded_models": loaded_models,
                "config_version": config_version
            }
        )
        return response.json()

    def fetch_next_job(self, worker_id: str) -> Optional[dict]:
        """获取下一个任务"""
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/workers/{worker_id}/next-job",
                headers=self._headers(),
                timeout=10  # 短超时
            )

            if response.status_code == 204:
                return None

            if response.status_code == 200:
                data = response.json()
                return data if data else None

            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        return None

    def complete_job(
        self,
        worker_id: str,
        job_id: str,
        success: bool,
        result: Optional[dict] = None,
        error: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> dict:
        """完成任务"""
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/api/v1/workers/{worker_id}/jobs/{job_id}/complete",
            headers=self._headers(),
            json={
                "success": success,
                "result": result,
                "error": error,
                "processing_time_ms": processing_time_ms
            }
        )
        return response.json()

    def notify_going_offline(
        self,
        worker_id: str,
        finish_current: bool = True
    ) -> dict:
        """通知即将下线"""
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/api/v1/workers/{worker_id}/going-offline",
            headers=self._headers(),
            params={"finish_current": finish_current}
        )
        return response.json()

    def notify_offline(self, worker_id: str) -> dict:
        """通知已下线"""
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/api/v1/workers/{worker_id}/offline",
            headers=self._headers()
        )
        return response.json()

    def verify_credentials(self, worker_id: str, token: str) -> bool:
        """验证凭据是否有效"""
        try:
            response = self.client.post(
                f"{self.base_url}/api/v1/workers/{worker_id}/verify",
                headers={"X-Worker-Token": token},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Credential verification error: {e}")
            return False

    def get_config(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """获取远程配置"""
        try:
            response = self.client.get(
                f"{self.base_url}/api/v1/workers/{worker_id}/config",
                headers=self._headers(),
                timeout=10
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Failed to get remote config: {e}")
            return None

    def refresh_token(
        self,
        worker_id: str,
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """刷新Token"""
        try:
            response = self._request_with_retry(
                "POST",
                f"{self.base_url}/api/v1/workers/{worker_id}/refresh-token",
                headers=self._headers(),
                json={"refresh_token": refresh_token}
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None

    def close(self):
        """关闭客户端"""
        self.client.close()
