"""
Python SDK - 增强版
支持：多区域、直连、重试、降级
"""
import httpx
from typing import Optional, List, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


class InferenceClient:
    """
    分布式GPU推理客户端SDK - 增强版

    Features:
    - 多服务器支持（主服务器 + 备用服务器）
    - 自动重试和降级
    - 直连模式支持
    - 区域选择
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        fallback_urls: List[str] = None
    ):
        """
        初始化客户端

        Args:
            base_url: 主服务器URL
            api_key: API密钥（可选）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            fallback_urls: 备用服务器URL列表
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_urls = fallback_urls or []
        self.client = httpx.Client(timeout=timeout)

        # 直连缓存
        self._direct_worker_cache: Dict[str, dict] = {}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request_with_fallback(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """带降级的请求"""
        urls = [self.base_url] + self.fallback_urls
        last_error = None

        for url in urls:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.request(
                        method,
                        f"{url}{path}",
                        **kwargs
                    )
                    response.raise_for_status()
                    return response

                except httpx.TimeoutException as e:
                    logger.warning(f"Timeout on {url}, attempt {attempt + 1}")
                    last_error = e

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 503:
                        logger.warning(f"Service unavailable on {url}")
                        last_error = e
                        break  # 尝试下一个服务器
                    elif 400 <= e.response.status_code < 500:
                        raise  # 客户端错误，不重试
                    last_error = e

                except httpx.RequestError as e:
                    logger.warning(f"Request error on {url}: {e}")
                    last_error = e

                # 重试前等待
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise last_error or Exception("All servers unavailable")

    # ==================== 聊天接口 ====================

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        region: Optional[str] = None,
        sync: bool = True,
        timeout: Optional[int] = None,
        use_direct: bool = False
    ) -> Dict[str, Any]:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user", "content": "Hello"}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            region: 指定区域（可选）
            sync: 是否同步等待结果
            timeout: 超时时间
            use_direct: 是否使用直连模式

        Returns:
            包含response的字典
        """
        # 直连模式
        if use_direct:
            return self._direct_inference(
                job_type="llm",
                params={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )

        endpoint = "/api/v1/jobs/sync" if sync else "/api/v1/jobs"

        response = self._request_with_fallback(
            "POST",
            endpoint,
            headers=self._headers(),
            json={
                "type": "llm",
                "params": {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                "region": region,
                "timeout_seconds": timeout or self.timeout
            },
            timeout=timeout or self.timeout
        )

        return response.json()

    # ==================== 图像生成接口 ====================

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: Optional[int] = None,
        region: Optional[str] = None,
        sync: bool = True,
        timeout: Optional[int] = None,
        use_direct: bool = False
    ) -> Dict[str, Any]:
        """
        生成图像

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图像宽度
            height: 图像高度
            steps: 推理步数
            seed: 随机种子
            region: 指定区域
            sync: 是否同步等待结果
            timeout: 超时时间
            use_direct: 是否使用直连模式

        Returns:
            包含image_base64的字典
        """
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed
        }

        if use_direct:
            return self._direct_inference(job_type="image_gen", params=params)

        endpoint = "/api/v1/jobs/sync" if sync else "/api/v1/jobs"

        response = self._request_with_fallback(
            "POST",
            endpoint,
            headers=self._headers(),
            json={
                "type": "image_gen",
                "params": params,
                "region": region,
                "timeout_seconds": timeout or 300
            },
            timeout=timeout or 300
        )

        return response.json()

    # ==================== 异步任务接口 ====================

    def create_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        priority: int = 0,
        region: Optional[str] = None,
        timeout_seconds: int = 300
    ) -> str:
        """
        创建异步任务

        Returns:
            任务ID
        """
        response = self._request_with_fallback(
            "POST",
            "/api/v1/jobs",
            headers=self._headers(),
            json={
                "type": job_type,
                "params": params,
                "priority": priority,
                "region": region,
                "timeout_seconds": timeout_seconds
            }
        )
        return response.json()["job_id"]

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """查询任务状态"""
        response = self._request_with_fallback(
            "GET",
            f"/api/v1/jobs/{job_id}",
            headers=self._headers()
        )
        return response.json()

    def wait_for_job(
        self,
        job_id: str,
        timeout: int = 300,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = time.time()

        while True:
            result = self.get_job(job_id)

            if result["status"] in ["completed", "failed"]:
                return result

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out")

            time.sleep(poll_interval)

    # ==================== 直连模式 ====================

    def _get_nearest_worker(self, job_type: str) -> dict:
        """获取最近的支持直连的Worker"""
        cache_key = f"direct_{job_type}"

        # 检查缓存
        if cache_key in self._direct_worker_cache:
            cached = self._direct_worker_cache[cache_key]
            if time.time() - cached["cached_at"] < 60:  # 1分钟缓存
                return cached["worker"]

        response = self._request_with_fallback(
            "GET",
            f"/api/v1/jobs/direct/nearest?job_type={job_type}",
            headers=self._headers()
        )

        worker = response.json()
        self._direct_worker_cache[cache_key] = {
            "worker": worker,
            "cached_at": time.time()
        }

        return worker

    def _direct_inference(
        self,
        job_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """直连推理"""
        worker = self._get_nearest_worker(job_type)
        direct_url = worker["direct_url"]

        logger.info(f"Using direct connection to {worker['region']}")

        response = self.client.post(
            f"{direct_url}/inference",
            json={
                "type": job_type,
                "params": params
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        return response.json()

    # ==================== 工具方法 ====================

    def get_queue_stats(self, region: Optional[str] = None) -> Dict[str, Any]:
        """获取队列统计"""
        params = {}
        if region:
            params["region"] = region

        response = self._request_with_fallback(
            "GET",
            "/api/v1/jobs/stats/queue",
            headers=self._headers(),
            params=params
        )
        return response.json()

    def list_workers(
        self,
        region: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出Workers"""
        params = {}
        if region:
            params["region"] = region
        if status:
            params["status"] = status

        response = self._request_with_fallback(
            "GET",
            "/api/v1/workers",
            headers=self._headers(),
            params=params
        )
        return response.json()

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 便捷函数 ====================

def chat(
    base_url: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """快速聊天接口"""
    with InferenceClient(base_url, api_key) as client:
        return client.chat(messages, **kwargs)


def generate_image(
    base_url: str,
    prompt: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """快速图像生成接口"""
    with InferenceClient(base_url, api_key) as client:
        return client.generate_image(prompt, **kwargs)
