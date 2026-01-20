"""地理位置检测服务 - 生产级实现"""
import logging
import time
from typing import Optional, Tuple
import httpx
from functools import lru_cache

logger = logging.getLogger(__name__)

# 区域映射
COUNTRY_TO_REGION = {
    # 东亚
    "CN": "asia-east", "JP": "asia-east", "KR": "asia-east",
    "TW": "asia-east", "HK": "asia-east", "MO": "asia-east",
    
    # 东南亚
    "SG": "asia-south", "TH": "asia-south", "VN": "asia-south",
    "MY": "asia-south", "ID": "asia-south", "PH": "asia-south",
    
    # 西欧
    "DE": "europe-west", "FR": "europe-west", "GB": "europe-west",
    "NL": "europe-west", "BE": "europe-west", "CH": "europe-west",
    "IT": "europe-west", "ES": "europe-west", "SE": "europe-west",
    
    # 东欧
    "PL": "europe-east", "RU": "europe-east", "UA": "europe-east",
    
    # 北美
    "US": "america-north", "CA": "america-north",
    
    # 南美
    "BR": "america-south", "AR": "america-south", "CL": "america-south",
    
    # 大洋洲
    "AU": "oceania", "NZ": "oceania",
}

# 内存缓存：IP -> (region, timestamp)
_region_cache: dict[str, Tuple[Optional[str], float]] = {}
_CACHE_TTL = 3600  # 1小时缓存
_MAX_CACHE_SIZE = 10000  # 最大缓存条目

# 共享 HTTP 客户端（复用连接）
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """获取共享的 HTTP 客户端"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=2.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    return _http_client


def _cleanup_cache():
    """清理过期缓存"""
    global _region_cache
    if len(_region_cache) > _MAX_CACHE_SIZE:
        now = time.time()
        _region_cache = {
            ip: (region, ts) 
            for ip, (region, ts) in _region_cache.items() 
            if now - ts < _CACHE_TTL
        }


async def detect_client_region(ip: Optional[str]) -> Optional[str]:
    """
    根据IP地址检测客户端区域（生产级实现）
    
    特性：
    - 内存缓存（1小时TTL）
    - 连接复用
    - 多API降级
    - 自动清理缓存
    
    Args:
        ip: 客户端IP地址
        
    Returns:
        区域代码，如 'asia-east'，检测失败返回 None
    """
    if not ip or ip in ["127.0.0.1", "localhost", "::1"]:
        return None
    
    # 检查缓存
    if ip in _region_cache:
        region, cached_time = _region_cache[ip]
        if time.time() - cached_time < _CACHE_TTL:
            return region
    
    # 尝试检测
    region = await _detect_from_api(ip)
    
    # 缓存结果（包括失败的结果，避免重复查询）
    _region_cache[ip] = (region, time.time())
    _cleanup_cache()
    
    return region


async def _detect_from_api(ip: str) -> Optional[str]:
    """从 API 检测区域（带降级）"""
    # 尝试主 API
    region = await _query_ip_api(ip)
    if region:
        return region
    
    # 降级：尝试备用 API
    region = await _query_ipinfo(ip)
    if region:
        return region
    
    logger.warning(f"All GeoIP APIs failed for {ip}")
    return None


async def _query_ip_api(ip: str) -> Optional[str]:
    """查询 ip-api.com（主API）"""
    try:
        client = _get_http_client()
        response = await client.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,countryCode"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                country_code = data.get("countryCode")
                if country_code:
                    region = COUNTRY_TO_REGION.get(country_code)
                    logger.info(f"GeoIP: {ip} -> {country_code} -> {region}")
                    return region
    except Exception as e:
        logger.debug(f"ip-api.com failed for {ip}: {e}")
    
    return None


async def _query_ipinfo(ip: str) -> Optional[str]:
    """查询 ipinfo.io（备用API）"""
    try:
        client = _get_http_client()
        response = await client.get(f"https://ipinfo.io/{ip}/json")
        
        if response.status_code == 200:
            data = response.json()
            country_code = data.get("country")
            if country_code:
                region = COUNTRY_TO_REGION.get(country_code)
                logger.info(f"GeoIP (fallback): {ip} -> {country_code} -> {region}")
                return region
    except Exception as e:
        logger.debug(f"ipinfo.io failed for {ip}: {e}")
    
    return None


@lru_cache(maxsize=128)
def get_region_info(region_code: str) -> dict:
    """获取区域信息（带缓存）"""
    regions = {
        "asia-east": {
            "name": "东亚",
            "description": "中国、日本、韩国",
            "countries": ["CN", "JP", "KR", "TW", "HK"]
        },
        "asia-south": {
            "name": "东南亚",
            "description": "新加坡、泰国、越南",
            "countries": ["SG", "TH", "VN", "MY", "ID"]
        },
        "europe-west": {
            "name": "西欧",
            "description": "德国、法国、英国",
            "countries": ["DE", "FR", "GB", "NL", "BE"]
        },
        "europe-east": {
            "name": "东欧",
            "description": "波兰、俄罗斯",
            "countries": ["PL", "RU", "UA"]
        },
        "america-north": {
            "name": "北美",
            "description": "美国、加拿大",
            "countries": ["US", "CA"]
        },
        "america-south": {
            "name": "南美",
            "description": "巴西、阿根廷",
            "countries": ["BR", "AR", "CL"]
        },
        "oceania": {
            "name": "大洋洲",
            "description": "澳大利亚、新西兰",
            "countries": ["AU", "NZ"]
        }
    }
    
    return regions.get(region_code, {
        "name": "未知",
        "description": "未知区域",
        "countries": []
    })


async def cleanup_resources():
    """清理资源（应用关闭时调用）"""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    _region_cache.clear()
