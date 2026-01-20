"""
地理位置服务 - IP到区域的映射
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# 简单的IP前缀到区域映射（生产环境应使用GeoIP数据库）
IP_REGION_MAP = {
    "1.": "asia-east",      # 中国部分IP段
    "14.": "asia-east",
    "27.": "asia-east",
    "36.": "asia-east",
    "39.": "asia-east",
    "42.": "asia-east",
    "49.": "asia-east",
    "58.": "asia-east",
    "59.": "asia-east",
    "60.": "asia-east",
    "61.": "asia-east",
    "101.": "asia-east",
    "106.": "asia-east",
    "110.": "asia-east",
    "111.": "asia-east",
    "112.": "asia-east",
    "113.": "asia-east",
    "114.": "asia-east",
    "115.": "asia-east",
    "116.": "asia-east",
    "117.": "asia-east",
    "118.": "asia-east",
    "119.": "asia-east",
    "120.": "asia-east",
    "121.": "asia-east",
    "122.": "asia-east",
    "123.": "asia-east",
    "124.": "asia-east",
    "125.": "asia-east",
    "126.": "asia-east",
    "139.": "asia-east",
    "140.": "asia-east",
    "171.": "asia-east",
    "175.": "asia-east",
    "180.": "asia-east",
    "182.": "asia-east",
    "183.": "asia-east",
    "202.": "asia-east",
    "203.": "asia-east",
    "210.": "asia-east",
    "211.": "asia-east",
    "218.": "asia-east",
    "219.": "asia-east",
    "220.": "asia-east",
    "221.": "asia-east",
    "222.": "asia-east",
    "223.": "asia-east",

    # 日本
    "133.": "asia-east",
    "150.": "asia-east",
    "153.": "asia-east",

    # 东南亚
    "103.": "asia-south",
    "128.": "asia-south",

    # 欧洲
    "2.": "europe-west",
    "5.": "europe-west",
    "31.": "europe-west",
    "37.": "europe-west",
    "46.": "europe-west",
    "51.": "europe-west",
    "62.": "europe-west",
    "77.": "europe-west",
    "78.": "europe-west",
    "79.": "europe-west",
    "80.": "europe-west",
    "81.": "europe-west",
    "82.": "europe-west",
    "83.": "europe-west",
    "84.": "europe-west",
    "85.": "europe-west",
    "86.": "europe-west",
    "87.": "europe-west",
    "88.": "europe-west",
    "89.": "europe-west",
    "90.": "europe-west",
    "91.": "europe-west",
    "92.": "europe-west",
    "93.": "europe-west",
    "94.": "europe-west",
    "95.": "europe-west",
    "109.": "europe-west",
    "176.": "europe-west",
    "178.": "europe-west",
    "185.": "europe-west",
    "188.": "europe-west",
    "193.": "europe-west",
    "194.": "europe-west",
    "195.": "europe-west",
    "212.": "europe-west",
    "213.": "europe-west",
    "217.": "europe-west",

    # 北美
    "3.": "america-north",
    "4.": "america-north",
    "8.": "america-north",
    "12.": "america-north",
    "13.": "america-north",
    "15.": "america-north",
    "16.": "america-north",
    "17.": "america-north",
    "18.": "america-north",
    "19.": "america-north",
    "20.": "america-north",
    "23.": "america-north",
    "24.": "america-north",
    "32.": "america-north",
    "34.": "america-north",
    "35.": "america-north",
    "38.": "america-north",
    "40.": "america-north",
    "44.": "america-north",
    "45.": "america-north",
    "47.": "america-north",
    "50.": "america-north",
    "52.": "america-north",
    "54.": "america-north",
    "63.": "america-north",
    "64.": "america-north",
    "65.": "america-north",
    "66.": "america-north",
    "67.": "america-north",
    "68.": "america-north",
    "69.": "america-north",
    "70.": "america-north",
    "71.": "america-north",
    "72.": "america-north",
    "73.": "america-north",
    "74.": "america-north",
    "75.": "america-north",
    "76.": "america-north",
    "96.": "america-north",
    "97.": "america-north",
    "98.": "america-north",
    "99.": "america-north",
    "100.": "america-north",
    "104.": "america-north",
    "107.": "america-north",
    "108.": "america-north",
    "129.": "america-north",
    "130.": "america-north",
    "131.": "america-north",
    "132.": "america-north",
    "134.": "america-north",
    "135.": "america-north",
    "136.": "america-north",
    "137.": "america-north",
    "138.": "america-north",
    "142.": "america-north",
    "143.": "america-north",
    "144.": "america-north",
    "146.": "america-north",
    "147.": "america-north",
    "148.": "america-north",
    "149.": "america-north",
    "152.": "america-north",
    "155.": "america-north",
    "156.": "america-north",
    "157.": "america-north",
    "158.": "america-north",
    "159.": "america-north",
    "160.": "america-north",
    "161.": "america-north",
    "162.": "america-north",
    "163.": "america-north",
    "164.": "america-north",
    "165.": "america-north",
    "166.": "america-north",
    "167.": "america-north",
    "168.": "america-north",
    "169.": "america-north",
    "170.": "america-north",
    "172.": "america-north",
    "173.": "america-north",
    "174.": "america-north",
    "184.": "america-north",
    "192.": "america-north",
    "198.": "america-north",
    "199.": "america-north",
    "204.": "america-north",
    "205.": "america-north",
    "206.": "america-north",
    "207.": "america-north",
    "208.": "america-north",
    "209.": "america-north",
    "214.": "america-north",
    "215.": "america-north",
    "216.": "america-north",

    # 南美
    "177.": "america-south",
    "179.": "america-south",
    "181.": "america-south",
    "186.": "america-south",
    "187.": "america-south",
    "189.": "america-south",
    "190.": "america-south",
    "191.": "america-south",
    "200.": "america-south",
    "201.": "america-south",

    # 大洋洲
    "1.1": "oceania",  # 澳大利亚部分
    "49.1": "oceania",
    "101.1": "oceania",
    "103.1": "oceania",
    "110.1": "oceania",
    "116.2": "oceania",
    "120.1": "oceania",
    "121.4": "oceania",
    "122.1": "oceania",
    "123.2": "oceania",
    "124.1": "oceania",
    "125.2": "oceania",
}

# 私有IP默认区域
PRIVATE_IP_PREFIXES = ["10.", "172.16.", "172.17.", "172.18.", "172.19.",
                        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                        "172.30.", "172.31.", "192.168.", "127."]


async def detect_client_region(ip: Optional[str]) -> str:
    """
    根据IP检测客户端区域

    生产环境建议使用：
    - MaxMind GeoIP2
    - IP2Location
    - Cloudflare/AWS等CDN的地理信息头
    """
    if not ip:
        return "asia-east"  # 默认区域

    # 检查是否是私有IP
    for prefix in PRIVATE_IP_PREFIXES:
        if ip.startswith(prefix):
            return "asia-east"  # 私有IP默认区域

    # 本地测试IP
    if ip == "localhost" or ip == "::1":
        return "asia-east"

    # 根据IP前缀判断
    for prefix, region in IP_REGION_MAP.items():
        if ip.startswith(prefix):
            return region

    # 默认返回
    return "asia-east"


def get_region_name(region_code: str) -> str:
    """获取区域的可读名称"""
    names = {
        "asia-east": "东亚（中国/日本/韩国）",
        "asia-south": "东南亚（新加坡/泰国）",
        "europe-west": "西欧（德国/法国/英国）",
        "europe-east": "东欧",
        "america-north": "北美（美国/加拿大）",
        "america-south": "南美",
        "oceania": "大洋洲（澳大利亚）"
    }
    return names.get(region_code, region_code)
