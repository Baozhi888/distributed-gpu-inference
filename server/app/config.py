"""服务端配置"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List


class Settings(BaseSettings):
    # 应用配置
    app_name: str = "Distributed GPU Inference"
    debug: bool = False

    @field_validator('debug', mode='before')
    @classmethod
    def parse_debug(cls, v):
        """解析debug字段，支持多种格式"""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return bool(v)

    # 区域配置
    region: str = "asia-east"  # 当前服务器所在区域

    # 数据库
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/inference"
    redis_url: str = "redis://localhost:6379/0"

    # 安全
    secret_key: str = "change-me-in-production"
    api_key_header: str = "X-API-Key"
    worker_token_header: str = "X-Worker-Token"

    # Worker配置
    heartbeat_timeout_seconds: int = 90  # Worker心跳超时（增加到90秒以应对网络抖动）
    job_timeout_seconds: int = 300  # 任务执行超时
    stale_job_check_interval: int = 30  # 检查超时任务的间隔（秒）

    # 限流
    rate_limit_per_minute: int = 60

    # 多区域配置
    enable_cross_region: bool = True  # 是否启用跨区域任务分配
    cross_region_penalty: float = 0.3  # 跨区域调度的权重惩罚（0-1）

    class Config:
        env_file = ".env"
        env_prefix = ""  # 允许环境变量覆盖


settings = Settings()
