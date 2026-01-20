"""
Worker配置服务 - 核心配置由服务端管理
Worker从服务端获取配置，确保安全和一致性
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field
import logging
import json

from app.models.models import Worker, WorkerStatus

logger = logging.getLogger(__name__)


# ==================== 配置模型 ====================

class LoadControlConfig(BaseModel):
    """负载控制配置"""
    # 任务接受率 (0.0-1.0)，1.0表示接受所有任务
    acceptance_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # 最大并发任务数
    max_concurrent_jobs: int = Field(default=1, ge=1, le=10)
    
    # 每小时最大任务数 (0=无限制)
    max_jobs_per_hour: int = Field(default=0, ge=0)
    
    # GPU显存使用上限 (0=无限制)
    max_gpu_memory_percent: float = Field(default=90.0, ge=0.0, le=100.0)
    
    # 工作时间段 (24小时制，空=全天)
    working_hours_start: Optional[int] = Field(default=None, ge=0, le=23)
    working_hours_end: Optional[int] = Field(default=None, ge=0, le=23)
    
    # 任务类型权重 (用于优先级)
    type_weights: Dict[str, float] = Field(default_factory=lambda: {
        "llm": 1.0,
        "image_gen": 1.0,
        "whisper": 1.0,
        "embedding": 1.0
    })
    
    # 冷却时间（任务完成后休息秒数）
    cooldown_seconds: int = Field(default=0, ge=0)


class SecurityConfig(BaseModel):
    """安全配置"""
    # Token有效期（小时）
    token_validity_hours: int = Field(default=24, ge=1)
    
    # 是否强制HTTPS
    require_https: bool = Field(default=True)
    
    # 是否启用请求签名
    enable_request_signing: bool = Field(default=True)
    
    # 允许的IP白名单（空=不限制）
    ip_whitelist: List[str] = Field(default_factory=list)
    
    # 是否允许直连
    allow_direct_connection: bool = Field(default=True)


class ModelConfig(BaseModel):
    """模型配置 - 由服务端统一管理"""
    model_id: str
    revision: Optional[str] = None
    
    # 模型参数
    max_new_tokens: int = 2048
    temperature: float = 0.7
    
    # 量化设置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # 其他参数
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class WorkerRemoteConfig(BaseModel):
    """Worker远程配置（从服务端获取）"""
    config_version: int = 1
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 负载控制
    load_control: LoadControlConfig = Field(default_factory=LoadControlConfig)
    
    # 安全配置
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # 支持的模型配置
    model_configs: Dict[str, ModelConfig] = Field(default_factory=dict)
    
    # 服务端消息
    server_message: Optional[str] = None
    
    # 是否需要更新
    update_required: bool = False
    update_url: Optional[str] = None


# ==================== 配置服务 ====================

class WorkerConfigService:
    """Worker配置管理服务"""
    
    # 默认模型配置
    DEFAULT_MODEL_CONFIGS = {
        "llm": ModelConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=2048,
            temperature=0.7
        ),
        "image_gen": ModelConfig(
            model_id="black-forest-labs/FLUX.1-schnell",
            extra_params={"num_inference_steps": 4}
        ),
        "whisper": ModelConfig(
            model_id="openai/whisper-large-v3"
        ),
        "embedding": ModelConfig(
            model_id="BAAI/bge-large-zh-v1.5"
        )
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_worker_config(self, worker: Worker) -> WorkerRemoteConfig:
        """获取Worker的配置"""
        # 从Worker的元数据中获取自定义配置
        custom_config = worker.config_override or {}
        
        # 构建负载控制配置
        load_control = LoadControlConfig(
            acceptance_rate=custom_config.get("acceptance_rate", 1.0),
            max_concurrent_jobs=custom_config.get("max_concurrent_jobs", 1),
            max_jobs_per_hour=custom_config.get("max_jobs_per_hour", 0),
            max_gpu_memory_percent=custom_config.get("max_gpu_memory_percent", 90.0),
            working_hours_start=custom_config.get("working_hours_start"),
            working_hours_end=custom_config.get("working_hours_end"),
            cooldown_seconds=custom_config.get("cooldown_seconds", 0)
        )
        
        # 构建安全配置
        security = SecurityConfig(
            token_validity_hours=24,
            require_https=True,
            enable_request_signing=True,
            allow_direct_connection=worker.supports_direct
        )
        
        # 获取该Worker支持的模型配置
        model_configs = {}
        for job_type in (worker.supported_types or []):
            if job_type in self.DEFAULT_MODEL_CONFIGS:
                model_configs[job_type] = self.DEFAULT_MODEL_CONFIGS[job_type]
        
        return WorkerRemoteConfig(
            load_control=load_control,
            security=security,
            model_configs=model_configs
        )
    
    async def update_worker_load_config(
        self,
        worker: Worker,
        config: LoadControlConfig
    ):
        """更新Worker的负载配置"""
        current_config = worker.config_override or {}
        
        # 更新配置
        current_config.update({
            "acceptance_rate": config.acceptance_rate,
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "max_jobs_per_hour": config.max_jobs_per_hour,
            "max_gpu_memory_percent": config.max_gpu_memory_percent,
            "working_hours_start": config.working_hours_start,
            "working_hours_end": config.working_hours_end,
            "cooldown_seconds": config.cooldown_seconds,
            "type_weights": config.type_weights
        })
        
        worker.config_override = current_config
        await self.db.commit()
        
        logger.info(f"Updated load config for worker {worker.id}")
    
    def should_accept_job(
        self,
        worker: Worker,
        job_type: str,
        current_hour: int
    ) -> tuple[bool, str]:
        """
        判断Worker是否应该接受任务
        
        Returns:
            (should_accept, reason)
        """
        config = worker.config_override or {}
        
        # 检查工作时间
        start_hour = config.get("working_hours_start")
        end_hour = config.get("working_hours_end")
        
        if start_hour is not None and end_hour is not None:
            if start_hour <= end_hour:
                # 同一天内的时间段
                if not (start_hour <= current_hour < end_hour):
                    return False, "outside_working_hours"
            else:
                # 跨天的时间段（如22:00-06:00）
                if not (current_hour >= start_hour or current_hour < end_hour):
                    return False, "outside_working_hours"
        
        # 检查接受率
        acceptance_rate = config.get("acceptance_rate", 1.0)
        if acceptance_rate < 1.0:
            import random
            if random.random() > acceptance_rate:
                return False, "rate_limited"
        
        # 检查任务类型权重
        type_weights = config.get("type_weights", {})
        if job_type in type_weights and type_weights[job_type] <= 0:
            return False, "type_disabled"
        
        return True, "accepted"
    
    async def get_hourly_job_count(self, worker: Worker) -> int:
        """获取Worker过去一小时的任务数"""
        from app.models.models import Job, JobStatus
        
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        result = await self.db.execute(
            select(Job)
            .where(
                Job.worker_id == worker.id,
                Job.started_at >= one_hour_ago
            )
        )
        
        return len(list(result.scalars().all()))
