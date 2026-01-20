"""
Worker配置 - 支持环境变量和YAML配置
优先级: 环境变量 > config.yaml > 默认值
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path
import os


def get_env(key: str, default: Any = None, cast: type = str) -> Any:
    """获取环境变量并转换类型"""
    value = os.getenv(key)
    if value is None:
        return default

    if cast == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif cast == list:
        return [x.strip() for x in value.split(',') if x.strip()]

    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


class ServerConfig(BaseModel):
    """服务器配置"""
    url: str = Field(default_factory=lambda: get_env('GPU_SERVER_URL', 'http://localhost:8000'))
    timeout: int = Field(default_factory=lambda: get_env('GPU_SERVER_TIMEOUT', 30, int))
    verify_ssl: bool = Field(default_factory=lambda: get_env('GPU_SERVER_VERIFY_SSL', True, bool))


class GPUConfig(BaseModel):
    """GPU配置"""
    enable_cpu_offload: bool = Field(default_factory=lambda: get_env('GPU_ENABLE_CPU_OFFLOAD', True, bool))
    max_memory_gb: Optional[float] = Field(default_factory=lambda: get_env('GPU_MAX_MEMORY_GB', None, float))
    device_id: int = Field(default_factory=lambda: get_env('GPU_DEVICE_ID', 0, int))


class DirectConfig(BaseModel):
    """直连配置"""
    enabled: bool = Field(default_factory=lambda: get_env('GPU_DIRECT_ENABLED', False, bool))
    host: str = Field(default_factory=lambda: get_env('GPU_DIRECT_HOST', '0.0.0.0'))
    port: int = Field(default_factory=lambda: get_env('GPU_DIRECT_PORT', 8080, int))
    public_url: Optional[str] = Field(default_factory=lambda: get_env('GPU_DIRECT_PUBLIC_URL', None))


class LoadControlConfig(BaseModel):
    """负载控制配置"""
    acceptance_rate: float = Field(default_factory=lambda: get_env('GPU_ACCEPTANCE_RATE', 1.0, float))
    max_concurrent_jobs: int = Field(default_factory=lambda: get_env('GPU_MAX_CONCURRENT_JOBS', 1, int))
    max_jobs_per_hour: int = Field(default_factory=lambda: get_env('GPU_MAX_JOBS_PER_HOUR', 0, int))
    working_hours_start: Optional[int] = Field(default_factory=lambda: get_env('GPU_WORKING_HOURS_START', None, int))
    working_hours_end: Optional[int] = Field(default_factory=lambda: get_env('GPU_WORKING_HOURS_END', None, int))


class WorkerConfig(BaseModel):
    """Worker配置"""

    # Worker标识（首次运行后自动填充）
    worker_id: Optional[str] = Field(default_factory=lambda: get_env('GPU_WORKER_ID', None))
    token: Optional[str] = Field(default_factory=lambda: get_env('GPU_WORKER_TOKEN', None))
    name: Optional[str] = Field(default_factory=lambda: get_env('GPU_WORKER_NAME', None))

    # 地理信息
    region: str = Field(default_factory=lambda: get_env('GPU_REGION', 'asia-east'))
    country: Optional[str] = Field(default_factory=lambda: get_env('GPU_COUNTRY', None))
    city: Optional[str] = Field(default_factory=lambda: get_env('GPU_CITY', None))
    timezone: Optional[str] = Field(default_factory=lambda: get_env('GPU_TIMEZONE', None))

    # 服务器配置
    server: ServerConfig = Field(default_factory=ServerConfig)

    # GPU配置
    gpu: GPUConfig = Field(default_factory=GPUConfig)

    # 直连配置
    direct: DirectConfig = Field(default_factory=DirectConfig)

    # 负载控制
    load_control: LoadControlConfig = Field(default_factory=LoadControlConfig)

    # 支持的任务类型
    supported_types: List[str] = Field(
        default_factory=lambda: get_env('GPU_SUPPORTED_TYPES', ['llm', 'image_gen'], list)
    )

    # 引擎配置
    engines: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # 轮询配置
    heartbeat_interval: int = Field(default_factory=lambda: get_env('GPU_HEARTBEAT_INTERVAL', 30, int))
    poll_interval: int = Field(default_factory=lambda: get_env('GPU_POLL_INTERVAL', 2, int))

    def save(self, path: str = "config.yaml"):
        """保存配置到YAML文件"""
        data = self.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        """从环境变量创建配置"""
        return cls()


def load_dotenv(path: str = ".env"):
    """加载.env文件到环境变量"""
    env_path = Path(path)
    if not env_path.exists():
        return

    with open(env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 解析 KEY=VALUE
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()

                # 移除引号
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]

                # 只设置未定义的环境变量
                if key and key not in os.environ:
                    os.environ[key] = value


def load_config(path: str = "config.yaml") -> WorkerConfig:
    """
    加载配置
    优先级: 环境变量 > config.yaml > 默认值
    """
    # 首先加载.env文件
    load_dotenv()

    config_path = Path(path)

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 处理嵌套配置
        if "server" in data and isinstance(data["server"], dict):
            data["server"] = ServerConfig(**data["server"])
        if "gpu" in data and isinstance(data["gpu"], dict):
            data["gpu"] = GPUConfig(**data["gpu"])
        if "direct" in data and isinstance(data["direct"], dict):
            data["direct"] = DirectConfig(**data["direct"])
        if "load_control" in data and isinstance(data["load_control"], dict):
            data["load_control"] = LoadControlConfig(**data["load_control"])

        config = WorkerConfig(**data)
    else:
        # 仅从环境变量创建配置
        config = WorkerConfig()

    # 从环境变量加载引擎配置
    _load_engine_configs_from_env(config)

    return config


def _load_engine_configs_from_env(config: WorkerConfig):
    """从环境变量加载引擎配置"""
    env_models = {
        'llm': get_env('GPU_LLM_MODEL'),
        'image_gen': get_env('GPU_IMAGE_MODEL'),
        'vision': get_env('GPU_VISION_MODEL'),
        'whisper': get_env('GPU_WHISPER_MODEL'),
        'embedding': get_env('GPU_EMBEDDING_MODEL'),
    }

    for engine_type, model_id in env_models.items():
        if model_id:
            if engine_type not in config.engines:
                config.engines[engine_type] = {}
            config.engines[engine_type]['model_id'] = model_id


# 默认引擎配置
DEFAULT_ENGINE_CONFIGS = {
    "llm": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 2048,
        "temperature": 0.7,
    },
    "image_gen": {
        "model_id": "Zhihu-ai/Z-Image-Turbo",
        "default_steps": 4,
        "default_width": 1024,
        "default_height": 1024,
    },
    "vision": {
        "model_id": "THUDM/glm-4v-9b",
        "max_new_tokens": 1024,
    },
    "whisper": {
        "model_id": "openai/whisper-large-v3",
    },
    "embedding": {
        "model_id": "BAAI/bge-large-zh-v1.5",
    }
}
