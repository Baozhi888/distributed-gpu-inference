"""引擎模块

支持多种推理后端：
- llm: 原生 Transformers 后端（兼容性好）
- llm_sglang: SGLang 高性能后端（推荐，RadixAttention）
- llm_vllm: vLLM 高性能后端（PagedAttention）
- llm_vllm_async: vLLM 异步引擎（支持流式）
- image_gen: 图像生成引擎
- vision: 视觉模型引擎

使用示例：
    # 方式1: 直接使用引擎类
    from engines import LLMEngine
    engine = LLMEngine(config)

    # 方式2: 通过配置选择后端
    from engines import create_llm_engine
    engine = create_llm_engine({"backend": "sglang", "model_id": "..."})

    # 方式3: 通过类型名获取
    from engines import get_engine
    EngineClass = get_engine("llm_sglang")
    engine = EngineClass(config)
"""
from typing import Dict, Any, Optional

from .base import BaseEngine
from .llm import LLMEngine
from .llm_base import LLMBaseEngine, LLMBackend, GenerationConfig, GenerationResult
from .image_gen import ImageGenEngine
from .vision import VisionEngine


# 延迟导入高性能引擎（可能需要额外依赖）
def _get_sglang_engine():
    from .llm_sglang import SGLangEngine
    return SGLangEngine


def _get_vllm_engine():
    from .llm_vllm import VLLMEngine
    return VLLMEngine


def _get_vllm_async_engine():
    from .llm_vllm import VLLMAsyncEngine
    return VLLMAsyncEngine


# 引擎注册表
ENGINE_REGISTRY = {
    # 原生后端
    "llm": LLMEngine,
    "image_gen": ImageGenEngine,
    "vision": VisionEngine,
}

# 高性能后端（延迟注册）
_LAZY_ENGINES = {
    "llm_sglang": _get_sglang_engine,
    "llm_vllm": _get_vllm_engine,
    "llm_vllm_async": _get_vllm_async_engine,
}

# 后端别名映射
_BACKEND_ALIASES = {
    "native": "llm",
    "transformers": "llm",
    "sglang": "llm_sglang",
    "vllm": "llm_vllm",
    "vllm_async": "llm_vllm_async",
}


def get_engine(engine_type: str) -> type:
    """
    获取引擎类

    Args:
        engine_type: 引擎类型名称

    Returns:
        引擎类

    Raises:
        ValueError: 未知的引擎类型
        ImportError: 引擎依赖未安装
    """
    # 处理别名
    engine_type = _BACKEND_ALIASES.get(engine_type, engine_type)

    if engine_type in ENGINE_REGISTRY:
        return ENGINE_REGISTRY[engine_type]

    if engine_type in _LAZY_ENGINES:
        try:
            engine_class = _LAZY_ENGINES[engine_type]()
            ENGINE_REGISTRY[engine_type] = engine_class  # 缓存
            return engine_class
        except ImportError as e:
            raise ImportError(
                f"Engine '{engine_type}' requires additional dependencies: {e}"
            )

    raise ValueError(f"Unknown engine type: {engine_type}")


def create_llm_engine(config: Dict[str, Any]) -> LLMBaseEngine:
    """
    根据配置创建 LLM 引擎

    这是创建 LLM 引擎的推荐方式，会根据配置中的 backend 字段
    自动选择合适的引擎实现。

    Args:
        config: 引擎配置，应包含：
            - backend: 后端类型 ("native", "sglang", "vllm", "vllm_async")
            - model_id: 模型 ID
            - 其他后端特定配置

    Returns:
        LLM 引擎实例

    示例:
        config = {
            "backend": "sglang",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "sglang": {
                "tp_size": 1,
                "mem_fraction_static": 0.85,
                "enable_prefix_caching": True,
            }
        }
        engine = create_llm_engine(config)
    """
    backend = config.get("backend", "native").lower()

    # 获取引擎类型
    engine_type = _BACKEND_ALIASES.get(backend, backend)

    # 验证是 LLM 引擎
    if not engine_type.startswith("llm"):
        raise ValueError(f"'{backend}' is not a valid LLM backend")

    # 获取引擎类
    engine_class = get_engine(engine_type)

    # 创建实例
    return engine_class(config)


def list_engines() -> dict:
    """列出所有可用引擎及其状态"""
    engines = {}

    # 已注册引擎
    for name in ENGINE_REGISTRY:
        engines[name] = {"available": True, "loaded": True}

    # 延迟加载引擎
    for name, loader in _LAZY_ENGINES.items():
        if name not in engines:
            try:
                loader()
                engines[name] = {"available": True, "loaded": False}
            except ImportError as e:
                engines[name] = {"available": False, "error": str(e)}

    return engines


def get_recommended_backend() -> str:
    """
    获取推荐的 LLM 后端

    按优先级尝试：SGLang > vLLM > Native
    """
    # 优先尝试 SGLang
    try:
        _get_sglang_engine()
        return "sglang"
    except ImportError:
        pass

    # 其次尝试 vLLM
    try:
        _get_vllm_engine()
        return "vllm"
    except ImportError:
        pass

    # 回退到原生
    return "native"


__all__ = [
    # 基类
    "BaseEngine",
    "LLMBaseEngine",
    "LLMBackend",
    "GenerationConfig",
    "GenerationResult",

    # 具体引擎
    "LLMEngine",
    "ImageGenEngine",
    "VisionEngine",

    # 工厂和注册
    "ENGINE_REGISTRY",
    "get_engine",
    "create_llm_engine",
    "list_engines",
    "get_recommended_backend",
]
