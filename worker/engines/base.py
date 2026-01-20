"""推理引擎基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """推理引擎基类 - 所有引擎必须继承此类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """加载模型到内存"""
        pass

    @abstractmethod
    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型，释放内存"""
        pass

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        status = {
            "loaded": self.loaded,
            "device": self.device,
        }

        if torch.cuda.is_available():
            status["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }

        return status

    def _get_gpu_memory(self) -> Optional[Dict[str, float]]:
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            return {
                "used_gb": torch.cuda.memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return None
