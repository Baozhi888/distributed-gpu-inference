"""图像生成推理引擎"""
from typing import Dict, Any
import torch
import io
import base64
import logging

from .base import BaseEngine

logger = logging.getLogger(__name__)


class ImageGenEngine(BaseEngine):
    """图像生成推理引擎"""

    def load_model(self) -> None:
        """加载图像生成模型"""
        from diffusers import DiffusionPipeline

        model_id = self.config.get("model_id", "Zhihu-ai/Z-Image-Turbo")
        logger.info(f"Loading image generation model: {model_id}")

        # 加载Pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )

        # 内存优化
        if self.config.get("enable_cpu_offload", True):
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        self.loaded = True
        logger.info("Image generation model loaded successfully")

    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像生成"""
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("steps", 4)
        seed = params.get("seed", None)

        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # 生成图像
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator
        )

        image = result.images[0]

        # 转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image_base64": image_base64,
            "width": width,
            "height": height,
            "seed": seed,
            "format": "png"
        }

    def unload_model(self) -> None:
        """卸载模型"""
        if hasattr(self, "pipe"):
            del self.pipe
        torch.cuda.empty_cache()
        self.loaded = False
        logger.info("Image generation model unloaded")
