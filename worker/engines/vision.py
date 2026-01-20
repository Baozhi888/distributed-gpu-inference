"""视觉理解推理引擎 - GLM-4V"""
from typing import Dict, Any
import torch
import base64
import io
import logging
from PIL import Image

from .base import BaseEngine

logger = logging.getLogger(__name__)


class VisionEngine(BaseEngine):
    """视觉理解推理引擎 - 支持图像识别、图像问答"""

    def load_model(self) -> None:
        """加载视觉语言模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config.get("model_id", "THUDM/glm-4v-9b")
        logger.info(f"Loading vision model: {model_id}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # 加载模型
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.config.get("enable_cpu_offload", True):
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

        self.loaded = True
        logger.info("Vision model loaded successfully")

    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行视觉理解推理

        支持的任务:
        - image_qa: 图像问答
        - image_caption: 图像描述
        - ocr: 文字识别
        """
        task = params.get("task", "image_qa")
        image_data = params.get("image_base64") or params.get("image")
        question = params.get("question", "请描述这张图片的内容")
        max_tokens = params.get("max_tokens", 1024)

        # 解码图像
        if isinstance(image_data, str):
            # Base64 编码的图像
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise ValueError("image_base64 is required")

        # 根据任务类型构建提示
        if task == "image_caption":
            prompt = "请详细描述这张图片的内容，包括场景、物体、人物、颜色等细节。"
        elif task == "ocr":
            prompt = "请识别并提取图片中的所有文字内容。"
        elif task == "image_qa":
            prompt = question
        else:
            prompt = question

        # 构建输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 应用聊天模板
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        output_length = outputs.shape[1] - input_length

        return {
            "response": response,
            "task": task,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": output_length,
                "total_tokens": input_length + output_length
            }
        }

    def unload_model(self) -> None:
        """卸载模型"""
        if self.model:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
        self.loaded = False
        logger.info("Vision model unloaded")
