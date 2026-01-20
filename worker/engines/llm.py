"""LLM推理引擎"""
from typing import Dict, Any, List
import torch
import logging

from .base import BaseEngine

logger = logging.getLogger(__name__)


class LLMEngine(BaseEngine):
    """大语言模型推理引擎"""

    def load_model(self) -> None:
        """加载LLM模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"Loading LLM model: {model_id}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # 加载模型
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        if self.config.get("enable_cpu_offload", True):
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        self.loaded = True
        logger.info("LLM model loaded successfully")

    def inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行LLM推理"""
        messages = params.get("messages", [])
        max_tokens = params.get("max_tokens", 2048)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)

        # 格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        output_length = outputs.shape[1] - input_length

        return {
            "response": response,
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
        logger.info("LLM model unloaded")
