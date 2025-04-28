from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
import torch

logger = logging.getLogger(__name__)

class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.cache_dir = "/models"

        try:
            logger.info(f"📦 Пытаемся загрузить модель {self.model_name} в {self.cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            logger.info("✅ Модель успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise e

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
        )

        new_tokens = output_ids[:, inputs.input_ids.shape[-1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return response


qwen_model = QwenModel()
