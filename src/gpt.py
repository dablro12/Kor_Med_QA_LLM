from transformers import pipeline
import torch


class GPT:
    def __init__(self, hg_model_id: str, device: str = "auto", cache_dir: str = None):
        self.hg_model_id = hg_model_id
        self.cache_dir = cache_dir
        self.device = device
        
        if self.hg_model_id == "gpt-oss-20b":
            self._init_gpt_oss_20b()
        else:
            raise ValueError(f"Unsupported model: {self.hg_model_id}")
        
    def _init_gpt_oss_20b(self):
        model_id = "openai/gpt-oss-20b"

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map=self.device,
            cache_dir= self.cache_dir
        )
        
    def _run_gpt_oss_20b(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        outputs = self.pipe(
            messages,
            # max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]
        
    def run(self, prompt: str):
        if self.hg_model_id == "gpt-oss-20b":
            return self._run_gpt_oss_20b(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.hg_model_id}")