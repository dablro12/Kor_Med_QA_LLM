from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

def get_hf_cache_path(cache_dir, model_id):
    """
    HuggingFace cache format:
    <cache>/models--ORG--MODEL/
    """
    if cache_dir is None:
        return None

    if "/" not in model_id:
        return None

    org, name = model_id.split("/")
    dir_name = f"models--{org}--{name}"
    return os.path.join(cache_dir, dir_name)


class Qwen:
    def __init__(self, hg_model_id: str, device: str = "auto", cache_dir: str = None):
        self.hg_model_id = hg_model_id
        self.cache_dir = cache_dir
        self.device = device

        # -------------------------------
        # 🔥 local_files_only 자동 판단
        # -------------------------------
        model_cache_dir = get_hf_cache_path(cache_dir, hg_model_id)

        self.local_files_only = (
            os.path.exists(model_cache_dir)
            if model_cache_dir is not None else False
        )

        print(f"[Qwen] cache_dir: {self.cache_dir}")
        print(f"[Qwen] model cache dir: {model_cache_dir}")
        print(f"[Qwen] local_files_only = {self.local_files_only}")

        if "qwen3" in self.hg_model_id.lower():
            self._init_qwen3_Series()
        else:
            raise ValueError(f"Unsupported model: {self.hg_model_id}")
        
    def _init_qwen3_Series(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hg_model_id,
            dtype=torch.float16,
            device_map=self.device,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device,
            dtype=torch.float16,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir
        )
        
    def _run_qwen3_Series(self, prompt: str, max_new_tokens: int = 256, **generate_kwargs):
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # 필요시 reasoning 모드 disable/enable 가능
            enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        # take only the generated part beyond input
        output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content
        
    def run(self, prompt: str, max_new_tokens: int = 256, **generate_kwargs):
        if "qwen3" in self.hg_model_id.lower().strip():
            return self._run_qwen3_Series(prompt, max_new_tokens=max_new_tokens, **generate_kwargs)
        else:
            raise ValueError(f"Unsupported model: {self.hg_model_id}")

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False
        )
        return len(encoded["input_ids"])

# 사용 예시
if __name__ == "__main__":
    model = Qwen(hg_model_id="Qwen/Qwen3-4B-Instruct-2507", device="cuda", cache_dir="/workspace/kor_med_opendataset/hg_cache")
    response = model.run("한국어로 간단히 대형 언어모델이란 무엇인지 설명해줘.", max_new_tokens=128, temperature=0.1, top_p=0.9)
    print("Response:", response)
