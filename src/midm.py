from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os

def get_hf_cache_path(cache_dir, model_id):
    if cache_dir is None:
        return None
    if "/" not in model_id:
        return None
    org, name = model_id.split("/")
    return os.path.join(cache_dir, f"models--{org}--{name}")

class Midm:
    """
    Midm wrapper for HF chat-style models.
    - 캐시/오프라인 지원
    - TEXT-only 및 messages 모두 지원
    - count_tokens() 제공
    """
    def __init__(self, hg_model_id: str, device="cuda", cache_dir=None):
        self.hg_model_id = hg_model_id
        self.cache_dir = cache_dir
        self.device = device

        model_cache_dir = get_hf_cache_path(cache_dir, hg_model_id)
        self.local_files_only = os.path.exists(model_cache_dir)

        print(f"[Midm] cache_dir = {cache_dir}")
        print(f"[Midm] model cache dir = {model_cache_dir}")
        print(f"[Midm] local_files_only = {self.local_files_only}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hg_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.generation_config = GenerationConfig.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    def run(self, prompt_or_messages, max_new_tokens=128, do_sample=False, **kwargs):
        # TEXT-only → 자동 변환
        if isinstance(prompt_or_messages, str):
            messages = [
                {"role": "system", "content": "You are a clinical doctor."},
                {"role": "user", "content": prompt_or_messages},
            ]
        else:
            messages = prompt_or_messages

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_ids = input_ids.to(self.device)

        output = self.model.generate(
            input_ids,
            generation_config=self.generation_config,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
        # 일반적으로 output shape: (1, 전체길이)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result.split('assistant')[-1].strip()
    
    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

if __name__ == "__main__":
    model = Midm(
        hg_model_id="K-intelligence/Midm-2.0-Mini-Instruct",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    )

    print(model.run("한국어로 대형 언어모델이 무엇인지 설명해줘.", max_new_tokens=128))
