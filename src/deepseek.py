# src/deepseek.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


def get_hf_cache_path(cache_dir, model_id):
    if cache_dir is None:
        return None
    if "/" not in model_id:
        return None
    org, name = model_id.split("/")
    return os.path.join(cache_dir, f"models--{org}--{name}")

class DeepSeek:
    """
    DeepSeek wrapper (AutoModelForCausalLM)
    - 캐시/오프라인 지원
    - TEXT-only 지원
    - count_tokens() 제공
    """
    def __init__(self, hg_model_id: str, device="cuda", cache_dir=None):
        self.hg_model_id = hg_model_id
        self.cache_dir = cache_dir
        self.device = device

        model_cache_dir = get_hf_cache_path(cache_dir, hg_model_id)
        self.local_files_only = os.path.exists(model_cache_dir)

        print(f"[DeepSeek] cache_dir = {cache_dir}")
        print(f"[DeepSeek] model cache dir = {model_cache_dir}")
        print(f"[DeepSeek] local_files_only = {self.local_files_only}")

        self._load_model()

        # tokenizer는 토큰 카운트용
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    # ----------------------------------------------------
    def _load_model(self):
        # flash_attention_2 미지원 환경에서는 attn_implementation="eager"를 명시적으로 지정
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hg_model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            attn_implementation="eager",
        )
        self.model.eval()

    # ----------------------------------------------------
    # run 메서드
    # ----------------------------------------------------
    def run(self, prompt_or_messages, max_new_tokens=2048, temperature=0, top_p=1.0, num_beams=1, do_sample=False, **kwargs):

        # TEXT-only → 자동 변환
        if isinstance(prompt_or_messages, str):
            messages = [
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        # apply_chat_template 사용 (fallback 포함)
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # fallback: 그냥 user 프롬프트 붙여서 토크나이즈
            input_text = prompt_or_messages if isinstance(prompt_or_messages, str) else messages[-1]["content"]
            input_ids = self.tokenizer(
                input_text,
                return_tensors="pt"
            ).input_ids.to(self.model.device)

        # generation 파라미터 설정
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "do_sample": do_sample,
        }
        # 추가 kwargs 병합
        gen_kwargs.update({k: v for k, v in kwargs.items() if k not in ["max_new_tokens", "temperature", "top_p", "num_beams", "do_sample"]})

        # generate 호출 (torch.no_grad 사용)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs,
            )

            # input_ids 제외하고 새로 생성된 토큰만 추출
            generated_ids = outputs[0][input_ids.shape[-1]:]
            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return result

    # ----------------------------------------------------
    # 토큰 카운트 (tokenizer로 계산)
    # ----------------------------------------------------
    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

# ----------------------------------------------------
# 사용 예시
# ----------------------------------------------------
if __name__ == "__main__":
    model = DeepSeek(
        hg_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    )

    print(model.run("한국어로 대형 언어모델이 무엇인지 설명해줘.", max_new_tokens=300))
