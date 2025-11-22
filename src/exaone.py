# src/exaone.py
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


class Exaone:
    """
    Exaone wrapper (AutoModelForCausalLM)
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

        print(f"[Exaone] cache_dir = {cache_dir}")
        print(f"[Exaone] model cache dir = {model_cache_dir}")
        print(f"[Exaone] local_files_only = {self.local_files_only}")

        self._load_model()

        # tokenizer는 토큰 카운트용
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    # ----------------------------------------------------
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hg_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=self.cache_dir,
        )

    # ----------------------------------------------------
    # run 메서드
    # ----------------------------------------------------
    def run(self, prompt_or_messages, max_new_tokens=512, **kwargs):

        # TEXT-only → 자동 변환
        if isinstance(prompt_or_messages, str):
            messages = [
                {"role": "system", "content": "You are a clinical doctor."},
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        # apply_chat_template 사용
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # input_ids 길이 저장 (나중에 새로 생성된 토큰만 추출하기 위해)
        input_length = input_ids.shape[1]

        # generate 호출
        outputs = self.model.generate(
            input_ids.to(self.device),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=kwargs.get("do_sample", False),
            **{k: v for k, v in kwargs.items() if k != "do_sample"}
        )

        # input_ids 제외하고 새로 생성된 토큰만 추출
        generated_ids = outputs[0][input_length:]
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
    model = Exaone(
        hg_model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    )

    print(model.run("한국어로 대형 언어모델이 무엇인지 설명해줘.", max_new_tokens=300))
