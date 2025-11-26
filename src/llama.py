# src/llama.py
from transformers import pipeline, AutoTokenizer
import torch
import os


def get_hf_cache_path(cache_dir, model_id):
    if cache_dir is None:
        return None
    if "/" not in model_id:
        return None
    org, name = model_id.split("/")
    return os.path.join(cache_dir, f"models--{org}--{name}")


class Llama:
    """
    Llama wrapper (text-generation pipeline)
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

        print(f"[Llama] cache_dir = {cache_dir}")
        print(f"[Llama] model cache dir = {model_cache_dir}")
        print(f"[Llama] local_files_only = {self.local_files_only}")

        self._init_pipeline()

        # tokenizer는 토큰 카운트용
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    # ----------------------------------------------------
    def _init_pipeline(self):
        # cache_dir은 pipeline에서 지원하지 않으므로 제거
        # 모델 로드는 내부적으로 HuggingFace 캐시를 사용
        self.pipe = pipeline(
            "text-generation",
            model=self.hg_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # HF 모델 본체 (파라미터 수 계산용)
        self.model = self.pipe.model

    # ----------------------------------------------------
    # pipeline 스타일 run
    # ----------------------------------------------------
    def run(self, prompt_or_messages, max_new_tokens=256, **kwargs):

        # TEXT-only → 자동 변환
        if isinstance(prompt_or_messages, str):
            messages = [
                {"role": "system", "content": "You are a clinical doctor"},
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        outputs = self.pipe(
            messages,
            # max_new_tokens=max_new_tokens,
            **kwargs
        )

        try:
            return outputs[0]["generated_text"][-1]["content"]
        except:
            return outputs

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
    model = Llama(
        hg_model_id="meta-llama/Llama-3.2-1B-Instruct",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    )

    print(model.run("한국어로 대형 언어모델이 무엇인지 설명해줘.", max_new_tokens=300))
