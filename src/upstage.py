from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

def get_hf_cache_path(cache_dir, model_id):
    if cache_dir is None:
        return None
    if "/" not in model_id:
        return None
    org, name = model_id.split("/")
    return os.path.join(cache_dir, f"models--{org}--{name}")

def upstage_response_parser(response):
    """
    파싱에 실패하지 않고, 여러 JSON 블록이 있으면 가장 마지막 JSON 블록만 반환
    """
    # 모든 { ... } 블록 추출
    json_blocks = re.findall(r'(\{[\s\S]*?\})', response)
    if json_blocks:
        # 여러 블록 중 가장 마지막(올바른 answer/explanation 포함될 가능성 높음) 사용
        preprocess_response = json_blocks[-1]
    else:
        # 못 찾으면 전체 응답을 반환
        preprocess_response = response.strip()
    return preprocess_response


class Upstage:
    """
    Upstage SOLAR wrapper (AutoModelForCausalLM)
    - 캐시/오프라인 지원
    - TEXT-only 지원
    - count_tokens() 제공
    """

    def __init__(
        self,
        hg_model_id: str = "Upstage/SOLAR-10.7B-Instruct-v1.0",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    ):
        self.hg_model_id = hg_model_id
        self.cache_dir = cache_dir
        self.device = device

        model_cache_dir = get_hf_cache_path(cache_dir, hg_model_id)
        self.local_files_only = os.path.exists(model_cache_dir)

        print(f"[Upstage] cache_dir = {cache_dir}")
        print(f"[Upstage] model cache dir = {model_cache_dir}")
        print(f"[Upstage] local_files_only = {self.local_files_only}")

        self._load_model()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hg_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    def _load_model(self):
        # Upstage SOLAR 권장: float16, device_map/cuda 활용
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hg_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir,
        )

    def run(self, prompt_or_messages, max_new_tokens=512, **kwargs):
        # prompt_or_messages: str(프롬프트) or list(dict)(대화형)
        if isinstance(prompt_or_messages, str):
            # Upstage는 시스템 프롬프트 필요시 아래처럼 삽입
            messages = [
                {"role": "system", "content": "You are a clinical doctor."},
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        # Upstage의 chat-template 사용
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 토큰화 및 tensor 변환 (모델 디바이스로)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        after_response = upstage_response_parser(output_text)
        return after_response

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

# 사용 예시
if __name__ == "__main__":
    model = Upstage(
        hg_model_id="Upstage/SOLAR-10.7B-Instruct-v1.0",
        device="cuda",
        cache_dir="/workspace/kor_med_opendataset/hg_cache"
    )

    prompt = "한국어로 대형 언어모델이 무엇인지 설명해줘."
    print(model.run(prompt, max_new_tokens=300))
