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



import re
import json


def clovax_response_parser(response):
    """
    Clovax output 예시(원본):

    tool_list

    system
    ...
    assistant

    {"answer":"A","explanation":"..."}

    에서 assistant 이후 JSON만 추출해서 반환
    """
    import re

    # assistant 다음의 JSON만 추출
    # 1) 'assistant'라는 단어와 그 이후 줄 바꿈 이후 {로 시작하는 부분 찾기
    assistant_json_match = re.search(
        r"assistant\s*\n\s*(\{[\s\S]+)", response
    )
    preprocess_response = None
    if assistant_json_match:
        # {로 시작하는 부분 추출 (줄 끝까지)
        json_part = assistant_json_match.group(1)
        # 추가로 assistant 이후에 다른 내용이 올 수 있지만, JSON만 추출
        # 중괄호 짝이 모두 맞는 첫번째 JSON을 추출하거나, 맨 끝 } 까지
        # 가장 단순하게는 첫 번째 }를 찾음
        # JSON 응답이 여러줄일 수 있으므로, 여러줄 허용
        # ({...}) 패턴 전체 추출
        json_match = re.search(r"(\{[\s\S]+\})", json_part)
        if json_match:
            preprocess_response = json_match.group(1)
        else:
            # 그래도 없으면 전체를 넘김
            preprocess_response = json_part.strip()
    else:
        # assistant 키워드 없으면 response 전체에서 처음 등장하는 JSON 추출
        json_match = re.search(r"(\{[\s\S]+\})", response)
        if json_match:
            preprocess_response = json_match.group(1)
        else:
            preprocess_response = response.strip()
    return preprocess_response

class Clovax:
    """
    Clovax wrapper for HyperCLOVAX-SEED-Text-Instruct-0.5B and compatible models (Chat template 기반)
    - 캐시/오프라인 지원
    - TEXT/메시지 리스트 모두 지원
    - count_tokens() 제공
    """

    def __init__(self, hg_model_id: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", device: str = "cuda", cache_dir: str = None):
        self.hg_model_id = hg_model_id
        self.device = device
        self.cache_dir = cache_dir

        model_cache_dir = get_hf_cache_path(cache_dir, hg_model_id)
        self.local_files_only = os.path.exists(model_cache_dir) if model_cache_dir else False

        print(f"[Clovax] cache_dir = {cache_dir}")
        print(f"[Clovax] model cache dir = {model_cache_dir}")
        print(f"[Clovax] local_files_only = {self.local_files_only}")

        self.model = AutoModelForCausalLM.from_pretrained(
            hg_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            hg_model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only
        )

    def run(self, chat_or_text, max_new_tokens=1024, stop_strings=None, repetition_penalty=1.0, temperature=0.0, **gen_kwargs):
        """
        chat_or_text: str or list-of-dict (chat template)
        """
        # 메시지 구성: str이면 기본 chat prompt로 변환
        if isinstance(chat_or_text, str):
            chat = [
                {"role": "tool_list", "content": ""},
                {"role": "system", "content": "You are a clinical doctor."},
                {"role": "user", "content": chat_or_text},
            ]
        else:
            chat = chat_or_text

        inputs = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # GPU로 이동
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # stop_strings default 처리
        stop_strings = stop_strings if stop_strings is not None else ["<|endofturn|>", "<|stop|>"]

        # generation 파라미터
        generate_args = dict(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_new_tokens,
            stop_strings=stop_strings,
            repetition_penalty=repetition_penalty,
            tokenizer=self.tokenizer,
            temperature=temperature
        )
        generate_args.update(gen_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(**generate_args)
        # 첫번째 (batch=1)
        raw_output =  self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return clovax_response_parser(raw_output)

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False
        )
        return len(encoded["input_ids"])

if __name__ == "__main__":
    model = Clovax(
        hg_model_id="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        device="cuda",
        cache_dir=None  # 필요시 지정
    )
    # 간단 사용
    print(model.run("슈뢰딩거 방정식과 양자역학의 관계를 최대한 자세히 알려줘.", max_new_tokens=512))
