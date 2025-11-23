# snuh_ClinicalQA_benchmark.py
import pandas as pd
import time
import torch
import json
import re
import os
import pynvml  # GPU 메모리 사용량 측정
from tqdm import tqdm
from src.qa_prompt import get_sean0042_KorMedMCQA_prompt
from getpass import getpass
import os

def parse_model_response(resp: str):
    if resp is None:
        return None

    text = resp.strip()

    # ------------------------------------------------------------
    # 1) 코드블록 제거 (```json ... ``` 또는 ```\n{...}\n```)
    # ------------------------------------------------------------
    # 다양한 코드블록 형식 처리
    codeblock_patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",  # ```json {...} ```
        r"```\s*(\{.*?\})\s*```",  # ``` {...} ```
        r"```\s*\n\s*(\{.*?\})\s*\n\s*```",  # ```\n{...}\n```
    ]
    
    for pattern in codeblock_patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    
    # ------------------------------------------------------------
    # 2) JSON 객체만 추출 ({ ... })
    # ------------------------------------------------------------
    # 불완전한 JSON도 추출 (닫는 중괄호가 없어도)
    json_pattern = r"\{[\s\S]*"
    m = re.search(json_pattern, text)
    if m:
        text = m.group(0).strip()

    # ------------------------------------------------------------
    # 3) JSON 파싱 시도
    # ------------------------------------------------------------
    data = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        # JSON 파싱 실패 시, 정규식으로 직접 필드 추출 시도
        # answer 필드 추출
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        
        # explanation 추출: "explanation":" 이후부터 마지막 } 전까지 추출
        # 중간에 있는 따옴표는 그대로 유지 (이스케이프되지 않은 따옴표 포함)
        explanation_text = ""
        
        explanation_start_match = re.search(r'"explanation"\s*:\s*"', text)
        if explanation_start_match:
            start_pos = explanation_start_match.end()
            # 마지막 } 찾기
            last_brace = text.rfind('}')
            if last_brace > start_pos:
                # start_pos부터 last_brace 전까지 추출
                raw_explanation = text[start_pos:last_brace]
                # 끝부분의 따옴표, 쉼표, 공백 제거
                explanation_text = raw_explanation.rstrip().rstrip('"').rstrip(',').rstrip('}').strip()
            else:
                # }가 없으면 끝까지 추출
                raw_explanation = text[start_pos:]
                # 끝부분의 따옴표, 공백 제거
                explanation_text = raw_explanation.rstrip().rstrip('"').strip()
        
        # explanation이 여전히 비어있으면 다른 패턴 시도
        if not explanation_text:
            # 패턴: "explanation":"...까지 (닫는 따옴표 찾기, 하지만 중간 따옴표는 무시)
            # 이 방법은 완벽하지 않지만, 간단한 경우에 작동
            explanation_match = re.search(r'"explanation"\s*:\s*"(.*?)(?:"\s*[,}]|$)', text, re.DOTALL)
            if explanation_match:
                explanation_text = explanation_match.group(1).strip()
        
        if answer_match:
            answer = answer_match.group(1).strip()
            
            # 1-5 숫자 선택지 추출 (KorMedMCQA는 1-5 선택지)
            option_numbers = re.findall(r'[1-5]', answer)
            if option_numbers:
                option_number = option_numbers[0]
                if option_number in ['1', '2', '3', '4', '5']:
                    return {
                        "pred_answer": option_number,
                        "pred_explanation": explanation_text
                    }
            
            # A-E 선택지 추출 (다른 벤치마크용)
            option_letters = re.findall(r'[A-Ea-e]', answer)
            if option_letters:
                option_letter = option_letters[0].upper()
                if option_letter in ['A', 'B', 'C', 'D', 'E']:
                    return {
                        "pred_answer": option_letter,
                        "pred_explanation": explanation_text
                    }
        
        # 정규식 추출도 실패한 경우
        print("❌ JSON decode error:", e)
        print("📝 RAW RESPONSE START\n", resp, "\n📝 RAW RESPONSE END")
        return None

    # ------------------------------------------------------------
    # 4) 필드 검증
    # ------------------------------------------------------------
    if not isinstance(data, dict):
        return None

    if "answer" not in data:
        return None

    answer = data["answer"].strip()
    explanation = data.get("explanation", "")

    # ------------------------------------------------------------
    # 5) 1~5 선택지 추출 (다양한 형식 지원: "1)", "2", "3/4", "5,4", "1 또는 2" 등)
    # ------------------------------------------------------------
    # 패턴 1: "1)", "2)", "3)" 형식
    # 패턴 2: "1/2", "3/4", "1,2,3" 형식
    # 패턴 3: "1", "2", "3" 단독 형식
    # 패턴 4: "1) 또는 2)", "1/2/3" 등 복합 형식

    # 모든 1-5 숫자 추출 (정답이 1~5 중에 있음)
    option_numbers = re.findall(r'[1-5]', answer)
    
    if not option_numbers:
        return None
    
    # 첫 번째 유효한 옵션 선택
    option_number = option_numbers[0]
    
    # 유효성 검증 (1-5 범위 내)
    if option_number not in ['1', '2', '3', '4', '5']:
        return None

    return {
        "pred_answer": option_number,
        "pred_explanation": explanation
    }


def get_gpu_memory_used(device_idx=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / (1024**2)  # MB
    total = info.total / (1024**2)
    pynvml.nvmlShutdown()
    return used, total

class BenchmarkProcessor:
    def __init__(self, hg_model_id: str, data_path: str, save_dir: str):
        self.hg_model_id = hg_model_id
        self.data_path = data_path
        self.save_dir = save_dir
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _load_model(self):
        if "qwen" in self.hg_model_id.lower():
            from src.qwen import Qwen
            self.model = Qwen(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "gpt" in self.hg_model_id.lower():
            from src.gpt import GPT
            self.model = GPT(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "deepseek" in self.hg_model_id.lower():
            from src.deepseek import DeepSeek
            self.model = DeepSeek(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "gemma" in self.hg_model_id.lower():
            from src.gemma import Gemma
            self.model = Gemma(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "llama" in self.hg_model_id.lower():
            from src.llama import Llama
            self.model = Llama(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "exaone" in self.hg_model_id.lower():
            from src.exaone import Exaone
            self.model = Exaone(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        elif "kanana" in self.hg_model_id.lower():
            from src.kanana import Kanana
            self.model = Kanana(
                hg_model_id=self.hg_model_id,
                device="cuda",
                cache_dir="/workspace/kor_med_opendataset/hg_cache"
            )
        else:
            raise ValueError(f"Unsupported model: {self.hg_model_id}")

        # 🔥 파라미터 수 캐싱 (HF 모델 본체는 self.model.model)
        hf_model = self.model.model
        print("📌 Counting model parameters... (only once)")
        self.num_params = sum(p.numel() for p in hf_model.parameters())
        print(f"📌 Total Parameters: {self.num_params:,}")

    def _load_data(self):
        self.df = pd.read_csv(self.data_path)

    def run(self):
        self._load_model()
        self._load_data()

        
        results = []
        total_tokens = 0
        total_flops = 0

        start_time_total = time.time()
        gpu_used_before, gpu_total = get_gpu_memory_used()

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing QA Benchmark", leave=False):

            prompt = get_sean0042_KorMedMCQA_prompt(row)

            t0 = time.time()
            response = self.model.run(prompt, max_new_tokens=512, temperature=0.1, top_p=0.9)
            first_token_latency = time.time() - t0

            parsed = parse_model_response(response)
            if parsed is None:
                continue

            # 토큰 수
            length_tokens = self.model.count_tokens(response)
            total_tokens += length_tokens

            full_time = time.time() - t0
            time_per_token = full_time / length_tokens if length_tokens > 0 else None

            # GPU 사용량
            gpu_used_after, _ = get_gpu_memory_used()
            vram_used = gpu_used_after - gpu_used_before

            # --------------------------------------------------------
            # 🔥 FLOPs 계산
            # --------------------------------------------------------
            N = self.num_params
            D = length_tokens

            flops_per_token = 6 * N
            flops_this = flops_per_token * D
            total_flops += flops_this

            cost_per_token = full_time / D if D > 0 else None
            # --------------------------------------------------------

            results.append({
                "question_id": row["question"],
                "gt_answer": row["answer"],
                "pred_answer": parsed["pred_answer"],
                "pred_explanation": parsed["pred_explanation"],
                "first_token_latency_s": first_token_latency,
                "time_per_token_s": time_per_token,
                "vram_used_MB": vram_used,

                # FLOPs 기록
                "flops_this": flops_this,
                "flops_per_token": flops_per_token,
                "cost_per_token_s": cost_per_token
            })

        total_time = time.time() - start_time_total
        throughput = total_tokens / total_time if total_time > 0 else None

        summary = {
            "model": self.hg_model_id,
            "total_time_s": total_time,
            "throughput_tokens_per_sec": throughput,
            "gpu_total_vram_MB": gpu_total,
            "average_vram_used_MB": sum(r["vram_used_MB"] for r in results) / len(results),

            # 🔥 FLOPs summary 추가
            "total_flops": total_flops,
            "total_tokens": total_tokens,
            "parameters": self.num_params
        }
        
        # 🔥 Detailed results → Parquet 저장
        df_results = pd.DataFrame(results)
        parquet_path = os.path.join(
            self.save_dir,
            f"{self.hg_model_id.replace('/', '_')}_detailed.parquet"
        )
        df_results.to_parquet(parquet_path, index=False)

        # 🔥 Summary → JSON 저장
        summary_path = os.path.join(
            self.save_dir,
            f"{self.hg_model_id.replace('/', '_')}_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("Saved detailed results (parquet) and summary (json).")
        print("Summary:", summary)


# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SNUH ClinicalQA Benchmark Runner")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to ClinicalQA CSV file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where benchmark results will be saved"
    )
    parser.add_argument(
        "--cuda_ids",
        type=str,
        default=None,
        help="CUDA device IDs to use (e.g., '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES environment variable."
    )

    args = parser.parse_args()

    # CUDA device 설정
    if args.cuda_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids
        print(f"🔧 CUDA_VISIBLE_DEVICES set to: {args.cuda_ids}")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"🔧 Using existing CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("🔧 CUDA_VISIBLE_DEVICES not set, using default (all available GPUs)")

    processor = BenchmarkProcessor(
        hg_model_id=args.model,
        data_path=args.data,
        save_dir=args.save_dir
    )
    processor.run()