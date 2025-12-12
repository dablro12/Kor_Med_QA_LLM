# snuh_ClinicalQA_benchmark.py
import pandas as pd
import time
import torch
import json
import re
import os
import pynvml  # GPU 메모리 사용량 측정
from tqdm import tqdm
from src.qa_prompt import get_snuh_ClinicalQA_prompt
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
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        
        # explanation 추출: 닫는 따옴표가 없어도 처리
        # "explanation":"로 시작해서 끝까지 또는 닫는 따옴표까지
        explanation_match = re.search(r'"explanation"\s*:\s*"(.*?)(?:"\s*[,}]|$)', text, re.DOTALL)
        if not explanation_match:
            # 더 관대한 패턴: "explanation":" 이후 모든 텍스트 (닫는 따옴표 없어도)
            explanation_match = re.search(r'"explanation"\s*:\s*"(.*)', text, re.DOTALL)
        
        explanation_text = explanation_match.group(1) if explanation_match else ""
        
        if answer_match:
            answer = answer_match.group(1).strip()
            
            # A-E 선택지 추출
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

    # 일부 모델이 answer를 문자열이 아닌 숫자로 반환하는 경우가 있어 문자열로 강제 변환
    answer = str(data["answer"]).strip()
    explanation = data.get("explanation", "")

    # ------------------------------------------------------------
    # 5) A-E 선택지 추출 (다양한 형식 지원)
    # ------------------------------------------------------------
    # 패턴 1: "A)", "B)", "C)" 형식
    # 패턴 2: "A/B", "B/E", "A/B/C" 형식
    # 패턴 3: "A", "B", "C" 단독 형식
    # 패턴 4: "A) 또는 B)", "A/B/C" 등 복합 형식
    
    # 모든 A-E 문자 추출
    option_letters = re.findall(r'[A-Ea-e]', answer)
    
    if not option_letters:
        return None
    
    # 첫 번째 유효한 옵션 선택 (대문자로 변환)
    option_letter = option_letters[0].upper()
    
    # 유효성 검증 (A-E 범위 내)
    if option_letter not in ['A', 'B', 'C', 'D', 'E']:
        return None

    return {
        "pred_answer": option_letter,
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

from src._Model_Loader import load_model
class BenchmarkProcessor:
    def __init__(self, hg_model_id: str, data_path: str, save_dir: str):
        self.hg_model_id = hg_model_id
        self.data_path = data_path
        self.save_dir = save_dir
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)

        # 🔥 파라미터 수 캐싱 (HF 모델 본체는 self.model.model)
        self.model = load_model(self.hg_model_id)
        print("📌 Counting model parameters... (only once)")
        self.num_params = sum(p.numel() for p in self.model.model.parameters())
        print(f"📌 Total Parameters: {self.num_params:,}")
    def _load_data(self):
        self.df = pd.read_csv(self.data_path)

    def run(self):
        self._load_data()

        
        results = []
        total_tokens = 0
        total_flops = 0

        start_time_total = time.time()
        gpu_used_before, gpu_total = get_gpu_memory_used()

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing QA Benchmark", leave=False):

            prompt = get_snuh_ClinicalQA_prompt(row)

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
                "question_id": row["question_id"],
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

    parquet_path = os.path.join(args.save_dir, f"{args.model.replace('/', '_')}_detailed.parquet")
    if os.path.exists(parquet_path):
        print(f"🔥 {parquet_path} already exists. BenchmarkProcessor will not run.")
        exit()

    processor = BenchmarkProcessor(
        hg_model_id=args.model,
        data_path=args.data,
        save_dir=args.save_dir
    )
    processor.run()
