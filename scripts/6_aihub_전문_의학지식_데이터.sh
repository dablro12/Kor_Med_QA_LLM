#!/bin/bash
set -eu  # pipefail 제외

# ============================================================
# SNUH ClinicalQA Benchmark Runner (with OOM cooldown & CUDA setting)
# ============================================================

WORKSPACE="/workspace"

## 실험 모델 목록
CUDA_IDS="0"

# gpu 마다 할당 할 모델 목록

MODELS=(
    # -------------------------------------------------------
    # Qwen Series
    # -------------------------------------------------------
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"

    # -------------------------------------------------------
    # Google Gemma / Med-Gemma Series
    # -------------------------------------------------------
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/medgemma-4b-it"

    # Gemma-3n (New Additions)
    "google/gemma-3n-E2B-it"
    "google/gemma-3n-E4B-it"

    # -------------------------------------------------------
    # Meta Llama Series
    # -------------------------------------------------------
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"

    # New addition
    "meta-llama/Meta-Llama-3-8B-Instruct"

    # -------------------------------------------------------
    # LGAI EXAONE Series
    # -------------------------------------------------------
    "LGAI-EXAONE/EXAONE-4.0-1.2B"
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    # -------------------------------------------------------
    # Kakao Kanana Series
    # -------------------------------------------------------
    "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    "kakaocorp/kanana-1.5-8b-instruct-2505"

    # -------------------------------------------------------
    # Naver ClovaX Series
    # -------------------------------------------------------
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    # -------------------------------------------------------
    # Upstage Solar Series
    # -------------------------------------------------------
    "upstage/SOLAR-10.7B-Instruct-v1.0"

    # -------------------------------------------------------
    # K-Intelligence Midm Series
    # -------------------------------------------------------
    "K-intelligence/Midm-2.0-Mini-Instruct"
    "K-intelligence/Midm-2.0-Base-Instruct"

)

# ANSI Colors
BOLD="\033[1m"
GREEN="\033[92m"
YELLOW="\033[93m"
BLUE="\033[94m"
RED="\033[91m"
RESET="\033[0m"

# CUDA 설정 추가
# 사용할 GPU 번호 지정: CUDA_IDS 사용자가 직접 환경 변수로 지정 가능
if [ -z "${CUDA_IDS:-}" ]; then
    echo -e "${BLUE}CUDA_IDS를 지정하지 않았습니다. 기본값: 0 사용${RESET}"
    export CUDA_VISIBLE_DEVICES=0
    CUDA_IDS="0"
else
    echo -e "${BLUE}CUDA_VISIBLE_DEVICES <- CUDA_IDS: ${CUDA_IDS}${RESET}"
    export CUDA_VISIBLE_DEVICES=${CUDA_IDS}
fi

# GPU VRAM 확인 함수
# 50% 이상 사용하고 있으면 5분 대기 후 진행
check_gpu_memory() {
    local gpu_ids="$1"
    local need_wait=0
    echo -e "${BLUE}📊 GPU VRAM 상태 확인 중...${RESET}"
    
    # CUDA_IDS를 쉼표로 분리하여 각 GPU 확인
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        # nvidia-smi로 GPU 메모리 정보 가져오기
        local memory_info=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ]; then
            local used=$(echo "$memory_info" | cut -d',' -f1 | tr -d ' ')
            local total=$(echo "$memory_info" | cut -d',' -f2 | tr -d ' ')
            local used_percent=$((used * 100 / total))
            echo -e "${BLUE}  GPU $gpu_id: ${used}MB / ${total}MB 사용 중 (${used_percent}%)${RESET}"
            
            # 메모리 사용률이 90% 이상이면 경고
            if [ $used_percent -ge 90 ]; then
                echo -e "${YELLOW}  ⚠️  GPU $gpu_id 메모리 사용률이 높습니다 (${used_percent}%)${RESET}"
            fi
            # 50% 이상일 경우 flag
            if [ $used_percent -ge 50 ]; then
                echo -e "${YELLOW}  ⏸️ GPU $gpu_id 메모리 사용률이 50% 이상입니다 (${used_percent}%). 5분 대기 후 재시도합니다.${RESET}"
                need_wait=1
            fi
        else
            echo -e "${RED}  ❌ GPU $gpu_id 정보를 가져올 수 없습니다${RESET}"
        fi
    done
    echo

    if [ $need_wait -eq 1 ]; then
        sleep 300
    fi
}

# 모델 실행 함수 (재시도 로직 포함)
run_model_with_retry() {
    local model_id="$1"
    local data_path="$2"
    local save_dir="$3"
    local cuda_ids="$4"
    local log_file="$5"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        # GPU 메모리 확인 및 50% 이상시 5분 대기
        check_gpu_memory "$cuda_ids"
        
        # 실행 및 로그 저장
        python aihub_전문_의학지식_데이터_benchmark.py \
            --model "$model_id" \
            --data "$data_path" \
            --save_dir "$save_dir" \
            --cuda_ids "$cuda_ids" \
            2>&1 | tee "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        
        # 성공한 경우
        if [ $exit_code -eq 0 ]; then
            return 0
        fi
        
        # OOM 에러 확인
        if grep -qi "out of memory\|CUDA out of memory\|RuntimeError.*CUDA" "$log_file"; then
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo -e "${RED}❌ [$model_id] OOM 발생! GPU 메모리 부족 (재시도 $retry_count/$max_retries)${RESET}"
                echo -e "${YELLOW}⏳ 5분(300초) 대기 후 재시도...${RESET}"
                sleep 300
                # GPU 메모리 정리 대기
                check_gpu_memory "$cuda_ids"
                continue
            else
                echo -e "${RED}❌ [$model_id] OOM 발생! 최대 재시도 횟수($max_retries) 초과${RESET}"
                return 1
            fi
        fi
        
        # 기타 에러
        echo -e "${RED}❌ [$model_id] 실행 중 오류 발생! (재시도 $retry_count/$max_retries)${RESET}"
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo -e "${YELLOW}⏳ 5분(300초) 대기 후 재시도...${RESET}"
            sleep 300
            continue
        else
            echo -e "${RED}❌ [$model_id] 최대 재시도 횟수($max_retries) 초과${RESET}"
            return 1
        fi
    done
    
    return 1
}

echo -e "${BOLD}${BLUE}✨ aihub_전문_의학지식_데이터 Benchmark 시작합니다 ✨${RESET}"
echo

TOTAL_START_TS=$(date +%s)

SAVE_ROOT="$WORKSPACE/kor_med_opendataset/results/aihub_전문_의학지식_데이터_benchmark/객관식"
DATA_PATH="$WORKSPACE/kor_med_opendataset/aihub_전문_의학지식_데이터/라벨링데이터_객관식.csv"
LOG_DIR="$SAVE_ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_ROOT" 

DOMAIN_START_TS=$(date +%s)

# -------- 실행 루프 --------
for MODEL_ID in "${MODELS[@]}"; do

    SAFE_NAME=$(echo "$MODEL_ID" | tr '/.' '_')
    MODEL_SAVE_DIR="$SAVE_ROOT/$SAFE_NAME"
    mkdir -p "$MODEL_SAVE_DIR"

    LOG_FILE="$LOG_DIR/benchmark_${SAFE_NAME}.log"

    echo -e "${YELLOW}--------------------------------------------------${RESET}"
    echo -e "${BOLD}모델: ${MODEL_ID}${RESET}"
    echo -e "결과 저장 위치: ${MODEL_SAVE_DIR}"
    echo -e "데이터: ${DATA_PATH}"
    echo -e "로그: ${LOG_FILE}"
    echo -e "${YELLOW}--------------------------------------------------${RESET}"
    echo -e "CUDA_IDS: ${CUDA_IDS}"
    
    MODEL_START=$(date +%s)

    # 재시도 로직이 포함된 모델 실행
    if run_model_with_retry "$MODEL_ID" "$DATA_PATH" "$MODEL_SAVE_DIR" "$CUDA_IDS" "$LOG_FILE"; then
        MODEL_END=$(date +%s)
        echo -e "${GREEN}✅ [$MODEL_ID] 실행 완료 (소요시간: $((MODEL_END - MODEL_START))초)${RESET}"
        echo
    else
        echo -e "${RED}❌ [$MODEL_ID] 실행 실패 (최대 재시도 횟수 초과)${RESET}"
        echo "로그 파일 확인: $LOG_FILE"
        echo -e "${YELLOW}⏳ 다음 모델로 진행합니다...${RESET}"
        echo
        continue
    fi
done

DOMAIN_END_TS=$(date +%s)
echo -e "${GREEN}🎉 벤치마크 완료! (소요시간: $((DOMAIN_END_TS - DOMAIN_START_TS))초) 🎉${RESET}"
echo "결과: $SAVE_ROOT"
echo

TOTAL_END_TS=$(date +%s)
echo -e "${GREEN}🎊 전체 벤치마크 완료! (총 소요시간: $((TOTAL_END_TS - TOTAL_START_TS))초) 🎊${RESET}"

