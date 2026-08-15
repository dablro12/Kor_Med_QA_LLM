# Bonsai 27B GGUF Overnight Benchmark

Date: 2026-08-12

## Goal

Run PrismML Bonsai 27B (ternary + 1-bit) on the existing Kor_MedQA benchmark framework, unattended: download first, then sequential evaluation on CUDA device 1 (RTX 3090 24GB).

## Models

| Variant | Hugging Face repo | Quantization |
|---------|-------------------|--------------|
| Ternary (performance) | `prism-ml/Ternary-Bonsai-27B-gguf` | `Ternary-Bonsai-27B-Q2_g64.gguf` (mainline llama.cpp) |
| 1-bit (size) | `prism-ml/Bonsai-27B-gguf` | `Bonsai-27B-Q1_0.gguf` |

Do not download F16, dspark, or mmproj files.

Cache: `/workspace/kor_med_opendataset/hg_cache`  
Results: `/workspace/kor_med_opendataset/results/<dataset>/...` matching the existing on-disk layout.

## Runtime

- Backend: `llama-cpp-python` with CUDA (`n_gpu_layers=-1`)
- Device: `CUDA_VISIBLE_DEVICES=1`
- CPU threads: `n_threads=4`, `n_threads_batch=4`
- Prompt/decode scratch: `n_batch=16384`, `n_ubatch=16384`
- Context: `n_ctx=131072` (fills leftover 3090 VRAM with KV; packed weights are only ~4–8GB)
- QA loop stays one sample at a time (no change to `*_benchmark.py`)

## Integration

Existing processors call `load_model(id)` then `.run()`, `.count_tokens()`, and `self.model.model.parameters()` for FLOPs.

- New wrapper: `src/bonsai.py`
- Loader match (before other families): `bonsai`, `prism`, or `ternary` in the model id
- FLOPs use a fixed 27B parameter stand-in (estimate, not measured matmul FLOPs)

Overnight orchestration does **not** rewrite historical model lists in `scripts/4`–`7`. It invokes the same Python benchmark entrypoints with only the two Bonsai ids.

## Datasets (all)

1. SNUH ClinicalQA → `results/snuh_ClinicalQA_benchmark/`
2. KorMedMCQA doctor / nurse / dentist / pharm → `results/sean0042_KorMedMCQA_benchmark/<domain>/`
3. AIHub 전문 의학지식 (객관식) → `results/aihub_전문_의학지식_데이터_benchmark_객관식/`
4. AIHub 필수의료 의학지식 (객관식) → `results/aihub_필수의료_의학지식_데이터_benchmark_객관식/`

Order: Ternary through all datasets, then 1-bit through all datasets.

## Overnight flow

1. `scripts/1b_download_bonsai.sh` downloads both GGUF repos into `hg_cache`
2. Smoke: load + one short generate per model on CUDA:1
3. Smoke failure: exit non-zero and print the log path (no automatic fallback)
4. Success: sequential full benches via `scripts/8_bonsai_overnight.sh`
5. Master log: `results/bonsai_overnight/`

## Out of scope

README / `benchmark/` table refresh, MLX, unpacked FP16, Prism developer API, prompt or parser changes.
