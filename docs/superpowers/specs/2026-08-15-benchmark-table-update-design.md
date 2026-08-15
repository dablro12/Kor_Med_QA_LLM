# Benchmark Table Update (Bonsai) — Design

**Date:** 2026-08-15  
**Status:** Approved for planning  
**Scope:** Add PrismML Ternary + 1-bit Bonsai rows to published `benchmark/` artifacts and README tables.

## Goal

Refresh the public leaderboard artifacts so both completed Bonsai GGUF runs appear alongside existing models, without changing evaluation code, prompts, or re-running inference.

## Non-goals

- Permanent/general aggregation pipeline beyond this update script
- Quality reports (parse-fail rates, null preds)
- Re-inference, prompt/parser changes
- Changing FLOPs methodology for GGUF

## Decisions

| Topic | Choice |
|---|---|
| Scope | A — add 2 Bonsai models to csv/md/png + README |
| Display names | `prism-ml` / `Ternary-Bonsai-27B` and `prism-ml` / `Bonsai-27B-1bit` |
| Metrics | Same columns as existing rows; parquet values as-is (incl. GGUF FLOPs stub) |
| Method | One-shot script (not notebook re-run loop) |

## Sources → outputs

### Inputs (per dataset)

`kor_med_opendataset/results/<dataset_path>/<model_folder>/*_detailed.parquet`

Evaluated with existing `src.metrics.ClinicalQAEvaluator`:

- `accuracy (%)` from `gt_answer == pred_answer` (string compare)
- `avg_time_per_token (s)` from `time_per_token_s` mean
- `mean_flops (GFlops)` from `flops_this` mean / 1e9

### Display-name override

| Result folder | model_group | model_name |
|---|---|---|
| `prism-ml_Ternary-Bonsai-27B-gguf` | `prism-ml` | `Ternary-Bonsai-27B` |
| `prism-ml_Bonsai-27B-gguf` | `prism-ml` | `Bonsai-27B-1bit` |
| other folders | `split('_')[0]` | `'_'.join(split('_')[1:])` |

### Dataset path → `benchmark/` stem

| Results path | Output stem (`benchmark/<stem>.{csv,md,png}`) |
|---|---|
| `snuh_ClinicalQA_benchmark` | `snuh_ClinicalQA_benchmark` |
| `sean0042_KorMedMCQA_benchmark/doctor` | `sean0042_KorMedMCQA_benchmark_doctor` |
| `sean0042_KorMedMCQA_benchmark/nurse` | `sean0042_KorMedMCQA_benchmark_nurse` |
| `sean0042_KorMedMCQA_benchmark/dentist` | `sean0042_KorMedMCQA_benchmark_dentist` |
| `sean0042_KorMedMCQA_benchmark/pharm` | `sean0042_KorMedMCQA_benchmark_pharm` |
| `aihub_전문_의학지식_데이터_benchmark_객관식` | `aihub_전문_의학지식_데이터_benchmark_객관식` |
| `aihub_필수의료_의학지식_데이터_benchmark_객관식` | `aihub_필수의료_의학지식_데이터_benchmark_객관식` |

### Outputs

1. Rewrite `benchmark/<stem>.csv` and `.md` (columns unchanged)
2. Regenerate `benchmark/<stem>.png` with notebook-equivalent Nature-style scatter
3. Replace markdown tables inside each matching README `<details>` block; keep image paths

## Script design

**Path:** `scripts/9_update_benchmark_tables.py`

**Behavior:**

1. For each of the 7 datasets, glob `*_detailed.parquet` under the results tree
2. Build `all_summary_df` via `ClinicalQAEvaluator`
3. Apply display-name overrides
4. Sort by `model_group`, then `accuracy (%)` ascending (match notebook)
5. Write csv + md
6. Plot with transplanted `plot_medical_llm_benchmark_nature` (palette +1 color for `prism-ml` if needed; use display short names on labels)
7. Sync README tables from the new `.md` files
8. `--dry-run`: print tables only, no writes

**Sorting / rounding:** match notebook — accuracy and time to 3 decimals for display columns; flops GFlops to 3 decimals.

## Known data caveats (document, do not “fix”)

- GGUF FLOPs are a fixed 27B stub; still published for schema consistency
- Some parquets have fewer rows than the full question set (e.g. KorMed doctor Ternary); accuracy is parquet-denominated like other incomplete historical runs
- 1-bit accuracy is materially lower than Ternary on several sets; publish as measured

## Verification

- Every output csv contains both Bonsai display rows under `prism-ml`
- Recomputed accuracy from parquet matches csv for those two rows
- README `<details>` tables match corresponding `benchmark/*.md`
- Image files exist and are freshly written for all 7 stems
- Dry-run then real run

## Out of scope follow-ups

- Factoring plot code into `src/` permanently
- CI hook to regenerate tables
- Annotating FLOPs-as-stub in README footnotes
