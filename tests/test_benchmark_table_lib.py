from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.benchmark_table_lib import (
    DISPLAY_COLS,
    dataframe_to_markdown,
    display_frame,
    replace_readme_table,
    split_model_folder,
    summarize_dataset,
    write_csv_md,
)


def test_split_override_ternary():
    assert split_model_folder("prism-ml_Ternary-Bonsai-27B-gguf") == (
        "prism-ml",
        "Ternary-Bonsai-27B",
    )


def test_split_override_onebit():
    assert split_model_folder("prism-ml_Bonsai-27B-gguf") == (
        "prism-ml",
        "Bonsai-27B-1bit",
    )


def test_split_default_qwen():
    assert split_model_folder("Qwen_Qwen3-8B") == ("Qwen", "Qwen3-8B")


def test_display_frame_sort_and_columns():
    raw = pd.DataFrame(
        {
            "model": [
                "prism-ml_Bonsai-27B-gguf",
                "prism-ml_Ternary-Bonsai-27B-gguf",
                "Qwen_Qwen3-8B",
            ],
            "accuracy (%)": [40.0, 70.0, 60.0],
            "avg_time_per_token (s)": [0.01234, 0.01789, 0.02111],
            "mean_flops": [1.62e11, 1.62e11, 5e12],
        }
    )
    out = display_frame(raw)
    cols = [
        "model_group",
        "model_name",
        "accuracy (%)",
        "avg_time_per_token (s)",
        "mean_flops (GFlops)",
    ]
    assert list(out.columns) == cols
    # sort: model_group ASC (ASCII: Qwen before prism-ml), then accuracy ASC
    assert out.iloc[0]["model_group"] == "Qwen"
    assert list(out.loc[out.model_group == "prism-ml", "model_name"]) == [
        "Bonsai-27B-1bit",
        "Ternary-Bonsai-27B",
    ]


def test_dataframe_to_markdown_header():
    df = pd.DataFrame(
        {
            "model_group": ["prism-ml"],
            "model_name": ["Ternary-Bonsai-27B"],
            "accuracy (%)": [70.33],
            "avg_time_per_token (s)": [0.017],
            "mean_flops (GFlops)": [162.0],
        }
    )
    md = dataframe_to_markdown(
        df,
        [
            "model_group",
            "model_name",
            "accuracy (%)",
            "avg_time_per_token (s)",
            "mean_flops (GFlops)",
        ],
    )
    assert md.startswith("| model_group | model_name |")
    assert "Ternary-Bonsai-27B" in md


def test_replace_readme_table_keeps_image_and_summary():
    readme = """## x
<details>
<summary><b>SNUH ClinicalQA</b> (Click to expand)</summary>

![SNUH ClinicalQA Benchmark](benchmark/snuh_ClinicalQA_benchmark.png)

| model_group | model_name | accuracy (%) | avg_time_per_token (s) | mean_flops (GFlops) |
|----|----|----|----|----|
| Qwen | Qwen3-8B | 63.72 | 0.021 | 12645.159 |

</details>
"""
    new_table = """| model_group | model_name | accuracy (%) | avg_time_per_token (s) | mean_flops (GFlops) |
|----|----|----|----|----|
| prism-ml | Ternary-Bonsai-27B | 70.33 | 0.017 | 162.0 |
| Qwen | Qwen3-8B | 63.72 | 0.021 | 12645.159 |
"""
    out = replace_readme_table(readme, "SNUH ClinicalQA", new_table)
    assert "![SNUH ClinicalQA Benchmark](benchmark/snuh_ClinicalQA_benchmark.png)" in out
    assert "Ternary-Bonsai-27B" in out
    assert "<summary><b>SNUH ClinicalQA</b>" in out
    assert out.count("| model_group |") == 1


def _write_tiny_parquet(path: Path, gt, pred, tpt=0.01, flops=1e11):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(gt)
    table = pa.table(
        {
            "question_id": list(range(n)),
            "gt_answer": gt,
            "pred_answer": pred,
            "pred_explanation": [""] * n,
            "first_token_latency_s": [0.1] * n,
            "time_per_token_s": [tpt] * n,
            "vram_used_MB": [100.0] * n,
            "flops_this": [flops] * n,
            "flops_per_token": [flops] * n,
            "cost_per_token_s": [tpt] * n,
        }
    )
    pq.write_table(table, path)


def test_summarize_dataset_accuracy(tmp_path: Path):
    root = tmp_path / "ds"
    _write_tiny_parquet(
        root / "prism-ml_Ternary-Bonsai-27B-gguf" / "x_detailed.parquet",
        gt=["A", "B"],
        pred=["A", "B"],
    )
    _write_tiny_parquet(
        root / "Qwen_Qwen3-8B" / "x_detailed.parquet",
        gt=["A", "B"],
        pred=["A", "C"],
    )
    summary = summarize_dataset(root)
    assert set(summary["model"]) == {
        "prism-ml_Ternary-Bonsai-27B-gguf",
        "Qwen_Qwen3-8B",
    }
    row = summary.set_index("model").loc["prism-ml_Ternary-Bonsai-27B-gguf"]
    assert row["accuracy (%)"] == 100.0
    row_q = summary.set_index("model").loc["Qwen_Qwen3-8B"]
    assert row_q["accuracy (%)"] == 50.0


def test_write_csv_md(tmp_path: Path):
    df = pd.DataFrame(
        {
            "model_group": ["prism-ml"],
            "model_name": ["Ternary-Bonsai-27B"],
            "accuracy (%)": [70.33],
            "avg_time_per_token (s)": [0.017],
            "mean_flops (GFlops)": [162.0],
        }
    )
    write_csv_md(df, tmp_path, "snuh_ClinicalQA_benchmark")
    csv_path = tmp_path / "snuh_ClinicalQA_benchmark.csv"
    md_path = tmp_path / "snuh_ClinicalQA_benchmark.md"
    assert csv_path.exists() and md_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == DISPLAY_COLS
    assert "Ternary-Bonsai-27B" in md_path.read_text(encoding="utf-8")
