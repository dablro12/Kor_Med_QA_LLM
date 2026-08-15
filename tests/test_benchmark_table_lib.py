import pandas as pd
import pytest

from scripts.benchmark_table_lib import (
    split_model_folder,
    display_frame,
    dataframe_to_markdown,
    replace_readme_table,
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
