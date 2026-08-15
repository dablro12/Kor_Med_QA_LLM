from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

DISPLAY_NAME_OVERRIDES = {
    "prism-ml_Ternary-Bonsai-27B-gguf": ("prism-ml", "Ternary-Bonsai-27B"),
    "prism-ml_Bonsai-27B-gguf": ("prism-ml", "Bonsai-27B-1bit"),
}

DISPLAY_COLS = [
    "model_group",
    "model_name",
    "accuracy (%)",
    "avg_time_per_token (s)",
    "mean_flops (GFlops)",
]

DATASETS = [
    {
        "results_rel": "snuh_ClinicalQA_benchmark",
        "stem": "snuh_ClinicalQA_benchmark",
        "readme_summary_substr": "SNUH ClinicalQA",
    },
    {
        "results_rel": "sean0042_KorMedMCQA_benchmark/doctor",
        "stem": "sean0042_KorMedMCQA_benchmark_doctor",
        "readme_summary_substr": "KorMedMCQA - Doctor",
    },
    {
        "results_rel": "sean0042_KorMedMCQA_benchmark/nurse",
        "stem": "sean0042_KorMedMCQA_benchmark_nurse",
        "readme_summary_substr": "KorMedMCQA - Nurse",
    },
    {
        "results_rel": "sean0042_KorMedMCQA_benchmark/dentist",
        "stem": "sean0042_KorMedMCQA_benchmark_dentist",
        "readme_summary_substr": "KorMedMCQA - Dentist",
    },
    {
        "results_rel": "sean0042_KorMedMCQA_benchmark/pharm",
        "stem": "sean0042_KorMedMCQA_benchmark_pharm",
        "readme_summary_substr": "KorMedMCQA - Pharm",
    },
    {
        "results_rel": "aihub_전문_의학지식_데이터_benchmark_객관식",
        "stem": "aihub_전문_의학지식_데이터_benchmark_객관식",
        "readme_summary_substr": "AIHub Professional Medical Knowledge",
    },
    {
        "results_rel": "aihub_필수의료_의학지식_데이터_benchmark_객관식",
        "stem": "aihub_필수의료_의학지식_데이터_benchmark_객관식",
        "readme_summary_substr": "AIHub Essential Medical Knowledge",
    },
]


def split_model_folder(folder: str) -> tuple[str, str]:
    if folder in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[folder]
    parts = folder.split("_")
    return parts[0], "_".join(parts[1:])


def display_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    groups, names = zip(*(split_model_folder(m) for m in df["model"]))
    df["model_group"] = list(groups)
    df["model_name"] = list(names)
    df["mean_flops (GFlops)"] = (df["mean_flops"] / 1e9).round(3)
    df["accuracy (%)"] = df["accuracy (%)"].round(3)
    df["avg_time_per_token (s)"] = df["avg_time_per_token (s)"].round(3)
    out = df[DISPLAY_COLS].sort_values(
        ["model_group", "accuracy (%)"], ascending=[True, True]
    )
    return out.reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, cols: Iterable[str]) -> str:
    cols = list(cols)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "----|" * len(cols),
    ]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(str(x) for x in row.values) + " |")
    return "\n".join(lines) + "\n"


def replace_readme_table(readme_text: str, summary_substr: str, new_md_table: str) -> str:
    """Replace the first markdown table inside the <details> whose <summary> contains summary_substr."""
    pattern = re.compile(
        r"(<details>\s*<summary><b>"
        + re.escape(summary_substr)
        + r".*?</summary>)(.*?)(</details>)",
        re.DOTALL,
    )

    def _sub(m: re.Match) -> str:
        head, body, tail = m.group(1), m.group(2), m.group(3)
        # Keep everything before the table (e.g. image), replace table only
        table_pat = re.compile(
            r"\| model_group \| model_name \|.*?\|(?:\n\|[^\n]+\|)+",
            re.DOTALL,
        )
        new_body, n = table_pat.subn("\n\n" + new_md_table.rstrip() + "\n", body, count=1)
        if n != 1:
            raise ValueError(f"Could not find markdown table for summary={summary_substr!r}")
        return head + new_body + tail

    out, n = pattern.subn(_sub, readme_text, count=1)
    if n != 1:
        raise ValueError(f"Could not find <details> for summary={summary_substr!r}")
    return out
