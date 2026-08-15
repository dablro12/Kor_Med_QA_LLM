from __future__ import annotations

import re
import sys
from glob import glob
from itertools import cycle
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import ClinicalQAEvaluator

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


def summarize_dataset(results_dir: Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = []
    for parquet_file in sorted(glob(str(results_dir / "*" / "*_detailed.parquet"))):
        folder = Path(parquet_file).parent.name
        ev = ClinicalQAEvaluator(parquet_file)
        summary_df = ev.summary()
        summary_df.insert(0, "model", folder)
        rows.append(summary_df)
    if not rows:
        raise FileNotFoundError(f"No *_detailed.parquet under {results_dir}")
    return pd.concat(rows, ignore_index=True)


def write_csv_md(display_df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    display_df[DISPLAY_COLS].to_csv(out_dir / f"{stem}.csv", index=False)
    md = dataframe_to_markdown(display_df, DISPLAY_COLS)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")


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


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
            "figure.dpi": 300,
        }
    )


def plot_medical_llm_benchmark_nature(
    df: pd.DataFrame,
    x_col: str = "mean_flops",
    y_col: str = "accuracy (%)",
    model_col: str = "model",
    figsize: tuple[float, float] = (11, 7),
    dpi: int = 300,
    title: str = "Medical Domain Evaluation of Open-Source Small Large Language Models",
    save_path: str | Path | None = "NEJM_style_plot.png",
) -> None:
    set_nature_style()

    plot_df = df.copy()
    plot_df[x_col] = plot_df[x_col] / 1e9
    plot_df["Group"] = [split_model_folder(m)[0] for m in plot_df[model_col]]
    plot_df["Short Name"] = [split_model_folder(m)[1] for m in plot_df[model_col]]

    unique_groups = plot_df["Group"].unique()
    palette_colors = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#ff7f0e",
        "#bcbd22",
        "#e377c2",
        "#7f7f7f",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]
    palette = {g: c for g, c in zip(unique_groups, cycle(palette_colors))}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    sorted_df = plot_df.sort_values(by=x_col)
    pareto_points = []
    current_max = -np.inf
    for _, row in sorted_df.iterrows():
        if row[y_col] > current_max:
            pareto_points.append((row[x_col], row[y_col]))
            current_max = row[y_col]

    if len(pareto_points) > 0:
        px, py = zip(*pareto_points)
        ax.plot(px, py, "--", color="#777777", linewidth=1.2, alpha=0.7)

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue="Group",
        palette=palette,
        style="Group",
        markers=True,
        s=140,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Computational Cost (GFLOPs, log-scale)", fontsize=14, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model Family", frameon=False, fontsize=11, title_fontsize=12)

    texts = []
    for _, row in plot_df.iterrows():
        t = ax.text(
            row[x_col],
            row[y_col],
            row["Short Name"],
            fontsize=10,
            color="black",
            ha="center",
            va="bottom",
        )
        texts.append(t)

    if HAS_ADJUST_TEXT:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
            force_text=(0.3, 0.4),
        )

    ax.margins(x=0.15, y=0.1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✔ Saved to: {save_path}")

    plt.close(fig)
