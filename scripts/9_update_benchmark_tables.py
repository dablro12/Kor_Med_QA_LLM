#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.benchmark_table_lib import (
    DATASETS,
    DISPLAY_COLS,
    dataframe_to_markdown,
    display_frame,
    plot_medical_llm_benchmark_nature,
    summarize_dataset,
    sync_readme,
    verify_bonsai_rows,
    write_csv_md,
)
from src.metrics import ClinicalQAEvaluator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kor_med_opendataset" / "results"
BENCH = ROOT / "benchmark"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-plot", action="store_true", help="csv/md/README only")
    ap.add_argument("--skip-readme", action="store_true")
    args = ap.parse_args()

    for ds in DATASETS:
        results_dir = RESULTS / ds["results_rel"]
        summary = summarize_dataset(results_dir)
        disp = display_frame(summary)
        print(f"\n===== {ds['stem']} ({len(disp)} models) =====")
        print(dataframe_to_markdown(disp, DISPLAY_COLS))
        # Task 2 only writes csv/md; plot/README in later tasks
        if not args.dry_run:
            write_csv_md(disp, BENCH, ds["stem"])
            print(f"wrote {ds['stem']}.csv/md")
        if not args.dry_run and not args.skip_plot:
            plot_medical_llm_benchmark_nature(
                summary,
                save_path=BENCH / f"{ds['stem']}.png",
            )

    if not args.dry_run and not args.skip_readme:
        sync_readme(ROOT / "README.md", BENCH)
        print("synced README.md")

    if not args.dry_run:
        verify_bonsai_rows(BENCH)
        parquet = (
            RESULTS
            / "snuh_ClinicalQA_benchmark"
            / "prism-ml_Ternary-Bonsai-27B-gguf"
            / "prism-ml_Ternary-Bonsai-27B-gguf_detailed.parquet"
        )
        acc = ClinicalQAEvaluator(str(parquet)).summary()["accuracy (%)"].iloc[0]
        csv = pd.read_csv(BENCH / "snuh_ClinicalQA_benchmark.csv")
        row = csv[
            (csv.model_group == "prism-ml") & (csv.model_name == "Ternary-Bonsai-27B")
        ].iloc[0]
        if abs(float(row["accuracy (%)"]) - float(acc)) > 0.01:
            raise AssertionError(
                f"SNUH Ternary accuracy mismatch csv={row['accuracy (%)']} parquet={acc}"
            )
        print("verified Bonsai rows + pngs + SNUH Ternary accuracy")


if __name__ == "__main__":
    main()
