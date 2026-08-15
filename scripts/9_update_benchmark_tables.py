#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark_table_lib import (
    DATASETS,
    DISPLAY_COLS,
    dataframe_to_markdown,
    display_frame,
    summarize_dataset,
    write_csv_md,
)

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


if __name__ == "__main__":
    main()
