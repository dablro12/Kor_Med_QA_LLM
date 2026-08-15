# Benchmark Table Update (Bonsai) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ternary-Bonsai-27B and Bonsai-27B-1bit rows to all seven `benchmark/` csv/md/png artifacts and sync the matching README tables, without re-running inference.

**Architecture:** One-shot script `scripts/9_update_benchmark_tables.py` scans each dataset’s `*_detailed.parquet`, aggregates with existing `ClinicalQAEvaluator`, applies display-name overrides, writes csv/md, regenerates Nature-style scatter PNGs (logic transplanted from `notebook/result_test.ipynb`), then replaces markdown tables inside README `<details>` blocks.

**Tech Stack:** Python 3, pandas, pyarrow/parquet, matplotlib, seaborn, optional adjustText, pytest

## Global Constraints

- Display names exactly: `prism-ml` / `Ternary-Bonsai-27B` and `prism-ml` / `Bonsai-27B-1bit`
- Metrics columns unchanged: `model_group,model_name,accuracy (%),avg_time_per_token (s),mean_flops (GFlops)`
- Use parquet values as-is (including GGUF FLOPs stub); do not filter incomplete rows
- Sort: `model_group` asc, then `accuracy (%)` asc
- Do not modify `src/metrics.py` behavior
- Spec: `docs/superpowers/specs/2026-08-15-benchmark-table-update-design.md`

## File Structure

| File | Responsibility |
|---|---|
| `scripts/benchmark_table_lib.py` | Pure helpers: dataset registry, display names, summarize, write csv/md, README table sync, plot function |
| `scripts/9_update_benchmark_tables.py` | CLI entry (`--dry-run`), loops datasets, calls lib |
| `tests/test_benchmark_table_lib.py` | Unit tests for naming, md write, README sync, aggregation smoke on tiny fixture |
| `benchmark/*.{csv,md,png}` | Regenerated outputs (7 stems) |
| `README.md` | Table bodies inside 7 `<details>` blocks updated |

---

### Task 1: Pure helpers + unit tests (naming, md, README sync)

**Files:**
- Create: `scripts/benchmark_table_lib.py`
- Create: `tests/test_benchmark_table_lib.py`
- Create: `tests/fixtures/tiny_detailed.parquet` (generated in test setup, not committed binary — build in test)

**Interfaces:**
- Produces:
  - `DISPLAY_NAME_OVERRIDES: dict[str, tuple[str, str]]`
  - `DATASETS: list[dict]` with keys `results_rel`, `stem`, `readme_summary_substr`
  - `def split_model_folder(folder: str) -> tuple[str, str]`
  - `def display_frame(summary_df: pd.DataFrame) -> pd.DataFrame`  # adds model_group, model_name, rounded metric cols
  - `def dataframe_to_markdown(df: pd.DataFrame, cols: list[str]) -> str`
  - `def replace_readme_table(readme_text: str, summary_substr: str, new_md_table: str) -> str`

- [ ] **Step 1: Write failing tests**

Create `tests/test_benchmark_table_lib.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL (import / missing module)**

```bash
cd /workspace && python -m pytest tests/test_benchmark_table_lib.py -v
```

Expected: `ModuleNotFoundError` or collection error for `scripts.benchmark_table_lib`

- [ ] **Step 3: Implement `scripts/benchmark_table_lib.py` (helpers only, no plot yet)**

```python
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
```

Also add empty `scripts/__init__.py` **only if** needed for imports; prefer running tests with `PYTHONPATH=/workspace` and importing `scripts.benchmark_table_lib`. If package init is awkward, put tests’ import path as:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_table_lib import ...
```

- [ ] **Step 4: Re-run tests — expect PASS**

```bash
cd /workspace && PYTHONPATH=/workspace python -m pytest tests/test_benchmark_table_lib.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_table_lib.py tests/test_benchmark_table_lib.py
git commit -m "$(cat <<'EOF'
Add benchmark table helper lib and unit tests for Bonsai display names.

EOF
)"
```

---

### Task 2: Aggregate parquets + write csv/md

**Files:**
- Modify: `scripts/benchmark_table_lib.py`
- Modify: `tests/test_benchmark_table_lib.py`
- Create: `scripts/9_update_benchmark_tables.py` (CLI stub that can `--dry-run` csv/md only)

**Interfaces:**
- Consumes: `ClinicalQAEvaluator` from `src.metrics`
- Produces:
  - `def summarize_dataset(results_dir: Path) -> pd.DataFrame`  # columns: model, accuracy (%), avg_time_per_token (s), mean_flops, …
  - `def write_csv_md(display_df: pd.DataFrame, out_dir: Path, stem: str) -> None`

- [ ] **Step 1: Write failing tests for summarize + write**

Append to `tests/test_benchmark_table_lib.py`:

```python
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.benchmark_table_lib import summarize_dataset, write_csv_md, DISPLAY_COLS


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
```

- [ ] **Step 2: Run new tests — expect FAIL**

```bash
cd /workspace && PYTHONPATH=/workspace python -m pytest tests/test_benchmark_table_lib.py::test_summarize_dataset_accuracy tests/test_benchmark_table_lib.py::test_write_csv_md -v
```

Expected: FAIL (`summarize_dataset` / `write_csv_md` not defined)

- [ ] **Step 3: Implement summarize + write helpers**

Add to `scripts/benchmark_table_lib.py`:

```python
from glob import glob
from pathlib import Path
import sys

# allow `from src.metrics import ClinicalQAEvaluator` when run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import ClinicalQAEvaluator


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
```

- [ ] **Step 4: Create CLI that dry-runs one dataset aggregation to stdout**

`scripts/9_update_benchmark_tables.py`:

```python
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
```

- [ ] **Step 5: Run unit tests + dry-run on real data**

```bash
cd /workspace && PYTHONPATH=/workspace python -m pytest tests/test_benchmark_table_lib.py -v
cd /workspace && PYTHONPATH=/workspace python scripts/9_update_benchmark_tables.py --dry-run | head -80
```

Expected: tests PASS; dry-run prints tables including `Ternary-Bonsai-27B` and `Bonsai-27B-1bit` for SNUH.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_table_lib.py scripts/9_update_benchmark_tables.py tests/test_benchmark_table_lib.py
git commit -m "$(cat <<'EOF'
Add parquet aggregation and csv/md writers for benchmark refresh.

EOF
)"
```

---

### Task 3: Plot PNG generation

**Files:**
- Modify: `scripts/benchmark_table_lib.py` (add `plot_medical_llm_benchmark_nature`, `set_nature_style`)
- Modify: `scripts/9_update_benchmark_tables.py` (call plot unless `--skip-plot` / `--dry-run`)

**Interfaces:**
- Produces: `def plot_medical_llm_benchmark_nature(df: pd.DataFrame, save_path: str | Path, ...) -> None`
- Input `df` must include columns `model` (folder id OK), `mean_flops`, `accuracy (%)` — **and** after display mapping, plot should label with display short names. Prefer building plot_df from `display_frame` output plus raw `mean_flops` before GFlops conversion, OR pass summary pre-display and apply overrides inside plot via `split_model_folder`.

Recommended: plot from **raw summary** (`model`, `mean_flops`, `accuracy (%)`) and set:

```python
plot_df["Group"] = [split_model_folder(m)[0] for m in plot_df["model"]]
plot_df["Short Name"] = [split_model_folder(m)[1] for m in plot_df["model"]]
```

so Bonsai labels are correct.

- [ ] **Step 1: Transplant plot function from notebook**

Copy `set_nature_style` + `plot_medical_llm_benchmark_nature` from `notebook/result_test.ipynb` into `benchmark_table_lib.py`.

Critical edits vs notebook:

1. Use `split_model_folder` for Group / Short Name (not naive `split("_")`).
2. Guard adjustText:

```python
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
```

3. Do **not** call `plt.show()` in the script path (headless); only `savefig` + `plt.close(fig)`.
4. Extend palette list if needed (12+ colors already in notebook).

- [ ] **Step 2: Wire CLI to write PNG**

In `main()`, after `write_csv_md`:

```python
        if not args.dry_run and not args.skip_plot:
            from scripts.benchmark_table_lib import plot_medical_llm_benchmark_nature
            plot_medical_llm_benchmark_nature(
                summary,
                save_path=BENCH / f"{ds['stem']}.png",
            )
```

- [ ] **Step 3: Smoke one dataset plot**

```bash
cd /workspace && PYTHONPATH=/workspace MPLBACKEND=Agg python - <<'PY'
from pathlib import Path
from scripts.benchmark_table_lib import summarize_dataset, plot_medical_llm_benchmark_nature
root = Path('/workspace')
summary = summarize_dataset(root/'kor_med_opendataset/results/snuh_ClinicalQA_benchmark')
out = root/'benchmark/_smoke_snuh_plot.png'
plot_medical_llm_benchmark_nature(summary, save_path=out)
assert out.exists() and out.stat().st_size > 10_000
print('OK', out.stat().st_size)
out.unlink()
PY
```

Expected: `OK` and file size > 10k; no exception.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_table_lib.py scripts/9_update_benchmark_tables.py
git commit -m "$(cat <<'EOF'
Add Nature-style benchmark scatter plots with Bonsai display labels.

EOF
)"
```

---

### Task 4: README sync + full refresh + verification

**Files:**
- Modify: `scripts/benchmark_table_lib.py` (optional `sync_readme`)
- Modify: `scripts/9_update_benchmark_tables.py`
- Modify: `benchmark/*.{csv,md,png}` (generated)
- Modify: `README.md`

**Interfaces:**
- Produces: `def sync_readme(readme_path: Path, stem_to_md: dict[str, str]) -> None`  
  where values are markdown table strings keyed by `readme_summary_substr` OR iterate `DATASETS` and read `benchmark/<stem>.md`.

- [ ] **Step 1: Implement README sync in lib + CLI**

```python
def sync_readme(readme_path: Path, benchmark_dir: Path, datasets=DATASETS) -> None:
    text = readme_path.read_text(encoding="utf-8")
    for ds in datasets:
        md_table = (benchmark_dir / f"{ds['stem']}.md").read_text(encoding="utf-8")
        text = replace_readme_table(text, ds["readme_summary_substr"], md_table)
    readme_path.write_text(text, encoding="utf-8")
```

Call at end of CLI when `not args.dry_run and not args.skip_readme`.

- [ ] **Step 2: Add verification assertions in CLI (or separate check)**

After writing all artifacts:

```python
def verify_bonsai_rows(benchmark_dir: Path, datasets=DATASETS) -> None:
    required = {("prism-ml", "Ternary-Bonsai-27B"), ("prism-ml", "Bonsai-27B-1bit")}
    for ds in datasets:
        df = pd.read_csv(benchmark_dir / f"{ds['stem']}.csv")
        got = set(zip(df["model_group"], df["model_name"]))
        missing = required - got
        if missing:
            raise AssertionError(f"{ds['stem']} missing {missing}")
        png = benchmark_dir / f"{ds['stem']}.png"
        if not png.exists() or png.stat().st_size < 10_000:
            raise AssertionError(f"bad/missing png for {ds['stem']}")
```

Cross-check accuracy for SNUH Ternary ≈ 70.33 from parquet (tolerance 0.01).

- [ ] **Step 3: Dry-run full, then real run**

```bash
cd /workspace && PYTHONPATH=/workspace python scripts/9_update_benchmark_tables.py --dry-run | tee /tmp/bench_dryrun.txt
grep -c 'Ternary-Bonsai-27B' /tmp/bench_dryrun.txt
# expect >= 7

cd /workspace && PYTHONPATH=/workspace MPLBACKEND=Agg python scripts/9_update_benchmark_tables.py
```

Expected: 7 csv/md/png updated; README contains both Bonsai names in all 7 details sections.

- [ ] **Step 4: Manual spot-check commands**

```bash
# both models in every csv
for f in /workspace/benchmark/*.csv; do
  echo "== $(basename $f)"
  rg 'prism-ml' "$f"
done

# README counts
rg -c 'Ternary-Bonsai-27B' /workspace/README.md
rg -c 'Bonsai-27B-1bit' /workspace/README.md
# expect 7 each

# SNUH accuracy sanity
python - <<'PY'
import pandas as pd
from pathlib import Path
from src.metrics import ClinicalQAEvaluator
p=Path('/workspace/kor_med_opendataset/results/snuh_ClinicalQA_benchmark/prism-ml_Ternary-Bonsai-27B-gguf/prism-ml_Ternary-Bonsai-27B-gguf_detailed.parquet')
acc=ClinicalQAEvaluator(str(p)).summary()['accuracy (%)'].iloc[0]
csv=pd.read_csv('/workspace/benchmark/snuh_ClinicalQA_benchmark.csv')
row=csv[(csv.model_group=='prism-ml')&(csv.model_name=='Ternary-Bonsai-27B')].iloc[0]
assert abs(row['accuracy (%)']-acc)<1e-6
print('SNUH Ternary OK', acc)
PY
```

- [ ] **Step 5: Commit artifacts**

```bash
git add scripts/benchmark_table_lib.py scripts/9_update_benchmark_tables.py \
  benchmark/*.csv benchmark/*.md benchmark/*.png README.md
git commit -m "$(cat <<'EOF'
Publish Ternary and 1-bit Bonsai rows in benchmark tables and README.

EOF
)"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Add 2 Bonsai models to csv/md/png | Tasks 2–4 |
| Display name overrides | Task 1 |
| Metrics as-is via ClinicalQAEvaluator | Task 2 |
| Sort group + accuracy | Task 1 `display_frame` |
| Nature plot with display short names | Task 3 |
| README table sync, keep images | Task 4 |
| `--dry-run` | Tasks 2–4 CLI |
| Verification asserts | Task 4 |
| No re-inference / no metrics.py change | All tasks |

## Placeholder / consistency check

- No TBD left
- Function names consistent: `split_model_folder`, `display_frame`, `summarize_dataset`, `write_csv_md`, `replace_readme_table`, `sync_readme`, `plot_medical_llm_benchmark_nature`
- `DATASETS` registry is single source for stems + README substrings
