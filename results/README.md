# Results

The complete numerical output of the reported experiment: all 12 datasets,
3 seeds each, produced by `full_pipeline/moxec_full_pipeline.py`. This is the
full result set behind every table and figure in §2 of the root
[`README.md`](../README.md) and in the manuscript — included so a reviewer
(or anyone re-running the pipeline) can check specific numbers without
re-executing a 25–40 hour job first.

Scratch artifacts from the run — Optuna SQLite study databases and AutoGluon's
serialized model binaries (`_work/`, several GB) — are intentionally not
included here; they're reproducible by re-running the pipeline and aren't
needed to verify any reported result.

## Contents

| Folder | Contents |
|---|---|
| `paper_ready/` | One `summary.json` per dataset (best configs, seed-level and dataset-level results, all statistical tests) plus `FINAL_paper_support_summary.json` — the single reference file used to write up results. |
| `raw_results/` | Every trial from every seed's search (`<dataset>_all_trials_seed<N>.parquet`), not just the Pareto front. |
| `processed_results/` | Per-seed Pareto fronts and best-MCC-by-seed, derived from `raw_results/`. |
| `metrics/` | Hypervolume and baseline point-result tables. |
| `tables/` | CSV tables in the exact form used in the manuscript (portfolio hypervolume summary, per-objective breakdown, ablation A1/A2, front-navigation agreement). |
| `statistical_analysis/` | Wilcoxon signed-rank, Friedman + Nemenyi, faithfulness/latency-proxy validation, domain-contrast Mann-Whitney. |
| `figures/` | Every Pareto front plot (12 datasets × 3 seeds, PNG 300 DPI + PDF) plus the two portfolio-level plots (critical-difference diagram, hypervolume by dataset). A subset of these is embedded directly in the root README. |
| `pareto_analysis/` | Front-navigation picks: knee-point, cost-constrained best-MCC, and weighted-scalarization (MES) selections per dataset. |
| `run_manifest.csv` | Which datasets succeeded, when, and how long each took. |

## Regenerating or extending this

If you re-run the pipeline (e.g. to add a seed or a dataset), regenerate this
folder's aggregate tables/figures without re-running any search:

```bash
python full_pipeline/moxec_full_pipeline.py --aggregate-only --output-dir <your-output-dir>
```

then copy the same set of subfolders (excluding `_work/`, `_datasets/`, and
`run_log_*.txt`) from `<your-output-dir>/` here.
