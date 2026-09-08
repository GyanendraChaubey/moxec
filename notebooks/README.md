# MOXEC — Notebooks

Interactive, no-server alternatives to [`../full_pipeline/`](../full_pipeline/) for
running MOXEC on Colab or Kaggle's free tiers. All three notebooks share the same
core pipeline code (preprocessing, NSGA-II search, baselines, evaluation) and carry
the same two correctness fixes described in
[`../full_pipeline/README.md §6`](../full_pipeline/README.md#6-two-correctness-fixes-baked-into-this-script-worth-knowing-about).

If you have access to a plain server (or just want one long-running background job
instead of three interactive notebooks), use `full_pipeline/moxec_full_pipeline.py`
instead — it runs the full 12-dataset portfolio unattended and is the more robust,
actively-maintained path.

## Files

| Notebook | Purpose |
|---|---|
| `MOXEC_single_dataset_experiment.ipynb` | Runs the complete pipeline on **one** dataset. Good first run to sanity-check the environment (~20–40 min on free-tier CPU at the default `n_trials=100`, `n_seeds=3`). |
| `MOXEC_kaggle_full_pipeline.ipynb` | Runs the complete pipeline across **all 12** datasets in the portfolio (see the [root README](../README.md#dataset-portfolio)) in one Kaggle session. |
| `MOXEC_aggregate_results.ipynb` | Run **after** one or more `MOXEC_single_dataset_experiment.ipynb` runs. Collects each run's `moxec_results/<dataset_name>/summary.json` and produces cross-dataset hypervolume, Friedman/Nemenyi, and domain-contrast statistics. |

## How to run

1. Upload the notebook you want to Colab or Kaggle.
2. Run the install cells first (§0 / §1). Safe to re-run.
3. On Colab, mount Drive when prompted — this is what makes the Optuna search
   resumable if your session disconnects mid-run.
4. Run every remaining cell top to bottom.
5. Check the faithfulness sanity check (Spearman ρ) before trusting a full run —
   if it warns below 0.80, raise `faithfulness_sample_size` / `faithfulness_steps`
   in `CONFIG` first.
6. Results are saved under `moxec_results/<dataset_name>/`: trial dataframes, the
   Pareto front, hypervolume table, baseline results, and a `summary.json`.

### Running a different dataset (single-dataset notebook)

Change only the `CONFIG` cell (`uci_id`, `dataset_name`, `task_type`). Every
downstream cell reads from `CONFIG` and adapts automatically, including binary vs.
multiclass MCC/SHAP handling. See the dataset table in the
[root README](../README.md#dataset-portfolio) for the full 12-dataset portfolio
and UCI ids.

## Known-fixed issues (already applied in these notebooks)

1. `trials_to_dataframe` assumed every Optuna study had 3 objective values — broke
   on the single-objective TPE baseline. Fixed to handle both cases.
2. The TPE baseline's post-hoc faithfulness computation was passing raw
   (unpreprocessed) data as the SHAP background against a model fit on preprocessed
   features — a silent feature-count mismatch. Fixed to transform both train and
   test through the same fitted pipeline steps before computing Φ.

The UCI fetch (`ucimlrepo`), the AutoGluon baseline, and timing/latency numbers
depend on network access and a full dependency install — verify those specifically
on your first real Colab/Kaggle run.
