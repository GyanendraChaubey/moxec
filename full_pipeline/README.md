# MOXEC Full Pipeline

Runs the complete MOXEC experiment (search + baselines + evaluation) across the
dataset portfolio in `DATASET_REGISTRY` and saves every result, figure, and table to
`./outputs/`.

The registry has two sources:
- **UCI (12 datasets, `M*`/`A*` keys)** — the original PALE-lean portfolio
  (`PALE_lean_protocol.md`), fetched via `ucimlrepo`. Unchanged; results reproduce as
  before.
- **KEEL (26 datasets, `K*` keys)** — downloaded on first use from
  [sci2s.ugr.es/keel](https://sci2s.ugr.es/keel), parsed from KEEL's native `.dat`
  format, and cached under `<output-dir>/_datasets_keel/`. Selected from all 75
  datasets in KEEL's "Standard Classification" catalog using the same suitability rule
  the original protocol used for UCI: n ≥ 900 rows, every class has ≥5 rows (required
  for `StratifiedKFold(5)`), not a duplicate of a UCI dataset already in the registry
  (KEEL's `thyroid`/`mushroom`/`satimage` are excluded on that basis), and not so large
  it makes the per-dataset search+SHAP+AutoML cost impractical (KEEL's `poker`,
  `kddcup`, `census`, `fars`, `connect-4`, `adult`, `shuttle`, `kr-vs-k`, and `abalone`
  are excluded on that basis). Of the 75, 26 pass and 49 are excluded (36 for n<900, 3
  as UCI duplicates, 2 for pathological class sparsity, 8 as too large/slow) — see the
  comment above `DATASET_REGISTRY` in `moxec_full_pipeline.py` for the full reasoning
  and the exact per-dataset breakdown.

## Setup

```bash
pip install -r requirements.txt
```

No extra dependencies are needed for KEEL — it's downloaded with the standard library
(`urllib`/`zipfile`), not a new package.

## Running

`--datasets` accepts either a group keyword or a comma-separated list of specific
registry keys from the tables below.

```bash
# Groups
python moxec_full_pipeline.py --datasets uci     # original 12 UCI datasets, ~25-40h
python moxec_full_pipeline.py --datasets keel    # all 26 KEEL datasets
python moxec_full_pipeline.py --datasets all     # all 38 -- expect several days

# Specific dataset(s) -- any key(s) from the tables below, UCI and KEEL freely mixed
python moxec_full_pipeline.py --datasets K13_phoneme                       # one KEEL dataset
python moxec_full_pipeline.py --datasets M6_diabetic_retinopathy           # one UCI dataset
python moxec_full_pipeline.py --datasets A1_dry_bean,K19_tic_tac_toe,M2_thyroid_disease
```

Unknown keys are rejected up front with the full valid-key list printed, so a typo
fails immediately rather than partway through a run.

Every run can be backgrounded (e.g. `nohup python moxec_full_pipeline.py --datasets keel &`)
or run in a persistent terminal session (`tmux`/`screen`), and resumes automatically if
interrupted — just re-run the same command with the same `--datasets`. Add
`--aggregate-only` to skip straight to rebuilding the cross-dataset summary/figures from
whatever `outputs/paper_ready/*_summary.json` files already exist, without launching any
new search.

For a fast sanity check before committing to a long run, cut every budget down, e.g.:

```bash
python moxec_full_pipeline.py --datasets K19_tic_tac_toe \
  --n-trials 5 --n-seeds 1 --cv-folds 2 --no-autogluon --baseline-time-budget-sec 5
```

`K07_letter` and `K08_magic` are the two largest KEEL additions (20k and 19k rows) and
will each take noticeably longer than the rest of the KEEL set at default settings.

### UCI datasets (`--datasets uci`)

| Key | Dataset | Domain | Task | n | d | classes |
|---|---|---|---|---|---|---|
| `M1_eeg_eye_state` | EEG Eye State | medical | binary | 14,980 | 14 | 2 |
| `M2_thyroid_disease` | Thyroid Disease (ann-thyroid) | medical | multiclass | 7,200 | 21 | 3 |
| `M3_aids_clinical_trials` | AIDS Clinical Trials Group Study 175 | medical | binary | 2,139 | 23 | 2 |
| `M4_cardiotocography` | Cardiotocography | medical | multiclass | 2,126 | 21 | 3 |
| `M5_obesity_levels` | Estimation of Obesity Levels | medical | multiclass | 2,111 | 16 | 7 |
| `M6_diabetic_retinopathy` | Diabetic Retinopathy (Debrecen) | medical | binary | 1,151 | 19 | 2 |
| `A1_dry_bean` | Dry Bean | agriculture | multiclass | 13,611 | 16 | 7 |
| `A2_mushroom` | Mushroom | agriculture | binary | 8,124 | 22 | 2 |
| `A3_wine_quality` | Wine Quality (red+white, binarized) | agriculture | binary | 6,497 | 11 | 2 |
| `A4_landsat_satellite` | Statlog (Landsat Satellite) | agriculture | multiclass | 6,435 | 36 | 6 |
| `A5_rice` | Rice (Cammeo and Osmancik) | agriculture | binary | 3,810 | 7 | 2 |
| `A6_raisin` | Raisin | agriculture | binary | 900 | 7 | 2 |

### KEEL datasets (`--datasets keel`)

| Key | Dataset | Domain | Task | n | d | classes |
|---|---|---|---|---|---|---|
| `K01_car` | Car Evaluation | consumer | multiclass | 1,728 | 6 | 4 |
| `K02_chess_krvkp` | Chess (King-Rook vs. King-Pawn) | game | binary | 3,196 | 36 | 2 |
| `K03_coil2000` | Insurance Company Benchmark (COIL 2000) | finance | binary | 9,822 | 85 | 2 |
| `K04_contraceptive` | Contraceptive Method Choice | social | multiclass | 1,473 | 9 | 3 |
| `K05_flare` | Solar Flare | physical | multiclass | 1,066 | 11 | 6 |
| `K06_german_credit` | German Credit (Statlog) | finance | binary | 1,000 | 20 | 2 |
| `K07_letter` | Letter Recognition | image | multiclass | 20,000 | 16 | 26 |
| `K08_magic` | MAGIC Gamma Telescope | physical | binary | 19,020 | 10 | 2 |
| `K09_marketing` | Marketing (Income Survey) | social | multiclass | 6,876 | 13 | 9 |
| `K10_optdigits` | Optical Recognition of Handwritten Digits | image | multiclass | 5,620 | 64 | 10 |
| `K11_page_blocks` | Page Blocks Classification | image | multiclass | 5,472 | 10 | 5 |
| `K12_penbased` | Pen-Based Recognition of Handwritten Digits | image | multiclass | 10,992 | 16 | 10 |
| `K13_phoneme` | Phoneme | signal | binary | 5,404 | 5 | 2 |
| `K14_ring` | Ring (synthetic) | synthetic | binary | 7,400 | 20 | 2 |
| `K15_segment` | Image Segmentation (Statlog) | image | multiclass | 2,310 | 19 | 7 |
| `K16_spambase` | Spambase | text | binary | 4,597 | 57 | 2 |
| `K17_splice` | Splice-Junction Gene Sequences | biology | multiclass | 3,190 | 60 | 3 |
| `K18_texture` | Texture | image | multiclass | 5,500 | 40 | 11 |
| `K19_tic_tac_toe` | Tic-Tac-Toe Endgame | game | binary | 958 | 9 | 2 |
| `K20_titanic` | Titanic Survival | social | binary | 2,201 | 3 | 2 |
| `K21_twonorm` | Twonorm (synthetic) | synthetic | binary | 7,400 | 20 | 2 |
| `K22_vowel` | Vowel Recognition (Deterding) | signal | multiclass | 990 | 13 | 11 |
| `K23_winequality_red` | Wine Quality — Red (multiclass) | agriculture | multiclass | 1,599 | 11 | 6 |
| `K24_winequality_white` | Wine Quality — White (multiclass) | agriculture | multiclass | 4,898 | 11 | 7 |
| `K25_yeast` | Yeast Protein Localization | biology | multiclass | 1,484 | 8 | 10 |
| `K26_banana` | Banana (synthetic) | synthetic | binary | 5,300 | 2 | 2 |

This table is generated from `DATASET_REGISTRY` in `moxec_full_pipeline.py` — if you add
or edit an entry there, update this table (or just run
`python -c "from moxec_full_pipeline import DATASET_REGISTRY as R; [print(k, v['display_name']) for k, v in R.items()]"`
to regenerate the key list).
