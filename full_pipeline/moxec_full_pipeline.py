#!/usr/bin/env python3
"""
moxec_full_pipeline.py

Standalone server script version of MOXEC_kaggle_full_pipeline.ipynb -- runs the full
MOXEC pipeline (leakage-safe preprocessing, 9-model NSGA-II CASH search over 3
objectives, 4 baselines, hypervolume/Wilcoxon evaluation, cross-dataset aggregation)
across the dataset portfolio in DATASET_REGISTRY -- the original 12-dataset UCI
PALE-lean portfolio plus 26 additional KEEL-sourced datasets (downloaded on first use
from sci2s.ugr.es/keel, no extra dependency required) -- unattended, on a plain server.

Ported verbatim from the notebook version, including two fixes discovered during the
local single-dataset runs that matter for correctness everywhere in the portfolio:
  1. MLP architecture encoding as a string key ("64_32"), never a raw tuple -- a raw
     tuple round-trips through SQLite's JSON encoding as a list, which crashes TPE
     when it tries to reconstruct historical trial values against the distribution.
  2. Hypervolume normalization bounds built from the union of every method's
     evaluated points, including TPE's -- omitting TPE lets its (reliably higher) raw
     MCC exceed the bounds and inflates its hypervolume past the indicator's
     theoretical maximum.

USAGE
-----
    pip install -r requirements.txt
    python moxec_full_pipeline.py --datasets uci          # original 12 UCI datasets
    python moxec_full_pipeline.py --datasets keel         # the 26 KEEL additions
    python moxec_full_pipeline.py --datasets all          # all 38 (expect several days)
    python moxec_full_pipeline.py --datasets M6_diabetic_retinopathy,A1_dry_bean
    python moxec_full_pipeline.py --aggregate-only        # skip the run, just aggregate
    python moxec_full_pipeline.py --n-trials 50 --n-seeds 2 --no-autogluon  # a fast smoke test

Every dataset's outputs are saved to disk the moment that dataset finishes (not
deferred to the end of the whole run), and each dataset is wrapped in its own
try/except so one failure does not lose the rest of a long batch. All stdout/stderr is
mirrored to a timestamped log file under --output-dir in addition to the console, so a
run submitted to a job scheduler (nohup, systemd, Slurm, etc.) still leaves a complete
record even if the console output itself isn't captured elsewhere.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import io
import sys
import json
import time
import math
import zipfile
import argparse
import traceback
import logging
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

# Headless-server-safe plotting backend -- must be set before pyplot is imported
# anywhere, or matplotlib will try (and fail) to find a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd

# All third-party, non-stdlib dependencies are imported inside this try/except so a
# missing package produces one clear, actionable message instead of a raw traceback
# from wherever the first missing import happens to occur.
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score, accuracy_score

    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    import xgboost as xgb
    import lightgbm as lgb

    import shap
    import optuna
    from optuna.samplers import NSGAIISampler, RandomSampler, TPESampler

    from pymoo.indicators.hv import HV
    from scipy.stats import wilcoxon, spearmanr, friedmanchisquare, mannwhitneyu

    import scikit_posthocs  # noqa: F401 -- checked here (not just at first use in
                             # run_cross_dataset_aggregation) so this is caught up front too
    import ucimlrepo        # noqa: F401 -- same reasoning; actually used inside
                             # load_and_validate_dataset via `from ucimlrepo import fetch_ucirepo`
    import flaml             # noqa: F401 -- actually used inside run_flaml_baseline
except ImportError as e:
    print(f"ERROR: missing required package -- {e}")
    print("Install everything with:  pip install -r requirements.txt")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.seterr(all="ignore")


class Tee:
    """Duplicates writes to two streams -- used so every print() in this script
    (unchanged from the notebook version) lands in both the console and a
    persistent log file, without needing to convert hundreds of print() calls to
    a logging call by hand."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()



# ======================================================================

# Dataset registry (ported verbatim from the notebook)

# ======================================================================

# domain: used later for the Mann-Whitney domain contrast, which specifically compares
# rows tagged "medical" vs "agriculture" -- any other domain string (e.g. "finance",
# "image", "game") is carried through the pipeline and reported, but simply isn't a
# member of that one pairwise test.
# expected_n / expected_d / expected_classes: from PALE_lean_protocol.md §5 for the
# original 12 (UCI); hand-verified against the actual downloaded file for the 26 KEEL
# additions below (see the "K" keys) -- used for sanity-checking, not enforcement,
# except task_type (binary/multiclass) which is enforced (§ load_and_validate_dataset).
# source: "uci" (via ucimlrepo, keyed by uci_id) or "keel" (downloaded + parsed from
# sci2s.ugr.es/keel by keel_name -- see fetch_keel_dataset / _parse_keel_dat below).
DATASET_REGISTRY = {
    # ---- Original 12 -- UCI, PALE_lean_protocol.md §5 (untouched, for reproducibility) ----
    "M1_eeg_eye_state":          {"source": "uci", "uci_id": 264, "display_name": "EEG Eye State",
                                    "domain": "medical", "task_type": "binary",
                                    "expected_n": 14980, "expected_d": 14, "expected_classes": 2},
    "M2_thyroid_disease":        {"source": "uci", "uci_id": 102, "display_name": "Thyroid Disease (ann-thyroid)",
                                    "domain": "medical", "task_type": "multiclass",
                                    "expected_n": 7200, "expected_d": 21, "expected_classes": 3},
    "M3_aids_clinical_trials":   {"source": "uci", "uci_id": 890, "display_name": "AIDS Clinical Trials Group Study 175",
                                    "domain": "medical", "task_type": "binary",
                                    "expected_n": 2139, "expected_d": 23, "expected_classes": 2},
    "M4_cardiotocography":       {"source": "uci", "uci_id": 193, "display_name": "Cardiotocography",
                                    "domain": "medical", "task_type": "multiclass",
                                    "expected_n": 2126, "expected_d": 21, "expected_classes": 3},
    "M5_obesity_levels":         {"source": "uci", "uci_id": 544, "display_name": "Estimation of Obesity Levels",
                                    "domain": "medical", "task_type": "multiclass",
                                    "expected_n": 2111, "expected_d": 16, "expected_classes": 7},
    "M6_diabetic_retinopathy":   {"source": "uci", "uci_id": 329, "display_name": "Diabetic Retinopathy (Debrecen)",
                                    "domain": "medical", "task_type": "binary",
                                    "expected_n": 1151, "expected_d": 19, "expected_classes": 2},
    "A1_dry_bean":               {"source": "uci", "uci_id": 602, "display_name": "Dry Bean",
                                    "domain": "agriculture", "task_type": "multiclass",
                                    "expected_n": 13611, "expected_d": 16, "expected_classes": 7},
    "A2_mushroom":                {"source": "uci", "uci_id": 73, "display_name": "Mushroom",
                                    "domain": "agriculture", "task_type": "binary",
                                    "expected_n": 8124, "expected_d": 22, "expected_classes": 2},
    "A3_wine_quality":            {"source": "uci", "uci_id": 186, "display_name": "Wine Quality (red+white, binarized)",
                                    "domain": "agriculture", "task_type": "binary",
                                    "expected_n": 6497, "expected_d": 11, "expected_classes": 2,
                                    "binarize_threshold": 6},
    "A4_landsat_satellite":       {"source": "uci", "uci_id": 146, "display_name": "Statlog (Landsat Satellite)",
                                    "domain": "agriculture", "task_type": "multiclass",
                                    "expected_n": 6435, "expected_d": 36, "expected_classes": 6},
    "A5_rice":                     {"source": "uci", "uci_id": 545, "display_name": "Rice (Cammeo and Osmancik)",
                                    "domain": "agriculture", "task_type": "binary",
                                    "expected_n": 3810, "expected_d": 7, "expected_classes": 2},
    "A6_raisin":                   {"source": "uci", "uci_id": 850, "display_name": "Raisin (substitutes Pumpkin Seeds -- protocol day-1 fallback)",
                                    "domain": "agriculture", "task_type": "binary",
                                    "expected_n": 900, "expected_d": 7, "expected_classes": 2},

    # ---- 26 KEEL additions -- github discussion: "make this run on KEEL datasets" ----
    # Selection criteria, applied identically to every one of the 75 datasets in KEEL's
    # "Standard Classification" catalog (sci2s.ugr.es/keel): n >= 900 (the protocol's own floor --
    # below that the Pareto front doesn't reproduce across seeds, PALE_lean_protocol.md
    # §5), classification (not regression), StratifiedKFold(5)-safe (every class has
    # >=5 rows -- this alone excludes "nursery", whose "recommend" class has only 2),
    # and not already a duplicate of one of the 12 above under a different name (this
    # excludes KEEL's "thyroid" [=M2], "mushroom" [=A2, but missing-rows dropped so a
    # different n], "satimage" [=A4]). Also excluded as impractical at this pipeline's
    # per-dataset cost (NSGA-II CASH search + SHAP + 2 AutoML baselines, x3 seeds):
    # anything >~20-30k rows (kr-vs-k, shuttle, adult, census, kddcup, fars, connect-4,
    # poker) and "abalone" (28 classes, most with a handful of rows -- same failure mode
    # as nursery). Domain is tagged honestly by real-world topic, not forced into
    # medical/agriculture -- the two winequality sets are genuinely agriculture (food
    # chemistry), extending that side of the existing domain contrast; no remaining
    # KEEL "Standard Classification" set is both genuinely medical AND >=900 rows
    # (mammographic, the closest candidate, is 830), so the medical side stays at 6.
    # Final tally out of the 75: 26 included below; 49 excluded (36 for n<900, 3 as
    # UCI duplicates, 2 for pathological class sparsity, 8 as too large/slow).
    "K01_car":                    {"source": "keel", "keel_name": "car", "display_name": "Car Evaluation",
                                    "domain": "consumer", "task_type": "multiclass",
                                    "expected_n": 1728, "expected_d": 6, "expected_classes": 4},
    "K02_chess_krvkp":            {"source": "keel", "keel_name": "chess", "display_name": "Chess (King-Rook vs. King-Pawn)",
                                    "domain": "game", "task_type": "binary",
                                    "expected_n": 3196, "expected_d": 36, "expected_classes": 2},
    "K03_coil2000":               {"source": "keel", "keel_name": "coil2000", "display_name": "Insurance Company Benchmark (COIL 2000)",
                                    "domain": "finance", "task_type": "binary",
                                    "expected_n": 9822, "expected_d": 85, "expected_classes": 2},
    "K04_contraceptive":          {"source": "keel", "keel_name": "contraceptive", "display_name": "Contraceptive Method Choice",
                                    "domain": "social", "task_type": "multiclass",
                                    "expected_n": 1473, "expected_d": 9, "expected_classes": 3},
    "K05_flare":                  {"source": "keel", "keel_name": "flare", "display_name": "Solar Flare",
                                    "domain": "physical", "task_type": "multiclass",
                                    "expected_n": 1066, "expected_d": 11, "expected_classes": 6},
    "K06_german_credit":          {"source": "keel", "keel_name": "german", "display_name": "German Credit (Statlog)",
                                    "domain": "finance", "task_type": "binary",
                                    "expected_n": 1000, "expected_d": 20, "expected_classes": 2},
    "K07_letter":                 {"source": "keel", "keel_name": "letter", "display_name": "Letter Recognition",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 20000, "expected_d": 16, "expected_classes": 26},
    "K08_magic":                  {"source": "keel", "keel_name": "magic", "display_name": "MAGIC Gamma Telescope",
                                    "domain": "physical", "task_type": "binary",
                                    "expected_n": 19020, "expected_d": 10, "expected_classes": 2},
    "K09_marketing":              {"source": "keel", "keel_name": "marketing", "display_name": "Marketing (Income Survey)",
                                    "domain": "social", "task_type": "multiclass",
                                    "expected_n": 6876, "expected_d": 13, "expected_classes": 9},
    "K10_optdigits":              {"source": "keel", "keel_name": "optdigits", "display_name": "Optical Recognition of Handwritten Digits",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 5620, "expected_d": 64, "expected_classes": 10},
    "K11_page_blocks":            {"source": "keel", "keel_name": "page-blocks", "display_name": "Page Blocks Classification",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 5472, "expected_d": 10, "expected_classes": 5},
    "K12_penbased":               {"source": "keel", "keel_name": "penbased", "display_name": "Pen-Based Recognition of Handwritten Digits",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 10992, "expected_d": 16, "expected_classes": 10},
    "K13_phoneme":                {"source": "keel", "keel_name": "phoneme", "display_name": "Phoneme",
                                    "domain": "signal", "task_type": "binary",
                                    "expected_n": 5404, "expected_d": 5, "expected_classes": 2},
    "K14_ring":                   {"source": "keel", "keel_name": "ring", "display_name": "Ring (synthetic)",
                                    "domain": "synthetic", "task_type": "binary",
                                    "expected_n": 7400, "expected_d": 20, "expected_classes": 2},
    "K15_segment":                {"source": "keel", "keel_name": "segment", "display_name": "Image Segmentation (Statlog)",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 2310, "expected_d": 19, "expected_classes": 7},
    "K16_spambase":               {"source": "keel", "keel_name": "spambase", "display_name": "Spambase",
                                    "domain": "text", "task_type": "binary",
                                    "expected_n": 4597, "expected_d": 57, "expected_classes": 2},
    "K17_splice":                 {"source": "keel", "keel_name": "splice", "display_name": "Splice-Junction Gene Sequences",
                                    "domain": "biology", "task_type": "multiclass",
                                    "expected_n": 3190, "expected_d": 60, "expected_classes": 3},
    "K18_texture":                {"source": "keel", "keel_name": "texture", "display_name": "Texture",
                                    "domain": "image", "task_type": "multiclass",
                                    "expected_n": 5500, "expected_d": 40, "expected_classes": 11},
    "K19_tic_tac_toe":            {"source": "keel", "keel_name": "tic-tac-toe", "display_name": "Tic-Tac-Toe Endgame",
                                    "domain": "game", "task_type": "binary",
                                    "expected_n": 958, "expected_d": 9, "expected_classes": 2},
    "K20_titanic":                {"source": "keel", "keel_name": "titanic", "display_name": "Titanic Survival",
                                    "domain": "social", "task_type": "binary",
                                    "expected_n": 2201, "expected_d": 3, "expected_classes": 2},
    "K21_twonorm":                {"source": "keel", "keel_name": "twonorm", "display_name": "Twonorm (synthetic)",
                                    "domain": "synthetic", "task_type": "binary",
                                    "expected_n": 7400, "expected_d": 20, "expected_classes": 2},
    "K22_vowel":                  {"source": "keel", "keel_name": "vowel", "display_name": "Vowel Recognition (Deterding)",
                                    "domain": "signal", "task_type": "multiclass",
                                    "expected_n": 990, "expected_d": 13, "expected_classes": 11},
    "K23_winequality_red":        {"source": "keel", "keel_name": "winequality-red", "display_name": "Wine Quality -- Red (multiclass)",
                                    "domain": "agriculture", "task_type": "multiclass",
                                    "expected_n": 1599, "expected_d": 11, "expected_classes": 6},
    "K24_winequality_white":      {"source": "keel", "keel_name": "winequality-white", "display_name": "Wine Quality -- White (multiclass)",
                                    "domain": "agriculture", "task_type": "multiclass",
                                    "expected_n": 4898, "expected_d": 11, "expected_classes": 7,
                                    "note": "smallest class has 5 rows -- SMOTE (k_neighbors=5) will prune "
                                            "some trials on this class in-fold; StratifiedKFold(5) itself is fine"},
    "K25_yeast":                  {"source": "keel", "keel_name": "yeast", "display_name": "Yeast Protein Localization",
                                    "domain": "biology", "task_type": "multiclass",
                                    "expected_n": 1484, "expected_d": 8, "expected_classes": 10,
                                    "note": "smallest class (ERL) has 5 rows -- same SMOTE caveat as K24"},
    "K26_banana":                 {"source": "keel", "keel_name": "banana", "display_name": "Banana (synthetic)",
                                    "domain": "synthetic", "task_type": "binary",
                                    "expected_n": 5300, "expected_d": 2, "expected_classes": 2},
}

def _dataset_source_label(entry):
    if entry.get("source", "uci") == "uci":
        return f"uci_id={entry['uci_id']}"
    return f"keel='{entry['keel_name']}'"

print(f"Registry loaded: {len(DATASET_REGISTRY)} datasets.")
for key, entry in DATASET_REGISTRY.items():
    print(f"  {key}: {_dataset_source_label(entry):<20} expected {entry['expected_n']:>6} x {entry['expected_d']:<3} "
          f"({entry['expected_classes']}-class, {entry['domain']})")


# ======================================================================

# Pipeline functions (ported verbatim from the notebook -- preprocessing,

# search space, objectives, faithfulness sanity check, NSGA-II search,

# Pareto extraction/plotting, latency validation, front navigation,

# baselines, hypervolume/Wilcoxon evaluation, dataset loading)

# ======================================================================

def build_preprocessing_steps(trial, n_features):
    '''Suggest a leakage-safe preprocessing configuration for this trial.
    Returns a list of (name, transformer) tuples for imblearn.Pipeline.'''
    steps = []

    scaler_choice = trial.suggest_categorical("scaler", ["none", "standard", "minmax", "robust"])
    if scaler_choice == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_choice == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_choice == "robust":
        steps.append(("scaler", RobustScaler()))

    fs_choice = trial.suggest_categorical("feature_selection", ["none", "kbest"])
    if fs_choice == "kbest":
        k_frac = trial.suggest_float("kbest_frac", 0.5, 1.0)
        k = max(1, int(round(k_frac * n_features)))
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k)))

    imb_choice = trial.suggest_categorical("imbalance", ["none", "smote"])
    if imb_choice == "smote":
        steps.append(("smote", SMOTE(random_state=0)))

    return steps

print("build_preprocessing_steps defined.")


MLP_ARCHS = {"32": (32,), "64": (64,), "64_32": (64, 32), "128_64": (128, 64)}


def suggest_model(trial, n_classes, random_state=0):
    '''Suggest a model family + its hyperparameters. Returns (name, estimator).'''
    family = trial.suggest_categorical(
        "model_family",
        ["logreg", "dtree", "rforest", "extratrees", "xgboost", "lightgbm", "knn", "nb", "mlp"],
    )

    if family == "logreg":
        C = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        model = LogisticRegression(C=C, penalty=penalty, solver="liblinear",
                                    max_iter=2000, random_state=random_state)

    elif family == "dtree":
        depth = trial.suggest_int("dt_max_depth", 2, 20)
        min_split = trial.suggest_int("dt_min_samples_split", 2, 20)
        model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split,
                                        random_state=random_state)

    elif family == "rforest":
        n_est = trial.suggest_int("rf_n_estimators", 20, 300)
        depth = trial.suggest_int("rf_max_depth", 3, 20)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                        n_jobs=-1, random_state=random_state)

    elif family == "extratrees":
        n_est = trial.suggest_int("et_n_estimators", 20, 300)
        depth = trial.suggest_int("et_max_depth", 3, 20)
        model = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth,
                                      n_jobs=-1, random_state=random_state)

    elif family == "xgboost":
        n_est = trial.suggest_int("xgb_n_estimators", 20, 300)
        depth = trial.suggest_int("xgb_max_depth", 2, 12)
        lr = trial.suggest_float("xgb_lr", 1e-3, 0.5, log=True)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        kwargs = {} if n_classes == 2 else {"num_class": n_classes}
        model = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                   objective=objective, eval_metric="logloss",
                                   n_jobs=-1, random_state=random_state, **kwargs)

    elif family == "lightgbm":
        n_est = trial.suggest_int("lgb_n_estimators", 20, 300)
        leaves = trial.suggest_int("lgb_num_leaves", 7, 127)
        lr = trial.suggest_float("lgb_lr", 1e-3, 0.5, log=True)
        model = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=lr,
                                    n_jobs=-1, random_state=random_state, verbosity=-1)

    elif family == "knn":
        k = trial.suggest_int("knn_k", 3, 31)
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    elif family == "nb":
        model = GaussianNB()

    elif family == "mlp":
        arch_key = trial.suggest_categorical("mlp_units", list(MLP_ARCHS.keys()))
        n_units = MLP_ARCHS[arch_key]
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True)
        model = MLPClassifier(hidden_layer_sizes=n_units, alpha=alpha,
                               max_iter=500, random_state=random_state)

    return family, model

print("suggest_model / MLP_ARCHS defined.")


def compute_mcc(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return matthews_corrcoef(y_te, pred), model


def structural_cost(model, family, n_train, n_features):
    '''Hardware-independent proxy for inference cost. Higher = more expensive.'''
    if family in ("rforest", "extratrees"):
        return float(sum(est.tree_.node_count for est in model.estimators_))
    if family == "dtree":
        return float(model.tree_.node_count)
    if family == "xgboost":
        booster = model.get_booster()
        df = booster.trees_to_dataframe()
        return float(len(df))  # total nodes across all boosted trees
    if family == "lightgbm":
        return float(model.booster_.num_trees() * (2 ** 6))  # trees x approx nodes/tree at default depth
    if family == "logreg":
        return float(np.sum(np.abs(model.coef_) > 1e-8))
    if family == "knn":
        return float(n_train * n_features)  # lazy learner — cost scales with training set
    if family == "nb":
        return float(n_features)
    if family == "mlp":
        return float(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    raise ValueError(f"Unknown family: {family}")


def get_shap_values(model, family, X_background, X_explain, n_classes):
    '''Dispatch to the fastest correct SHAP explainer per model family.'''
    tree_families = {"dtree", "rforest", "extratrees", "xgboost", "lightgbm"}
    if family in tree_families:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain, check_additivity=False)
    elif family == "logreg":
        explainer = shap.LinearExplainer(model, X_background)
        sv = explainer.shap_values(X_explain)
    else:
        # KernelSHAP fallback for KNN / NB / MLP -- capped budget to stay tractable
        bg = shap.kmeans(X_background, min(30, len(X_background)))
        predict_fn = model.predict_proba
        explainer = shap.KernelExplainer(predict_fn, bg)
        sv = explainer.shap_values(X_explain, nsamples=100, silent=True)

    # Normalize output shape to (n_classes, n_instances, n_features). SHAP's return shape
    # is inconsistent across explainers/model families/versions:
    #   - list of per-class arrays  -> stack directly
    #   - (n, d, n_classes)         -> RF/DT/KernelExplainer on multi-output models
    #   - (n, d)                   -> XGBoost/LightGBM/LinearExplainer on BINARY problems:
    #                                  a single array for class 1 only.
    # That last case is a real pitfall: naively wrapping it as shape (1, n, d) and then
    # indexing by predicted class (0 or 1) IndexErrors whenever the predicted class is 0.
    # The fix is to expand it into both classes: class-1 attribution is the returned
    # array, class-0 attribution is its negation (the two classes' probabilities sum to
    # 1, so their local attributions are equal and opposite).
    if isinstance(sv, list):
        return np.array(sv)                  # (n_classes, n, d)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return np.transpose(sv, (2, 0, 1))    # (n, d, n_classes) -> (n_classes, n, d)
    if n_classes == 2:
        return np.stack([-sv, sv])            # (2, n, d)
    raise ValueError(f"Unexpected SHAP output shape {sv.shape} for a {n_classes}-class problem")


def faithfulness(model, family, X_background, X_explain, k_steps=8):
    '''Φ = InsertionAUC - DeletionAUC, per-instance on the predicted class, averaged.
    See notebook markdown above for why this must be per-instance, not pooled.'''
    n, d = X_explain.shape
    background_vec = np.median(X_background, axis=0)
    proba_full = model.predict_proba(X_explain)
    target_cls = proba_full.argmax(axis=1)
    n_classes_local = proba_full.shape[1]

    sv_all = get_shap_values(model, family, X_background, X_explain, n_classes_local)  # (C, n, d)
    # select each instance's SHAP vector for ITS OWN predicted class
    sv = np.stack([sv_all[target_cls[i], i, :] for i in range(n)])  # (n, d)

    order = np.argsort(-np.abs(sv), axis=1)
    steps = np.unique(np.linspace(0, d, k_steps, dtype=int))
    x = steps / d

    del_curves = np.zeros((n, len(steps)))
    ins_curves = np.zeros((n, len(steps)))
    for si, k in enumerate(steps):
        Xdel = X_explain.copy()
        Xins = np.tile(background_vec, (n, 1))
        for i in range(n):
            idx = order[i, :k]
            Xdel[i, idx] = background_vec[idx]
            Xins[i, idx] = X_explain[i, idx]
        del_curves[:, si] = model.predict_proba(Xdel)[np.arange(n), target_cls]
        ins_curves[:, si] = model.predict_proba(Xins)[np.arange(n), target_cls]

    del_auc = np.array([np.trapezoid(del_curves[i], x) for i in range(n)])
    ins_auc = np.array([np.trapezoid(ins_curves[i], x) for i in range(n)])
    phi = float((ins_auc - del_auc).mean())
    return phi, float(del_auc.mean()), float(ins_auc.mean())


def measured_latency_ms(model, X_sample, n_warmup=20, n_repeats=100):
    '''Single-sample (batch=1) wall-clock latency, median over repeats. Used only to
    VALIDATE the structural proxy, never as the optimized objective (shared-VM timing
    is noisy on both Colab and Kaggle).'''
    x1 = X_sample[0:1]
    for _ in range(n_warmup):
        model.predict(x1)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(x1)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))

print("Objective functions (compute_mcc, structural_cost, get_shap_values, faithfulness, measured_latency_ms) defined.")


def pilot_faithfulness_check(X, y, n_classes, run_config, n_configs=8, seed=0):
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=run_config["cv_folds"], shuffle=True, random_state=seed)
    tr_idx, te_idx = next(skf.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    small_n = min(int(run_config["faithfulness_sample_size"] * 0.5), len(X_te))
    large_n = min(len(X_te), max(small_n * 2, small_n + 1))

    phi_small, phi_large = [], []
    study_pilot = optuna.create_study(directions=["maximize"], sampler=RandomSampler(seed=seed))

    for i in range(n_configs):
        trial = study_pilot.ask()
        family, model = suggest_model(trial, n_classes, random_state=seed)
        try:
            model.fit(X_tr, y_tr)
        except Exception:
            study_pilot.tell(trial, 0.0)
            continue
        idx_small = rng.choice(len(X_te), size=small_n, replace=False)
        idx_large = rng.choice(len(X_te), size=large_n, replace=False)
        try:
            p_s, _, _ = faithfulness(model, family, X_tr, X_te[idx_small], k_steps=max(4, run_config["faithfulness_steps"] // 2))
            p_l, _, _ = faithfulness(model, family, X_tr, X_te[idx_large], k_steps=run_config["faithfulness_steps"] * 2)
        except Exception as e:
            study_pilot.tell(trial, 0.0)
            continue
        phi_small.append(p_s); phi_large.append(p_l)
        study_pilot.tell(trial, 0.0)

    if len(phi_small) < 3:
        print("Too few successful pilot configs to compute correlation — inspect exceptions above.")
        return None, None, len(phi_small)
    rho, pval = spearmanr(phi_small, phi_large)
    print(f"Pilot configs evaluated: {len(phi_small)}")
    print(f"Spearman correlation (reduced-budget Φ vs larger-budget Φ): rho={rho:.3f}  p={pval:.4f}")
    if rho < 0.80:
        print("WARNING: correlation below 0.80 — faithfulness estimate may be noisy for this dataset.")
    else:
        print("OK: reduced-budget faithfulness estimate is adequately correlated with the larger-budget estimate.")
    return rho, pval, len(phi_small)

print("pilot_faithfulness_check defined.")


def make_objective(X, y, seed):
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        pre_steps = build_preprocessing_steps(trial, X.shape[1])
        family, model = suggest_model(trial, n_classes, random_state=seed)

        mccs, phis, costs = [], [], []
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            pipe = ImbPipeline(pre_steps + [("clf", model)]) if pre_steps else ImbPipeline([("clf", model)])
            try:
                pipe.fit(X_tr, y_tr)
            except Exception:
                raise optuna.TrialPruned()

            pred = pipe.predict(X_te)
            mccs.append(matthews_corrcoef(y_te, pred))

            fitted_model = pipe.named_steps["clf"]
            X_te_pre = X_te
            for name, step in pipe.steps[:-1]:
                if name != "smote":
                    X_te_pre = step.transform(X_te_pre)
            X_tr_pre = X_tr
            for name, step in pipe.steps[:-1]:
                if name != "smote":
                    X_tr_pre = step.transform(X_tr_pre) if not hasattr(step, "fit_resample") else X_tr_pre

            n_faith = min(CONFIG["faithfulness_sample_size"], len(X_te_pre))
            idx = np.random.RandomState(seed).choice(len(X_te_pre), size=n_faith, replace=False)
            try:
                phi, _, _ = faithfulness(fitted_model, family, X_tr_pre, X_te_pre[idx],
                                          k_steps=CONFIG["faithfulness_steps"])
            except Exception:
                phi = 0.0
            phis.append(phi)

            costs.append(structural_cost(fitted_model, family, len(X_tr), X_tr_pre.shape[1]))

        trial.set_user_attr("model_family", family)
        return float(np.mean(mccs)), float(np.mean(phis)), float(np.mean(costs))

    return objective


def run_nsga2_search(X, y, seed, n_trials, study_name_suffix=""):
    study = optuna.create_study(
        study_name=f"{DATASET_KEY}_nsga2_seed{seed}{study_name_suffix}",
        directions=["maximize", "maximize", "minimize"],
        sampler=NSGAIISampler(seed=seed),
        storage=STUDY_DB,
        load_if_exists=True,
    )
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        print(f"[{DATASET_KEY} seed {seed}] running {remaining} new trials ({len(study.trials)} already done)...")
        study.optimize(make_objective(X, y, seed), n_trials=remaining, show_progress_bar=False)
    else:
        print(f"[{DATASET_KEY} seed {seed}] already has {len(study.trials)} trials — nothing to do (resumed).")
    return study


def run_all_nsga2(X, y, n_seeds, n_trials):
    studies = {}
    for seed in range(n_seeds):
        studies[seed] = run_nsga2_search(X, y, seed, n_trials)
    print(f"[{DATASET_KEY}] NSGA-II search complete for all {n_seeds} seeds.")
    for seed, study in studies.items():
        print(f"  seed {seed}: {len(study.trials)} trials, {len(study.best_trials)} on the Pareto front")
    return studies

print("NSGA-II search functions defined.")


def trials_to_dataframe(study):
    '''Handles both the 3-objective MOXEC/random studies and the single-objective
    TPE baseline study -- the latter\'s trials carry only one value (MCC), and
    indexing t.values[1]/[2] unconditionally IndexErrors on it.'''
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {
            "trial": t.number,
            "model_family": t.user_attrs.get("model_family", "unknown"),
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        if len(t.values) == 3:
            row["mcc"], row["phi"], row["cost"] = t.values
        elif len(t.values) == 1:
            row["mcc"] = t.values[0]
        else:
            raise ValueError(f"Unexpected number of objective values: {len(t.values)}")
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front_df(study):
    df = trials_to_dataframe(study)
    front_numbers = {t.number for t in study.best_trials}
    return df[df["trial"].isin(front_numbers)]


def plot_pareto_front(dataset_key, seed, nsga2_studies):
    '''Build and save the 3-panel Pareto front figure for one (dataset, seed), as
    both PNG (300 DPI) and PDF, directly into the categorized figures/ output folder.'''
    df_s = trials_to_dataframe(nsga2_studies[seed])
    front_s = pareto_front_df(nsga2_studies[seed])

    fig = plt.figure(figsize=(15, 5))
    families = df_s["model_family"].unique()
    family_color = {fam: FAMILY_PALETTE.get(fam, "#333333") for fam in families}

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    for fam in families:
        sub = df_s[df_s["model_family"] == fam]
        ax1.scatter(sub["mcc"], sub["phi"], sub["cost"], label=fam, alpha=0.5, s=20, color=family_color[fam])
    ax1.scatter(front_s["mcc"], front_s["phi"], front_s["cost"],
                color="black", marker="*", s=140, label="Pareto front", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("MCC"); ax1.set_ylabel("Faithfulness (Φ)"); ax1.set_zlabel("Cost proxy")
    ax1.set_title(f"3D objective space — {dataset_key} (seed {seed})")

    ax2 = fig.add_subplot(1, 3, 2)
    for fam in families:
        sub = df_s[df_s["model_family"] == fam]
        ax2.scatter(sub["mcc"], sub["phi"], alpha=0.5, s=20, color=family_color[fam], label=fam)
    ax2.scatter(front_s["mcc"], front_s["phi"], color="black", marker="*", s=140, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("MCC"); ax2.set_ylabel("Faithfulness (Φ)"); ax2.set_title(f"MCC vs. Faithfulness (seed {seed})")

    ax3 = fig.add_subplot(1, 3, 3)
    for fam in families:
        sub = df_s[df_s["model_family"] == fam]
        ax3.scatter(sub["mcc"], sub["cost"], alpha=0.5, s=20, color=family_color[fam], label=fam)
    ax3.scatter(front_s["mcc"], front_s["cost"], color="black", marker="*", s=140, edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("MCC"); ax3.set_ylabel("Cost proxy"); ax3.set_title(f"MCC vs. Cost (seed {seed})")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    stem = OUTPUT_DIRS["figures"] / f"{dataset_key}_pareto_front_seed{seed}"
    png_path, pdf_path = savefig_dual(fig, stem)
    plt.show()
    plt.close(fig)
    print(f"Saved: {png_path.name}, {pdf_path.name}")
    return png_path

print("trials_to_dataframe / pareto_front_df / plot_pareto_front defined.")


def refit_trial_pipeline(trial_params, X_tr, y_tr, seed):
    '''Rebuild and refit the exact pipeline a given Optuna trial specified.'''
    fixed_trial = optuna.trial.FixedTrial(trial_params)
    pre_steps = build_preprocessing_steps(fixed_trial, X_tr.shape[1])
    family, model = suggest_model(fixed_trial, n_classes, random_state=seed)
    pipe = ImbPipeline(pre_steps + [("clf", model)]) if pre_steps else ImbPipeline([("clf", model)])
    pipe.fit(X_tr, y_tr)
    return family, pipe


def validate_latency_proxy(X, y, seed0_front, n_sample=10):
    skf_val = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=0)
    tr_idx, te_idx = next(skf_val.split(X, y))
    X_tr_v, X_te_v = X[tr_idx], X[te_idx]

    proxy_vals, latency_vals = [], []
    sample_front = seed0_front.sample(min(n_sample, len(seed0_front)), random_state=0)

    for _, row in sample_front.iterrows():
        params = {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_")}
        try:
            family, pipe = refit_trial_pipeline(params, X_tr_v, y[tr_idx], seed=0)
            fitted_model = pipe.named_steps["clf"]
            X_te_pre = X_te_v
            for name, step in pipe.steps[:-1]:
                if name != "smote":
                    X_te_pre = step.transform(X_te_pre)
            med_ms, p95_ms = measured_latency_ms(fitted_model, X_te_pre)
            proxy_vals.append(row["cost"]); latency_vals.append(med_ms)
        except Exception as e:
            print(f"  skipped one config due to: {e}")

    n_configs = len(proxy_vals)
    if n_configs >= 3:
        rho, pval = spearmanr(proxy_vals, latency_vals)
        print(f"Spearman(proxy, measured median latency) = {rho:.3f}  (p={pval:.4f}) over {n_configs} configs")
    else:
        rho, pval = None, None
        print("Not enough successful refits to compute a correlation.")
    return rho, pval, n_configs

print("refit_trial_pipeline / validate_latency_proxy defined.")


def knee_point(df):
    '''Point of maximum distance from the line connecting the two normalized extremes.'''
    pts = df[["mcc", "phi", "cost"]].copy()
    pts["cost_inv"] = -pts["cost"]  # flip so all three are "higher is better"
    norm = (pts[["mcc", "phi", "cost_inv"]] - pts[["mcc", "phi", "cost_inv"]].min()) / \
           (pts[["mcc", "phi", "cost_inv"]].max() - pts[["mcc", "phi", "cost_inv"]].min() + 1e-9)
    dist_from_ideal = np.sqrt(((1 - norm) ** 2).sum(axis=1))
    return df.loc[dist_from_ideal.idxmin()]


def cost_constrained_best(df, cost_budget):
    feasible = df[df["cost"] <= cost_budget]
    if feasible.empty:
        return None
    return feasible.loc[feasible["mcc"].idxmax()]


def weighted_scalarization(df, w_mcc=0.5, w_phi=0.3, w_cost=0.2):
    '''MES, repurposed here as a front-navigation aid rather than the search objective itself.'''
    norm = df[["mcc", "phi"]].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
    cost_norm = 1 - (df["cost"] - df["cost"].min()) / (df["cost"].max() - df["cost"].min() + 1e-9)
    mes = w_mcc * norm["mcc"] + w_phi * norm["phi"] + w_cost * cost_norm
    return df.loc[mes.idxmax()]


def build_navigation_table(front_df):
    knee = knee_point(front_df)
    cost_budget = front_df["cost"].median()
    cost_pick = cost_constrained_best(front_df, cost_budget)
    mes_pick = weighted_scalarization(front_df)

    nav_table = pd.DataFrame([
        {"rule": "Knee point", "model_family": knee["model_family"], "mcc": knee["mcc"], "phi": knee["phi"], "cost": knee["cost"]},
        {"rule": f"Cost-constrained (budget={cost_budget:.0f})", "model_family": cost_pick["model_family"], "mcc": cost_pick["mcc"], "phi": cost_pick["phi"], "cost": cost_pick["cost"]},
        {"rule": "Weighted MES", "model_family": mes_pick["model_family"], "mcc": mes_pick["mcc"], "phi": mes_pick["phi"], "cost": mes_pick["cost"]},
    ])
    # agreement: do knee/cost-constrained/MES land on the same model family?
    families_picked = nav_table["model_family"].tolist()
    all_agree = len(set(families_picked)) == 1
    knee_mes_agree = knee["model_family"] == mes_pick["model_family"]
    return nav_table, {"all_three_agree": bool(all_agree), "knee_and_mes_agree": bool(knee_mes_agree)}

print("Front navigation functions defined.")


def run_flaml_baseline(X, y, seed):
    from flaml import AutoML
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)
    tr_idx, te_idx = next(skf.split(X, y))
    automl = AutoML()
    automl.fit(X[tr_idx], y[tr_idx], task="classification", time_budget=CONFIG["baseline_time_budget_sec"],
               metric="macro_f1" if n_classes > 2 else "roc_auc", verbose=0, seed=seed)
    pred = automl.predict(X[te_idx])
    mcc = matthews_corrcoef(y[te_idx], pred)
    return {"method": "FLAML", "seed": seed, "mcc": mcc, "phi": None, "cost": None}


def run_autogluon_baseline(X, y, seed, feature_names):
    from autogluon.tabular import TabularPredictor
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)
    tr_idx, te_idx = next(skf.split(X, y))
    train_df = pd.DataFrame(X[tr_idx], columns=feature_names); train_df["target"] = y[tr_idx]
    test_df = pd.DataFrame(X[te_idx], columns=feature_names)
    predictor = TabularPredictor(label="target", verbosity=0,
                                  path=str(BASE_DIR / f"autogluon_seed{seed}")).fit(
        train_df, time_limit=CONFIG["baseline_time_budget_sec"], presets="medium_quality")
    pred = predictor.predict(test_df)
    mcc = matthews_corrcoef(y[te_idx], pred)
    return {"method": "AutoGluon", "seed": seed, "mcc": mcc, "phi": None, "cost": None}


def run_random_search_3obj(X, y, seed, n_trials):
    study = optuna.create_study(
        study_name=f"{DATASET_KEY}_random_seed{seed}", directions=["maximize", "maximize", "minimize"],
        sampler=RandomSampler(seed=seed), storage=STUDY_DB, load_if_exists=True,
    )
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(make_objective(X, y, seed), n_trials=remaining, show_progress_bar=False)
    return study


def run_tpe_mcc_only(X, y, seed, n_trials):
    skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)

    def objective(trial):
        pre_steps = build_preprocessing_steps(trial, X.shape[1])
        family, model = suggest_model(trial, n_classes, random_state=seed)
        mccs = []
        for tr_idx, te_idx in skf.split(X, y):
            pipe = ImbPipeline(pre_steps + [("clf", model)]) if pre_steps else ImbPipeline([("clf", model)])
            try:
                pipe.fit(X[tr_idx], y[tr_idx])
            except Exception:
                raise optuna.TrialPruned()
            mccs.append(matthews_corrcoef(y[te_idx], pipe.predict(X[te_idx])))
        trial.set_user_attr("model_family", family)
        return float(np.mean(mccs))

    study = optuna.create_study(
        study_name=f"{DATASET_KEY}_tpe_seed{seed}", direction="maximize",
        sampler=TPESampler(seed=seed), storage=STUDY_DB, load_if_exists=True,
    )
    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    return study


def run_all_baselines(X, y, n_seeds, n_trials, run_autogluon, feature_names):
    baseline_results = []
    random_studies, tpe_studies = {}, {}

    for seed in range(n_seeds):
        print(f"--- {DATASET_KEY} seed {seed} ---")
        baseline_results.append(run_flaml_baseline(X, y, seed))
        print("  FLAML done.")

        if run_autogluon:
            try:
                baseline_results.append(run_autogluon_baseline(X, y, seed, feature_names))
                print("  AutoGluon done.")
            except Exception as e:
                print(f"  AutoGluon skipped (error: {e})")

        random_studies[seed] = run_random_search_3obj(X, y, seed, n_trials)
        print(f"  Random search (3-obj) done: {len(random_studies[seed].trials)} trials.")

        tpe_studies[seed] = run_tpe_mcc_only(X, y, seed, n_trials)
        print(f"  TPE (MCC-only) done: {len(tpe_studies[seed].trials)} trials.")

    baseline_df = pd.DataFrame(baseline_results)
    return baseline_df, random_studies, tpe_studies

print("Baseline functions defined.")


def compute_hv_for_front(front_points, global_min, global_max, ref=(1.05, 1.05, 1.05)):
    '''front_points: array of (mcc, phi, cost) tuples (cost already 'lower is better').
    Normalizes against the GLOBAL min/max across all methods being compared, then
    converts to minimization form for pymoo's HV indicator. Normalized values are
    clipped to [0,1] as a safety net against any point landing outside the bounds.'''
    pts = np.array(front_points, dtype=float)
    mn, mx = np.array(global_min), np.array(global_max)
    norm = np.clip((pts - mn) / (mx - mn + 1e-9), 0.0, 1.0)
    F = np.column_stack([1 - norm[:, 0], 1 - norm[:, 1], norm[:, 2]])
    ind = HV(ref_point=np.array(ref))
    return float(ind(F))


def get_tpe_point(X, y, seed, tpe_studies):
    '''Refit TPE's best-MCC trial to recover phi/cost, for a fair HV comparison
    against the 3-objective methods. Returns (mcc, phi, cost, model_family).'''
    tpe_df = trials_to_dataframe(tpe_studies[seed])
    best_tpe_row = tpe_df.loc[tpe_df["mcc"].idxmax()]
    params_tpe = {k.replace("param_", ""): v for k, v in best_tpe_row.items() if k.startswith("param_")}
    skf_tmp = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)
    tr_idx, te_idx = next(skf_tmp.split(X, y))
    fam_tpe, pipe_tpe = refit_trial_pipeline(params_tpe, X[tr_idx], y[tr_idx], seed)
    X_te_pre_tpe = X[te_idx]
    X_tr_pre_tpe = X[tr_idx]
    for name, step in pipe_tpe.steps[:-1]:
        if name != "smote":
            X_te_pre_tpe = step.transform(X_te_pre_tpe)
            X_tr_pre_tpe = step.transform(X_tr_pre_tpe)
    phi_tpe, _, _ = faithfulness(pipe_tpe.named_steps["clf"], fam_tpe, X_tr_pre_tpe, X_te_pre_tpe[:CONFIG["faithfulness_sample_size"]])
    cost_tpe = structural_cost(pipe_tpe.named_steps["clf"], fam_tpe, len(tr_idx), X_te_pre_tpe.shape[1])
    return float(best_tpe_row["mcc"]), phi_tpe, cost_tpe, fam_tpe


def evaluate_hypervolume(X, y, n_seeds, nsga2_studies, random_studies, tpe_studies):
    tpe_points = {seed: get_tpe_point(X, y, seed, tpe_studies) for seed in range(n_seeds)}

    all_mcc, all_phi, all_cost = [], [], []
    for seed in range(n_seeds):
        df_s = trials_to_dataframe(nsga2_studies[seed])
        all_mcc += df_s["mcc"].tolist(); all_phi += df_s["phi"].tolist(); all_cost += df_s["cost"].tolist()
        df_r = trials_to_dataframe(random_studies[seed])
        all_mcc += df_r["mcc"].tolist(); all_phi += df_r["phi"].tolist(); all_cost += df_r["cost"].tolist()
        mcc_t, phi_t, cost_t, _fam_t = tpe_points[seed]
        all_mcc.append(mcc_t); all_phi.append(phi_t); all_cost.append(cost_t)

    global_min = (min(all_mcc), min(all_phi), min(all_cost))
    global_max = (max(all_mcc), max(all_phi), max(all_cost))
    print(f"[{DATASET_KEY}] Global bounds — MCC: [{global_min[0]:.3f}, {global_max[0]:.3f}]  "
          f"Phi: [{global_min[1]:.3f}, {global_max[1]:.3f}]  Cost: [{global_min[2]:.1f}, {global_max[2]:.1f}]")

    hv_results = []
    for seed in range(n_seeds):
        nsga2_front = pareto_front_df(nsga2_studies[seed])[["mcc", "phi", "cost"]].values
        hv_nsga2 = compute_hv_for_front(nsga2_front, global_min, global_max)
        random_front = pareto_front_df(random_studies[seed])[["mcc", "phi", "cost"]].values
        hv_random = compute_hv_for_front(random_front, global_min, global_max)
        hv_tpe = compute_hv_for_front([list(tpe_points[seed][:3])], global_min, global_max)
        hv_results.append({"seed": seed, "MOXEC (NSGA-II)": hv_nsga2, "Random Search (3-obj)": hv_random, "TPE (MCC-only)": hv_tpe})

    hv_df = pd.DataFrame(hv_results)
    return hv_df, tpe_points, global_min, global_max


def run_wilcoxon_tests(hv_df):
    print("Wilcoxon signed-rank test (MOXEC vs. baseline), paired across seeds:\n")
    wilcoxon_results = {}
    for baseline in ["Random Search (3-obj)", "TPE (MCC-only)"]:
        moxec_mean = float(hv_df["MOXEC (NSGA-II)"].mean())
        baseline_mean = float(hv_df[baseline].mean())
        if hv_df["seed"].nunique() >= 3:
            stat, p = wilcoxon(hv_df["MOXEC (NSGA-II)"], hv_df[baseline])
            print(f"  MOXEC vs. {baseline}: W={stat:.3f}, p={p:.4f}  (mean HV {moxec_mean:.4f} vs {baseline_mean:.4f})")
            wilcoxon_results[baseline] = {"statistic": float(stat), "p_value": float(p),
                                            "moxec_mean_hv": moxec_mean, "baseline_mean_hv": baseline_mean,
                                            "n_seeds": int(hv_df["seed"].nunique())}
        else:
            print(f"  MOXEC vs. {baseline}: need >=3 seeds — currently {hv_df['seed'].nunique()}. "
                  f"Means only: {moxec_mean:.4f} vs {baseline_mean:.4f}")
            wilcoxon_results[baseline] = {"statistic": None, "p_value": None,
                                            "moxec_mean_hv": moxec_mean, "baseline_mean_hv": baseline_mean,
                                            "n_seeds": int(hv_df["seed"].nunique())}
    return wilcoxon_results

print("Hypervolume (compute_hv_for_front, get_tpe_point, evaluate_hypervolume) and Wilcoxon functions defined.")


KEEL_BASE_URL = "https://sci2s.ugr.es/keel/dataset/data/classification/"


def _parse_keel_dat(text):
    '''Parses a KEEL .dat file (ARFF-like: @relation/@attribute/@inputs/@outputs/@data)
    into (X_raw, y_raw) pandas objects shaped exactly like what UCI\'s fetch_ucirepo
    returns, so downstream cleaning (cat_cols/get_dummies/fillna below) is unchanged.
    Numeric attributes ("real"/"integer") are cast to float64; nominal attributes
    ("{a,b,c}") are left as strings for later one-hot encoding. "?" -> NaN either way,
    though every dataset in DATASET_REGISTRY is a KEEL complete-case file with none.'''
    attr_names, attr_is_numeric = [], []
    output_name = None
    data_start = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low.startswith("@attribute"):
            rest = stripped[len("@attribute"):].strip()
            m = re.match(r"^'?([^'\s{]+)'?\s*(.*)$", rest)
            name, spec = m.group(1), m.group(2).strip()
            attr_names.append(name)
            attr_is_numeric.append(not spec.startswith("{"))
        elif low.startswith("@output"):  # matches both "@output" (car.dat) and "@outputs"
            output_name = stripped.split(None, 1)[1].split(",")[0].strip()
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("KEEL file has no @data section -- malformed download?")

    data_lines = [l for l in lines[data_start:] if l.strip()]
    rows = [[v.strip().strip("'\"") for v in l.split(",")] for l in data_lines]
    df = pd.DataFrame(rows, columns=attr_names).replace("?", np.nan)

    if output_name is None or output_name not in df.columns:
        output_name = attr_names[-1]  # KEEL convention: target is always the last attribute
    for name, is_numeric in zip(attr_names, attr_is_numeric):
        if name != output_name and is_numeric:
            df[name] = pd.to_numeric(df[name], errors="coerce")

    y_raw = df[output_name]
    X_raw = df.drop(columns=[output_name])
    return X_raw, y_raw


def fetch_keel_dataset(keel_name, cache_dir):
    '''Downloads (once; cached thereafter) and parses one KEEL "Standard Classification"
    dataset by its KEEL name, returning (X_raw, y_raw). The plain "<name>.zip" archive
    (as opposed to the "<name>-5-fold.zip" partitioned ones) contains exactly one
    <name>.dat file holding the complete, unpartitioned dataset.'''
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dat_path = cache_dir / f"{keel_name}.dat"

    if not dat_path.exists():
        url = f"{KEEL_BASE_URL}{keel_name}.zip"
        print(f"  Downloading KEEL dataset '{keel_name}' from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                zip_bytes = resp.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download KEEL dataset '{keel_name}' from {url}: {e}") from e
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            dat_members = [n for n in zf.namelist() if n.lower().endswith(".dat")]
            if not dat_members:
                raise RuntimeError(f"No .dat file found inside {url} -- got: {zf.namelist()}")
            with zf.open(dat_members[0]) as f:
                dat_path.write_bytes(f.read())

    text = dat_path.read_text(encoding="latin-1")
    return _parse_keel_dat(text)

print("KEEL loader (fetch_keel_dataset / _parse_keel_dat) defined.")


def load_and_validate_dataset(dataset_key):
    entry = DATASET_REGISTRY[dataset_key]
    source = entry.get("source", "uci")

    if source == "uci":
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=entry["uci_id"])
        X_raw = ds.data.features.copy()
        y_raw = ds.data.targets.copy()
        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]
        dataset_name = ds.metadata.name
    elif source == "keel":
        keel_cache_dir = OUTPUT_ROOT / "_datasets_keel"
        X_raw, y_raw = fetch_keel_dataset(entry["keel_name"], keel_cache_dir)
        dataset_name = f"KEEL: {entry['keel_name']}"
    else:
        raise ValueError(f"[{dataset_key}] unknown dataset source '{source}' -- expected 'uci' or 'keel'.")

    if entry.get("binarize_threshold") is not None:
        y_raw = (pd.to_numeric(y_raw, errors="coerce") >= entry["binarize_threshold"]).astype(int)

    load_warnings = []
    actual_n, actual_d = X_raw.shape
    if actual_n != entry["expected_n"]:
        load_warnings.append(f"row count mismatch: expected {entry['expected_n']}, got {actual_n}")
    if actual_d != entry["expected_d"]:
        load_warnings.append(f"feature count mismatch: expected {entry['expected_d']}, got {actual_d}")
    n_missing = int(X_raw.isna().sum().sum())
    if n_missing > 0:
        load_warnings.append(f"{n_missing} missing values present (median-imputed below)")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes_actual = len(le.classes_)
    if n_classes_actual != entry["expected_classes"]:
        load_warnings.append(f"class count mismatch: expected {entry['expected_classes']}, got {n_classes_actual}")

    expected_task = "binary" if n_classes_actual == 2 else "multiclass"
    if expected_task != entry["task_type"]:
        raise ValueError(f"[{dataset_key}] task_type mismatch: registry says '{entry['task_type']}', "
                          f"but {n_classes_actual} classes were detected ({expected_task}). "
                          f"Fix DATASET_REGISTRY before proceeding -- every objective function "
                          f"branches on task_type and will silently misbehave otherwise.")

    if load_warnings:
        print(f"[{dataset_key}] SANITY WARNINGS:")
        for w in load_warnings:
            print(f"  - {w}")
    else:
        print(f"[{dataset_key}] Sanity check OK: shape, feature count, and class count match the registry.")

    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_enc = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=True)
    else:
        X_enc = X_raw.copy()

    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

    def _dedupe_columns(cols):
        seen, out = {}, []
        for c in cols:
            if c not in seen:
                seen[c] = 0; out.append(c)
            else:
                seen[c] += 1; out.append(f"{c}__{seen[c]}")
        return out

    if X_enc.columns.duplicated().any():
        dup_count = int(X_enc.columns.duplicated().sum())
        print(f"[{dataset_key}] WARNING: {dup_count} duplicate column name(s) -- renaming for AutoGluon compatibility.")
        X_enc.columns = _dedupe_columns(X_enc.columns.tolist())

    X = X_enc.values.astype(np.float64)
    feature_names = X_enc.columns.tolist()

    class_counts = dict(pd.Series(y).value_counts().sort_index())
    print(f"[{dataset_key}] Loaded: {dataset_name} -- X.shape={X.shape}, "
          f"classes={list(le.classes_)}, class distribution={class_counts}")

    return X, y, n_classes_actual, feature_names, dataset_name, load_warnings

print("load_and_validate_dataset defined.")


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full MOXEC pipeline across the UCI PALE-lean portfolio and/or the KEEL additions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", type=str, default="all",
                    help="Comma-separated dataset keys to run (see DATASET_REGISTRY), or one of: "
                         "'all' (every registry entry), 'uci' (the original 12), 'keel' (the 26 KEEL "
                         "additions). 'all' is a lot of compute -- see full_pipeline/README.md before "
                         "using it unmodified.")
    p.add_argument("--output-dir", type=str, default="./outputs",
                    help="Root output directory (the categorized tree is created under this).")
    p.add_argument("--n-trials", type=int, default=100, help="NSGA-II / Random / TPE trial budget per seed.")
    p.add_argument("--n-seeds", type=int, default=3, help="Number of independent seeds per dataset.")
    p.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds for every objective evaluation.")
    p.add_argument("--faithfulness-sample-size", type=int, default=200)
    p.add_argument("--faithfulness-steps", type=int, default=8)
    p.add_argument("--baseline-time-budget-sec", type=int, default=300,
                    help="Wall-clock budget per seed for FLAML and AutoGluon each.")
    p.add_argument("--no-autogluon", action="store_true", help="Skip the AutoGluon baseline (FLAML still runs).")
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--aggregate-only", action="store_true",
                    help="Skip running datasets; only (re)build the cross-dataset aggregation "
                         "and paper-support summary from whatever is already in --output-dir/paper_ready/.")
    return p.parse_args()


def init_globals(args):
    '''Sets the module-level globals every ported function reads (OUTPUT_ROOT,
    OUTPUT_DIRS, MASTER_SEED, plotting style) from parsed CLI args -- the CLI
    equivalent of the notebook's hardcoded Kaggle-path / MASTER_SEED cell.'''
    global MASTER_SEED, OUTPUT_ROOT, OUTPUT_DIRS, DB_ROOT

    MASTER_SEED = args.master_seed
    np.random.seed(MASTER_SEED)

    OUTPUT_ROOT = Path(args.output_dir)
    OUTPUT_DIRS = {
        "raw": OUTPUT_ROOT / "raw_results",
        "processed": OUTPUT_ROOT / "processed_results",
        "metrics": OUTPUT_ROOT / "metrics",
        "tables": OUTPUT_ROOT / "tables",
        "figures": OUTPUT_ROOT / "figures",
        "stats": OUTPUT_ROOT / "statistical_analysis",
        "pareto": OUTPUT_ROOT / "pareto_analysis",
        "paper": OUTPUT_ROOT / "paper_ready",
    }
    for d in OUTPUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    DB_ROOT = OUTPUT_ROOT / "optuna_dbs"
    DB_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {OUTPUT_ROOT.resolve()}")

    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


FAMILY_PALETTE = {
    "logreg": "#1f77b4", "dtree": "#d62728", "rforest": "#ff7f0e", "extratrees": "#2ca02c",
    "xgboost": "#9467bd", "lightgbm": "#8c564b", "knn": "#e377c2", "nb": "#17becf", "mlp": "#7f7f7f",
}


def savefig_dual(fig, path_stem):
    '''Save a figure as both PNG (300 DPI) and PDF (vector) under the same stem.'''
    png_path = path_stem.with_suffix(".png")
    pdf_path = path_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


# ======================================================================
# Per-dataset pipeline orchestrator (ported verbatim from the notebook)
# ======================================================================
def run_dataset_pipeline(dataset_key):
    global DATASET_KEY, CONFIG, BASE_DIR, STUDY_DB, n_classes

    DATASET_KEY = dataset_key
    entry = DATASET_REGISTRY[dataset_key]
    t_start = time.time()

    CONFIG = {
        **GLOBAL_RUN_CONFIG,
        "dataset_source": entry.get("source", "uci"),
        "uci_id": entry.get("uci_id"),
        "keel_name": entry.get("keel_name"),
        "dataset_name": dataset_key,
        "task_type": entry["task_type"],
    }

    BASE_DIR = OUTPUT_ROOT / "_work" / dataset_key
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DB = f"sqlite:///{BASE_DIR / 'optuna_studies.db'}"

    print("\n" + "=" * 78)
    print(f"DATASET: {dataset_key}  ({entry['display_name']})  {_dataset_source_label(entry)}")
    print("=" * 78)

    X, y, n_classes_actual, feature_names, dataset_name, load_warnings = load_and_validate_dataset(dataset_key)
    n_classes = n_classes_actual

    print(f"\n[{dataset_key}] Faithfulness sanity check...")
    faith_rho, faith_pval, faith_n = pilot_faithfulness_check(X, y, n_classes, CONFIG, n_configs=8, seed=0)

    print(f"\n[{dataset_key}] NSGA-II search (all {CONFIG['n_seeds']} seeds)...")
    nsga2_studies = run_all_nsga2(X, y, CONFIG["n_seeds"], CONFIG["n_trials"])

    print(f"\n[{dataset_key}] Pareto front extraction + per-seed plots...")
    per_seed_records = []
    seed0_front = None
    for seed in range(CONFIG["n_seeds"]):
        df_s = trials_to_dataframe(nsga2_studies[seed])
        front_s = pareto_front_df(nsga2_studies[seed])
        if seed == 0:
            seed0_front = front_s

        df_s.to_parquet(OUTPUT_DIRS["raw"] / f"{dataset_key}_all_trials_seed{seed}.parquet")
        front_s.to_parquet(OUTPUT_DIRS["processed"] / f"{dataset_key}_pareto_front_seed{seed}.parquet")
        plot_pareto_front(dataset_key, seed, nsga2_studies)

        moxec_best_row = df_s.loc[df_s["mcc"].idxmax()]
        per_seed_records.append({
            "dataset": dataset_key, "seed": seed,
            "moxec_best_mcc": float(moxec_best_row["mcc"]),
            "moxec_best_mcc_family": moxec_best_row["model_family"],
            "n_trials": len(df_s), "n_pareto_front": len(front_s),
        })

    print(f"\n[{dataset_key}] Latency proxy validation...")
    latency_rho, latency_pval, latency_n = validate_latency_proxy(X, y, seed0_front)

    print(f"\n[{dataset_key}] Front navigation...")
    nav_table, nav_agreement = build_navigation_table(seed0_front)
    nav_table.to_csv(OUTPUT_DIRS["pareto"] / f"{dataset_key}_navigation_picks.csv", index=False)

    print(f"\n[{dataset_key}] Baselines (FLAML / AutoGluon / Random / TPE)...")
    baseline_df, random_studies, tpe_studies = run_all_baselines(
        X, y, CONFIG["n_seeds"], CONFIG["n_trials"], GLOBAL_RUN_CONFIG["run_autogluon"], feature_names)

    for rec in per_seed_records:
        seed = rec["seed"]
        for _, brow in baseline_df[baseline_df["seed"] == seed].iterrows():
            rec[f"{brow['method'].lower()}_mcc"] = float(brow["mcc"]) if pd.notna(brow["mcc"]) else None

    best_mcc_df = pd.DataFrame(per_seed_records)
    best_mcc_df.to_csv(OUTPUT_DIRS["processed"] / f"{dataset_key}_best_mcc_by_seed.csv", index=False)

    print(f"\n[{dataset_key}] Hypervolume + Wilcoxon evaluation...")
    hv_df, tpe_points, global_min, global_max = evaluate_hypervolume(
        X, y, CONFIG["n_seeds"], nsga2_studies, random_studies, tpe_studies)
    hv_df.to_csv(OUTPUT_DIRS["metrics"] / f"{dataset_key}_hypervolume_by_seed.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIRS["metrics"] / f"{dataset_key}_baseline_results.csv", index=False)
    wilcoxon_results = run_wilcoxon_tests(hv_df)

    elapsed_sec = time.time() - t_start

    artifact_summary = {
        "dataset_key": dataset_key,
        "config": CONFIG,
        "dataset_meta": {"name": dataset_name, "domain": entry["domain"],
                          "n_instances": int(X.shape[0]), "n_features": int(X.shape[1]),
                          "n_classes": int(n_classes_actual)},
        "load_warnings": load_warnings,
        "hypervolume_by_seed": hv_df.to_dict(orient="records"),
        "hv_normalization_bounds": {"mcc": [global_min[0], global_max[0]],
                                      "phi": [global_min[1], global_max[1]],
                                      "cost": [global_min[2], global_max[2]]},
        "tpe_points_by_seed": {str(s): {"mcc": p[0], "phi": p[1], "cost": p[2], "model_family": p[3]}
                                 for s, p in tpe_points.items()},
        "wilcoxon_tests": wilcoxon_results,
        "faithfulness_sanity_check": {"spearman_rho": faith_rho, "p_value": faith_pval, "n_configs": faith_n},
        "latency_proxy_validation": {"spearman_rho": latency_rho, "p_value": latency_pval, "n_configs": latency_n},
        "baseline_point_results": baseline_df.to_dict(orient="records"),
        "best_mcc_by_seed": best_mcc_df.to_dict(orient="records"),
        "navigation_picks": nav_table.to_dict(orient="records"),
        "navigation_agreement": nav_agreement,
        "elapsed_seconds": elapsed_sec,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(OUTPUT_DIRS["stats"] / f"{dataset_key}_wilcoxon.json", "w") as f:
        json.dump(wilcoxon_results, f, indent=2, default=str)
    with open(OUTPUT_DIRS["stats"] / f"{dataset_key}_faithfulness_sanity_check.json", "w") as f:
        json.dump({"spearman_rho": faith_rho, "p_value": faith_pval, "n_configs": faith_n}, f, indent=2, default=str)
    with open(OUTPUT_DIRS["stats"] / f"{dataset_key}_latency_proxy_validation.json", "w") as f:
        json.dump({"spearman_rho": latency_rho, "p_value": latency_pval, "n_configs": latency_n}, f, indent=2, default=str)
    with open(OUTPUT_DIRS["paper"] / f"{dataset_key}_summary.json", "w") as f:
        json.dump(artifact_summary, f, indent=2, default=str)

    summary_table = pd.DataFrame([{
        "dataset": dataset_key, "domain": entry["domain"],
        "n_instances": int(X.shape[0]), "n_features": int(X.shape[1]), "n_classes": int(n_classes_actual),
        "moxec_best_mcc_mean": best_mcc_df["moxec_best_mcc"].mean(),
        "moxec_best_mcc_std": best_mcc_df["moxec_best_mcc"].std(),
        "flaml_mcc_mean": best_mcc_df["flaml_mcc"].mean() if "flaml_mcc" in best_mcc_df else None,
        "autogluon_mcc_mean": best_mcc_df["autogluon_mcc"].mean() if "autogluon_mcc" in best_mcc_df else None,
        "hv_moxec_mean": hv_df["MOXEC (NSGA-II)"].mean(), "hv_moxec_std": hv_df["MOXEC (NSGA-II)"].std(),
        "hv_random_mean": hv_df["Random Search (3-obj)"].mean(), "hv_random_std": hv_df["Random Search (3-obj)"].std(),
        "hv_tpe_mean": hv_df["TPE (MCC-only)"].mean(), "hv_tpe_std": hv_df["TPE (MCC-only)"].std(),
        "wilcoxon_vs_random_p": wilcoxon_results["Random Search (3-obj)"]["p_value"],
        "wilcoxon_vs_tpe_p": wilcoxon_results["TPE (MCC-only)"]["p_value"],
        "faithfulness_sanity_rho": faith_rho,
        "latency_proxy_rho": latency_rho,
        "n_pareto_front_mean": best_mcc_df["n_pareto_front"].mean(),
        "nav_all_three_agree": nav_agreement["all_three_agree"],
        "elapsed_minutes": elapsed_sec / 60,
    }])
    summary_table.to_csv(OUTPUT_DIRS["tables"] / f"{dataset_key}_summary_table.csv", index=False)

    print(f"\n[{dataset_key}] DONE in {elapsed_sec/60:.1f} min. Artifacts saved under {OUTPUT_ROOT}.")
    return artifact_summary

print("run_dataset_pipeline defined.")


# ======================================================================
# Master loop over a batch of datasets, with per-dataset error isolation
# ======================================================================
def run_all_datasets(dataset_keys):
    run_manifest = []

    for dataset_key in dataset_keys:
        entry_start = time.time()
        try:
            summary = run_dataset_pipeline(dataset_key)
            run_manifest.append({
                "dataset": dataset_key, "status": "SUCCESS", "error": None,
                "elapsed_minutes": summary["elapsed_seconds"] / 60,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[{dataset_key}] FAILED: {e}\n{tb}")
            run_manifest.append({
                "dataset": dataset_key, "status": "FAILED", "error": f"{type(e).__name__}: {e}",
                "elapsed_minutes": (time.time() - entry_start) / 60,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            continue

    manifest_df = pd.DataFrame(run_manifest)
    manifest_path = OUTPUT_ROOT / "run_manifest.csv"
    if manifest_path.exists():
        prior = pd.read_csv(manifest_path)
        manifest_df = pd.concat([prior[~prior["dataset"].isin(manifest_df["dataset"])], manifest_df], ignore_index=True)
    manifest_df.to_csv(manifest_path, index=False)

    print("\n" + "=" * 78)
    print("RUN MANIFEST (this invocation)")
    print("=" * 78)
    print(manifest_df.to_string(index=False))

    n_failed = int((manifest_df["status"] == "FAILED").sum())
    if n_failed > 0:
        print(f"\n{n_failed} dataset(s) FAILED -- see the 'error' column above. "
              f"Re-run those specific keys with --datasets after investigating; every "
              f"other dataset in this batch completed and was saved regardless.")
    return manifest_df



# ======================================================================
# Cross-dataset aggregation -- combines every §18/19 analysis from the
# notebook (per-dataset HV summary, Friedman+Nemenyi+CD diagram, domain
# contrast, ablations A1/A2, front-navigation agreement, per-objective
# breakdown, final paper-support summary) into one callable pass. Reads
# whatever *_summary.json files are present in OUTPUT_DIRS["paper"] -- works
# on however many datasets have completed, from this run or any prior one.
# ======================================================================
def run_cross_dataset_aggregation():
    import scikit_posthocs as sp

    summary_paths = sorted(p for p in OUTPUT_DIRS["paper"].glob("*_summary.json")
                            if not p.name.startswith("FINAL_"))
    all_summaries = {}
    for p in summary_paths:
        with open(p) as f:
            s = json.load(f)
        all_summaries[s["dataset_key"]] = s

    n_done = len(all_summaries)
    n_total = len(DATASET_REGISTRY)
    print(f"Found {n_done} / {n_total} completed dataset summaries: {sorted(all_summaries.keys())}")
    missing = sorted(set(DATASET_REGISTRY.keys()) - set(all_summaries.keys()))
    if missing:
        print(f"NOT yet completed: {missing}")
    if n_done == 0:
        print("\nNo completed datasets found -- run at least one dataset via the master loop "
              "above (or attach a prior session's outputs/) before this section produces anything.")

    portfolio_hv_rows = []
    for dataset_key, s in all_summaries.items():
        hv_records = s.get("hypervolume_by_seed", [])
        if not hv_records:
            continue
        hv_df_ds = pd.DataFrame(hv_records)
        row = {"dataset": dataset_key, "domain": s["dataset_meta"]["domain"],
               "n_instances": s["dataset_meta"]["n_instances"], "n_classes": s["dataset_meta"]["n_classes"]}
        for method in ["MOXEC (NSGA-II)", "Random Search (3-obj)", "TPE (MCC-only)"]:
            row[f"{method}_mean"] = hv_df_ds[method].mean()
            row[f"{method}_std"] = hv_df_ds[method].std()
        row["moxec_wins_vs_random"] = bool(row["MOXEC (NSGA-II)_mean"] > row["Random Search (3-obj)_mean"])
        row["moxec_wins_vs_tpe"] = bool(row["MOXEC (NSGA-II)_mean"] > row["TPE (MCC-only)_mean"])
        portfolio_hv_rows.append(row)

    portfolio_hv_df = pd.DataFrame(portfolio_hv_rows)
    if len(portfolio_hv_df):
        portfolio_hv_df.to_csv(OUTPUT_DIRS["tables"] / "portfolio_hypervolume_summary.csv", index=False)
        print(f"n_datasets = {len(portfolio_hv_df)}")
        print(f"MOXEC beats Random Search on {int(portfolio_hv_df['moxec_wins_vs_random'].sum())}/{len(portfolio_hv_df)} datasets")
        print(f"MOXEC beats TPE on {int(portfolio_hv_df['moxec_wins_vs_tpe'].sum())}/{len(portfolio_hv_df)} datasets")
    print(portfolio_hv_df.to_string(index=False))


    if len(portfolio_hv_df) >= 1:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(portfolio_hv_df))))
        y_pos = np.arange(len(portfolio_hv_df))
        width = 0.25
        methods_plot = ["MOXEC (NSGA-II)", "Random Search (3-obj)", "TPE (MCC-only)"]
        colors_plot = ["#2a6f4f", "#5b8fd4", "#c9622a"]
        for i, method in enumerate(methods_plot):
            means = portfolio_hv_df[f"{method}_mean"].values
            stds = portfolio_hv_df[f"{method}_std"].values
            ax.barh(y_pos + (i - 1) * width, means, height=width, xerr=stds, label=method,
                    color=colors_plot[i], alpha=0.9, capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(portfolio_hv_df["dataset"])
        ax.set_xlabel("Hypervolume (mean ± std across seeds)")
        ax.set_title("Hypervolume by dataset and method")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        savefig_dual(fig, OUTPUT_DIRS["figures"] / "portfolio_hypervolume_by_dataset")
        plt.show()
        plt.close(fig)
    else:
        print("No datasets completed yet -- nothing to plot.")

    FRIEDMAN_METHODS = ["MOXEC (NSGA-II)", "Random Search (3-obj)", "TPE (MCC-only)"]

    def plot_cd_diagram(avg_ranks, method_names, cd, n_datasets, output_stem):
        '''Minimal critical-difference diagram (Demsar, 2006): average rank per method on a
        number line (rank 1 = best, left), with a bar of width `cd` and a thick horizontal
        line connecting any group of methods whose ranks are not significantly different.'''
        k = len(method_names)
        fig, ax = plt.subplots(figsize=(7, 1.8 + 0.3 * k))
        order = np.argsort(avg_ranks)
        ranks_sorted = np.array(avg_ranks)[order]
        names_sorted = np.array(method_names)[order]

        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", linewidth=1)
        for r in range(1, k + 1):
            ax.plot([r, r], [0.47, 0.53], color="black", linewidth=1)
            ax.text(r, 0.40, str(r), ha="center", fontsize=9)

        for i, (r, name) in enumerate(zip(ranks_sorted, names_sorted)):
            y = 0.75 if i % 2 == 0 else 0.25
            ax.plot([r, r], [0.5, y], color="gray", linewidth=1)
            ax.plot(r, 0.5, "o", color="black", markersize=4)
            ax.text(r, y + (0.05 if i % 2 == 0 else -0.09), f"{name}\n(rank={r:.2f})",
                    ha="center", fontsize=8)

        # CD bar
        cd_y = 0.9
        ax.plot([ranks_sorted.min(), ranks_sorted.min() + cd], [cd_y, cd_y], color="red", linewidth=2)
        ax.text((ranks_sorted.min() + ranks_sorted.min() + cd) / 2, cd_y + 0.04, f"CD = {cd:.3f}",
                ha="center", fontsize=8, color="red")

        # thick bars connecting statistically-indistinguishable groups
        connected_y = 0.6
        i = 0
        while i < k:
            j = i
            while j + 1 < k and ranks_sorted[j + 1] - ranks_sorted[i] <= cd:
                j += 1
            if j > i:
                ax.plot([ranks_sorted[i], ranks_sorted[j]], [connected_y, connected_y], color="black", linewidth=3)
                connected_y += 0.05
            i = j + 1

        ax.set_yticks([])
        ax.set_xlabel("Average rank (1 = best)")
        ax.set_title(f"Critical-difference diagram — hypervolume ranks across {n_datasets} datasets")
        plt.tight_layout()
        return savefig_dual(fig, output_stem)


    MIN_DATASETS_FOR_FRIEDMAN = 3
    if len(portfolio_hv_df) >= MIN_DATASETS_FOR_FRIEDMAN:
        hv_matrix = portfolio_hv_df[[f"{m}_mean" for m in FRIEDMAN_METHODS]].values
        stat, p = friedmanchisquare(*[hv_matrix[:, i] for i in range(hv_matrix.shape[1])])
        print(f"Friedman test over {len(portfolio_hv_df)} datasets, {len(FRIEDMAN_METHODS)} methods: "
              f"chi2={stat:.4f}, p={p:.4f}")
        if len(portfolio_hv_df) < 8:
            print(f"CAUTION: only {len(portfolio_hv_df)} datasets -- Friedman/Nemenyi has limited power "
                  f"below ~10-12 blocks; treat this as a preview, not a final result, until more of the "
                  f"portfolio is complete.")

        # ranks: rank 1 = best (highest HV) -> use -hv_matrix so scipy/pandas rank ascending gives rank 1 to best
        ranks_df = pd.DataFrame(-hv_matrix, columns=FRIEDMAN_METHODS).rank(axis=1)
        avg_ranks = ranks_df.mean(axis=0).values

        nemenyi_df = sp.posthoc_nemenyi_friedman(hv_matrix)
        nemenyi_df.columns = FRIEDMAN_METHODS
        nemenyi_df.index = FRIEDMAN_METHODS
        nemenyi_df.to_csv(OUTPUT_DIRS["stats"] / "nemenyi_posthoc_pvalues.csv")
        print("\nNemenyi post-hoc pairwise p-values:")
        print(nemenyi_df.to_string())

        # Nemenyi critical value (q_alpha, alpha=0.05) for k treatments -- studentized range
        # statistic constants from Demsar (2006) Table; standard, widely-reproduced values.
        Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                       8: 3.031, 9: 3.102, 10: 3.164}
        k = len(FRIEDMAN_METHODS)
        N = len(portfolio_hv_df)
        q_alpha = Q_ALPHA_005.get(k, 2.343)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
        print(f"\nCritical difference (alpha=0.05, k={k}, N={N}): CD = {cd:.4f}")

        friedman_results = {
            "n_datasets": int(N), "n_methods": int(k), "statistic": float(stat), "p_value": float(p),
            "average_ranks": {m: float(r) for m, r in zip(FRIEDMAN_METHODS, avg_ranks)},
            "critical_difference": float(cd), "q_alpha_005": float(q_alpha),
            "caution_low_power": bool(N < 8),
        }
        with open(OUTPUT_DIRS["stats"] / "friedman_nemenyi_results.json", "w") as f:
            json.dump(friedman_results, f, indent=2)

        plot_cd_diagram(avg_ranks, FRIEDMAN_METHODS, cd, N, OUTPUT_DIRS["figures"] / "critical_difference_diagram")
        plt.show()
    else:
        print(f"Only {len(portfolio_hv_df)} dataset(s) completed -- need at least "
              f"{MIN_DATASETS_FOR_FRIEDMAN} for the Friedman test to run at all, and substantially more "
              f"({MIN_DATASETS_FOR_FRIEDMAN}-12+) for it to be a meaningful, citable result. Skipping.")
        friedman_results = None

    MIN_PER_DOMAIN_FOR_MANNWHITNEY = 3
    if len(portfolio_hv_df):
        portfolio_hv_df["hv_gain_vs_random"] = (
            portfolio_hv_df["MOXEC (NSGA-II)_mean"] - portfolio_hv_df["Random Search (3-obj)_mean"]
        )
        domain_counts = portfolio_hv_df["domain"].value_counts().to_dict()
        print(f"Datasets per domain so far: {domain_counts}")

        medical_gain = portfolio_hv_df.loc[portfolio_hv_df["domain"] == "medical", "hv_gain_vs_random"]
        agri_gain = portfolio_hv_df.loc[portfolio_hv_df["domain"] == "agriculture", "hv_gain_vs_random"]

        if len(medical_gain) >= MIN_PER_DOMAIN_FOR_MANNWHITNEY and len(agri_gain) >= MIN_PER_DOMAIN_FOR_MANNWHITNEY:
            u_stat, u_p = mannwhitneyu(medical_gain, agri_gain, alternative="two-sided")
            print(f"\nMann-Whitney U (medical HV-gain vs. agricultural HV-gain): U={u_stat:.3f}, p={u_p:.4f}")
            print(f"  medical   (n={len(medical_gain)}): mean gain = {medical_gain.mean():.4f}")
            print(f"  agriculture (n={len(agri_gain)}): mean gain = {agri_gain.mean():.4f}")
            domain_contrast_results = {
                "u_statistic": float(u_stat), "p_value": float(u_p),
                "medical_n": int(len(medical_gain)), "medical_mean_gain": float(medical_gain.mean()),
                "agriculture_n": int(len(agri_gain)), "agriculture_mean_gain": float(agri_gain.mean()),
                "interpretation": ("no significant domain difference -- advantage appears domain-agnostic"
                                    if u_p >= 0.05 else "significant domain difference -- advantage is NOT uniform across domains"),
            }
        else:
            print(f"\nNeed >= {MIN_PER_DOMAIN_FOR_MANNWHITNEY} datasets per domain for Mann-Whitney U -- "
                  f"currently medical={len(medical_gain)}, agriculture={len(agri_gain)}. Skipping test.")
            domain_contrast_results = None

        if domain_contrast_results:
            with open(OUTPUT_DIRS["stats"] / "domain_contrast_mannwhitney.json", "w") as f:
                json.dump(domain_contrast_results, f, indent=2)
    else:
        domain_contrast_results = None
        print("No datasets completed yet.")

    if len(portfolio_hv_df):
        ablation_rows = []
        for _, row in portfolio_hv_df.iterrows():
            ablation_rows.append({
                "dataset": row["dataset"],
                "A1_moxec_hv": row["MOXEC (NSGA-II)_mean"], "A1_tpe_hv": row["TPE (MCC-only)_mean"],
                "A1_moxec_wins": row["MOXEC (NSGA-II)_mean"] > row["TPE (MCC-only)_mean"],
                "A2_moxec_hv": row["MOXEC (NSGA-II)_mean"], "A2_random_hv": row["Random Search (3-obj)_mean"],
                "A2_moxec_wins": row["MOXEC (NSGA-II)_mean"] > row["Random Search (3-obj)_mean"],
            })
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(OUTPUT_DIRS["tables"] / "ablation_A1_A2_by_dataset.csv", index=False)

        n = len(ablation_df)
        a1_wins = int(ablation_df["A1_moxec_wins"].sum())
        a2_wins = int(ablation_df["A2_moxec_wins"].sum())
        print(f"Ablation A1 (3-obj vs. MCC-only): MOXEC beats TPE on hypervolume in {a1_wins}/{n} datasets "
              f"({100*a1_wins/n:.0f}%)")
        print(f"Ablation A2 (NSGA-II vs. random): MOXEC beats Random Search on hypervolume in {a2_wins}/{n} datasets "
              f"({100*a2_wins/n:.0f}%)")

        if n >= 3:
            stat_a1, p_a1 = wilcoxon(ablation_df["A1_moxec_hv"], ablation_df["A1_tpe_hv"])
            stat_a2, p_a2 = wilcoxon(ablation_df["A2_moxec_hv"], ablation_df["A2_random_hv"])
            print(f"\nPortfolio-level Wilcoxon (paired across datasets, on per-dataset mean HV):")
            print(f"  A1 MOXEC vs. TPE:    W={stat_a1:.3f}, p={p_a1:.4f}")
            print(f"  A2 MOXEC vs. Random: W={stat_a2:.3f}, p={p_a2:.4f}")
            if a2_wins / n < 0.6:
                print("\nNOTE: A2 win rate is not overwhelming -- per PALE_lean_protocol.md §12 risk table, "
                      "if random search matches NSGA-II on the full portfolio, reframe the paper's claim "
                      "toward the 3-objective FORMULATION rather than the NSGA-II sampler specifically.")
        else:
            print(f"\nNeed >= 3 datasets for a portfolio-level Wilcoxon test -- currently {n}.")
    else:
        print("No datasets completed yet.")

    if all_summaries:
        agreement_rows = []
        for dataset_key, s in all_summaries.items():
            agr = s.get("navigation_agreement", {})
            agreement_rows.append({"dataset": dataset_key, **agr})
        agreement_df = pd.DataFrame(agreement_rows)
        agreement_df.to_csv(OUTPUT_DIRS["tables"] / "navigation_agreement_by_dataset.csv", index=False)

        n = len(agreement_df)
        all3 = int(agreement_df["all_three_agree"].sum()) if n else 0
        kneemes = int(agreement_df["knee_and_mes_agree"].sum()) if n else 0
        print(f"All three navigation rules agree on {all3}/{n} datasets ({100*all3/n:.0f}%)" if n else "No data.")
        print(f"Knee point and weighted MES agree on {kneemes}/{n} datasets ({100*kneemes/n:.0f}%)" if n else "")
    else:
        print("No datasets completed yet.")

    per_objective_rows = []
    for dataset_key, s in all_summaries.items():
        best_mcc_records = s.get("best_mcc_by_seed", [])
        if not best_mcc_records:
            continue
        bdf = pd.DataFrame(best_mcc_records)
        tpe_points = s.get("tpe_points_by_seed", {})
        tpe_mcc = np.mean([p["mcc"] for p in tpe_points.values()]) if tpe_points else None
        tpe_phi = np.mean([p["phi"] for p in tpe_points.values()]) if tpe_points else None
        tpe_cost = np.mean([p["cost"] for p in tpe_points.values()]) if tpe_points else None

        per_objective_rows.append({
            "dataset": dataset_key,
            "moxec_best_mcc_mean": bdf["moxec_best_mcc"].mean(),
            "flaml_mcc_mean": bdf["flaml_mcc"].mean() if "flaml_mcc" in bdf else None,
            "autogluon_mcc_mean": bdf["autogluon_mcc"].mean() if "autogluon_mcc" in bdf else None,
            "tpe_mcc_mean": tpe_mcc, "tpe_phi_mean": tpe_phi, "tpe_cost_mean": tpe_cost,
            "faithfulness_sanity_rho": s.get("faithfulness_sanity_check", {}).get("spearman_rho"),
            "latency_proxy_rho": s.get("latency_proxy_validation", {}).get("spearman_rho"),
        })

    per_objective_df = pd.DataFrame(per_objective_rows)
    if len(per_objective_df):
        per_objective_df.to_csv(OUTPUT_DIRS["tables"] / "per_objective_breakdown.csv", index=False)
    print(per_objective_df.to_string(index=False) if len(per_objective_df) else "No datasets completed yet.")

    paper_support = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio_coverage": {
            "n_datasets_completed": len(all_summaries),
            "n_datasets_total": len(DATASET_REGISTRY),
            "datasets_completed": sorted(all_summaries.keys()),
            "datasets_missing": sorted(set(DATASET_REGISTRY.keys()) - set(all_summaries.keys())),
        },
        "run_config": GLOBAL_RUN_CONFIG,
        "master_seed": MASTER_SEED,
    }

    # --- best-performing configuration per dataset (accuracy-only, and joint-objective) ---
    best_configs = {}
    for dataset_key, s in all_summaries.items():
        bdf = pd.DataFrame(s["best_mcc_by_seed"])
        best_row = bdf.loc[bdf["moxec_best_mcc"].idxmax()]
        nav = pd.DataFrame(s["navigation_picks"])
        knee_row = nav[nav["rule"] == "Knee point"].iloc[0] if len(nav) else None
        best_configs[dataset_key] = {
            "best_accuracy_model": {
                "family": best_row["moxec_best_mcc_family"], "mcc": float(best_row["moxec_best_mcc"]),
                "seed": int(best_row["seed"]),
            },
            "best_joint_objective_model_knee_point": (
                {"family": knee_row["model_family"], "mcc": float(knee_row["mcc"]),
                 "phi": float(knee_row["phi"]), "cost": float(knee_row["cost"])}
                if knee_row is not None else None
            ),
        }
    paper_support["best_configs_by_dataset"] = best_configs

    # --- dataset-wise and seed-wise results ---
    paper_support["dataset_wise_results"] = {
        k: {"hypervolume_by_seed": s["hypervolume_by_seed"], "best_mcc_by_seed": s["best_mcc_by_seed"],
            "baseline_point_results": s["baseline_point_results"]}
        for k, s in all_summaries.items()
    }

    # --- aggregate statistics ---
    paper_support["aggregate_statistics"] = {
        "portfolio_hypervolume_summary": portfolio_hv_df.to_dict(orient="records") if len(portfolio_hv_df) else [],
        "friedman_nemenyi": friedman_results,
        "domain_contrast_mannwhitney": domain_contrast_results,
    }

    # --- ablation findings ---
    paper_support["ablation_findings"] = ablation_df.to_dict(orient="records") if len(portfolio_hv_df) else []

    # --- statistical test results (per-dataset Wilcoxon + faithfulness/latency sanity) ---
    paper_support["statistical_test_results_by_dataset"] = {
        k: {"wilcoxon_tests": s["wilcoxon_tests"],
            "faithfulness_sanity_check": s["faithfulness_sanity_check"],
            "latency_proxy_validation": s["latency_proxy_validation"]}
        for k, s in all_summaries.items()
    }

    # --- Pareto-optimal configurations (full seed-0 front per dataset, family + objectives) ---
    pareto_optimal_configs = {}
    for dataset_key in all_summaries:
        front_path = OUTPUT_DIRS["processed"] / f"{dataset_key}_pareto_front_seed0.parquet"
        if front_path.exists():
            front_df = pd.read_parquet(front_path)
            pareto_optimal_configs[dataset_key] = front_df[["trial", "model_family", "mcc", "phi", "cost"]].to_dict(orient="records")
    paper_support["pareto_optimal_configs_seed0"] = pareto_optimal_configs

    # --- front-navigation agreement ---
    paper_support["front_navigation_agreement"] = agreement_df.to_dict(orient="records") if all_summaries else []

    # --- important observations (auto-generated from the numbers above; edit/extend by hand before submission) ---
    observations = []
    if len(portfolio_hv_df):
        a2_wins = int(portfolio_hv_df["moxec_wins_vs_random"].sum())
        a1_wins = int(portfolio_hv_df["moxec_wins_vs_tpe"].sum())
        n = len(portfolio_hv_df)
        observations.append(f"MOXEC (NSGA-II) beats Random Search on hypervolume in {a2_wins}/{n} datasets.")
        observations.append(f"MOXEC (NSGA-II) beats single-objective TPE on hypervolume in {a1_wins}/{n} datasets.")
        if a2_wins / n < 0.6:
            observations.append("NSGA-II's advantage over random search is not overwhelming on this portfolio -- "
                                  "per the protocol's own risk framing, the paper's claim should lean on the "
                                  "3-objective FORMULATION, not the NSGA-II sampler specifically.")
    if domain_contrast_results:
        observations.append(f"Domain contrast: {domain_contrast_results['interpretation']} "
                             f"(Mann-Whitney p={domain_contrast_results['p_value']:.4f}).")
    for dataset_key, s in all_summaries.items():
        frho = s["faithfulness_sanity_check"]["spearman_rho"]
        if frho is not None and frho < 0.80:
            observations.append(f"{dataset_key}: faithfulness sanity-check rho={frho:.3f} is BELOW the 0.80 "
                                 f"threshold -- Φ may be noisy for this dataset at the configured sample size.")
        lrho = s["latency_proxy_validation"]["spearman_rho"]
        if lrho is not None and lrho < 0.30:
            observations.append(f"{dataset_key}: latency-proxy validation rho={lrho:.3f} is weak -- the "
                                 f"structural cost proxy may not track real inference latency well here.")
        if s["load_warnings"]:
            observations.append(f"{dataset_key}: dataset load warnings recorded -- {s['load_warnings']}")
    paper_support["important_observations"] = observations

    with open(OUTPUT_DIRS["paper"] / "FINAL_paper_support_summary.json", "w") as f:
        json.dump(paper_support, f, indent=2, default=str)

    print(f"Paper-support summary written: {OUTPUT_DIRS['paper'] / 'FINAL_paper_support_summary.json'}")
    print(f"\nPortfolio coverage: {len(all_summaries)}/{len(DATASET_REGISTRY)} datasets completed.")
    print("\nKey observations:")
    for obs in observations:
        print(f"  - {obs}")
    print(f"\nCross-dataset aggregation complete. Portfolio coverage: "
          f"{len(all_summaries)}/{len(DATASET_REGISTRY)} datasets.")
    return paper_support



# ======================================================================
# main()
# ======================================================================
def main():
    args = parse_args()
    init_globals(args)

    log_path = OUTPUT_ROOT / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Logging to {log_path.resolve()} (mirrored to console)")
    print(f"Run started: {datetime.now(timezone.utc).isoformat()}")

    global GLOBAL_RUN_CONFIG
    GLOBAL_RUN_CONFIG = {
        "n_trials": args.n_trials,
        "n_seeds": args.n_seeds,
        "cv_folds": args.cv_folds,
        "faithfulness_sample_size": args.faithfulness_sample_size,
        "faithfulness_steps": args.faithfulness_steps,
        "run_autogluon": not args.no_autogluon,
        "baseline_time_budget_sec": args.baseline_time_budget_sec,
    }
    print(f"Run config: {GLOBAL_RUN_CONFIG}")

    datasets_arg = args.datasets.strip().lower()
    if datasets_arg == "all":
        dataset_keys = list(DATASET_REGISTRY.keys())
    elif datasets_arg == "uci":
        dataset_keys = [k for k, v in DATASET_REGISTRY.items() if v.get("source", "uci") == "uci"]
    elif datasets_arg == "keel":
        dataset_keys = [k for k, v in DATASET_REGISTRY.items() if v.get("source") == "keel"]
    else:
        dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
        unknown = [k for k in dataset_keys if k not in DATASET_REGISTRY]
        if unknown:
            print(f"ERROR: unknown dataset key(s) in --datasets: {unknown}")
            print(f"Valid keys: {list(DATASET_REGISTRY.keys())}")
            sys.exit(1)

    if not args.aggregate_only:
        print(f"\nRunning {len(dataset_keys)} dataset(s): {dataset_keys}")
        run_all_datasets(dataset_keys)
    else:
        print("\n--aggregate-only set: skipping the dataset run, aggregating existing outputs only.")

    print("\nRunning cross-dataset aggregation...")
    run_cross_dataset_aggregation()

    print(f"\nRun finished: {datetime.now(timezone.utc).isoformat()}")
    print(f"All outputs under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Whatever datasets already finished were saved to disk; "
              "re-run with --datasets to pick up the remaining ones, or --aggregate-only to build the "
              "cross-dataset summary from what's already there.")
        sys.exit(130)
    except Exception:
        print("\n\nUNHANDLED ERROR -- full traceback follows. Any dataset that already completed before "
              "this point was saved to disk regardless.")
        traceback.print_exc()
        sys.exit(1)
