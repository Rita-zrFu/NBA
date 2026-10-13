"""Fixed, small temporal baseline comparison. Reads train/validation only."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="nba_stage4_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="nba_stage4_cache_"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from nba_injury.features.historical import FEATURE_NAMES
from nba_injury.features.preprocessing import make_preprocessor

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_ERROR = None
except (ImportError, OSError) as exc:
    LGBMClassifier = None
    LIGHTGBM_ERROR = str(exc)

THRESHOLD = 0.5
FOLDS = [("2019-01-01", "2019-03-01"),
         ("2019-03-01", "2019-04-11"),
         ("2019-11-01", "2020-02-28")]
CONFIGS = {
    "Logistic Regression": dict(C=1.0, solver="lbfgs", max_iter=1000,
                                class_weight="balanced", random_state=42),
    "Random Forest": dict(n_estimators=200, max_depth=8, min_samples_leaf=5,
                          max_features="sqrt", class_weight="balanced",
                          n_jobs=4, random_state=42),
    "XGBoost": dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                    min_child_weight=5, subsample=1.0, colsample_bytree=1.0,
                    reg_lambda=1.0, tree_method="hist", objective="binary:logistic",
                    eval_metric="logloss", n_jobs=4, random_state=42),
    "LightGBM": dict(n_estimators=150, max_depth=4, num_leaves=15,
                     learning_rate=0.05, min_child_samples=20, reg_lambda=1.0,
                     subsample=1.0, colsample_bytree=1.0, objective="binary",
                     n_jobs=4, random_state=42, verbosity=-1),
}
METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1")
READ_PATHS = []
INPUT_HASHES = {}


def read_input(split, name):
    # An explicit allowlist makes accidental test evaluation impossible here.
    if split not in {"train", "validation"} or name not in {"X", "y", "row_manifest"}:
        raise ValueError("Stage 4 permits only frozen train/validation matrices and manifests")
    path = ROOT / "data/processed/stage3" / f"{name}_{split}.csv"
    READ_PATHS.append(str(path.relative_to(ROOT)))
    INPUT_HASHES[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def load_split(split, expected_n, expected_positives):
    X, y_frame, manifest = [read_input(split, name) for name in ("X", "y", "row_manifest")]
    assert list(X.columns) == list(FEATURE_NAMES) and X.shape == (expected_n, 36)
    assert y_frame.shape == (expected_n, 1) and y_frame.columns.tolist() == ["y"]
    y = y_frame.iloc[:, 0].astype(int)
    assert y.isin([0, 1]).all() and y.sum() == expected_positives
    assert len(manifest) == expected_n and manifest.observation_id.is_unique
    assert manifest.split.eq(split).all()
    assert np.array_equal(y.to_numpy(), manifest.target_14d.to_numpy())
    for col in ("index_date", "history_cutoff_exclusive", "assumed_label_available_date_14d"):
        manifest[col] = pd.to_datetime(manifest[col])
    assert (manifest.history_cutoff_exclusive < manifest.index_date).all()
    assert (manifest.assumed_label_available_date_14d >= manifest.index_date).all()
    return X, y, manifest


def class_ratio(y):
    assert y.nunique() == 2
    return float((y == 0).sum() / (y == 1).sum())


def make_model(name, y):
    settings = CONFIGS[name].copy()
    constructors = {"Logistic Regression": LogisticRegression,
                    "Random Forest": RandomForestClassifier,
                    "XGBoost": XGBClassifier, "LightGBM": LGBMClassifier}
    if name in {"XGBoost", "LightGBM"}:
        settings["scale_pos_weight"] = class_ratio(y)
    family = "linear" if name == "Logistic Regression" else "tree"
    # Each call produces a fresh, unfitted imputer/scaler, including each CV fold.
    preprocessing = make_preprocessor(family).set_output(transform="pandas")
    return Pipeline([("preprocess", preprocessing), ("model", constructors[name](**settings))])


def fit_predict(name, X_train, y_train, X_eval):
    pipeline = make_model(name, y_train)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    pre = pipeline.named_steps["preprocess"]
    medians = pre.named_steps["impute"].statistics_.copy()
    scale_mean = pre.named_steps["scale"].mean_.copy() if "scale" in pre.named_steps else None
    probabilities = pipeline.predict_proba(X_eval)[:, 1]
    assert probabilities.shape == (len(X_eval),)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(medians, pre.named_steps["impute"].statistics_)
    if scale_mean is not None:
        np.testing.assert_array_equal(scale_mean, pre.named_steps["scale"].mean_)
    return probabilities, time.perf_counter() - started


def metrics(y, probabilities):
    predicted = probabilities >= THRESHOLD
    return dict(roc_auc=float(roc_auc_score(y, probabilities)),
                average_precision=float(average_precision_score(y, probabilities)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)))


def markdown_table(frame):
    def value(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(map(value, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def main():
    started = time.perf_counter()
    X, y, train = load_split("train", 4957, 593)
    X_val, y_val, validation = load_split("validation", 2143, 293)
    assert train.index_date.max() < validation.index_date.min() - pd.Timedelta(days=22)
    assert train.assumed_label_available_date_14d.max() < validation.index_date.min()
    folds, fold_docs = [], []
    for fold_no, (start_text, end_text) in enumerate(FOLDS, 1):
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        training_mask = ((train.index_date < start - pd.Timedelta(days=22)) &
                         (train.assumed_label_available_date_14d < start))
        eval_mask = (train.index_date >= start) & (train.index_date < end)
        tr, ev = np.flatnonzero(training_mask), np.flatnonzero(eval_mask)
        assert len(tr) and len(ev) and not set(tr).intersection(ev)
        assert y.iloc[tr].nunique() == y.iloc[ev].nunique() == 2
        assert (train.iloc[ev].index_date.min() - train.iloc[tr].index_date.max()).days >= 23
        assert train.iloc[tr].assumed_label_available_date_14d.max() < start
        eligible_earlier = train.index_date < start
        folds.append((tr, ev))
        fold_docs.append(dict(fold=fold_no,
            train_dates=f"{train.iloc[tr].index_date.min().date()}–{train.iloc[tr].index_date.max().date()}",
            eval_dates=f"{train.iloc[ev].index_date.min().date()}–{train.iloc[ev].index_date.max().date()}",
            train_n=len(tr), train_positive=int(y.iloc[tr].sum()), eval_n=len(ev),
            eval_positive=int(y.iloc[ev].sum()), purged=int(eligible_earlier.sum()-len(tr)),
            train_negative_positive_ratio=class_ratio(y.iloc[tr]),
            latest_train_label_available=str(train.iloc[tr].assumed_label_available_date_14d.max().date())))
    names = list(CONFIGS) if LGBMClassifier is not None else list(CONFIGS)[:-1]
    rows, fold_metrics, validation_probabilities = [], [], {}
    for name in names:
        model_started = time.perf_counter()
        scores = []
        for fold_no, (tr, ev) in enumerate(folds, 1):
            p, seconds = fit_predict(name, X.iloc[tr], y.iloc[tr], X.iloc[ev])
            score = metrics(y.iloc[ev], p)
            scores.append(score)
            fold_metrics.append(dict(model=name, fold=fold_no, **score, seconds=seconds))
        p, seconds = fit_predict(name, X, y, X_val)
        validation_probabilities[name] = p
        row = {"model": name}
        for metric in METRICS:
            row[f"cv_{metric}_mean"] = float(np.mean([s[metric] for s in scores]))
            row[f"cv_{metric}_sd"] = float(np.std([s[metric] for s in scores], ddof=1))
        row.update({f"validation_{key}": value for key, value in metrics(y_val, p).items()})
        row["validation_fit_predict_seconds"] = seconds
        row["all_fit_predict_seconds"] = time.perf_counter() - model_started
        rows.append(row)
        print(f"{name}: validation ROC-AUC={row['validation_roc_auc']:.4f}, "
              f"AP={row['validation_average_precision']:.4f}, elapsed={row['all_fit_predict_seconds']:.2f}s", flush=True)
    comparison = pd.DataFrame(rows)
    table_path = ROOT / "results/tables/stage4_model_comparison.csv"
    comparison.to_csv(table_path, index=False)
    output_dir = ROOT / "data/processed/stage4"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = validation[["observation_id", "index_date"]].copy()
    predictions["y_14d"] = y_val
    for name, p in validation_probabilities.items():
        slug = name.lower().replace(" ", "_")
        predictions[f"{slug}_probability"] = p
        predictions[f"{slug}_prediction_at_0_5"] = (p >= THRESHOLD).astype(int)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    figure_dir = ROOT / "results/figures/stage4"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for curve in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for name, p in validation_probabilities.items():
            if curve == "roc":
                horizontal, vertical, _ = roc_curve(y_val, p)
                score = roc_auc_score(y_val, p)
                label = f"{name} (AUC {score:.3f})"
            else:
                vertical, horizontal, _ = precision_recall_curve(y_val, p)
                score = average_precision_score(y_val, p)
                label = f"{name} (AP {score:.3f})"
            ax.plot(horizontal, vertical, label=label, linewidth=1.6)
        if curve == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance ranking")
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
        else:
            ax.axhline(y_val.mean(), linestyle="--", color="gray", label=f"Prevalence {y_val.mean():.3f}")
            ax.set(xlabel="Recall", ylabel="Precision")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"Frozen validation: {'ROC' if curve == 'roc' else 'precision–recall'}")
        ax.legend(loc="lower right" if curve == "roc" else "upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{curve}.png", dpi=160)
        plt.close(fig)
    assert len(READ_PATHS) == 6 and all("test" not in path for path in READ_PATHS)
    for path, expected_hash in INPUT_HASHES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    best = comparison.loc[comparison.validation_average_precision.idxmax(), "model"]
    elapsed = time.perf_counter() - started
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "xgboost"] +
                (["lightgbm"] if LGBMClassifier is not None else [])}
    headline_cols = ["model", "cv_roc_auc_mean", "cv_average_precision_mean", "validation_roc_auc",
                     "validation_average_precision", "validation_precision", "validation_recall", "validation_f1"]
    report = f"""# Stage 4: fixed temporal baseline comparison

Four prescribed baseline configurations were fixed before fitting. No hyperparameter search, early stopping, feature selection, calibration, threshold optimization, or SMOTE was used. Stage 2/3 cohort and features are unchanged. **The held-out test matrices, labels, and manifests were never opened.**

## Population and preprocessing

Frozen training: 4,957 snapshots, 593 positives (11.96%), 2018-11-16–2020-02-27. Frozen validation: 2,143 snapshots, 293 positives (13.67%), 2021-01-22–2021-04-23. The task remains a documented new injury-related inactive-list episode in 14 days, conditional on the Stage 2 ascertainment/eligibility rules. All 36 Stage 3 numeric predictors are retained; player/date/target metadata never enter X.

Each fold has a fresh sklearn Pipeline with the Stage 3 schema guard and training-fold median imputation. Logistic Regression additionally uses training-fold StandardScaler; trees are not scaled. Full-training models learn preprocessing on training only and evaluate the frozen validation once each. No fitted models are persisted; the script reproduces the comparison.

## Temporal validation

Three fixed expanding chronological folds use only the frozen training period. Training index dates must be at least **23 calendar days** before the evaluation start, and `assumed_label_available_date_14d` must be strictly earlier than that start. This conservatively enforces the Stage 2 horizon/witness availability policy. No shuffled CV is used. Class weights and boosting `scale_pos_weight` are recomputed from each fold's training labels, never evaluation labels.

{markdown_table(pd.DataFrame(fold_docs))}

CV metrics are the unweighted mean across three folds; sample standard deviations and all CV precision/recall/F1 values are in the single comparison CSV. They describe temporal variability, not independent-sample confidence intervals. Repeated players across periods are allowed for future-time prediction; these results do not establish generalization to entirely unseen players.

## Baseline settings and imbalance

"""
    for name in names:
        settings = CONFIGS[name].copy()
        if name in {"XGBoost", "LightGBM"}:
            settings["scale_pos_weight"] = class_ratio(y)
        report += f"- **{name}**: `{settings}`\n"
    report += f"""
All unlisted parameters use installed-library defaults. Logistic Regression/Random Forest use `class_weight=\"balanced\"`; boosted models use training negatives/positives, **{int((y==0).sum())}/593 = {class_ratio(y):.6f}** for the final training fit. Probability cutoffs for every precision/recall/F1 calculation are fixed at **0.5**. These class-weighted scores are uncalibrated; they are not established clinical injury probabilities. See [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) and [LightGBM API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) for weight semantics.

## Comparison

PR-AUC is reported as **Average Precision (AP)**, not trapezoidal area under the interpolated PR curve. The primary comparison criterion was frozen-validation AP, selected before fitting.

{markdown_table(comparison[headline_cols])}

**Highest validation AP: {best}.** Validation no-skill AP reference is its prevalence, {y_val.mean():.4f}. Selection remains provisional until a later approved final test evaluation. The old course ROC-AUC uses a different outcome/population/random split and cannot serve as an improvement comparator.

Per-fold results (same fixed configurations; no searches):

{markdown_table(pd.DataFrame(fold_metrics))}

![Validation ROC](../results/figures/stage4/roc.png)

![Validation precision–recall](../results/figures/stage4/precision_recall.png)

## Sanity checks and runtime

All {len(names)*4} fits succeeded (three CV and one full-training fit per model). All probabilities are finite and within [0,1]; imputation/scaling statistics do not change during held-out prediction. Fold ordering, 23-day separation, training-label availability, frozen row counts/positive counts/schema/alignment, and unchanged six input-file SHA-256 values passed runtime assertions. The read allowlist admits only train/validation X, y, and manifests. No Stage 4 test suite was added; Stage 2/3 source code was not modified.

Measured script runtime: **{elapsed:.2f} seconds**, excluding imports, package installation, and document review. Per-model fitting/prediction times are in the comparison CSV. Python: {sys.version.split()[0]}; packages: `{versions}`. XGBoost and LightGBM installed from precompiled wheels in ignored `.venv-stage4`, using existing Anaconda scientific packages; no global package changes.

## Outputs and reproduction

- `results/tables/stage4_model_comparison.csv`: the only comparison table (CV mean/SD for five metrics, validation five metrics, timing).
- `results/figures/stage4/roc.png` and `precision_recall.png`: validation curves only.
- `data/processed/stage4/validation_predictions.csv`: 2,143 validation observations with IDs, dates, truth, four scores and four fixed-threshold predictions; local and ignored by Git.
- `scripts/run_stage4_baselines.py`, `requirements-stage4.txt`, and this report: fixed settings, split definitions, checks, and reproduction.
- Existing `.gitignore` and `README.md` updated for the local environment and Stage 4 status. No Stage 2/3 data, scripts, feature rules, or split definitions modified.

From the repository root:

```bash
/opt/anaconda3/bin/python3 -m venv --system-site-packages .venv-stage4
.venv-stage4/bin/python -m pip install --only-binary=:all: --no-deps xgboost==3.4.1 lightgbm==4.7.0
.venv-stage4/bin/python scripts/run_stage4_baselines.py
git status --short
```

The tested existing scientific dependency versions are listed in `requirements-stage4.txt`; a new isolated environment can install that file instead. If LightGBM import is unavailable, the script explicitly reports its skip reason and runs the other three models.

## Remaining limits

This benchmark conditions on retrospective label ascertainment/follow-up and appearance-based eligibility. Injury logs do not establish physiological onset, a continuously active roster, causal workload effects, or calibrated risk. Reporting differences and repeated-player dependence remain. No uncertainty intervals or detailed diagnostics were added in this streamlined stage. Résumé claims may describe four-model temporal comparison and validation results, with the recorded-event task specified; do not call these test results or clinical injury prediction accuracy.

Stage 4 ends here. No tuning, Stage 5 work, held-out test evaluation, commits, or remote push was performed.
"""
    if LIGHTGBM_ERROR:
        report += f"\nLightGBM skipped: `{LIGHTGBM_ERROR}`\n"
    (ROOT / "docs/stage4_report.md").write_text(report, encoding="utf-8")
    print(f"Best validation AP: {best}; script runtime {elapsed:.2f}s; sanity checks passed; test reads=0", flush=True)


if __name__ == "__main__":
    main()
