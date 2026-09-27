"""Leakage-resistant reproduction for the UCI SECOM case study.

The script downloads the public UCI archive, verifies its SHA-256, performs a
fixed stratified holdout, fits preprocessing inside training folds, selects an
operating threshold from training OOF scores, and evaluates the untouched
holdout once.  ``--feature-audit`` repeats feature-count CV with importance
ranking fitted inside every fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ARCHIVE_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
RANDOM_STATE = 42


def ensure_data() -> None:
    """Download and verify the official public archive if data is absent."""
    data_file = DATA / "secom.data"
    label_file = DATA / "secom_labels.data"
    if data_file.exists() and label_file.exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "secom.zip"
    urllib.request.urlretrieve(ARCHIVE_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"archive SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA)
    if not data_file.exists() or not label_file.exists():
        raise FileNotFoundError("SECOM files were not found after extraction")


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    ensure_data()
    X = pd.read_csv(DATA / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(
        DATA / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "date", "time"],
        quotechar='"',
    )
    X.columns = [f"S{i + 1:03d}" for i in range(X.shape[1])]
    y = (labels["label"].to_numpy() == 1).astype(int)
    return X, y


def prepare(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit filtering and imputation on the training partition only."""
    missing = train_df.isna().mean()
    unique = train_df.nunique(dropna=True)
    keep = [c for c in train_df.columns if missing[c] <= 0.50 and unique[c] > 1]
    medians = train_df[keep].median(axis=0)
    train = train_df[keep].fillna(medians).to_numpy(dtype=float)
    valid = valid_df[keep].fillna(medians).to_numpy(dtype=float)
    return train, valid, keep


def make_rf(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def fit_predict(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    valid_df: pd.DataFrame,
    feature_count: int | None,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, list[str]]:
    train, valid, columns = prepare(train_df, valid_df)
    if feature_count is None or feature_count >= train.shape[1]:
        selected = np.arange(train.shape[1])
    else:
        ranker = make_rf(seed, n_estimators)
        ranker.fit(train, y_train)
        selected = np.argsort(ranker.feature_importances_)[::-1][:feature_count]
    model = make_rf(seed + 1, n_estimators)
    model.fit(train[:, selected], y_train)
    probability = model.predict_proba(valid[:, selected])[:, 1]
    return probability, [columns[i] for i in selected]


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "reinspection_rate": float((tp + fp) / len(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    repetitions: int = 2000,
) -> dict:
    rng = np.random.default_rng(20260927)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    values = {k: [] for k in ("recall", "precision", "pr_auc", "reinspection_rate")}
    for _ in range(repetitions):
        index = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        row = metric_row(y_true[index], probability[index], threshold)
        for key in values:
            values[key].append(row[key])
    return {
        key: {
            "low": float(np.quantile(item, 0.025)),
            "high": float(np.quantile(item, 0.975)),
        }
        for key, item in values.items()
    }


def fixed_split(X: pd.DataFrame, y: np.ndarray):
    index = np.arange(len(y))
    train_index, test_index = train_test_split(
        index, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    return (
        X.iloc[train_index].reset_index(drop=True),
        X.iloc[test_index].reset_index(drop=True),
        y[train_index],
        y[test_index],
    )


def run_final_validation() -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = fixed_split(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train), dtype=float)
    fold_ap, fold_feature_counts = [], []
    for train, valid in cv.split(X_train, y_train):
        probability, columns = fit_predict(
            X_train.iloc[train],
            y_train[train],
            X_train.iloc[valid],
            feature_count=None,
            seed=41,
            n_estimators=500,
        )
        oof[valid] = probability
        fold_ap.append(float(average_precision_score(y_train[valid], probability)))
        fold_feature_counts.append(len(columns))

    threshold = float(np.quantile(oof[y_train == 1], 0.30))
    test_probability, final_columns = fit_predict(
        X_train,
        y_train,
        X_test,
        feature_count=None,
        seed=41,
        n_estimators=500,
    )
    result = {
        "split_random_state": 42,
        "cv_random_state": 42,
        "rf_random_state": 42,
        "n_estimators": 500,
        "train_n": len(y_train),
        "train_fails": int(y_train.sum()),
        "holdout_n": len(y_test),
        "holdout_fails": int(y_test.sum()),
        "fold_feature_counts": fold_feature_counts,
        "final_feature_count": len(final_columns),
        "oof_pr_auc": float(average_precision_score(y_train, oof)),
        "fold_pr_auc": fold_ap,
        "threshold": threshold,
        "threshold_rule": "train 5-fold OOF positive-score 30th percentile (target recall 70%)",
        "holdout_default_0_5": metric_row(y_test, test_probability, 0.5),
        "holdout_tuned": metric_row(y_test, test_probability, threshold),
        "bootstrap_95_ci": bootstrap_ci(y_test, test_probability, threshold),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_seed42_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"y": y_train, "probability": oof}).to_csv(
        RESULTS / "final_seed42_train_oof.csv", index=False
    )
    pd.DataFrame({"y": y_test, "probability": test_probability}).to_csv(
        RESULTS / "final_seed42_holdout.csv", index=False
    )
    return result


def run_feature_audit() -> list[dict]:
    X, y = load_data()
    X_train, _, y_train, _ = fixed_split(X, y)
    records = []
    candidates: list[int | None] = [20, 40, 80, None]
    for repeat, cv_seed in enumerate((42, 123, 2026), start=1):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        for position, feature_count in enumerate(candidates):
            oof = np.zeros(len(y_train), dtype=float)
            for fold, (train, valid) in enumerate(cv.split(X_train, y_train), start=1):
                probability, _ = fit_predict(
                    X_train.iloc[train],
                    y_train[train],
                    X_train.iloc[valid],
                    feature_count=feature_count,
                    seed=50000 + repeat * 1000 + position * 100 + fold,
                    n_estimators=220,
                )
                oof[valid] = probability
            records.append(
                {
                    "repeat": repeat,
                    "cv_seed": cv_seed,
                    "feature_count": "all" if feature_count is None else feature_count,
                    "oof_pr_auc": float(average_precision_score(y_train, oof)),
                }
            )
    frame = pd.DataFrame(records)
    summary = []
    for feature_count, group in frame.groupby("feature_count", sort=False):
        summary.append(
            {
                "feature_count": feature_count,
                "mean_pr_auc": float(group["oof_pr_auc"].mean()),
                "std_pr_auc": float(group["oof_pr_auc"].std(ddof=1)),
                "min_pr_auc": float(group["oof_pr_auc"].min()),
                "max_pr_auc": float(group["oof_pr_auc"].max()),
            }
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "repeated_feature_count_cv.csv", index=False)
    (RESULTS / "repeated_feature_count_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-audit",
        action="store_true",
        help="also run repeated fold-local feature-count comparison",
    )
    args = parser.parse_args()
    print(json.dumps(run_final_validation(), ensure_ascii=False, indent=2))
    if args.feature_audit:
        print(json.dumps(run_feature_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
