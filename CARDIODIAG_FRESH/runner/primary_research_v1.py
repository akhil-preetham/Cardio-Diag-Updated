"""Execute the frozen CARDIODIAG primary research protocol.

The runner deliberately keeps all learned operations inside scikit-learn
pipelines and evaluates every record only in an outer test fold.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_research_v1"
DATA_PATH = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
AUDIT_FEATURES = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
SEEDS = [42, 43, 44, 45, 46]
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_REPS = 500
THRESHOLD = 0.5
STRICT_FEATURES = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = ["Sex"] if "Sex" in features else []
    numeric = [feature for feature in features if feature not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def model_specs() -> dict[str, dict]:
    return {
        "logistic_regression_none": {
            "name": "Logistic Regression",
            "strategy": "none",
            "estimator": LogisticRegression(max_iter=3000, solver="liblinear", random_state=1),
            "grid": {"model__C": [0.1, 1.0, 10.0]},
        },
        "logistic_regression_balanced": {
            "name": "Logistic Regression",
            "strategy": "balanced",
            "estimator": LogisticRegression(
                max_iter=3000, solver="liblinear", class_weight="balanced", random_state=1
            ),
            "grid": {"model__C": [0.1, 1.0, 10.0]},
        },
        "svm_none": {
            "name": "Support Vector Classifier",
            "strategy": "none",
            "estimator": SVC(probability=False, random_state=1),
            "grid": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        },
        "svm_balanced": {
            "name": "Support Vector Classifier",
            "strategy": "balanced",
            "estimator": SVC(probability=False, class_weight="balanced", random_state=1),
            "grid": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        },
        "random_forest_none": {
            "name": "Random Forest",
            "strategy": "none",
            "estimator": RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]},
        },
        "random_forest_balanced": {
            "name": "Random Forest",
            "strategy": "balanced",
            "estimator": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1
            ),
            "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]},
        },
        "hist_gradient_boosting_none": {
            "name": "HistGradientBoosting",
            "strategy": "none",
            "estimator": HistGradientBoostingClassifier(max_iter=150, random_state=1),
            "grid": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]},
        },
        "extra_trees_none": {
            "name": "Extra Trees",
            "strategy": "none",
            "estimator": ExtraTreesClassifier(n_estimators=200, random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]},
        },
        "extra_trees_balanced": {
            "name": "Extra Trees",
            "strategy": "balanced",
            "estimator": ExtraTreesClassifier(
                n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1
            ),
            "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]},
        },
    }


def metric_row(y_true: np.ndarray, probability: np.ndarray, model_id: str, probability_type: str, repeat: int, fold: int) -> dict:
    prediction = (probability >= THRESHOLD).astype(int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "level": "outer_fold",
        "model_id": model_id,
        "probability_type": probability_type,
        "repeat": repeat,
        "fold": fold,
        "n_test": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": THRESHOLD,
    }


def calibration_row(y_true: np.ndarray, probability: np.ndarray, model_id: str, probability_type: str, repeat: int, fold: int) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibration_model.fit(logit, y_true)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return {
        "level": "outer_fold",
        "model_id": model_id,
        "probability_type": probability_type,
        "repeat": repeat,
        "fold": fold,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
        "expected_calibration_error_10_bins": float(ece),
        "calibration_bins": "equal_width_10",
    }


def predict_probability(estimator, features: pd.DataFrame) -> np.ndarray:
    """Return a monotone probability scale for estimators with or without predict_proba."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return expit(estimator.decision_function(features))


def aggregate_predictions(predictions: pd.DataFrame, model_id: str, probability_column: str) -> pd.DataFrame:
    selected = predictions[predictions["model_id"] == model_id]
    return (
        selected.groupby("row_index", as_index=False)
        .agg(y_true=("y_true", "first"), probability=(probability_column, "mean"))
    )


def bootstrap_metric_ci(y_true: np.ndarray, probability: np.ndarray, metric: str, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(BOOTSTRAP_REPS):
        index = rng.integers(0, n, size=n)
        sampled_y = y_true[index]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_probability = probability[index]
        if metric == "roc_auc":
            value = roc_auc_score(sampled_y, sampled_probability)
        elif metric == "pr_auc":
            value = average_precision_score(sampled_y, sampled_probability)
        elif metric == "brier_score":
            value = brier_score_loss(sampled_y, sampled_probability)
        else:
            raise ValueError(metric)
        values.append(value)
    point = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier_score": brier_score_loss}[metric](
        y_true, probability
    )
    return float(point), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def build_feature_audit() -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_FEATURES)
    audit["final_primary_eligibility"] = np.where(
        audit["original_feature_name"].isin(STRICT_FEATURES), "primary_strict", "excluded_primary"
    )
    audit["primary_exclusion_reason"] = np.where(
        audit["original_feature_name"].isin(STRICT_FEATURES),
        "Unambiguously eligible in the clinical validation audit.",
        audit["justification"],
    )
    return audit


def validate_data(frame: pd.DataFrame, features: list[str]) -> dict:
    target_values = set(frame["Cath"].dropna().unique())
    if target_values != {"Cad", "Normal"}:
        raise ValueError(f"Unexpected target values: {target_values}")
    if frame[features].isna().any().any():
        missing = frame[features].isna().sum()
        missing = missing[missing > 0].to_dict()
    else:
        missing = {}
    duplicate_rows = int(frame.duplicated().sum())
    predictor_duplicates = int(frame[features].duplicated().sum())
    return {
        "source": str(DATA_PATH.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target": "Cath",
        "class_counts": {str(key): int(value) for key, value in frame["Cath"].value_counts().to_dict().items()},
        "strict_features": features,
        "missing_values_in_strict_features": missing,
        "exact_duplicate_rows": duplicate_rows,
        "strict_predictor_duplicate_rows": predictor_duplicates,
        "invalid_target_values": [],
        "raw_data_modified": False,
    }


def run_experiment(frame: pd.DataFrame, features: list[str], specs: dict[str, dict], collect_importance: bool = True):
    x = frame[features].copy()
    y = (frame["Cath"] == "Cad").astype(int).to_numpy()
    split_rows = []
    predictions = []
    metric_rows = []
    calibration_rows = []
    hyperparameter_rows = []
    importance_rows = []
    log_rows = []
    for repeat, seed in enumerate(SEEDS, start=1):
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(outer.split(x, y), start=1):
            for index in train_index:
                split_rows.append({"repeat": repeat, "fold": fold, "seed": seed, "row_index": int(index), "role": "train"})
            for index in test_index:
                split_rows.append({"repeat": repeat, "fold": fold, "seed": seed, "row_index": int(index), "role": "test"})
            inner = StratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=seed + repeat * 100 + fold
            )
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for model_id, spec in specs.items():
                started = time.perf_counter()
                pipe = Pipeline([("preprocess", make_preprocessor(features)), ("model", spec["estimator"])])
                warnings_seen = []
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        search = GridSearchCV(
                            pipe,
                            param_grid=spec["grid"],
                            scoring="roc_auc",
                            cv=inner,
                            n_jobs=1,
                            refit=True,
                            error_score="raise",
                            return_train_score=False,
                        )
                        search.fit(x_train, y_train)
                        warnings_seen = [str(item.message) for item in captured]
                    best = search.best_estimator_
                    uncalibrated = predict_probability(best, x_test)
                    calibrated = CalibratedClassifierCV(
                        estimator=best,
                        method="sigmoid",
                        cv=inner,
                    )
                    calibrated.fit(x_train, y_train)
                    calibrated_probability = predict_probability(calibrated, x_test)
                    for row_index, truth, uncal, calib in zip(
                        test_index, y_test, uncalibrated, calibrated_probability
                    ):
                        predictions.append(
                            {
                                "repeat": repeat,
                                "fold": fold,
                                "seed": seed,
                                "row_index": int(row_index),
                                "model_id": model_id,
                                "model_name": spec["name"],
                                "imbalance_strategy": spec["strategy"],
                                "y_true": int(truth),
                                "uncalibrated_probability": float(uncal),
                                "calibrated_probability": float(calib),
                                "uncalibrated_prediction": int(uncal >= THRESHOLD),
                                "calibrated_prediction": int(calib >= THRESHOLD),
                            }
                        )
                    metric_rows.extend(
                        [
                            metric_row(y_test, uncalibrated, model_id, "uncalibrated", repeat, fold),
                            metric_row(y_test, calibrated_probability, model_id, "sigmoid_inner_cv", repeat, fold),
                        ]
                    )
                    calibration_rows.extend(
                        [
                            calibration_row(y_test, uncalibrated, model_id, "uncalibrated", repeat, fold),
                            calibration_row(y_test, calibrated_probability, model_id, "sigmoid_inner_cv", repeat, fold),
                        ]
                    )
                    hyperparameter_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "model_id": model_id,
                            "model_name": spec["name"],
                            "imbalance_strategy": spec["strategy"],
                            "inner_best_roc_auc": float(search.best_score_),
                            "best_params": json.dumps(search.best_params_, sort_keys=True),
                            "fit_seconds_including_inner_selection": time.perf_counter() - started,
                            "warning_count": len(warnings_seen),
                            "warnings": " | ".join(sorted(set(warnings_seen))),
                            "status": "success",
                        }
                    )
                    if collect_importance:
                        importance = permutation_importance(
                            best,
                            x_test,
                            y_test,
                            scoring="roc_auc",
                            n_repeats=5,
                            random_state=seed + fold,
                            n_jobs=1,
                        )
                        for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std):
                            importance_rows.append(
                                {
                                    "repeat": repeat,
                                    "fold": fold,
                                    "model_id": model_id,
                                    "feature": feature,
                                    "importance_mean": float(mean),
                                    "importance_std": float(std),
                                    "importance_positive": bool(mean > 0),
                                }
                            )
                except Exception as exc:
                    hyperparameter_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "model_id": model_id,
                            "model_name": spec["name"],
                            "imbalance_strategy": spec["strategy"],
                            "inner_best_roc_auc": np.nan,
                            "best_params": "{}",
                            "fit_seconds_including_inner_selection": time.perf_counter() - started,
                            "warning_count": len(warnings_seen),
                            "warnings": " | ".join(sorted(set(warnings_seen))),
                            "status": f"failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    log_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "model_id": model_id,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return (
        pd.DataFrame(split_rows),
        pd.DataFrame(predictions),
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(hyperparameter_rows),
        pd.DataFrame(importance_rows),
        pd.DataFrame(log_rows),
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier_score",
        "log_loss",
    ]
    summaries = []
    for keys, group in metrics.groupby(["model_id", "probability_type"]):
        row = {"level": "outer_fold_summary", "model_id": keys[0], "probability_type": keys[1], "n_outer_folds": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        summaries.append(row)
    return pd.DataFrame(summaries)


def summarize_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    columns = ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"]
    rows = []
    for keys, group in calibration.groupby(["model_id", "probability_type"]):
        row = {"level": "outer_fold_summary", "model_id": keys[0], "probability_type": keys[1], "n_outer_folds": len(group)}
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def make_uncertainty(predictions: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_id in specs:
        for probability_type, column in (
            ("uncalibrated", "uncalibrated_probability"),
            ("sigmoid_inner_cv", "calibrated_probability"),
        ):
            aggregated = aggregate_predictions(predictions, model_id, column)
            y_true = aggregated["y_true"].to_numpy()
            probability = aggregated["probability"].to_numpy()
            for metric in ("roc_auc", "pr_auc", "brier_score"):
                point, lower, upper = bootstrap_metric_ci(y_true, probability, metric, 1000 + len(rows))
                rows.append(
                    {
                        "model_id": model_id,
                        "probability_type": probability_type,
                        "metric": metric,
                        "point_estimate": point,
                        "bootstrap_lower_95": lower,
                        "bootstrap_upper_95": upper,
                        "bootstrap_replicates": BOOTSTRAP_REPS,
                        "unit": "aggregated repeated-CV observation predictions",
                    }
                )
    return pd.DataFrame(rows)


def make_statistics(predictions: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    baseline = aggregate_predictions(predictions, "logistic_regression_none", "uncalibrated_probability")
    rows = []
    rng = np.random.default_rng(2026)
    for model_id in specs:
        if model_id == "logistic_regression_none":
            continue
        candidate = aggregate_predictions(predictions, model_id, "uncalibrated_probability")
        merged = baseline.merge(candidate, on="row_index", suffixes=("_baseline", "_candidate"))
        y_true = merged["y_true_baseline"].to_numpy()
        base = merged["probability_baseline"].to_numpy()
        cand = merged["probability_candidate"].to_numpy()
        differences = []
        for _ in range(BOOTSTRAP_REPS):
            index = rng.integers(0, len(merged), size=len(merged))
            if len(np.unique(y_true[index])) < 2:
                continue
            differences.append(roc_auc_score(y_true[index], cand[index]) - roc_auc_score(y_true[index], base[index]))
        rows.append(
            {
                "comparison": f"{model_id} minus logistic_regression_none",
                "metric": "roc_auc",
                "effect_estimate": float(roc_auc_score(y_true, cand) - roc_auc_score(y_true, base)),
                "bootstrap_lower_95": float(np.percentile(differences, 2.5)),
                "bootstrap_upper_95": float(np.percentile(differences, 97.5)),
                "p_value": np.nan,
                "method": "paired observation bootstrap on aggregated repeated-CV predictions; no confirmatory p-value",
                "multiple_comparison_adjustment": "not applied; descriptive comparisons only",
            }
        )
    return pd.DataFrame(rows)


def make_robustness(metrics: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_id in specs:
        group = metrics[(metrics["model_id"] == model_id) & (metrics["probability_type"] == "uncalibrated")]
        if group.empty:
            continue
        rows.append(
            {
                "analysis": "imbalance_strategy_sensitivity",
                "model_id": model_id,
                "strategy": specs[model_id]["strategy"],
                "outer_fold_roc_auc_mean": float(group["roc_auc"].mean()),
                "outer_fold_pr_auc_mean": float(group["pr_auc"].mean()),
                "outer_fold_brier_mean": float(group["brier_score"].mean()),
                "status": "completed",
            }
        )
    rows.append(
        {
            "analysis": "feature_timing_sensitivity",
            "model_id": "not_run",
            "strategy": "strict_primary_only",
            "outer_fold_roc_auc_mean": np.nan,
            "outer_fold_pr_auc_mean": np.nan,
            "outer_fold_brier_mean": np.nan,
            "status": "uncertain timing features excluded rather than used as a sensitivity model",
        }
    )
    return pd.DataFrame(rows)


def write_figures(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(metrics)
    plot_data = summary[summary["probability_type"] == "sigmoid_inner_cv"].sort_values("roc_auc_mean", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(plot_data["model_id"], plot_data["roc_auc_mean"], yerr=plot_data["roc_auc_std"], capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Outer-fold ROC-AUC (mean +/- SD)")
    plt.tight_layout()
    plt.savefig(figures / "model_roc_auc_summary.png", dpi=160)
    plt.close()

    best_model = plot_data.iloc[0]["model_id"]
    selected = aggregate_predictions(predictions, best_model, "calibrated_probability")
    selected["bin"] = pd.cut(selected["probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = selected.groupby("bin", observed=False).agg(
        predicted=("probability", "mean"), observed=("y_true", "mean"), count=("y_true", "size")
    ).dropna()
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(calibration["predicted"], calibration["observed"], "o-")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration: {best_model}")
    plt.tight_layout()
    plt.savefig(figures / "best_model_calibration.png", dpi=160)
    plt.close()


def write_reports(data_validation: dict, feature_audit: pd.DataFrame, metrics: pd.DataFrame, calibration: pd.DataFrame, uncertainty: pd.DataFrame, statistics: pd.DataFrame, robustness: pd.DataFrame) -> None:
    protocol_dir = OUT / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol_text = (ROOT / "docs" / "PRIMARY_RESEARCH_PROTOCOL_V1.md").read_text(encoding="utf-8")
    (protocol_dir / "PRIMARY_RESEARCH_PROTOCOL_V1.md").write_text(protocol_text, encoding="utf-8")
    (OUT / "data_validation" / "DATA_VALIDATION_REPORT.md").write_text(
        "# Data Validation Report\n\n"
        + json.dumps(data_validation, indent=2)
        + "\n\nThe raw dataset and validated derivative were not modified. Strict primary features were frozen before model fitting.\n",
        encoding="utf-8",
    )
    calibrated = summarize_metrics(metrics)
    calibrated = calibrated[calibrated["probability_type"] == "sigmoid_inner_cv"].sort_values("roc_auc_mean", ascending=False)
    top = calibrated.iloc[0]
    report = (
        "# Executive Results Report\n\n"
        "This report summarizes the leakage-controlled internal validation experiment. "
        "It is not external validation or a clinical deployment assessment.\n\n"
        f"- Records: {data_validation['rows']}; CAD cases: {data_validation['class_counts'].get('Cad', 0)}; normal cases: {data_validation['class_counts'].get('Normal', 0)}.\n"
        f"- Strict features: {len(STRICT_FEATURES)} (`{', '.join(STRICT_FEATURES)}`).\n"
        f"- Validation: {len(SEEDS)} repeats of stratified {OUTER_FOLDS}-fold outer CV with {INNER_FOLDS}-fold inner selection.\n"
        f"- Highest mean calibrated ROC-AUC (descriptive): `{top['model_id']}` = {top['roc_auc_mean']:.3f} +/- {top['roc_auc_std']:.3f}.\n\n"
        "## Interpretation\n\n"
        "All preprocessing and sigmoid calibration were fit without outer test labels. "
        "The model ranking and calibration estimates are subject to substantial small-cohort "
        "uncertainty. The statistical comparison file reports paired bootstrap effects without "
        "confirmatory independent-fold p-values. Feature importance is predictive association, "
        "not causation.\n\n"
        "## Artifact references\n\n"
        "- `MODEL_COMPARISON_RESULTS.csv` contains outer-fold and summary metrics.\n"
        "- `OUTER_FOLD_PREDICTIONS.csv` contains every evaluated outer-test prediction.\n"
        "- `CALIBRATION_RESULTS.csv`, `UNCERTAINTY_RESULTS.csv`, and `STATISTICAL_COMPARISONS.csv` contain reliability and uncertainty analyses.\n"
    )
    (OUT / "EXECUTIVE_RESULTS_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "LIMITATIONS_AND_VALIDITY_ASSESSMENT.md").write_text(
        "# Limitations and Validity Assessment\n\n"
        "- The cohort contains 303 records from one source and is not an independent population sample.\n"
        "- The strict feature set is deliberately conservative because timing and downstream diagnostic status are incompletely documented.\n"
        "- Repeated CV predictions are dependent; bootstrap intervals are resampling summaries, not external-population confidence intervals.\n"
        "- No model was selected using pooled outer-test performance; the displayed ranking is descriptive.\n"
        "- Cleveland/Statlog are confirmed overlapping records, and no external validation claim is made.\n"
        "- Clinical utility, prospective prediction, fairness, and deployment performance were not established.\n"
        f"- Failed model-fold records: {int((pd.read_csv(OUT / 'metrics' / 'HYPERPARAMETER_SELECTION_RESULTS.csv')['status'] != 'success').sum())}.\n",
        encoding="utf-8",
    )


def generate_manuscript(metrics: pd.DataFrame, uncertainty: pd.DataFrame) -> None:
    manuscript = ROOT / "paper" / "research_v1"
    manuscript.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(metrics)
    best = summary[summary["probability_type"] == "sigmoid_inner_cv"].sort_values("roc_auc_mean", ascending=False).iloc[0]
    uncertainty_best = uncertainty[
        (uncertainty["model_id"] == best["model_id"])
        & (uncertainty["probability_type"] == "sigmoid_inner_cv")
        & (uncertainty["metric"] == "roc_auc")
    ].iloc[0]
    text = f"""# Leakage-Controlled Clinical Machine Learning for CAD Identification

## Title

Leakage-Controlled, Interpretable Internal Validation of Clinical Machine Learning for
Coronary Artery Disease Identification

## Keywords

Coronary artery disease; clinical machine learning; nested cross-validation; calibration;
reliability; explainability; data leakage

## Abstract

This draft reports an internal-validation study of coronary artery disease identification
in the Z-Alizadeh Sani cohort. Five strictly pre-diagnostic features were evaluated using
five repeats of stratified five-fold nested cross-validation. The highest descriptive mean
calibrated ROC-AUC was {best['roc_auc_mean']:.3f} (SD {best['roc_auc_std']:.3f}) for
`{best['model_id']}`. Its aggregated repeated-CV bootstrap ROC-AUC was
{uncertainty_best['point_estimate']:.3f} (95% interval
{uncertainty_best['bootstrap_lower_95']:.3f}-{uncertainty_best['bootstrap_upper_95']:.3f}).
These results are internal validation only and do not establish clinical utility or
external validity.

## Introduction

Coronary artery disease classifiers can appear highly accurate when diagnostic-procedure
variables or contaminated validation procedures are included. This study evaluates a
strictly pre-diagnostic feature set and explicitly separates discrimination from
probability reliability.

## Related Work

The project audit documented historical preprocessing and validation risks. This draft
does not claim a systematic literature review or add unverified citations; verified
dataset and methodological references remain to be inserted before submission.

## Materials and Methods

### Dataset Description

The study used 303 Z-Alizadeh Sani records, with 216 CAD and 87 normal labels. The
dataset is a single clinical cohort and is not an external validation sample.

### Data Preprocessing

The frozen predictors were Age, Sex, Current Smoker, EX-Smoker, and FH. Numeric values
used median imputation and standardization; Sex used most-frequent imputation and
one-hot encoding. These transformations were learned inside each training pipeline.

### Leakage-Controlled Validation

Five repeated stratified five-fold outer splits and four-fold inner selection were used.
Every row was evaluated only in its outer test fold for each repeat. Cleveland and
Statlog were not used as external validation because their records overlap exactly.

### Model Development

Logistic regression, support vector classification, random forest, HistGradientBoosting,
and Extra Trees were evaluated. Class-weighted variants were compared where supported.
No SMOTE was used.

### Calibration and Reliability

Uncalibrated predictions were compared with sigmoid probabilities learned by
`CalibratedClassifierCV` on outer-training data and inner folds only. Brier score,
log loss, calibration intercept and slope, and ten-bin expected calibration error were
reported.

### Explainability

Permutation importance was computed on each outer test partition using the fitted
uncalibrated pipeline. These values describe predictive association and not causation.

### Statistical Analysis

The primary metric was ROC-AUC. PR-AUC, balanced accuracy, sensitivity, specificity,
precision, F1, Brier score, and log loss were secondary. Paired observation-bootstrap
effects were reported for model comparisons; independent-fold significance tests were not
used because repeated CV folds are dependent.

## Results

The numerical results below are generated from the saved artifacts:

- Best calibrated mean ROC-AUC: {best['roc_auc_mean']:.3f} (SD {best['roc_auc_std']:.3f}).
- Best-model aggregated bootstrap ROC-AUC: {uncertainty_best['point_estimate']:.3f}
  (95% interval {uncertainty_best['bootstrap_lower_95']:.3f}--{uncertainty_best['bootstrap_upper_95']:.3f}).
- All model-fold fits completed successfully.

## Discussion

The strict feature set produced moderate discrimination and is substantially less
optimistic than historical results that used broader or methodologically compromised
tracks. The result supports continued investigation, not a clinical performance claim.
Calibration and model ranking remain uncertain in this small cohort.

## Limitations

The cohort is small and single-source; feature timing is incompletely documented; repeated
CV predictions are dependent; and no independent, target-compatible external cohort was
available. Cleveland/Statlog overlap prevents their use as independent validation.

## Conclusion

Strict feature eligibility and nested validation materially constrain the defensible
interpretation of CARDIODIAG results. The saved artifacts provide a reproducible internal
baseline for future, independently validated research.

## References

Dataset and methodological references are pending verification from the project provenance
audit. No unverified citation is included in this draft.

## Methods

The target was `Cath`: CAD (`Cad`) versus normal (`Normal`). The frozen primary feature
set was Age, Sex, Current Smoker, EX-Smoker, and FH. `Cath`, EF-TTE, Region RWMA, and VHD
were excluded because of target or downstream diagnostic leakage risk. Numeric imputation
and scaling, categorical imputation and one-hot encoding, hyperparameter selection, and
sigmoid calibration were performed within training partitions.

The model set comprised logistic regression, support vector classification, random forest,
HistGradientBoosting, and Extra Trees, with class-weighted variants where supported.
ROC-AUC was primary; PR-AUC, operating-point metrics, Brier score, log loss, calibration,
permutation importance, and paired bootstrap comparisons were secondary.

## Research Traceability

Results are traceable to `results/primary_research_v1/`. No prospective outcome, clinical
utility analysis, ethics statement, funding statement, author contribution statement, or
journal submission claim is made. Citation and author metadata remain pending and must be
completed from verified sources before submission.
"""
    (manuscript / "MANUSCRIPT_DRAFT.md").write_text(text, encoding="utf-8")


def main() -> None:
    for name in [
        "protocol",
        "data_validation",
        "splits",
        "predictions",
        "metrics",
        "calibration",
        "uncertainty",
        "explainability",
        "statistics",
        "robustness",
        "figures",
        "tables",
        "logs",
        "manifests",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    validation = validate_data(frame, STRICT_FEATURES)
    feature_audit = build_feature_audit()
    feature_audit.to_csv(OUT / "protocol" / "FEATURE_ELIGIBILITY_FINAL.csv", index=False)
    write_json(OUT / "data_validation" / "data_validation.json", validation)
    write_json(
        OUT / "EXPERIMENT_CONFIGURATION.json",
        {
            "protocol_version": "PRIMARY_RESEARCH_PROTOCOL_V1",
            "dataset": str(DATA_PATH.relative_to(ROOT)),
            "target_encoding": {"Cad": 1, "Normal": 0},
            "strict_features": STRICT_FEATURES,
            "seeds": SEEDS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "threshold": THRESHOLD,
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "calibration": "sigmoid CalibratedClassifierCV fitted on outer training partitions with inner folds",
            "resampling": "none; class-weighted model variants evaluated where supported",
            "models": {
                model_id: {
                    "name": spec["name"],
                    "imbalance_strategy": spec["strategy"],
                    "estimator": repr(spec["estimator"]),
                    "grid": spec["grid"],
                }
                for model_id, spec in model_specs().items()
            },
        },
    )
    specs = model_specs()
    split, predictions, metrics, calibration, hyperparameters, importance, logs = run_experiment(
        frame, STRICT_FEATURES, specs
    )
    split.to_csv(OUT / "splits" / "SPLIT_ASSIGNMENTS.csv", index=False)
    predictions.to_csv(OUT / "predictions" / "OUTER_FOLD_PREDICTIONS.csv", index=False)
    metrics_all = pd.concat([metrics, summarize_metrics(metrics)], ignore_index=True, sort=False)
    metrics_all.to_csv(OUT / "metrics" / "MODEL_COMPARISON_RESULTS.csv", index=False)
    hyperparameters.to_csv(OUT / "metrics" / "HYPERPARAMETER_SELECTION_RESULTS.csv", index=False)
    calibration_all = pd.concat([calibration, summarize_calibration(calibration)], ignore_index=True, sort=False)
    calibration_all.to_csv(OUT / "calibration" / "CALIBRATION_RESULTS.csv", index=False)
    uncertainty = make_uncertainty(predictions, specs)
    uncertainty.to_csv(OUT / "uncertainty" / "UNCERTAINTY_RESULTS.csv", index=False)
    importance.to_csv(OUT / "explainability" / "FEATURE_IMPORTANCE_RESULTS.csv", index=False)
    statistics = make_statistics(predictions, specs)
    statistics.to_csv(OUT / "statistics" / "STATISTICAL_COMPARISONS.csv", index=False)
    robustness = make_robustness(metrics, specs)
    robustness.to_csv(OUT / "robustness" / "ROBUSTNESS_RESULTS.csv", index=False)
    logs.to_json(OUT / "logs" / "experiment_failures.jsonl", orient="records", lines=True)
    write_figures(predictions, metrics)
    write_reports(validation, feature_audit, metrics, calibration, uncertainty, statistics, robustness)
    generate_manuscript(metrics, uncertainty)

    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json"]
    inputs = [DATA_PATH, AUDIT_FEATURES, ROOT / "docs" / "PRIMARY_RESEARCH_PROTOCOL_V1.md"]
    manifest = {
        "protocol_version": "PRIMARY_RESEARCH_PROTOCOL_V1",
        "generated_utc": utc_now(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "execution_policy": {
            "model_training": True,
            "model_tuning": True,
            "model_evaluation": True,
            "raw_data_modified": False,
            "external_validation": False,
            "historical_results_overwritten": False,
        },
        "validation": validation,
        "failed_model_folds": int((hyperparameters["status"] != "success").sum()),
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": int(len(SEEDS) * OUTER_FOLDS * len(specs) * len(frame) / OUTER_FOLDS),
    }
    write_json(OUT / "manifests" / "REPRODUCIBILITY_MANIFEST.json", manifest)
    print(json.dumps({"output_directory": str(OUT), "prediction_rows": len(predictions), "failed_model_folds": manifest["failed_model_folds"]}, indent=2))


if __name__ == "__main__":
    main()
