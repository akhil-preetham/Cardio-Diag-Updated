"""Execute CARDIODIAG Research Protocol v2 using the frozen Stage 6 manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "primary_research_v2"
DATA_PATH = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
STAGE6_MANIFEST = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
STAGE6_METADATA = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "experiment_metadata_v2.json"
FEATURES = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]
THRESHOLD = 0.5
BOOTSTRAP_REPS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                [feature for feature in FEATURES if feature != "Sex"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["Sex"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def model_specs() -> dict[str, dict]:
    return {
        "logistic_regression_none": {"name": "Logistic Regression", "strategy": "none", "estimator": LogisticRegression(max_iter=3000, solver="liblinear", random_state=1), "grid": {"model__C": [0.1, 1.0, 10.0]}},
        "logistic_regression_balanced": {"name": "Logistic Regression", "strategy": "balanced", "estimator": LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=1), "grid": {"model__C": [0.1, 1.0, 10.0]}},
        "svm_none": {"name": "Support Vector Classifier", "strategy": "none", "estimator": SVC(probability=False, random_state=1), "grid": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]}},
        "svm_balanced": {"name": "Support Vector Classifier", "strategy": "balanced", "estimator": SVC(probability=False, class_weight="balanced", random_state=1), "grid": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]}},
        "random_forest_none": {"name": "Random Forest", "strategy": "none", "estimator": RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=1), "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]}},
        "random_forest_balanced": {"name": "Random Forest", "strategy": "balanced", "estimator": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1), "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]}},
        "hist_gradient_boosting_none": {"name": "HistGradientBoosting", "strategy": "none", "estimator": HistGradientBoostingClassifier(max_iter=150, random_state=1), "grid": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}},
        "extra_trees_none": {"name": "Extra Trees", "strategy": "none", "estimator": ExtraTreesClassifier(n_estimators=200, random_state=1, n_jobs=1), "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]}},
        "extra_trees_balanced": {"name": "Extra Trees", "strategy": "balanced", "estimator": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1), "grid": {"model__max_depth": [None, 4], "model__min_samples_leaf": [1, 3]}},
    }


def load_manifest() -> dict:
    manifest = json.loads(STAGE6_MANIFEST.read_text(encoding="utf-8"))
    from runner.methodology_corrections_v2.split_manifest import validate_nested_split_manifest

    validation = validate_nested_split_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(f"Stage 6 split manifest is invalid: {validation}")
    return manifest


def assignment_groups(manifest: dict) -> tuple[dict, dict]:
    outer: dict[tuple[int, int], dict[str, set[int]]] = {}
    inner: dict[tuple[int, int, int], dict[str, set[int]]] = {}
    for row in manifest["outer_assignments"]:
        outer.setdefault((row["repeat"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    for row in manifest["inner_assignments"]:
        inner.setdefault((row["repeat"], row["outer_fold"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    return outer, inner


def output_value(estimator, frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(frame)[:, 1], "native_probability"
    return estimator.decision_function(frame), "decision_score"


def fit_pipeline(spec: dict, params: dict | None = None) -> Pipeline:
    pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", clone(spec["estimator"]))])
    if params:
        pipeline.set_params(**params)
    return pipeline


def inner_splits_for_group(outer_train: list[int], group: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = {row: position for position, row in enumerate(outer_train)}
    splits = []
    for fold in sorted(group):
        train = np.array([positions[row] for row in sorted(group[fold]["train"])], dtype=int)
        test = np.array([positions[row] for row in sorted(group[fold]["test"])], dtype=int)
        splits.append((train, test))
    return splits


def cross_fitted_calibration(
    spec: dict,
    best_params: dict,
    x: pd.DataFrame,
    y: np.ndarray,
    outer_train: list[int],
    inner_group: dict[int, dict[str, set[int]]],
) -> tuple[LogisticRegression, dict[int, float], list[str]]:
    """Fit Platt calibration on cross-fitted inner-training predictions.

    Each calibration prediction is generated by a pipeline fit without that row.
    The resulting score vector, and only that vector, is used to fit the calibrator.
    """
    scores: dict[int, float] = {}
    warnings_seen: list[str] = []
    for fold in sorted(inner_group):
        train_rows = sorted(inner_group[fold]["train"])
        calibration_rows = sorted(inner_group[fold]["test"])
        pipeline = fit_pipeline(spec, best_params)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(x.iloc[train_rows], y[train_rows])
            values, _ = output_value(pipeline, x.iloc[calibration_rows])
        warnings_seen.extend(str(item.message) for item in captured)
        scores.update({row: float(value) for row, value in zip(calibration_rows, values)})
    if set(scores) != set(outer_train):
        raise ValueError("Cross-fitted calibration predictions do not cover outer training rows exactly")
    raw = np.array([scores[row] for row in outer_train], dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        calibrator.fit(raw, y[outer_train])
    warnings_seen.extend(str(item.message) for item in captured)
    return calibrator, scores, sorted(set(warnings_seen))


def calibrated_probability(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


def metric_row(y_true: np.ndarray, values: np.ndarray, model_id: str, probability_type: str, repeat: int, fold: int, thresholded: bool) -> dict:
    row = {
        "level": "outer_fold",
        "model_id": model_id,
        "probability_type": probability_type,
        "repeat": repeat,
        "fold": fold,
        "n_test": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, values)),
        "pr_auc": float(average_precision_score(y_true, values)),
    }
    if not thresholded:
        row.update({key: np.nan for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss", "tn", "fp", "fn", "tp", "threshold"]})
        return row
    prediction = (values >= THRESHOLD).astype(int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    row.update(
        {
            "accuracy": float(accuracy_score(y_true, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
            "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp) if tn + fp else np.nan),
            "precision": float(precision_score(y_true, prediction, zero_division=0)),
            "f1": float(f1_score(y_true, prediction, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, values)),
            "log_loss": float(log_loss(y_true, values, labels=[0, 1])),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "threshold": THRESHOLD,
        }
    )
    return row


def calibration_row(y_true: np.ndarray, probability: np.ndarray, model_id: str, repeat: int, fold: int) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    evaluation_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluation_model.fit(logit, y_true)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return {
        "level": "outer_fold",
        "model_id": model_id,
        "probability_type": "nested_sigmoid",
        "repeat": repeat,
        "fold": fold,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "calibration_intercept": float(evaluation_model.intercept_[0]),
        "calibration_slope": float(evaluation_model.coef_[0, 0]),
        "expected_calibration_error_10_bins": float(ece),
        "calibration_bins": "equal_width_10_supplementary",
    }


def summarize(frame: pd.DataFrame, keys: list[str], metrics: list[str], level: str) -> pd.DataFrame:
    rows = []
    for group_keys, group in frame.groupby(keys, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(keys, group_keys))
        row.update({"level": level, "n_rows": len(group)})
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean()) if group[metric].notna().any() else np.nan
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if group[metric].notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_values(y_true: np.ndarray, values: np.ndarray, metric: str, rng: np.random.Generator) -> tuple[float, float, float]:
    point = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier_score": brier_score_loss}[metric](y_true, values)
    estimates = []
    for _ in range(BOOTSTRAP_REPS):
        index = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[index])) < 2:
            continue
        estimates.append({"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier_score": brier_score_loss}[metric](y_true[index], values[index]))
    return float(point), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def aggregate(predictions: pd.DataFrame, model_id: str, column: str) -> pd.DataFrame:
    selected = predictions[predictions["model_id"] == model_id]
    return selected.groupby("row_index", as_index=False).agg(y_true=("y_true", "first"), value=(column, "mean"))


def run() -> None:
    manifest = load_manifest()
    frame = pd.read_csv(DATA_PATH)
    if list(frame.columns).count("Cath") != 1 or any(feature not in frame for feature in FEATURES):
        raise ValueError("Dataset does not match the frozen target and strict feature policy")
    y = (frame["Cath"] == "Cad").astype(int).to_numpy()
    x = frame[FEATURES].copy()
    outer_groups, inner_groups = assignment_groups(manifest)
    if len(outer_groups) != 25 or len(inner_groups) != 100:
        raise ValueError("Unexpected persisted split-manifest dimensions")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SPLIT_MANIFEST_USED.json").write_bytes(STAGE6_MANIFEST.read_bytes())

    specs = model_specs()
    split_rows, inner_rows, prediction_rows = [], [], []
    metric_rows, calibration_rows, hyper_rows, importance_rows = [], [], [], []
    failed_rows = []
    for row in manifest["outer_assignments"]:
        split_rows.append(row)
    for row in manifest["inner_assignments"]:
        inner_rows.append(row)

    for (repeat, fold), group in sorted(outer_groups.items()):
        outer_train = sorted(group["train"])
        outer_test = sorted(group["test"])
        local_inner = {key[2]: value for key, value in inner_groups.items() if key[:2] == (repeat, fold)}
        x_train, x_test = x.iloc[outer_train], x.iloc[outer_test]
        y_train, y_test = y[outer_train], y[outer_test]
        cv = inner_splits_for_group(outer_train, local_inner)
        for model_id, spec in specs.items():
            started = time.perf_counter()
            warnings_seen: list[str] = []
            try:
                pipeline = fit_pipeline(spec)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    search = GridSearchCV(pipeline, spec["grid"], scoring="roc_auc", cv=cv, n_jobs=1, refit=True, error_score="raise")
                    search.fit(x_train, y_train)
                warnings_seen.extend(str(item.message) for item in captured)
                best_params = search.best_params_
                calibrator, _, calibration_warnings = cross_fitted_calibration(spec, best_params, x, y, outer_train, local_inner)
                warnings_seen.extend(calibration_warnings)
                best = search.best_estimator_
                native_test, native_type = output_value(best, x_test)
                calibrated_test = calibrated_probability(calibrator, native_test)
                for row_index, truth, native, calibrated in zip(outer_test, y_test, native_test, calibrated_test):
                    prediction_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "seed": manifest["outer_seeds"][repeat - 1],
                            "row_index": row_index,
                            "source_row_id": f"source_row:{row_index}",
                            "model_id": model_id,
                            "model_name": spec["name"],
                            "imbalance_strategy": spec["strategy"],
                            "y_true": int(truth),
                            "native_output": float(native),
                            "native_output_type": native_type,
                            "calibrated_probability": float(calibrated),
                            "calibrated_prediction": int(calibrated >= THRESHOLD),
                        }
                    )
                metric_rows.append(metric_row(y_test, native_test, model_id, native_type, repeat, fold, native_type == "native_probability"))
                metric_rows.append(metric_row(y_test, calibrated_test, model_id, "nested_sigmoid", repeat, fold, True))
                calibration_rows.append(calibration_row(y_test, calibrated_test, model_id, repeat, fold))
                if native_type == "native_probability":
                    calibration_rows.append(
                        {
                            **calibration_row(y_test, native_test, model_id, repeat, fold),
                            "probability_type": "native_probability",
                        }
                    )
                importance = permutation_importance(best, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=manifest["outer_seeds"][repeat - 1] + fold, n_jobs=1)
                for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std):
                    importance_rows.append({"repeat": repeat, "fold": fold, "model_id": model_id, "feature": feature, "importance_mean": float(mean), "importance_std": float(std), "importance_positive": bool(mean > 0)})
                hyper_rows.append({"repeat": repeat, "fold": fold, "model_id": model_id, "model_name": spec["name"], "imbalance_strategy": spec["strategy"], "inner_best_roc_auc": float(search.best_score_), "best_params": json.dumps(best_params, sort_keys=True), "fit_seconds": time.perf_counter() - started, "warning_count": len(set(warnings_seen)), "warnings": " | ".join(sorted(set(warnings_seen))), "status": "success", "calibration_method": "nested_cross_fitted_sigmoid"})
            except Exception as exc:
                failed_rows.append({"repeat": repeat, "fold": fold, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
                hyper_rows.append({"repeat": repeat, "fold": fold, "model_id": model_id, "model_name": spec["name"], "imbalance_strategy": spec["strategy"], "inner_best_roc_auc": np.nan, "best_params": "{}", "fit_seconds": time.perf_counter() - started, "warning_count": len(set(warnings_seen)), "warnings": " | ".join(sorted(set(warnings_seen))), "status": f"failed: {type(exc).__name__}: {exc}", "calibration_method": "nested_cross_fitted_sigmoid"})

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    hyperparameters = pd.DataFrame(hyper_rows)
    if failed_rows:
        raise RuntimeError(f"Model-fold failures occurred: {failed_rows}")
    if len(predictions) != 303 * 5 * len(specs):
        raise RuntimeError(f"Unexpected prediction count: {len(predictions)}")
    metrics = pd.concat(
        [
            metrics,
            summarize(metrics, ["model_id", "probability_type", "repeat"], ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"], "repeat_summary"),
            summarize(metrics, ["model_id", "probability_type"], ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"], "aggregate_summary"),
        ],
        ignore_index=True,
    )
    calibration = pd.concat(
        [
            calibration,
            summarize(calibration, ["model_id", "probability_type", "repeat"], ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"], "repeat_summary"),
            summarize(calibration, ["model_id", "probability_type"], ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"], "aggregate_summary"),
        ],
        ignore_index=True,
    )

    uncertainty_rows = []
    rng = np.random.default_rng(20261113)
    for model_id in specs:
        for column, output_type in [("calibrated_probability", "nested_sigmoid"), ("native_output", "native_probability")]:
            if output_type == "native_probability" and predictions[predictions.model_id == model_id]["native_output_type"].eq("decision_score").all():
                continue
            aggregated = aggregate(predictions, model_id, column)
            for metric in ("roc_auc", "pr_auc", "brier_score"):
                point, lower, upper = bootstrap_values(aggregated.y_true.to_numpy(), aggregated.value.to_numpy(), metric, rng)
                uncertainty_rows.append({"model_id": model_id, "probability_type": output_type, "metric": metric, "point_estimate": point, "bootstrap_lower_95": lower, "bootstrap_upper_95": upper, "bootstrap_replicates": BOOTSTRAP_REPS, "unit": "aggregated repeated-CV source-row predictions", "interpretation": "conditional resampling summary; not a population confidence interval; no refitting"})
    uncertainty = pd.DataFrame(uncertainty_rows)

    baseline = aggregate(predictions, "logistic_regression_none", "calibrated_probability")
    statistics_rows = []
    for model_id in specs:
        if model_id == "logistic_regression_none":
            continue
        candidate = aggregate(predictions, model_id, "calibrated_probability")
        merged = baseline.merge(candidate, on="row_index", suffixes=("_baseline", "_candidate"))
        y_values = merged["y_true_baseline"].to_numpy()
        differences = []
        for _ in range(BOOTSTRAP_REPS):
            index = rng.integers(0, len(merged), size=len(merged))
            if len(np.unique(y_values[index])) == 2:
                differences.append(roc_auc_score(y_values[index], merged["value_candidate"].to_numpy()[index]) - roc_auc_score(y_values[index], merged["value_baseline"].to_numpy()[index]))
        statistics_rows.append({"comparison": f"{model_id} minus logistic_regression_none", "metric": "roc_auc", "effect_estimate": float(roc_auc_score(y_values, merged["value_candidate"]) - roc_auc_score(y_values, merged["value_baseline"])), "bootstrap_lower_95": float(np.percentile(differences, 2.5)), "bootstrap_upper_95": float(np.percentile(differences, 97.5)), "p_value": np.nan, "method": "paired source-row bootstrap on aggregated repeated-CV predictions; descriptive only", "multiple_comparison_adjustment": "not applied; no confirmatory p-values"})
    statistics = pd.DataFrame(statistics_rows)

    robustness = pd.DataFrame(
        [
            {"analysis": "imbalance_strategy_sensitivity", "model_id": model_id, "strategy": spec["strategy"], "calibrated_roc_auc_mean": float(metrics[(metrics.model_id == model_id) & (metrics.probability_type == "nested_sigmoid") & (metrics.level == "outer_fold")].roc_auc.mean()), "status": "completed"}
            for model_id, spec in specs.items()
        ]
        + [{"analysis": "feature_timing_sensitivity", "model_id": "not_run", "strategy": "strict_primary_only", "calibrated_roc_auc_mean": np.nan, "status": "uncertain-timing features excluded by policy"}]
    )

    audit = pd.read_csv(FEATURE_AUDIT)
    validation = {"source": str(DATA_PATH.relative_to(ROOT)), "rows": len(frame), "columns": frame.shape[1], "target": "Cath", "class_counts": {str(k): int(v) for k, v in frame.Cath.value_counts().to_dict().items()}, "strict_features": FEATURES, "feature_definitions": {str(row.original_feature_name): str(row.clinical_meaning) for row in audit[audit.original_feature_name.isin(FEATURES)].itertuples()}, "missing_values": {str(k): int(v) for k, v in frame[FEATURES].isna().sum().to_dict().items()}, "exact_duplicate_rows": int(frame.duplicated().sum()), "strict_predictor_duplicate_rows": int(frame[FEATURES].duplicated().sum()), "raw_data_modified": False}

    for path, data in [
        ("SPLIT_ASSIGNMENTS.csv", pd.DataFrame(split_rows)),
        ("INNER_FOLD_ASSIGNMENTS.csv", pd.DataFrame(inner_rows)),
        ("OUTER_FOLD_PREDICTIONS.csv", predictions),
        ("MODEL_COMPARISON_RESULTS.csv", metrics),
        ("CALIBRATION_RESULTS.csv", calibration),
        ("HYPERPARAMETER_SELECTION_RESULTS.csv", hyperparameters),
        ("UNCERTAINTY_RESULTS.csv", uncertainty),
        ("STATISTICAL_COMPARISONS.csv", statistics),
        ("FEATURE_IMPORTANCE_RESULTS.csv", pd.DataFrame(importance_rows)),
        ("ROBUSTNESS_RESULTS.csv", robustness),
    ]:
        data.to_csv(OUT / path, index=False)
    (OUT / "DATA_VALIDATION_REPORT.md").write_text("# Data Validation Report\n\n" + json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    best = metrics[(metrics.level == "aggregate_summary") & (metrics.probability_type == "nested_sigmoid")].sort_values("roc_auc_mean", ascending=False).iloc[0]
    (OUT / "EXECUTIVE_RESULTS_REPORT.md").write_text(
        f"# Executive Results Report\n\n"
        f"This is the corrected V2 internal-validation experiment. It is not external validation or clinical deployment.\n\n"
        f"- Records: {len(frame)}; class counts: {validation['class_counts']}.\n"
        f"- Strict features: {', '.join(FEATURES)}.\n"
        f"- Persisted Stage 6 manifest used exactly: yes.\n"
        f"- Nested cross-fitted sigmoid calibration: yes.\n"
        f"- Descriptive highest calibrated ROC-AUC: `{best.model_id}` = {best.roc_auc_mean:.3f} +/- {best.roc_auc_std:.3f}.\n\n"
        f"Model selection and calibration used only outer-training data. SVM decision scores are not treated as native probabilities. "
        f"Comparisons are paired descriptive summaries without confirmatory p-values. Bootstrap outputs are conditional source-row resampling summaries, not population confidence intervals.\n",
        encoding="utf-8",
    )
    (OUT / "PRIMARY_RESEARCH_PROTOCOL_EXECUTION.md").write_text(
        "# Primary Research Protocol V2 Execution\n\n"
        "The frozen Stage 6 protocol was executed using the persisted nested split manifest. "
        "The strict five-feature set was used without undocumented predictors. The experiment "
        "did not use external validation, patient identifiers, SMOTE, or threshold optimization. "
        "Sigmoid calibration was fit to cross-fitted inner-training predictions and applied "
        "only after the final outer-training estimator was fitted.\n",
        encoding="utf-8",
    )
    (OUT / "LIMITATIONS_AND_VALIDITY_ASSESSMENT.md").write_text(
        "# Limitations and Validity Assessment\n\n"
        "- This is repeated internal validation on 303 records, not independent external validation.\n"
        "- Source-row labels are not patient identifiers; patient-level grouping cannot be proven from the derivative.\n"
        "- Calibration metrics and ten-bin ECE are unstable at outer-fold sample sizes.\n"
        "- Conditional bootstrap summaries do not capture refitting or population sampling uncertainty.\n"
        "- Predictive permutation importance is not causal evidence.\n"
        "- Strict-feature timing is conservative but not completely documented.\n",
        encoding="utf-8",
    )
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    plot = metrics[(metrics.level == "aggregate_summary") & (metrics.probability_type == "nested_sigmoid")].sort_values("roc_auc_mean", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot.model_id, plot.roc_auc_mean, yerr=plot.roc_auc_std, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Calibrated outer ROC-AUC (mean +/- SD)")
    plt.tight_layout()
    plt.savefig(figures / "calibrated_roc_auc_summary.png", dpi=160)
    plt.close()
    selected = aggregate(predictions, best.model_id, "calibrated_probability")
    fraction, mean_pred = calibration_curve(selected.y_true, selected.value, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(mean_pred, fraction, "o-")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.tight_layout()
    plt.savefig(figures / "best_model_reliability.png", dpi=160)
    plt.close()

    config = {
        "protocol_version": "RESEARCH_PROTOCOL_V2",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "target": "Cath",
        "target_encoding": {"Cad": 1, "Normal": 0},
        "features": FEATURES,
        "outer_seeds": manifest["outer_seeds"],
        "outer_folds": manifest["outer_folds"],
        "inner_folds": manifest["inner_folds"],
        "threshold": THRESHOLD,
        "model_configurations": {key: {"name": value["name"], "strategy": value["strategy"], "grid": value["grid"]} for key, value in specs.items()},
        "calibration": {"method": "sigmoid", "strategy": "cross-fitted inner-training predictions", "outer_test_labels_used_for_fit": False},
        "patient_identifier_available": False,
        "split_manifest_sha256": sha256(STAGE6_MANIFEST),
        "source_row_identifier_policy": "not a patient identifier",
    }
    write_json(OUT / "EXPERIMENT_CONFIGURATION.json", config)
    inputs = [DATA_PATH, FEATURE_AUDIT, STAGE6_MANIFEST, STAGE6_METADATA]
    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json"]
    manifest_out = {
        "experiment_version": "primary_research_v2",
        "generated_utc": utc_now(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "split_manifest_used": {"path": str(STAGE6_MANIFEST.relative_to(ROOT)), "sha256": sha256(STAGE6_MANIFEST), "copied_output": "SPLIT_MANIFEST_USED.json"},
        "execution_policy": {"raw_data_modified": False, "historical_results_modified": False, "external_validation": False, "new_model_families": False, "patient_identifiers_created": False, "model_training_executed": True},
        "failed_model_folds": len(failed_rows),
        "prediction_rows": len(predictions),
        "calibration_policy": "nested cross-fitted sigmoid; outer test labels never used for calibrator fit",
    }
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", manifest_out)


if __name__ == "__main__":
    run()
