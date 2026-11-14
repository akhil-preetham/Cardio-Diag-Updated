"""Run a compact, fully nested performance optimization study."""
from __future__ import annotations

import hashlib
import itertools
import json
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from runner.primary_research_v2.run import (
    FEATURES,
    ROOT,
    THRESHOLD,
    aggregate,
    bootstrap_values,
    calibration_row,
    fit_pipeline,
    inner_splits_for_group,
    metric_row,
    output_value,
    sha256,
    summarize,
    write_json,
)
from runner.primary_research_v3.run import (
    assignment_groups,
    calibrated_probability,
    fit_calibrator,
    fit_calibration_fold,
    load_manifest,
)

OUT = ROOT / "results" / "primary_research_v4_performance"
DATA_PATH = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
STAGE6_MANIFEST = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
STAGE6_METADATA = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "experiment_metadata_v2.json"
BOOTSTRAP_REPS = 500


def performance_specs() -> dict[str, dict]:
    """Compact predeclared extensions of the V3 search spaces."""
    return {
        "logistic_regression_none": {
            "name": "Logistic Regression",
            "strategy": "none",
            "estimator": LogisticRegression(max_iter=3000, solver="liblinear", random_state=1),
            "grid": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "complexity": "linear; five regularization values",
        },
        "logistic_regression_balanced": {
            "name": "Logistic Regression",
            "strategy": "balanced",
            "estimator": LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=1),
            "grid": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "complexity": "linear; five regularization values",
        },
        "svm_none": {
            "name": "Support Vector Classifier",
            "strategy": "none",
            "estimator": SVC(probability=False, random_state=1),
            "grid": {"model__C": [0.1, 1.0, 10.0, 100.0], "model__gamma": ["scale", "auto"]},
            "complexity": "kernel; compact C/gamma grid",
        },
        "svm_balanced": {
            "name": "Support Vector Classifier",
            "strategy": "balanced",
            "estimator": SVC(probability=False, class_weight="balanced", random_state=1),
            "grid": {"model__C": [0.1, 1.0, 10.0, 100.0], "model__gamma": ["scale", "auto"]},
            "complexity": "kernel; compact C/gamma grid",
        },
        "random_forest_none": {
            "name": "Random Forest",
            "strategy": "none",
            "estimator": RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 3, 5]},
            "complexity": "200 trees; depth/leaf grid",
        },
        "random_forest_balanced": {
            "name": "Random Forest",
            "strategy": "balanced",
            "estimator": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 3, 5]},
            "complexity": "200 trees; depth/leaf grid",
        },
        "hist_gradient_boosting_none": {
            "name": "HistGradientBoosting",
            "strategy": "none",
            "estimator": HistGradientBoostingClassifier(max_iter=150, random_state=1),
            "grid": {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0.0, 1.0]},
            "complexity": "boosting; learning-rate/tree/l2 grid",
        },
        "extra_trees_none": {
            "name": "Extra Trees",
            "strategy": "none",
            "estimator": ExtraTreesClassifier(n_estimators=200, random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 3, 5]},
            "complexity": "200 randomized trees; depth/leaf grid",
        },
        "extra_trees_balanced": {
            "name": "Extra Trees",
            "strategy": "balanced",
            "estimator": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=1, n_jobs=1),
            "grid": {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 3, 5]},
            "complexity": "200 randomized trees; depth/leaf grid",
        },
    }


def metric_columns() -> list[str]:
    return ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    frame = pd.read_csv(DATA_PATH)
    x = frame[FEATURES].copy()
    y = (frame["Cath"] == "Cad").astype(int).to_numpy()
    outer_groups, inner_groups = assignment_groups(manifest)
    specs = performance_specs()
    split_rows = list(manifest["outer_assignments"])
    inner_rows = list(manifest["inner_assignments"])
    selection_rows, prediction_rows, metric_rows = [], [], []
    calibration_rows, hyper_rows, importance_rows = [], [], []
    failures = []

    for (repeat, outer_fold), outer_group in sorted(outer_groups.items()):
        outer_train = sorted(outer_group["train"])
        outer_test = sorted(outer_group["test"])
        local_inner = {key[2]: value for key, value in inner_groups.items() if key[:2] == (repeat, outer_fold)}
        final_cv = inner_splits_for_group(outer_train, local_inner)
        for model_id, spec in specs.items():
            started = time.perf_counter()
            try:
                scores = {}
                for calibration_fold in sorted(local_inner):
                    score, record, detail = fit_calibration_fold(spec, model_id, x, y, local_inner, calibration_fold)
                    scores.update({row: float(value) for row, value in zip(detail["rows"], score)})
                    record.update({"repeat": repeat, "outer_fold": outer_fold, "study": "performance_optimization_v4"})
                    selection_rows.append(record)
                calibrator, fit_details = fit_calibrator(scores, outer_train, y)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    final_search = GridSearchCV(fit_pipeline(spec), spec["grid"], scoring="roc_auc", cv=final_cv, n_jobs=1, refit=True, error_score="raise")
                    final_search.fit(x.iloc[outer_train], y[outer_train])
                best = final_search.best_estimator_
                native, native_type = output_value(best, x.iloc[outer_test])
                calibrated = calibrated_probability(calibrator, native)
                for row_index, truth, native_value, calibrated_value in zip(outer_test, y[outer_test], native, calibrated):
                    prediction_rows.append(
                        {
                            "study": "performance_optimization_v4",
                            "repeat": repeat,
                            "fold": outer_fold,
                            "seed": manifest["outer_seeds"][repeat - 1],
                            "row_index": row_index,
                            "source_row_id": f"source_row:{row_index}",
                            "model_id": model_id,
                            "model_name": spec["name"],
                            "imbalance_strategy": spec["strategy"],
                            "complexity_description": spec["complexity"],
                            "y_true": int(truth),
                            "native_output": float(native_value),
                            "native_output_type": native_type,
                            "calibrated_probability": float(calibrated_value),
                            "calibrated_prediction": int(calibrated_value >= THRESHOLD),
                        }
                    )
                metric_rows.extend(
                    [
                        metric_row(y[outer_test], native, model_id, native_type, repeat, outer_fold, native_type == "native_probability"),
                        metric_row(y[outer_test], calibrated, model_id, "fully_nested_sigmoid", repeat, outer_fold, True),
                    ]
                )
                calibration_metric = calibration_row(y[outer_test], calibrated, model_id, repeat, outer_fold)
                calibration_metric.update(fit_details)
                calibration_metric["probability_type"] = "fully_nested_sigmoid"
                calibration_rows.append(calibration_metric)
                if native_type == "native_probability":
                    native_calibration = calibration_row(y[outer_test], native, model_id, repeat, outer_fold)
                    native_calibration["probability_type"] = "native_probability"
                    calibration_rows.append(native_calibration)
                importance = permutation_importance(best, x.iloc[outer_test], y[outer_test], scoring="roc_auc", n_repeats=5, random_state=manifest["outer_seeds"][repeat - 1] + outer_fold, n_jobs=1)
                for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std):
                    importance_rows.append({"level": "outer_fold", "repeat": repeat, "fold": outer_fold, "model_id": model_id, "feature": feature, "importance_mean": float(mean), "importance_std": float(std), "importance_positive": bool(mean > 0)})
                hyper_rows.append({"repeat": repeat, "fold": outer_fold, "model_id": model_id, "model_name": spec["name"], "imbalance_strategy": spec["strategy"], "complexity_description": spec["complexity"], "inner_best_roc_auc": float(final_search.best_score_), "best_params": json.dumps(final_search.best_params_, sort_keys=True), "fit_seconds": time.perf_counter() - started, "warning_count": len(captured), "warnings": " | ".join(sorted(set(str(item.message) for item in captured))), "status": "success", "calibration_method": "fully_nested_cross_fitted_sigmoid", **fit_details})
            except Exception as exc:
                failures.append({"repeat": repeat, "fold": outer_fold, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"})
                raise

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    if len(predictions) != 303 * 5 * len(specs):
        raise RuntimeError(f"Unexpected prediction count: {len(predictions)}")
    metrics = pd.concat([metrics, summarize(metrics, ["model_id", "probability_type", "repeat"], metric_columns(), "repeat_summary"), summarize(metrics, ["model_id", "probability_type"], metric_columns(), "aggregate_summary")], ignore_index=True)
    cal_cols = ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"]
    calibration = pd.concat([calibration, summarize(calibration, ["model_id", "probability_type", "repeat"], cal_cols, "repeat_summary"), summarize(calibration, ["model_id", "probability_type"], cal_cols, "aggregate_summary")], ignore_index=True)

    importance = pd.DataFrame(importance_rows)
    repeat_importance = importance.groupby(["repeat", "model_id", "feature"]).agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), importance_positive=("importance_positive", "mean")).reset_index()
    repeat_importance["level"] = "repeat_summary"
    aggregate_importance = importance.groupby(["model_id", "feature"]).agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), importance_positive=("importance_positive", "mean")).reset_index()
    aggregate_importance["level"] = "aggregate_summary"
    importance = pd.concat([importance, repeat_importance, aggregate_importance], ignore_index=True)

    rng = np.random.default_rng(20261113)
    uncertainty_rows = []
    for model_id in specs:
        aggregate_predictions = aggregate(predictions, model_id, "calibrated_probability")
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            point, lower, upper = bootstrap_values(aggregate_predictions.y_true.to_numpy(), aggregate_predictions.value.to_numpy(), metric, rng)
            uncertainty_rows.append({"model_id": model_id, "probability_type": "fully_nested_sigmoid", "metric": metric, "point_estimate": point, "bootstrap_lower_95": lower, "bootstrap_upper_95": upper, "bootstrap_replicates": BOOTSTRAP_REPS, "unit": "aggregated repeated-CV source-row predictions", "interpretation": "conditional resampling summary; not population confidence interval; no refitting"})
    uncertainty = pd.DataFrame(uncertainty_rows)

    baseline = aggregate(predictions, "logistic_regression_none", "calibrated_probability")
    statistics_rows = []
    for model_id in specs:
        if model_id == "logistic_regression_none":
            continue
        candidate = aggregate(predictions, model_id, "calibrated_probability")
        merged = baseline.merge(candidate, on="row_index", suffixes=("_baseline", "_candidate"))
        y_values = merged.y_true_baseline.to_numpy()
        differences = []
        for _ in range(BOOTSTRAP_REPS):
            index = rng.integers(0, len(merged), size=len(merged))
            if len(np.unique(y_values[index])) == 2:
                differences.append(roc_auc_score(y_values[index], merged.value_candidate.to_numpy()[index]) - roc_auc_score(y_values[index], merged.value_baseline.to_numpy()[index]))
        statistics_rows.append({"comparison": f"{model_id} minus logistic_regression_none", "metric": "roc_auc", "effect_estimate": float(roc_auc_score(y_values, merged.value_candidate) - roc_auc_score(y_values, merged.value_baseline)), "bootstrap_lower_95": float(np.percentile(differences, 2.5)), "bootstrap_upper_95": float(np.percentile(differences, 97.5)), "p_value": np.nan, "method": "paired source-row bootstrap on aggregated repeated-CV predictions; descriptive only", "multiple_comparison_adjustment": "not applied; no confirmatory p-values"})
    statistics = pd.DataFrame(statistics_rows)

    summary = metrics[(metrics.level == "aggregate_summary") & (metrics.probability_type == "fully_nested_sigmoid")].copy()
    pareto = summary[["model_id", "roc_auc_mean", "pr_auc_mean", "brier_score_mean", "log_loss_mean"]].copy()
    pareto["complexity"] = pareto.model_id.map({key: value["complexity"] for key, value in specs.items()})
    pareto["roc_auc_rank"] = pareto.roc_auc_mean.rank(ascending=False, method="min").astype(int)
    pareto["brier_rank"] = pareto.brier_score_mean.rank(ascending=True, method="min").astype(int)
    pareto["pareto_efficient"] = ~((pareto.roc_auc_mean.values[:, None] <= pareto.roc_auc_mean.values).all(axis=1) & (pareto.brier_score_mean.values[:, None] >= pareto.brier_score_mean.values).all(axis=1))

    feature_sensitivity = pd.DataFrame([{"analysis": "strict_primary", "features": "|".join(FEATURES), "status": "completed", "reason": "prespecified primary feature set"}, {"analysis": "additional_features", "features": "", "status": "not_run", "reason": "clinical timing/definition evidence insufficient under Stage 6 policy; no outer-test-driven selection"}])
    robustness = pd.DataFrame([{"analysis": "regularization_and_complexity", "model_id": key, "strategy": spec["strategy"], "complexity": spec["complexity"], "status": "completed"} for key, spec in specs.items()] + [{"analysis": "feature_timing_sensitivity", "model_id": "not_run", "strategy": "strict_primary_only", "complexity": "", "status": "additional features excluded by policy"}])

    for name, data in [("CALIBRATION_SELECTION_RESULTS.csv", pd.DataFrame(selection_rows)), ("MODEL_COMPARISON_RESULTS.csv", metrics), ("OUTER_FOLD_PREDICTIONS.csv", predictions), ("SPLIT_ASSIGNMENTS.csv", pd.DataFrame(split_rows)), ("INNER_FOLD_ASSIGNMENTS.csv", pd.DataFrame(inner_rows)), ("HYPERPARAMETER_SELECTION_RESULTS.csv", pd.DataFrame(hyper_rows)), ("CALIBRATION_RESULTS.csv", calibration), ("UNCERTAINTY_RESULTS.csv", uncertainty), ("STATISTICAL_COMPARISONS.csv", statistics), ("FEATURE_IMPORTANCE_RESULTS.csv", importance), ("ROBUSTNESS_RESULTS.csv", robustness), ("FEATURE_SET_SENSITIVITY.csv", feature_sensitivity), ("PARETO_ANALYSIS.csv", pareto)]:
        data.to_csv(OUT / name, index=False)
    best = summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    (OUT / "PERFORMANCE_OPTIMIZATION_PLAN.md").write_text("# Performance Optimization Plan\n\nThe study preserves the five strict predictors and the five existing model families. It expands regularization and compact complexity grids only within training folds, retains balanced/unbalanced configurations, and uses the V3 fully nested calibration architecture. Additional clinical features were not tested because timing and definitions remain insufficiently documented.\n", encoding="utf-8")
    (OUT / "EXECUTIVE_PERFORMANCE_REPORT.md").write_text(f"# Executive Performance Report\n\nThe structured performance study used exact Stage 6 splits and fully nested calibration. No new predictors or model families were added.\n\n- Best descriptive calibrated ROC-AUC: `{best.model_id}` = {best.roc_auc_mean:.3f} +/- {best.roc_auc_std:.3f}.\n- Performance and calibration were evaluated separately.\n- Statistical comparisons and bootstrap intervals are descriptive, not confirmatory population inference.\n- The preferred model must be selected by the prespecified primary metric and stability, not by isolated best scores.\n", encoding="utf-8")
    (OUT / "LIMITATIONS_AND_VALIDITY_ASSESSMENT.md").write_text("# Performance Study Limitations\n\n- This remains repeated internal validation on 303 records.\n- The compact grid expansion increases selection opportunity and requires independent validation.\n- Conditional source-row bootstrap summaries do not capture refitting or population uncertainty.\n- Additional features were not evaluated because their clinical timing/definitions were not sufficiently documented.\n- Feature importance is predictive, not causal.\n", encoding="utf-8")
    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    plot = summary.sort_values("roc_auc_mean", ascending=False)
    plt.figure(figsize=(10, 5)); plt.bar(plot.model_id, plot.roc_auc_mean, yerr=plot.roc_auc_std, capsize=3); plt.xticks(rotation=45, ha="right"); plt.ylabel("Fully nested calibrated ROC-AUC"); plt.tight_layout(); plt.savefig(figures / "performance_roc_auc.png", dpi=160); plt.close()

    fit_count = 25 * sum(16 * len(list(itertools.product(*spec["grid"].values()))) + 5 for spec in specs.values())
    config = {"study_version": "primary_research_v4_performance", "dataset": str(DATA_PATH.relative_to(ROOT)), "target": "Cath", "features": FEATURES, "outer_seeds": manifest["outer_seeds"], "outer_folds": manifest["outer_folds"], "inner_folds": manifest["inner_folds"], "calibration": "V3 fully nested fold-specific hyperparameter selection", "model_configurations": {key: {"name": value["name"], "strategy": value["strategy"], "grid": value["grid"], "complexity": value["complexity"]} for key, value in specs.items()}, "additional_features_evaluated": False, "reason_additional_features_not_evaluated": "timing/definition evidence insufficient", "fit_count_estimate": fit_count, "n_jobs": 1, "split_manifest_sha256": sha256(STAGE6_MANIFEST)}
    write_json(OUT / "EXPERIMENT_CONFIGURATION.json", config)
    inputs = [DATA_PATH, FEATURE_AUDIT, STAGE6_MANIFEST, STAGE6_METADATA]
    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json"]
    reproducibility = {"study_version": "primary_research_v4_performance", "generated_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha256(Path(__file__)), "environment": {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs}, "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths}, "split_manifest_sha256": sha256(STAGE6_MANIFEST), "failed_model_folds": len(failures), "prediction_rows": len(predictions), "historical_results_modified": False, "model_families_added": False}
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", reproducibility)


if __name__ == "__main__":
    run()
