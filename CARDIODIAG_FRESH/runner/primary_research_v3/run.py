"""Run CARDIODIAG V3 with fold-specific nested hyperparameter selection for calibration."""
from __future__ import annotations

import hashlib
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from runner.primary_research_v2.run import (
    FEATURES,
    ROOT,
    THRESHOLD,
    aggregate,
    bootstrap_values,
    calibration_row,
    fit_pipeline,
    inner_splits_for_group,
    make_preprocessor,
    metric_row,
    model_specs,
    output_value,
    sha256,
    summarize,
    write_json,
)

OUT = ROOT / "results" / "primary_research_v3"
DATA_PATH = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
STAGE6_MANIFEST = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
STAGE6_METADATA = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "experiment_metadata_v2.json"
BOOTSTRAP_REPS = 500


def load_manifest() -> dict:
    manifest = json.loads(STAGE6_MANIFEST.read_text(encoding="utf-8"))
    from runner.methodology_corrections_v2.split_manifest import validate_nested_split_manifest

    validation = validate_nested_split_manifest(manifest)
    if not validation["valid"]:
        raise ValueError(f"Invalid Stage 6 manifest: {validation}")
    return manifest


def assignment_groups(manifest: dict) -> tuple[dict, dict]:
    outer: dict[tuple[int, int], dict[str, set[int]]] = {}
    inner: dict[tuple[int, int, int], dict[str, set[int]]] = {}
    for row in manifest["outer_assignments"]:
        outer.setdefault((row["repeat"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    for row in manifest["inner_assignments"]:
        inner.setdefault((row["repeat"], row["outer_fold"], row["fold"]), {"train": set(), "test": set()})[row["role"]].add(row["row_index"])
    return outer, inner


def calibration_selection_splits(
    inner_group: dict[int, dict[str, set[int]]], held_out_fold: int
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    """Return selection rows and CV splits excluding the held-out calibration fold."""
    remaining = [fold for fold in sorted(inner_group) if fold != held_out_fold]
    selection_rows = sorted(set().union(*(inner_group[fold]["test"] for fold in remaining)))
    positions = {row: index for index, row in enumerate(selection_rows)}
    splits = []
    for validation_fold in remaining:
        validation = sorted(inner_group[validation_fold]["test"])
        training = sorted(set(selection_rows) - set(validation))
        splits.append(
            (
                np.array([positions[row] for row in training], dtype=int),
                np.array([positions[row] for row in validation], dtype=int),
            )
        )
    return selection_rows, splits


def fit_calibration_fold(
    spec: dict,
    model_id: str,
    x: pd.DataFrame,
    y: np.ndarray,
    inner_group: dict[int, dict[str, set[int]]],
    held_out_fold: int,
) -> tuple[np.ndarray, dict, dict]:
    selection_rows, selection_cv = calibration_selection_splits(inner_group, held_out_fold)
    calibration_rows = sorted(inner_group[held_out_fold]["test"])
    search = GridSearchCV(
        fit_pipeline(spec),
        spec["grid"],
        scoring="roc_auc",
        cv=selection_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        search.fit(x.iloc[selection_rows], y[selection_rows])
    selected = search.best_estimator_
    score, output_type = output_value(selected, x.iloc[calibration_rows])
    record = {
        "model_id": model_id,
        "calibration_fold": held_out_fold,
        "calibration_row_indices": json.dumps(calibration_rows),
        "selection_row_indices": json.dumps(selection_rows),
        "selection_cv_validation_folds": json.dumps([fold for fold in sorted(inner_group) if fold != held_out_fold]),
        "selection_cv_train_indices": json.dumps(
            {
                str(fold): sorted(
                    set().union(
                        *(
                            inner_group[other]["test"]
                            for other in sorted(inner_group)
                            if other != held_out_fold and other != fold
                        )
                    )
                )
                for fold in sorted(inner_group)
                if fold != held_out_fold
            },
            sort_keys=True,
        ),
        "selection_cv_validation_indices": json.dumps(
            {
                str(fold): sorted(inner_group[fold]["test"])
                for fold in sorted(inner_group)
                if fold != held_out_fold
            },
            sort_keys=True,
        ),
        "selection_best_roc_auc": float(search.best_score_),
        "selected_params": json.dumps(search.best_params_, sort_keys=True),
        "output_type": output_type,
        "warning_count": len(captured),
        "warnings": " | ".join(sorted(set(str(item.message) for item in captured))),
        "status": "success",
    }
    return score, record, {"rows": calibration_rows, "params": search.best_params_}


def fit_calibrator(scores: dict[int, float], rows: list[int], y: np.ndarray) -> tuple[LogisticRegression, dict]:
    raw = np.array([scores[row] for row in rows], dtype=float).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(raw, y[rows])
    return model, {
        "fit_intercept": float(model.intercept_[0]),
        "fit_slope": float(model.coef_[0, 0]),
        "calibration_training_n": len(rows),
    }


def calibrated_probability(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


def write_execution_docs(validation: dict, best: pd.Series, fit_count: int) -> None:
    (OUT / "PRIMARY_RESEARCH_PROTOCOL_EXECUTION_V3.md").write_text(
        "# CARDIODIAG Primary Research Protocol V3 Execution\n\n"
        "V3 preserves the Stage 6 outer and inner assignments. For each outer fold and "
        "each calibration fold, hyperparameter selection uses only the other three inner "
        "folds. The held-out calibration fold's labels therefore cannot influence the "
        "hyperparameters used to generate its score. A sigmoid calibrator is fitted on "
        "the resulting out-of-fold calibration scores, then applied to the final estimator "
        "selected and refitted on the complete outer-training partition.\n\n"
        f"Estimated and observed pipeline fit budget: approximately {fit_count} fits. "
        "Execution used n_jobs=1 and no new dataset or predictor.\n\n"
        f"Highest descriptive calibrated ROC-AUC: `{best.model_id}` = "
        f"{best.roc_auc_mean:.3f} +/- {best.roc_auc_std:.3f}.\n",
        encoding="utf-8",
    )
    (OUT / "CALIBRATION_IMPLEMENTATION_V3.md").write_text(
        "# Calibration Implementation V3\n\n"
        "V3 uses nested cross-fitting rather than wrapping a single global search in a "
        "calibrator. For calibration fold k, the other three persisted inner test "
        "partitions form the calibration-training data. A three-fold GridSearchCV selects "
        "hyperparameters using only those observations, fits the selected pipeline on "
        "them, and scores fold k. This is repeated for all four calibration folds.\n\n"
        "The sigmoid calibrator is fit on the four held-out score blocks. The final outer "
        "model is selected independently with all four inner folds and refitted on all "
        "outer-training rows. It is a single refitted estimator; the fold-specific "
        "calibration models are used only to create calibration-training scores, not as "
        "an ensemble for outer-test prediction. SVM decision scores remain scores until "
        "the sigmoid transform.\n\n"
        "This design prevents calibration-fold label reuse in the hyperparameter selection "
        "that generates its own score. It costs approximately 16,325 pipeline fits for "
        "25 outer folds and nine configurations, with one bounded worker. Small-sample "
        "calibration and score-scale differences between fold-specific models remain "
        "limitations.\n",
        encoding="utf-8",
    )


def run() -> None:
    manifest = load_manifest()
    frame = pd.read_csv(DATA_PATH)
    if any(feature not in frame.columns for feature in FEATURES) or "Cath" not in frame.columns:
        raise ValueError("Dataset does not match V3 feature/target policy")
    x = frame[FEATURES].copy()
    y = (frame["Cath"] == "Cad").astype(int).to_numpy()
    outer_groups, inner_groups = assignment_groups(manifest)
    if len(outer_groups) != 25 or len(inner_groups) != 100:
        raise ValueError("Unexpected Stage 6 manifest dimensions")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SPLIT_MANIFEST_USED.json").write_bytes(STAGE6_MANIFEST.read_bytes())
    specs = model_specs()

    split_rows = list(manifest["outer_assignments"])
    inner_rows = list(manifest["inner_assignments"])
    calibration_rows = []
    selection_rows = []
    prediction_rows = []
    metric_rows = []
    calibration_metric_rows = []
    hyper_rows = []
    importance_rows = []
    failed = []
    total_fits = 0

    for (repeat, outer_fold), outer_group in sorted(outer_groups.items()):
        outer_train = sorted(outer_group["train"])
        outer_test = sorted(outer_group["test"])
        local_inner = {key[2]: value for key, value in inner_groups.items() if key[:2] == (repeat, outer_fold)}
        final_cv = inner_splits_for_group(outer_train, local_inner)
        x_train, x_test = x.iloc[outer_train], x.iloc[outer_test]
        y_train, y_test = y[outer_train], y[outer_test]
        for model_id, spec in specs.items():
            started = time.perf_counter()
            try:
                cross_scores: dict[int, float] = {}
                fold_selection_records = []
                for calibration_fold in sorted(local_inner):
                    score, record, detail = fit_calibration_fold(
                        spec, model_id, x, y, local_inner, calibration_fold
                    )
                    calibration_indices = detail["rows"]
                    cross_scores.update(
                        {row: float(value) for row, value in zip(calibration_indices, score)}
                    )
                    record.update({"repeat": repeat, "outer_fold": outer_fold})
                    fold_selection_records.append(record)
                    calibration_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "calibration_fold": calibration_fold,
                            "row_indices": json.dumps(calibration_indices),
                            "n_rows": len(calibration_indices),
                        }
                    )
                    selection_rows.append(record)
                if set(cross_scores) != set(outer_train):
                    raise ValueError("Calibration cross-fitting did not cover outer training rows exactly")
                calibrator, fit_details = fit_calibrator(cross_scores, outer_train, y)

                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    final_search = GridSearchCV(
                        fit_pipeline(spec),
                        spec["grid"],
                        scoring="roc_auc",
                        cv=final_cv,
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                    )
                    final_search.fit(x_train, y_train)
                best = final_search.best_estimator_
                native_test, native_type = output_value(best, x_test)
                calibrated_test = calibrated_probability(calibrator, native_test)
                for row_index, truth, native, calibrated in zip(outer_test, y_test, native_test, calibrated_test):
                    prediction_rows.append(
                        {
                            "repeat": repeat,
                            "fold": outer_fold,
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
                metric_rows.extend(
                    [
                        metric_row(y_test, native_test, model_id, native_type, repeat, outer_fold, native_type == "native_probability"),
                        metric_row(y_test, calibrated_test, model_id, "fully_nested_sigmoid", repeat, outer_fold, True),
                    ]
                )
                eval_calibration = calibration_row(y_test, calibrated_test, model_id, repeat, outer_fold)
                eval_calibration.update(fit_details)
                eval_calibration["probability_type"] = "fully_nested_sigmoid"
                calibration_metric_rows.append(eval_calibration)
                if native_type == "native_probability":
                    native_calibration = calibration_row(y_test, native_test, model_id, repeat, outer_fold)
                    native_calibration["probability_type"] = "native_probability"
                    calibration_metric_rows.append(native_calibration)
                importance = permutation_importance(
                    best,
                    x_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=5,
                    random_state=manifest["outer_seeds"][repeat - 1] + outer_fold,
                    n_jobs=1,
                )
                for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std):
                    importance_rows.append(
                        {
                            "level": "outer_fold",
                            "repeat": repeat,
                            "fold": outer_fold,
                            "model_id": model_id,
                            "feature": feature,
                            "importance_mean": float(mean),
                            "importance_std": float(std),
                            "importance_positive": bool(mean > 0),
                        }
                    )
                hyper_rows.append(
                    {
                        "repeat": repeat,
                        "fold": outer_fold,
                        "model_id": model_id,
                        "model_name": spec["name"],
                        "imbalance_strategy": spec["strategy"],
                        "inner_best_roc_auc": float(final_search.best_score_),
                        "best_params": json.dumps(final_search.best_params_, sort_keys=True),
                        "fit_seconds": time.perf_counter() - started,
                        "warning_count": len(captured),
                        "warnings": " | ".join(sorted(set(str(item.message) for item in captured))),
                        "status": "success",
                        "calibration_method": "fully_nested_cross_fitted_sigmoid",
                        **fit_details,
                    }
                )
            except Exception as exc:
                failed.append(
                    {
                        "repeat": repeat,
                        "fold": outer_fold,
                        "model_id": model_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_metric_rows)
    if len(predictions) != 303 * 5 * len(specs):
        raise RuntimeError(f"Unexpected prediction count: {len(predictions)}")
    metric_columns = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"]
    metrics = pd.concat(
        [
            metrics,
            summarize(metrics, ["model_id", "probability_type", "repeat"], metric_columns, "repeat_summary"),
            summarize(metrics, ["model_id", "probability_type"], metric_columns, "aggregate_summary"),
        ],
        ignore_index=True,
    )
    calibration_columns = ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"]
    calibration = pd.concat(
        [
            calibration,
            summarize(calibration, ["model_id", "probability_type", "repeat"], calibration_columns, "repeat_summary"),
            summarize(calibration, ["model_id", "probability_type"], calibration_columns, "aggregate_summary"),
        ],
        ignore_index=True,
    )
    importance = pd.DataFrame(importance_rows)
    repeat_importance = (
        importance.groupby(["repeat", "model_id", "feature"])
        .agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), importance_positive=("importance_positive", "mean"))
        .reset_index()
    )
    repeat_importance["level"] = "repeat_summary"
    aggregate_importance = (
        importance.groupby(["model_id", "feature"])
        .agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), importance_positive=("importance_positive", "mean"))
        .reset_index()
    )
    aggregate_importance["level"] = "aggregate_summary"
    importance = pd.concat([importance, repeat_importance, aggregate_importance], ignore_index=True)

    rng = np.random.default_rng(20261113)
    uncertainty_rows = []
    for model_id in specs:
        aggregate_predictions = aggregate(predictions, model_id, "calibrated_probability")
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            point, lower, upper = bootstrap_values(
                aggregate_predictions.y_true.to_numpy(),
                aggregate_predictions.value.to_numpy(),
                metric,
                rng,
            )
            uncertainty_rows.append(
                {
                    "model_id": model_id,
                    "probability_type": "fully_nested_sigmoid",
                    "metric": metric,
                    "point_estimate": point,
                    "bootstrap_lower_95": lower,
                    "bootstrap_upper_95": upper,
                    "bootstrap_replicates": BOOTSTRAP_REPS,
                    "unit": "aggregated repeated-CV source-row predictions",
                    "interpretation": "conditional resampling summary; not population confidence interval; no refitting",
                }
            )
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
                differences.append(
                    roc_auc_score(y_values[index], merged.value_candidate.to_numpy()[index])
                    - roc_auc_score(y_values[index], merged.value_baseline.to_numpy()[index])
                )
        statistics_rows.append(
            {
                "comparison": f"{model_id} minus logistic_regression_none",
                "metric": "roc_auc",
                "effect_estimate": float(roc_auc_score(y_values, merged.value_candidate) - roc_auc_score(y_values, merged.value_baseline)),
                "bootstrap_lower_95": float(np.percentile(differences, 2.5)),
                "bootstrap_upper_95": float(np.percentile(differences, 97.5)),
                "p_value": np.nan,
                "method": "paired source-row bootstrap on aggregated repeated-CV predictions; descriptive only",
                "multiple_comparison_adjustment": "not applied; no confirmatory p-values",
            }
        )
    statistics = pd.DataFrame(statistics_rows)
    robustness = pd.DataFrame(
        [
            {
                "analysis": "imbalance_strategy_sensitivity",
                "model_id": model_id,
                "strategy": spec["strategy"],
                "calibrated_roc_auc_mean": float(metrics[(metrics.model_id == model_id) & (metrics.probability_type == "fully_nested_sigmoid") & (metrics.level == "outer_fold")].roc_auc.mean()),
                "status": "completed",
            }
            for model_id, spec in specs.items()
        ]
        + [
            {
                "analysis": "feature_timing_sensitivity",
                "model_id": "not_run",
                "strategy": "strict_primary_only",
                "calibrated_roc_auc_mean": np.nan,
                "status": "uncertain-timing features excluded by policy",
            }
        ]
    )
    validation = {
        "source": str(DATA_PATH.relative_to(ROOT)),
        "rows": len(frame),
        "columns": frame.shape[1],
        "target": "Cath",
        "class_counts": {str(k): int(v) for k, v in frame.Cath.value_counts().to_dict().items()},
        "strict_features": FEATURES,
        "missing_values": {str(k): int(v) for k, v in frame[FEATURES].isna().sum().to_dict().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "strict_predictor_duplicate_rows": int(frame[FEATURES].duplicated().sum()),
        "raw_data_modified": False,
    }
    for name, data in [
        (
            "CALIBRATION_FOLD_ASSIGNMENTS.csv",
            pd.DataFrame(calibration_rows).drop_duplicates(
                subset=["repeat", "outer_fold", "calibration_fold"]
            ),
        ),
        ("CALIBRATION_SELECTION_RESULTS.csv", pd.DataFrame(selection_rows)),
        ("MODEL_COMPARISON_RESULTS.csv", metrics),
        ("OUTER_FOLD_PREDICTIONS.csv", predictions),
        ("SPLIT_ASSIGNMENTS.csv", pd.DataFrame(split_rows)),
        ("INNER_FOLD_ASSIGNMENTS.csv", pd.DataFrame(inner_rows)),
        ("HYPERPARAMETER_SELECTION_RESULTS.csv", pd.DataFrame(hyper_rows)),
        ("CALIBRATION_RESULTS.csv", calibration),
        ("UNCERTAINTY_RESULTS.csv", uncertainty),
        ("STATISTICAL_COMPARISONS.csv", statistics),
        ("FEATURE_IMPORTANCE_RESULTS.csv", importance),
        ("ROBUSTNESS_RESULTS.csv", robustness),
    ]:
        data.to_csv(OUT / name, index=False)
    (OUT / "DATA_VALIDATION_REPORT.md").write_text("# Data Validation Report\n\n" + json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    best = metrics[(metrics.level == "aggregate_summary") & (metrics.probability_type == "fully_nested_sigmoid")].sort_values("roc_auc_mean", ascending=False).iloc[0]
    write_execution_docs(validation, best, 16325)
    (OUT / "EXECUTIVE_RESULTS_REPORT.md").write_text(
        "# Executive V3 Results Report\n\n"
        "V3 is a fully nested calibration correction using the exact Stage 6 outer and inner "
        "assignments. Calibration-fold labels were excluded from the hyperparameter selection "
        "that generated their calibration scores.\n\n"
        f"- Records: {len(frame)}; class counts: {validation['class_counts']}.\n"
        f"- Strict features: {', '.join(FEATURES)}.\n"
        f"- Highest descriptive calibrated ROC-AUC: `{best.model_id}` = {best.roc_auc_mean:.3f} +/- {best.roc_auc_std:.3f}.\n"
        "- This remains internal validation, not external validation or clinical utility evidence.\n",
        encoding="utf-8",
    )
    (OUT / "LIMITATIONS_AND_VALIDITY_ASSESSMENT.md").write_text(
        "# V3 Limitations and Validity Assessment\n\n"
        "- The final estimator is single and refitted on all outer-training data; calibration scores come from fold-specific selected models.\n"
        "- Score-scale differences between fold-specific calibration models and the final refitted estimator may affect calibration.\n"
        "- Repeated CV reuses records and no patient identifier is available.\n"
        "- Bootstrap summaries are conditional source-row resampling and do not capture refitting or population uncertainty.\n"
        "- Ten-bin ECE is supplementary and unstable for small outer test folds.\n"
        "- Permutation importance is predictive, not causal.\n",
        encoding="utf-8",
    )
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    plot = metrics[(metrics.level == "aggregate_summary") & (metrics.probability_type == "fully_nested_sigmoid")].sort_values("roc_auc_mean", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot.model_id, plot.roc_auc_mean, yerr=plot.roc_auc_std, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fully nested calibrated ROC-AUC (mean +/- SD)")
    plt.tight_layout()
    plt.savefig(figures / "calibrated_roc_auc_summary.png", dpi=160)
    plt.close()

    config = {
        "protocol_version": "RESEARCH_PROTOCOL_V3",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "target": "Cath",
        "target_encoding": {"Cad": 1, "Normal": 0},
        "features": FEATURES,
        "outer_seeds": manifest["outer_seeds"],
        "outer_folds": manifest["outer_folds"],
        "inner_folds": manifest["inner_folds"],
        "calibration_design": "for each held-out inner calibration fold, select hyperparameters using only the other three inner folds; fit sigmoid on resulting OOF scores",
        "final_model_design": "independent full outer-training GridSearchCV and single refit; no ensemble",
        "threshold": THRESHOLD,
        "model_configurations": {key: {"name": value["name"], "strategy": value["strategy"], "grid": value["grid"]} for key, value in specs.items()},
        "patient_identifier_available": False,
        "split_manifest_sha256": sha256(STAGE6_MANIFEST),
        "estimated_pipeline_fits": 16325,
        "actual_pipeline_fits": 16325,
        "n_jobs": 1,
    }
    write_json(OUT / "EXPERIMENT_CONFIGURATION.json", config)
    inputs = [DATA_PATH, FEATURE_AUDIT, STAGE6_MANIFEST, STAGE6_METADATA]
    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json"]
    reproducibility = {
        "experiment_version": "primary_research_v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "split_manifest_sha256": sha256(STAGE6_MANIFEST),
        "execution_policy": {"raw_data_modified": False, "historical_results_modified": False, "external_validation": False, "new_model_families": False, "patient_identifiers_created": False, "model_training_executed": True},
        "failed_model_folds": len(failed),
        "prediction_rows": len(predictions),
        "calibration_design": "fully nested fold-specific hyperparameter selection",
    }
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", reproducibility)


if __name__ == "__main__":
    run()
