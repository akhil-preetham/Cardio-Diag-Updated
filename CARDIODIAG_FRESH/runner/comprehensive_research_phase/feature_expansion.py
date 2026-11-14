"""Exploratory, fully nested feature-group sensitivity analysis."""
from __future__ import annotations

import itertools
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from runner.primary_research_v2.run import (
    FEATURES as STRICT_FEATURES,
    ROOT,
    THRESHOLD,
    calibration_row,
    inner_splits_for_group,
    metric_row,
    sha256,
    summarize,
    write_json,
)
from runner.primary_research_v3.run import (
    assignment_groups,
    calibrated_probability,
    calibration_selection_splits,
    fit_calibrator,
    load_manifest,
)

OUT = ROOT / "results" / "comprehensive_research_phase" / "feature_expansion"
DATA_PATH = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
STAGE6_MANIFEST = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"]

GROUPS = {
    "strict_primary": STRICT_FEATURES,
    "demographic_anthropometric_exploratory": STRICT_FEATURES + ["Weight", "Length", "BMI"],
    "medical_history_exploratory": STRICT_FEATURES + ["DM", "HTN", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "DLP"],
    "symptoms_exploratory": STRICT_FEATURES + ["Typical Chest Pain", "Dyspnea", "Atypical", "Nonanginal"],
    "ecg_exploratory": STRICT_FEATURES + ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"],
    "laboratory_exploratory": STRICT_FEATURES + ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
}


def spec(features: list[str], balanced: bool) -> dict:
    return {
        "features": features,
        "strategy": "balanced" if balanced else "none",
        "estimator": LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced" if balanced else None, random_state=1),
        "grid": {"model__C": GRID},
    }


def pipeline(model_spec: dict) -> Pipeline:
    features = model_spec["features"]
    numeric = [feature for feature in features if feature != "Sex" and pd.api.types.is_numeric_dtype(model_spec["frame"][feature])]
    categorical = [feature for feature in features if feature not in numeric]
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model_spec["estimator"])])


def fit_calibration_fold(model_spec: dict, model_id: str, x: pd.DataFrame, y: np.ndarray, inner_group: dict, held_out_fold: int) -> tuple[dict[int, float], dict]:
    selection_rows, selection_cv = calibration_selection_splits(inner_group, held_out_fold)
    calibration_rows = sorted(inner_group[held_out_fold]["test"])
    search = GridSearchCV(pipeline(model_spec), model_spec["grid"], scoring="roc_auc", cv=selection_cv, n_jobs=1, refit=True, error_score="raise")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        search.fit(x.iloc[selection_rows], y[selection_rows])
    score = search.best_estimator_.predict_proba(x.iloc[calibration_rows])[:, 1]
    return dict(zip(calibration_rows, score)), {
        "model_id": model_id,
        "calibration_fold": held_out_fold,
        "calibration_row_indices": json.dumps(calibration_rows),
        "selection_row_indices": json.dumps(selection_rows),
        "selection_cv_validation_folds": json.dumps([fold for fold in sorted(inner_group) if fold != held_out_fold]),
        "selected_params": json.dumps(search.best_params_, sort_keys=True),
        "selection_best_roc_auc": float(search.best_score_),
        "warning_count": len(captured),
        "warnings": " | ".join(sorted(set(str(item.message) for item in captured))),
        "status": "success",
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    missing_groups = {name: sorted(set(features) - set(frame.columns)) for name, features in GROUPS.items()}
    if any(missing_groups.values()):
        raise ValueError(f"Missing feature columns: {missing_groups}")
    y = (frame["Cath"] == "Cad").astype(int).to_numpy()
    manifest = load_manifest()
    outer_groups, inner_groups = assignment_groups(manifest)
    selections, predictions, metrics, calibrations = [], [], [], []

    for (repeat, outer_fold), outer_group in sorted(outer_groups.items()):
        outer_train, outer_test = sorted(outer_group["train"]), sorted(outer_group["test"])
        local_inner = {key[2]: value for key, value in inner_groups.items() if key[:2] == (repeat, outer_fold)}
        final_cv = inner_splits_for_group(outer_train, local_inner)
        for group_id, features in GROUPS.items():
            for balanced in (False, True):
                model_id = f"{group_id}__{'balanced' if balanced else 'none'}"
                model_spec = spec(features, balanced)
                model_spec["frame"] = frame
                calibration_scores = {}
                for calibration_fold in sorted(local_inner):
                    fold_scores, record = fit_calibration_fold(model_spec, model_id, frame[features], y, local_inner, calibration_fold)
                    calibration_scores.update(fold_scores)
                    record.update({"repeat": repeat, "outer_fold": outer_fold, "feature_group": group_id, "features": "|".join(features)})
                    selections.append(record)
                calibrator, fit_details = fit_calibrator(calibration_scores, outer_train, y)
                final_search = GridSearchCV(pipeline(model_spec), model_spec["grid"], scoring="roc_auc", cv=final_cv, n_jobs=1, refit=True, error_score="raise")
                final_search.fit(frame.iloc[outer_train][features], y[outer_train])
                native = final_search.best_estimator_.predict_proba(frame.iloc[outer_test][features])[:, 1]
                calibrated = calibrated_probability(calibrator, native)
                predictions.extend({"repeat": repeat, "fold": outer_fold, "row_index": int(row), "source_row_id": f"source_row:{row}", "feature_group": group_id, "model_id": model_id, "imbalance_strategy": model_spec["strategy"], "y_true": int(truth), "probability": float(prob), "prediction": int(prob >= THRESHOLD)} for row, truth, prob in zip(outer_test, y[outer_test], calibrated))
                metrics.append(metric_row(y[outer_test], calibrated, model_id, "fully_nested_sigmoid", repeat, outer_fold, True) | {"feature_group": group_id})
                calibrations.append(calibration_row(y[outer_test], calibrated, model_id, repeat, outer_fold) | {"feature_group": group_id, "probability_type": "fully_nested_sigmoid", **fit_details})

    predictions_frame = pd.DataFrame(predictions)
    metrics_frame = pd.DataFrame(metrics)
    calibration_frame = pd.DataFrame(calibrations)
    metrics_frame = pd.concat([metrics_frame, summarize(metrics_frame, ["feature_group", "model_id"], METRICS, "aggregate_summary")], ignore_index=True)
    calibration_frame = pd.concat([calibration_frame, summarize(calibration_frame, ["feature_group", "model_id"], ["brier_score", "log_loss", "calibration_intercept", "calibration_slope", "expected_calibration_error_10_bins"], "aggregate_summary")], ignore_index=True)
    comparison = metrics_frame[(metrics_frame.level == "aggregate_summary")].copy()
    comparison["eligibility_status"] = comparison.feature_group.map(lambda value: "strict_primary" if value == "strict_primary" else "exploratory_pending_timing_unit_confirmation")
    comparison.to_csv(OUT / "FEATURE_GROUP_COMPARISON.csv", index=False)
    metrics_frame.to_csv(OUT / "MODEL_PERFORMANCE_RESULTS.csv", index=False)
    predictions_frame.to_csv(OUT / "OUTER_FOLD_PREDICTIONS.csv", index=False)
    calibration_frame.to_csv(OUT / "CALIBRATION_RESULTS.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "FEATURE_GROUP_SELECTION_RESULTS.csv", index=False)
    pd.DataFrame([{"feature_group": key, "features": "|".join(value), "status": "strict_primary" if key == "strict_primary" else "exploratory_only", "reason": "Units or prediction-time timing require confirmation before clinical eligibility." if key != "strict_primary" else "Prespecified V3 primary feature set."} for key, value in GROUPS.items()]).to_csv(OUT / "FEATURE_GROUP_ELIGIBILITY.csv", index=False)
    fit_count = 25 * len(GROUPS) * 2 * (16 * len(GRID) + 5)
    config = {"study_version": "comprehensive_research_phase_feature_expansion", "features_groups": GROUPS, "target": "Cath", "calibration": "fully nested V3 fold-specific selection and sigmoid calibration", "outer_folds": 5, "outer_repeats": 5, "inner_folds": 4, "model_family": "logistic_regression", "search_grid": GRID, "fit_count_estimate": fit_count, "n_jobs": 1, "split_manifest_sha256": sha256(STAGE6_MANIFEST), "additional_groups_finally_eligible": False}
    write_json(OUT / "EXPERIMENT_CONFIGURATION.json", config)
    manifest_output = {"generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)), "environment": {"python": sys.version, "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [DATA_PATH, STAGE6_MANIFEST]}, "outputs": {}, "fit_count_estimate": fit_count}
    for path in OUT.rglob("*"):
        if path.is_file() and path.name != "REPRODUCIBILITY_MANIFEST.json":
            manifest_output["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUT / "REPRODUCIBILITY_MANIFEST.json", manifest_output)


if __name__ == "__main__":
    run()
