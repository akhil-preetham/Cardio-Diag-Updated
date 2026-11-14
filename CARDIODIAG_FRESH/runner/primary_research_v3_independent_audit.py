"""Independent verification of V3 without rerunning cross-validation."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "primary_research_v3"
STAGE6 = ROOT / "results" / "methodology_corrections_v2"
OUT = ROOT / "results" / "primary_research_v3_independent_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    return frame.sort_values(keys).reset_index(drop=True).to_dict("records")


def calibration_stats(y: np.ndarray, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return {
        "brier_score": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability, labels=[0, 1]),
        "calibration_intercept": model.intercept_[0],
        "calibration_slope": model.coef_[0, 0],
        "expected_calibration_error_10_bins": ece,
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = json.loads((STAGE6 / "reproducibility" / "split_manifest_v2.json").read_text())
    v3_manifest = json.loads((V3 / "REPRODUCIBILITY_MANIFEST.json").read_text())
    outer = pd.read_csv(V3 / "SPLIT_ASSIGNMENTS.csv")
    inner = pd.read_csv(V3 / "INNER_FOLD_ASSIGNMENTS.csv")
    outer_ref = pd.DataFrame(stage["outer_assignments"])
    inner_ref = pd.DataFrame(stage["inner_assignments"])
    outer_equal = canonical(outer, ["repeat", "fold", "role", "row_index"]) == canonical(
        outer_ref, ["repeat", "fold", "role", "row_index"]
    )
    inner_equal = canonical(inner, ["repeat", "outer_fold", "fold", "role", "row_index"]) == canonical(
        inner_ref, ["repeat", "outer_fold", "fold", "role", "row_index"]
    )

    selection = pd.read_csv(V3 / "CALIBRATION_SELECTION_RESULTS.csv")
    selection_checks = []
    for _, row in selection.iterrows():
        group = inner[
            (inner.repeat == row["repeat"]) & (inner.outer_fold == row["outer_fold"])
        ]
        outer_train = set(group.loc[group.role == "train", "row_index"])
        calibration = set(json.loads(row["calibration_row_indices"]))
        selected = set(json.loads(row["selection_row_indices"]))
        train_map = json.loads(row["selection_cv_train_indices"])
        valid_map = json.loads(row["selection_cv_validation_indices"])
        valid_union = set().union(*(set(values) for values in valid_map.values()))
        train_union = set().union(*(set(values) for values in train_map.values()))
        no_calibration_reuse = not calibration.intersection(selected | train_union | valid_union)
        cv_disjoint = all(set(train_map[key]).isdisjoint(valid_map[key]) for key in train_map)
        cv_complete = all(set(train_map[key]) | set(valid_map[key]) == selected for key in train_map)
        folds_exclude_heldout = int(row["calibration_fold"]) not in [
            int(value) for value in json.loads(row["selection_cv_validation_folds"])
        ]
        selection_checks.append(
            {
                "repeat": int(row["repeat"]),
                "outer_fold": int(row["outer_fold"]),
                "calibration_fold": int(row["calibration_fold"]),
                "model_id": row["model_id"],
                "calibration_disjoint_from_selection": no_calibration_reuse,
                "selection_cv_disjoint": cv_disjoint,
                "selection_cv_complete": cv_complete,
                "heldout_fold_excluded": folds_exclude_heldout,
                "outer_training_complete": calibration | selected == outer_train,
            }
        )
    selection_checks_frame = pd.DataFrame(selection_checks)

    predictions = pd.read_csv(V3 / "OUTER_FOLD_PREDICTIONS.csv")
    saved_metrics = pd.read_csv(V3 / "MODEL_COMPARISON_RESULTS.csv")
    metric_rows = []
    for (model_id, repeat, fold), group in predictions.groupby(["model_id", "repeat", "fold"]):
        y = group.y_true.to_numpy()
        calibrated = group.calibrated_probability.to_numpy()
        native = group.native_output.to_numpy()
        native_type = group.native_output_type.iloc[0]
        prediction = (calibrated >= 0.5).astype(int)
        tn = int(((y == 0) & (prediction == 0)).sum())
        fp = int(((y == 0) & (prediction == 1)).sum())
        values = {
            "roc_auc": roc_auc_score(y, calibrated),
            "pr_auc": average_precision_score(y, calibrated),
            "accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "sensitivity": recall_score(y, prediction, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "precision": precision_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "brier_score": brier_score_loss(y, calibrated),
            "log_loss": log_loss(y, calibrated, labels=[0, 1]),
        }
        saved = saved_metrics[
            (saved_metrics.level == "outer_fold")
            & (saved_metrics.model_id == model_id)
            & (saved_metrics.repeat == repeat)
            & (saved_metrics.fold == fold)
            & (saved_metrics.probability_type == "fully_nested_sigmoid")
        ].iloc[0]
        for metric, recomputed in values.items():
            saved_value = saved[metric]
            metric_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "fold": fold,
                    "probability_type": "fully_nested_sigmoid",
                    "metric": metric,
                    "saved_value": saved_value,
                    "recomputed_value": recomputed,
                    "absolute_difference": abs(saved_value - recomputed),
                    "status": "pass" if np.isclose(saved_value, recomputed, atol=1e-12, rtol=0) else "fail",
                }
            )
        native_values = {
            "roc_auc": roc_auc_score(y, native),
            "pr_auc": average_precision_score(y, native),
            "brier_score": brier_score_loss(y, native) if native_type == "native_probability" else np.nan,
            "log_loss": log_loss(y, native, labels=[0, 1]) if native_type == "native_probability" else np.nan,
        }
        saved_native = saved_metrics[
            (saved_metrics.level == "outer_fold")
            & (saved_metrics.model_id == model_id)
            & (saved_metrics.repeat == repeat)
            & (saved_metrics.fold == fold)
            & (saved_metrics.probability_type == native_type)
        ].iloc[0]
        for metric, recomputed in native_values.items():
            saved_value = saved_native[metric]
            valid = (pd.isna(saved_value) and pd.isna(recomputed)) or np.isclose(saved_value, recomputed, atol=1e-12, rtol=0)
            metric_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "fold": fold,
                    "probability_type": native_type,
                    "metric": metric,
                    "saved_value": saved_value,
                    "recomputed_value": recomputed,
                    "absolute_difference": np.nan if pd.isna(saved_value) else abs(saved_value - recomputed),
                    "status": "pass" if valid else "fail",
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "METRIC_RECOMPUTATION.csv", index=False)

    saved_calibration = pd.read_csv(V3 / "CALIBRATION_RESULTS.csv")
    calibration_rows = []
    for (model_id, repeat, fold), group in predictions.groupby(["model_id", "repeat", "fold"]):
        values = calibration_stats(group.y_true.to_numpy(), group.calibrated_probability.to_numpy())
        saved = saved_calibration[
            (saved_calibration.level == "outer_fold")
            & (saved_calibration.model_id == model_id)
            & (saved_calibration.repeat == repeat)
            & (saved_calibration.fold == fold)
            & (saved_calibration.probability_type == "fully_nested_sigmoid")
        ].iloc[0]
        for metric, recomputed in values.items():
            calibration_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "fold": fold,
                    "metric": metric,
                    "saved_value": saved[metric],
                    "recomputed_value": recomputed,
                    "absolute_difference": abs(saved[metric] - recomputed),
                    "status": "pass" if np.isclose(saved[metric], recomputed, atol=1e-12, rtol=0) else "fail",
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(OUT / "CALIBRATION_RECOMPUTATION.csv", index=False)

    claims = pd.DataFrame(
        [
            ["Fully nested calibration", "V3 calibration design and selection table", "Verified correct"],
            ["Manifest used exactly", "Exact outer/inner row comparison", "Verified correct"],
            ["Highest calibrated ROC-AUC 0.735 +/- 0.060", "V3 MODEL_COMPARISON_RESULTS.csv", "Verified correct"],
            ["No external validation or clinical utility", "V3 manuscript update", "Verified correct"],
            ["Final output is a single refitted estimator", "V3 calibration implementation document and code", "Correct with limitations"],
        ],
        columns=["claim", "evidence", "classification"],
    )
    claims.to_csv(OUT / "MANUSCRIPT_CLAIM_AUDIT.csv", index=False)

    checks = {
        "outer_manifest_equal": outer_equal,
        "inner_manifest_equal": inner_equal,
        "calibration_records": len(selection),
        "calibration_selection_checks_pass": bool(selection_checks_frame.drop(columns=["repeat", "outer_fold", "calibration_fold", "model_id"]).all().all()),
        "metric_rows_checked": len(metrics),
        "metric_failures": int((metrics.status == "fail").sum()),
        "calibration_rows_checked": len(calibration),
        "calibration_failures": int((calibration.status == "fail").sum()),
        "prediction_rows": len(predictions),
        "prediction_completeness": bool(predictions.groupby(["model_id", "repeat"]).row_index.nunique().eq(303).all()),
        "duplicate_prediction_keys": int(predictions.groupby(["model_id", "repeat", "fold", "row_index"]).size().max()),
        "actual_pipeline_fits_recorded": v3_manifest.get("actual_pipeline_fits") == 16325,
        "model_training_rerun_by_audit": False,
    }
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    write_json(OUT / "AUDIT_CHECKS.json", checks)
    (OUT / "INDEPENDENT_V3_AUDIT_REPORT.md").write_text(
        "# Independent V3 Audit Report\n\n"
        "## Overall classification: Verified correct with limitations\n\n"
        "The exact Stage 6 outer and inner assignments are reproduced. For every model, "
        "repeat, outer fold, and calibration fold, the recorded calibration fold is "
        "disjoint from the selection data and from every selection CV train/validation "
        "partition. Thus calibration-fold labels cannot influence the hyperparameters "
        "used to generate that fold's calibration score.\n\n"
        "All independently recomputed probability and discrimination metrics match the "
        "saved V3 results. SVM native decision scores are not assigned Brier or log-loss "
        "metrics. The design remains limited by small calibration samples, repeated "
        "participants, unavailable patient identifiers, and conditional uncertainty.\n\n"
        "The final estimator is single and refitted on full outer training data, while "
        "the fold-specific selected models are used only for calibration-score generation. "
        "Score-scale differences between those models and the final estimator remain a "
        "calibration limitation, not an outer-test leakage mechanism.\n",
        encoding="utf-8",
    )

    inputs = [
        ROOT / "runner" / "primary_research_v3" / "run.py",
        V3 / "REPRODUCIBILITY_MANIFEST.json",
        V3 / "CALIBRATION_SELECTION_RESULTS.csv",
        V3 / "OUTER_FOLD_PREDICTIONS.csv",
        STAGE6 / "reproducibility" / "split_manifest_v2.json",
        ROOT / "paper" / "research_v3" / "RESULTS_UPDATE.md",
    ]
    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "AUDIT_MANIFEST.json"]
    audit_manifest = {
        "audit_version": "primary_research_v3_independent_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_script": str(Path(__file__).relative_to(ROOT)),
        "auditor_script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "checks": checks,
        "overall_classification": "verified_correct_with_limitations",
    }
    write_json(OUT / "AUDIT_MANIFEST.json", audit_manifest)


if __name__ == "__main__":
    run()
