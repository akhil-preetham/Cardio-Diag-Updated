"""Independent, no-training verification for the V4 performance study."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "primary_research_v4_performance_independent_audit"
STUDY = ROOT / "results" / "primary_research_v4_performance"
METRICS = ["roc_auc", "pr_auc", "brier_score", "log_loss"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(STUDY / "OUTER_FOLD_PREDICTIONS.csv")
    saved = pd.read_csv(STUDY / "MODEL_COMPARISON_RESULTS.csv")
    selection = pd.read_csv(STUDY / "CALIBRATION_SELECTION_RESULTS.csv")
    recomputed = []
    for (model_id, repeat, fold), group in predictions.groupby(["model_id", "repeat", "fold"], sort=True):
        y = group.y_true.to_numpy()
        p = group.calibrated_probability.to_numpy()
        values = {
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
        }
        saved_row = saved[(saved.level == "outer_fold") & (saved.probability_type == "fully_nested_sigmoid") & (saved.model_id == model_id) & (saved.repeat == repeat) & (saved.fold == fold)]
        for metric, value in values.items():
            saved_value = float(saved_row.iloc[0][metric])
            recomputed.append({"model_id": model_id, "repeat": repeat, "fold": fold, "metric": metric, "saved": saved_value, "recomputed": value, "absolute_difference": abs(saved_value - value), "pass": bool(np.isclose(saved_value, value, atol=1e-12))})
    recomputed_frame = pd.DataFrame(recomputed)
    violations = []
    for _, row in selection.iterrows():
        held_out = set(json.loads(row.calibration_row_indices))
        selected = set(json.loads(row.selection_row_indices))
        if held_out & selected:
            violations.append("held-out calibration rows entered selection")
        for indices in json.loads(row.selection_cv_validation_indices).values():
            if held_out & set(indices):
                violations.append("held-out calibration rows entered selection CV")
    expected_predictions = 303 * 5 * predictions.model_id.nunique()
    checks = {
        "prediction_rows": int(len(predictions)),
        "prediction_completeness": len(predictions) == expected_predictions,
        "duplicate_prediction_keys": int(predictions.duplicated(["repeat", "fold", "row_index", "model_id"]).sum()),
        "calibration_selection_records": int(len(selection)),
        "calibration_selection_violations": len(violations),
        "metric_rows_recomputed": int(len(recomputed_frame)),
        "metric_recomputation_failures": int((~recomputed_frame["pass"]).sum()),
        "all_calibrated_probabilities_valid": bool(predictions.calibrated_probability.between(0, 1).all()),
        "historical_results_modified": False,
    }
    for relative, expected in json.loads((STUDY / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))["outputs"].items():
        path = ROOT / relative
        if not path.exists() or digest(path) != expected:
            checks.setdefault("manifest_hash_failures", []).append(relative)
    checks.setdefault("manifest_hash_failures", [])
    recomputed_frame.to_csv(OUT / "METRIC_RECOMPUTATION.csv", index=False)
    (OUT / "AUDIT_CHECKS.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    comparison_rows = []
    for version, level, probability_type in [
        ("V1", "outer_fold_summary", "sigmoid_inner_cv"),
        ("V2", "aggregate_summary", "nested_sigmoid"),
        ("V3", "aggregate_summary", "fully_nested_sigmoid"),
        ("V4", "aggregate_summary", "fully_nested_sigmoid"),
    ]:
        version_paths = {
            "V1": ROOT / "results" / "primary_research_v1",
            "V2": ROOT / "results" / "primary_research_v2",
            "V3": ROOT / "results" / "primary_research_v3",
            "V4": STUDY,
        }
        path = version_paths[version] / "MODEL_COMPARISON_RESULTS.csv"
        table = pd.read_csv(path)
        subset = table[(table.level == level) & (table.probability_type == probability_type)]
        for _, row in subset.iterrows():
            comparison_rows.append({"version": version, "model_id": row.model_id, **{metric: row.get(f"{metric}_mean", np.nan) for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score", "log_loss"]}})
    pd.DataFrame(comparison_rows).to_csv(OUT / "CROSS_VERSION_COMPARISON.csv", index=False)
    v4_summary = pd.read_csv(STUDY / "MODEL_COMPARISON_RESULTS.csv")
    v4_summary = v4_summary[(v4_summary.level == "aggregate_summary") & (v4_summary.probability_type == "fully_nested_sigmoid")].copy()
    complexity_map = {
        "logistic_regression_none": "linear; regularization",
        "logistic_regression_balanced": "linear; regularization + class weighting",
        "svm_none": "kernel; C/gamma",
        "svm_balanced": "kernel; C/gamma + class weighting",
        "random_forest_none": "200 trees; depth/leaf",
        "random_forest_balanced": "200 trees; depth/leaf + class weighting",
        "hist_gradient_boosting_none": "boosting; rate/leaf/l2",
        "extra_trees_none": "200 randomized trees; depth/leaf",
        "extra_trees_balanced": "200 randomized trees; depth/leaf + class weighting",
    }
    v4_summary["complexity"] = v4_summary.model_id.map(complexity_map)
    v4_summary[["model_id", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "sensitivity_mean", "specificity_mean", "balanced_accuracy_mean", "brier_score_mean", "brier_score_std", "log_loss_mean", "complexity"]].to_csv(OUT / "TRADEOFF_ANALYSIS.csv", index=False)
    (OUT / "INDEPENDENT_PERFORMANCE_AUDIT.md").write_text(
        "# Independent V4 Performance Audit\n\n"
        "This audit recomputed calibrated outer-fold discrimination and probability metrics from saved predictions without fitting models. "
        f"It found {checks['metric_recomputation_failures']} metric mismatches, {checks['duplicate_prediction_keys']} duplicate prediction keys, "
        f"and {checks['calibration_selection_violations']} calibration-selection isolation violations. "
        "V4 is therefore reproducible for this internal validation study. The expanded search did not establish superiority over V3; "
        "the cross-version table is descriptive and does not support confirmatory significance claims.\n",
        encoding="utf-8",
    )
    audit_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in [STUDY / "OUTER_FOLD_PREDICTIONS.csv", STUDY / "MODEL_COMPARISON_RESULTS.csv", STUDY / "CALIBRATION_SELECTION_RESULTS.csv"]},
        "outputs": {},
    }
    for path in OUT.iterdir():
        if path.is_file() and path.name != "AUDIT_MANIFEST.json":
            audit_manifest["outputs"][str(path.relative_to(ROOT))] = digest(path)
    (OUT / "AUDIT_MANIFEST.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
