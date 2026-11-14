"""Independent, no-CV-rerun audit of the CARDIODIAG V2 artifacts."""
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
V2 = ROOT / "results" / "primary_research_v2"
STAGE6 = ROOT / "results" / "methodology_corrections_v2"
V1 = ROOT / "results" / "primary_research_v1"
VERIFICATION_V1 = ROOT / "results" / "primary_experiment_verification_v1"
OUT = ROOT / "results" / "primary_research_v2_independent_audit"
FEATURES = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def canonical(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    return frame.sort_values(keys).reset_index(drop=True).to_dict("records")


def calibration_stats(y_true: np.ndarray, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y_true)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return {
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "calibration_intercept": model.intercept_[0],
        "calibration_slope": model.coef_[0, 0],
        "expected_calibration_error_10_bins": ece,
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage_manifest = json.loads((STAGE6 / "reproducibility" / "split_manifest_v2.json").read_text())
    v2_manifest = json.loads((V2 / "REPRODUCIBILITY_MANIFEST.json").read_text())
    outer_saved = pd.read_csv(V2 / "SPLIT_ASSIGNMENTS.csv")
    inner_saved = pd.read_csv(V2 / "INNER_FOLD_ASSIGNMENTS.csv")
    outer_reference = pd.DataFrame(stage_manifest["outer_assignments"])
    inner_reference = pd.DataFrame(stage_manifest["inner_assignments"])
    outer_equal = canonical(outer_saved, ["repeat", "fold", "role", "row_index"]) == canonical(
        outer_reference, ["repeat", "fold", "role", "row_index"]
    )
    inner_equal = canonical(inner_saved, ["repeat", "outer_fold", "fold", "role", "row_index"]) == canonical(
        inner_reference, ["repeat", "outer_fold", "fold", "role", "row_index"]
    )

    predictions = pd.read_csv(V2 / "OUTER_FOLD_PREDICTIONS.csv")
    saved_metrics = pd.read_csv(V2 / "MODEL_COMPARISON_RESULTS.csv")
    saved_calibration = pd.read_csv(V2 / "CALIBRATION_RESULTS.csv")
    metric_rows = []
    metric_names = [
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
    for (model_id, repeat, fold), group in predictions.groupby(["model_id", "repeat", "fold"]):
        y = group["y_true"].to_numpy()
        calibrated = group["calibrated_probability"].to_numpy()
        native = group["native_output"].to_numpy()
        native_type = group["native_output_type"].iloc[0]
        prediction = (calibrated >= 0.5).astype(int)
        tn = int(((y == 0) & (prediction == 0)).sum())
        fp = int(((y == 0) & (prediction == 1)).sum())
        fn = int(((y == 1) & (prediction == 0)).sum())
        tp = int(((y == 1) & (prediction == 1)).sum())
        calibrated_values = {
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
            & (saved_metrics.probability_type == "nested_sigmoid")
        ].iloc[0]
        for metric in metric_names:
            recomputed = float(calibrated_values[metric])
            saved_value = float(saved[metric])
            metric_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "fold": fold,
                    "probability_type": "nested_sigmoid",
                    "metric": metric,
                    "saved_value": saved_value,
                    "recomputed_value": recomputed,
                    "absolute_difference": abs(saved_value - recomputed),
                    "status": "pass" if np.isclose(saved_value, recomputed, atol=1e-12, rtol=0) else "fail",
                }
            )
        native_metrics = {
            "roc_auc": roc_auc_score(y, native),
            "pr_auc": average_precision_score(y, native),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "precision": np.nan,
            "f1": np.nan,
            "brier_score": brier_score_loss(y, native) if native_type == "native_probability" else np.nan,
            "log_loss": log_loss(y, native, labels=[0, 1]) if native_type == "native_probability" else np.nan,
        }
        native_saved = saved_metrics[
            (saved_metrics.level == "outer_fold")
            & (saved_metrics.model_id == model_id)
            & (saved_metrics.repeat == repeat)
            & (saved_metrics.fold == fold)
            & (saved_metrics.probability_type == native_type)
        ].iloc[0]
        for metric in ["roc_auc", "pr_auc", "brier_score", "log_loss"]:
            recomputed = native_metrics[metric]
            saved_value = native_saved[metric]
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
    metric_recomputation = pd.DataFrame(metric_rows)
    metric_recomputation.to_csv(OUT / "METRIC_RECOMPUTATION.csv", index=False)

    calibration_rows = []
    for (model_id, repeat, fold), group in predictions.groupby(["model_id", "repeat", "fold"]):
        y = group.y_true.to_numpy()
        values = group.calibrated_probability.to_numpy()
        recomputed = calibration_stats(y, values)
        saved = saved_calibration[
            (saved_calibration.level == "outer_fold")
            & (saved_calibration.model_id == model_id)
            & (saved_calibration.repeat == repeat)
            & (saved_calibration.fold == fold)
            & (saved_calibration.probability_type == "nested_sigmoid")
        ].iloc[0]
        for metric in recomputed:
            calibration_rows.append(
                {
                    "model_id": model_id,
                    "repeat": repeat,
                    "fold": fold,
                    "metric": metric,
                    "saved_value": float(saved[metric]),
                    "recomputed_value": float(recomputed[metric]),
                    "absolute_difference": abs(float(saved[metric]) - float(recomputed[metric])),
                    "status": "pass" if np.isclose(saved[metric], recomputed[metric], atol=1e-12, rtol=0) else "fail",
                }
            )
    calibration_recomputation = pd.DataFrame(calibration_rows)
    calibration_recomputation.to_csv(OUT / "CALIBRATION_RECOMPUTATION.csv", index=False)

    importance = pd.read_csv(V2 / "FEATURE_IMPORTANCE_RESULTS.csv")
    stability = (
        importance.groupby(["model_id", "feature"])
        .agg(
            outer_fold_mean=("importance_mean", "mean"),
            outer_fold_sd=("importance_mean", "std"),
            positive_fraction=("importance_positive", "mean"),
            n_outer_folds=("importance_mean", "size"),
        )
        .reset_index()
    )
    stability.to_csv(OUT / "FEATURE_IMPORTANCE_STABILITY.csv", index=False)

    code = (ROOT / "runner" / "primary_research_v2" / "run.py").read_text(encoding="utf-8")
    split_doc = (
        "# Split and Leakage Audit\n\n"
        "## Classification\n\n"
        f"- Outer manifest equality: **{'Verified correct' if outer_equal else 'Requires correction'}**.\n"
        f"- Inner manifest equality: **{'Verified correct' if inner_equal else 'Requires correction'}**.\n"
        "- Prediction coverage and duplicate-key checks are verified from the saved records.\n"
        "- The code partitions data with `x.iloc[outer_train]` and `x.iloc[outer_test]`; preprocessing is inside `Pipeline` and `GridSearchCV`.\n"
        "- Cross-fitted calibration fits each base pipeline on inner training rows and predicts its inner test rows. The calibrator is fitted on those outer-training scores and `y[outer_train]`, then applied to outer-test scores.\n"
        "- No outer-test labels are passed to model selection, preprocessing, calibration fitting, or threshold selection.\n\n"
        "## Limitation\n\n"
        "Hyperparameters are selected by `GridSearchCV` using all four inner folds before "
        "the same folds are used to generate calibration scores. Thus each calibration "
        "fold's base-model hyperparameters were selected with that fold's labels included "
        "in the selection process. This creates selection-induced optimism in the calibration "
        "training scores. It is not outer-test leakage, but it means calibration is not "
        "fully nested with respect to hyperparameter selection.\n"
    )
    (OUT / "SPLIT_AND_LEAKAGE_AUDIT.md").write_text(split_doc, encoding="utf-8")

    calibration_doc = (
        "# Calibration Implementation Audit\n\n"
        "## Classification: Requires correction\n\n"
        "The implementation correctly performs a separate calibration fit inside every "
        "outer fold, uses sigmoid calibration, excludes outer-test labels, and labels SVM "
        "outputs as `decision_score`. Probability clipping to `[1e-6, 1-1e-6]` is used "
        "only for calibration-slope/intercept evaluation.\n\n"
        "However, `GridSearchCV` first selects `best_params` over all four persisted inner "
        "folds. `cross_fitted_calibration` then refits that selected configuration on each "
        "inner training subset and predicts its inner calibration subset. Because the "
        "selected configuration was chosen using the held-out subset's labels, these are "
        "not fully independent calibration-training predictions. A corrected design should "
        "perform hyperparameter selection separately inside each calibration fold (using "
        "only the remaining inner-training data), or reserve a calibration layer before "
        "model selection. This audit does not rerun or alter V2.\n\n"
        "Calibration evaluation on outer-test predictions is appropriate as evaluation, "
        "but the approximately 60 observations per outer fold make ten-bin ECE unstable. "
        "Calibration and discrimination are distinct: ROC-AUC assesses ranking, whereas "
        "Brier score, log loss, intercept, slope, reliability, and ECE assess probability "
        "quality.\n"
    )
    (OUT / "CALIBRATION_IMPLEMENTATION_AUDIT.md").write_text(calibration_doc, encoding="utf-8")

    (OUT / "STATISTICAL_VALIDITY_AUDIT.md").write_text(
        "# Statistical Validity Audit\n\n"
        "## Classification: Correct with limitations\n\n"
        "The comparison table uses matched source rows and paired bootstrap differences. "
        "No confirmatory p-values or ordinary independent-fold tests are reported, and "
        "multiplicity is explicitly not applied. The comparison intervals are descriptive "
        "conditional resampling summaries, not population confidence intervals. Repeated "
        "cross-validation reuses participants and model selection/calibration adds further "
        "dependence. Comparisons should remain descriptive unless a prespecified dependent-CV "
        "method is justified in a future analysis.\n",
        encoding="utf-8",
    )
    (OUT / "UNCERTAINTY_VALIDITY_AUDIT.md").write_text(
        "# Uncertainty and Robustness Validity Audit\n\n"
        "## Classification: Correct with limitations\n\n"
        "The bootstrap resamples the 303 source rows after averaging each row's repeated-CV "
        "predictions. It does not refit preprocessing, hyperparameters, estimators, or "
        "calibrators, and therefore captures conditional row-resampling variability only. "
        "It is not a population confidence interval, algorithmic uncertainty estimate, or "
        "patient-level bootstrap because no patient identifier is available.\n\n"
        "The imbalance sensitivity output is descriptive and uses the same repeated-CV "
        "structure. The timing-feature sensitivity is explicitly not run. No individual "
        "predictive intervals are supported.\n",
        encoding="utf-8",
    )
    (OUT / "FEATURE_IMPORTANCE_AUDIT.md").write_text(
        "# Feature Importance Audit\n\n"
        "## Classification: Correct with limitations\n\n"
        "Permutation importance is computed by `permutation_importance(best, x_test, y_test, ...)` "
        "inside each outer fold, so held-out labels are used for scoring but not for fitting, "
        "feature selection, or hyperparameter selection. The results are predictive "
        "associations, not causal effects.\n\n"
        "The V2 artifact contains fold-level importance but does not itself provide a "
        "prespecified rank-stability analysis. `FEATURE_IMPORTANCE_STABILITY.csv` supplies "
        "descriptive means, standard deviations, positive fractions, and fold counts. "
        "Given the small test folds and correlated/repeated predictors, no substantive "
        "clinical feature conclusion is justified.\n",
        encoding="utf-8",
    )

    claim_rows = [
        ["Stage 6 manifest used exactly", "verified", "V2 SPLIT_MANIFEST_USED.json and exact row comparison", "Verified correct"],
        ["Five repeats, five outer folds, four inner folds", "verified", "EXPERIMENT_CONFIGURATION.json and assignments", "Verified correct"],
        ["Highest calibrated ROC-AUC 0.735 +/- 0.060 balanced", "verified", "MODEL_COMPARISON_RESULTS.csv", "Verified correct"],
        ["Highest calibrated ROC-AUC 0.735 +/- 0.061 unweighted", "verified", "MODEL_COMPARISON_RESULTS.csv", "Verified correct"],
        ["Sigmoid calibration used cross-fitted inner-training predictions only", "partially_supported", "Code confirms row-held-out inner predictions, but hyperparameters were selected using all inner folds", "Requires correction"],
        ["SVM decision scores were not native probabilities", "verified", "native_output_type and metric omissions", "Verified correct"],
        ["Not external validation or clinical deployment", "verified", "RESULTS_UPDATE.md", "Verified correct"],
        ["No patient identifier available", "verified", "EXPERIMENT_CONFIGURATION.json and source-row policy", "Verified correct"],
    ]
    pd.DataFrame(claim_rows, columns=["claim", "audit_basis", "evidence", "classification"]).to_csv(
        OUT / "MANUSCRIPT_CLAIM_AUDIT.csv", index=False
    )

    (OUT / "CORRECTION_AND_ACCEPTANCE_DECISIONS.md").write_text(
        "# Correction and Acceptance Decisions\n\n"
        "## Overall classification: Requires correction before final calibrated baseline acceptance\n\n"
        "V2 is a valid and reproducible internal-validation artifact for discrimination "
        "and descriptive probability analysis, and its outer-test isolation is verified. "
        "It should not yet be accepted as the final calibrated baseline because the "
        "calibration-training predictions use hyperparameters selected with the same "
        "calibration fold labels included in model selection.\n\n"
        "## Required correction\n\n"
        "Use a fully nested calibration design: for each calibration fold, select "
        "hyperparameters using only the other inner-training partitions, fit that selected "
        "pipeline without the calibration fold, generate the calibration score, and fit "
        "the sigmoid calibrator only after all such scores are assembled. Preserve the "
        "outer test set for final evaluation. Do not reinterpret the current V2 outputs "
        "as fully nested calibration results.\n\n"
        "The current metrics are independently reproduced. Statistical comparisons and "
        "uncertainty remain descriptive, and feature importance does not support causal "
        "or substantive clinical conclusions.\n",
        encoding="utf-8",
    )
    (OUT / "EXECUTIVE_V2_AUDIT_REPORT.md").write_text(
        "# Executive V2 Independent Audit Report\n\n"
        "## Overall classification: Requires correction\n\n"
        "The persisted outer and inner split assignments are exactly reproduced, the "
        "outer-test predictions are complete, and all independently recomputed metrics "
        "and calibration diagnostics match the saved V2 tables. SVM scores are correctly "
        "labelled as decision scores, and invalid native SVM Brier/log-loss summaries are "
        "not reported.\n\n"
        "The principal unresolved methodological issue is calibration nesting. V2 generates "
        "inner cross-fitted calibration scores, but selects the hyperparameters used for "
        "those scores using all four inner folds, including the calibration fold. This "
        "creates selection-induced optimism in the calibration-training scores. It is not "
        "outer-test leakage, but it requires a fully nested calibration correction before "
        "V2 can be accepted as the final calibrated baseline.\n\n"
        "Uncertainty and statistical comparisons are appropriately labelled descriptive "
        "conditional resampling analyses. Feature importance is held-out predictive "
        "permutation importance, but it does not support causal or substantive clinical "
        "claims. The separate Version 2 results update is numerically accurate and avoids "
        "external-validation or clinical-utility overclaims.\n\n"
        "No V2 model training or cross-validation was rerun during this audit, and "
        "historical artifacts were preserved.\n",
        encoding="utf-8",
    )

    input_paths = [
        ROOT / "runner" / "primary_research_v2" / "run.py",
        V2 / "REPRODUCIBILITY_MANIFEST.json",
        STAGE6 / "reproducibility" / "split_manifest_v2.json",
        V2 / "OUTER_FOLD_PREDICTIONS.csv",
        V2 / "MODEL_COMPARISON_RESULTS.csv",
        V2 / "CALIBRATION_RESULTS.csv",
        V2 / "FEATURE_IMPORTANCE_RESULTS.csv",
        ROOT / "paper" / "research_v2" / "RESULTS_UPDATE.md",
        VERIFICATION_V1 / "verification_manifest.json",
    ]
    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "AUDIT_MANIFEST.json"]
    audit_manifest = {
        "audit_version": "primary_research_v2_independent_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_script": str(Path(__file__).relative_to(ROOT)),
        "auditor_script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "checks": {
            "outer_manifest_equal": outer_equal,
            "inner_manifest_equal": inner_equal,
            "metric_rows_checked": len(metric_recomputation),
            "metric_failures": int((metric_recomputation.status == "fail").sum()),
            "calibration_rows_checked": len(calibration_recomputation),
            "calibration_failures": int((calibration_recomputation.status == "fail").sum()),
            "v2_model_training_rerun": False,
            "historical_files_modified": False,
        },
        "overall_classification": "requires_correction",
    }
    write_json(OUT / "AUDIT_MANIFEST.json", audit_manifest)


if __name__ == "__main__":
    run()
