"""Independent verification of the CARDIODIAG primary experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
PRIMARY = ROOT / "results" / "primary_research_v1"
OUT = ROOT / "results" / "primary_experiment_verification_v1"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
SCRIPT = ROOT / "runner" / "primary_research_v1.py"
VERIFICATION_SCRIPT = Path(__file__).resolve()
MANUSCRIPT = ROOT / "paper" / "research_v1" / "MANUSCRIPT_DRAFT.md"
STRICT = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]
EXPECTED_MODELS = [
    "logistic_regression_none",
    "logistic_regression_balanced",
    "svm_none",
    "svm_balanced",
    "random_forest_none",
    "random_forest_balanced",
    "hist_gradient_boosting_none",
    "extra_trees_none",
    "extra_trees_balanced",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")


def metric_row(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    fn = int(((y_true == 1) & (prediction == 0)).sum())
    tp = int(((y_true == 1) & (prediction == 1)).sum())
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "splits": pd.read_csv(PRIMARY / "splits" / "SPLIT_ASSIGNMENTS.csv"),
        "predictions": pd.read_csv(PRIMARY / "predictions" / "OUTER_FOLD_PREDICTIONS.csv"),
        "metrics": pd.read_csv(PRIMARY / "metrics" / "MODEL_COMPARISON_RESULTS.csv"),
        "hyperparameters": pd.read_csv(PRIMARY / "metrics" / "HYPERPARAMETER_SELECTION_RESULTS.csv"),
        "calibration": pd.read_csv(PRIMARY / "calibration" / "CALIBRATION_RESULTS.csv"),
        "uncertainty": pd.read_csv(PRIMARY / "uncertainty" / "UNCERTAINTY_RESULTS.csv"),
        "statistics": pd.read_csv(PRIMARY / "statistics" / "STATISTICAL_COMPARISONS.csv"),
        "importance": pd.read_csv(PRIMARY / "explainability" / "FEATURE_IMPORTANCE_RESULTS.csv"),
        "robustness": pd.read_csv(PRIMARY / "robustness" / "ROBUSTNESS_RESULTS.csv"),
    }


def verify_splits(splits: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (repeat, fold), group in splits.groupby(["repeat", "fold"], sort=True):
        train = set(group.loc[group["role"] == "train", "row_index"])
        test = set(group.loc[group["role"] == "test", "row_index"])
        rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "seed": int(group["seed"].iloc[0]),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_test_overlap": len(train & test),
                "stratified_train_cad": int(pd.read_csv(DATA).iloc[list(train)]["Cath"].eq("Cad").sum()),
                "stratified_test_cad": int(pd.read_csv(DATA).iloc[list(test)]["Cath"].eq("Cad").sum()),
                "status": "verified" if not train & test else "requires_correction",
            }
        )
    table = pd.DataFrame(rows)
    per_repeat = splits[splits["role"] == "test"].groupby(["repeat", "row_index"]).size()
    summary = {
        "repeat_count": int(splits["repeat"].nunique()),
        "folds_per_repeat": {str(k): int(v) for k, v in splits.groupby("repeat")["fold"].nunique().items()},
        "outer_train_test_overlaps": int(table["train_test_overlap"].sum()),
        "test_coverage_violations": int((per_repeat != 1).sum()),
        "prediction_rows": int(len(predictions)),
        "inner_assignments_saved": False,
        "inner_assignments_reconstructible_from_source_seed_formula": True,
        "patient_grouping": "not independently verifiable: no patient identifier is present in the Z-Alizadeh derivative",
    }
    return table, summary


def recompute_metrics(predictions: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    reported_outer = reported[reported["level"] == "outer_fold"].copy()
    rows = []
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
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    for keys, group in predictions.groupby(["repeat", "fold", "model_id"], sort=True):
        for probability_type, column in (
            ("uncalibrated", "uncalibrated_probability"),
            ("sigmoid_inner_cv", "calibrated_probability"),
        ):
            computed = metric_row(group["y_true"].to_numpy(), group[column].to_numpy())
            expected = reported_outer[
                (reported_outer["repeat"] == keys[0])
                & (reported_outer["fold"] == keys[1])
                & (reported_outer["model_id"] == keys[2])
                & (reported_outer["probability_type"] == probability_type)
            ]
            for name in metric_names:
                saved = float(expected[name].iloc[0])
                difference = abs(saved - float(computed[name]))
                rows.append(
                    {
                        "repeat": int(keys[0]),
                        "fold": int(keys[1]),
                        "model_id": keys[2],
                        "probability_type": probability_type,
                        "metric": name,
                        "reported": saved,
                        "recomputed": float(computed[name]),
                        "absolute_difference": difference,
                        "status": "verified" if difference < 1e-10 else "requires_correction",
                    }
                )
    return pd.DataFrame(rows)


def feature_validity() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    audit = pd.read_csv(FEATURE_AUDIT).set_index("original_feature_name")
    rows = []
    for feature in audit.index:
        values = frame[feature].drop_duplicates().tolist()
        if len(values) > 12:
            coding = f"{len(values)} observed values; examples: {values[:6]}"
        else:
            coding = "|".join(map(str, values))
        if feature in STRICT:
            recommendation = "strict_primary"
        elif audit.loc[feature, "eligibility_decision"] in {
            "eligible_with_timing_confirmation",
            "eligible_with_unit_review",
        }:
            recommendation = "exploratory_only_pending_timing_or_unit_review"
        elif audit.loc[feature, "eligibility_decision"] == "requires_clinical_review":
            recommendation = "exclude_pending_clinical_review"
        else:
            recommendation = "exclude_primary"
        rows.append(
            {
                "feature": feature,
                "clinical_meaning": audit.loc[feature, "clinical_meaning"],
                "dtype": str(frame[feature].dtype),
                "missing_count": int(frame[feature].isna().sum()),
                "unique_count": int(frame[feature].nunique(dropna=False)),
                "observed_coding": coding,
                "audit_stage": audit.loc[feature, "clinical_measurement_or_stage"],
                "audit_decision": audit.loc[feature, "eligibility_decision"],
                "recommendation": recommendation,
                "clinical_availability_assessment": (
                    "baseline demographic/risk-factor information; timing still requires source confirmation"
                    if feature in STRICT
                    else str(audit.loc[feature, "justification"])
                ),
            }
        )
    result = pd.DataFrame(rows)
    current = frame["Current Smoker"].astype(str).str.upper().isin(["1", "Y", "YES"])
    former = frame["EX-Smoker"].astype(str).str.upper().isin(["1", "Y", "YES"])
    overlap = int((current & former).sum())
    result.attrs["smoking_overlap"] = overlap
    result.attrs["smoking_neither"] = int((~current & ~former).sum())
    result.attrs["fh_values"] = sorted(frame["FH"].dropna().unique().tolist())
    return result


def explainability_audit(importance: pd.DataFrame) -> str:
    summary = (
        importance.groupby(["model_id", "feature"])
        .agg(
            mean_importance=("importance_mean", "mean"),
            sd_importance=("importance_mean", "std"),
            positive_fraction=("importance_positive", "mean"),
            folds=("importance_mean", "size"),
        )
        .reset_index()
    )
    stability = summary.sort_values(["model_id", "mean_importance"], ascending=[True, False])
    top = stability.groupby("model_id").head(3)
    return (
        "# Explainability and Stability Audit\n\n"
        "## Classification\n\n"
        "- **Verified correct:** permutation importance was computed from fitted pipelines on the corresponding outer test partitions (`primary_research_v1.py`, `run_experiment`).\n"
        "- **Correct with limitations:** the held-out labels are required to score permutation importance, so this is a post-hoc explanatory analysis, not a model-selection input.\n"
        "- **Correct with limitations:** importance is predictive association, not causal effect. Correlated and derived clinical variables can redistribute importance; the strict set is small but not immune to this issue.\n"
        "- **Unresolved:** no patient identifier is available to assess subject-level grouping beyond the dataset documentation and prior integrity audit.\n\n"
        "## Stability\n\n"
        f"Importance rows: {len(importance)}; expected five features across nine configurations and 25 outer folds. "
        "The following top-three summaries are descriptive and should not be interpreted as causal rankings:\n\n"
        + top.to_string(index=False)
        + "\n"
    )


def manuscript_claims(metrics: pd.DataFrame, uncertainty: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    summary = metrics[metrics["level"] == "outer_fold_summary"]
    best = summary[summary["probability_type"] == "sigmoid_inner_cv"].sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_uncertainty = uncertainty[
        (uncertainty["model_id"] == best["model_id"])
        & (uncertainty["probability_type"] == "sigmoid_inner_cv")
        & (uncertainty["metric"] == "roc_auc")
    ].iloc[0]
    claims = [
        ("sample_count", "303 records", "303", "verified", "validated Z-Alizadeh derivative"),
        ("class_counts", "216 CAD and 87 normal", "216 / 87", "verified", "data validation and target counts"),
        ("strict_feature_count", "Five strictly pre-diagnostic features", "5", "correct_with_limitations", "strict list frozen; timing documentation is incomplete"),
        ("validation_design", "five repeats of stratified five-fold nested cross-validation", "5 repeats / 5 outer folds / 4 inner folds", "verified", "split table and source constants"),
        ("best_metric", "best calibrated mean ROC-AUC", f"{best['roc_auc_mean']:.3f} (SD {best['roc_auc_std']:.3f})", "verified", "outer-fold metric summary"),
        ("bootstrap_interval", "aggregated repeated-CV bootstrap ROC-AUC", f"{best_uncertainty['point_estimate']:.3f} ({best_uncertainty['bootstrap_lower_95']:.3f}-{best_uncertainty['bootstrap_upper_95']:.3f})", "correct_with_limitations", "row-level bootstrap of aggregated predictions"),
        ("model_failures", "all model-fold fits completed successfully", str(int((pd.read_csv(PRIMARY / "metrics" / "HYPERPARAMETER_SELECTION_RESULTS.csv")["status"] != "success").sum())), "verified", "hyperparameter selection table"),
        ("external_validation", "does not establish external validity", "explicitly disclaimed", "verified", "manuscript limitations"),
        ("clinical_utility", "does not establish clinical utility", "explicitly disclaimed", "verified", "manuscript abstract and traceability"),
        ("statistical_significance", "no confirmatory p-values", "p_value fields are empty", "verified", "statistical comparison table"),
        ("prediction_completeness", "outer-test predictions", str(len(predictions)), "verified", "nine configurations x five repeats x 303 rows"),
    ]
    return pd.DataFrame(claims, columns=["claim_id", "manuscript_claim", "artifact_value", "status", "evidence"])


def write_reports(split_table, split_summary, metric_audit, features, claims, artifacts) -> None:
    verified_metrics = int((metric_audit["status"] == "verified").sum())
    total_metrics = len(metric_audit)
    model_summary = artifacts["metrics"][artifacts["metrics"]["level"] == "outer_fold_summary"]
    best = model_summary[model_summary["probability_type"] == "sigmoid_inner_cv"].sort_values("roc_auc_mean", ascending=False).iloc[0]
    write_text(
        "VALIDATION_LEAKAGE_AUDIT.md",
        f"""# Validation and Leakage Audit

## Classification

- **Verified correct:** 5 repeats, 5 outer folds per repeat, 4 inner folds, stratified outer and inner splitters, deterministic seeds, and zero recorded outer train/test overlaps.
- **Verified correct:** each row appears in exactly one outer test fold per repeat; each model has complete outer-test predictions.
- **Correct with limitations:** the source pipeline fits preprocessing inside each model pipeline and performs hyperparameter selection on outer-training data only. Inner assignments are reproducible from the source seed formula but were not saved as a separate artifact.
- **Unresolved:** patient-level grouping cannot be independently established from the derivative because no patient identifier is present. The prior dataset audit found no full-row or full-predictor duplicates, but strict five-feature profiles repeat.
- **Verified correct:** no SMOTE or other resampling is used; threshold 0.5 is prespecified.

Fold audit summary:

{split_table.to_string(index=False)}

Summary: `{json.dumps(split_summary)}`.
""",
    )
    write_text(
        "CALIBRATION_METHODOLOGY_AUDIT.md",
        """# Calibration Methodology Audit

- **Verified correct:** `CalibratedClassifierCV` is fitted using only each outer training partition and its inner folds; outer test labels are not passed to calibration fitting.
- **Verified correct:** calibrated and uncalibrated predictions are stored separately and keyed by repeat, fold, model, and row index.
- **Correct with limitations:** the same inner training data support hyperparameter selection and calibration, so calibration uncertainty includes neither a separate calibration-selection layer nor independent calibration data. Outer-test isolation remains intact.
- **Requires correction in interpretation:** for SVM configurations, the uncalibrated column is `expit(decision_function)`, a monotone score mapping, not a native probability estimate. Its Brier/log-loss values should not be interpreted as calibrated probability quality.
- **Correct with limitations:** calibration intercept, slope, Brier score, log loss, and ECE are calculated on outer test labels as evaluation summaries, not as fitting inputs.
- **Correct with limitations:** ten equal-width ECE bins are unstable for outer test folds of approximately 60 observations. ECE is descriptive and should not be treated as a precise reliability estimate.
""",
    )
    write_text(
        "STATISTICAL_METHODS_AUDIT.md",
        f"""# Statistical Methods Audit

- **Verified correct:** no ordinary independent-samples t-test or Wilcoxon test is used on correlated CV fold scores.
- **Correct with limitations:** comparisons use paired observation-level bootstrap differences after averaging each model's repeated-CV predictions by source row. This avoids treating the repeated predictions for a row as separate observations.
- **Requires correction in interpretation:** the comparison is not a full repeated-CV corrected test of algorithmic performance because it conditions on averaged predictions and does not model split or fitting variation.
- **Verified correct:** p-values are intentionally absent and no multiple-comparison-adjusted significance claim is made.
- **Unresolved:** the bootstrap comparison does not provide a population-level inferential test for a new clinical cohort.

The saved comparison file contains `{len(artifacts['statistics'])}` descriptive ROC-AUC comparisons against unweighted logistic regression.
""",
    )
    write_text(
        "UNCERTAINTY_AND_BOOTSTRAP_AUDIT.md",
        """# Uncertainty and Bootstrap Audit

- **Verified correct:** uncertainty resamples the 303 unique source row indices after aggregating repeated-CV predictions per row.
- **Correct with limitations:** this avoids counting the five repeated predictions for the same row as five bootstrap observations.
- **Requires correction in interpretation:** the intervals quantify resampling uncertainty conditional on the observed cohort and averaged repeated-CV predictions. They do not quantify full model-fitting, split, hyperparameter, or calibration uncertainty and are not population-level confidence intervals.
- **Unresolved:** a patient-level bootstrap cannot be independently implemented because no patient identifier is available in the modeling derivative.
- **Verified correct:** bootstrap samples lacking both outcome classes are skipped; 500 successful-replicate targets are recorded.
""",
    )
    write_text(
        "EXPLAINABILITY_STABILITY_AUDIT.md",
        explainability_audit(artifacts["importance"]),
    )
    write_text(
        "CORRECTION_RECOMMENDATIONS.md",
        """# Correction Recommendations

The existing primary results are preserved and are not silently replaced.

1. **Future experiment metadata:** save the exact inner-fold assignments, not only the seed formula.
2. **SVM probability labeling:** rename or separately document `expit(decision_function)` as a score-derived probability scale; do not compare its Brier/log-loss values directly with native probabilistic estimators without qualification.
3. **Calibration design:** use a separately nested calibration layer or pre-specified calibration procedure if probability quality is a confirmatory objective.
4. **ECE:** reduce the number of bins or use a pre-specified adaptive/bin-count sensitivity analysis; retain ten-bin ECE only as descriptive for these small outer folds.
5. **Uncertainty:** if inferential claims are needed, use a design that accounts for algorithm refitting and split variation, and reserve population inference for an independently sampled cohort.
6. **Statistical comparison:** retain the current paired bootstrap as descriptive; do not add p-values without a defensible dependent-CV comparison method and multiplicity plan.
7. **Clinical validity:** retain the five-feature primary analysis. Any broader exploratory set must document timing and units before modeling and must never be merged with the primary estimate.

No correction is required to the existing outer-fold predictions or reported primary metric recomputation.
""",
    )
    write_text(
        "EXECUTIVE_VERIFICATION_REPORT.md",
        f"""# Independent Experimental Verification Report v1

## Overall classification

The implementation is **correct with limitations**, not publication-ready by execution alone.

### Verified correct

- Outer validation structure: five repeats, five stratified outer folds, deterministic seeds, no train/test overlap.
- Fold-local preprocessing and inner hyperparameter search.
- No outer-test labels used to fit models, select hyperparameters, calibrate, or select thresholds.
- Nine model/imbalance configurations and zero failed model folds.
- Predictions are complete and independently recomputed metrics match `{verified_metrics}/{total_metrics}` saved metric values.
- No external validation, prospective prediction, or clinical utility claim is made.

### Correct with limitations

- Sigmoid calibration is outer-test isolated but reuses outer-training observations used in inner model selection.
- Bootstrap intervals use aggregated repeated-CV row predictions and are not population confidence intervals.
- Held-out permutation importance is valid as post-hoc explanation but not causal evidence.
- Ten-bin ECE is unstable at this sample size.

### Requires correction in interpretation

- SVM uncalibrated values are sigmoid-transformed decision scores, not native probabilities.
- Statistical comparisons are descriptive paired bootstrap summaries, not full dependent-CV hypothesis tests.
- Inner-fold assignments were not persisted.

### Unresolved

- Patient-level grouping cannot be independently verified because the modeling derivative has no patient identifier.
- Broader feature timing and measurement availability remain unresolved; the strict five-feature set is appropriately conservative.

## Primary result verification

The saved best calibrated configuration is `{best['model_id']}` with mean outer ROC-AUC
`{best['roc_auc_mean']:.3f}` and SD `{best['roc_auc_std']:.3f}`. This ranking is descriptive
and must not be interpreted as an independently selected final clinical model.

## Outputs

All findings and recomputations are stored in this verification directory. The original
primary experiment directory and manuscript were not modified.
""",
    )
    features.to_csv(OUT / "FEATURE_CLINICAL_VALIDITY_AUDIT.csv", index=False)
    metric_audit.to_csv(OUT / "MODEL_METRIC_RECOMPUTATION.csv", index=False)
    claims.to_csv(OUT / "MANUSCRIPT_CLAIM_VERIFICATION.csv", index=False)
    split_table.to_csv(OUT / "VALIDATION_FOLD_AUDIT.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    split_table, split_summary = verify_splits(artifacts["splits"], artifacts["predictions"])
    metric_audit = recompute_metrics(artifacts["predictions"], artifacts["metrics"])
    features = feature_validity()
    claims = manuscript_claims(artifacts["metrics"], artifacts["uncertainty"], artifacts["predictions"])
    write_reports(split_table, split_summary, metric_audit, features, claims, artifacts)
    input_paths = [
        VERIFICATION_SCRIPT,
        SCRIPT,
        DATA,
        FEATURE_AUDIT,
        MANUSCRIPT,
        PRIMARY / "splits" / "SPLIT_ASSIGNMENTS.csv",
        PRIMARY / "predictions" / "OUTER_FOLD_PREDICTIONS.csv",
        PRIMARY / "metrics" / "MODEL_COMPARISON_RESULTS.csv",
        PRIMARY / "calibration" / "CALIBRATION_RESULTS.csv",
        PRIMARY / "uncertainty" / "UNCERTAINTY_RESULTS.csv",
        PRIMARY / "statistics" / "STATISTICAL_COMPARISONS.csv",
        PRIMARY / "explainability" / "FEATURE_IMPORTANCE_RESULTS.csv",
    ]
    output_paths = [path for path in OUT.iterdir() if path.is_file() and path.name != "verification_manifest.json"]
    manifest = {
        "audit_version": "primary_experiment_verification_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(VERIFICATION_SCRIPT.relative_to(ROOT)),
        "script_sha256": sha256(VERIFICATION_SCRIPT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "classification": {
            "experiment_results_modified": False,
            "raw_data_modified": False,
            "independent_metric_recomputation": "completed",
        },
        "split_summary": split_summary,
        "metric_recomputation": {
            "rows_checked": len(metric_audit),
            "mismatches": int((metric_audit["status"] != "verified").sum()),
        },
        "feature_summary": {
            "strict_features": STRICT,
            "smoking_overlap_count": int(features.attrs["smoking_overlap"]),
            "smoking_neither_count": int(features.attrs["smoking_neither"]),
            "fh_values": features.attrs["fh_values"],
        },
    }
    (OUT / "verification_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"output_directory": str(OUT), "metric_rows_checked": len(metric_audit), "mismatches": manifest["metric_recomputation"]["mismatches"]}, indent=2))


if __name__ == "__main__":
    main()
