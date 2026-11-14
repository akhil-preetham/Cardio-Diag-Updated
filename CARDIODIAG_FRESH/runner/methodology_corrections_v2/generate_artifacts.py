"""Generate Stage 6 methodology policies and reproducibility metadata only."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .split_manifest import create_nested_split_manifest, save_nested_split_manifest
except ImportError:
    from runner.methodology_corrections_v2.split_manifest import (
        create_nested_split_manifest,
        save_nested_split_manifest,
    )


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "methodology_corrections_v2"
PRIMARY = ROOT / "results" / "primary_research_v1"
VERIFICATION = ROOT / "results" / "primary_experiment_verification_v1"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
STRICT = ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    audit = pd.read_csv(FEATURE_AUDIT)

    issue_rows = [
        ["SVM score probability labeling", "requires_correction", "expit(decision_function) is not native probability calibration", "Use nested Platt calibration; label native scores separately."],
        ["Inner assignments", "requires_correction", "only seed formula was saved in Version 1", "Persist exact outer and inner row assignments."],
        ["Bootstrap uncertainty", "correct_with_limitations", "row bootstrap of aggregated repeated-CV predictions", "Report as conditional resampling uncertainty, not population CI."],
        ["Statistical comparisons", "correct_with_limitations", "paired descriptive bootstrap; no confirmatory p-values", "Retain descriptive analysis; consider dependent-CV sensitivity only if prespecified."],
        ["Ten-bin ECE", "correct_with_limitations", "approximately 60 observations per outer fold", "Use fewer/adaptive bins as sensitivity; keep ECE supplementary."],
        ["Patient grouping", "unresolved", "no patient identifier in modeling derivative", "Do not manufacture IDs; obtain a legitimate identifier for future patient bootstrap/splitting."],
        ["Strict features", "verified_correct", "five features pass current audit", "Preserve as primary baseline."],
    ]
    pd.DataFrame(issue_rows, columns=["issue", "classification", "evidence", "decision"]).to_csv(
        OUT / "ISSUE_RESOLUTION_MATRIX.csv", index=False
    )

    probability_rows = [
        ["Logistic Regression", "native predict_proba", "sigmoid calibration optional", "Report native and calibrated separately; primary probability metric uses held-out calibrated output."],
        ["Random Forest", "native predict_proba", "nested sigmoid calibration", "Do not assume native probabilities are calibrated."],
        ["Extra Trees", "native predict_proba", "nested sigmoid calibration", "Do not compare Brier/log loss without labeling calibration status."],
        ["HistGradientBoosting", "native predict_proba", "nested sigmoid calibration", "Use held-out calibration assessment."],
        ["SVM", "decision_function score", "nested Platt calibration required", "Never call expit(score) native probability; isotonic is exploratory only."],
    ]
    pd.DataFrame(probability_rows, columns=["model_family", "native_output", "primary_calibration_policy", "interpretation"]).to_csv(
        OUT / "MODEL_PROBABILITY_POLICY.csv", index=False
    )

    uncertainty_rows = [
        ["Outer repeat/fold variation", "validation variability", "reportable descriptively", "repeated participants make folds dependent"],
        ["Row-aggregated bootstrap", "conditional resampling", "reportable with qualification", "not population CI; does not refit algorithms"],
        ["Patient bootstrap", "sampling uncertainty", "requires new experiment", "patient IDs unavailable in current derivative"],
        ["Refit bootstrap", "algorithmic uncertainty", "requires new experiment", "must refit preprocessing, tuning, model, and calibration"],
        ["Individual predictive interval", "predictive uncertainty", "deferred", "requires a prespecified probabilistic/conformal method"],
    ]
    pd.DataFrame(uncertainty_rows, columns=["quantity", "uncertainty_source", "Version_1_status", "limitation"]).to_csv(
        OUT / "UNCERTAINTY_REPORTING_POLICY.csv", index=False
    )

    statistics_rows = [
        ["Primary paper comparison", "paired descriptive difference on identical outer predictions", "yes", "No confirmatory p-value"],
        ["Ordinary independent t-test", "independent fold scores", "no", "Violates repeated-CV dependence"],
        ["Nadeau-Bengio corrected resampled test", "dependent resampling sensitivity", "conditional", "Assumptions and multiplicity must be prespecified"],
        ["Multiplicity", "limited prespecified comparisons", "required", "Do not search across models and report a single favorable p-value"],
    ]
    pd.DataFrame(statistics_rows, columns=["analysis", "method", "recommended", "caveat"]).to_csv(
        OUT / "STATISTICAL_COMPARISON_POLICY.csv", index=False
    )

    feature_rows = []
    for _, row in audit.iterrows():
        decision = row["eligibility_decision"]
        if row["original_feature_name"] in STRICT:
            category = "strict_primary"
        elif decision in {"eligible", "eligible_with_timing_confirmation", "eligible_with_unit_review"}:
            category = "exploratory_pending_evidence"
        elif decision == "requires_clinical_review":
            category = "excluded_pending_clinical_review"
        else:
            category = "excluded_target_or_downstream"
        feature_rows.append(
            {
                "feature": row["original_feature_name"],
                "category": category,
                "audit_decision": decision,
                "clinical_meaning": row["clinical_meaning"],
                "primary_inclusion": row["original_feature_name"] in STRICT,
                "reason": row["justification"],
            }
        )
    pd.DataFrame(feature_rows).to_csv(OUT / "FEATURE_POLICY_FINAL.csv", index=False)

    source_row_ids = [f"source_row:{index}" for index in range(len(frame))]
    split_manifest = create_nested_split_manifest(
        (frame["Cath"] == "Cad").astype(int).tolist(),
        outer_seeds=[42, 43, 44, 45, 46],
        outer_folds=5,
        inner_folds=4,
        source_row_ids=source_row_ids,
    )
    save_nested_split_manifest(OUT / "reproducibility" / "split_manifest_v2.json", split_manifest)
    metadata = {
        "protocol_version": "RESEARCH_PROTOCOL_V2",
        "dataset": str(DATA.relative_to(ROOT)),
        "target": "Cath",
        "target_encoding": {"Cad": 1, "Normal": 0},
        "strict_features": STRICT,
        "exploratory_feature_policy": "timing/unit/clinical-review evidence required before use",
        "outer_seeds": [42, 43, 44, 45, 46],
        "outer_folds": 5,
        "inner_folds": 4,
        "calibration": "nested sigmoid/Platt calibration; isotonic exploratory only",
        "threshold": 0.5,
        "resampling": "no SMOTE by default; class weighting only where supported",
        "patient_identifier_available": False,
        "source_row_identifier_policy": "stable source-row labels only; not patient identifiers",
        "code_version": "Git unavailable in repository workspace",
        "historical_results_preserved": True,
        "model_training_launched": False,
    }
    write_json(OUT / "reproducibility" / "experiment_metadata_v2.json", metadata)
    decisions = {
        "stage": "Stage 6",
        "status": "protocol_finalization_without_new_training",
        "primary_feature_set": STRICT,
        "primary_comparison": "descriptive paired comparisons on identical outer predictions",
        "probability_decision": "native and calibrated outputs must be separated; SVM requires nested calibration",
        "uncertainty_decision": "report validation variation and conditional bootstrap only; defer population inference",
        "calibration_decision": "sigmoid primary; isotonic exploratory only",
        "unresolved": ["patient-level identifier unavailable", "feature timing for broader variables", "independent external cohort unavailable"],
    }
    write_json(OUT / "METHODOLOGY_DECISIONS.json", decisions)

    write_json(
        OUT / "EXECUTIVE_METHODOLOGY_REVIEW.json",
        {
            "note": "The human-readable executive review is written below by this generator.",
            "source_experiment_unchanged": True,
        },
    )
    (OUT / "EXECUTIVE_METHODOLOGY_REVIEW.md").write_text(
        "# Executive Methodology Review v2\n\n"
        "The Version 1 experiment is retained as a valid preliminary internal-validation "
        "baseline with explicit limitations. No new model training was launched.\n\n"
        "## Decisions\n\n"
        "- Preserve the five-feature strict primary analysis.\n"
        "- Prohibit treating SVM expit-transformed scores as native probabilities.\n"
        "- Use nested sigmoid calibration as the primary future probability protocol.\n"
        "- Treat current bootstrap intervals and paired comparisons as descriptive.\n"
        "- Persist exact inner folds and source-row reproducibility labels in future runs.\n"
        "- Do not claim population confidence intervals, clinical utility, or external validation.\n\n"
        "The full policy tables and exact split-manifest module are versioned in this directory "
        "and `runner/methodology_corrections_v2/`.\n",
        encoding="utf-8",
    )

    output_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "METHODOLOGY_MANIFEST.json"]
    inputs = [
        DATA,
        FEATURE_AUDIT,
        PRIMARY / "metrics" / "MODEL_COMPARISON_RESULTS.csv",
        PRIMARY / "uncertainty" / "UNCERTAINTY_RESULTS.csv",
        PRIMARY / "statistics" / "STATISTICAL_COMPARISONS.csv",
        VERIFICATION / "EXECUTIVE_VERIFICATION_REPORT.md",
        ROOT / "docs" / "research_protocol_v2" / "RESEARCH_PROTOCOL_V2.md",
    ]
    manifest = {
        "methodology_version": "methodology_corrections_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "policy": {
            "new_model_training": False,
            "historical_results_modified": False,
            "patient_ids_created": False,
        },
    }
    write_json(OUT / "METHODOLOGY_MANIFEST.json", manifest)


if __name__ == "__main__":
    generate()
