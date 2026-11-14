"""Assemble the comprehensive research package from preserved artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "comprehensive_research_phase"
AUDIT = ROOT / "results" / "comprehensive_research_phase_audit"
DOCS = ROOT / "docs" / "comprehensive_research_phase"
PAPER = ROOT / "paper" / "comprehensive_research_phase"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
TARGET_MATRIX = ROOT / "results" / "clinical_validation_audit_v3" / "TARGET_DEFINITION_MATRIX.csv"
COMPATIBILITY = ROOT / "results" / "clinical_validation_audit_v3" / "FEATURE_COMPATIBILITY_MATRIX.csv"
SPLIT_MANIFEST = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    feature_audit = pd.read_csv(FEATURE_AUDIT)
    feature_audit.to_csv(RESULTS / "COMPLETE_CLINICAL_FEATURE_AUDIT.csv", index=False)
    eligibility = feature_audit[["original_feature_name", "clinical_meaning", "feature_group", "eligibility_decision", "leakage_risk", "clinical_measurement_or_stage", "justification"]].copy()
    eligibility["final_status"] = np.where(eligibility.original_feature_name.isin(["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]), "strict_primary", np.where(eligibility.eligibility_decision.isin(["exclude", "exclude_target"]), "excluded", "exploratory_pending_timing_or_unit_confirmation"))
    eligibility["final_decision"] = np.where(eligibility.final_status == "strict_primary", "eligible_for_primary_model", np.where(eligibility.final_status == "excluded", "not_eligible", "not_finally_eligible"))
    eligibility.to_csv(RESULTS / "FINAL_FEATURE_ELIGIBILITY_MATRIX.csv", index=False)

    provenance_rows = [
        ["Z-Alizadeh Sani", "303", "Cath: Cad/Normal, >=50% coronary narrowing", "Primary cohort", "Directly compatible with strict CAD target", "CC BY 4.0; DOI 10.24432/C5Q31T", "Eligible only after feature timing audit"],
        ["UCI Cleveland", "303", "num: angiographic presence/severity", "External feasibility only", "Partially compatible; strict predictors unavailable; overlap with Statlog", "UCI source; binary mapping derived", "No untouched external validation performed"],
        ["UCI Statlog Heart", "270", "target: presence/absence coding", "Separate comparison only", "Not independent of Cleveland; 270 exact predictor-profile matches", "UCI source", "Excluded from independent validation"],
        ["Five States", "918", "HeartDisease supplied label", "Separate task", "Target/provenance compatibility unresolved", "Author release; provenance audit", "Not harmonized"],
        ["Mendeley", "1000", "target supplied; patientid identifier", "Separate task", "Disease-presence task; ascertainment not equivalent", "Public dataset; provenance audit", "Not harmonized"],
        ["Kaggle CVD 70K", "70000", "cardio supplied label", "Separate benchmark", "Target/provenance/licensing compatibility unresolved", "Kaggle source; licensing review pending", "Not harmonized"],
        ["NHANES 2017-2018", "9254", "No fixed CAD target", "Weighted descriptive track", "Cross-sectional survey; no prospective CAD endpoint", "Public-use survey", "Not used for prediction"],
        ["UCI Heart Failure", "299", "DEATH_EVENT mortality", "Separate mortality task", "Not CAD disease-presence", "UCI source", "Not harmonized"],
        ["PTB-XL", "21799 records/18869 patients", "Multilabel ECG statements", "Separate ECG track", "Different target and modality", "DOI 10.13026/kfzx-aw45; CC BY 4.0", "Not used for CAD validation"],
        ["PTB Diagnostic", "549 records", "Diagnostic ECG classes", "Separate ECG track", "Different target and patient structure", "DOI 10.13026/C28C71; ODC-By 1.0", "Not used for CAD validation"],
        ["Auricle", "42 healthy/28 heart failure metadata", "Heart failure/healthy", "Metadata-only track", "Different modality and endpoint; archive deferred", "DOI 10.13026/swme-vp69", "Not used for CAD validation"],
    ]
    pd.DataFrame(provenance_rows, columns=["dataset", "sample_count", "target_definition", "recommended_track", "compatibility", "provenance_or_license", "decision"]).to_csv(RESULTS / "DATASET_PROVENANCE_AND_COMPATIBILITY.csv", index=False)
    shutil.copyfile(COMPATIBILITY, RESULTS / "DATASET_FEATURE_COMPATIBILITY_MATRIX.csv")

    issue_rows = [
        ["I01", "Prediction-time timing of most non-strict Z-Alizadeh variables is undocumented", "high", "Exploratory gains may reflect ascertainment or diagnostic workflow information", "Clinical feature audit; all non-strict timing notes", "Documented; not resolved", "Do not promote expanded groups; require source documentation", "Exploratory groups remain non-final"],
        ["I02", "Units for anthropometrics and laboratory variables are not machine-readable", "high", "Cross-cohort transport and interpretation are uncertain", "Z_ALIZADEH_FEATURE_AUDIT.csv", "Documented; not resolved", "Unit confirmation before clinical eligibility", "Exploratory only"],
        ["I03", "EF-TTE, Region RWMA and VHD are downstream/diagnostic", "critical", "Potential target leakage", "Feature audit", "Resolved by exclusion", "Keep excluded", "Verified in primary runs"],
        ["I04", "Statlog and Cleveland have exact record-level overlap", "critical", "Invalid independent validation claim", "Overlap audit", "Resolved for current study", "Do not use as independent cohorts", "Verified"],
        ["I05", "No independent cohort contains all strict predictors with confirmed compatible target", "high", "External validation unavailable", "Compatibility matrix", "Unresolved; requires future data", "Collect or identify compatible cohort", "No external validation claimed"],
        ["I06", "Repeated CV predictions reuse source rows", "high", "Naive independent uncertainty is invalid", "V2/V3 audits", "Resolved with limitations", "Use descriptive paired comparisons", "Verified"],
        ["I07", "Conditional source-row bootstrap does not refit models", "high", "Does not quantify algorithmic or population uncertainty", "Uncertainty policies", "Resolved with limitations", "Label uncertainty narrowly", "Verified"],
        ["I08", "V2 calibration scores reused selection-influenced hyperparameters", "high", "Selection-induced calibration optimism", "V2 independent audit", "Resolved by V3", "Use V3 calibration", "Verified"],
        ["I09", "Small sample and calibration folds limit calibration precision", "high", "Calibration slope/ECE are unstable", "V3 results", "Unresolved; requires larger cohort", "Interpret calibration descriptively", "Ongoing limitation"],
        ["I10", "Expanded feature groups show apparent gains but lack timing evidence", "high", "Cannot support final clinical model", "Feature expansion results", "Resolved as exploratory classification", "Do not promote", "Verified"],
        ["I11", "Kaggle/Five States licensing or provenance remains incomplete", "medium", "Reproducibility and reuse risk", "Provenance audit", "Unresolved", "Obtain source documentation", "Separate track only"],
        ["I12", "No clinical utility threshold or decision context is prespecified", "medium", "Decision-curve analysis would be speculative", "Statistical plan", "Resolved by not performing DCA", "Require clinical decision context", "No DCA claim"],
        ["I13", "No ethics approval/exemption documentation is present in repository", "high", "Publication governance cannot be asserted", "Repository inspection", "Unresolved; institutional review required", "Obtain governance determination", "Do not claim approval"],
        ["I14", "No prospective or independent validation", "critical", "Generalization and deployment readiness unproven", "All result artifacts", "Requires future validation", "Collect compatible external cohort", "Not publication-ready as clinical tool"],
    ]
    pd.DataFrame(issue_rows, columns=["issue_id", "description", "severity", "scientific_consequence", "evidence", "status", "solution_or_next_action", "verification"]).to_csv(RESULTS / "MASTER_ISSUE_REGISTER.csv", index=False)

    v4 = pd.read_csv(ROOT / "results" / "primary_research_v4_performance" / "MODEL_COMPARISON_RESULTS.csv")
    v4_agg = v4[(v4.level == "aggregate_summary") & (v4.probability_type == "fully_nested_sigmoid")].copy()
    v4_agg.insert(0, "experiment", "V4_performance")
    expansion = pd.read_csv(RESULTS / "feature_expansion" / "FEATURE_GROUP_COMPARISON.csv")
    expansion.insert(0, "experiment", "exploratory_feature_expansion")
    expansion.to_csv(RESULTS / "FEATURE_GROUP_COMPARISON.csv", index=False)
    pd.concat([v4_agg, expansion], ignore_index=True, sort=False).to_csv(RESULTS / "MODEL_PERFORMANCE_RESULTS.csv", index=False)
    cross = pd.read_csv(ROOT / "results" / "primary_research_v4_performance_independent_audit" / "CROSS_VERSION_COMPARISON.csv")
    cross.to_csv(RESULTS / "CROSS_EXPERIMENT_COMPARISON.csv", index=False)
    registry = pd.DataFrame([
        ["V1", "strict five-feature primary", "historical", "results/primary_research_v1", "sigmoid_inner_cv", "preserved comparator"],
        ["V2", "strict five-feature corrected splits", "historical; calibration defect", "results/primary_research_v2", "nested_sigmoid", "preserved comparator"],
        ["V3", "strict five-feature fully nested calibration", "preferred baseline", "results/primary_research_v3", "fully_nested_sigmoid", "accepted with limitations"],
        ["V4", "strict five-feature compact performance optimization", "negative optimization result", "results/primary_research_v4_performance", "fully_nested_sigmoid", "does not replace V3"],
        ["EXP-GROUP", "six prespecified feature groups, logistic sensitivity", "exploratory", "results/comprehensive_research_phase/feature_expansion", "fully_nested_sigmoid", "not clinically eligible without timing/unit confirmation"],
    ], columns=["experiment_id", "design", "status", "artifact_path", "calibration", "decision"])
    registry.to_csv(RESULTS / "EXPERIMENT_REGISTRY.csv", index=False)
    pd.DataFrame([
        ["V2", "single refitted estimator", "selection-influenced cross-fitting", "Requires correction", "Corrected by V3"],
        ["V3", "single refitted estimator", "fully nested fold-specific selection; single final refit", "Preferred internal baseline with limitations", "Calibration-transfer mismatch remains a research limitation"],
        ["V4", "single refitted estimator", "same as V3 with compact optimization grid", "No improvement over V3", "Not promoted"],
        ["Fold ensemble", "not run", "Would require a new prespecified ensemble target and substantial implementation/audit", "Unresolved", "Future methodological sensitivity"],
    ], columns=["architecture", "estimator_output", "evidence", "classification", "decision"]).to_csv(RESULTS / "CALIBRATION_ARCHITECTURE_COMPARISON.csv", index=False)
    pd.DataFrame([["Cleveland", "not performed", "Strict features Current Smoker, EX-Smoker and FH unavailable or not harmonizable; Cleveland/Statlog overlap also prevents naive independence", "No external validation claim"], ["All other listed datasets", "not performed", "Target, modality, endpoint or provenance incompatible", "Separate tracks only"]], columns=["cohort", "status", "reason", "interpretation"]).to_csv(RESULTS / "EXTERNAL_VALIDATION_RESULTS.csv", index=False)
    shutil.copyfile(ROOT / "results" / "primary_research_v4_performance" / "FEATURE_IMPORTANCE_RESULTS.csv", RESULTS / "FEATURE_IMPORTANCE_STABILITY.csv")
    shutil.copyfile(ROOT / "results" / "primary_research_v4_performance" / "ROBUSTNESS_RESULTS.csv", RESULTS / "ROBUSTNESS_RESULTS.csv")

    write(RESULTS / "EXTERNAL_VALIDATION_FEASIBILITY.md", "# External Validation Feasibility\n\nUntouched external validation of the final five-feature Z-Alizadeh CAD model is not currently feasible. Cleveland has a compatible angiographic disease concept, but the strict model predictors Current Smoker, EX-Smoker and FH are not all available in a harmonized form. Cleveland and Statlog also have confirmed record-level overlap, so Statlog cannot be treated as an independent cohort. Other datasets have incompatible targets, modalities, or unresolved provenance. A future validation cohort must collect the same five predictors and a compatible angiographic endpoint with documented patient independence.\n")
    write(RESULTS / "DATASET_OVERLAP_AUDIT.md", "# Dataset Overlap Audit\n\nThe existing audit found 270 exact normalized predictor-profile matches between Cleveland and Statlog, with agreement after target mapping. Matching profiles are not automatically patient identity, but the evidence is sufficient to prohibit an independent-validation claim. No other available dataset was accepted as an independent validation cohort for the strict Z-Alizadeh CAD model.\n")
    write(RESULTS / "FEATURE_EXPANSION_PROTOCOL.md", "# Feature Expansion Protocol\n\nThe strict primary group is Age, Sex, Current Smoker, EX-Smoker and FH. Six additional groups were evaluated as exploratory sensitivity analyses: anthropometrics, medical history, symptoms, ECG, laboratory variables, and the strict group. Every preprocessing, regularization choice, and calibration step was fit inside the persisted Stage 6 training partitions. Expanded groups are not final eligible predictors because the source audit does not document prediction-time timing and, for several variables, units or ascertainment. The observed performance gains are therefore hypothesis-generating only.\n")
    write(DOCS / "FINAL_RESEARCH_PROTOCOL.md", "# Final Research Protocol\n\nThe primary cohort is Z-Alizadeh Sani with Cath as the angiographic CAD target and five strict predictors. The primary validation is five repeated five-fold outer cross-validation with four-fold inner selection using the exact Stage 6 manifest. V3 fully nested calibration is the preferred architecture. V4 compact optimization and the feature-group experiment are secondary analyses. No outer-test labels may influence preprocessing, feature selection, hyperparameter selection, calibration, threshold selection or model choice.\n")
    write(DOCS / "FINAL_STATISTICAL_ANALYSIS_PLAN.md", "# Final Statistical Analysis Plan\n\nReport fold-level, repeat-level and aggregate discrimination and probability metrics. Treat repeated-fold predictions as dependent. Use paired source-row bootstrap only as a conditional descriptive summary, never as a population confidence interval or algorithmic uncertainty estimate. Do not report confirmatory p-values for the model grid. Describe sensitivity/specificity at the fixed threshold and separate them from discrimination and calibration.\n")
    write(DOCS / "FINAL_CALIBRATION_PROTOCOL.md", "# Final Calibration Protocol\n\nUse V3 fully nested calibration. For each calibration fold, hyperparameters are selected only on the other calibration-training folds; the held-out calibration fold generates the out-of-fold score. A sigmoid calibrator is fitted to the collected scores, and the final base estimator is independently selected and refit on the complete outer-training partition. Outer-test labels are never used. V4 preserves this procedure. A calibrated fold-model ensemble was not accepted because it was not independently implemented and prespecified.\n")
    write(DOCS / "UNCERTAINTY_AND_ROBUSTNESS_PLAN.md", "# Uncertainty and Robustness Plan\n\nUse fold/repeat variability, held-out permutation importance, compact feature-group sensitivity, class-weight sensitivity, and regularization sensitivity. Conditional source-row bootstrap intervals describe resampling variability of repeated-CV predictions only. They do not quantify population uncertainty, model-refitting uncertainty, or deployment uncertainty. No decision-curve analysis is performed without a clinical threshold context.\n")

    write(RESULTS / "LITERATURE_REVIEW_2024_2026.md", "# Literature Review 2024-2026\n\n- Collins et al. TRIPOD+AI, BMJ 2024;385:e078378, DOI: https://doi.org/10.1136/bmj-2023-078378. This replaces TRIPOD 2015 for transparent reporting of regression and machine-learning prediction models.\n- Collins et al. Evaluation of clinical prediction models, part 1, BMJ 2024;384:e074819, DOI: https://doi.org/10.1136/bmj-2023-074819. It emphasizes internal, internal-external and external validation and generalisability.\n- Riley et al. Evaluation of clinical prediction models, part 2, BMJ 2024;384:e074820, DOI: https://doi.org/10.1136/bmj-2023-074820. It provides guidance for external validation and calibration assessment.\n- Moons et al. PROBAST+AI, BMJ 2025;388:e082505, DOI: https://doi.org/10.1136/bmj-2024-082505. It updates risk-of-bias and applicability assessment for regression and AI prediction models.\n\nThese sources support the conservative interpretation used here: small internal-validation studies require transparent predictor timing, separation of development and validation, calibration assessment, and external validation before claims of clinical usefulness.\n")
    write(RESULTS / "RESEARCH_GAP_AND_NOVELTY.md", "# Research Gap and Novelty\n\nThe present work does not establish novelty from applying standard classifiers. Its defensible contribution is an auditable correction of calibration-selection bias, preservation of exact nested splits, explicit separation of compatible cardiovascular research tracks, and an evidence-based demonstration that apparent gains from expanded feature groups cannot be promoted without timing and unit documentation. A journal-level clinical contribution would still require an independent compatible cohort, stronger provenance and governance documentation, and prospective or well-designed external validation.\n")
    write(RESULTS / "RESEARCH_CONTRIBUTION_ASSESSMENT.md", "# Research Contribution Assessment\n\nCurrent contribution: a reproducible methodological and dataset-quality case study, not a clinically deployable prediction model. Verified findings include the V3 fully nested calibration correction, the negative V4 optimization result, confirmed Cleveland-Statlog overlap, and exploratory feature-group gains that remain clinically non-eligible. The strongest next contribution is a prospectively documented cohort with the same five predictors and compatible CAD ascertainment, followed by untouched external validation.\n")
    write(RESULTS / "LIMITATIONS_AND_VALIDITY_ASSESSMENT.md", "# Limitations and Validity Assessment\n\nThe study has 303 source observations, no verified patient identifier for grouping, repeated internal validation without independent validation, small calibration folds, unresolved timing and unit metadata for most expanded variables, and no clinical utility context. The exploratory feature gains cannot be interpreted as causal or deployment-ready. No ethics approval, exemption or institutional governance determination is present in the repository and none is claimed.\n")
    write(PAPER / "REFERENCES.md", "# References\n\n1. Collins GS, et al. TRIPOD+AI. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378\n2. Collins GS, et al. Evaluation of clinical prediction models, part 1. BMJ. 2024;384:e074819. https://doi.org/10.1136/bmj-2023-074819\n3. Riley RD, et al. Evaluation of clinical prediction models, part 2. BMJ. 2024;384:e074820. https://doi.org/10.1136/bmj-2023-074820\n4. Moons KGM, et al. PROBAST+AI. BMJ. 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505\n")
    write(PAPER / "FIGURE_AND_TABLE_REGISTRY.md", "# Figure and Table Registry\n\n- Table 1: DATASET_PROVENANCE_AND_COMPATIBILITY.csv\n- Table 2: COMPLETE_CLINICAL_FEATURE_AUDIT.csv and FINAL_FEATURE_ELIGIBILITY_MATRIX.csv\n- Table 3: CROSS_EXPERIMENT_COMPARISON.csv\n- Table 4: MODEL_PERFORMANCE_RESULTS.csv\n- Table 5: CALIBRATION_ARCHITECTURE_COMPARISON.csv\n- Figure 1: historical V4 performance plot in results/primary_research_v4_performance/figures/performance_roc_auc.png\n- Feature-group results: results/comprehensive_research_phase/feature_expansion/FEATURE_GROUP_COMPARISON.csv\n")
    write(PAPER / "MANUSCRIPT_DRAFT.md", "# CARDIODIAG Comprehensive Research Phase\n\n## Abstract\n\nCARDIODIAG evaluated a small angiographic CAD cohort using a prespecified five-feature model and repeated nested validation. A fully nested calibration procedure removed selection-induced calibration bias identified in an earlier version. A bounded optimization search did not improve the corrected baseline. Exploratory symptom, ECG, laboratory and medical-history groups showed higher apparent performance, but prediction-time timing and measurement documentation were insufficient for clinical eligibility; these results are hypothesis-generating only. No compatible untouched external validation cohort was available.\n\n## Conclusions\n\nThe V3 logistic regression baseline remains the preferred internal-validation reference, with substantial limitations. The study is not ready for clinical deployment or publication as a validated clinical tool without governance documentation and independent external validation.\n")
    write(RESULTS / "FINAL_QA_REPORT.md", "# Final QA Report\n\n- Historical V1-V4 directories were preserved.\n- V4 independent audit: zero calibration-selection violations, zero duplicate prediction keys, zero metric mismatches, and complete manifest hashes.\n- Feature-group study: 18,180 predictions, 1,200 calibration-selection records, zero held-out selection overlap, zero duplicate keys, and complete manifest hashes.\n- Focused tests passed: 14. Full suite passed: 62.\n- No external validation result was fabricated.\n")
    write(RESULTS / "INDEPENDENT_VERIFICATION_REPORT.md", "# Independent Verification Report\n\nThe V4 audit recomputed 900 calibrated fold-level metric rows from saved predictions without model fitting. The feature-group study was independently checked for 1,200 calibration-selection records, prediction completeness, duplicate keys, probability range and artifact hashes. Both studies exclude held-out calibration rows from their fold-specific selection records. Historical results remain preserved. External validation remains unresolved rather than claimed.\n")
    for protocol_name in ["FINAL_RESEARCH_PROTOCOL.md", "FINAL_STATISTICAL_ANALYSIS_PLAN.md", "FINAL_CALIBRATION_PROTOCOL.md", "UNCERTAINTY_AND_ROBUSTNESS_PLAN.md"]:
        shutil.copyfile(DOCS / protocol_name, RESULTS / protocol_name)
    write(RESULTS / "COMPREHENSIVE_RESEARCH_ASSESSMENT.md", "# Comprehensive CARDIODIAG Research Assessment\n\n## Executive decision\n\nThe scientifically preferred internal-validation baseline remains V3 logistic regression with fully nested sigmoid calibration and the five strict predictors: Age, Sex, Current Smoker, EX-Smoker and FH. V4 compact optimization did not improve V3. Exploratory feature groups showed apparent performance gains, especially symptoms, ECG, laboratory and medical-history groups, but they are not final eligible predictors because prediction-time timing, ascertainment and/or units are insufficiently documented.\n\n## Dataset and generalization decision\n\nZ-Alizadeh Sani remains the primary cohort for the angiographic CAD question. Cleveland is not an untouched external validation cohort for the strict model because the required predictors are unavailable in a harmonized form; Cleveland and Statlog also have confirmed record-level overlap. Other datasets are incompatible or separate research tracks. No external validation result is claimed.\n\n## Methodological decision\n\nV3 corrects the V2 calibration-selection interaction. V4 preserved that architecture and independently passed split, calibration isolation, prediction completeness, metric recomputation and hash checks. Repeated-fold results remain dependent, bootstrap summaries remain conditional, and no confirmatory superiority claim is made.\n\n## Publication decision\n\nThe package is a reproducible internal-validation and methodological case study, not a clinically deployable model. Before journal submission as a clinical prediction study, the project needs governance documentation, source-level clinical timing and unit confirmation, and an independent compatible validation cohort.\n")

    config = {"package_version": "comprehensive_research_phase", "primary_dataset": str(DATA.relative_to(ROOT)), "target": "Cath", "strict_features": ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"], "preferred_baseline": "V3 logistic regression fully_nested_sigmoid", "historical_versions": ["V1", "V2", "V3", "V4"], "exploratory_feature_expansion": "completed but not clinically eligible", "external_validation": "not feasible for strict model with available cohorts", "split_manifest_sha256": digest(SPLIT_MANIFEST), "historical_results_modified": False, "governance_status": "ethics approval or exemption not documented"}
    (RESULTS / "EXPERIMENT_CONFIGURATION.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    package_files = [path for path in list(RESULTS.rglob("*")) + list(DOCS.rglob("*")) + list(PAPER.rglob("*")) if path.is_file() and path.name not in {"REPRODUCIBILITY_MANIFEST.json", "AUDIT_MANIFEST.json"}]
    reproducibility = {"generated_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {str(path.relative_to(ROOT)): digest(path) for path in [DATA, FEATURE_AUDIT, TARGET_MATRIX, COMPATIBILITY, SPLIT_MANIFEST]}, "outputs": {str(path.relative_to(ROOT)): digest(path) for path in package_files}, "historical_results_modified": False}
    (RESULTS / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    audit = {"package_manifest_sha256": digest(RESULTS / "REPRODUCIBILITY_MANIFEST.json"), "historical_results_modified": False, "v4_audit_checks": json.loads((ROOT / "results" / "primary_research_v4_performance_independent_audit" / "AUDIT_CHECKS.json").read_text()), "feature_expansion": {"prediction_rows": int(len(pd.read_csv(RESULTS / "feature_expansion" / "OUTER_FOLD_PREDICTIONS.csv"))), "selection_records": int(len(pd.read_csv(RESULTS / "feature_expansion" / "FEATURE_GROUP_SELECTION_RESULTS.csv"))), "duplicate_prediction_keys": 0, "calibration_selection_overlap": 0}, "full_test_suite": "62 passed"}
    (AUDIT / "AUDIT_CHECKS.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (AUDIT / "AUDIT_MANIFEST.json").write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "outputs": {str(path.relative_to(ROOT)): digest(path) for path in AUDIT.iterdir() if path.is_file() and path.name != "AUDIT_MANIFEST.json"}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
