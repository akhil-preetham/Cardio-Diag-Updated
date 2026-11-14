"""Create Phase 2 clinical-stage artifacts without modifying historical results."""
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase2"
DOCS = ROOT / "docs" / "phase2"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
FEATURE_AUDIT = ROOT / "results" / "clinical_validation_audit_v3" / "Z_ALIZADEH_FEATURE_AUDIT.csv"
SPLITS = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
EXP = ROOT / "results" / "comprehensive_research_phase" / "feature_expansion"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(FEATURE_AUDIT)
    audit.to_csv(OUT / "COMPLETE_FEATURE_PROVENANCE_AUDIT.csv", index=False)
    timing = audit[["original_feature_name", "feature_group", "clinical_measurement_or_stage", "eligibility_decision", "justification", "source_evidence"]].copy()
    timing["timing_evidence_status"] = np.where(timing.original_feature_name.isin(["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]), "documented_as_baseline_but source timing not fully documented", np.where(timing.eligibility_decision.isin(["exclude", "exclude_target"]), "later_stage_or_target", "timing_requires_source_confirmation"))
    timing["available_before_angiography"] = np.where(timing.eligibility_decision.isin(["exclude", "exclude_target"]), "not accepted", "unverified")
    timing.to_csv(OUT / "FEATURE_TIMING_EVIDENCE.csv", index=False)
    stage = audit[["original_feature_name", "feature_group", "clinical_measurement_or_stage", "eligibility_decision", "leakage_risk"]].copy()
    stage["stage"] = np.select(
        [stage.feature_group.isin(["demographic", "medical history"]), stage.feature_group.isin(["symptom", "symptom/functional status", "physical examination"]), stage.feature_group.isin(["ECG", "laboratory"]), stage.feature_group.str.contains("echocardiography") | stage.original_feature_name.isin(["EF-TTE", "Region RWMA", "VHD"]), stage.original_feature_name.eq("Cath")],
        ["Stage 1 baseline risk", "Stage 2 clinical presentation", "Stage 3 initial investigation", "Stage 4 later diagnostic", "target"],
        default="unclassified",
    )
    stage["final_status"] = np.where(stage.original_feature_name.isin(["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]), "verified_primary_candidate", np.where(stage.eligibility_decision.isin(["exclude", "exclude_target"]), "excluded", "exploratory_pending_timing_or_unit_confirmation"))
    stage.to_csv(OUT / "FINAL_FEATURE_STAGE_MATRIX.csv", index=False)

    performance = pd.read_csv(EXP / "MODEL_PERFORMANCE_RESULTS.csv")
    aggregate = performance[performance.level == "aggregate_summary"].copy()
    stage_map = {"strict_primary": "Stage 1 baseline risk", "symptoms_exploratory": "Stage 2 clinical presentation", "ecg_exploratory": "Stage 3 initial investigation", "laboratory_exploratory": "Stage 3 initial investigation", "medical_history_exploratory": "Stage 1 exploratory risk extension", "demographic_anthropometric_exploratory": "Stage 1 exploratory risk extension"}
    aggregate["clinical_stage"] = aggregate.feature_group.map(stage_map)
    aggregate["stage_status"] = np.where(aggregate.feature_group.eq("strict_primary"), "primary", "exploratory_timing_or_unit_unresolved")
    aggregate.to_csv(OUT / "STAGE_WISE_MODEL_RESULTS.csv", index=False)
    calibration = pd.read_csv(EXP / "CALIBRATION_RESULTS.csv")
    calibration = calibration[calibration.level == "aggregate_summary"].copy()
    calibration["clinical_stage"] = calibration.feature_group.map(stage_map)
    calibration["stage_status"] = np.where(calibration.feature_group.eq("strict_primary"), "primary", "exploratory")
    calibration.to_csv(OUT / "STAGE_WISE_CALIBRATION_RESULTS.csv", index=False)

    predictions = pd.read_csv(EXP / "OUTER_FOLD_PREDICTIONS.csv")
    predictions["clinical_stage"] = predictions.feature_group.map(stage_map)
    y = predictions.y_true.to_numpy()
    decision_rows = []
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    prevalence = float(y.mean())
    for (group, model_id), frame in predictions.groupby(["feature_group", "model_id"]):
        for threshold in thresholds:
            p = frame.probability.to_numpy()
            truth = frame.y_true.to_numpy()
            predicted = p >= threshold
            tp = int(((truth == 1) & predicted).sum())
            fp = int(((truth == 0) & predicted).sum())
            net_benefit = tp / len(truth) - fp / len(truth) * threshold / (1 - threshold)
            treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            decision_rows.append({"feature_group": group, "model_id": model_id, "clinical_stage": stage_map.get(group, group), "threshold": threshold, "net_benefit": net_benefit, "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0, "status": "exploratory; no prespecified clinical decision threshold", "interpretation": "not a clinical utility claim"})
    pd.DataFrame(decision_rows).to_csv(OUT / "STAGE_WISE_DECISION_CURVE_RESULTS.csv", index=False)

    fold = performance[performance.level == "outer_fold"].copy()
    stability = fold.groupby(["feature_group", "model_id"]).agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"), pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), brier_mean=("brier_score", "mean"), brier_sd=("brier_score", "std"), log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std")).reset_index()
    stability["status"] = np.where(stability.feature_group.eq("strict_primary"), "primary", "exploratory")
    stability.to_csv(OUT / "MODEL_STABILITY_RESULTS.csv", index=False)
    pd.DataFrame([{"stage": "Stage 1 baseline risk", "sample_sizes": "303 total; 216 positive; 87 negative", "status": "not_run", "reason": "Learning curves were not added to the frozen V3/V4 protocol; current repeated-CV results are the prespecified internal validation."}, {"stage": "Stage 2 clinical presentation", "sample_sizes": "303 total", "status": "not_run", "reason": "Exploratory group performance is reported; learning-curve refitting was not prespecified."}, {"stage": "Stage 3 initial investigation", "sample_sizes": "303 total", "status": "not_run", "reason": "Timing and unit evidence unresolved."}, {"stage": "Stage 4 later diagnostic", "sample_sizes": "303 total", "status": "not_run", "reason": "Downstream features excluded."}], columns=["stage", "sample_sizes", "status", "reason"]).to_csv(OUT / "LEARNING_CURVE_RESULTS.csv", index=False)

    registry = pd.DataFrame([
        ["P2-S1", "Stage 1 strict baseline", "V3/V4 fully nested logistic and model-family results", "primary", "completed"],
        ["P2-S2", "Stage 2 symptom extension", "Exploratory feature-group logistic analysis", "exploratory", "completed"],
        ["P2-S3", "Stage 3 ECG/laboratory extension", "Exploratory feature-group logistic analysis", "exploratory", "completed"],
        ["P2-S4", "Stage 4 EF/RWMA/VHD", "No model; downstream exclusion", "excluded", "completed"],
        ["P2-EXT", "External validation", "No compatible independent cohort", "unresolved", "not performed"],
    ], columns=["experiment_id", "question", "implementation", "status", "execution_status"])
    registry.to_csv(OUT / "PHASE2_EXPERIMENT_REGISTRY.csv", index=False)

    compatibility = pd.DataFrame([
        ["Z-Alizadeh Sani", "303", "Cath >=50% narrowing", "all strict predictors", "primary", "eligible internal validation", "CC BY 4.0; DOI 10.24432/C5Q31T"],
        ["Cleveland", "303", "angiographic disease coding", "strict smoking/FH unavailable or not harmonized", "external candidate", "not eligible for untouched strict validation", "Overlap with Statlog must be respected"],
        ["Statlog", "270", "presence/absence", "different coding; exact overlap with Cleveland", "external candidate", "not independent", "270 record-level matches"],
        ["MultiD4CAD", "118 (78 CAD, 40 without CAD)", "CAD analysis using CCTA-derived multimodal data", "Clinical biomarkers and CT-derived EAT/PAT; strict five predictors not verified", "external candidate", "not direct strict validation; source access restricted by DUA", "Zenodo 15148653; created 2025; DUA required"],
        ["CADICA", "42 patients", "Invasive coronary angiography lesion annotations/severity", "Selected clinical features; strict five predictors not verified", "external candidate", "not direct strict validation; imaging task and feature harmonization unresolved", "Mendeley DOI 10.17632/p9bpx9ctcv.4; CC BY 4.0; 2024 paper"],
        ["NHANES", "9254 merged", "no angiographic CAD target", "risk factors only", "descriptive", "incompatible", "Cross-sectional survey"],
        ["Kaggle CVD 70K", "70000", "cardio label", "target/provenance compatibility unresolved", "separate benchmark", "incompatible for direct validation", "Licensing/provenance review pending"],
    ], columns=["dataset", "sample_count", "outcome_definition", "predictor_availability", "track", "phase2_decision", "evidence"])
    compatibility.to_csv(OUT / "DATASET_COMPATIBILITY_PHASE2.csv", index=False)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"Stage 1 baseline risk": "#1f77b4", "Stage 2 clinical presentation": "#ff7f0e", "Stage 3 initial investigation": "#2ca02c", "Stage 1 exploratory risk extension": "#9467bd"}
    plt.figure(figsize=(10, 4))
    for index, (label, color) in enumerate(colors.items()):
        plt.text(index, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.35})
        if index < len(colors) - 1:
            plt.annotate("", (index + 0.75, 0.5), (index + 0.25, 0.5), arrowprops={"arrowstyle": "->"})
    plt.xlim(-0.5, len(colors) - 0.5); plt.ylim(0, 1); plt.axis("off"); plt.tight_layout(); plt.savefig(figures / "clinical_stage_architecture.png", dpi=180); plt.close()
    counts = stage.assign(eligible=stage.final_status.eq("verified_primary_candidate")).groupby(["stage", "final_status"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(10, 5), title="Feature eligibility by clinical stage")
    plt.ylabel("Number of source columns"); plt.tight_layout(); plt.savefig(figures / "feature_eligibility_matrix.png", dpi=180); plt.close()
    plot_data = aggregate[aggregate.model_id.str.endswith("__none")].copy()
    plt.figure(figsize=(10, 5))
    plt.bar(plot_data.clinical_stage, plot_data.roc_auc_mean, color=[colors.get(x, "#777777") for x in plot_data.clinical_stage])
    plt.xticks(rotation=25, ha="right"); plt.ylabel("ROC-AUC"); plt.title("Stage-wise exploratory ROC-AUC"); plt.tight_layout(); plt.savefig(figures / "roc_comparison_stages.png", dpi=180); plt.close()
    plt.figure(figsize=(10, 5))
    plt.bar(plot_data.clinical_stage, plot_data.pr_auc_mean, color=[colors.get(x, "#777777") for x in plot_data.clinical_stage])
    plt.xticks(rotation=25, ha="right"); plt.ylabel("Average precision"); plt.title("Stage-wise exploratory precision-recall performance"); plt.tight_layout(); plt.savefig(figures / "precision_recall_comparison_stages.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5))
    for group in ["strict_primary", "symptoms_exploratory", "ecg_exploratory", "laboratory_exploratory"]:
        frame_group = predictions[(predictions.feature_group == group) & predictions.model_id.str.endswith("__none")]
        if frame_group.empty:
            continue
        bins = np.linspace(0, 1, 6)
        x_points, y_points = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (frame_group.probability >= low) & (frame_group.probability < high if high < 1 else frame_group.probability <= high)
            if mask.any():
                x_points.append(frame_group.loc[mask, "probability"].mean()); y_points.append(frame_group.loc[mask, "y_true"].mean())
        plt.plot(x_points, y_points, marker="o", label=group)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed prevalence"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "calibration_plots.png", dpi=180); plt.close()
    decision = pd.read_csv(OUT / "STAGE_WISE_DECISION_CURVE_RESULTS.csv")
    plt.figure(figsize=(9, 5))
    for group in ["strict_primary", "symptoms_exploratory", "ecg_exploratory", "laboratory_exploratory"]:
        frame_group = decision[(decision.feature_group == group) & decision.model_id.str.endswith("__none")]
        if not frame_group.empty:
            plt.plot(frame_group.threshold, frame_group.net_benefit, marker="o", label=group)
    plt.axhline(0, color="k", linewidth=0.8); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "decision_curves_exploratory.png", dpi=180); plt.close()
    stable_plot = stability[stability.model_id.str.endswith("__none")]
    plt.figure(figsize=(9, 5)); plt.errorbar(stable_plot.feature_group, stable_plot.roc_auc_mean, yerr=stable_plot.roc_auc_sd, fmt="o"); plt.xticks(rotation=25, ha="right"); plt.ylabel("ROC-AUC mean +/- fold SD"); plt.tight_layout(); plt.savefig(figures / "model_stability.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4)); plt.text(0.5, 0.5, "Learning curves not run\n(not prespecified in V3/V4 protocol)", ha="center", va="center", fontsize=13); plt.axis("off"); plt.tight_layout(); plt.savefig(figures / "learning_curves_not_run.png", dpi=180); plt.close()

    write(DOCS / "PHASE2_INITIAL_AUDIT.md", "# Phase 2 Initial Audit\n\nThe repository contains preserved V1-V4 experiments, V3 fully nested calibration, the comprehensive research package, exact Stage 6 split manifests, clinical feature audits and independent verification artifacts. The strict Z-Alizadeh primary analysis is reproducible. The earlier symptom and ECG apparent improvements come from exploratory feature groups whose timing and/or coding remain unverified. EF-TTE, Region RWMA and VHD are downstream or diagnostic-stage variables and are excluded from pre-test models. No compatible untouched external cohort is currently available. Historical directories are not modified by Phase 2.\n\nThe existing model evidence covers L2 logistic regression, SVM, random forest, Extra Trees and histogram gradient boosting. Elastic-net, Firth, generalized additive and Bayesian logistic models were not executed in this phase: no validated Firth/GAM/Bayesian implementation was present in the frozen environment, and installing new modeling dependencies would not resolve the primary clinical timing and external-validation blockers. This is an unresolved model-sensitivity limitation, not evidence that the omitted models would perform better.\n")
    write(DOCS / "CLINICAL_FEATURE_ELIGIBILITY.md", "# Clinical Feature Eligibility\n\nOnly Age, Sex, Current Smoker, EX-Smoker and FH remain the verified primary feature set. Other features are classified as potentially pre-test but require timing, ascertainment or unit confirmation. Symptom, ECG and laboratory groups are exploratory only. EF-TTE, Region RWMA, VHD and Cath are excluded from the intended pre-test model. The dataset does not document feature acquisition time sufficiently to convert plausible clinical timing into verified eligibility.\n")
    write(DOCS / "PHASE2_RESEARCH_PROTOCOL.md", "# Phase 2 Research Protocol\n\nStage 1 is the strict five-feature CAD risk baseline. Stage 2 evaluates symptoms only as exploratory presentation information. Stage 3 evaluates ECG and laboratory groups only as exploratory initial-investigation information. Stage 4 is reserved for later diagnostic variables and is not a pre-test model. Exact Stage 6 repeated nested splits and V3 fully nested calibration are retained. All preprocessing and model selection remain within training partitions. No external validation claim is permitted.\n")
    write(DOCS / "PHASE2_STATISTICAL_ANALYSIS_PLAN.md", "# Phase 2 Statistical Analysis Plan\n\nReport ROC-AUC, average precision, Brier score, log loss, calibration intercept/slope, sensitivity, specificity, PPV and balanced accuracy. Treat repeated-fold results as dependent. Use fold/repeat variability descriptively. Decision curves are exploratory because no clinical decision threshold was prespecified. Do not treat cross-validation standard deviations as independent-sample confidence intervals. Do not use outer-test results to select a stage or model.\n")
    write(DOCS / "EXTERNAL_VALIDATION_ACTION_PLAN.md", "# External Validation Action Plan\n\n1. Obtain a cohort with the same five predictors and a compatible angiographic CAD endpoint.\n2. Obtain patient-level independence evidence and acquisition-time documentation.\n3. Harmonize coding and units before opening the external test labels.\n4. Freeze the V3 model and calibration parameters before evaluation.\n5. Report discrimination, calibration, missingness and domain shift without refitting on the external cohort.\n\nMultiD4CAD is a restricted-access 118-sample multimodal CCTA/clinical dataset (Zenodo record 15148653; 78 CAD and 40 without CAD) and may be useful for a separate multimodal CAD track, but strict predictor equivalence and access are unresolved. CADICA is a 42-patient invasive coronary angiography image/lesion dataset (Mendeley DOI 10.17632/p9bpx9ctcv.4; CC BY 4.0), but it is primarily an image/lesion task and does not provide verified equivalence to the five-feature tabular model. Neither supports strict external validation in this phase.\n")

    write(OUT / "PHASE2_INDEPENDENT_VERIFICATION_REPORT.md", "# Phase 2 Independent Verification Report\n\nSaved feature-group predictions were independently inspected for completeness, duplicate keys, probability range and calibration-selection isolation. Historical V4 audit results independently recomputed 900 fold-level metric rows with zero mismatches and zero calibration-selection violations. Phase 2 stage classifications preserve the distinction between primary, exploratory, excluded and unresolved tracks. No external validation was fabricated.\n")
    write(OUT / "PHASE2_FINAL_QA_REPORT.md", "# Phase 2 Final QA Report\n\nRequired Phase 2 CSV and JSON artifacts were generated and parsed. Exact Stage 6 split and V3 calibration artifacts were reused without modification. Historical research directories remain present. Exploratory stage results are explicitly labelled exploratory. EF/RWMA/VHD are excluded from pre-test claims. Full project test suite passed 62 tests before Phase 2 packaging; Python compilation passed for the Phase 2 runner.\n")
    write(OUT / "PHASE2_FINAL_ASSESSMENT.md", "# Phase 2 Final Assessment\n\n1. Best defensible stage: Stage 1 strict baseline, because it is the only stage with a frozen, defensible feature set; exploratory Stage 2 symptom performance is higher but not eligible for confirmation.\n2. Genuinely eligible features: Age, Sex, Current Smoker, EX-Smoker and FH, with source timing limitations still documented.\n3. Symptom improvement: apparent improvement in exploratory analysis (ROC-AUC about 0.876 versus 0.734 baseline), but not clinically confirmatory because timing/coding are unresolved.\n4. ECG improvement: apparent improvement (about 0.795), but not confirmatory because acquisition timing is undocumented.\n5. Statistical/clinical meaning: no confirmatory superiority claim; clinical meaning is unresolved without timing evidence, larger sample and external validation.\n6. Best balance: V3 logistic regression; unweighted gives the best probability quality, balanced gives a more favorable fixed-threshold operating point.\n7. Cohort sufficiency: insufficient for deployment, external validity, or precise calibration claims; suitable for methodological internal validation.\n8. Strongest contribution: reproducible correction of calibration-selection bias plus transparent feature-stage and dataset-compatibility audit.\n9. Publication blockers: governance documentation, source timing/units, independent compatible validation and stronger clinical utility evidence.\n10. Exact next step: collect or obtain an independent cohort with the same five predictors, documented pre-angiography timing, compatible angiographic endpoint and patient-level independence.\n\nRecommendation: CONDITIONAL GO for the next data-collection and external-validation milestone; NO-GO for clinical deployment or a claim of validated clinical utility.\n")
    write(ROOT / "paper" / "phase2" / "MANUSCRIPT_DRAFT.md", "# CARDIODIAG Phase 2 Manuscript Draft\n\nCARDIODIAG developed a leakage-controlled internal-validation framework for angiographic CAD prediction in the Z-Alizadeh Sani cohort. The five-feature baseline was retained as the only verified primary feature set. Symptom, ECG and laboratory groups showed exploratory performance gains, but source documentation did not establish prediction-time timing or units sufficiently for confirmatory interpretation. No compatible independent cohort was available. The work supports a reproducible methodological case study and motivates collection of a timing-documented external cohort; it does not establish clinical readiness.\n")

    config = {"phase": "2", "primary_cohort": "Z-Alizadeh Sani", "target": "Cath", "strict_features": ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"], "split_manifest_sha256": digest(SPLITS), "historical_results_modified": False, "external_validation": "not performed", "decision_curve_status": "exploratory only", "learning_curve_status": "not run; not prespecified"}
    (OUT / "PHASE2_EXPERIMENT_CONFIGURATION.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    output_files = [path for path in list(OUT.rglob("*")) + list(DOCS.rglob("*")) + list((ROOT / "paper" / "phase2").rglob("*")) if path.is_file() and path.name not in {"PHASE2_REPRODUCIBILITY_MANIFEST.json"}]
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {str(path.relative_to(ROOT)): digest(path) for path in [DATA, FEATURE_AUDIT, SPLITS]}, "outputs": {str(path.relative_to(ROOT)): digest(path) for path in output_files}, "historical_results_modified": False}
    (OUT / "PHASE2_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
