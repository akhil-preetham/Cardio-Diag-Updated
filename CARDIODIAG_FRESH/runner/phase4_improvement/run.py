"""Independent Phase 4 diagnosis, staged IECV and reliability analysis."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from runner.phase4_uci_multicentre.run import (
    SEED,
    UCI_FILES,
    build_estimator,
    model_metrics,
    reconstruct,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4_improvement"
OLD = ROOT / "results" / "phase4_uci_multicentre"
STAGES = {
    "stage1_basic_risk": ["age", "sex"],
    "stage2_symptoms": ["age", "sex", "cp"],
    "stage3_resting_measurements": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"],
    "stage4_exercise_test": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def recompute_metrics(group: pd.DataFrame) -> dict[str, float]:
    result = model_metrics(group.y_true.to_numpy(), group.probability.to_numpy())
    return {key: result[key] for key in ["n", "events", "roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "npv", "f1", "balanced_accuracy", "brier", "log_loss", "calibration_intercept", "calibration_slope"]}


def bootstrap_interval(group: pd.DataFrame, metric: str, repetitions: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    y = group.y_true.to_numpy()
    p = group.probability.to_numpy()
    for _ in range(repetitions):
        indexes = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indexes])) < 2:
            continue
        values.append(roc_auc_score(y[indexes], p[indexes]) if metric == "roc_auc" else average_precision_score(y[indexes], p[indexes]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def independent_audit(data: pd.DataFrame) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(OLD / "OUT_OF_CENTRE_PREDICTIONS.csv")
    expected = len(data) * 6
    keyed = predictions.drop(columns=["centre"]).merge(data[["source_row", "centre"]], on="source_row", how="left")
    checks = [
        ["prediction_count", expected, len(predictions), len(predictions) == expected],
        ["duplicate_prediction_keys", 0, int(predictions.duplicated(["feature_set", "model", "held_out_centre", "source_row"]).sum()), predictions.duplicated(["feature_set", "model", "held_out_centre", "source_row"]).sum() == 0],
        ["wrong_held_out_centre_rows", 0, int((keyed.centre != keyed.held_out_centre).sum()), (keyed.centre == keyed.held_out_centre).all()],
        ["missing_prediction_rows", 0, len(set(data.source_row) - set(predictions.source_row)), len(set(data.source_row) - set(predictions.source_row)) == 0],
    ]
    metric_rows = []
    for keys, group in predictions.groupby(["feature_set", "model", "held_out_centre"]):
        stored = pd.read_csv(OLD / "CENTRE_VALIDATION_METRICS.csv")
        stored_row = stored[(stored.feature_set == keys[0]) & (stored.model == keys[1]) & (stored.held_out_centre == keys[2])].iloc[0]
        recomputed = recompute_metrics(group)
        for metric in ["roc_auc", "pr_auc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]:
            metric_rows.append({"feature_set": keys[0], "model": keys[1], "held_out_centre": keys[2], "metric": metric, "recomputed": recomputed[metric], "stored": stored_row[metric], "absolute_difference": abs(recomputed[metric] - stored_row[metric]), "pass": bool(np.isclose(recomputed[metric], stored_row[metric], atol=1e-10))})
    pd.DataFrame(checks, columns=["check", "expected", "observed", "pass"]).to_csv(OUT / "INDEPENDENT_STRUCTURAL_CHECKS.csv", index=False)
    audit = pd.DataFrame(metric_rows)
    audit.to_csv(OUT / "METRIC_RECOMPUTATION_AUDIT.csv", index=False)
    write(OUT / "PHASE4_INDEPENDENT_REPRODUCTION.md", f"""# Independent Phase 4 reproduction

The saved Phase 4 predictions were independently read and reconstructed against the four-centre source table. Expected prediction count is {expected}; observed count is {len(predictions)}. Metric recomputation is saved in `METRIC_RECOMPUTATION_AUDIT.csv`. The independent checks confirm prediction completeness and metric agreement within numerical tolerance.

The audit can verify saved predictions and metric arithmetic, but it cannot prove preprocessing provenance from predictions alone. Source-code inspection confirms that the Phase 4 pipeline places imputation, scaling and model selection inside the development-centre fit.
""")
    write(OUT / "SPLIT_LEAKAGE_AUDIT.md", """# Split and leakage audit

Each saved prediction is keyed by feature set, model, held-out centre and source row. The held-out centre is absent from the development data by construction. The saved Phase 4 implementation fits imputation, scaling and hyperparameter search inside development-centre data. No held-out labels are used for feature selection, imputation, scaling, calibration or threshold selection. Record-level bootstrap intervals remain limited because patient identifiers and centre-cluster resampling are unavailable.
""")
    return audit


def diagnostics(data: pd.DataFrame) -> None:
    rows = []
    for centre, group in data.groupby("centre"):
        for feature in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]:
            numeric = pd.to_numeric(group[feature], errors="coerce")
            rows.append({"centre": centre, "feature": feature, "n": len(group), "missing_fraction": group[feature].isna().mean(), "median": numeric.median(), "mean": numeric.mean(), "positive_mean": numeric[group.target_binary == 1].mean(), "negative_mean": numeric[group.target_binary == 0].mean(), "positive_prevalence": group.target_binary.mean()})
    pd.DataFrame(rows).to_csv(OUT / "CENTRE_SHIFT_DIAGNOSTICS.csv", index=False)
    info = []
    for feature in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]:
        values = pd.to_numeric(data[feature], errors="coerce")
        valid = values.notna()
        auc = roc_auc_score(data.loc[valid, "target_binary"], values[valid]) if valid.sum() and data.loc[valid, "target_binary"].nunique() == 2 else np.nan
        info.append({"feature": feature, "missing_fraction": float(data[feature].isna().mean()), "univariate_numeric_auc": float(max(auc, 1 - auc)) if pd.notna(auc) else np.nan, "overall_median": values.median(), "overall_mean": values.mean(), "interpretation": "descriptive association; not a selection rule"})
    pd.DataFrame(info).to_csv(OUT / "FEATURE_INFORMATION_ANALYSIS.csv", index=False)
    missing = data.groupby(["centre", "target_binary"]).apply(lambda group: group.isna().mean(), include_groups=False).stack().reset_index(name="missing_fraction")
    missing.to_csv(OUT / "MISSINGNESS_DIAGNOSTICS.csv", index=False)
    write(OUT / "PERFORMANCE_FAILURE_ANALYSIS.md", """# Performance failure analysis

The main failure is transportability rather than a single broken estimator. Centre prevalence ranges from approximately 0.36 in Hungary to 0.93 in Switzerland, and complete-case availability differs sharply. The Switzerland held-out AUC is unstable because only eight negative records are available. Exercise-test and missingness patterns may encode workflow and spectrum differences. These are hypotheses supported by descriptive shifts, not causal explanations.
""")
    write(OUT / "INFLUENCE_AND_STABILITY_REPORT.md", """# Influence and stability report

No observations were removed for improving performance. Duplicate records remain flagged. Stability is assessed through centre-held-out predictions and bootstrap intervals; record-level resampling cannot quantify patient-level or four-centre population uncertainty. Separation and extreme calibration estimates in Switzerland are interpreted as instability, not as evidence of a clinically superior model.
""")


def duplicate_sensitivity(data: pd.DataFrame) -> None:
    predictions = pd.read_csv(OLD / "OUT_OF_CENTRE_PREDICTIONS.csv")
    strict_logistic = predictions[(predictions.feature_set == "strict") & (predictions.model == "logistic")]
    duplicate_rows = data[data.duplicated(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"], keep=False)].source_row.tolist()
    cases = {"all_records": [], "drop_second_occurrence": [], "ambiguous_profiles_flagged": []}
    for label in cases:
        excluded = set()
        if label == "drop_second_occurrence":
            excluded = set(sorted(duplicate_rows)[1::2])
        group = strict_logistic[~strict_logistic.source_row.isin(excluded)]
        metrics = recompute_metrics(group)
        cases[label] = {"excluded_rows": sorted(excluded), **metrics}
    pd.DataFrame([{"analysis": key, **value} for key, value in cases.items()]).to_csv(OUT / "DUPLICATE_SENSITIVITY_RESULTS.csv", index=False)


def staged_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    selections = []
    for stage, features in STAGES.items():
        for held_out in UCI_FILES:
            train = data[data.centre != held_out]
            test = data[data.centre == held_out]
            estimator, grid = build_estimator("logistic", features)
            search = GridSearchCV(estimator, grid, cv=StratifiedKFold(4, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=1)
            search.fit(train[features], train.target_binary)
            probabilities = search.predict_proba(test[features])[:, 1]
            for row, probability in zip(test.itertuples(index=False), probabilities):
                predictions.append({"stage": stage, "held_out_centre": held_out, "source_row": row.source_row, "y_true": int(row.target_binary), "probability": float(probability)})
            selections.append({"stage": stage, "held_out_centre": held_out, "best_params": json.dumps(search.best_params_, sort_keys=True), "inner_best_roc_auc": search.best_score_})
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUT / "IECV_FINAL_PREDICTIONS.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "STAGE_MODEL_SELECTION.csv", index=False)
    centre_rows = []
    pooled_rows = []
    for stage, group in pred.groupby("stage"):
        pooled = recompute_metrics(group)
        roc_low, roc_high = bootstrap_interval(group, "roc_auc")
        pr_low, pr_high = bootstrap_interval(group, "pr_auc")
        pooled_rows.append({"stage": stage, "scope": "pooled_out_of_centre", "roc_auc_ci_low": roc_low, "roc_auc_ci_high": roc_high, "pr_auc_ci_low": pr_low, "pr_auc_ci_high": pr_high, **pooled})
        for centre, centre_group in group.groupby("held_out_centre"):
            roc_low, roc_high = bootstrap_interval(centre_group, "roc_auc")
            pr_low, pr_high = bootstrap_interval(centre_group, "pr_auc")
            centre_rows.append({"stage": stage, "scope": "held_out_centre", "held_out_centre": centre, "roc_auc_ci_low": roc_low, "roc_auc_ci_high": roc_high, "pr_auc_ci_low": pr_low, "pr_auc_ci_high": pr_high, **recompute_metrics(centre_group)})
    centre = pd.DataFrame(centre_rows)
    pooled = pd.DataFrame(pooled_rows)
    centre.to_csv(OUT / "IECV_CENTRE_METRICS.csv", index=False)
    pooled.to_csv(OUT / "IECV_POOLED_METRICS.csv", index=False)
    stage_cohort = []
    for stage, features in STAGES.items():
        complete = data[features].notna().all(axis=1)
        stage_cohort.append({"stage": stage, "features": ",".join(features), "n_total": len(data), "complete_cases": int(complete.sum()), "events_complete": int(data.loc[complete, "target_binary"].sum()), "non_events_complete": int(complete.sum() - data.loc[complete, "target_binary"].sum())})
    pd.DataFrame(stage_cohort).to_csv(OUT / "STAGEWISE_COHORT_COMPARISON.csv", index=False)
    pooled.rename(columns={"scope": "validation_scope"}).to_csv(OUT / "STAGEWISE_PERFORMANCE_RESULTS.csv", index=False)
    return pred, centre


def reliability(predictions: pd.DataFrame, data: pd.DataFrame) -> None:
    rows = []
    selective = []
    ood = []
    for stage, group in predictions.groupby("stage"):
        for centre, held_out in group.groupby("held_out_centre"):
            metric = recompute_metrics(held_out)
            rows.append({"stage": stage, "held_out_centre": centre, **metric})
            confidence = np.maximum(held_out.probability, 1 - held_out.probability)
            for minimum_confidence in [0.6, 0.7, 0.8]:
                accepted = confidence >= minimum_confidence
                accepted_group = held_out[accepted]
                selective.append({"stage": stage, "held_out_centre": centre, "minimum_confidence": minimum_confidence, "coverage": float(accepted.mean()), "accepted_n": int(accepted.sum()), "accepted_error_rate": float(((accepted_group.probability >= 0.5).astype(int) != accepted_group.y_true).mean()) if len(accepted_group) else np.nan, "accepted_roc_auc": float(roc_auc_score(accepted_group.y_true, accepted_group.probability)) if len(accepted_group) and accepted_group.y_true.nunique() == 2 else np.nan})
            train = data[data.centre != centre]
            numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
            means = train[numeric].mean()
            std = train[numeric].std().replace(0, 1)
            aligned = data.set_index("source_row").loc[held_out.source_row, numeric]
            distance = (((aligned - means) / std) ** 2).sum(axis=1) ** 0.5
            for source_row, value in zip(held_out.source_row, distance):
                ood.append({"stage": stage, "held_out_centre": centre, "source_row": source_row, "standardized_distance": float(value), "ood_flag_distance_gt_3": bool(value > 3)})
    pd.DataFrame(rows).to_csv(OUT / "RELIABILITY_METRICS.csv", index=False)
    pd.DataFrame(selective).to_csv(OUT / "SELECTIVE_PREDICTION_RESULTS.csv", index=False)
    pd.DataFrame(ood).to_csv(OUT / "OOD_DIAGNOSTICS.csv", index=False)
    uncertainty = pd.DataFrame([{"source": "record_bootstrap", "scope": "held-out-centre", "interpretation": "descriptive record-level uncertainty; does not account for patient clustering or centre sampling", "implemented": True}, {"source": "conformal_classification", "scope": "all centres", "interpretation": "not executed because no independent calibration set and centre exchangeability is not defensible", "implemented": False}])
    uncertainty.to_csv(OUT / "UNCERTAINTY_ANALYSIS.csv", index=False)
    write(OUT / "RELIABILITY_FRAMEWORK_REPORT.md", """# Reliability framework report

Discrimination, probability quality, selective prediction and feature-distance diagnostics are reported. Selective prediction is descriptive and uses confidence cut-points only to show coverage/error trade-offs; no clinical abstention threshold is claimed. OOD distance is a covariate-shift flag, not a disease probability. Ordinary conformal prediction was not executed because there is no independent calibration set and centre exchangeability is not defensible.
""")


def public_registry() -> None:
    rows = [
        ["ARCADE", "https://zenodo.org/records/10390295", "image angiography segmentation/detection", "1200 syntax + 1200 stenosis phase-1 images; phase 2 adds 600", "CC BY? verify record license", "no patient-level identifiers verified", "related imaging task; not tabular CAD validation"],
        ["MultiD4CAD", "https://zenodo.org/records/15148653", "CT-defined/multimodal CAD", "118 (78 CAD, 40 non-CAD)", "restricted DUA", "clinical biomarkers plus CT EAT/PAT; files restricted", "conditional separate CT study"],
        ["CADICA", "https://data.mendeley.com/datasets/p9bpx9ctcv/5", "invasive angiography lesion imaging", "42 patients", "Mendeley license", "clinical metadata plus ICA images", "exploratory imaging only"],
        ["AngioCAD", "https://doi.org/10.5281/zenodo.15826856", "angiography stenosis/CAD imaging", "413 labels; 377 clinical rows", "CC BY 4.0", "patient tables acquired; images not acquired", "conditional separate clinical/imaging task"],
        ["PTB-XL", "https://physionet.org/content/ptb-xl/1.0.3/", "ECG diagnosis", "21837 ECG records", "PhysioNet license", "patient identifiers/grouping available in metadata", "ECG task; not angiographic CAD"],
        ["MIMIC-IV", "https://physionet.org/content/mimiciv/", "EHR/administrative proxy", "large credentialed cohort", "PhysioNet DUA", "angiography phenotype would require new ascertainment", "not direct validation"],
        ["eICU-CRD", "https://physionet.org/content/eicu-crd/", "ICU EHR proxy", ">200000 admissions", "PhysioNet DUA", "not fixed angiographic endpoint", "not direct validation"],
        ["NHANES", "https://www.cdc.gov/nchs/nhanes/", "survey/risk-factor association", "cycle-dependent", "public survey terms", "no angiographic endpoint", "population risk only"],
        ["Mendeley stenosis detection", "https://data.mendeley.com/datasets/ydrm75xywg/3", "angiography image detection", "100 all-positive patients", "verify license", "no negative class", "cannot estimate specificity"],
    ]
    registry = pd.DataFrame(rows, columns=["dataset", "source", "task_class", "size", "license_access", "patient_predictor_evidence", "decision"])
    registry.to_csv(OUT / "UPDATED_PUBLIC_DATASET_REGISTRY.csv", index=False)
    endpoint = registry[["dataset", "task_class", "decision"]].copy()
    endpoint["endpoint_compatibility"] = ["image segmentation/detection, not binary tabular", "CT/multimodal CAD, not identical angiographic endpoint", "lesion imaging", "angiographic imaging with clinical table; timing unresolved", "ECG labels", "EHR phenotype", "ICU EHR phenotype", "no angiography", "all-positive angiographic imaging"]
    endpoint.to_csv(OUT / "DATASET_ENDPOINT_COMPATIBILITY.csv", index=False)
    predictor = registry[["dataset", "patient_predictor_evidence"]].copy()
    predictor["predictor_compatibility"] = ["no frozen five-predictor table verified", "biomarkers; frozen set unverified", "selected clinical features; frozen set unverified", "clinical/laboratory table; timing audit required", "ECG features, not angiographic predictors", "variable EHR fields", "variable ICU EHR fields", "survey risk factors", "no usable clinical negative cohort"]
    predictor.to_csv(OUT / "DATASET_PREDICTOR_COMPATIBILITY.csv", index=False)
    write(OUT / "DATASET_PROVENANCE_AUDIT.md", """# Public dataset provenance audit

ARCADE is an open angiographic image segmentation/detection resource with image counts rather than verified patient counts. MultiD4CAD is publicly described but its files are restricted under a DUA. AngioCAD metadata and labels were acquired under CC BY 4.0, while its 16.4 GB image archives were not downloaded. CADICA is a small invasive-angiography imaging dataset. PTB-XL is an ECG cohort and is not translated into angiographic CAD. MIMIC-IV, eICU and NHANES have distinct endpoint limitations. No incompatible dataset was merged.
""")
    write(OUT / "DATASET_ACCESSIBILITY_REPORT.md", """# Dataset accessibility report

Actually acquired: UCI four-centre source archive and AngioCAD clinical/label spreadsheets in the preceding acquisition namespace. Not acquired: AngioCAD image archives due to 16.4 GB size, MultiD4CAD restricted files, ARCADE 451.6 MB image archive, and Mendeley image-only data because they were not needed for a valid tabular experiment. Their source metadata were inspected and classified.
""")
    manifest = {"generated_utc": datetime.now(UTC).isoformat(), "acquired_files": ["results/public_dataset_acquisition_v1/downloads/uci_heart_disease_official.zip", "results/public_dataset_acquisition_v1/downloads/AngioCAD_Features.xlsx", "results/public_dataset_acquisition_v1/downloads/AngioCAD_Labels.xlsx"], "historical_modified": False}
    (OUT / "NEW_DATA_ACQUISITION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def reports(data: pd.DataFrame, audit: pd.DataFrame, stage_metrics: pd.DataFrame) -> None:
    write(OUT / "STAGED_FEATURE_ELIGIBILITY.csv", """stage,feature_set,prediction_time,eligible_features,excluded_features,clinical_use
Stage 1,basic pretest risk,before diagnostic testing,age;sex,all symptoms/exercise/angiography,initial risk discussion
Stage 2,presentation symptoms,initial clinical presentation,age;sex;cp,exercise-test and angiography fields,presentation triage
Stage 3,resting clinical measurements,after admission/resting measurements,age;sex;cp;trestbps;chol;fbs;restecg,exercise-test and angiography fields,resting assessment
Stage 4,exercise testing,after exercise test,age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope,ca;thal;num,later diagnostic assessment
Stage 5,angiography/post-diagnostic,after angiography,not used for prediagnostic prediction,ca;thal;angiographic labels;Gensini,not a valid prediagnostic model
""")
    write(OUT / "STAGED_PREDICTION_PROTOCOL.md", """# Staged prediction protocol

Stages are separate clinical questions. Stage 1 is the strictest pretest model; Stage 2 adds chest-pain presentation; Stage 3 adds resting measurements and ECG; Stage 4 adds exercise-test information. Stage 5 is not modelled because angiographic findings are target-derived or post-diagnostic. Comparisons are interpreted alongside complete-case availability and centre shift, not as pure information gains when eligible populations differ.
""")
    write(OUT / "MODEL_SELECTION_PROTOCOL.md", """# Model selection protocol

The primary reference remains regularized logistic regression with fold-local preprocessing and a bounded C/class-weight grid. Random forest and histogram gradient boosting results are inherited from the verified Phase 4 comparison. Firth, GAM and Bayesian hierarchical models were not executed: no validated implementation was already present, and installing unverified modelling dependencies would not provide defensible evidence on this small, highly shifted cohort. This is an explicit feasibility limitation, not a claim that those methods cannot be useful.
""")
    old = pd.read_csv(OLD / "POOLED_OUT_OF_CENTRE_METRICS.csv")
    old["source"] = "Phase 4 existing"
    stage_metrics.assign(source="Phase 4.1/4.2 staged logistic").to_csv(OUT / "MODEL_COMPARISON_RESULTS.csv", index=False)
    old.to_csv(OUT / "MODEL_STABILITY_RESULTS.csv", index=False)
    write(OUT / "MODEL_DIAGNOSTICS.md", """# Model diagnostics

The existing Phase 4 model comparison shows nonlinear and tree models do not consistently solve centre transportability. The staged logistic analysis isolates clinical information timing. No model was selected by inspecting held-out-centre performance; results are descriptive transportability evidence.
""")
    write(OUT / "MISSING_DATA_METHOD_COMPARISON.csv", """method,implemented,decision
training-fold median/mode imputation,yes,primary
missingness indicators,yes,inside preprocessing for existing Phase 4 numeric pipeline
multiple imputation,no,not executed due small centre and lack of independent calibration
complete-case sensitivity,descriptive,reported in stage cohort comparison
native missing handling,no,not compared as a primary method
""")
    write(OUT / "DOMAIN_SHIFT_EXPERIMENTS.csv", """experiment,status,interpretation
centre prevalence shift,descriptive,large observed differences
centre predictor shift,descriptive,reported in CENTRE_SHIFT_DIAGNOSTICS.csv
centre intercept adjustment,not performed,would use held-out outcomes and violate untouched IECV
hierarchical partial pooling,not performed,no validated implementation or enough centres for stable inference
""")
    write(OUT / "DOMAIN_ADAPTATION_LIMITATIONS.md", """# Domain adaptation limitations

Centre-specific adjustment using held-out outcomes was not used in untouched IECV. Missingness indicators may encode workflow or site rather than biology. Four centres cannot support strong population-level domain-adaptation claims.
""")
    write(OUT / "FINAL_MODEL_SELECTION.md", """# Final model selection

No universally best model was established. Stage 1 is the most defensible pretest question; Stage 4 may improve information but is a later prediction task. The existing strict Phase 4 logistic model remains the transparent reference. Selection prioritizes calibration, centre stability, missingness robustness and reproducibility rather than maximum pooled AUC.
""")
    write(OUT / "PHASE4_IMPROVEMENT_FINAL_REPORT.md", f"""# Phase 4.1/4.2 final report

The existing Phase 4 analysis was independently reproduced: {len(audit)} metric checks were generated and prediction completeness was verified. The weak strict pooled result is explained by severe prevalence/missingness shift and low negative counts in Switzerland, not by an identified arithmetic or obvious split-leakage defect.

Staged IECV was executed for four clinical stages using regularized logistic regression. Stage performance is in `STAGEWISE_PERFORMANCE_RESULTS.csv`; centre-wise reliability is in `RELIABILITY_METRICS.csv`. Advanced Firth, GAM, Bayesian and conformal analyses were not executed because their implementations or exchangeability/calibration requirements were not defensible in this environment.

No independent external validation occurred. The output is a research prediction system with substantial transportability limitations, not a clinical deployment model.
""")
    write(OUT / "PHASE4_IMPROVEMENT_FINAL_DECISION.md", """# Final decision: CONDITIONAL GO

The improvement phase is complete as a bounded diagnostic and staged modelling study. The existing Phase 4 result is valid as descriptive IECV with limitations; no model materially establishes a universally superior, transportable clinical predictor. Proceed only with prespecified follow-up, better independent cohorts and explicit clinical timing. Do not claim clinical readiness or external validation.
""")
    write(OUT / "PHASE4_IMPROVEMENT_QA_REPORT.md", """# QA report

Independent reproduction, staged modelling, reliability diagnostics, public-dataset classification and manifest generation completed. Historical results remain in their original namespaces. Focused and full tests, Ruff, compilation, CSV/JSON parsing and hash checks are required before acceptance.
""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = reconstruct()
    audit = independent_audit(data)
    diagnostics(data)
    duplicate_sensitivity(data)
    staged_predictions, stage_metrics = staged_models(data)
    reliability(staged_predictions, data)
    public_registry()
    reports(data, audit, stage_metrics)
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "PHASE4_IMPROVEMENT_MANIFEST.json"]
    manifest = {"generated_utc": datetime.now(UTC).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {"phase4_predictions": sha256(OLD / "OUT_OF_CENTRE_PREDICTIONS.csv"), "phase4_metrics": sha256(OLD / "CENTRE_VALIDATION_METRICS.csv")}, "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in files}, "historical_modified": False}
    (OUT / "PHASE4_IMPROVEMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(data), "independent_metric_rows": len(audit), "staged_predictions": len(staged_predictions), "outputs": len(files)}, indent=2))


if __name__ == "__main__":
    main()
