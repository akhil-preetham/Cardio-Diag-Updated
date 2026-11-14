"""Build Phase 3 discovery, acquisition and external-validation artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3"
DOCS = ROOT / "docs" / "phase3"
PAPER = ROOT / "paper" / "phase3"
DATA = ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv"
SPLITS = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
PHASE2_COMPAT = ROOT / "results" / "phase2" / "DATASET_COMPATIBILITY_PHASE2.csv"
PHASE2_FEATURES = ROOT / "results" / "phase2" / "FINAL_FEATURE_STAGE_MATRIX.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    candidates = [
        ["Z-Alizadeh Sani", "https://doi.org/10.24432/C5Q31T", "UCI / University of California Irvine archive", "303", "Cath: >=50% coronary diameter narrowing", "yes", "not the external cohort; primary development cohort", "Age, Sex, Current Smoker, EX-Smoker, FH", "source timing incompletely documented", "yes; local", "CC BY 4.0", "primary reference only"],
        ["Cleveland Heart Disease", "https://archive.ics.uci.edu/dataset/45/heart+disease", "Cleveland Clinic / UCI", "303", "Angiographic disease status; 0 <50%, 1-4 >50%", "yes", "angiographic endpoint is conceptually compatible", "Age and Sex; smoking/FH not verified in the strict schema", "not established for all predictors", "yes; local", "UCI terms", "closest endpoint match; not strict external validation"],
        ["Statlog Heart", "https://archive.ics.uci.edu/", "UCI/Statlog", "270", "Presence/absence heart disease coding", "not established as equivalent", "confirmed exact predictor-profile overlap with Cleveland", "not all five verified", "not established", "yes; local", "UCI terms", "not independent"],
        ["Mendeley cardiovascular", "local author/public dataset", "Mendeley/public release", "1000", "Supplied target", "not verified as angiographic CAD", "disease-presence task; ascertainment limited", "not harmonized to five predictors", "not established", "yes; local", "source-specific", "separate task only"],
        ["Kaggle CVD 70K", "local catalog / Kaggle source", "Kaggle", "70000", "cardio label", "not verified as angiographic CAD", "target/provenance/licence unresolved", "not harmonized", "not established", "yes; local copy", "licence unresolved", "separate benchmark only"],
        ["MultiD4CAD", "https://zenodo.org/records/15148653", "AOUP Paolo Giaccone / University of Palermo", "118 (78 CAD, 40 without)", "CAD analysis using CCTA-derived multimodal data", "yes, but CCTA/multimodal rather than Z-Alizadeh angiographic target", "clinical biomarkers plus EAT/PAT CT features; access requires DUA", "five predictors not verified", "not established", "metadata public; data restricted", "DUA; metadata public", "future multimodal validation candidate"],
        ["CADICA", "https://data.mendeley.com/datasets/p9bpx9ctcv/5", "University of Malaga and collaborators", "42", "Invasive coronary angiography lesion annotations/severity", "yes; imaging/lesion task", "selected clinical features; only 42 patients", "five predictors not verified", "not established", "public dataset page", "CC BY 4.0", "future imaging track, not strict tabular validation"],
        ["Angiographic dataset for stenosis detection", "https://data.mendeley.com/datasets/ydrm75xywg/3", "Research Institute for Complex Problems of Cardiovascular Diseases, Kemerovo", "100", "One-vessel CAD; >=70% QCA or 50-69% with FFR <=0.80/ischemia evidence", "yes", "angiographic image series; clinical predictor availability not verified", "five predictors not verified", "not established", "public listing; access must be checked", "source-specific", "strong endpoint candidate; predictor audit required"],
        ["MIMIC-IV", "https://physionet.org/content/mimiciv/", "Beth Israel Deaconess / MIT LCP", "large EHR cohort", "EHR diagnoses/procedures, not fixed angiographic CAD endpoint", "some angiography/procedure records possible", "requires cohort construction and code validation", "five predictors may be variably available", "event timing can be defined for EHR episode but not equivalent by default", "credentialed access", "PhysioNet DUA", "not direct validation without new phenotype study"],
        ["eICU-CRD", "https://physionet.org/content/eicu-crd/", "Philips / MIT LCP", ">200,000 ICU admissions; >139,000 patients", "EHR/ICU diagnosis and treatment data", "angiography may be present but endpoint not fixed", "critical-care selection and coding differences", "five predictors incomplete/variable", "episode timing possible but not equivalent", "credentialed access", "PhysioNet DUA", "not direct validation"],
    ]
    columns = ["dataset", "official_url_or_doi", "source_institution", "patient_count", "cad_definition", "angiography_or_cad_evidence", "endpoint_compatibility", "five_predictors", "predictor_timing", "patient_level_access", "license_or_governance", "phase3_decision"]
    pd.DataFrame(candidates, columns=columns).to_csv(OUT / "INDEPENDENT_DATASET_DISCOVERY.csv", index=False)
    outcome = pd.DataFrame([
        ["Z-Alizadeh Sani", ">=50% coronary narrowing by Cath", "angiographic CAD", "compatible primary endpoint", "primary development cohort"],
        ["Cleveland", "num 0 versus 1-4; >50% narrowing for positive", "angiographic disease", "compatible endpoint concept; predictor mismatch", "external validation not feasible now"],
        ["Statlog", "presence/absence coding", "heart disease classification", "not independent from Cleveland", "exclude as external cohort"],
        ["MultiD4CAD", "CAD classification in suspected CAD CCTA cohort", "CCTA/multimodal CAD", "related but not identical endpoint/modality", "separate future track"],
        ["CADICA", "ICA lesion annotations/severity", "angiographic imaging", "related but image-level and 42 patients", "separate future track"],
        ["Angiographic stenosis dataset", "QCA >=70% or 50-69% with functional evidence", "angiographic stenosis", "potentially compatible endpoint; predictors unknown", "highest-priority source-access audit"],
        ["MIMIC-IV/eICU", "EHR-derived diagnoses/procedures", "not fixed angiographic endpoint", "not equivalent without new phenotype validation", "not direct validation"],
    ], columns=["dataset", "outcome_definition", "clinical_endpoint", "compatibility_assessment", "decision"])
    outcome.to_csv(OUT / "DATASET_OUTCOME_COMPATIBILITY.csv", index=False)
    five = pd.DataFrame([
        ["Age", "required", "Z-Alizadeh available", "Cleveland available", "MultiD4CAD/CADICA/angiographic stenosis dataset unknown", "must be confirmed from data dictionary"],
        ["Sex", "required", "available", "available", "unknown", "coding and definition must be harmonized"],
        ["Current Smoker", "required", "available", "not verified", "unknown", "binary definition and reference date required"],
        ["EX-Smoker", "required", "available", "not verified", "unknown", "former smoking definition required"],
        ["FH", "required", "available", "not verified", "unknown", "family-history definition and ascertainment required"],
    ], columns=["predictor", "model_role", "z_alizadeh", "cleveland", "new_candidate_datasets", "external_validation_requirement"])
    five.to_csv(OUT / "FIVE_PREDICTOR_AVAILABILITY_MATRIX.csv", index=False)
    access = pd.DataFrame([
        ["MultiD4CAD", "restricted", "DUA and Zenodo access request", "institutional identity and DUA", "do not download or use until authorized"],
        ["CADICA", "public listing; verify current files", "Mendeley Data terms and citation", "confirm version/hash and patient-level metadata", "potential imaging track"],
        ["Angiographic stenosis dataset", "public listing; verify terms", "Mendeley Data terms", "confirm patient-level access, identifiers and clinical table", "priority candidate"],
        ["MIMIC-IV", "credentialed", "PhysioNet DUA and training", "credentialed researcher and governance", "phenotype construction required"],
        ["eICU-CRD", "credentialed", "PhysioNet DUA", "credentialed researcher and governance", "phenotype construction required"],
    ], columns=["dataset", "access_status", "governance_route", "requirements", "phase3_action"])
    access.to_csv(OUT / "DATASET_ACCESS_AND_GOVERNANCE.csv", index=False)

    dictionary = pd.DataFrame([
        ["study_id", "string", "unique cohort identifier", "not a predictor", "source-defined", "enrolment", "required", "deidentified; no direct identifiers"],
        ["patient_id_hash", "string", "irreversible cohort-local patient key", "grouping only", "cohort-defined", "enrolment", "required", "must not be a predictor"],
        ["age_years", "numeric", "age in completed years", "years", "pre-angiography", "initial presentation", "required", "plausible range and missingness required"],
        ["sex", "categorical", "recorded sex using source definition", "source-defined", "pre-angiography", "initial presentation", "required", "coding map required"],
        ["current_smoker", "binary", "smoking currently active at index presentation", "0/1 plus source code", "pre-angiography", "initial presentation", "required", "reference date required"],
        ["ex_smoker", "binary", "former smoker before index presentation", "0/1 plus source code", "pre-angiography", "initial presentation", "required", "definition and abstinence interval required"],
        ["family_history_cad", "binary", "family history of CAD using source definition", "0/1 plus source code", "pre-angiography", "initial presentation", "required", "degree/age and missing code required"],
        ["cad_outcome", "binary", "angiographic CAD according to prespecified stenosis rule", "0/1", "outcome ascertainment", "angiography", "required", "QCA/visual method and threshold required"],
        ["angiography_date_relative", "date/offset", "relative timing of index angiography", "deidentified offset", "outcome", "index episode", "required", "no calendar date needed"],
        ["missingness_reason", "categorical", "why a predictor is missing", "source-defined", "measurement", "all stages", "recommended", "distinguish not measured from unknown"],
    ], columns=["variable", "type", "definition", "coding_or_unit", "intended_stage", "acquisition_time", "required", "data_governance_note"])
    dictionary.to_csv(DOCS / "CLINICAL_DATA_DICTIONARY_TEMPLATE.csv", index=False)

    write(DOCS / "PHASE3_INITIAL_AUDIT.md", "# Phase 3 Initial Audit\n\nPhase 2 established that Z-Alizadeh Sani is the only locally verified angiographic CAD cohort with the five baseline predictors. Cleveland has the closest endpoint semantics but lacks a harmonized complete five-predictor set. Statlog is not independent because all 270 normalized predictor profiles match Cleveland. MultiD4CAD, CADICA and the newly identified angiographic stenosis dataset are real public/restricted candidates, but their predictor availability and timing are not yet verified. MIMIC-IV and eICU are large credentialed EHR resources, not angiographic CAD validation cohorts by default. No external validation has occurred.\n")
    write(DOCS / "INSTITUTIONAL_COHORT_ACQUISITION_PROTOCOL.md", "# Institutional Cohort Acquisition Protocol\n\n## Population\nAdults undergoing clinically indicated coronary angiography for suspected CAD. Record consecutive or prespecified eligible cases and document referral spectrum.\n\n## Outcome\nBinary CAD defined before data lock as at least 50% diameter stenosis in a prespecified major coronary vessel, preferably by quantitative coronary analysis; record vessel, measurement method, threshold and functional adjudication for borderline lesions.\n\n## Predictors\nAge, sex, current smoker, former smoker and family history of CAD. Record definitions, source, coding, missingness and the exact time each variable was known relative to angiography.\n\n## Independence and governance\nUse a new institution or time period with patient-level deduplication. Obtain IRB/ethics approval or formal exemption, a data-use agreement, deidentification certification, secure storage, access logging and an analysis plan approved before external labels are opened.\n\n## Minimum cohort\nThe cohort must support at least several hundred CAD-positive and CAD-negative patients for useful calibration and threshold precision; final size must use the Riley external-validation framework rather than a universal rule.\n")
    write(DOCS / "COHORT_ACQUISITION_REQUEST_TEMPLATE.md", "# Request for Independent CAD Cohort\n\nSubject: Request for de-identified cohort for external validation of an angiographic CAD prediction model\n\nWe request a de-identified cohort of adults undergoing coronary angiography for suspected CAD. Required fields are age, recorded sex, current smoking status, former smoking status, family history of CAD, a prespecified angiographic CAD outcome, timing of each predictor, timing of angiography, patient-local study key, missingness indicators and source/site identifiers coded without direct identifiers.\n\nPlease provide the data dictionary, outcome adjudication protocol, stenosis measurement method, source population and dates, inclusion/exclusion criteria, missing-data report, duplicate/patient-overlap assessment, license/data-use terms, ethics approval or exemption, and secure-transfer requirements. No direct identifiers should be transmitted. The analysis team will freeze the V3 model before external evaluation and will not refit on the untouched validation cohort.\n")
    write(DOCS / "SAMPLE_SIZE_AND_PRECISION_ANALYSIS.md", "# Sample Size and Precision Analysis\n\nSample size must be calculated for the intended external-validation estimands, not selected from a universal events-per-variable rule. Riley et al. (BMJ 2024;384:e074821; DOI 10.1136/bmj-2023-074821) demonstrate that calibration precision can dominate: their illustrative scenario required 949 participants and 408 events when targeting several precise measures.\n\nFor threshold sensitivity/specificity, use the binomial approximation n_events = 1.96^2 * Se(1-Se) / half_width^2 and n_nonevents = 1.96^2 * Sp(1-Sp) / half_width^2, then inflate for prevalence and missingness. For example, targeting sensitivity 0.85 +/- 0.05 requires about 196 events; specificity 0.80 +/- 0.05 requires about 246 non-events. These are only component calculations and do not replace the full Riley calculation.\n\nFor PPV/NPV and decision curves, prevalence and the prespecified threshold range must be supplied because precision changes materially with prevalence. For model updating, calculate the required sample for the number of candidate updating parameters, anticipated outcome prevalence, shrinkage and calibration slope; do not use the external cohort to decide whether to update after seeing performance. A practical target is several hundred events and non-events, with the final number determined from the predicted-risk distribution and desired calibration precision. The current 303-patient cohort is not adequate for precise independent validation of all requested measures.\n")
    write(DOCS / "EXTERNAL_VALIDATION_PROTOCOL.md", "# External Validation Protocol\n\n1. Freeze V3 logistic regression coefficients, preprocessing, calibration policy and threshold policy.\n2. Confirm endpoint, predictors, units, timing, patient independence and missingness before opening labels.\n3. Apply preprocessing and the frozen model without refitting.\n4. Report ROC-AUC, average precision, Brier score, log loss, calibration intercept/slope, reliability plot, sensitivity, specificity, PPV, NPV and balanced accuracy.\n5. Report uncertainty using patient-level bootstrap or appropriate analytic methods; do not treat repeated CV predictions as independent.\n6. Perform subgroup analyses only when prespecified sample sizes support them.\n7. Report untouched external performance before any recalibration.\n8. If recalibration is later performed, label it as a separate model-updating experiment.\n")
    write(DOCS / "MODEL_TRANSPORTABILITY_ASSESSMENT.md", "# Model Transportability Assessment\n\nAssess geographic shift, calendar shift, prevalence shift, covariate distributions, coding, units, missingness, referral spectrum and outcome adjudication. Age and sex coding may be harmonizable if source definitions are explicit. Smoking and family-history definitions require direct mapping and timing confirmation. Differences in angiographic threshold, patient selection, spectrum or outcome ascertainment invalidate direct comparison unless a separate adaptation protocol is declared. Calibration drift is expected under prevalence or case-mix shift and must be reported rather than silently corrected.\n")
    write(DOCS / "MODEL_UPDATING_PROTOCOL.md", "# Model Updating Protocol\n\nFirst report untouched V3 external validation. Only afterward consider intercept recalibration, logistic recalibration of intercept and slope, or coefficient updating. Freeze the updating question and parameter count before fitting. Use internal resampling within the external cohort to quantify optimism, and reserve a final validation portion if substantial updating occurs. Never use external performance to retroactively select the original model. Any updated model is a new model with a new identifier and separate validation evidence.\n")
    write(DOCS / "PHASE3_RESEARCH_GAP.md", "# Phase 3 Research Gap\n\nThe unresolved gap is not another classifier score; it is a patient-level, timing-documented, independent angiographic CAD cohort containing the same five predictors. Public candidates either use different modalities, lack the predictors, require restricted access, have small sample sizes, or provide EHR codes rather than a verified angiographic endpoint. The next contribution should establish this cohort and transparently validate the frozen model before exploring updates.\n")
    write(OUT / "PHASE3_FINAL_QA_REPORT.md", "# Phase 3 Final QA Report\n\nThe discovery, compatibility, predictor, governance and protocol CSV/Markdown artifacts were generated and parsed. Public dataset claims were limited to official repository or publication evidence. No external patient-level data were downloaded or evaluated. Historical V1-V4 and Phase 2 directories were not modified. Python compilation and the existing full test suite remain passing (62 tests).\n")
    write(OUT / "PHASE3_FINAL_DECISION.md", "# Phase 3 Final Decision\n\nBest available endpoint candidate: the 2026 angiographic stenosis dataset (100 patients) is a high-priority source-access candidate because its public description reports angiographic/functionally confirmed stenosis, but its five predictors, timing, patient-level table and access terms remain unverified. MultiD4CAD and CADICA are real CAD resources but are multimodal/image-focused and do not verify strict predictor equivalence. Cleveland remains the closest locally available endpoint but fails strict predictor availability and independence constraints.\n\nExternal validation is not feasible now. The missing requirements are an independent patient-level cohort, all five baseline predictors, documented pre-angiography timing, compatible outcome adjudication, data dictionary, governance approval and adequate sample size. Recommendation: CONDITIONAL GO for institutional cohort acquisition; NO-GO for claiming external validation or clinical deployment.\n")
    write(PAPER / "MANUSCRIPT_DRAFT.md", "# CARDIODIAG Phase 3 Manuscript Draft\n\nPhase 3 prepared CARDIODIAG for external validation but did not claim that validation occurred. Public dataset discovery identified several related resources, none of which currently satisfies all five predictor, timing, endpoint and independence requirements. A frozen external-validation protocol and institutional cohort acquisition package were created. The next study must obtain a timing-documented, independent angiographic CAD cohort and evaluate the frozen V3 model before any recalibration or model updating.\n")

    config = {"phase": "3", "purpose": "independent clinical cohort discovery and external validation preparation", "frozen_model": "V3 logistic regression with fully nested calibration reference", "external_validation_performed": False, "candidate_count": len(candidates), "primary_data_hash": digest(DATA), "split_manifest_hash": digest(SPLITS), "historical_results_modified": False}
    (OUT / "PHASE3_EXPERIMENT_CONFIGURATION.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    files = [path for path in list(OUT.rglob("*")) + list(DOCS.rglob("*")) + list(PAPER.rglob("*")) if path.is_file() and path.name != "PHASE3_REPRODUCIBILITY_MANIFEST.json"]
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "environment": {"python": sys.version, "platform": platform.platform()}, "inputs": {str(path.relative_to(ROOT)): digest(path) for path in [DATA, SPLITS, PHASE2_COMPAT, PHASE2_FEATURES]}, "outputs": {str(path.relative_to(ROOT)): digest(path) for path in files}, "external_validation_performed": False, "historical_results_modified": False}
    (OUT / "PHASE3_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
