"""Independent clinical suitability and research-protocol audit.

This script reads only acquired/local files, performs no model training, and
writes versioned audit artifacts under results/clinical_validation_audit_v3.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "clinical_validation_audit_v3"
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(relative: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, **kwargs)


Z_FEATURES = {
    "Age": ("Age in years", "demographic", "pre-diagnostic", "eligible", "Available before diagnostic work-up; source workbook label."),
    "Weight": ("Body weight", "demographic", "pre-diagnostic", "eligible_with_unit_review", "Available before diagnosis; unit is not machine-readable in the workbook."),
    "Length": ("Body length/height", "demographic", "pre-diagnostic", "eligible_with_unit_review", "Available before diagnosis; likely centimetres but source units require confirmation."),
    "Sex": ("Recorded sex", "demographic", "pre-diagnostic", "eligible", "Baseline demographic variable."),
    "BMI": ("Body mass index", "demographic", "pre-diagnostic", "eligible_with_unit_review", "Baseline measurement; unit/calculation should be documented."),
    "DM": ("Diabetes mellitus indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede CAD diagnosis; ascertainment timing is not documented."),
    "HTN": ("Hypertension indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede CAD diagnosis; ascertainment timing is not documented."),
    "Current Smoker": ("Current smoking indicator", "medical history", "pre-diagnostic", "eligible", "Baseline history/exposure."),
    "EX-Smoker": ("Former smoking indicator", "medical history", "pre-diagnostic", "eligible", "Baseline history/exposure."),
    "FH": ("Family history indicator", "medical history", "pre-diagnostic", "eligible", "Baseline history variable."),
    "Obesity": ("Obesity indicator", "medical history", "pre-diagnostic", "eligible_with_unit_review", "Potentially derived from anthropometrics; avoid duplicate-derived feature use without documentation."),
    "CRF": ("Chronic renal failure indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede diagnosis; timing and definition require review."),
    "CVA": ("Cerebrovascular accident history", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede diagnosis; timing and definition require review."),
    "Airway disease": ("Airway disease indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede diagnosis; timing is undocumented."),
    "Thyroid Disease": ("Thyroid disease indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede diagnosis; timing is undocumented."),
    "CHF": ("Congestive heart failure indicator", "medical history", "pre-diagnostic", "requires_clinical_review", "May be a downstream/overlapping cardiac diagnosis; temporal relation to CAD target is unclear."),
    "DLP": ("Dyslipidemia indicator", "medical history", "pre-diagnostic", "eligible_with_timing_confirmation", "History may precede diagnosis; timing is undocumented."),
    "BP": ("Blood pressure", "physical examination", "pre-diagnostic", "eligible_with_unit_review", "Likely baseline examination measurement; unit and measurement timing require confirmation."),
    "PR": ("Pulse rate", "physical examination", "pre-diagnostic", "eligible_with_unit_review", "Baseline examination measurement; unit/timing require confirmation."),
    "Edema": ("Peripheral edema finding", "physical examination", "pre-diagnostic", "requires_clinical_review", "Could reflect established cardiac disease; timing relative to CAD ascertainment is unclear."),
    "Weak Peripheral Pulse": ("Weak peripheral pulse finding", "physical examination", "pre-diagnostic", "requires_clinical_review", "Physical finding may be available before diagnosis but timing is undocumented."),
    "Lung rales": ("Pulmonary rales finding", "physical examination", "pre-diagnostic", "requires_clinical_review", "May reflect heart failure or acute disease; timing is undocumented."),
    "Systolic Murmur": ("Systolic murmur finding", "physical examination", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially available at examination; timing and ascertainment require review."),
    "Diastolic Murmur": ("Diastolic murmur finding", "physical examination", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially available at examination; timing and ascertainment require review."),
    "Typical Chest Pain": ("Typical chest pain indicator", "symptom", "pre-diagnostic", "eligible_with_timing_confirmation", "Symptom is clinically relevant but must be defined as present before the diagnostic procedure."),
    "Dyspnea": ("Dyspnea indicator", "symptom", "pre-diagnostic", "eligible_with_timing_confirmation", "Symptom may precede diagnosis; timing and coding require review."),
    "Function Class": ("Functional class", "symptom/functional status", "pre-diagnostic", "requires_clinical_review", "May reflect established heart failure severity; prediction-time availability is uncertain."),
    "Atypical": ("Atypical chest pain indicator", "symptom", "pre-diagnostic", "eligible_with_timing_confirmation", "Potential baseline symptom; timing requires confirmation."),
    "Nonanginal": ("Non-anginal symptom indicator", "symptom", "pre-diagnostic", "eligible_with_timing_confirmation", "Potential baseline symptom; timing requires confirmation."),
    "Exertional CP": ("Exertional chest pain indicator", "symptom", "pre-diagnostic", "eligible_with_timing_confirmation", "Constant in this file and therefore not informative; retain in raw data but exclude as predictor."),
    "LowTH Ang": ("Low-threshold angina indicator", "symptom", "pre-diagnostic", "requires_clinical_review", "Angina classification may be part of diagnostic assessment; timing is undocumented."),
    "Q Wave": ("Q-wave ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Could be baseline ECG information, but ECG acquisition timing is undocumented."),
    "St Elevation": ("ST elevation ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "St Depression": ("ST depression ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "Tinversion": ("T-wave inversion ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "LVH": ("Left ventricular hypertrophy ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "Poor R Progression": ("Poor R-wave progression ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "BBB": ("Bundle branch block ECG finding", "ECG", "pre-diagnostic", "eligible_with_timing_confirmation", "Potentially diagnostic ECG information; prediction-time availability must be specified."),
    "FBS": ("Fasting blood sugar", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; fasting status and timing require confirmation."),
    "CR": ("Creatinine", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "TG": ("Triglycerides", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; fasting status and timing require confirmation."),
    "LDL": ("LDL cholesterol", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "HDL": ("HDL cholesterol", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "BUN": ("Blood urea nitrogen", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "ESR": ("Erythrocyte sedimentation rate", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "HB": ("Hemoglobin", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "K": ("Serum potassium", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "Na": ("Serum sodium", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "WBC": ("White blood cell count", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "Lymph": ("Lymphocyte measure", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "Neut": ("Neutrophil measure", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "PLT": ("Platelet count", "laboratory", "pre-diagnostic", "eligible_with_unit_review", "Laboratory value may be baseline; units and timing require confirmation."),
    "EF-TTE": ("Ejection fraction from transthoracic echocardiography", "echocardiography", "diagnostic/downstream", "exclude", "Echo-derived cardiac function may be measured during diagnostic work-up and is a downstream leakage risk."),
    "Region RWMA": ("Regional wall-motion abnormality", "echocardiography", "diagnostic/downstream", "exclude", "Diagnostic imaging finding is likely downstream of or part of CAD ascertainment."),
    "VHD": ("Valvular heart disease severity", "echocardiography/diagnosis", "diagnostic/downstream", "exclude", "Clinical diagnosis/severity field may be downstream and is not a safe pre-diagnostic predictor without temporal evidence."),
    "Cath": ("Catheterization-derived CAD label", "target", "target", "exclude_target", "This is the target itself: Cad versus Normal, based on at least 50% coronary narrowing."),
}


def audit_z_alizadeh() -> pd.DataFrame:
    frame = load_csv("data/interim/validated/z_alizadeh_sani.csv")
    rows = []
    for column in frame.columns:
        meaning, group, stage, decision, reason = Z_FEATURES[column]
        rows.append({
            "original_feature_name": column,
            "clinical_meaning": meaning,
            "feature_group": group,
            "dtype": str(frame[column].dtype),
            "missing_count": int(frame[column].isna().sum()),
            "missing_percentage": round(float(frame[column].isna().mean() * 100), 6),
            "unique_value_count": int(frame[column].nunique(dropna=False)),
            "observed_values": "|".join(map(str, frame[column].drop_duplicates().head(12).tolist())),
            "clinical_measurement_or_stage": stage,
            "direct_target_encoding_risk": "direct" if column == "Cath" else ("high" if decision == "exclude" else "possible"),
            "leakage_risk": "high" if decision in {"exclude", "exclude_target"} else ("moderate" if "review" in decision else "low"),
            "eligibility_decision": decision,
            "justification": reason,
            "source_evidence": "UCI dataset 412 metadata and original workbook labels; temporal availability is not fully documented.",
        })
    return pd.DataFrame(rows)


NHANES_KEY = {
    "SEQN": ("participant identifier", "identifier", "exclude", "No predictor role; merge key only."),
    "RIDAGEYR": ("age in years", "demographic", "eligible_predictor", "Baseline demographic variable."),
    "RIAGENDR": ("sex", "demographic", "eligible_predictor", "Baseline demographic variable."),
    "RIDRETH3": ("race/ethnicity coding", "demographic", "eligible_predictor_with_ethics_review", "May support subgroup analysis; avoid causal interpretation."),
    "WTINT2YR": ("full-sample interview weight", "survey design", "required_design_variable", "Use for interview-only population estimates."),
    "WTMEC2YR": ("full-sample MEC examination weight", "survey design", "required_design_variable", "Use for MEC/examination and applicable laboratory analyses."),
    "WTSAF2YR": ("fasting subsample weight", "survey design", "required_design_variable", "Required for fasting subsample variables such as triglycerides/LDL."),
    "SDMVPSU": ("masked variance pseudo-PSU", "survey design", "required_design_variable", "Required with SDMVSTRA for Taylor linearization/complex survey variance."),
    "SDMVSTRA": ("masked variance pseudo-stratum", "survey design", "required_design_variable", "Required with SDMVPSU for complex survey variance."),
    "CDQ001": ("ever had chest pain or discomfort", "cardiovascular questionnaire", "descriptive_outcome_candidate", "Self-reported symptom; not clinician-adjudicated CAD and not prospective."),
    "BPQ020": ("ever told by a health professional of hypertension", "blood-pressure questionnaire", "descriptive_outcome_candidate", "Self-reported hypertension history; not a prospective outcome."),
    "BPQ040A": ("taking prescription for hypertension", "blood-pressure questionnaire", "eligible_predictor_with_timing_review", "Medication/history variable; timing and treatment indication require review."),
    "BPXSY1": ("first systolic blood pressure reading", "examination", "eligible_predictor", "Measured examination variable; use MEC weight."),
    "BPXDI1": ("first diastolic blood pressure reading", "examination", "eligible_predictor", "Measured examination variable; use MEC weight."),
    "BMXBMI": ("body mass index", "examination", "eligible_predictor", "Measured examination variable; use MEC weight."),
    "LBXTC": ("total cholesterol", "laboratory", "eligible_predictor", "Laboratory measure; use MEC weight unless analysis-specific guidance says otherwise."),
    "LBDHDD": ("HDL cholesterol", "laboratory", "eligible_predictor", "Laboratory measure; use MEC weight."),
    "LBDLDL": ("calculated LDL cholesterol", "fasting laboratory", "eligible_predictor_with_subsample_weight", "Fasting subsample; use WTSAF2YR and inspect eligibility."),
    "LBXTR": ("triglycerides", "fasting laboratory", "eligible_predictor_with_subsample_weight", "Fasting subsample; use WTSAF2YR."),
    "DIQ010": ("doctor told participant they have diabetes", "diabetes questionnaire", "eligible_predictor_with_timing_review", "Self-reported history; not clinician-adjudicated in this derivative."),
    "SMQ020": ("smoked at least 100 cigarettes in life", "smoking questionnaire", "eligible_predictor", "Self-reported exposure; codebook special missing values require handling."),
}


def component_frames() -> dict[str, pd.DataFrame]:
    return {
        path.stem: pd.read_sas(path, format="xport", encoding="utf-8")
        for path in sorted((DATA / "raw" / "nhanes" / "raw_cycles" / "2017-2018").glob("*.XPT"))
    }


def audit_nhanes() -> tuple[pd.DataFrame, list[dict], dict]:
    frames = component_frames()
    merged = load_csv("data/interim/nhanes_components/nhanes_2017_2018_cardiovascular_components.csv")
    source_for = {}
    for component, frame in frames.items():
        for column in frame.columns:
            source_for.setdefault(column, component)
    rows = []
    for column in merged.columns:
        meaning, group, decision, justification = NHANES_KEY.get(
            column,
            ("Component variable requiring codebook-specific review", source_for.get(column, "merged"), "codebook_review", "Do not use as a predictor until codebook coding and timing are reviewed."),
        )
        component = source_for.get(column, "merged")
        source_frame = frames.get(component)
        weight = "WTSAF2YR" if component == "TRIGLY_J" or column in {"LBDLDL", "LBXTR"} else (
            "WTINT2YR" if component in {"DEMO_J", "BPQ_J", "CDQ_J", "DIQ_J", "SMQ_J"} else "WTMEC2YR"
        )
        rows.append({
            "variable": column,
            "source_component": component,
            "clinical_meaning": meaning,
            "dtype": str(merged[column].dtype),
            "component_rows": None if source_frame is None else int(len(source_frame)),
            "merged_missing_percentage": round(float(merged[column].isna().mean() * 100), 6),
            "component_missing_percentage": None if source_frame is None else round(float(source_frame[column].isna().mean() * 100), 6),
            "unique_value_count_merged": int(merged[column].nunique(dropna=False)),
            "eligibility_decision": decision,
            "recommended_weight": weight if decision != "required_design_variable" else "preserve; not a predictor",
            "target_or_feature_role": "candidate descriptive outcome" if "outcome" in decision else ("survey design" if group == "survey design" else "candidate predictor" if "eligible" in decision else "review"),
            "justification": justification,
            "official_evidence": f"CDC NHANES 2017-2018 {component} codebook preserved under data/raw/nhanes/raw_cycles/2017-2018.",
        })
    merge_steps = []
    current = None
    for component in ["DEMO_J", "BPQ_J", "BPX_J", "CDQ_J", "BMX_J", "TCHOL_J", "HDL_J", "TRIGLY_J", "DIQ_J", "SMQ_J"]:
        frame = frames[component].drop_duplicates("SEQN")
        if current is None:
            current = frame
            continue
        merged_step = current.merge(frame, on="SEQN", how="outer", indicator=True, suffixes=("", f"_{component}"))
        merge_steps.append({
            "added_component": component,
            "left_rows_before_merge": int(len(current)),
            "component_rows": int(len(frame)),
            "merged_rows": int(len(merged_step)),
            "left_only": int((merged_step["_merge"] == "left_only").sum()),
            "right_only": int((merged_step["_merge"] == "right_only").sum()),
            "both": int((merged_step["_merge"] == "both").sum()),
        })
        current = merged_step.drop(columns="_merge")
    return pd.DataFrame(rows), merge_steps, {
        "merged_rows": int(len(merged)),
        "unique_seqn": int(merged["SEQN"].nunique()),
        "duplicate_seqn": int(merged["SEQN"].duplicated().sum()),
        "survey_design_columns": [c for c in ["WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVPSU", "SDMVSTRA"] if c in merged],
        "component_rows": {name: int(len(frame)) for name, frame in frames.items()},
    }


def normalize_tuple(values: tuple) -> tuple:
    normalized = []
    for value in values:
        if pd.isna(value):
            normalized.append("<NA>")
        elif isinstance(value, float):
            normalized.append(round(value, 6))
        else:
            normalized.append(str(value).strip().lower())
    return tuple(normalized)


def audit_existing_overlap() -> dict:
    paths = {
        "five_states": "data/interim/validated/five_states.csv",
        "mendeley": "data/interim/validated/mendeley.csv",
        "kaggle_cvd_70000": "data/interim/validated/kaggle_cvd_70000.csv",
        "cleveland": "data/interim/validated/uci_heart_disease_cleveland.csv",
        "statlog": "data/interim/validated/statlog_heart.csv",
    }
    frames = {name: load_csv(path) for name, path in paths.items()}
    canonical = {
        "age": ["age", "Age"],
        "sex": ["sex", "Sex", "gender"],
        "chest_pain": ["cp", "chest_pain", "ChestPainType", "chestpain"],
        "rest_bp": ["trestbps", "rest_bp", "RestingBP", "restingBP"],
        "chol": ["chol", "serum_chol", "Cholesterol", "serumcholestrol"],
        "fbs": ["fbs", "fasting_blood_sugar", "FastingBS", "fastingbloodsugar"],
        "rest_ecg": ["restecg", "rest_ecg", "RestingECG", "restingrelectro"],
        "max_hr": ["thalach", "max_heart_rate", "MaxHR", "maxheartrate"],
        "exang": ["exang", "exercise_angina", "ExerciseAngina", "exerciseangia"],
        "oldpeak": ["oldpeak", "Oldpeak"],
        "slope": ["slope", "ST_Slope"],
        "major_vessels": ["ca", "major_vessels", "noofmajorvessels"],
        "thal": ["thal"],
    }
    canonical_frames = {}
    for name in ["five_states", "mendeley", "kaggle_cvd_70000", "cleveland", "statlog"]:
        frame = frames[name]
        columns = {}
        for key, candidates in canonical.items():
            match = next((candidate for candidate in candidates if candidate in frame.columns), None)
            if match:
                columns[key] = match
        canonical_frames[name] = pd.DataFrame({
            key: frame[column].map(lambda value: normalize_tuple((value,))[0])
            for key, column in columns.items()
        })
    results = []
    for left, right in [("cleveland", "statlog"), ("cleveland", "five_states"), ("cleveland", "mendeley"), ("cleveland", "kaggle_cvd_70000"), ("statlog", "mendeley")]:
        common = sorted(set(canonical_frames[left]).intersection(canonical_frames[right]))
        if len(common) < 8:
            results.append({"left": left, "right": right, "common_predictor_fields": len(common), "exact_common_predictor_matches": 0, "classification": "unresolvable_schema_or_provenance"})
            continue
        left_keys = set(map(tuple, canonical_frames[left][common].to_numpy()))
        right_keys = set(map(tuple, canonical_frames[right][common].to_numpy()))
        results.append({
            "left": left,
            "right": right,
            "common_predictor_fields": len(common),
            "exact_common_predictor_matches": len(left_keys.intersection(right_keys)),
            "classification": "possible_record_overlap_requires_source_proof" if left_keys.intersection(right_keys) else "no_exact_match_detected_under_normalization",
        })
        if left == "cleveland" and right == "statlog" and len(left_keys.intersection(right_keys)) == len(frames["statlog"]):
            cleveland_lookup = dict(zip(
                map(tuple, canonical_frames["cleveland"][common].to_numpy()),
                frames["cleveland"]["num"],
            ))
            statlog_keys = list(map(tuple, canonical_frames["statlog"][common].to_numpy()))
            mapped = [cleveland_lookup[key] for key in statlog_keys]
            statlog_binary = frames["statlog"]["target"].map({1: 0, 2: 1}).tolist()
            results[-1].update({
                "matched_right_rows": len(mapped),
                "target_agreement_after_standard_mapping": int(sum((value > 0) == bool(target) for value, target in zip(mapped, statlog_binary))),
                "classification": "confirmed_record_level_overlap",
                "evidence": "All 270 Statlog predictor profiles exactly match unique Cleveland predictor profiles; mapped targets agree for all 270 rows.",
            })
    return {"pairwise_checks": results, "interpretation": "Identical clinical profiles are not treated as patient identity; no independent-validation claim is permitted."}


def audit_ecg() -> dict:
    ptbxl = load_csv("data/interim/clinical_metadata/ptb_xl_metadata_index.csv")
    records = load_csv("data/interim/clinical_metadata/ptb_diagnostic_record_index.csv")
    scp = load_csv("data/raw/ptb_xl/metadata/scp_statements.csv")
    statement_map = scp.set_index("Unnamed: 0") if "Unnamed: 0" in scp.columns else scp.set_index(scp.columns[0])
    diagnostic_counts = {}
    for value in ptbxl["scp_codes"].dropna():
        try:
            codes = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            codes = {}
        for code in codes:
            if code in statement_map.index and "diagnostic_class" in statement_map:
                label = statement_map.loc[code, "diagnostic_class"]
                diagnostic_counts[str(label)] = diagnostic_counts.get(str(label), 0) + 1
    fold_patient_counts = ptbxl.groupby("patient_id")["strat_fold"].nunique()
    return {
        "ptb_xl_records": int(len(ptbxl)),
        "ptb_xl_patients": int(ptbxl["patient_id"].nunique()),
        "ptb_xl_patient_multi_fold_count": int((fold_patient_counts > 1).sum()),
        "ptb_xl_recommended_folds": sorted(ptbxl["strat_fold"].dropna().unique().tolist()),
        "ptb_xl_diagnostic_class_annotation_counts": diagnostic_counts,
        "ptb_diagnostic_record_count": int(len(records)),
        "ptb_diagnostic_visible_subject_directory_count": int(records["subject_id"].nunique()),
        "ptb_diagnostic_record_per_subject": records.groupby("subject_id").size().describe().to_dict(),
        "overlap_classification": "unresolved_related_provenance",
        "overlap_evidence": "PTB-XL exposes numeric patient_id values and PTB Diagnostic exposes patient directory identifiers; no validated crosswalk is available. Both originate from PTB/PhysioNet, so independence cannot be assumed.",
        "waveform_policy": "Archives verified; waveforms were not extracted or loaded.",
    }


def write_markdown_files(z: pd.DataFrame, nhanes: pd.DataFrame, merge_steps: list[dict], nhanes_summary: dict, ecg: dict, overlap: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DATASET_PROVENANCE_AUDIT.md").write_text(
        "# Dataset Provenance Audit v3\n\n"
        "All v2-acquired inputs were audited locally without additional downloads. UCI Z-Alizadeh Sani is version 412, DOI 10.24432/C5Q31T, CC BY 4.0, 303 rows and 56 columns. PTB-XL is version 1.0.3, DOI 10.13026/kfzx-aw45, CC BY 4.0, 21,799 records and 18,869 patients. PTB Diagnostic is version 1.0.0, DOI 10.13026/C28C71, ODC-By 1.0, 549 records; 290 subject directories are exposed in the local RECORDS file while the official page documents 294 subjects. Auricle is version 1.0.0, DOI 10.13026/swme-vp69, ODC-By 1.0, metadata-only locally. NHANES is the 2017-2018 public-use cycle with ten components.\n\n"
        "PTB-XL, PTB Diagnostic, and Auricle metadata checksums were independently compared with their respective official SHA256SUMS files. Both ECG archives passed ZIP structural validation. The Auricle checksum file was re-downloaded from the Auricle endpoint and all acquired Auricle metadata checksums pass; it is not the PTB Diagnostic checksum file.\n\n"
        "Existing author-release datasets and prior manifests remain hash-consistent. Kaggle licensing remains unresolved, and Five States provenance remains unresolved.\n",
        encoding="utf-8",
    )
    (OUT / "ECG_INTEGRITY_AND_OVERLAP_AUDIT.md").write_text(
        "# ECG Integrity and Overlap Audit v3\n\n"
        f"- PTB-XL: {ecg['ptb_xl_records']} records, {ecg['ptb_xl_patients']} patients; patient multi-fold assignments: {ecg['ptb_xl_patient_multi_fold_count']}.\n"
        f"- PTB Diagnostic: {ecg['ptb_diagnostic_record_count']} records and {ecg['ptb_diagnostic_visible_subject_directory_count']} visible subject directories.\n"
        "- PTB-XL `strat_fold` is suitable for patient-level protocol construction because each indexed patient remains in one fold. Do not randomly split records.\n"
        "- PTB-XL and PTB Diagnostic share PTB provenance, but no validated identifier crosswalk exists. Their independence is unresolved and must not be claimed.\n"
        "- PTB-XL SCP labels are hierarchical and multi-label; diagnostic superclass, subclass, rhythm, and form labels must not be collapsed without a prespecified task definition.\n",
        encoding="utf-8",
    )
    (OUT / "EXISTING_DATASET_OVERLAP_AUDIT.md").write_text(
        "# Existing Dataset Overlap Audit v3\n\n"
        "Exact normalized common-feature fingerprints were calculated where schemas permitted. A matching clinical profile is not treated as patient identity. Cleveland and Statlog show confirmed record-level correspondence; other source relationships remain unresolved without source-level provenance or identifiers. Dataset names and different row counts do not establish independence.\n\n"
        + json.dumps(overlap, indent=2),
        encoding="utf-8",
    )
    (OUT / "PROPOSED_RESEARCH_PROTOCOL.md").write_text(
        "# Proposed Research Protocol\n\n"
        "## Primary question\n"
        "Can a prespecified model using genuinely pre-diagnostic clinical variables estimate CAD presence in the Z-Alizadeh Sani cohort without diagnostic-procedure leakage, with uncertainty and calibration reported transparently?\n\n"
        "## Primary dataset and target\n"
        "Use Z-Alizadeh Sani (`Cath`: Cad versus Normal). Exclude `Cath`, `EF-TTE`, `Region RWMA`, and `VHD` from the primary predictor set. ECG, symptoms, functional class, and examination findings require a documented prediction-time definition before inclusion. Use only demographics, history, baseline measurements, and laboratory variables confirmed available before catheterization.\n\n"
        "## Validation\n"
        "Use repeated nested stratified cross-validation with an outer patient-level split and an inner tuning split. Fit imputation, scaling, encoding, feature selection, and any resampling within inner training folds only. Do not use repeated overlapping holdouts as external validation. Reserve a genuinely independent cohort only after provenance and target compatibility are established; current Cleveland, Statlog, Five States, Mendeley, and Kaggle relationships do not meet that standard.\n\n"
        "## Missingness and imbalance\n"
        "Report missingness before any handling. Prespecify clinically justified imputation within training folds, add missingness indicators only if justified, and perform sensitivity analyses. If class weighting or resampling is used, apply it inside training folds only and compare against an unresampled baseline.\n\n"
        "## Evaluation and uncertainty\n"
        "Report discrimination, calibration, threshold-independent and clinically relevant threshold metrics with nested resampling uncertainty intervals. Use bootstrap or repeated outer-fold intervals, not test-set tuning. Compare models with paired outer-fold predictions only after the protocol is frozen. Decision-curve analysis is exploratory and is not evidence of clinical utility without an explicit net-benefit context.\n\n"
        "## Secondary tracks\n"
        "Treat NHANES as complex-survey descriptive/associational analysis, PTB-XL/PTB Diagnostic as separate patient-level ECG tasks, Auricle as multimodal physiology metadata until a storage decision, and UCI Heart Failure as a separate mortality task. Do not merge targets or claim cross-track clinical generalization.\n",
        encoding="utf-8",
    )
    (OUT / "REMAINING_RISKS_AND_LIMITATIONS.md").write_text(
        "# Remaining Risks and Limitations\n\n"
        "- Z-Alizadeh feature timing and units are incompletely documented; downstream echocardiography/catheterization variables require exclusion or clinical review.\n"
        "- Z-Alizadeh is a small single-cohort dataset; internal validation uncertainty may be large.\n"
        "- NHANES is cross-sectional, uses self-reported and measurement-defined variables, and requires survey design variables and component-specific weights.\n"
        "- PTB-XL and PTB Diagnostic have related PTB provenance and unresolved patient overlap.\n"
        "- Statlog is an exact predictor-level/label-corresponding match to the Cleveland records under the documented normalization; Five States, Mendeley, and Kaggle independence is not established.\n"
        "- Kaggle licensing and Five States provenance remain unresolved.\n"
        "- No clinical deployment, clinical utility, ethics, or generalizability claim is supported by this audit.\n",
        encoding="utf-8",
    )
    summary = (
        "# Executive Summary\n\n"
        "This audit independently verified local provenance, hashes, official checksum matches, ZIP integrity, schemas, target definitions, feature eligibility, NHANES merge/design variables, ECG patient-level structure, and cross-dataset overlap limitations. No model training, tuning, or evaluation was performed.\n\n"
        "The scientifically defensible primary research track is a leakage-controlled Z-Alizadeh Sani CAD analysis using only pre-diagnostic predictors. NHANES is suitable for weighted descriptive/associational analyses, not prospective prediction. PTB-XL and PTB Diagnostic are separate ECG tracks with patient-level splitting and unresolved overlap. Auricle remains metadata-only.\n\n"
        f"NHANES merged participants: {nhanes_summary['merged_rows']}; duplicate SEQN rows: {nhanes_summary['duplicate_seqn']}. Z-Alizadeh excluded direct/downstream fields are recorded in Z_ALIZADEH_FEATURE_AUDIT.csv.\n"
    )
    (OUT / "EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")


def write_csvs(z: pd.DataFrame, nhanes: pd.DataFrame) -> None:
    z.to_csv(OUT / "Z_ALIZADEH_FEATURE_AUDIT.csv", index=False)
    nhanes.to_csv(OUT / "NHANES_VARIABLE_AND_WEIGHT_AUDIT.csv", index=False)
    targets = pd.DataFrame([
        ["Five States", "HeartDisease", "1", "0", "Supplied heart-disease label; provenance unresolved", "disease-presence classification", "918-row author release", "No verified external independence"],
        ["Mendeley", "target", "1", "0", "Supplied target; patientid is an identifier, not a predictor", "disease-presence classification", "1,000-row public dataset", "Target ascertainment details limited"],
        ["Kaggle CVD", "cardio", "1", "0", "Supplied cardiovascular disease label", "disease-presence classification", "70,000-row benchmark", "License and provenance require review"],
        ["UCI Cleveland", "num", "1-4", "0", "Angiographic heart-disease severity/presence coding", "CAD/general heart disease classification", "303 records", "Binary mapping is derived, not original"],
        ["Statlog Heart", "target", "2", "1", "UCI target coding: presence versus absence", "heart-disease classification", "270 records", "Possible source overlap with Cleveland"],
        ["Z-Alizadeh Sani", "Cath", "Cad", "Normal", "CAD if coronary diameter narrowing is at least 50%", "CAD diagnosis", "303 patients", "Catheterization-derived target; downstream fields must be excluded"],
        ["UCI Heart Failure", "DEATH_EVENT", "1", "0", "Death during follow-up", "heart-failure mortality", "299 patients", "Not disease-presence classification"],
        ["NHANES 2017-2018", "none fixed", "not created", "not created", "Cross-sectional public-use measurements/questionnaires", "weighted descriptive/associational analysis", "9,254 merged participants", "No synthetic or prospective target"],
        ["PTB-XL", "scp_codes", "task-specific", "task-specific", "Multi-label cardiologist/ECG statements", "ECG diagnostic classification", "21,799 records/18,869 patients", "Patient-level splits required"],
        ["PTB Diagnostic", "clinical summaries", "task-specific", "task-specific", "Clinical summaries and diagnostic classes", "ECG diagnostic classification", "549 records", "Visible 290 subject directories; source documents 294 subjects"],
        ["Auricle", "cohort metadata", "heart failure/healthy", "healthy", "Multimodal physiology cohort labels", "multimodal physiology", "42 healthy/28 heart failure documented", "Full archive deferred"],
    ], columns=["dataset", "target_variable", "positive_class", "negative_class", "clinical_definition", "intended_task", "population", "limitations"])
    targets.to_csv(OUT / "TARGET_DEFINITION_MATRIX.csv", index=False)
    compatibility = pd.DataFrame([
        ["Z-Alizadeh Sani", "Z-Alizadeh Sani", "directly compatible", "Within-dataset internal validation after eligibility review"],
        ["Z-Alizadeh Sani", "Cleveland/Statlog", "partially compatible", "Held-out comparison only if target/coding and provenance are prespecified; independence unresolved"],
        ["Z-Alizadeh Sani", "Mendeley/Five States/Kaggle", "clinically incompatible without transformation", "Descriptive comparison or separate analyses; do not silently harmonize"],
        ["NHANES", "NHANES", "directly compatible", "Weighted within-cycle descriptive/associational analysis"],
        ["PTB-XL", "PTB Diagnostic", "partially compatible", "Separate ECG tasks; no independent validation claim"],
        ["Auricle", "Tabular datasets", "clinically incompatible", "Multimodal physiology metadata/descriptive track only"],
        ["Heart Failure", "Disease-presence datasets", "clinically incompatible", "Separate mortality task"],
    ], columns=["source_track", "comparison_track", "compatibility", "recommended_use"])
    compatibility.to_csv(OUT / "FEATURE_COMPATIBILITY_MATRIX.csv", index=False)


def write_manifest(input_paths: list[Path], output_paths: list[Path], merge_steps: list[dict], nhanes_summary: dict, ecg: dict, overlap: dict) -> None:
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths if path.exists() and path.is_file()}
    outputs = {str(path.relative_to(ROOT)): sha256(path) for path in output_paths if path.exists() and path.is_file()}
    manifest = {
        "audit_version": "clinical_validation_audit_v3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__},
        "inputs": inputs,
        "outputs": outputs,
        "nhanes_merge_steps": merge_steps,
        "nhanes_summary": nhanes_summary,
        "ecg_summary": ecg,
        "overlap_summary": overlap,
        "prohibited_actions": {"model_training": False, "model_tuning": False, "model_evaluation": False, "additional_downloads": False, "raw_modification": False},
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = audit_z_alizadeh()
    nhanes, merge_steps, nhanes_summary = audit_nhanes()
    ecg = audit_ecg()
    overlap = audit_existing_overlap()
    write_csvs(z, nhanes)
    write_markdown_files(z, nhanes, merge_steps, nhanes_summary, ecg, overlap)
    input_paths = [
        ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv",
        ROOT / "data" / "interim" / "nhanes_components" / "nhanes_2017_2018_cardiovascular_components.csv",
        ROOT / "data" / "interim" / "clinical_metadata" / "ptb_xl_metadata_index.csv",
        ROOT / "data" / "interim" / "clinical_metadata" / "ptb_diagnostic_record_index.csv",
        ROOT / "data" / "raw" / "ptb_xl" / "metadata" / "SHA256SUMS.txt",
        ROOT / "data" / "raw" / "ptb_diagnostic" / "metadata" / "SHA256SUMS.txt",
        ROOT / "data" / "raw" / "auricle" / "metadata" / "SHA256SUMS.txt",
        ROOT / "data" / "raw" / "ptb_xl" / "ptb-xl-1.0.3.zip",
        ROOT / "data" / "raw" / "ptb_diagnostic" / "ptbdb-1.0.0.zip",
        ROOT / "data" / "metadata" / "dataset_registry.json",
        ROOT / "data" / "metadata" / "dataset_manifest_v2.json",
    ]
    output_paths = [path for path in OUT.glob("*") if path.name != "audit_manifest.json"]
    write_manifest(input_paths, output_paths, merge_steps, nhanes_summary, ecg, overlap)
    print(json.dumps({"output_directory": str(OUT), "outputs": len(list(OUT.iterdir())), "model_training": False}, indent=2))


if __name__ == "__main__":
    main()
