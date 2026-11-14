"""Phase 5 provenance, feature-timing and compatibility audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "phase5" / "data_registry"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / "data" / "metadata" / "dataset_registry.json").read_text(encoding="utf-8"))
    rows = []
    for name, item in registry.items():
        rows.append(
            {
                "dataset": name,
                "title": item.get("title"),
                "source": item.get("source_url"),
                "doi": item.get("doi"),
                "version": item.get("version"),
                "license": item.get("license"),
                "access": item.get("access"),
                "rows": item.get("rows", item.get("metadata_record_count", item.get("records_documented_by_source"))),
                "target": item.get("target"),
                "role": item.get("role", item.get("research_track")),
                "status": item.get("status"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "PUBLIC_DATASET_REGISTRY.csv", index=False)
    compatibility = pd.DataFrame(
        [
            ["z_alizadeh_sani", "angiographic CAD >=50%", "303 patients; Cath target", "direct development cohort", "A"],
            ["uci_heart_disease_cleveland", "angiographic num 0 versus 1-4", "303 records; source target", "related centre benchmark; not ZAS external validation", "B"],
            ["statlog_heart", "heart disease classification", "270 records; potential Cleveland provenance relationship", "possible related benchmark; independence unresolved", "B"],
            ["mendeley", "supplied cardiovascular target", "1000 records; patientid", "separate benchmark; endpoint compatibility unresolved", "B"],
            ["kaggle_cvd_70000", "cardio disease presence", "70000 records; no angiographic endpoint", "proxy endpoint only", "D"],
            ["AngioCAD", "angiographic image labels", "413 labels; 377 clinical rows; image archive not acquired", "separate image/clinical task", "C"],
            ["MultiD4CAD", "CT/multimodal CAD", "restricted DUA", "conditional separate CT study", "B"],
            ["CADICA", "invasive angiography imaging", "small imaging cohort", "modality-specific research", "C"],
            ["PTB-XL", "ECG diagnoses", "patient metadata and ECG records", "ECG task; not angiographic validation", "C"],
            ["MIMIC-IV", "EHR phenotype", "phenotype must be newly ascertained", "proxy endpoint only", "D"],
            ["eICU", "ICU EHR phenotype", "endpoint not fixed", "proxy endpoint only", "D"],
            ["NHANES", "survey risk factors", "no angiographic endpoint", "proxy endpoint only", "D"],
        ],
        columns=["dataset", "endpoint", "evidence", "decision", "class"],
    )
    compatibility.to_csv(OUT / "DATASET_COMPATIBILITY_MATRIX.csv", index=False)
    overlap = pd.read_json(ROOT / "data" / "metadata" / "overlap_reports" / "clinical_dataset_overlap.json")
    overlap.to_csv(OUT / "DATASET_OVERLAP_AUDIT.csv")
    source_files = {}
    for relative in [
        "data/interim/validated/z_alizadeh_sani.csv",
        "data/interim/validated/uci_heart_disease_cleveland.csv",
        "data/interim/validated/statlog_heart.csv",
        "data/interim/validated/mendeley.csv",
        "data/interim/validated/kaggle_cvd_70000.csv",
    ]:
        path = ROOT / relative
        source_files[relative] = {"sha256": sha256(path), "rows": len(pd.read_csv(path)), "columns": len(pd.read_csv(path, nrows=1).columns)}
    (OUT / "DATASET_ACQUISITION_MANIFEST.json").write_text(json.dumps({"sources": source_files, "new_downloads": [], "policy": "No new incompatible or unnecessary downloads in Phase 5."}, indent=2), encoding="utf-8")
    write_feature_audits()
    (OUT / "DATASET_PROVENANCE_REPORT.md").write_text(
        "# Phase 5 dataset provenance\n\n"
        "Phase 5 reuses locally verified UCI, Z-Alizadeh Sani, Mendeley, Kaggle, PhysioNet and prior acquisition metadata. No new dataset was merged. The compatibility matrix distinguishes direct angiographic endpoints from proxy and modality-specific resources. Cleveland and Statlog are not assumed independent; PTB datasets are treated as related provenance until linkage is established.\n",
        encoding="utf-8",
    )
    (OUT / "DATASET_ACCESSIBILITY_REPORT.md").write_text(
        "# Phase 5 dataset accessibility\n\n"
        "All Phase 5 experiments use existing local files with recorded hashes. Restricted MultiD4CAD files and large image archives were not downloaded. Optional model libraries for GAM, EBM, CatBoost and Bayesian logistic regression are absent from `.venv-repro`; these experiments are documented as infeasible rather than approximated silently.\n",
        encoding="utf-8",
    )


def write_feature_audits() -> None:
    zas = pd.read_csv(ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv")
    rows = []
    for feature in zas.columns:
        rows.append(
            {
                "feature": feature,
                "dtype": str(zas[feature].dtype),
                "missing": int(zas[feature].isna().sum()),
                "clinical_group": "target" if feature == "Cath" else "basic_risk" if feature in {"Age", "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH"} else "symptoms_examination" if feature in {"Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "BP", "PR", "Edema"} else "resting_ecg" if feature in {"Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"} else "laboratory" if feature in {"FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"} else "echo_or_diagnostic",
                "timing": "outcome" if feature == "Cath" else "prediagnostic plausible but source timing incomplete" if feature not in {"EF-TTE", "Region RWMA", "VHD", "Cath"} else "timing uncertain; keep exploratory or exclude",
                "leakage_risk": "target" if feature == "Cath" else "high/uncertain" if feature in {"EF-TTE", "Region RWMA", "VHD"} else "requires clinical timing confirmation",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "ZAS_FEATURE_DICTIONARY.csv", index=False)
    pd.DataFrame(
        [
            ["ZAS V3", "Age; Sex; Current Smoker; EX-Smoker; FH", "frozen prediagnostic baseline", "primary reference"],
            ["basic_risk", "Age; Sex; DM; HTN; Current Smoker; EX-Smoker; FH", "prediagnostic plausible; timing not fully documented", "development experiment"],
            ["symptoms_examination", "Typical Chest Pain; Dyspnea; Function Class; Atypical; Nonanginal; Exertional CP; LowTH Ang; BP; PR; Edema", "presentation-stage candidate", "development experiment"],
            ["resting_ecg", "Q Wave; St Elevation; St Depression; Tinversion; LVH; Poor R Progression; BBB", "resting ECG candidate; acquisition timing requires confirmation", "development experiment"],
            ["laboratory", "FBS; CR; TG; LDL; HDL; BUN; ESR; HB; K; Na; WBC; Lymph; Neut; PLT", "laboratory candidate; timing and units incomplete", "development experiment"],
            ["eligible_combined", "basic_risk + symptoms_examination + resting_ecg + laboratory", "exploratory prediagnostic composite; no echo variables", "development experiment"],
        ],
        columns=["feature_group", "features", "timing_assessment", "role"],
    ).to_csv(OUT / "ZAS_FEATURE_TIMING_AUDIT.csv", index=False)
    uci = pd.read_csv(ROOT / "results" / "phase4_uci_multicentre" / "uci_four_centre_patient_table.csv")
    pd.DataFrame(
        [{"feature": feature, "dtype": str(uci[feature].dtype), "missing": int(uci[feature].isna().sum()), "timing": "presentation/resting" if feature in {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"} else "exercise-test or timing unresolved" if feature in {"thalach", "exang", "oldpeak", "slope"} else "post-diagnostic/target-derived; exclude"} for feature in uci.columns]
    ).to_csv(OUT / "UCI_FEATURE_DICTIONARY.csv", index=False)
    pd.DataFrame(
        [{"feature": feature, "timing": "presentation/resting" if feature in {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"} else "exercise-test timing unresolved", "eligible_stage": "stage2/3" if feature in {"age", "sex", "cp"} else "stage4"} for feature in uci.columns if feature not in {"centre", "source_row", "target_binary", "num", "ca", "thal"}]
    ).to_csv(OUT / "UCI_FEATURE_TIMING_AUDIT.csv", index=False)
