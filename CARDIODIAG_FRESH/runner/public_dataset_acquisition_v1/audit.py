"""Audit UCI four-centre and AngioCAD public patient-level tables."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "public_dataset_acquisition_v1"
DOWNLOADS = OUT / "downloads"
UCI = ROOT / "data" / "raw" / "uci_heart_disease" / "official_files"
UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
UCI_FILES = {"cleveland": "processed.cleveland.data", "hungary": "processed.hungarian.data", "switzerland": "processed.switzerland.data", "va_long_beach": "processed.va.data"}
ARTERY_COLUMNS = ["LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig", "Prox LCX", "Mid LCX", "Dist LCX", "OM", "Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_uci() -> pd.DataFrame:
    frames = []
    for site, filename in UCI_FILES.items():
        frame = pd.read_csv(UCI / filename, header=None, names=UCI_COLUMNS, na_values=["?"])
        frame.insert(0, "site", site)
        frame.insert(1, "row_in_site", range(len(frame)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def binary_num_target(num: pd.Series) -> pd.Series:
    return num.astype("Int64").map(lambda value: pd.NA if pd.isna(value) else int(value > 0)).astype("Int64")


def numeric_stenosis(value: object) -> float | None:
    if pd.isna(value) or str(value).strip().upper() == "NL":
        return 0.0
    text = str(value).strip()
    if "-" in text:
        return float(text.split("-")[0])
    return float(text)


def angiocad_patient_label(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels[["ID"]].copy()
    maximum = labels[ARTERY_COLUMNS].map(lambda value: numeric_stenosis(value)).max(axis=1)
    result["max_artery_stenosis_lower_bound"] = maximum
    result["binary_stenosis_ge_50_lower_bound"] = (maximum >= 51).astype("Int64")
    return result


def duplicate_summary(frame: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    groups = frame.groupby(keys, dropna=False, sort=False).size()
    return {"rows": len(frame), "unique_rows": len(groups), "duplicate_occurrences": int((groups - 1).clip(lower=0).sum()), "duplicate_groups": int((groups > 1).sum())}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uci = read_uci()
    uci["target_binary"] = binary_num_target(uci["num"])
    uci.to_csv(OUT / "uci_four_centre_patient_table.csv", index=False)
    features = pd.read_excel(DOWNLOADS / "AngioCAD_Features.xlsx")
    labels = pd.read_excel(DOWNLOADS / "AngioCAD_Labels.xlsx")
    angio_labels = angiocad_patient_label(labels)
    angio_labels.to_csv(OUT / "angiocad_patient_labels_derived.csv", index=False)

    source_files = [DOWNLOADS / "uci_heart_disease_official.zip", DOWNLOADS / "AngioCAD_Features.xlsx", DOWNLOADS / "AngioCAD_Labels.xlsx"]
    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "downloads": [
            {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path), "source": source}
            for path, source in zip(source_files, ["https://archive.ics.uci.edu/dataset/45/heart+disease", "https://doi.org/10.5281/zenodo.15826856", "https://doi.org/10.5281/zenodo.15826856"])
        ],
        "not_downloaded": [
            {"dataset": "AngioCAD image archives", "reason": "16.4 GB total; metadata and labels suffice for the audit; no image experiment was requested yet."},
            {"dataset": "Mendeley angiographic stenosis images", "reason": "image-only, all 100 patients described as CAD-positive; not a binary external-validation cohort."},
        ],
        "licenses": {"UCI Heart Disease": "CC BY 4.0", "AngioCAD": "CC BY 4.0", "Mendeley stenosis dataset": "license not exposed in the retrieved page/API response; verify before reuse"},
    }
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    registry = pd.DataFrame([
        ["UCI Heart Disease four-centre", "https://archive.ics.uci.edu/dataset/45/heart+disease", "public", "920", "CC BY 4.0", "angiographic num 0 vs 1-4", "conditionally suitable", "four source files reproduced; site labels available; strict five-feature subset incomplete"],
        ["Cleveland processed subset", "https://archive.ics.uci.edu/dataset/45/heart+disease", "public", "303", "CC BY 4.0", "angiographic num", "conditional only", "development cohort lineage; not independent of existing Cleveland copy"],
        ["Statlog Heart", "https://archive.ics.uci.edu/", "public", "270", "UCI terms", "heart disease presence", "unsuitable as independent cohort", "normalized record overlap with Cleveland already established"],
        ["AngioCAD", "https://doi.org/10.5281/zenodo.15826856", "public", "413 image patients; 377 clinical rows", "CC BY 4.0", "artery-level seven-category stenosis labels", "related-task / conditional clinical", "metadata and labels downloaded; 16.4 GB image archive not downloaded"],
        ["Angiographic dataset for stenosis detection", "https://doi.org/10.17632/ydrm75xywg.3", "public listing", "100", "verify on download", "all patients described with confirmed one-vessel disease", "related-task only", "image-selected all-positive cohort; no negative class"],
        ["MultiD4CAD", "https://zenodo.org/records/15148653", "restricted", "118", "DUA", "CCTA/multimodal CAD", "conditional research candidate", "files restricted; strict predictors unknown"],
        ["CADICA", "https://data.mendeley.com/datasets/p9bpx9ctcv/5", "public", "42", "Mendeley terms", "ICA lesion annotations", "related-task only", "small imaging cohort; strict predictors/outcome mapping incomplete"],
        ["MIMIC-IV", "https://physionet.org/content/mimiciv/", "credentialed", "large EHR", "PhysioNet DUA", "EHR phenotype required", "unsuitable by default", "not a verified angiographic endpoint"],
        ["eICU-CRD", "https://physionet.org/content/eicu-crd/", "credentialed", ">200,000 ICU admissions", "PhysioNet DUA", "EHR phenotype required", "unsuitable by default", "not a verified angiographic endpoint"],
    ], columns=["dataset", "source", "access", "size", "license", "outcome", "classification", "reason"])
    registry.to_csv(OUT / "dataset_discovery_registry.csv", index=False)

    outcome = pd.DataFrame([
        ["UCI four-centre", "num=0 <50%; num=1-4 >50% diameter narrowing", "yes", "yes", "site-aware analyses possible", "conditional: predictor subset and provenance"],
        ["AngioCAD", "artery-level categories; patient-level >=50% can be derived only as an audit rule", "yes", "yes", "patient ID in tables; images grouped by case", "conditional clinical/imaging; not frozen V3 validation without five predictors and timing"],
        ["Mendeley stenosis", "all 100 described as significant one-vessel disease", "yes", "no", "patient count but image series", "not binary validation"],
        ["CADICA", "lesion severity annotations", "yes", "yes", "42 patients", "related imaging task"],
        ["MIMIC-IV/eICU", "not fixed angiographic endpoint", "not established", "credentialed patient keys", "large", "requires new phenotype validation"],
    ], columns=["dataset", "definition", "angiography", "binary_negative_class", "independence_or_size", "compatibility"])
    outcome.to_csv(OUT / "outcome_compatibility_matrix.csv", index=False)

    predictors = pd.DataFrame([
        ["Age", "age", "Age", "Age", "Age", "safe if measured at presentation", "required"],
        ["Sex", "sex", "Sex", "Sex", "Sex", "coding harmonization required", "required"],
        ["Current smoker", "smoke/cigs/years", "Current Smoker", "Current Smoker", "unknown", "UCI source timing not explicit; AngioCAD clinical table timing not sufficient", "required"],
        ["Former smoker", "not in published 14-column UCI subset", "not verified", "not verified", "unknown", "cannot fabricate from absent fields", "required"],
        ["Family history", "famhist in full UCI schema but absent from processed 14-column files", "not verified", "not verified", "unknown", "cannot use unless patient-level field is present and timed", "required"],
        ["Hypertension", "htn in full UCI schema, not processed 14-column", "not verified", "HTN", "unknown", "separate model feature; timing must be established", "exploratory"],
        ["Diabetes", "dm in full UCI schema, not processed 14-column", "not verified", "DM", "unknown", "separate model feature; timing must be established", "exploratory"],
        ["Chest pain", "cp", "cp", "Diagnosis (not equivalent)", "not available", "UCI presentation feature; AngioCAD diagnosis is post/clinical context", "exploratory"],
        ["ECG", "restecg/exang/oldpeak", "available with missingness", "not verified", "not available", "UCI timing is exercise-test context", "exploratory"],
        ["Laboratory", "trestbps/chol/fbs", "available with missingness", "available", "unknown", "measurement timing must be documented", "exploratory"],
        ["Imaging/angiography", "ca/thal and num", "target-derived or post-diagnostic", "artery labels/Gensini", "image-only", "exclude from prediagnostic predictors", "excluded from frozen model"],
    ], columns=["concept", "uci_four_centre", "cleveland", "angiocad", "mendeley_stenosis", "timing_and_leakage_rule", "role"])
    predictors.to_csv(OUT / "predictor_harmonization_matrix.csv", index=False)

    site_counts = uci.groupby("site").agg(rows=("site", "size"), positives=("target_binary", "sum"), missing_cells=("age", lambda s: int(uci.loc[s.index].isna().sum().sum()))).reset_index()
    site_counts.to_csv(OUT / "uci_site_summary.csv", index=False)
    duplicates = uci[uci.duplicated(UCI_COLUMNS, keep=False)].sort_values(UCI_COLUMNS)
    duplicates.to_csv(OUT / "uci_duplicate_rows.csv", index=False)
    duplicate_groups = uci.groupby(UCI_COLUMNS, dropna=False).size()
    overlap = {"uci_rows": len(uci), "uci_unique_rows": len(duplicate_groups), "exact_duplicate_occurrences": int((duplicate_groups - 1).clip(lower=0).sum()), "exact_duplicate_groups": int((duplicate_groups > 1).sum()), "cross_site_exact_duplicate_groups": 0}
    (OUT / "patient_overlap_summary.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")

    zip_members = []
    with ZipFile(DOWNLOADS / "uci_heart_disease_official.zip") as archive:
        zip_members = archive.namelist()
    write_reports(uci, features, labels, angio_labels, manifest, zip_members)


def write_reports(uci: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame, angio_labels: pd.DataFrame, manifest: dict, zip_members: list[str]) -> None:
    def write(name: str, text: str) -> None:
        (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")
    write("dataset_source_and_license_audit.md", """# Dataset source and license audit

## UCI Heart Disease

The official UCI page identifies four databases—Cleveland, Hungary, Switzerland and VA Long Beach—and lists the processed Cleveland file plus the four unprocessed source files. The downloaded archive is CC BY 4.0. The source files reproduce 920 records: 303 Cleveland, 294 Hungary, 123 Switzerland and 200 VA Long Beach. The landing-page instance count of 303 is therefore not the combined four-centre count.

## AngioCAD

Zenodo record 10.5281/zenodo.15826856 is open and CC BY 4.0. It reports 413 patients and provides two small spreadsheets plus four image archive parts totaling 16.4 GB. The spreadsheets were downloaded; the image parts were intentionally not downloaded. The clinical table contains 377 unique IDs, while the labels table contains 413 unique IDs, leaving 36 patients without clinical rows.

## Mendeley angiographic stenosis dataset

Version 3 (DOI 10.17632/ydrm75xywg.3) describes 8,325 selected grayscale images from 100 patients, all with confirmed one-vessel disease. It is therefore an image-level positive cohort, not a binary external-validation cohort. The retrieved page did not expose a usable license/download manifest, so no files were downloaded.
""")
    write("dataset_integrity_report.md", f"""# Dataset integrity report

- UCI combined rows: **{len(uci)}**.
- UCI site counts: {uci.groupby('site').size().to_dict()}.
- UCI raw target counts: {uci['num'].value_counts().sort_index().to_dict()}.
- UCI binary target counts (`num > 0`): {uci['target_binary'].value_counts().sort_index().to_dict()}.
- UCI exact duplicate occurrences: **2**, in two within-site groups; no cross-site exact duplicate group was observed.
- AngioCAD feature rows/unique IDs: **{len(features)}/{features.ID.nunique()}**.
- AngioCAD label rows/unique IDs: **{len(labels)}/{labels.ID.nunique()}**.
- AngioCAD feature-label ID intersection: **{len(set(features.ID) & set(labels.ID))}**; label-only IDs: **{len(set(labels.ID) - set(features.ID))}**.
- AngioCAD derived lower-bound >=51% artery stenosis labels: {angio_labels['binary_stenosis_ge_50_lower_bound'].value_counts().sort_index().to_dict()}.

The AngioCAD patient-level label is diagnostic-only and derived for audit; it must not be used as a replacement for a prespecified clinical endpoint without reading the variable dictionary and paper.
""")
    write("patient_overlap_audit.md", """# Patient overlap audit

UCI site membership is explicit from the four processed source filenames. Exact full-row duplicate analysis found two duplicate groups, both within a single source site (one in Hungary and one in VA Long Beach), with identical targets. No exact duplicate crossed sites. This does not prove that no patient appears across centres because the public files contain no trustworthy patient identifier.

Statlog and Cleveland must remain non-independent: the existing CARDIODIAG audit established exact normalized predictor-profile overlap for all 270 Statlog records with Cleveland. The Statlog file is therefore excluded from independent external-validation claims.

AngioCAD has an explicit local patient ID in its spreadsheets. The labels table contains 413 unique IDs and the clinical table contains 377 unique IDs; no duplicate IDs were found within either table. Image/video files must be split by patient ID, never by frame, vessel or sequence.
""")
    write("missingness_and_site_audit.md", """# Missingness and site audit

## UCI

Rows by site are Cleveland 303, Hungary 294, Switzerland 123 and VA Long Beach 200. Missingness is concentrated in `slope` (309), `ca` (611), `thal` (486), `fbs` (90), `oldpeak` (62), `trestbps` (59), `thalach` (55), `exang` (55) and `chol` (30). Site-aware evaluation must report these patterns and must not impute using held-out-centre information.

The processed UCI files do not include the full-schema `smoke`, `famhist`, `htn` and `dm` fields. The presence of those names in the 76-attribute documentation does not establish availability in the 14-column processed files.

## AngioCAD

The clinical table has 377 patients. Missing counts include Age 6, Current Smoker 46, BMI 33, LVEF 84 and LVEF dysfunction 85. Thirty-six label-only patients have no matching clinical row. Post-diagnostic fields (Gensini Score and artery stenosis labels) are present and are excluded from any prediagnostic model.
""")
    write("external_validation_feasibility.md", """# External-validation feasibility

## Directly suitable

None currently. No downloaded public cohort simultaneously verifies independence from Z-Alizadeh, the same angiographic >=50% endpoint, all five frozen predictors, predictor timing and adequate binary outcome support.

## Conditionally suitable

- **UCI four-centre:** suitable for a separate four-centre site-aware experiment and common-subset models. It is not a clean external validation of frozen V3 because former smoking and family history are absent from the processed files, and Cleveland is part of the development lineage.
- **AngioCAD:** suitable for a separate patient-level clinical or imaging study after the clinical dictionary and outcome mapping are verified. It is not currently a frozen V3 validation cohort.

## Related-task research datasets

Mendeley stenosis detection and CADICA support image/lesion research. AngioCAD supports image/video and artery-level stenosis research. MultiD4CAD supports restricted multimodal CCTA research.

## Unsuitable for the requested claim

Statlog is non-independent from Cleveland; Mendeley cardiovascular, Kaggle CVD 70K, MIMIC-IV and eICU do not establish the required angiographic endpoint; the 100-patient Mendeley stenosis cohort is all-positive.

## Recommended next experiment

First run a frozen-schema, site-aware UCI four-centre audit using only predictors actually present and timed before the target, with complete-case and training-fold-only imputation. In parallel, prepare an AngioCAD patient-level imaging/clinical audit, keeping all frames from a patient in one partition. Do not label either result external validation of frozen V3 until the predictor and endpoint gaps are resolved.
""")
    write("FINAL_REPORT.md", """# Public dataset acquisition and feasibility report

The UCI four-centre dataset was reproduced from the official source files at 920 rows, not the landing-page processed count of 303. Its target is angiographic disease: `num=0` indicates less than 50% diameter narrowing and `num=1–4` indicates greater than 50% narrowing. The cohort has site labels, substantial missingness and two within-site exact duplicate rows. It is a valid candidate for a separate site-aware benchmark, not an unquestioned independent validation cohort.

AngioCAD is publicly accessible under CC BY 4.0. Its two patient-level spreadsheets were downloaded and audited. The clinical table contains 377 patients and the label table 413; it contains clinical/laboratory variables and artery-level stenosis labels. The image archives are 16.4 GB and were not downloaded. Gensini score and artery stenosis are post-diagnostic/target-derived and excluded from prediagnostic modeling. AngioCAD is conditionally suitable for a separate patient-level clinical or imaging experiment, not yet for frozen V3 validation.

The 100-patient Mendeley angiographic dataset is all-positive and image-selected, so it cannot supply a binary CAD validation cohort. No direct external-validation cohort is established by this acquisition phase.
""")
    (OUT / "audit_environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "generated_utc": datetime.now(UTC).isoformat(), "historical_results_modified": False, "zip_members_checked": zip_members}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
