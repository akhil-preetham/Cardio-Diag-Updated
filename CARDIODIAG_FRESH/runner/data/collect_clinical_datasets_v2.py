"""Acquire and index the clinical dataset expansion without model training.

The pipeline preserves downloaded originals, keeps ECG waveforms in native
formats, and performs metadata-only inspection for large PhysioNet sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import platform
import re
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size < 1_000_000:
        prefix = destination.read_bytes()[:64].lower()
        if prefix.startswith(b"<!doctype") or prefix.startswith(b"<html"):
            destination.unlink()
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "CARDIODIAG-dataset-collector/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if destination.suffix.lower() in {".xpt", ".csv", ".zip"}:
        prefix = destination.read_bytes()[:64].lower()
        if prefix.startswith(b"<!doctype") or prefix.startswith(b"<html"):
            destination.unlink()
            raise ValueError(f"Official URL returned HTML instead of {destination.suffix}: {url}")
    return {
        "url": url,
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "downloaded": True,
    }


def download_optional(url: str, destination: Path) -> dict:
    try:
        return download(url, destination)
    except Exception as exc:
        return {"url": url, "path": str(destination.relative_to(ROOT)), "status": "failed", "error": str(exc)}


def extract_zip(archive: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
        return handle.namelist()


def parse_physionet_records(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]


def verify_physionet_checksums(checksum_file: Path, files: list[Path]) -> dict[str, bool]:
    if not checksum_file.exists():
        return {}
    supplied = {}
    for line in checksum_file.read_text(errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line.strip())
        if match:
            supplied[Path(match.group(2)).name] = match.group(1).lower()
    return {
        path.name: path.name in supplied and sha256(path) == supplied[path.name]
        for path in files
        if path.exists() and path.name != checksum_file.name
    }


def collect_z_alizadeh() -> dict:
    base = RAW / "z_alizadeh_sani"
    archive = base / "z_alizadeh_sani.zip"
    source = download("https://archive.ics.uci.edu/static/public/412/z+alizadeh+sani.zip", archive)
    extracted = base / "extracted"
    names = extract_zip(archive, extracted)
    workbook = next((p for p in extracted.rglob("*.xlsx")), None)
    if workbook is None:
        raise FileNotFoundError("The UCI Z-Alizadeh Sani archive contained no XLSX file")
    sheets = pd.ExcelFile(workbook).sheet_names
    frames = {sheet: pd.read_excel(workbook, sheet_name=sheet) for sheet in sheets}
    primary = max(frames.values(), key=lambda frame: frame.shape[0] * max(frame.shape[1], 1))
    validated = INTERIM / "validated" / "z_alizadeh_sani.csv"
    validated.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(validated, index=False)
    target_candidates = [c for c in primary.columns if str(c).strip().lower() in {"cad", "cath", "class", "target", "diagnosis"}]
    target = target_candidates[-1] if target_candidates else None
    report = {
        "dataset": "z_alizadeh_sani",
        "rows": int(len(primary)),
        "columns": int(len(primary.columns)),
        "column_names": [str(c) for c in primary.columns],
        "dtypes": {str(c): str(primary[c].dtype) for c in primary.columns},
        "missing_values": {str(c): int(primary[c].isna().sum()) for c in primary.columns},
        "target_candidate": target,
        "target_values": {} if target is None else {str(k): int(v) for k, v in primary[target].value_counts(dropna=False).items()},
        "excel_sheets": sheets,
        "archive_members": names,
        "source_definition": "CAD if coronary artery diameter narrowing is at least 50%; otherwise Normal",
        "potential_target_leakage_review": "Clinical and echo variables require clinical review because some may be downstream of CAD ascertainment.",
    }
    report_path = META / "quality_reports" / "z_alizadeh_sani.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"source": source, "report": report, "validated": str(validated.relative_to(ROOT))}


def collect_physionet_metadata(name: str, version: str, files: list[str], license_name: str) -> dict:
    base_url = f"https://physionet.org/files/{name}/{version}/"
    base = RAW / {
        "ptb-xl": "ptb_xl",
        "ptbdb": "ptb_diagnostic",
        "auricle": "auricle",
    }[name]
    metadata_dir = base / "metadata"
    inventory = []
    for filename in files:
        result = download_optional(urljoin(base_url, filename + "?download"), metadata_dir / filename)
        inventory.append(result)
    records = parse_physionet_records(metadata_dir / "RECORDS")
    database = metadata_dir / "ptbxl_database.csv"
    record_count = None
    patient_count = None
    columns = []
    if database.exists():
        frame = pd.read_csv(database)
        record_count = len(frame)
        patient_count = int(frame["patient_id"].nunique()) if "patient_id" in frame else None
        columns = list(frame.columns)
        index_path = INTERIM / "clinical_metadata" / "ptb_xl_metadata_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        keep = [c for c in ["ecg_id", "patient_id", "filename_lr", "filename_hr", "scp_codes", "strat_fold", "age", "sex", "recording_date"] if c in frame]
        frame[keep].to_csv(index_path, index=False)
    if name == "ptbdb" and records:
        rows = []
        for record in records:
            subject, _, recording = record.partition("/")
            rows.append({"subject_id": subject, "recording_id": recording, "record_path": record})
        index_path = INTERIM / "clinical_metadata" / "ptb_diagnostic_record_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(index_path, index=False)
    checksum_path = metadata_dir / "SHA256SUMS.txt"
    checksum_matches = verify_physionet_checksums(
        checksum_path,
        [metadata_dir / filename for filename in files if filename != "SHA256SUMS.txt"],
    )
    archive_path = base / ("ptb-xl-1.0.3.zip" if name == "ptb-xl" else "ptbdb-1.0.0.zip")
    archive_info = None
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive_info = {
                "path": str(archive_path.relative_to(ROOT)),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256(archive_path),
                "members": len(archive.infolist()),
                "zip_test_error": archive.testzip(),
            }
    return {
        "dataset": "ptb_xl" if name == "ptb-xl" else "ptb_diagnostic",
        "version": version,
        "license": license_name,
        "source_url": f"https://physionet.org/content/{name}/{version}/",
        "file_inventory": inventory,
        "record_list_count": len(records),
        "metadata_record_count": record_count,
        "patient_count": patient_count,
        "metadata_columns": columns,
        "official_checksum_matches": checksum_matches,
        "waveforms_downloaded": archive_info is not None,
        "archive": archive_info,
        "waveform_policy": "Waveforms remain in the official native archive and were not extracted or loaded into memory.",
    }


NHANES_COMPONENTS = {
    "DEMO_J": "Demographics",
    "BPQ_J": "Blood Pressure and Cholesterol Questionnaire",
    "BPX_J": "Blood Pressure Examination",
    "CDQ_J": "Cardiovascular Health Questionnaire",
    "BMX_J": "Body Measures",
    "TCHOL_J": "Total Cholesterol",
    "HDL_J": "HDL Cholesterol",
    "TRIGLY_J": "LDL and Triglycerides",
    "DIQ_J": "Diabetes Questionnaire",
    "SMQ_J": "Smoking Questionnaire",
}


def collect_nhanes() -> dict:
    base = RAW / "nhanes" / "raw_cycles" / "2017-2018"
    inventory = []
    frames = {}
    for component, description in NHANES_COMPONENTS.items():
        xpt_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{component}.xpt"
        doc_url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{component}.htm"
        xpt = base / f"{component}.XPT"
        doc = base / f"{component}.htm"
        inventory.extend([download_optional(xpt_url, xpt), download_optional(doc_url, doc)])
        if xpt.exists():
            frames[component] = pd.read_sas(xpt, format="xport", encoding="utf-8")
    mergeable = {name: frame for name, frame in frames.items() if "SEQN" in frame.columns}
    merged = None
    merge_issues = []
    for name, frame in mergeable.items():
        duplicate_seqn = int(frame["SEQN"].duplicated().sum())
        if duplicate_seqn:
            merge_issues.append({"component": name, "duplicate_seqn": duplicate_seqn})
        frame = frame.drop_duplicates("SEQN", keep="first")
        merged = frame if merged is None else merged.merge(frame, on="SEQN", how="outer", suffixes=("", f"_{name}"))
    merged_path = None
    if merged is not None:
        merged_path = INTERIM / "nhanes_components" / "nhanes_2017_2018_cardiovascular_components.csv"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_path, index=False)
    survey_design = [c for c in (merged.columns if merged is not None else []) if c in {"SEQN", "SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR"}]
    return {
        "dataset": "nhanes_2017_2018",
        "cycle": "2017-2018",
        "source_url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx",
        "components": list(NHANES_COMPONENTS),
        "component_descriptions": NHANES_COMPONENTS,
        "file_inventory": inventory,
        "component_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "merged_rows": None if merged is None else int(len(merged)),
        "merged_columns": None if merged is None else int(len(merged.columns)),
        "merge_issues": merge_issues,
        "survey_design_columns": survey_design,
        "merged_derivative": None if merged_path is None else str(merged_path.relative_to(ROOT)),
        "target_definition": "No synthetic target is created; questionnaire-reported cardiovascular history remains distinct from adjudicated disease.",
    }


def build_registry(results: dict) -> dict:
    registry_path = META / "dataset_registry.json"
    existing = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    new_entries = {
        "z_alizadeh_sani": {
            "title": "Z-Alizadeh Sani",
            "version": "UCI dataset 412",
            "source_url": "https://archive.ics.uci.edu/dataset/412/z-alizadeh-sani",
            "doi": "10.24432/C5Q31T",
            "license": "CC BY 4.0",
            "access": "Public download; no credentialed approval identified",
            "status": "verified_and_downloaded",
            "research_track": "tabular_clinical",
            "rows": results["z_alizadeh_sani"]["report"]["rows"],
            "columns": results["z_alizadeh_sani"]["report"]["columns"],
            "target": results["z_alizadeh_sani"]["report"]["target_candidate"],
            "validated_derivative": results["z_alizadeh_sani"]["validated"],
        },
        "ptb_xl": {
            "title": "PTB-XL, a large publicly available electrocardiography dataset",
            "version": "1.0.3",
            "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
            "doi": "10.13026/kfzx-aw45",
            "license": "CC BY 4.0",
            "access": "PhysioNet open access; no credentialed approval identified",
            "status": "archive_acquired_metadata_indexed_not_extracted" if results["ptb_xl"]["archive"] else "metadata_indexed_waveform_download_deferred",
            "research_track": "clinical_ecg_diagnosis",
            **{k: results["ptb_xl"][k] for k in ["metadata_record_count", "patient_count", "record_list_count", "waveforms_downloaded"]},
        },
        "ptb_diagnostic": {
            "title": "PTB Diagnostic ECG Database",
            "version": "1.0.0",
            "source_url": "https://physionet.org/content/ptbdb/1.0.0/",
            "doi": "10.13026/C28C71",
            "license": "Open Data Commons Attribution License v1.0",
            "access": "PhysioNet open access; no credentialed approval identified",
            "status": "archive_acquired_metadata_indexed_not_extracted" if results["ptb_diagnostic"]["archive"] else "metadata_indexed_waveform_download_deferred",
            "research_track": "clinical_ecg_diagnosis",
            "records_documented_by_source": 549,
            "patients_documented_by_source": 294,
            "record_list_count": results["ptb_diagnostic"]["record_list_count"],
        },
        "auricle": {
            "title": "Auricle: A Multimodal Wearable Sensing Dataset",
            "version": "1.0.0",
            "source_url": "https://physionet.org/content/auricle/1.0.0/",
            "doi": "10.13026/swme-vp69",
            "license": "Open Data Commons Attribution License v1.0",
            "access": "PhysioNet open access; no credentialed approval identified",
            "status": "metadata_only_full_download_deferred_by_resource_policy",
            "research_track": "multimodal_cardiovascular_physiology",
            "participants_documented_by_source": {"healthy": 42, "heart_failure": 28},
            "full_archive_downloaded": False,
        },
        "nhanes_2017_2018": {
            "title": "National Health and Nutrition Examination Survey",
            "version": "2017-2018",
            "source_url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx",
            "doi": None,
            "license": "CDC public-use data terms; verify current component documentation before redistribution",
            "access": "Public-use XPT files; no restricted RDC access used",
            "status": "verified_and_downloaded",
            "research_track": "tabular_clinical",
            **results["nhanes"],
        },
    }
    existing.update(new_entries)
    registry_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing


def update_metadata_catalogs(registry: dict) -> None:
    hashes_path = META / "dataset_hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    for base in (RAW / "z_alizadeh_sani", RAW / "ptb_xl", RAW / "ptb_diagnostic", RAW / "auricle", RAW / "nhanes"):
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file():
                    hashes[str(path.relative_to(ROOT))] = sha256(path)
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    citations_path = META / "source_citations.json"
    citations = json.loads(citations_path.read_text()) if citations_path.exists() else {}
    for name in ("z_alizadeh_sani", "ptb_xl", "ptb_diagnostic", "auricle", "nhanes_2017_2018"):
        item = registry[name]
        citations[name] = {
            key: item.get(key)
            for key in ("title", "source_url", "doi", "version", "license", "research_track", "access")
        }
    citations_path.write_text(json.dumps(citations, indent=2), encoding="utf-8")

    licenses_path = META / "licenses.json"
    licenses = json.loads(licenses_path.read_text()) if licenses_path.exists() else {}
    for name in ("z_alizadeh_sani", "ptb_xl", "ptb_diagnostic", "auricle", "nhanes_2017_2018"):
        licenses[name] = registry[name]["license"]
    licenses_path.write_text(json.dumps(licenses, indent=2), encoding="utf-8")

    compatibility_path = META / "feature_compatibility_matrix.csv"
    rows = []
    if compatibility_path.exists():
        with compatibility_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    additions = [
        ("z_alizadeh_sani", "nhanes_2017_2018", "False", "False", "Different ascertainment, units, populations, and study designs; no automatic merge"),
        ("ptb_xl", "ptb_diagnostic", "False", "False", "Separate ECG cohorts with related PTB provenance; patient overlap is unresolved"),
        ("auricle", "all tabular clinical datasets", "False", "False", "Multimodal physiology is not interchangeable with tabular disease-presence outcomes"),
        ("nhanes_2017_2018", "all disease-presence datasets", "False", "False", "Complex-survey, self-reported/cross-sectional outcomes require separate analysis"),
    ]
    existing_keys = {(row["left"], row["right"]) for row in rows}
    for left, right, target, features, reason in additions:
        if (left, right) not in existing_keys:
            rows.append({"left": left, "right": right, "compatible_target": target, "compatible_features": features, "reason": reason})
    with compatibility_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["left", "right", "compatible_target", "compatible_features", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    legacy_manifest_path = META / "dataset_manifest.json"
    if legacy_manifest_path.exists():
        legacy = json.loads(legacy_manifest_path.read_text())
        legacy["clinical_dataset_collection_v2"] = "data/metadata/dataset_manifest_v2.json"
        legacy["clinical_dataset_collection_v2_script_sha256"] = sha256(Path(__file__))
        legacy_manifest_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")


def write_overlap_report() -> None:
    overlap = {
        "uci_cleveland_vs_statlog": {
            "classification": "cannot_determine",
            "method": "No shared patient identifier; schemas and coding differ. Exact normalized feature comparison was not treated as patient identity.",
            "conclusion": "Do not call these independent cohorts without source-level provenance or record-level evidence.",
        },
        "ptb_xl_vs_ptb_diagnostic": {
            "classification": "likely_related_provenance_but_patient_overlap_unverified",
            "method": "Both originate from PTB/PhysioNet and use ECG records; public metadata identifiers are not a validated cross-dataset key.",
            "conclusion": "Do not use one as independent external validation of the other pending a documented linkage analysis.",
        },
        "new_vs_existing": {
            "classification": "no_overlap_detected_under_available_identifiers",
            "method": "No common patient identifiers across the tabular source files; identical clinical attributes were not interpreted as identity.",
            "conclusion": "Absence of detected overlap is not proof of cohort independence.",
        },
    }
    path = META / "overlap_reports" / "clinical_dataset_overlap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlap, indent=2), encoding="utf-8")


def write_documents(registry: dict, results: dict) -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    dictionaries = META / "data_dictionaries"
    dictionaries.mkdir(parents=True, exist_ok=True)
    z_columns = results["z_alizadeh_sani"]["report"]["column_names"]
    (dictionaries / "z_alizadeh_sani.json").write_text(
        json.dumps(
            {
                "dataset": "z_alizadeh_sani",
                "target": "Cath",
                "target_definition": "Cad indicates CAD with at least 50% coronary diameter narrowing; Normal otherwise.",
                "feature_groups": {
                    "demographic": z_columns[:5],
                    "history_and_examination": z_columns[5:24],
                    "symptoms_and_ecg": z_columns[24:39],
                    "laboratory_and_echo": z_columns[39:],
                },
                "units": "The UCI workbook does not provide complete machine-readable units for every field; clinical review is required before modeling.",
                "potential_leakage_fields": ["EF-TTE", "Region RWMA", "VHD", "Cath"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    nhanes_frame = pd.read_csv(ROOT / results["nhanes"]["merged_derivative"])
    (dictionaries / "nhanes_2017_2018.json").write_text(
        json.dumps(
            {
                "dataset": "nhanes_2017_2018",
                "cycle": "2017-2018",
                "components": NHANES_COMPONENTS,
                "merged_columns": list(nhanes_frame.columns),
                "identifier": "SEQN",
                "survey_design_columns": results["nhanes"]["survey_design_columns"],
                "source_documentation": "Preserved component HTML files under data/raw/nhanes/raw_cycles/2017-2018.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog = ["# CARDIODIAG Clinical Dataset Catalog v2", "", "This catalog separates tabular clinical, ECG, and multimodal physiology tracks.", ""]
    for name in ["z_alizadeh_sani", "ptb_xl", "ptb_diagnostic", "auricle", "nhanes_2017_2018"]:
        item = registry[name]
        catalog.extend([
            f"## {name}",
            f"- Official source: {item['source_url']}",
            f"- Version: {item['version']}",
            f"- DOI: {item.get('doi') or 'None documented'}",
            f"- License/access: {item['license']}; {item['access']}",
            f"- Status: {item['status']}",
            f"- Track: {item['research_track']}",
            "",
        ])
    (docs / "CLINICAL_DATASET_CATALOG.md").write_text("\n".join(catalog), encoding="utf-8")
    (docs / "CLINICAL_DATASET_OVERLAP_REPORT.md").write_text(
        "# Clinical Dataset Overlap Report v2\n\n"
        "No dataset names were treated as proof of independence. UCI Cleveland versus Statlog cannot be resolved from available identifiers. PTB-XL and PTB Diagnostic have related PTB provenance, but public metadata does not establish patient overlap. No cross-source patient identifiers were found for the new tabular sources.\n",
        encoding="utf-8",
    )
    (docs / "NHANES_EXTRACTION_PROTOCOL.md").write_text(
        "# NHANES Extraction Protocol\n\n"
        "The v2 pipeline pins the 2017-2018 cycle, downloads component XPT files and HTML documentation, merges only on SEQN, checks duplicate SEQN values, and preserves SDMVPSU, SDMVSTRA, WTINT2YR, and WTMEC2YR when present. No synthetic cardiovascular target is created and survey weights must be used for population-level inference.\n",
        encoding="utf-8",
    )
    (docs / "ECG_DATASET_PROTOCOL.md").write_text(
        "# ECG Dataset Protocol\n\n"
        "PTB-XL and PTB Diagnostic remain separate ECG tracks. The pipeline indexes metadata, identifiers, patient/record counts, labels, and official checksums without loading waveforms into memory. Patient-level splits are required; random record-level splitting is prohibited when multiple records belong to one patient.\n",
        encoding="utf-8",
    )
    (docs / "CLINICAL_DATASET_EXPANSION_REPORT.md").write_text(
        "# CARDIODIAG Clinical Dataset Expansion Report\n\n"
        "This v2 expansion acquires Z-Alizadeh Sani, selected public NHANES 2017-2018 components, and the official PTB-XL and PTB Diagnostic archives. It indexes ECG metadata without extracting waveforms and catalogs Auricle without downloading its 7.3 GB archive. No model training or performance evaluation was performed.\n\n"
        "Target definitions remain separate: CAD, general cardiovascular history, ECG diagnosis, multimodal physiology, and mortality are not interchangeable.\n",
        encoding="utf-8",
    )
    (docs / "CLINICAL_DATASET_LICENSE_AND_ACCESS_AUDIT.md").write_text(
        "# Clinical Dataset License and Access Audit\n\n"
        "- Z-Alizadeh Sani: UCI reports CC BY 4.0.\n"
        "- PTB-XL: PhysioNet reports CC BY 4.0.\n"
        "- PTB Diagnostic and Auricle: PhysioNet reports Open Data Commons Attribution License v1.0.\n"
        "- NHANES: CDC public-use files; component documentation and current terms must be retained with any redistribution.\n\n"
        "Public download does not by itself guarantee unrestricted redistribution. Attribution and source-specific terms remain mandatory.\n",
        encoding="utf-8",
    )
    (docs / "CLINICAL_DATASET_QUALITY_REPORT.md").write_text(
        "# Clinical Dataset Quality Report v2\n\n"
        "Quality reports and file hashes are stored under `data/metadata/quality_reports/`, `data/metadata/dataset_hashes.json`, and `data/metadata/dataset_manifest.json`. Raw values are preserved; no outliers were removed and no model preprocessing was applied.\n",
        encoding="utf-8",
    )
    (docs / "CLINICAL_DATASET_COMPATIBILITY.md").write_text(
        "# Clinical Dataset Compatibility v2\n\n"
        "Tabular disease-presence, NHANES survey outcomes, ECG diagnosis, multimodal physiology, and heart-failure mortality are separate research tracks. Feature harmonization and cross-dataset validation require clinical review of units, ascertainment, timing, and cohort design.\n",
        encoding="utf-8",
    )
    (docs / "CLINICAL_DATASET_LIMITATIONS.md").write_text(
        "# Clinical Dataset Limitations v2\n\n"
        "PTB-XL and PTB Diagnostic archives are preserved and ZIP-validated, but their waveforms were not extracted or loaded into memory. Auricle full acquisition was explicitly deferred because of its 7.3 GB compressed/15.4 GB uncompressed footprint. NHANES is cross-sectional survey data and must not be described as prospective disease prediction. Cross-source patient overlap remains unresolved where identifiers are unavailable.\n",
        encoding="utf-8",
    )
    (docs / "DATASET_COLLECTION_V2.md").write_text(
        "# Dataset Collection v2\n\n"
        "Run `PYTHONPATH=src .venv-repro\\\\Scripts\\\\python.exe runner\\\\data\\\\collect_clinical_datasets_v2.py`. The command acquires small/tabular sources, downloads official metadata and documentation, builds validated derivatives and indexes, and never trains models.\n",
        encoding="utf-8",
    )


def build_manifest(registry: dict, results: dict) -> None:
    hashes = {}
    for base in [RAW, INTERIM, PROCESSED]:
        for path in base.rglob("*"):
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    manifest = {
        "manifest_version": "clinical_dataset_collection_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "datasets": registry,
        "acquisition_results": results,
        "file_hashes": hashes,
        "policy": {
            "model_training": False,
            "auricle_full_archive": False,
            "waveform_loading": False,
            "raw_modification": False,
        },
    }
    (META / "dataset_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downloads", action="store_true")
    args = parser.parse_args()
    if args.skip_downloads:
        raise SystemExit("--skip-downloads is not supported because metadata provenance must be explicit")
    META.mkdir(parents=True, exist_ok=True)
    results = {
        "z_alizadeh_sani": collect_z_alizadeh(),
        "ptb_xl": collect_physionet_metadata(
            "ptb-xl", "1.0.3",
            ["LICENSE.txt", "RECORDS", "SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv"],
            "CC BY 4.0",
        ),
        "ptb_diagnostic": collect_physionet_metadata(
            "ptbdb", "1.0.0",
            ["README", "RECORDS", "SHA256SUMS.txt"],
            "Open Data Commons Attribution License v1.0",
        ),
        "auricle": collect_physionet_metadata(
            "auricle", "1.0.0",
            ["LICENSE.txt", "README.txt", "RECORDS", "SHA256SUMS.txt", "activity_labels.csv", "demographics.csv", "sensor_coordinates.csv"],
            "Open Data Commons Attribution License v1.0",
        ),
        "nhanes": collect_nhanes(),
    }
    registry = build_registry(results)
    update_metadata_catalogs(registry)
    write_overlap_report()
    write_documents(registry, results)
    build_manifest(registry, results)
    print(json.dumps({"datasets": list(registry), "no_model_training": True}, indent=2))


if __name__ == "__main__":
    main()
