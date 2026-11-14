"""Build the CARDIODIAG dataset registry and diagnostic derived datasets.

Raw archives and source files are never modified. The script is deterministic
apart from timestamps and writes all derived artifacts under data/ and docs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cardiodiag.data.quality import quality_summary


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def prepare_sources() -> dict[str, dict]:
    heart5 = read_csv(ROOT / "reference" / "author_release" / "heart5.csv")
    mendeley_local = read_csv(ROOT / "reference" / "author_release" / "MendeleyCardiovascular_Disease_Dataset.csv")
    mendeley_official = read_csv(RAW / "mendeley" / "Cardiovascular_Disease_Dataset_official.csv")
    kaggle = read_csv(RAW / "kaggle_cvd_70000" / "official_files" / "cardio_train.csv", sep=";")
    uci_heart = read_csv(
        RAW / "uci_heart_disease" / "official_files" / "processed.cleveland.data",
        header=None,
        na_values=["?"],
    )
    uci_heart.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    statlog = read_csv(
        RAW / "statlog_heart" / "official_files" / "heart.dat",
        sep=r"\s+",
        header=None,
    )
    statlog.columns = [
        "age", "sex", "chest_pain", "rest_bp", "serum_chol",
        "fasting_blood_sugar", "rest_ecg", "max_heart_rate",
        "exercise_angina", "oldpeak", "slope", "major_vessels", "thal",
        "target",
    ]
    failure = read_csv(
        RAW / "uci_heart_failure" / "official_files" / "heart_failure_clinical_records_dataset.csv"
    )
    return {
        "five_states": {
            "frame": heart5,
            "target": "HeartDisease",
            "raw_files": [ROOT / "reference" / "author_release" / "heart5.csv"],
        },
        "mendeley": {
            "frame": mendeley_local,
            "target": "target",
            "raw_files": [
                ROOT / "reference" / "author_release" / "MendeleyCardiovascular_Disease_Dataset.csv",
                RAW / "mendeley" / "Cardiovascular_Disease_Dataset_official.csv",
            ],
            "official_frame": mendeley_official,
        },
        "kaggle_cvd_70000": {
            "frame": kaggle,
            "target": "cardio",
            "raw_files": [
                RAW / "kaggle_cvd_70000" / "cardiovascular-disease-dataset.zip",
                RAW / "kaggle_cvd_70000" / "official_files" / "cardio_train.csv",
            ],
        },
        "uci_heart_disease_cleveland": {
            "frame": uci_heart,
            "target": "num",
            "raw_files": [
                RAW / "uci_heart_disease" / "heart_disease.zip",
                RAW / "uci_heart_disease" / "official_files" / "processed.cleveland.data",
            ],
        },
        "statlog_heart": {
            "frame": statlog,
            "target": "target",
            "raw_files": [
                RAW / "statlog_heart" / "statlog_heart.zip",
                RAW / "statlog_heart" / "official_files" / "heart.dat",
            ],
        },
        "uci_heart_failure": {
            "frame": failure,
            "target": "DEATH_EVENT",
            "raw_files": [
                RAW / "uci_heart_failure" / "heart_failure_clinical_records.zip",
                RAW / "uci_heart_failure" / "official_files" / "heart_failure_clinical_records_dataset.csv",
            ],
        },
    }


def source_catalog() -> dict[str, dict]:
    return {
        "five_states": {
            "title": "Five States / Heart Disease Prediction",
            "source_url": None,
            "doi": None,
            "authors": "Source not established from the local author release",
            "version": "local author-release copy",
            "license": "Unknown; do not redistribute beyond project permissions",
            "role": "original baseline disease-presence dataset",
            "target_meaning": "HeartDisease as supplied by the author release; exact upstream definition requires provenance confirmation",
        },
        "mendeley": {
            "title": "Cardiovascular_Disease_Dataset",
            "source_url": "https://data.mendeley.com/datasets/dzz48mvjht/1",
            "doi": "10.17632/dzz48mvjht.1",
            "authors": "Bhanu Prakash Doppala; Debnath Bhattacharyya",
            "publication_date": "2021-04-16",
            "version": "1",
            "license": "CC BY 4.0",
            "role": "original baseline disease-presence dataset",
            "target_meaning": "Supplied target for early-stage heart disease detection; source description reports 1,000 subjects and 12 features, while the file includes patientid and 13 predictors",
        },
        "kaggle_cvd_70000": {
            "title": "Cardiovascular Disease Dataset",
            "source_url": "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
            "doi": None,
            "authors": "Svetlana Ulianova",
            "version": "downloaded from official Kaggle dataset endpoint",
            "license": "Kaggle dataset license not independently resolved; retain attribution and check terms before redistribution",
            "role": "additional disease-presence classification benchmark",
            "target_meaning": "cardio: cardiovascular disease presence as supplied by the dataset",
        },
        "uci_heart_disease_cleveland": {
            "title": "Heart Disease",
            "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
            "doi": "10.24432/C52P4X",
            "authors": "Andras Janosi; William Steinbrunn; Matthias Pfisterer; Robert Detrano",
            "publication_date": "1988-06-30 donation; 1989 introductory publication",
            "version": "UCI repository download; processed Cleveland file",
            "license": "CC BY 4.0",
            "role": "additional/source-held-out disease-presence benchmark",
            "target_meaning": "num: original 0–4 angiographic diagnosis; binary presence mapping must remain derived",
        },
        "statlog_heart": {
            "title": "Statlog (Heart)",
            "source_url": "https://archive.ics.uci.edu/dataset/145/statlog+heart",
            "doi": "10.24432/C57303",
            "authors": "UCI repository record; original contributor attribution requires the included documentation",
            "version": "UCI repository download",
            "license": "CC BY 4.0",
            "role": "additional disease-presence benchmark; overlap with UCI Heart must be assessed",
            "target_meaning": "Original target 1=absence and 2=presence according to UCI documentation; no recoding in raw data",
        },
        "uci_heart_failure": {
            "title": "Heart Failure Clinical Records",
            "source_url": "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
            "doi": "10.24432/C5Z89R",
            "authors": "Davide Chicco; Giuseppe Jurman",
            "publication_date": "2020",
            "version": "UCI repository download",
            "license": "CC BY 4.0",
            "role": "separate cardiovascular outcome task; not disease-presence classification",
            "target_meaning": "DEATH_EVENT: death during follow-up, not disease presence",
        },
    }


def build() -> dict:
    sources = prepare_sources()
    catalog = source_catalog()
    (META / "data_dictionaries").mkdir(parents=True, exist_ok=True)
    (META / "quality_reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "interim" / "validated").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed" / "disease_classification").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed" / "separate_outcome_tasks").mkdir(parents=True, exist_ok=True)
    registry = {}
    hashes = {}
    quality = {}
    for name, spec in sources.items():
        frame = spec["frame"]
        target = spec["target"]
        implausible = {}
        if name == "kaggle_cvd_70000":
            implausible = {
                "ap_hi_nonpositive_or_over_250": int(((frame["ap_hi"] <= 0) | (frame["ap_hi"] > 250)).sum()),
                "ap_lo_nonpositive_or_over_200": int(((frame["ap_lo"] <= 0) | (frame["ap_lo"] > 200)).sum()),
                "height_nonpositive_or_outside_100_250_cm": int(((frame["height"] < 100) | (frame["height"] > 250)).sum()),
                "weight_nonpositive_or_over_300_kg": int(((frame["weight"] <= 0) | (frame["weight"] > 300)).sum()),
            }
        q = quality_summary(frame, target=target, dataset=name, implausible=implausible)
        quality[name] = q
        out_dir = ROOT / "data" / "interim" / "validated"
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        for raw in spec["raw_files"]:
            hashes[str(raw.relative_to(ROOT))] = sha256(raw)
        registry[name] = {
            **catalog[name],
            "target": target,
            "rows": len(frame),
            "columns": len(frame.columns),
            "raw_files": [str(path.relative_to(ROOT)) for path in spec["raw_files"]],
            "validated_derivative": str((out_dir / f"{name}.csv").relative_to(ROOT)),
            "quality_report": str((META / "quality_reports" / f"{name}.json").relative_to(ROOT)),
        }
        (META / "quality_reports" / f"{name}.json").write_text(
            json.dumps(q, indent=2), encoding="utf-8"
        )
        dictionary = {
            column: {
                "original_name": column,
                "dtype": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
                "predictor": column != target and column.lower() not in {"patientid", "id"},
            }
            for column in frame.columns
        }
        (META / "data_dictionaries" / f"{name}.json").write_text(
            json.dumps({"dataset": name, "target": target, "columns": dictionary}, indent=2),
            encoding="utf-8",
        )
    # Keep a derived binary Cleveland target separate from the original num.
    cleveland = sources["uci_heart_disease_cleveland"]["frame"].copy()
    cleveland["disease_present"] = (pd.to_numeric(cleveland["num"], errors="coerce") > 0).astype("Int64")
    cleveland.to_csv(
        ROOT / "data" / "processed" / "disease_classification" / "uci_heart_disease_cleveland_binary.csv",
        index=False,
    )
    failure = sources["uci_heart_failure"]["frame"]
    failure.to_csv(
        ROOT / "data" / "processed" / "separate_outcome_tasks" / "uci_heart_failure.csv",
        index=False,
    )
    m_local = sources["mendeley"]["frame"]
    m_official = sources["mendeley"]["official_frame"]
    registry["mendeley"]["official_comparison"] = {
        "same_schema": list(m_local.columns) == list(m_official.columns),
        "same_shape": m_local.shape == m_official.shape,
        "same_sha256": sha256(sources["mendeley"]["raw_files"][0]) == sha256(sources["mendeley"]["raw_files"][1]),
        "local_columns": list(m_local.columns),
        "official_columns": list(m_official.columns),
        "local_sha256": sha256(sources["mendeley"]["raw_files"][0]),
        "official_sha256": sha256(sources["mendeley"]["raw_files"][1]),
    }
    compatibility = [
        {"left": "five_states", "right": "mendeley", "compatible_target": True, "compatible_features": False, "reason": "Different schemas and clinical coding; no automatic merge"},
        {"left": "five_states", "right": "kaggle_cvd_70000", "compatible_target": True, "compatible_features": False, "reason": "Different feature names, measurement protocols, and coding; harmonization requires explicit mapping"},
        {"left": "uci_heart_disease_cleveland", "right": "statlog_heart", "compatible_target": True, "compatible_features": "partial", "reason": "Overlapping heart-disease concepts and feature families, but different encodings and possible source overlap"},
        {"left": "uci_heart_failure", "right": "all disease-presence datasets", "compatible_target": False, "compatible_features": False, "reason": "DEATH_EVENT is follow-up mortality, not disease presence"},
    ]
    (META / "dataset_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (META / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (META / "source_citations.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (META / "licenses.json").write_text(json.dumps({key: value["license"] for key, value in catalog.items()}, indent=2), encoding="utf-8")
    pd.DataFrame(compatibility).to_csv(META / "feature_compatibility_matrix.csv", index=False)
    (ROOT / "docs" / "DATASET_CATALOG.md").write_text(
        "# CARDIODIAG dataset catalog\n\n"
        + "\n".join(
            f"## {name}\n\n- Title: {catalog[name]['title']}\n- Source: {catalog[name]['source_url'] or 'Unresolved local provenance'}\n"
            f"- DOI: {catalog[name]['doi'] or 'none'}\n- License: {catalog[name]['license']}\n"
            f"- Rows × columns: {registry[name]['rows']} × {registry[name]['columns']}\n"
            f"- Target: `{registry[name]['target']}`; role: {catalog[name]['role']}\n"
            for name in registry
        )
        + "\nRaw files are preserved under `data/raw/` or their existing reference path. See the registry and hashes for exact provenance.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "DATA_DICTIONARY.md").write_text(
        "# CARDIODIAG data dictionary\n\n"
        "Machine-readable per-dataset dictionaries are under `data/metadata/data_dictionaries/`. "
        "Original target columns and codings are retained. The UCI Cleveland binary target is derived "
        "from original `num` and is stored separately.\n\n"
        + "\n".join(
            f"- `{name}`: target `{spec['target']}`, columns: {', '.join(spec['frame'].columns)}"
            for name, spec in sources.items()
        ) + "\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "DATASET_PREPARATION_PROTOCOL.md").write_text(
        "# Dataset preparation protocol\n\n"
        "Official archives are downloaded into `data/raw/` and never edited. ZIP contents are extracted "
        "for inspection. Validated CSV derivatives are deterministic copies with no automatic row deletion. "
        "Quality reports record missingness, duplicates, identifiers, ranges, quantiles, constants, and "
        "dataset-specific implausible-value counts. Disease-presence targets are not merged with mortality "
        "targets. Any harmonization must use an explicit clinical mapping and remain derived.\n\n"
        "The script is `runner/data/prepare_dataset_collection.py` and uses `.venv-repro`.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "DATASET_QUALITY_REPORT.md").write_text(
        "# Dataset quality report\n\n"
        + "\n".join(
            f"- `{name}`: {q['rows']} rows × {q['columns']} columns; missing values total "
            f"{sum(q['missing_values'].values())}; exact duplicate occurrences "
            f"{q['duplicates']['exact_duplicate_occurrences']}; predictor duplicate occurrences "
            f"{q['duplicates']['predictor_duplicate_occurrences']}; implausible flags "
            f"{q['implausible_value_counts']}"
            for name, q in quality.items()
        )
        + "\n\nNo records are automatically removed. Detailed JSON reports are under `data/metadata/quality_reports/`.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "DATASET_COMPATIBILITY.md").write_text(
        "# Dataset compatibility\n\n"
        "The compatibility matrix is `data/metadata/feature_compatibility_matrix.csv`. "
        "Five States and Mendeley are baseline disease-presence datasets but have different schemas. "
        "Kaggle CVD is a separate 70,000-row disease-presence benchmark. UCI Cleveland and Statlog "
        "share heart-disease concepts but may overlap and require source-level overlap analysis. "
        "UCI Heart Failure is a separate mortality/follow-up task and must not be merged.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "DATASET_LIMITATIONS.md").write_text(
        "# Dataset limitations\n\n"
        "- Five States upstream provenance and license are unresolved from the local author release.\n"
        "- Kaggle access succeeded, but its license terms must be checked before redistribution.\n"
        "- Similar UCI heart datasets may share records or source populations; they are not automatically independent validation sets.\n"
        "- No clinical outcome target should be equated with disease presence.\n"
        "- Range flags are diagnostic only and do not justify deletion.\n",
        encoding="utf-8",
    )
    return {"registry": registry, "quality": quality, "hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = build()
    print(json.dumps({"datasets": list(result["registry"]), "raw_files": len(result["hashes"])}, indent=2))


if __name__ == "__main__":
    main()
