"""Independent aggregate raw-source checks; no model fitting or patient output."""
from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "research_publication_plan/data_screening/angiocad"
VESSELS = ["LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig", "Prox LCX", "Mid LCX", "Dist LCX", "OM", "Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"]
ALLOWED = {"NL", "1-25", "26-50", "51-75", "76-90", "91-99", "100"}
SEVERE = {"51-75", "76-90", "91-99", "100"}
MAPPING = {"Age": "Age", "Sex": "Sex", "DM": "DM", "HTN": "HTN", "Current Smoker": "Current Smoker", "TG": "TG", "HDL": "HDL", "LDL": "LDL", "FBS": "FBS", "CR": "Cr", "BUN": "BUN", "HB": "Hgb", "WBC": "leukocytes", "Lymph": "Lymph", "Neut": "Neut", "PLT": "PLT"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name, source in mapping.items():
        if name == "Sex":
            out[name] = frame[source].astype(str).str.strip().replace({"Fmale": "Female"})
        else:
            out[name] = pd.to_numeric(frame[source], errors="raise")
    return out


def fingerprints(frame: pd.DataFrame) -> set[str]:
    # Values, not dataset-specific identifiers. Missing values remain explicit.
    return {"|".join("<NA>" if pd.isna(v) else format(float(v), ".12g") if isinstance(v, (float, int)) else str(v) for v in row) for row in frame.itertuples(index=False, name=None)}


def main() -> None:
    raw = next((ROOT / "data/raw/z_alizadeh_sani/extracted").glob("*.xlsx"))
    sheets = pd.read_excel(raw, sheet_name=None)
    zas = pd.read_csv(ROOT / "data/interim/validated/z_alizadeh_sani.csv")
    primary = next(frame for frame in sheets.values() if len(frame) == 303)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pd.read_excel(SOURCE / "AngioCAD_Features.xlsx").dropna(subset=["ID"])
        labels = pd.read_excel(SOURCE / "AngioCAD_Labels.xlsx").dropna(subset=["ID"])
    features.columns = features.columns.astype(str).str.strip()
    labels.columns = labels.columns.astype(str).str.strip()
    assert not features.ID.duplicated().any()
    assert not labels.ID.duplicated().any()
    joined = labels.set_index("ID").loc[features.ID]
    observed = set(labels[VESSELS].stack().astype(str).str.strip())
    assert observed <= ALLOWED
    assert not labels[VESSELS].isna().any().any()
    positive = joined[VESSELS].isin(SEVERE).any(axis=1)
    ambiguous = ~positive & joined[VESSELS].eq("26-50").any(axis=1)
    mapped_zas = canonical(zas, {x: x for x in MAPPING})
    mapped_angio = canonical(features, MAPPING)
    all_shared = list(MAPPING)
    core = ["Age", "Sex", "DM", "HTN", "Current Smoker"]
    qc = {
        "model_fits": 0,
        "source_zas_workbook_sha256": digest(raw),
        "source_angio_features_sha256": digest(SOURCE / "AngioCAD_Features.xlsx"),
        "source_angio_labels_sha256": digest(SOURCE / "AngioCAD_Labels.xlsx"),
        "zas_sheets": {name: {"rows": len(frame), "columns": len(frame.columns)} for name, frame in sheets.items()},
        "zas_primary_matches_validated_dataframe": primary.equals(zas),
        "zas_predictor_columns_excluding_Cath": len(zas.columns) - 1,
        "zas_full_record_duplicates": int(zas.duplicated().sum()),
        "zas_predictor_duplicates": int(zas.drop(columns="Cath").duplicated().sum()),
        "angio_feature_rows": len(features),
        "angio_label_rows": len(labels),
        "angio_matched_rows": len(joined),
        "angio_case_id_duplicates_features": int(features.ID.duplicated().sum()),
        "angio_case_id_duplicates_labels": int(labels.ID.duplicated().sum()),
        "angio_label_only_case_ids": len(set(labels.ID) - set(features.ID)),
        "angio_vessel_label_missing_cells": int(labels[VESSELS].isna().sum().sum()),
        "angio_vessel_allowed_tokens": sorted(observed),
        "angio_embedded_vessel_labels_match_label_workbook": features.set_index("ID")[VESSELS].astype(str).equals(joined[VESSELS].astype(str)),
        "angio_gt50_matched_positive": int(positive.sum()),
        "angio_gt50_matched_negative": int((~positive).sum()),
        "angio_negative_26_50_boundary_ambiguous": int(ambiguous.sum()),
        "angio_features_duplicates_without_ID": int(features.drop(columns="ID").duplicated().sum()),
        "angio_core_missingness": mapped_angio[core].isna().sum().to_dict(),
        "zas_core_fingerprint_collisions_with_angio": len(fingerprints(mapped_zas[core]) & fingerprints(mapped_angio[core])),
        "zas_all16_name_mapped_fingerprint_collisions_with_angio": len(fingerprints(mapped_zas[all_shared]) & fingerprints(mapped_angio[all_shared])),
        "name_mapping_is_not_clinical_unit_or_timing_equivalence": True,
        "fingerprint_noncollision_does_not_prove_patient_independence": True,
        "limited_core_collisions_could_occur_by_chance": True,
    }
    official = OUT / "official_zas_20261114.zip"
    if official.exists():
        with zipfile.ZipFile(io.BytesIO(official.read_bytes())) as archive:
            member = next(name for name in archive.namelist() if name.endswith(".xlsx"))
            qc["new_official_workbook_byte_matches_existing_raw"] = archive.read(member) == raw.read_bytes()
    (OUT / "independent_raw_checks.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
