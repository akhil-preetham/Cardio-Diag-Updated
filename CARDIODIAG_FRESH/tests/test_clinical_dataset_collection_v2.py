import json
from pathlib import Path

import pandas as pd

from runner.data.collect_clinical_datasets_v2 import parse_physionet_records, sha256, verify_physionet_checksums


ROOT = Path(__file__).resolve().parents[1]


def test_physionet_record_parser_ignores_blank_lines(tmp_path):
    records = tmp_path / "RECORDS"
    records.write_text("00001\n\n00002\n", encoding="utf-8")
    assert parse_physionet_records(records) == ["00001", "00002"]


def test_official_checksum_parser_validates_local_file(tmp_path):
    payload = tmp_path / "metadata.csv"
    payload.write_bytes(b"clinical metadata\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{sha256(payload)}  metadata.csv\n", encoding="utf-8")
    assert verify_physionet_checksums(checksums, [payload]) == {"metadata.csv": True}


def test_z_alizadeh_derivative_preserves_source_shape_and_target():
    frame = pd.read_csv(ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv")
    assert frame.shape == (303, 56)
    assert set(frame["Cath"].dropna()) == {"Cad", "Normal"}


def test_ptb_xl_metadata_is_patient_level_indexed():
    frame = pd.read_csv(ROOT / "data" / "interim" / "clinical_metadata" / "ptb_xl_metadata_index.csv")
    assert len(frame) == 21799
    assert frame["patient_id"].nunique() == 18869
    assert frame["strat_fold"].between(1, 10).all()


def test_nhanes_derivative_preserves_survey_design_columns():
    frame = pd.read_csv(
        ROOT / "data" / "interim" / "nhanes_components" / "nhanes_2017_2018_cardiovascular_components.csv"
    )
    assert "SEQN" in frame
    for column in ("SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR"):
        assert column in frame


def test_auricle_is_metadata_only():
    auricle = ROOT / "data" / "raw" / "auricle"
    assert (auricle / "metadata" / "demographics.csv").exists()
    assert not list(auricle.rglob("*.dat"))
    assert not list(auricle.rglob("*.hea"))


def test_v2_manifest_records_no_model_training():
    manifest = json.loads((ROOT / "data" / "metadata" / "dataset_manifest_v2.json").read_text())
    assert manifest["policy"]["model_training"] is False
    assert manifest["policy"]["waveform_loading"] is False
    assert manifest["datasets"]["nhanes_2017_2018"]["cycle"] == "2017-2018"
