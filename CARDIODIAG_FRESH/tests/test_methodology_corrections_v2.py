import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from runner.methodology_corrections_v2.split_manifest import (
    create_nested_split_manifest,
    load_and_validate_nested_split_manifest,
    save_nested_split_manifest,
    validate_nested_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "methodology_corrections_v2"


def test_nested_manifest_has_exact_outer_and_inner_coverage(tmp_path):
    manifest = create_nested_split_manifest([0, 1] * 30, [42, 42], outer_folds=5, inner_folds=4)
    assert validate_nested_split_manifest(manifest)["valid"] is True
    assert manifest["format_version"] == "nested_split_manifest_v2"
    assert all("outer_fold" in row for row in manifest["inner_assignments"])
    path = tmp_path / "manifest.json"
    save_nested_split_manifest(path, manifest)
    loaded = load_and_validate_nested_split_manifest(path)
    assert loaded["validation"]["valid"] is True


def test_manifest_rejects_duplicate_source_row_ids():
    with pytest.raises(ValueError, match="source_row_ids must be unique"):
        create_nested_split_manifest([0, 1, 0, 1], [1], outer_folds=2, inner_folds=2, source_row_ids=["x"] * 4)


def test_manifest_rejects_inner_rows_outside_outer_training():
    manifest = create_nested_split_manifest([0, 1] * 10, [42], outer_folds=2, inner_folds=2)
    manifest["inner_assignments"][0]["row_index"] = 999
    validation = validate_nested_split_manifest(manifest)
    assert validation["valid"] is False
    assert any("outside outer training" in error for error in validation["errors"])


def test_stage6_outputs_have_required_schemas_and_hashes():
    manifest = json.loads((OUT / "METHODOLOGY_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["new_model_training"] is False
    assert manifest["policy"]["historical_results_modified"] is False
    assert manifest["policy"]["patient_ids_created"] is False
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative

    required_csvs = {
        "ISSUE_RESOLUTION_MATRIX.csv": {"issue", "classification", "evidence", "decision"},
        "MODEL_PROBABILITY_POLICY.csv": {"model_family", "native_output", "primary_calibration_policy"},
        "UNCERTAINTY_REPORTING_POLICY.csv": {"quantity", "uncertainty_source", "Version_1_status"},
        "STATISTICAL_COMPARISON_POLICY.csv": {"analysis", "method", "recommended"},
        "FEATURE_POLICY_FINAL.csv": {"feature", "category", "primary_inclusion"},
    }
    for name, columns in required_csvs.items():
        assert columns <= set(pd.read_csv(OUT / name, nrows=0).columns)


def test_stage6_manifest_contains_no_fake_patient_identifier():
    split = json.loads((OUT / "reproducibility" / "split_manifest_v2.json").read_text(encoding="utf-8"))
    assert "patient_id" not in json.dumps(split).lower()
    assert split["source_row_identifier_policy"].endswith("not a patient identifier")
