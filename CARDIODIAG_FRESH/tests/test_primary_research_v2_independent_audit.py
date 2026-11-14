import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_research_v2_independent_audit"


def test_audit_manifest_reports_complete_recomputation():
    manifest = json.loads((OUT / "AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["checks"]["outer_manifest_equal"] is True
    assert manifest["checks"]["inner_manifest_equal"] is True
    assert manifest["checks"]["metric_rows_checked"] == 3150
    assert manifest["checks"]["metric_failures"] == 0
    assert manifest["checks"]["calibration_rows_checked"] == 1125
    assert manifest["checks"]["calibration_failures"] == 0
    assert manifest["checks"]["v2_model_training_rerun"] is False


def test_audit_artifacts_match_manifest_hashes_and_schemas():
    manifest = json.loads((OUT / "AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    metric = pd.read_csv(OUT / "METRIC_RECOMPUTATION.csv")
    calibration = pd.read_csv(OUT / "CALIBRATION_RECOMPUTATION.csv")
    assert {"saved_value", "recomputed_value", "absolute_difference", "status"} <= set(metric.columns)
    assert {"saved_value", "recomputed_value", "absolute_difference", "status"} <= set(calibration.columns)
    assert set(metric.status) == {"pass"}
    assert set(calibration.status) == {"pass"}


def test_audit_identifies_calibration_selection_limitation():
    text = (OUT / "CALIBRATION_IMPLEMENTATION_AUDIT.md").read_text(encoding="utf-8")
    assert "Requires correction" in text
    assert "selected configuration was chosen using the held-out subset's labels" in text


def test_audit_preserves_historical_v2_inputs():
    manifest = json.loads((OUT / "AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["inputs"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
