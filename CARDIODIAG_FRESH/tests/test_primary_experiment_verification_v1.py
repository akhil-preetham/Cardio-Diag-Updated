import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_experiment_verification_v1"


def test_verification_manifest_records_zero_metric_mismatches():
    manifest = json.loads((OUT / "verification_manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification"]["experiment_results_modified"] is False
    assert manifest["metric_recomputation"]["rows_checked"] == 6300
    assert manifest["metric_recomputation"]["mismatches"] == 0
    assert manifest["script"].endswith("primary_experiment_verification_v1.py")


def test_verification_outputs_match_manifest_hashes():
    manifest = json.loads((OUT / "verification_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_recomputed_metric_audit_contains_no_mismatches():
    audit = pd.read_csv(OUT / "MODEL_METRIC_RECOMPUTATION.csv")
    assert len(audit) == 6300
    assert (audit["status"] == "verified").all()


def test_feature_audit_separates_strict_and_exploratory_recommendations():
    audit = pd.read_csv(OUT / "FEATURE_CLINICAL_VALIDITY_AUDIT.csv")
    strict = audit[audit["recommendation"] == "strict_primary"]["feature"].tolist()
    assert strict == ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]
    assert (audit["recommendation"] == "exploratory_only_pending_timing_or_unit_review").any()
