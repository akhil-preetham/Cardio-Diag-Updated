import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "primary_research_v3"
AUDIT = ROOT / "results" / "primary_research_v3_independent_audit"


def test_v3_manifest_and_fit_budget():
    manifest = json.loads((V3 / "REPRODUCIBILITY_MANIFEST.json").read_text())
    config = json.loads((V3 / "EXPERIMENT_CONFIGURATION.json").read_text())
    assert manifest["failed_model_folds"] == 0
    assert manifest["prediction_rows"] == 303 * 5 * 9
    assert manifest["actual_pipeline_fits"] == 16325
    assert config["actual_pipeline_fits"] == 16325


def test_v3_calibration_assignments_are_unique_and_selection_records_complete():
    assignments = pd.read_csv(V3 / "CALIBRATION_FOLD_ASSIGNMENTS.csv")
    selection = pd.read_csv(V3 / "CALIBRATION_SELECTION_RESULTS.csv")
    assert len(assignments) == 5 * 5 * 4
    assert assignments.duplicated(["repeat", "outer_fold", "calibration_fold"]).sum() == 0
    assert len(selection) == 5 * 5 * 9 * 4
    assert selection.status.eq("success").all()
    assert selection.selection_cv_train_indices.notna().all()
    assert selection.selection_cv_validation_indices.notna().all()


def test_v3_predictions_and_probability_types_are_complete():
    predictions = pd.read_csv(V3 / "OUTER_FOLD_PREDICTIONS.csv")
    assert predictions.groupby(["model_id", "repeat"]).row_index.nunique().eq(303).all()
    assert predictions.groupby(["model_id", "repeat", "fold", "row_index"]).size().max() == 1
    svm = predictions[predictions.model_id.str.startswith("svm_")]
    assert set(svm.native_output_type) == {"decision_score"}
    assert svm.calibrated_probability.between(0, 1).all()


def test_v3_audit_passes_independent_checks():
    checks = json.loads((AUDIT / "AUDIT_CHECKS.json").read_text())
    assert checks["outer_manifest_equal"] is True
    assert checks["inner_manifest_equal"] is True
    assert checks["calibration_selection_checks_pass"] is True
    assert checks["metric_failures"] == 0
    assert checks["calibration_failures"] == 0


def test_v3_output_hashes_and_historical_inputs_are_stable():
    manifest = json.loads((V3 / "REPRODUCIBILITY_MANIFEST.json").read_text())
    for relative, expected in manifest["outputs"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    audit = json.loads((AUDIT / "AUDIT_MANIFEST.json").read_text())
    for relative, expected in audit["inputs"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
