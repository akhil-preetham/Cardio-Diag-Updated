import json
from pathlib import Path

import pandas as pd

from runner.primary_research_v4_performance.run import FEATURES, performance_specs


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "primary_research_v4_performance"
AUDIT = ROOT / "results" / "primary_research_v4_performance_independent_audit"


def test_performance_grid_is_bounded_and_preserves_existing_families():
    specs = performance_specs()
    assert len(specs) == 9
    assert set(FEATURES) == {"Age", "Sex", "Current Smoker", "EX-Smoker", "FH"}
    assert all(len(spec["grid"]) <= 3 for spec in specs.values())


def test_performance_predictions_are_complete_and_probabilities_valid():
    predictions = pd.read_csv(STUDY / "OUTER_FOLD_PREDICTIONS.csv")
    assert len(predictions) == 303 * 5 * 9
    assert predictions.duplicated(["repeat", "fold", "row_index", "model_id"]).sum() == 0
    assert predictions["calibrated_probability"].between(0, 1).all()


def test_independent_audit_passed_without_calibration_selection_reuse():
    checks = json.loads((AUDIT / "AUDIT_CHECKS.json").read_text(encoding="utf-8"))
    assert checks["calibration_selection_violations"] == 0
    assert checks["metric_recomputation_failures"] == 0
    assert checks["manifest_hash_failures"] == []
