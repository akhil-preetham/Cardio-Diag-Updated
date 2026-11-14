import pandas as pd

from runner.phase4_improvement.run import (
    STAGES,
    independent_audit,
    recompute_metrics,
    reconstruct,
)


def test_stages_are_temporally_ordered():
    assert set(STAGES["stage1_basic_risk"]).issubset(STAGES["stage3_resting_measurements"])
    assert "oldpeak" in STAGES["stage4_exercise_test"]
    assert "ca" not in STAGES["stage4_exercise_test"]


def test_recompute_metrics_returns_calibration_fields():
    group = pd.DataFrame({"y_true": [0, 0, 1, 1], "probability": [0.1, 0.2, 0.8, 0.9]})
    metrics = recompute_metrics(group)
    assert metrics["roc_auc"] == 1.0
    assert "calibration_intercept" in metrics
    assert "calibration_slope" in metrics


def test_reconstruction_is_four_centre_and_binary():
    data = reconstruct()
    assert len(data) == 920
    assert data.target_binary.isin([0, 1]).all()


def test_saved_phase4_predictions_can_be_audited():
    audit = independent_audit(reconstruct())
    assert not audit.empty
    assert audit["pass"].all()
