import numpy as np

from runner.phase4_uci_multicentre.run import (
    STRICT_FEATURES,
    build_estimator,
    model_metrics,
    reconstruct,
)


def test_reconstructs_expected_four_centre_total():
    data = reconstruct()
    assert len(data) == 920
    assert data.groupby("centre").size().to_dict() == {"Cleveland": 303, "Hungary": 294, "Switzerland": 123, "VA Long Beach": 200}


def test_centre_separation_and_target_encoding():
    data = reconstruct()
    assert set(data.centre) == {"Cleveland", "Hungary", "Switzerland", "VA Long Beach"}
    assert set(data.target_binary.unique()) == {0, 1}
    assert (data.num > 0).equals(data.target_binary.astype(bool))


def test_fold_local_pipeline_has_imputation_and_scaling():
    estimator, grid = build_estimator("logistic", STRICT_FEATURES)
    assert "preprocess" in estimator.named_steps
    assert "model__C" in grid
    assert estimator.named_steps["preprocess"].transformers[0][1].steps[0][0] == "imputer"
    assert estimator.named_steps["preprocess"].transformers[0][1].steps[1][0] == "scale"


def test_metric_recomputation_is_complete():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = model_metrics(y, p)
    assert metrics["n"] == 4
    assert metrics["roc_auc"] == 1.0
    assert metrics["tp"] == 2 and metrics["tn"] == 2
