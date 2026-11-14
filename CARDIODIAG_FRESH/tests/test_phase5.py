import hashlib
import json

import numpy as np
import pandas as pd

from runner.phase5.common.metrics import metrics
from runner.phase5.common.models import estimator
from runner.phase5.zas_track.run import GROUPS


def test_metric_recomputation_is_complete():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["roc_auc"] == 1.0
    assert "calibration_slope" in result


def test_fold_local_estimator_has_pipeline_and_transformer():
    frame = pd.DataFrame({"numeric": [1.0, 2.0], "category": ["a", "b"]})
    model, grid = estimator("logistic", ["numeric", "category"], frame)
    assert "preprocess" in model.named_steps
    assert grid["model__C"]


def test_zas_v3_features_are_preserved():
    assert GROUPS["v3_five_predictor"] == ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]


def test_phase5_manifest_hashes_if_generated():
    path = "results/phase5/manifests/PHASE5_MANIFEST.json"
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except FileNotFoundError:
        return
    for relative, expected in manifest["outputs"].items():
        with open(relative, "rb") as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == expected
