from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from runner.research_extension_v1 import zas


def test_nested_memberships_exclude_calibration_and_outer_test():
    frame = pd.read_csv(zas.legacy.DATA_PATH)
    y = frame.Cath.map({"Cad": 1, "Normal": 0}).to_numpy(dtype=int)
    outer, inner = zas.legacy.load_splits(zas.legacy.SPLIT_PATH, y)
    for split in outer:
        training, test = set(split["train"]), set(split["test"])
        _, components = zas.split_trace(split, inner[split["repeat"], split["fold"]], y[split["train"]])
        seen = []
        for comp in components:
            dev, cal = set(comp["development_rows"]), set(comp["calibration_rows"])
            assert not dev & cal and dev | cal == training
            assert not (dev | cal) & test
            seen.extend(cal)
            for cv in comp["inner_cv"]:
                a, b = set(cv["fit_rows"]), set(cv["valid_rows"])
                assert not a & b and a | b == dev
                assert not (a | b) & (cal | test)
        assert set(seen) == training and len(seen) == len(training)


def test_training_api_receives_no_outer_targets_and_feature_sets_are_fixed():
    for function in (zas.train_corrected_component, zas.train_historical_control):
        assert list(inspect.signature(function).parameters)[:3] == ["x_training", "y_training", "x_outer_unlabeled"]
    assert [len(cols) for cols in zas.feature_sets().values()] == [38, 28, 17, 5]
    for cols in zas.feature_sets().values():
        assert not set(cols) & {"Cath", "EF-TTE", "Region RWMA", "VHD"}


def test_immutable_artifacts_refuse_changes(tmp_path):
    path = tmp_path / "test.json"
    zas.write_immutable(path, {"frozen": 1})
    zas.write_immutable(path, {"frozen": 1})
    with pytest.raises(FileExistsError):
        zas.write_immutable(path, {"frozen": 2})


def test_calibration_targets_do_not_enter_base_selection(monkeypatch):
    seen = []

    class FixedBase:
        def predict_proba(self, frame):
            return np.tile([.3, .7], (len(frame), 1))

    def select(x_development, y_development, inner_cv):
        seen.append((x_development.Age.tolist(), y_development.tolist()))
        return {"features": ["Age"]}, FixedBase(), [], []

    class FixedCalibrator:
        intercept_ = np.array([0.])
        coef_ = np.array([[1.]])

        def predict_proba(self, frame):
            return np.tile([.25, .75], (len(frame), 1))

    monkeypatch.setattr(zas, "select_configuration", select)
    monkeypatch.setattr(zas, "sigmoid", lambda scores, labels: FixedCalibrator())
    x = pd.DataFrame({"Age": [10, 20, 30, 40]})
    outer_unlabeled = pd.DataFrame({"Age": [50, 60]})
    dev, cal = np.array([0, 1]), np.array([2, 3])
    a = zas.train_corrected_component(x, np.array([0, 1, 0, 1]), outer_unlabeled, dev, cal, [])
    b = zas.train_corrected_component(x, np.array([0, 1, 1, 0]), outer_unlabeled, dev, cal, [])
    assert seen == [([10, 20], [0, 1]), ([10, 20], [0, 1])]
    assert a["raw_probability"] == b["raw_probability"]
    assert a["calibration_y"] != b["calibration_y"]


def test_aggregate_auc_uses_complete_repeat_instead_of_fold_mean():
    # Individual folds are perfectly ranked; between-fold probability scales reverse comparisons.
    frame = pd.DataFrame({"repeat": [1] * 4, "fold": [1, 1, 2, 2], "model_id": ["example"] * 4,
                          "y_true": [0, 1, 0, 1], "probability": [.7, .8, .1, .2], "threshold": [.5] * 4})
    folds, repeats, _ = zas.summarize(frame)
    assert folds.roc_auc.mean() == 1
    assert repeats.roc_auc.iloc[0] == roc_auc_score(frame.y_true, frame.probability) == .75


@pytest.mark.skipif(not (zas.OUT / "MANIFEST.json").exists(), reason="Full experiment not completed yet")
def test_completed_predictions_hashes_selection_and_component_averages():
    manifest = json.loads((zas.OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["input_hashes"] == zas.input_hashes()
    assert manifest["corrected_component_searches"] == 45
    for relative, digest in manifest["outputs"].items():
        assert zas.sha256(zas.ROOT / relative) == digest
    pred = pd.read_csv(zas.OUT / "PREDICTIONS.csv")
    y = pd.read_csv(zas.legacy.DATA_PATH).Cath.map({"Cad": 1, "Normal": 0}).to_numpy(dtype=int)
    assert len(pred) == 303 * 3 * 8
    assert not pred.duplicated(["repeat", "model_id", "row_index"]).any()
    for _, group in pred.groupby(["repeat", "model_id"]):
        assert set(group.row_index) == set(range(303))
        assert np.array_equal(group.y_true, y[group.row_index])
    for path in sorted((zas.OUT / "checkpoints").glob("*corrected_component*.json")):
        cp = json.loads(path.read_text(encoding="utf-8"))
        assert len(cp["candidates"]) == 30 and len(cp["grid_scores"]) == 120
        winner = max(cp["candidates"], key=lambda row: row["best_score"])
        assert cp["selection"]["inner_auc"] == winner["best_score"]
    for (r, f), group in pred.groupby(["repeat", "fold"]):
        cps = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((zas.OUT / "checkpoints").glob(f"r{r}_f{f}_corrected_component*.json"))]
        for model, field in ((zas.MODEL_IDS[4], "calibrated_probability"), (zas.MODEL_IDS[5], "raw_probability")):
            actual = group[group.model_id == model].sort_values("row_index")
            assert np.allclose(actual.probability, np.mean([cp[field] for cp in cps], axis=0), atol=1e-14)
            assert actual.threshold.eq(.5).all()


@pytest.mark.skipif(not (zas.OUT / "MANIFEST.json").exists(), reason="Full experiment not completed yet")
def test_completed_metrics_recalculate_from_saved_predictions():
    pred = pd.read_csv(zas.OUT / "PREDICTIONS.csv")
    saved = pd.read_csv(zas.OUT / "REPEAT_METRICS.csv").set_index(["repeat", "model_id"])
    for key, group in pred.groupby(["repeat", "model_id"]):
        assert saved.loc[key, "n"] == 303
        assert saved.loc[key, "roc_auc"] == pytest.approx(roc_auc_score(group.y_true, group.probability), abs=1e-12)
        assert saved.loc[key, "brier"] == pytest.approx(brier_score_loss(group.y_true, group.probability), abs=1e-12)


@pytest.mark.skipif(not (zas.OUT / "MANIFEST.json").exists(), reason="Full experiment not completed yet")
def test_equal_feature_control_reproduces_historical_l2_winning_folds():
    old = pd.read_csv(zas.INPUTS["historical_predictions"])
    old = old[(old.model_id == "selection_aware_procedure") & (old.selected_model_family == "logistic_l2")]
    current = pd.read_csv(zas.OUT / "PREDICTIONS.csv")
    control = current[current.model_id == zas.MODEL_IDS[0]]
    joined = old.merge(control, on=["repeat", "fold", "row_index"])
    assert old.groupby(["repeat", "fold"]).ngroups == 9
    assert len(joined) == len(old)
    assert np.allclose(joined.calibrated_probability, joined.probability, atol=1e-9)
    assert np.allclose(joined.raw_probability_x, joined.raw_probability_y, atol=1e-9)
