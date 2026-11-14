"""Meaningful split, semantic, model-fitting and saved-artifact checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from runner.research_extension_v1 import transport as tr


def toy_frame(n=60):
    rng = np.random.default_rng(831)
    return pd.DataFrame({"Age": rng.uniform(30, 80, n), "Sex": rng.integers(0, 2, n).astype(float), "DM": rng.integers(0, 2, n).astype(float), "HTN": rng.integers(0, 2, n).astype(float), "Current Smoker": rng.integers(0, 2, n).astype(float)})


def test_fixed_mapping_and_missingness_are_not_outcome_guided():
    source = pd.DataFrame({"Age": [30, 55], "Sex": ["Male", "Fmale"], "DM": [0, 1], "HTN": [1, 0], "Current Smoker": [0, 1]})
    normalized = tr.normalize_core(source, source=True)
    assert normalized.Sex.tolist() == [1, 0]
    target = source.assign(Sex=["Female", "Male"], **{"Current Smoker": ["", "1"]})
    mapped = tr.normalize_core(target, source=False)
    assert mapped.Sex.tolist() == [0, 1]
    assert pd.isna(mapped.loc[0, "Current Smoker"])
    with pytest.raises(ValueError, match="code"):
        tr.normalize_core(target.assign(DM=[2, 0]), source=False)


def test_ambiguous_stenosis_is_not_silently_relabelled():
    rows = pd.DataFrame("NL", index=range(3), columns=tr.VESSELS)
    rows.loc[1, "LM"] = "26-50"
    rows.loc[2, "LM"] = "51-75"
    y, boundary = tr.endpoints(rows)
    assert y.tolist() == [0, 0, 1]
    assert boundary.tolist() == [False, True, False]
    rows.loc[0, "LM"] = ""
    with pytest.raises(ValueError, match="Unknown"):
        tr.endpoints(rows)


def test_search_is_nested_in_each_calibration_training_subset(monkeypatch):
    x = toy_frame()
    y = np.tile([0, 1], 30)
    observed_sizes = []
    original = GridSearchCV.fit
    def spy(self, x_fit, y_fit, **kwargs):
        observed_sizes.append(len(y_fit))
        return original(self, x_fit, y_fit, **kwargs)
    monkeypatch.setattr(GridSearchCV, "fit", spy)
    model = tr.make_calibrated(151)
    model.fit(x, y)
    assert observed_sizes == [40, 40, 40]
    assert len(model.calibrated_classifiers_) == 3
    assert all(pair.estimator.n_splits_ == 3 for pair in model.calibrated_classifiers_)


def test_target_missing_values_use_unchanged_source_imputer():
    x = toy_frame()
    model = tr.make_search(51).fit(x, np.tile([0, 1], 30))
    imputer = model.best_estimator_.named_steps["preprocess"].named_transformers_["age"].named_steps["impute"]
    frozen_median = imputer.statistics_.copy()
    target = toy_frame(4)
    target.loc[:, "Age"] = np.nan
    probabilities = model.predict_proba(target)[:, 1]
    np.testing.assert_equal(imputer.statistics_, frozen_median)
    assert np.isfinite(probabilities).all()


def test_joint_calibration_matches_independent_logistic_evaluation():
    rng = np.random.default_rng(4241)
    y = rng.binomial(1, .6, 400)
    p = np.clip(.45 + .13*y + rng.normal(0, .1, 400), .02, .98)
    intercept, slope = tr.joint_calibration(y, p)
    independent = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, tol=1e-10).fit(logit(p).reshape(-1, 1), y)
    np.testing.assert_allclose([intercept, slope], [independent.intercept_[0], independent.coef_[0, 0]], atol=1e-4)


def test_record_bootstrap_is_deterministic_and_finite():
    y = np.tile([0, 1], 25)
    p = np.linspace(.15, .9, 50)
    first, first_info = tr.bootstrap(y, p, draws=30, seed=17)
    second, second_info = tr.bootstrap(y, p, draws=30, seed=17)
    assert first == second
    assert first_info == second_info
    assert all(item["bootstrap_025"] <= item["bootstrap_975"] for item in first)


def test_saved_transport_manifest_predictions_endpoints_and_memberships():
    if not (tr.OUT / "TRANSPORT_MANIFEST.json").exists():
        pytest.skip("Run frozen transport study before artifact verification")
    manifest = json.loads((tr.OUT/"TRANSPORT_MANIFEST.json").read_text())
    assert manifest["target_label_fits"] == 0
    for name, digest in manifest["outputs"].items():
        assert tr.sha(tr.OUT/name) == digest, name
    predictions = pd.read_csv(tr.OUT/"TARGET_PREDICTIONS.csv")
    assert len(predictions) == predictions.case_id.nunique() == 377
    assert predictions.y_gt50.sum() == 298
    assert predictions.boundary.sum() == 10
    for name in ["prob_raw", "prob_calibrated"]:
        assert predictions[name].between(0, 1).all()
    table = pd.read_csv(tr.OUT/"TARGET_METRICS.csv")
    sizes = table.groupby("endpoint")["n"].unique().to_dict()
    assert sizes["base_gt50"].tolist() == [377]
    assert sizes["exclude_ambiguous26_50"].tolist() == [367]
    assert sizes["upper_assignment26_50_positive"].tolist() == [377]
    membership = pd.read_csv(tr.OUT/"SOURCE_CALIBRATION_MEMBERSHIP.csv")
    for _, g in membership.groupby(["scope", "repeat", "fold", "component"]):
        fit = set(g.loc[g.role == "component_fit_and_inner_selection", "source_row_index"])
        calibrate = set(g.loc[g.role == "sigmoid_fit_only", "source_row_index"])
        assert fit
        assert calibrate
        assert not fit & calibrate
    masks = pd.read_csv(tr.OUT/"SOURCE_MCAR_MASKS.csv")
    assert not masks.loc[masks.mask_rate == 0, [f"mask_{name}" for name in tr.CORE]].any().any()
    raw = pd.read_csv(tr.OUT/"SOURCE_OOF_PREDICTIONS.csv").query("arm == 'raw_full_source_fit'")
    stress = pd.read_csv(tr.OUT/"SOURCE_MCAR_STRESS_PREDICTIONS.csv").query("mask_rate == 0")
    joined = raw.merge(stress, on=["repeat", "fold", "source_row_index"], suffixes=("_raw", "_stress"))
    assert len(joined) == 303*3
    np.testing.assert_allclose(joined.probability_raw, joined.probability_stress, atol=1e-15)
