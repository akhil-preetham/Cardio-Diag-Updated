"""Independent provider-workbook and endpoint checks; no predictive model fit."""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from runner.research_extension_v1 import transport


@pytest.fixture(scope="module")
def workbooks():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        features = pd.read_excel(transport.SCREEN / "AngioCAD_Features.xlsx")
        labels = pd.read_excel(transport.SCREEN / "AngioCAD_Labels.xlsx")
    for frame in [features, labels]:
        frame.columns = frame.columns.astype(str).str.strip()
    return features.dropna(subset=["ID"]), labels.dropna(subset=["ID"])


def test_provider_zas_workbook_matches_validated_analysis_rows():
    workbook = next((transport.ROOT / "data/raw/z_alizadeh_sani/extracted").glob("*.xlsx"))
    original = pd.read_excel(workbook, sheet_name=0)
    validated = pd.read_csv(transport.SOURCE)
    assert original.shape == (303, 56)
    assert_frame_equal(original, validated, check_exact=True)


def test_xml_import_and_core_mapping_agree_with_independent_excel_reader(workbooks):
    features, _ = workbooks
    imported = transport.load_workbook(transport.SCREEN / "AngioCAD_Features.xlsx")
    assert imported.ID.astype(int).tolist() == features.ID.astype(int).tolist()
    expected = features[transport.CORE].copy()
    expected["Sex"] = expected.Sex.replace({"Male": 1, "Female": 0}).astype(float)
    expected = expected.astype(float).reset_index(drop=True)
    actual = transport.normalize_core(imported, source=False).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.isna().sum().to_dict() == {"Age": 6, "Sex": 0, "DM": 0, "HTN": 0, "Current Smoker": 46}


def test_endpoint_aggregation_agrees_with_independent_severity_lower_bound_rule(workbooks):
    features, labels = workbooks
    joined = labels.set_index("ID").loc[features.ID]
    # Independent scalar interpretation: a category whose lower bound exceeds
    # 50 is positive; exact-50 cannot be disambiguated within 26-50.
    def lower_bound(value):
        text = str(value)
        return 0 if text == "NL" else int(text.split("-")[0])
    expected = joined[transport.VESSELS].map(lower_bound).max(axis=1).gt(50).to_numpy(dtype=int)
    expected_boundary = (expected == 0) & joined[transport.VESSELS].astype(str).eq("26-50").any(axis=1).to_numpy()
    actual, boundary = transport.endpoints(joined)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(boundary, expected_boundary)
    assert actual.sum() == 298
    assert (~actual.astype(bool)).sum() == 79
    assert boundary.sum() == 10
    assert not joined[transport.VESSELS].isna().any().any()
    assert features.set_index("ID")[transport.VESSELS].astype(str).equals(joined[transport.VESSELS].astype(str))


def test_unknown_or_missing_artery_labels_are_not_silently_negative():
    row = pd.DataFrame([{name: "NL" for name in transport.VESSELS}])
    row.loc[0, "LM"] = np.nan
    with pytest.raises(ValueError, match="Unknown/missing"):
        transport.endpoints(row)


def test_diagnostic_columns_are_not_mapped_into_predictor_core(workbooks):
    features, _ = workbooks
    original = transport.normalize_core(features, source=False)
    changed = features.copy()
    changed["Diagnosis"] = "diagnostic-field-must-be-ignored"
    changed["Gensini Score"] = -999999
    changed["LM"] = "100"
    changed["ID"] = np.arange(len(changed)) + 1000000
    assert_frame_equal(transport.normalize_core(changed, source=False), original)
    assert set(original.columns) == {"Age", "Sex", "DM", "HTN", "Current Smoker"}


@pytest.mark.skipif(not (transport.OUT / "TRANSPORT_MANIFEST.json").exists(), reason="Transport not completed")
def test_transport_target_metrics_reconcile_with_independent_raw_workbooks(workbooks):
    _features, labels = workbooks
    saved = pd.read_csv(transport.OUT / "TARGET_PREDICTIONS.csv")
    raw_labels = labels.set_index("ID").loc[saved.target_case_id]
    severe = raw_labels[transport.VESSELS].astype(str).isin(["51-75", "76-90", "91-99", "100"]).any(axis=1).to_numpy(dtype=int)
    boundary = (severe == 0) & raw_labels[transport.VESSELS].astype(str).eq("26-50").any(axis=1).to_numpy()
    np.testing.assert_array_equal(saved.y_base_gt50, severe)
    np.testing.assert_array_equal(saved.boundary_ambiguous26_50, boundary)
    variants = {"base_gt50": (np.ones(len(saved), dtype=bool), severe), "exclude_ambiguous26_50": (~boundary, severe), "upper_assignment26_50_positive": (np.ones(len(saved), dtype=bool), np.where(boundary, 1, severe))}
    summary = pd.read_csv(transport.OUT / "TARGET_METRICS.csv").set_index(["endpoint", "arm"])
    for name, (keep, y) in variants.items():
        for arm in transport.ARMS:
            p = saved.loc[keep, arm].to_numpy()
            expected = {"roc_auc": roc_auc_score(y[keep], p), "average_precision": average_precision_score(y[keep], p), "brier_score": brier_score_loss(y[keep], p), "log_loss": log_loss(y[keep], p)}
            for metric, value in expected.items():
                assert summary.loc[(name, arm), metric] == pytest.approx(value, abs=1e-12)


@pytest.mark.skipif(not (transport.OUT / "TRANSPORT_MANIFEST.json").exists(), reason="Transport not completed")
def test_trusted_source_models_reproduce_saved_target_probabilities(workbooks):
    import joblib

    manifest = json.loads((transport.OUT / "TRANSPORT_MANIFEST.json").read_text())
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((transport.OUT / name).read_bytes()).hexdigest() == digest
    frozen = json.loads((transport.OUT / "FROZEN_PROTOCOL.json").read_text())
    from portable_release import rebase_path

    executed = transport.Path(transport.__file__).resolve()
    manifest_path = transport.OUT / "FROZEN_PROTOCOL.json"
    source_hashes = {rebase_path(key, transport.ROOT.parent, manifest_path): digest
                     for key, digest in frozen["input_hashes"].items()}
    assert source_hashes[executed] == hashlib.sha256(executed.read_bytes()).hexdigest()
    freeze = json.loads((transport.OUT / "PREDICTION_FREEZE.json").read_text())
    bundle = transport.OUT / "SOURCE_MODELS.joblib"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == freeze["models_sha256"]
    models = joblib.load(bundle)  # fixed local artifact, verified before loading
    features, _ = workbooks
    predictions = pd.read_csv(transport.OUT / "TARGET_PREDICTIONS.csv")
    indexed = features.set_index("ID").loc[predictions.target_case_id]
    x = transport.normalize_core(indexed, source=False)
    for arm in transport.ARMS:
        np.testing.assert_allclose(models[arm].predict_proba(x)[:, 1], predictions[arm], atol=1e-14)


@pytest.mark.skipif(not (transport.OUT / "TRANSPORT_MANIFEST.json").exists(), reason="Transport not completed")
def test_mcar_masks_are_nested_and_stress_metrics_reconcile():
    masks = pd.read_csv(transport.OUT / "SOURCE_MCAR_MASKS.csv")
    mask_cols = [f"mask_{name}" for name in transport.CORE]
    assert set(masks.mask_rate) == {0.0, .1, .25, .5}
    for _, group in masks.groupby(["repeat", "fold", "source_row_index"]):
        ordered = group.sort_values("mask_rate")[mask_cols].to_numpy(dtype=int)
        assert (np.diff(ordered, axis=0) >= 0).all()
        assert not ordered[0].any()
    predictions = pd.read_csv(transport.OUT / "SOURCE_MCAR_STRESS_PREDICTIONS.csv")
    summary = pd.read_csv(transport.OUT / "SOURCE_MCAR_STRESS_REPEAT_METRICS.csv").set_index(["repeat", "mask_rate"])
    for key, group in predictions.groupby(["repeat", "mask_rate"]):
        assert group.source_row_index.nunique() == len(group) == 303
        assert summary.loc[key, "roc_auc"] == pytest.approx(roc_auc_score(group.y_true, group.probability), abs=1e-12)
        assert summary.loc[key, "brier_score"] == pytest.approx(brier_score_loss(group.y_true, group.probability), abs=1e-12)
