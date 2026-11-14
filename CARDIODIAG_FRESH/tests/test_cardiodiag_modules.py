import numpy as np
import pandas as pd
import pytest

from cardiodiag.calibration import calibration_summary
from cardiodiag.clinical_utility import net_benefit
from cardiodiag.evaluation import classification_metrics
from cardiodiag.feature_selection import XAIHDRCSelector
from cardiodiag.imbalance import loras_resample
from cardiodiag.preprocessing import build_preprocessor
from cardiodiag.statistics import delong_auc_test, paired_bootstrap_difference


def test_preprocessor_is_constructed_without_fitting_on_input():
    preprocessor = build_preprocessor(["age"], ["sex"])
    assert not hasattr(preprocessor, "transformers_")


def test_xaih_selector_is_deterministic_and_selects_requested_features():
    frame = pd.DataFrame({"strong": [0, 0, 1, 1], "weak": [0, 1, 0, 1]})
    selector = XAIHDRCSelector(n_features=1).fit(
        frame,
        [0, 0, 1, 1],
        lime_importances={"strong": 1.0, "weak": 0.1},
    )
    assert selector.selected_features_ == ["strong"]
    assert list(selector.transform(frame).columns) == ["strong"]


def test_metrics_and_calibration_have_expected_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9])
    metrics = classification_metrics(y_true, y_pred, probabilities)
    assert metrics["accuracy"] == 0.75
    assert metrics["specificity"] == 0.5
    assert calibration_summary(y_true, probabilities)["bins"] == 10
    assert net_benefit(y_true, probabilities, 0.5) == pytest.approx(0.25)


def test_paired_bootstrap_requires_aligned_observations():
    result = paired_bootstrap_difference(
        np.array([0, 1, 1]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        metric=lambda truth, prediction: np.mean(truth == prediction),
        n_bootstrap=50,
    )
    assert set(result) == {"difference", "ci_low", "ci_high"}
    with pytest.raises(ValueError):
        paired_bootstrap_difference(np.array([0]), np.array([0, 1]), np.array([0]), metric=np.mean)


def test_delong_requires_aligned_vectors_and_returns_probability():
    result = delong_auc_test(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.2, 0.3, 0.7, 0.8]),
    )
    assert result["auc_first"] == pytest.approx(1.0)
    assert 0.0 <= result["p_value"] <= 1.0


def test_loras_generates_minority_samples_when_dependency_is_available():
    pytest.importorskip("loras")
    features = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6]])
    labels = np.array([0] * 7 + [1] * 5)
    sampled_features, sampled_labels = loras_resample(features, labels, target_count=7)
    assert sampled_features.shape[0] == sampled_labels.shape[0] == 14
    assert np.sum(sampled_labels == 0) == np.sum(sampled_labels == 1) == 7
