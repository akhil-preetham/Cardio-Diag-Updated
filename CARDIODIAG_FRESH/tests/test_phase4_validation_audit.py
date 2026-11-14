import numpy as np
import pandas as pd

from runner.phase4_improvement.run import reconstruct
from runner.phase4_validation_audit.run import (
    STAGES,
    bootstrap_values,
    ece,
)


def test_calibration_ece_preserves_bin_counts():
    group = pd.DataFrame({"y_true": [0, 0, 1, 1], "probability": [0.1, 0.2, 0.8, 0.9]})
    value, bins = ece(group, bins=2)
    assert np.isclose(value, 0.15)
    assert sum(item["n"] for item in bins) == len(group)


def test_bootstrap_is_reproducible():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.7, 0.9, 0.4, 0.8])
    first = bootstrap_values(y, p, "roc_auc", np.random.default_rng(42), repetitions=50)
    second = bootstrap_values(y, p, "roc_auc", np.random.default_rng(42), repetitions=50)
    assert first == second


def test_stages_do_not_include_target_derived_fields():
    assert all(feature not in STAGES["stage4_exercise_test"] for feature in ["ca", "thal", "num"])


def test_reconstructed_keys_are_unique_within_centre():
    data = reconstruct()
    assert not data.duplicated(["centre", "source_row"]).any()
