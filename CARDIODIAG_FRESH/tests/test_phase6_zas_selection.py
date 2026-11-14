from __future__ import annotations

import numpy as np
import pandas as pd

from runner.phase6_zas_selection.run import (
    FAMILY_ORDER,
    GROUPS,
    MODEL_GRIDS,
    choose_threshold,
    load_splits,
    make_pipeline,
)


def test_candidate_universe_has_no_logistic_ridge_alias_or_diagnostic_fields():
    assert len(GROUPS) == 6
    assert FAMILY_ORDER == ["logistic_l2", "elastic_net", "spline_logistic", "random_forest", "hist_gradient_boosting"]
    assert "ridge" not in FAMILY_ORDER
    all_features = set().union(*(set(features) for features in GROUPS.values()))
    assert not all_features.intersection({"Cath", "EF-TTE", "Region RWMA", "VHD"})
    assert len(MODEL_GRIDS) == 5


def test_frozen_outer_assignments_and_inner_partitions_are_disjoint():
    frame = pd.DataFrame({"Cath": ["Cad"] * 216 + ["Normal"] * 87})
    y = frame.Cath.map({"Cad": 1, "Normal": 0}).to_numpy()
    outer, inner = load_splits(
        __import__("pathlib").Path("results/methodology_corrections_v2/reproducibility/split_manifest_v2.json"),
        y,
    )
    assert len(outer) == 15
    assert len(inner) == 15
    for split in outer:
        folds = inner[(split["repeat"], split["fold"])]
        valid = np.concatenate([test for _, test in folds])
        assert len(valid) == len(set(valid.tolist()))
        assert set(valid.tolist()) == set(range(len(split["train"])))


def test_threshold_uses_inner_scores_and_resolves_ties_deterministically():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    assert choose_threshold(y, probability) == 0.6


def test_pipeline_keeps_imputation_and_encoding_in_transformer():
    frame = pd.DataFrame({"Age": [50, np.nan, 70], "Sex": ["Male", "Female", "Male"]})
    model = make_pipeline(frame, ["Age", "Sex"], "logistic")
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps
    assert model.named_steps["preprocess"].transformers[0][0] == "numeric"
