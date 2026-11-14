from __future__ import annotations

import numpy as np
import pytest

from runner.phase6_1_audit.audit import (
    OUT,
    reconcile,
    reconstruct_metrics,
    verify_audit_manifest,
    verify_phase6_manifest,
)


@pytest.fixture(scope="module")
def reconciled():
    return reconcile()


def test_saved_predictions_cover_each_outer_test_partition_and_source_labels(reconciled):
    coverage = reconciled["coverage"]
    assert len(coverage) == 15 * 2
    assert coverage.expected_test_rows.eq(coverage.observed_prediction_rows).all()
    assert coverage.expected_test_rows.eq(coverage.unique_source_rows).all()
    assert coverage.target_labels_match_source.all()
    assert coverage.source_row_label_consistent.all()
    assert coverage.manifest_partition_match.all()


def test_metric_reconstruction_uses_complete_repeat_predictions(reconciled):
    repeats = reconciled["repeats"]
    assert len(repeats) == 3 * 2
    assert repeats.n.eq(303).all()
    assert repeats.events.eq(216).all()
    assert repeats.benchmark_metrics_match.all()
    # The AUC over all 303 OOF rows is explicitly distinguished from the
    # unweighted arithmetic mean of the five fold AUCs.
    assert repeats.repeat_auc_not_fold_mean.all()


def test_selected_candidate_and_calibration_metadata(reconciled):
    winners = reconciled["candidate_selections"]
    assert len(winners) == 15
    assert winners.candidate_count.eq(30).all()
    assert winners.selection_metadata_matches_saved_inner_ranking.all()
    corrected = reconciled["metadata"]
    assert len(corrected) == 15 * 2 * 3
    assert set(corrected.calibration_fold) == {1, 2, 3}
    assert corrected.groupby(["repeat", "outer_fold", "model_id"]).calibration_fold.nunique().eq(3).all()


def test_metric_helper_and_phase6_manifest_hashes():
    y = np.array([0, 0, 1, 1], dtype=int)
    probability = np.array([0.1, 0.3, 0.7, 0.9])
    result = reconstruct_metrics(y, probability, 0.5)
    assert result["n"] == 4
    assert result["events"] == 2
    assert result["roc_auc"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    manifest = verify_phase6_manifest()
    assert manifest["all_match"]
    assert len(manifest["outputs"]) >= 15
    audit_manifest = verify_audit_manifest()
    assert audit_manifest["all_match"]
    assert OUT.name == "phase6_1_audit"
