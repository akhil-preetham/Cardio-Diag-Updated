import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_research_v2"


def test_v2_manifest_and_saved_split_hash_are_consistent():
    manifest = json.loads((OUT / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["failed_model_folds"] == 0
    assert manifest["prediction_rows"] == 303 * 5 * 9
    source = ROOT / "results" / "methodology_corrections_v2" / "reproducibility" / "split_manifest_v2.json"
    copied = OUT / "SPLIT_MANIFEST_USED.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["split_manifest_used"]["sha256"]
    assert source.read_bytes() == copied.read_bytes()


def test_v2_predictions_are_complete_and_svm_scores_are_not_probabilities():
    predictions = pd.read_csv(OUT / "OUTER_FOLD_PREDICTIONS.csv")
    assert predictions.groupby(["model_id", "repeat"]).row_index.nunique().eq(303).all()
    assert predictions.groupby(["model_id", "repeat", "fold", "row_index"]).size().max() == 1
    svm = predictions[predictions.model_id.str.startswith("svm_")]
    assert set(svm.native_output_type) == {"decision_score"}
    assert svm.calibrated_probability.between(0, 1).all()


def test_v2_metrics_exclude_invalid_svm_probability_metrics():
    metrics = pd.read_csv(OUT / "MODEL_COMPARISON_RESULTS.csv")
    svm_native = metrics[
        metrics.model_id.str.startswith("svm_")
        & metrics.probability_type.eq("decision_score")
        & metrics.level.eq("outer_fold")
    ]
    assert svm_native.brier_score.isna().all()
    assert svm_native.log_loss.isna().all()
    calibrated = metrics[
        metrics.probability_type.eq("nested_sigmoid") & metrics.level.eq("outer_fold")
    ]
    assert calibrated.brier_score.notna().all()
    assert calibrated.log_loss.notna().all()


def test_v2_split_assignments_have_no_overlap_or_outer_test_contamination():
    outer = pd.read_csv(OUT / "SPLIT_ASSIGNMENTS.csv")
    inner = pd.read_csv(OUT / "INNER_FOLD_ASSIGNMENTS.csv")
    for (repeat, fold), group in outer.groupby(["repeat", "fold"]):
        train = set(group.loc[group.role == "train", "row_index"])
        test = set(group.loc[group.role == "test", "row_index"])
        assert train.isdisjoint(test)
        assert len(train | test) == 303
    for (repeat, outer_fold, fold), group in inner.groupby(["repeat", "outer_fold", "fold"]):
        train = set(group.loc[group.role == "train", "row_index"])
        test = set(group.loc[group.role == "test", "row_index"])
        outer_train = set(
            outer.loc[
                (outer.repeat == repeat) & (outer.fold == outer_fold) & outer.role.eq("train"),
                "row_index",
            ]
        )
        assert train.isdisjoint(test)
        assert train | test == outer_train


def test_v2_configuration_records_calibration_isolation():
    config = json.loads((OUT / "EXPERIMENT_CONFIGURATION.json").read_text(encoding="utf-8"))
    assert config["calibration"]["method"] == "sigmoid"
    assert config["calibration"]["outer_test_labels_used_for_fit"] is False
    assert config["patient_identifier_available"] is False
    hyperparameters = pd.read_csv(OUT / "HYPERPARAMETER_SELECTION_RESULTS.csv")
    assert set(hyperparameters.status) == {"success"}
    assert set(hyperparameters.calibration_method) == {"nested_cross_fitted_sigmoid"}


def test_v2_historical_artifacts_remain_hash_stable():
    verification = json.loads(
        (ROOT / "results" / "primary_experiment_verification_v1" / "verification_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for relative, expected in verification["inputs"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
