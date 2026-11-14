import hashlib
import json
from pathlib import Path

import pandas as pd

from runner.primary_research_v1 import STRICT_FEATURES, build_feature_audit, metric_row, validate_data


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "primary_research_v1"


def test_strict_feature_audit_excludes_target_and_downstream_fields():
    audit = build_feature_audit().set_index("original_feature_name")
    assert STRICT_FEATURES == ["Age", "Sex", "Current Smoker", "EX-Smoker", "FH"]
    assert "Cath" not in STRICT_FEATURES
    assert audit.loc["Cath", "final_primary_eligibility"] == "excluded_primary"
    assert audit.loc["EF-TTE", "final_primary_eligibility"] == "excluded_primary"


def test_fold_assignments_have_no_train_test_overlap():
    splits = pd.read_csv(OUT / "splits" / "SPLIT_ASSIGNMENTS.csv")
    for (repeat, fold), group in splits.groupby(["repeat", "fold"]):
        train = set(group.loc[group["role"] == "train", "row_index"])
        test = set(group.loc[group["role"] == "test", "row_index"])
        assert train.isdisjoint(test), (repeat, fold)
    for repeat, group in splits.groupby("repeat"):
        assert group.loc[group["role"] == "test", "row_index"].nunique() == 303


def test_prediction_completeness_and_outer_test_isolation():
    predictions = pd.read_csv(OUT / "predictions" / "OUTER_FOLD_PREDICTIONS.csv")
    assert predictions["row_index"].notna().all()
    assert predictions[["uncalibrated_probability", "calibrated_probability"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    counts = predictions.groupby(["repeat", "model_id"])["row_index"].nunique()
    assert (counts == 303).all()
    assert predictions.groupby(["repeat", "fold", "model_id", "row_index"]).size().max() == 1


def test_metric_calculation_has_primary_and_secondary_values():
    row = metric_row(
        y_true=pd.Series([0, 0, 1, 1]).to_numpy(),
        probability=pd.Series([0.1, 0.4, 0.6, 0.9]).to_numpy(),
        model_id="test",
        probability_type="uncalibrated",
        repeat=1,
        fold=1,
    )
    assert row["roc_auc"] == 1.0
    assert row["pr_auc"] == 1.0
    assert row["threshold"] == 0.5


def test_manifest_hashes_and_execution_policy_are_consistent():
    manifest = json.loads((OUT / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["execution_policy"]["raw_data_modified"] is False
    assert manifest["execution_policy"]["external_validation"] is False
    assert manifest["failed_model_folds"] == 0
    for relative, expected in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_data_validation_preserves_target_and_duplicate_checks():
    frame = pd.read_csv(ROOT / "data" / "interim" / "validated" / "z_alizadeh_sani.csv")
    validation = validate_data(frame, STRICT_FEATURES)
    assert validation["rows"] == 303
    assert validation["class_counts"] == {"Cad": 216, "Normal": 87}
    assert validation["exact_duplicate_rows"] == 0
    assert validation["strict_predictor_duplicate_rows"] == 147
