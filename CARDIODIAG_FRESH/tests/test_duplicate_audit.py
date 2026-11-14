import pandas as pd

from runner.duplicate_audit import (
    duplicate_group_summary,
    near_duplicate_candidates,
    partition_overlaps,
)


def marked(df):
    df.attrs["dataset_name"] = "test"
    return df


def test_exact_duplicate_rows_and_first_occurrence():
    df = marked(pd.DataFrame({"x": [1, 1, 2], "target": [0, 0, 1]}))
    summary, groups = duplicate_group_summary(df, "target", list(df.columns), "exact")
    assert summary["duplicated_records"] == 1
    assert groups[0]["first_occurrence"] == 0
    assert groups[0]["subsequent_occurrences"] == [1]


def test_predictor_duplicates_identical_and_conflicting_labels():
    df = marked(pd.DataFrame({"x": [1, 1, 2, 2], "target": [0, 0, 0, 1]}))
    summary, groups = duplicate_group_summary(df, "target", ["x"], "predictor")
    assert summary["duplicate_groups"] == 2
    assert summary["conflicting_target_groups"] == 1
    assert any(not group["target_consistent"] for group in groups)


def test_missing_values_are_equal_for_exact_duplicates():
    df = marked(pd.DataFrame({"x": [None, None], "target": [0, 0]}))
    summary, _ = duplicate_group_summary(df, "target", list(df.columns), "exact")
    assert summary["duplicate_groups"] == 1


def test_mixed_near_duplicate_screen_is_conservative():
    df = marked(pd.DataFrame({"age": [50.0, 50.0000000001], "sex": ["M", "M"], "target": [0, 0]}))
    candidates = near_duplicate_candidates(df, "target", ["age", "sex"])
    assert len(candidates) == 1
    assert candidates[0]["target_conflict"] is False


def test_patient_identifier_is_excluded_from_predictor_overlap():
    df = marked(pd.DataFrame({
        "patientid": [10, 11, 12, 13],
        "age": [50, 50, 60, 70],
        "target": [0, 0, 1, 1],
    }))
    summary, _ = duplicate_group_summary(df, "target", ["age"], "predictor_without_patientid")
    assert summary["duplicate_groups"] == 1
    rows = partition_overlaps(df, "target", ["age"], [42], test_size=0.5)
    assert rows[0]["predictor_overlap_groups"] >= 0
