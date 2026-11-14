import pandas as pd
import pytest

from cardiodiag.data.quality import duplicate_summary, identifier_columns, quality_summary


def test_quality_summary_tracks_conflicting_predictors():
    frame = pd.DataFrame({"patientid": [1, 2, 3], "age": [50, 50, 60], "target": [0, 1, 1]})
    result = quality_summary(frame, target="target", dataset="test")
    assert result["rows"] == 3
    assert result["duplicates"]["predictor_duplicate_groups"] == 1
    assert result["duplicates"]["conflicting_predictor_groups"] == 1
    assert identifier_columns(frame) == ["patientid"]


def test_duplicate_summary_keeps_missing_values_comparable():
    frame = pd.DataFrame({"x": [None, None], "target": [0, 0]})
    result = duplicate_summary(frame, "target")
    assert result["exact_duplicate_occurrences"] == 1


def test_identifier_columns_do_not_depend_on_values():
    frame = pd.DataFrame({"subject_id": [10, 11], "value": [1.0, 2.0]})
    assert identifier_columns(frame) == ["subject_id"]


def test_quality_rejects_no_data_conceptually_via_empty_summary():
    frame = pd.DataFrame({"target": pd.Series(dtype=int)})
    result = quality_summary(frame, target="target", dataset="empty")
    assert result["rows"] == 0
