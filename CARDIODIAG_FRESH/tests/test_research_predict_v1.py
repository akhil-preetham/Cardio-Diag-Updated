from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runner.research_extension_v1.predict_research import validate_input


def valid_frame():
    return pd.DataFrame({"Age": [60.0], "Sex": ["Male"], "DM": [0], "HTN": [1], "Current Smoker": [0]})


def test_scoring_rejects_outcome_columns_and_incomplete_schema():
    with pytest.raises(ValueError, match="exactly five"):
        validate_input(valid_frame().assign(Cath="Cad"))
    with pytest.raises(ValueError, match="exactly five"):
        validate_input(valid_frame().drop(columns="HTN"))


def test_scoring_rejects_unsupported_categories_and_invalid_units():
    with pytest.raises(ValueError, match="Unsupported sex"):
        validate_input(valid_frame().assign(Sex="Unknown"))
    with pytest.raises(ValueError, match="Age outside"):
        validate_input(valid_frame().assign(Age=600))
    with pytest.raises(ValueError, match="Unexpected DM"):
        validate_input(valid_frame().assign(DM=2))


def test_missing_values_preserved_for_source_trained_imputer():
    result = validate_input(valid_frame().assign(Age=np.nan, **{"Current Smoker": np.nan}))
    assert result.Age.isna().all()
    assert result["Current Smoker"].isna().all()
    assert result.Sex.iloc[0] == 1.0


def test_empty_or_entirely_missing_record_is_rejected():
    with pytest.raises(ValueError, match="no records"):
        validate_input(valid_frame().iloc[:0])
    blank = pd.DataFrame({c: [np.nan] for c in valid_frame().columns})
    with pytest.raises(ValueError, match="every predictor missing"):
        validate_input(blank)
