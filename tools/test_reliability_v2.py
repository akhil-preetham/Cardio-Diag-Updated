"""Leakage boundaries, fixed schemas and probability transformations."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_validate, StratifiedKFold
from runner.research_extension_v2 import run as study


def test_schema_partition_and_invalid_codes():
    groups = study.CONTINUOUS + study.ORDINAL + study.BINARY + study.NOMINAL
    assert len(groups) == len(set(groups)) == 38
    assert set(groups) == set(study.FULL)
    frame = pd.read_csv(study.legacy.DATA_PATH)
    normalized = study.normalize_typed(frame)
    assert normalized.shape == (303, 38)
    frame.loc[0, 'DM'] = 2
    with pytest.raises(ValueError, match='binary'):
        study.normalize_typed(frame)


def test_offset_preserves_rank_and_matches_training_prevalence():
    p = np.linspace(.02, .95, 60)
    y = np.tile([0, 1, 1], 20)
    mapping = study.fit_map(p, y, 'offset')
    corrected = study.apply_map(p, mapping)
    assert corrected.mean() == pytest.approx(y.mean(), abs=1e-10)
    assert np.array_equal(np.argsort(p), np.argsort(corrected))
    for mode in ['logit_c1', 'logit_cv']:
        mapping = study.fit_map(p, y, mode)
        predicted = study.apply_map([0, 1, .5], mapping)
        assert np.isfinite(predicted).all()
        assert ((predicted > 0) & (predicted < 1)).all()


def test_augmentation_fits_only_fold_training_rows():
    frame = pd.read_csv(study.legacy.DATA_PATH)
    x = study.tr.normalize_core(frame, source=True)
    y = frame.Cath.map({'Cad': 1, 'Normal': 0}).to_numpy()
    cv = list(StratifiedKFold(3, shuffle=True, random_state=31).split(x, y))
    result = cross_validate(clone(study.AugmentedCore()), x, y, cv=cv,
                            scoring='roc_auc', return_estimator=True)
    for estimator, (train, test) in zip(result['estimator'], cv):
        assert estimator.training_input_rows_ == len(train)
        assert not set(train) & set(test)
        assert estimator.predict_proba(x.iloc[test]).shape == (len(test), 2)
    assert np.isfinite(result['test_score']).all()


def test_json_outputs_cannot_overwrite(tmp_path):
    path = tmp_path / 'frozen.json'
    study.json_write(path, {'frozen': True})
    with pytest.raises(FileExistsError):
        study.json_write(path, {'frozen': False})
