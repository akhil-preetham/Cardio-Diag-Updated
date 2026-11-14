"""Recompute follow-up metrics and check calibration/adaptation separation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from portable_release import PROJECT, discover_workspace, verify_manifest
sys.path.insert(0, str(discover_workspace() / PROJECT))
from runner.research_extension_v2.run import metric


def verify(root):
    result = root / PROJECT / 'results/research_extension_v2'
    verify_manifest(result / 'MANIFEST.json', root)
    source = pd.read_csv(root / PROJECT / 'data/interim/validated/z_alizadeh_sani.csv')
    source_y = source.Cath.map({'Cad': 1, 'Normal': 0})
    target = pd.read_csv(root / PROJECT / 'results/research_extension_v1/transport/TARGET_PREDICTIONS.csv').set_index('case_id')
    count = 0
    for name, models in [('TYPING', 3), ('CALIBRATION', 5), ('MISSINGNESS', 2), ('ADAPTATION', 6)]:
        predictions = pd.read_csv(result / (name + '_PREDICTIONS.csv'))
        saved = pd.read_csv(result / (name + '_REPEAT_METRICS.csv'))
        keys = ['repeat', 'model_id'] + (['scenario'] if name == 'MISSINGNESS' else [])
        expected_groups = 3 * models * (7 if name == 'MISSINGNESS' else 1)
        assert predictions.model_id.nunique() == models
        assert len(saved) == expected_groups
        for key, group in predictions.groupby(keys):
            truth = target.y_base_gt50 if name == 'ADAPTATION' else source_y
            assert len(group) == len(truth) and group.row_id.nunique() == len(truth)
            assert set(group.row_id) == set(truth.index)
            assert np.array_equal(group.y_true, truth.loc[group.row_id])
            selected = saved
            for field, value in zip(keys, key):
                selected = selected[selected[field] == value]
            assert len(selected) == 1
            for field, value in metric(group.y_true, group.probability).items():
                assert np.isclose(value, selected.iloc[0][field], atol=1e-10, rtol=0), (name, key, field)
            count += 1
    for item in json.loads((result / 'CALIBRATION_MAPS.json').read_text()):
        d, c, t = (set(item[k]) for k in ['development_row_ids', 'calibration_row_ids', 'outer_test_row_ids'])
        assert not (d & c or d & t or c & t)
        assert d | c | t == set(range(303))
    for item in json.loads((result / 'TARGET_ADAPTATION_SPLITS.json').read_text()):
        a, e = set(item['adaptation_case_ids']), set(item['evaluation_case_ids'])
        assert not a & e and a | e == set(target.index)
    for item in json.loads((result / 'BOOTSTRAP_REFIT_MEMBERSHIP.json').read_text()):
        train, test = set(item['sampled_source_rows']), set(item['out_of_bag_rows'])
        assert not train & test and train | test == set(range(303))
    assert not pd.read_csv(result / 'DECISION_REGISTER.csv').default_changed.any()
    # The ordinary clean arm must exactly reproduce the established source arm.
    new = pd.read_csv(result / 'MISSINGNESS_PREDICTIONS.csv')
    new = new[(new.model_id == 'ordinary_core') & (new.scenario == 'mcar_0')]
    old = pd.read_csv(root / PROJECT / 'results/research_extension_v1/transport/SOURCE_OOF_PREDICTIONS.csv')
    old = old[old.arm == 'raw_full_source_fit']
    paired = new.merge(old, left_on=['repeat', 'fold', 'row_id'], right_on=['repeat', 'fold', 'source_row_index'])
    assert len(paired) == 909
    assert np.allclose(paired.probability_x, paired.probability_y, atol=1e-12, rtol=0)
    return {'verified_repeat_groups': count, 'default_changed': False}


if __name__ == '__main__':
    print(json.dumps(verify(discover_workspace()), indent=2))
