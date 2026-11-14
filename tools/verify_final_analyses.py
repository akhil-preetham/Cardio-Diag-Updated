"""Verify final-analysis hashes, metric reconstruction and split boundaries."""
import json
import numpy as np
import pandas as pd
from pathlib import PurePosixPath
from portable_release import discover_workspace, PROJECT, sha256
from finalize_analyses import net_benefit, v2


def verify():
    root = discover_workspace()
    out = root / PROJECT / 'results/final_analyses_v3'
    manifest=json.loads((out/'MANIFEST.json').read_text())
    for section in ['inputs','outputs']:
        for name,digest in manifest[section].items():
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            current=(root/name).resolve()
            assert current.is_relative_to(root) and sha256(current)==digest, name
    for item in json.loads((out/'THRESHOLD_SPLIT_TRACE.json').read_text()):
        a,b,c = [set(item[key]) for key in ['search_fit_rows','threshold_prediction_rows','outer_test_rows']]
        assert not (a&b or a&c or b&c) and a|b|c == set(range(303))
    for item in json.loads((out/'ADAPTATION_LEARNING_TRACE.json').read_text()):
        assert not set(item['adaptation_case_ids']) & set(item['evaluation_case_ids'])
    count = 0
    for name,extras,n,groups in [('SOURCE_LEARNING',['fraction'],303,12),
        ('PERTURBATION',['scenario'],303,15),('ADAPTATION_LEARNING',['fraction','model_id'],377,30)]:
        predictions = pd.read_csv(out/(name+'_PREDICTIONS.csv'))
        saved = pd.read_csv(out/(name+'_REPEAT_METRICS.csv'))
        assert len(saved) == groups
        for key,g in predictions.groupby(['repeat',*extras]):
            assert len(g)==n and g.row_id.nunique()==n
            selected = saved
            for field,value in zip(['repeat',*extras],key): selected=selected[selected[field]==value]
            assert len(selected)==1
            for field,value in v2.metric(g.y_true,g.probability).items():
                assert np.isclose(value,selected.iloc[0][field],atol=1e-10,rtol=0)
            count += 1
    # Recompute every decision curve from the frozen predictions.
    decisions = pd.read_csv(out/'DECISION_CURVES.csv')
    assert len(decisions)==627
    for cohort,name in [('source','CALIBRATION'),('target_adaptation','ADAPTATION')]:
        data = pd.read_csv(v2.OUT/(name+'_PREDICTIONS.csv'))
        for (r,m),g in data.groupby(['repeat','model_id']):
            saved=decisions[(decisions.cohort==cohort)&(decisions.repeat==r)&(decisions.model_id==m)]
            assert len(saved)==19
            for row in saved.itertuples():
                assert np.isclose(row.net_benefit,net_benefit(g.y_true,g.probability,row.threshold),atol=1e-12)
    # Reproduce source sensitivity diagnostic counts.
    predictions=pd.read_csv(out/'SENSITIVITY_PREDICTIONS.csv')
    diagnostics=pd.read_csv(out/'SENSITIVITY_REPEAT_METRICS.csv').set_index('repeat')
    for r,g in predictions.groupby('repeat'):
        assert len(g)==303 and g.row_id.nunique()==303
        assert np.array_equal(g.predicted,(g.probability>=g.threshold).astype(int))
        y,p=g.y_true.to_numpy(),g.predicted.to_numpy()
        tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
        tn=int(((y==0)&(p==0)).sum());fn=int(((y==1)&(p==0)).sum())
        for field,value in {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'sensitivity':tp/(tp+fn),
            'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'npv':tn/(tn+fn)}.items():
            assert np.isclose(value,diagnostics.loc[r,field],atol=1e-12)
    # Full-data learning curve must reproduce existing ordinary-core predictions.
    a=pd.read_csv(out/'SOURCE_LEARNING_PREDICTIONS.csv');a=a[a.fraction==1]
    b=pd.read_csv(v2.OUT/'MISSINGNESS_PREDICTIONS.csv');b=b[(b.model_id=='ordinary_core')&(b.scenario=='mcar_0')]
    joined=a.merge(b,on=['repeat','fold','row_id'])
    assert len(joined)==909 and np.allclose(joined.probability_x,joined.probability_y,atol=1e-12,rtol=0)
    return {'verified_metric_groups':count,'decision_curve_points':len(decisions),'default_changed':False}


if __name__=='__main__': print(json.dumps(verify(),indent=2))
