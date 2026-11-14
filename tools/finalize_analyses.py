"""Prespecified additive final analyses; never replaces a released model."""
from pathlib import Path
import json
import sys
import warnings
from datetime import datetime, UTC
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from portable_release import PROJECT, discover_workspace, sha256, verify_release

ROOT = discover_workspace()
sys.path.insert(0, str(ROOT / PROJECT))
from runner.research_extension_v2 import run as v2
tr = v2.tr
OUT = ROOT / PROJECT / 'results/final_analyses_v3'
SEED = 941117
FRACTIONS = [.25, .5, .75, 1.]


def threshold_for_sensitivity(y, p, target=.9):
    """Largest observed threshold attaining target training-OOF sensitivity."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Threshold selection needs finite probabilities and both classes')
    if not 0 < target <= 1:
        raise ValueError('Invalid sensitivity target')
    positives = np.sort(p[y == 1])[::-1]
    return float(positives[int(np.ceil(target * len(positives))) - 1])


def net_benefit(y, p, threshold):
    if not 0 < threshold < 1:
        raise ValueError('Decision threshold must be within (0,1)')
    y = np.asarray(y, dtype=int)
    positive = np.asarray(p) >= threshold
    return float(np.mean(positive & (y == 1)) - np.mean(positive & (y == 0))*threshold/(1-threshold))


def write_json(name, data):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def save(name, rows):
    frame = pd.DataFrame(rows)
    with (OUT / name).open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)
    return frame


def subset(ids, y, fraction, seed):
    ids = np.asarray(ids)
    if fraction == 1:
        return ids
    return train_test_split(ids, train_size=fraction, stratify=y[ids], random_state=seed)[0]


def run():
    verify_release(ROOT, check_runtime=True)
    from verify_reliability_v2 import verify
    verify(ROOT)
    OUT.mkdir(exist_ok=True)
    if any(OUT.glob('*.json')) or any(OUT.glob('*.csv')):
        raise FileExistsError('Final analysis namespace already used; verify, do not overwrite')
    protocol = {'frozen_utc': datetime.now(UTC).isoformat(), 'seed': SEED,
        'scope': 'Retrospective exploratory follow-up on previously examined cohorts; no clinical threshold recommendation',
        'source_fractions': FRACTIONS, 'target_adaptation_fractions': [.1, .25, .5, .75, 1.],
        'sensitivity_target': .9, 'decision_thresholds': np.arange(.05, 1., .05).round(2).tolist(),
        'coefficient_bootstrap_refits': 50, 'default_model_changed': False,
        'inputs': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [Path(__file__),
            tr.SOURCE, tr.SPLITS, tr.OUT/'TARGET_PREDICTIONS.csv',
            v2.OUT/'MANIFEST.json', v2.OUT/'CALIBRATION_PREDICTIONS.csv', v2.OUT/'ADAPTATION_PREDICTIONS.csv']}}
    write_json('FROZEN_PROTOCOL.json', protocol)
    frame = pd.read_csv(tr.SOURCE)
    y = frame.Cath.map({'Cad': 1, 'Normal': 0}).to_numpy()
    x = tr.normalize_core(frame, source=True)
    outer, _ = v2.legacy.load_splits(tr.SPLITS, y)
    sensitivity, traces, learning, perturbations, coefficients = [], [], [], [], []
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        for split in outer:
            r, f = split['repeat'], split['fold']
            train, test = np.asarray(split['train']), np.asarray(split['test'])
            seed = tr.SEED + 100*r + f
            base = tr.make_search(seed).fit(x.iloc[train], y[train])
            oof = np.full(len(train), np.nan)
            for k, (a,b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED+100*r+f).split(train, y[train])):
                estimator = tr.make_search(SEED+1000*r+100*f+k).fit(x.iloc[train[a]], y[train[a]])
                oof[b] = estimator.predict_proba(x.iloc[train[b]])[:,1]
                traces.append({'repeat':r, 'fold':f, 'threshold_inner_fold':k,
                    'search_fit_rows':train[a].tolist(), 'threshold_prediction_rows':train[b].tolist(), 'outer_test_rows':test.tolist()})
            threshold = threshold_for_sensitivity(y[train], oof)
            p = base.predict_proba(x.iloc[test])[:,1]
            sensitivity.extend({'repeat':r,'fold':f,'row_id':int(i),'y_true':int(y[i]),
                'probability':float(v),'threshold':threshold,'predicted':int(v>=threshold)} for i,v in zip(test,p))
            for fraction in FRACTIONS:
                ids = subset(train,y,fraction,SEED+100*r+f)
                est = base if fraction == 1 else tr.make_search(seed).fit(x.iloc[ids],y[ids])
                q = est.predict_proba(x.iloc[test])[:,1]
                learning.extend({'repeat':r,'fold':f,'fraction':fraction,'training_n':len(ids),
                    'row_id':int(i),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,q))
            scenarios = {'unchanged':x.iloc[test].copy()}
            for name, delta in [('age_minus5',-5),('age_plus5',5)]:
                scenarios[name] = x.iloc[test].copy()
                scenarios[name]['Age'] = (scenarios[name].Age+delta).clip(18,110)
            scenarios['age_round5'] = x.iloc[test].copy()
            scenarios['age_round5']['Age'] = (scenarios['age_round5'].Age/5).round()*5
            scenarios['binary_flip5pct'] = x.iloc[test].copy()
            rng = np.random.default_rng(SEED+100*r+f)
            flips = rng.random((len(test),4)) < .05
            z = scenarios['binary_flip5pct'][tr.BINARY].to_numpy()
            scenarios['binary_flip5pct'].loc[:,tr.BINARY] = np.where(flips,1-z,z)
            for name, values in scenarios.items():
                q = base.predict_proba(values)[:,1]
                perturbations.extend({'repeat':r,'fold':f,'scenario':name,'row_id':int(i),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,q))
            print(f'Final source analyses {r}/{f}',flush=True)
        rng = np.random.default_rng(SEED)
        for b in range(50):
            ids = rng.integers(0,len(y),len(y))
            # Duplicate record copies stay in the same C-tuning fold.
            from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
            cv = list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+b).split(x.iloc[ids],y[ids],groups=ids))
            for a,c in cv:
                assert not set(ids[a]) & set(ids[c])
            template = tr.make_search(SEED+b)
            est = GridSearchCV(template.estimator,template.param_grid,cv=cv,scoring='roc_auc',n_jobs=1,error_score='raise').fit(x.iloc[ids],y[ids])
            values = est.best_estimator_.named_steps['model'].coef_[0]
            coefficients.extend({'bootstrap':b,'feature':name,'coefficient':float(value),
                'C':est.best_params_['model__C']} for name,value in zip(tr.CORE,values))
        target = pd.read_csv(tr.OUT/'TARGET_PREDICTIONS.csv')
        yt = target.y_base_gt50.to_numpy();raw = target.raw_full_source_fit.to_numpy()
        adapted, adaptation_trace = [], []
        for r in [1,2,3]:
            for f,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=v2.SEED+r).split(raw,yt),1):
                for fraction in [.1,.25,.5,.75,1.]:
                    ids = subset(train,yt,fraction,SEED+100*r+f)
                    mapping = v2.fit_map(raw[ids],yt[ids],'offset')
                    q = v2.apply_map(raw[test],mapping)
                    for model, values in [('target_offset',q),('adaptation_prevalence',np.full(len(test),yt[ids].mean()))]:
                        adapted.extend({'repeat':r,'fold':f,'fraction':fraction,'model_id':model,'adaptation_n':len(ids),
                            'row_id':int(target.case_id.iloc[i]),'y_true':int(yt[i]),'probability':float(v)} for i,v in zip(test,values))
                    adaptation_trace.append({'repeat':r,'fold':f,'fraction':fraction,
                        'adaptation_case_ids':target.case_id.iloc[ids].astype(int).tolist(),
                        'evaluation_case_ids':target.case_id.iloc[test].astype(int).tolist(),**mapping})
        for name,rows,extras in [('SOURCE_LEARNING',learning,['fraction']),('PERTURBATION',perturbations,['scenario']),
                                  ('ADAPTATION_LEARNING',adapted,['fraction','model_id'])]:
            data = save(name+'_PREDICTIONS.csv',rows)
            results = []
            for keys,g in data.groupby(['repeat',*extras]):
                results.append({**dict(zip(['repeat',*extras],keys)),**v2.metric(g.y_true,g.probability)})
            metrics = save(name+'_REPEAT_METRICS.csv',results)
            save(name+'_SUMMARY.csv',metrics.groupby(extras).mean(numeric_only=True).reset_index())
        data = save('SENSITIVITY_PREDICTIONS.csv',sensitivity)
        diagnostics = []
        for r,g in data.groupby('repeat'):
            tn,fp,fn,tp = confusion_matrix(g.y_true,g.predicted,labels=[0,1]).ravel()
            diagnostics.append({'repeat':int(r),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),
                'ppv':tp/(tp+fp),'npv':tn/(tn+fn),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)})
        save('SENSITIVITY_REPEAT_METRICS.csv',diagnostics)
        write_json('THRESHOLD_SPLIT_TRACE.json',traces)
        write_json('ADAPTATION_LEARNING_TRACE.json',adaptation_trace)
        coef = save('COEFFICIENT_BOOTSTRAPS.csv',coefficients)
        save('COEFFICIENT_STABILITY.csv',[{'feature':name,'median':g.coefficient.median(),
            'q025':g.coefficient.quantile(.025),'q975':g.coefficient.quantile(.975),
            'positive_fraction':(g.coefficient>0).mean()} for name,g in coef.groupby('feature')])
        decisions = []
        for cohort,name in [('source','CALIBRATION'),('target_adaptation','ADAPTATION')]:
            data = pd.read_csv(v2.OUT/(name+'_PREDICTIONS.csv'))
            for (r,m),g in data.groupby(['repeat','model_id']):
                for t in protocol['decision_thresholds']:
                    prevalence = float(g.y_true.mean())
                    decisions.append({'cohort':cohort,'repeat':r,'model_id':m,'threshold':t,
                        'net_benefit':net_benefit(g.y_true,g.probability,t),
                        'investigate_all':prevalence-(1-prevalence)*t/(1-t),'investigate_none':0.})
        save('DECISION_CURVES.csv',decisions)
    write_json('WARNINGS.json',[{'category':w.category.__name__,'message':str(w.message)} for w in captured])
    for name,digest in protocol['inputs'].items():
        if sha256(ROOT/name) != digest: raise ValueError('Input changed during final run')
    write_json('MANIFEST.json',{'inputs':protocol['inputs'],'outputs':{
        p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'completed_utc':datetime.now(UTC).isoformat(),'default_model_changed':False})
    print('Final analyses complete; released defaults unchanged',flush=True)


if __name__ == '__main__':
    run()
