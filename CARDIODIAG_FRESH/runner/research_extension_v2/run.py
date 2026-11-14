"""Frozen additive reliability studies. No historical output or default is replaced."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import warnings

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from runner.phase6_zas_selection import run as legacy
from runner.research_extension_v1 import transport as tr
from runner.research_extension_v1 import zas as v1

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/research_extension_v2'
SEED = 921117
FULL = list(legacy.GROUPS['eligible_combined'])
CONTINUOUS = ['Age','BP','PR'] + list(legacy.GROUPS['laboratory'])
ORDINAL = ['Function Class']
BINARY = ['DM','HTN','Current Smoker','EX-Smoker','FH','Edema','Typical Chest Pain','Q Wave','St Elevation','St Depression','Tinversion']
NOMINAL = ['Sex','Dyspnea','Atypical','Nonanginal','Exertional CP','LowTH Ang','LVH','Poor R Progression','BBB']
GRIDS = [.1,1.,10.]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def csv_write(name, records):
    frame = records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/name,index=False)
    return frame


def schema():
    return {'continuous':CONTINUOUS,'ordinal':{'Function Class':[0,1,2,3]},'binary':{x:[0,1] for x in BINARY},
            'nominal':{x:(['Fmale','Male'] if x=='Sex' else ['N','LBBB','RBBB'] if x=='BBB' else ['N','Y']) for x in NOMINAL},
            'scope':'Recorded types/codes, not confirmed acquisition timing or clinical unit harmonization'}


def normalize_typed(frame):
    if frame.columns.duplicated().any() or not set(FULL).issubset(frame.columns):
        raise ValueError('Missing/duplicate typed source fields')
    x = frame[FULL].copy().replace('',np.nan)
    for c in CONTINUOUS+ORDINAL+BINARY:
        x[c] = pd.to_numeric(x[c],errors='raise')
        if np.isinf(x[c].to_numpy(dtype=float)).any(): raise ValueError(f'Infinite {c}')
    for c in BINARY:
        if not x[c].dropna().isin([0,1]).all(): raise ValueError(f'Invalid binary {c}')
    if not x['Function Class'].dropna().isin([0,1,2,3]).all(): raise ValueError('Unknown Function Class')
    x['Sex'] = x['Sex'].replace({'Female':'Fmale'})
    for c, categories in schema()['nominal'].items():
        if not x[c].dropna().isin(categories).all(): raise ValueError(f'Unknown nominal {c}')
    return x


def typed_pipeline():
    cat = BINARY+NOMINAL
    categories = [[0,1] for _ in BINARY]+[schema()['nominal'][c] for c in NOMINAL]
    transform = ColumnTransformer([
        ('continuous',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),CONTINUOUS),
        ('ordinal',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('scale',StandardScaler())]),ORDINAL),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(categories=categories,drop='if_binary',handle_unknown='error',sparse_output=False))]),cat),
    ],sparse_threshold=0)
    return Pipeline([('preprocess',transform),('model',LogisticRegression(solver='lbfgs',max_iter=3000,random_state=SEED))])


def fit_map(scores, labels, mode, seed=SEED):
    scores = np.asarray(scores,dtype=float)
    labels = np.asarray(labels,dtype=int)
    if len(scores)!=len(labels) or set(np.unique(labels))!={0,1}: raise ValueError('Calibration needs both classes')
    z = logit(np.clip(scores,1e-6,1-1e-6))
    if mode=='offset':
        offset = brentq(lambda a:float(expit(z+a).mean()-labels.mean()),-40,40)
        return {'mode':mode,'offset':float(offset)}
    est = LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,random_state=SEED)
    selection = []
    if mode=='logit_cv':
        search = GridSearchCV(est,{'C':GRIDS},scoring='neg_log_loss',cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1,error_score='raise').fit(z.reshape(-1,1),labels)
        est = search.best_estimator_
        selection = [{'C':float(p['C']),'mean_neg_log_loss':float(s)} for p,s in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'])]
    else: est.fit(z.reshape(-1,1),labels)
    return {'mode':mode,'C':float(est.C),'intercept':float(est.intercept_[0]),'slope':float(est.coef_[0,0]),'inner_scores':selection}


def apply_map(scores, mapping):
    z = logit(np.clip(np.asarray(scores,dtype=float),1e-6,1-1e-6))
    return expit(z+mapping['offset']) if mapping['mode']=='offset' else expit(mapping['intercept']+mapping['slope']*z)


def metric(y,p):
    y,p = np.asarray(y,dtype=int),np.asarray(p,dtype=float)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all(): raise ValueError('Invalid predictions')
    return {'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
            'ap':float(average_precision_score(y,p)),'brier':float(np.mean((p-y)**2)),
            'log_loss':float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]))}


def summarize(name, rows, extra=()):
    pred = csv_write(name+'_PREDICTIONS.csv',rows)
    keys = ['repeat','model_id',*extra]
    if pred.duplicated([*keys,'row_id']).any(): raise ValueError('Duplicate OOF row')
    records = []
    for key,g in pred.groupby(keys,sort=True):
        if not isinstance(key,tuple): key=(key,)
        records.append({**dict(zip(keys,key)),**metric(g.y_true,g.probability)})
    metrics = csv_write(name+'_REPEAT_METRICS.csv',records)
    means = metrics.groupby(['model_id',*extra]).mean(numeric_only=True).reset_index()
    csv_write(name+'_SUMMARY.csv',means)
    return pred,metrics,means


class AugmentedCore(ClassifierMixin,BaseEstimator):
    """All augmentation occurs inside estimator.fit on its training rows only."""
    def __init__(self,C=1.,seed=SEED,rate=.2): self.C,self.seed,self.rate=C,seed,rate

    def fit(self,X,y):
        X = pd.DataFrame(X,columns=tr.CORE).copy()
        y = np.asarray(y,dtype=int)
        masked = X.mask(np.random.default_rng(self.seed).random(X.shape)<self.rate)
        augmented = pd.concat([X,masked],ignore_index=True)
        values = ColumnTransformer([
            ('age',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),['Age']),
            ('binary',SimpleImputer(strategy='most_frequent'),tr.BINARY),
        ])
        self.pipeline_ = Pipeline([('features',FeatureUnion([('values',values),('flags',MissingIndicator(features='all',error_on_new=False))])),
                                   ('model',LogisticRegression(C=self.C,solver='lbfgs',max_iter=3000,random_state=SEED))])
        self.pipeline_.fit(augmented,np.tile(y,2),model__sample_weight=np.full(2*len(y),.5))
        self.classes_ = self.pipeline_.classes_
        self.n_features_in_ = len(tr.CORE)
        self.training_input_rows_ = len(X)
        return self

    def predict_proba(self,X): return self.pipeline_.predict_proba(X)


def input_paths():
    return [Path(__file__),OUT/'PROTOCOL.md',legacy.DATA_PATH,legacy.SPLIT_PATH,
            v1.OUT/'MANIFEST.json',v1.OUT/'PREDICTIONS.csv',v1.OUT/'SPLIT_TRACE.json',
            *sorted((v1.OUT/'checkpoints').glob('*corrected_component*.json')),
            tr.OUT/'TRANSPORT_MANIFEST.json',tr.OUT/'TARGET_PREDICTIONS.csv',tr.OUT/'SOURCE_OOF_PREDICTIONS.csv',
            tr.OUT/'SOURCE_MCAR_MASKS.csv',tr.SCREEN/'AngioCAD_Features.xlsx',tr.SCREEN/'AngioCAD_Labels.xlsx']


def freeze():
    from portable_release import verify_release
    verify_release(ROOT.parent,check_runtime=True)
    OUT.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in input_paths()}
    path = OUT/'FROZEN_PROTOCOL.json'
    if path.exists():
        frozen = json.loads(path.read_text(encoding='utf-8'))
        if frozen['inputs']!=hashes: raise RuntimeError('Frozen v2 input mismatch; create a new version')
        if (OUT/'MANIFEST.json').exists(): raise FileExistsError('v2 already complete; verify instead of overwriting')
        if any(OUT.glob('*PREDICTIONS.csv')): raise FileExistsError('Partial v2 outputs exist; investigate and use new namespace')
        return frozen
    frozen = {'frozen_utc_before_experiments':datetime.now(UTC).isoformat(),'inputs':hashes,
              'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']},
              'seed':SEED,'target_adaptation':'Retrospective cross-validation on previously explored cohort; not new external validation',
              'default_model_replacement':False,'missingness_augmentation_rate':.2,'target_folds':5,'repeats':3,'paired_bootstrap_draws':1000,'refit_bootstrap_draws':50}
    json_write(path,frozen)
    json_write(OUT/'FEATURE_SCHEMA.json',schema())
    return frozen


def source_typing(frame,y,outer,inner):
    x = normalize_typed(frame)
    rows, choices = [],[]
    for split in outer:
        r,f=split['repeat'],split['fold'];train,test=split['train'],split['test'];cv=inner[r,f]
        search=GridSearchCV(typed_pipeline(),{'model__C':GRIDS},cv=cv,scoring='roc_auc',n_jobs=1,error_score='raise').fit(x.iloc[train],y[train])
        oof=np.full(len(train),np.nan)
        for a,b in cv:
            est=typed_pipeline().set_params(**search.best_params_).fit(x.iloc[np.asarray(train)[a]],y[np.asarray(train)[a]])
            oof[b]=est.predict_proba(x.iloc[np.asarray(train)[b]])[:,1]
        mapping=legacy.fit_sigmoid(oof,y[train])
        p=mapping.predict_proba(search.predict_proba(x.iloc[test])[:,1].reshape(-1,1))[:,1]
        rows.extend({'repeat':r,'fold':f,'model_id':'typed38_historical_calibration','row_id':int(i),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
        choices.append({'repeat':r,'fold':f,'C':search.best_params_['model__C'],'inner_auc':float(search.best_score_)})
        print(f'Typed control {r}/{f}',flush=True)
    old=pd.read_csv(v1.OUT/'PREDICTIONS.csv')
    for model in ['l2_full38_historical_calibration','historical_selection']:
        g=old[old.model_id==model]
        rows.extend({'repeat':int(v.repeat),'fold':int(v.fold),'model_id':model,'row_id':int(v.row_index),'y_true':int(v.y_true),'probability':float(v.probability)} for v in g.itertuples())
    csv_write('TYPED_SELECTIONS.csv',choices)
    return summarize('TYPING',rows)


def calibration_reuse(y):
    rows,mappings=[] ,[]
    traces=json.loads((v1.OUT/'SPLIT_TRACE.json').read_text())
    for split in traces:
        r,f=split['repeat'],split['fold'];test=split['outer_test']
        arrays={m:[] for m in ['raw','weak_v1','offset','logit_c1','logit_cv']}
        for k in range(1,4):
            cp=json.loads((v1.OUT/'checkpoints'/f'r{r}_f{f}_corrected_component{k}.json').read_text())
            for m,key in [('raw','raw_probability'),('weak_v1','calibrated_probability')]: arrays[m].append(cp[key])
            for mode in ['offset','logit_c1','logit_cv']:
                mapping=fit_map(cp['calibration_raw_scores'],cp['calibration_y'],mode,SEED+1000*r+100*f+k)
                arrays[mode].append(apply_map(cp['raw_probability'],mapping))
                mappings.append({'repeat':r,'fold':f,'component':k,'calibration_row_ids':cp['trace']['calibration_rows'],'development_row_ids':cp['trace']['development_rows'],'outer_test_row_ids':test,**mapping})
        for mode,values in arrays.items():
            p=np.mean(values,axis=0)
            rows.extend({'repeat':r,'fold':f,'model_id':mode,'row_id':int(i),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
    json_write(OUT/'CALIBRATION_MAPS.json',mappings)
    return summarize('CALIBRATION',rows)


def missingness(frame,y,outer):
    x=tr.normalize_core(frame,source=True); masks=pd.read_csv(tr.OUT/'SOURCE_MCAR_MASKS.csv')
    rows, choices=[] ,[]
    for split in outer:
        r,f=split['repeat'],split['fold'];train,test=split['train'],split['test'];seed=tr.SEED+100*r+f
        raw=tr.make_search(seed).fit(x.iloc[train],y[train])
        augmented=GridSearchCV(AugmentedCore(seed=SEED+100*r+f),{'C':GRIDS},cv=StratifiedKFold(3,shuffle=True,random_state=seed),scoring='roc_auc',n_jobs=1,error_score='raise').fit(x.iloc[train],y[train])
        choices.append({'repeat':r,'fold':f,'baseline_C':raw.best_params_['model__C'],'augmented_C':augmented.best_params_['C']})
        scenarios={}
        for rate in [0.,.1,.25,.5]:
            m=masks[(masks.repeat==r)&(masks.fold==f)&(masks.mask_rate==rate)].set_index('source_row_index').loc[test]
            scenarios[f'mcar_{rate:g}']=x.iloc[test].mask(m[[f'mask_{c}' for c in tr.CORE]].to_numpy(dtype=bool))
        for name,cols in [('age_missing',['Age']),('smoker_missing',['Current Smoker']),('risk_block_missing',['DM','HTN','Current Smoker'])]:
            scenarios[name]=x.iloc[test].copy();scenarios[name].loc[:,cols]=np.nan
        for name,est in [('ordinary_core',raw),('augmented_core',augmented)]:
            for scenario,inputs in scenarios.items():
                p=est.predict_proba(inputs)[:,1]
                rows.extend({'repeat':r,'fold':f,'model_id':name,'scenario':scenario,'row_id':int(i),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
    csv_write('MISSINGNESS_SELECTIONS.csv',choices)
    return summarize('MISSINGNESS',rows,('scenario',))


def adaptation():
    target=pd.read_csv(tr.OUT/'TARGET_PREDICTIONS.csv')
    y=target.y_base_gt50.to_numpy(dtype=int);raw=target.raw_full_source_fit.to_numpy()
    rows, traces, maps=[] ,[],[]
    for repeat in [1,2,3]:
        for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED+repeat).split(raw,y),1):
            traces.append({'repeat':repeat,'fold':fold,'adaptation_case_ids':target.case_id.iloc[train].astype(int).tolist(),'evaluation_case_ids':target.case_id.iloc[test].astype(int).tolist()})
            probabilities={'frozen_raw':raw[test],'frozen_source_calibrated':target.source_calibrated_ensemble.iloc[test].to_numpy(),'adaptation_prevalence':np.full(len(test),y[train].mean())}
            for mode in ['offset','logit_c1','logit_cv']:
                mapping=fit_map(raw[train],y[train],mode,SEED+100*repeat+fold)
                probabilities['target_'+mode]=apply_map(raw[test],mapping)
                maps.append({'repeat':repeat,'fold':fold,**mapping})
            for model,p in probabilities.items():
                rows.extend({'repeat':repeat,'fold':fold,'model_id':model,'row_id':int(target.case_id.iloc[i]),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
    json_write(OUT/'TARGET_ADAPTATION_SPLITS.json',traces);json_write(OUT/'TARGET_ADAPTATION_MAPS.json',maps)
    pred,metrics,summary=summarize('ADAPTATION',rows)
    scenario_rows=[]
    indexed=target.set_index('case_id')
    for (repeat,model),g in pred.groupby(['repeat','model_id']):
        boundary=indexed.loc[g.row_id,'boundary'].to_numpy(dtype=bool)
        for scenario,keep,truth in [('base_gt50',np.ones(len(g),dtype=bool),g.y_true.to_numpy()),('exclude_boundary',~boundary,g.y_true.to_numpy()),('upper_assignment',np.ones(len(g),dtype=bool),np.where(boundary,1,g.y_true))]:
            scenario_rows.append({'repeat':repeat,'model_id':model,'endpoint_sensitivity':scenario,**metric(truth[keep],g.probability.to_numpy()[keep])})
    csv_write('ADAPTATION_ENDPOINT_SENSITIVITY.csv',scenario_rows)
    return pred,metrics,summary


def uncertainty(frame,y,cal_pred):
    rng=np.random.default_rng(SEED+40000)
    models=['raw','weak_v1','offset','logit_c1','logit_cv']
    arrays={m:np.stack([cal_pred[(cal_pred.model_id==m)&(cal_pred.repeat==r)].set_index('row_id').loc[range(303)].probability.to_numpy() for r in [1,2,3]]) for m in models}
    draws=[]
    for i in range(1000):
        ids=rng.integers(0,303,303)
        if len(np.unique(y[ids]))<2: continue
        for m in models:
            p=arrays[m][:,ids]
            draws.append({'draw':i,'model_id':m,'auc':float(np.mean([roc_auc_score(y[ids],a) for a in p])),'brier':float(np.mean((p-y[ids])**2))})
    draw_frame=csv_write('PAIRED_RECORD_BOOTSTRAP_DRAWS.csv',draws)
    intervals=[]
    for m in models:
        a=draw_frame[draw_frame.model_id==m].set_index('draw')
        ref=draw_frame[draw_frame.model_id=='raw'].set_index('draw')
        for col in ['auc','brier']:
            delta=a[col]-ref[col]
            intervals.append({'model_id':m,'contrast':'minus identical raw ensemble','metric':col,'record_bootstrap_025':float(delta.quantile(.025)),'record_bootstrap_975':float(delta.quantile(.975)),'scope':'conditional saved source fits, shared record draws across repeats; not patient/refitting confidence'})
    csv_write('PAIRED_RECORD_BOOTSTRAP_INTERVALS.csv',intervals)
    x=normalize_typed(frame);refits=[];membership=[]
    for i in range(50):
        ids=rng.integers(0,303,303);oob=np.setdiff1d(np.arange(303),np.unique(ids))
        split=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+i).split(x.iloc[ids],y[ids],groups=ids))
        for fit,valid in split:
            if set(ids[fit])&set(ids[valid]): raise ValueError('Bootstrap copy leaked between tuning folds')
        search=GridSearchCV(typed_pipeline(),{'model__C':GRIDS},cv=split,scoring='roc_auc',n_jobs=1,error_score='raise').fit(x.iloc[ids],y[ids])
        p=search.predict_proba(x.iloc[oob])[:,1]
        refits.append({'bootstrap_refit':i,'unique_training_rows':len(np.unique(ids)),'C':search.best_params_['model__C'],**metric(y[oob],p)})
        membership.append({'bootstrap_refit':i,'sampled_source_rows':ids.tolist(),'out_of_bag_rows':oob.tolist(),'inner_groups_disjoint':True})
    csv_write('BOOTSTRAP_REFIT_OOB_SENSITIVITY.csv',refits);json_write(OUT/'BOOTSTRAP_REFIT_MEMBERSHIP.json',membership)


def subgroups(frame,typing,calibration,adapted):
    source_meta=pd.DataFrame({'row_id':range(303),'Sex':frame.Sex.replace({'Fmale':'Female'}),'Age':frame.Age})
    target=tr.load_workbook(tr.SCREEN/'AngioCAD_Features.xlsx')
    target_meta=target[['ID','Sex','Age']].rename(columns={'ID':'row_id'});target_meta['row_id']=target_meta.row_id.astype(int);target_meta.Age=pd.to_numeric(target_meta.Age,errors='coerce')
    records=[]
    for cohort,pred,meta in [('ZAS typing',typing,source_meta),('ZAS calibration',calibration,source_meta),('AngioCAD adaptation',adapted,target_meta)]:
        joined=pred[pred.repeat==1].merge(meta,on='row_id',validate='many_to_one')
        joined['age_group']=np.where(joined.Age.isna(),'missing',np.where(joined.Age<=60,'<=60','>60'))
        for model,g in joined.groupby('model_id'):
            for feature in ['Sex','age_group']:
                for value,s in g.groupby(feature):
                    positive=int(s.y_true.sum());negative=len(s)-positive
                    records.append({'cohort':cohort,'model_id':model,'subgroup_variable':feature,'subgroup':value,'repeat':1,'unstable_small_class':min(positive,negative)<10,**metric(s.y_true,s.probability)})
    csv_write('SUBGROUP_DIAGNOSTICS.csv',records)


def decision(typing,calibration,missing,adapted):
    records=[]
    for track,data,candidate,reference in [
        ('typing',typing,'typed38_historical_calibration','l2_full38_historical_calibration'),
        *[('calibration',calibration,m,'weak_v1') for m in ['offset','logit_c1','logit_cv']],
        ('missingness_clean',missing[missing.scenario=='mcar_0'],'augmented_core','ordinary_core'),
        *[('target_adaptation',adapted,'target_'+m,'frozen_raw') for m in ['offset','logit_c1','logit_cv']],
    ]:
        means=data.groupby('model_id').mean(numeric_only=True)
        a,b=means.loc[candidate],means.loc[reference]
        gate=bool(a.roc_auc>=b.roc_auc-1e-10 and a.brier<=b.brier+1e-10)
        records.append({'track':track,'candidate':candidate,'reference':reference,'auc_delta':a.roc_auc-b.roc_auc,'brier_delta':a.brier-b.brier,'descriptive_nonregression_gate':gate,
                        'default_changed':False,'disposition':'additive research result; no automatic promotion' if gate else 'not promoted; retain established default and report trade-off'})
    return csv_write('DECISION_REGISTER.csv',records)


def run():
    frozen=freeze();started=datetime.now(UTC)
    frame=pd.read_csv(legacy.DATA_PATH);y=frame.Cath.map({'Cad':1,'Normal':0}).to_numpy(dtype=int)
    outer,inner=legacy.load_splits(legacy.SPLIT_PATH,y)
    with warnings.catch_warnings(record=True) as captured,threadpool_limits(limits=1):
        warnings.simplefilter('always')
        t=source_typing(frame,y,outer,inner)
        c=calibration_reuse(y);print('Saved-component calibration complete',flush=True)
        m=missingness(frame,y,outer);print('Missingness comparisons complete',flush=True)
        a=adaptation();print('Target adaptation complete',flush=True)
        uncertainty(frame,y,c[0]);print('Bootstrap sensitivity complete',flush=True)
        subgroups(frame,t[0],c[0],a[0]);d=decision(t[1],c[1],m[1],a[1])
    counts=Counter((w.category.__name__,str(w.message)) for w in captured)
    json_write(OUT/'WARNINGS.json',[{'category':k[0],'message':k[1],'count':v} for k,v in counts.items()])
    now=datetime.now(UTC)
    if frozen['inputs']!={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in input_paths()}: raise ValueError('Frozen inputs changed during run')
    outputs={p.relative_to(ROOT.parent).as_posix():sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
    json_write(OUT/'MANIFEST.json',{'completed_utc':now.isoformat(),'elapsed_seconds':(now-started).total_seconds(),'inputs':frozen['inputs'],'outputs':outputs,'default_model_changed':False,'retrospective_followup':True})
    print(d.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run: run()
    else: parser.print_help()
