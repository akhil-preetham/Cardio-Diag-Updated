"""Reconstruct paper metrics from frozen predictions; no training or unpickling."""
import argparse
import json
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from portable_release import discover_workspace, PROJECT, sha256
from paper_editorial import EVIDENCE_COMMIT

ROOT=discover_workspace();OUT=ROOT/PROJECT/'paper/final';RESULT=ROOT/PROJECT/'results'
SPECS=[
 ('source',RESULT/'research_extension_v1/zas/PREDICTIONS.csv',['repeat','model_id']),
 ('calibration',RESULT/'research_extension_v2/CALIBRATION_PREDICTIONS.csv',['repeat','model_id']),
 ('adaptation',RESULT/'research_extension_v2/ADAPTATION_PREDICTIONS.csv',['repeat','model_id']),
 ('typing',RESULT/'research_extension_v2/TYPING_PREDICTIONS.csv',['repeat','model_id']),
 ('missingness_v2',RESULT/'research_extension_v2/MISSINGNESS_PREDICTIONS.csv',['repeat','model_id','scenario']),
 ('mcar',RESULT/'research_extension_v1/transport/SOURCE_MCAR_STRESS_PREDICTIONS.csv',['repeat','mask_rate']),
 ('source_learning',RESULT/'final_analyses_v3/SOURCE_LEARNING_PREDICTIONS.csv',['repeat','fraction']),
 ('adaptation_learning',RESULT/'final_analyses_v3/ADAPTATION_LEARNING_PREDICTIONS.csv',['repeat','model_id','fraction']),
 ('perturbation',RESULT/'final_analyses_v3/PERTURBATION_PREDICTIONS.csv',['repeat','scenario']),
 ('sensitivity',RESULT/'final_analyses_v3/SENSITIVITY_PREDICTIONS.csv',['repeat']),
]


def collect():
    rows=[]
    for analysis,path,keys in SPECS:
        data=pd.read_csv(path);digest=sha256(path)
        for key,g in data.groupby(keys,sort=True):
            if not isinstance(key,tuple):key=(key,)
            fields=dict(zip(keys,key));y=g.y_true.to_numpy();p=g.probability.to_numpy()
            metrics={'roc_auc':roc_auc_score(y,p),'ap':average_precision_score(y,p),
                     'brier':float(np.mean((p-y)**2)),
                     'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])}
            if analysis=='sensitivity':
                label=g.predicted.to_numpy();tp=((y==1)&(label==1)).sum();fp=((y==0)&(label==1)).sum()
                tn=((y==0)&(label==0)).sum();fn=((y==1)&(label==0)).sum()
                metrics.update(sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),ppv=tp/(tp+fp),npv=tn/(tn+fn))
            condition={k:v for k,v in fields.items() if k not in ['repeat','model_id']}
            for metric,value in metrics.items():
                rows.append({'analysis':analysis,'repeat':int(fields['repeat']),
                    'model_id':fields.get('model_id','ordinary_core'),'condition':json.dumps(condition,sort_keys=True),
                    'n':len(g),'events':int(y.sum()),'metric':metric,'value':float(value),
                    'source_file':path.relative_to(ROOT).as_posix(),'source_sha256':digest})
    return pd.DataFrame(rows)


def write_manifest():
    paths=[p for _,p,_ in SPECS]
    paths += [ROOT/'AUTHORS.json']
    paths += [ROOT/'tools'/name for name in ['paper_evidence.py','paper_editorial.py',
        'build_final_paper.py','plot_final_analyses.py','build_submission_documents.py',
        'qa_final_paper.py','qa_submission_documents.py']]
    paths += [ROOT/PROJECT/'results'/name for name in [
        'research_extension_v1/zas/MANIFEST.json','research_extension_v1/transport/TRANSPORT_MANIFEST.json',
        'research_extension_v2/MANIFEST.json','final_analyses_v3/MANIFEST.json']]
    paths += [ROOT/PROJECT/'paper/research_extension_v2/CARDIODIAG_RESEARCH_REPORT_V2.tex']
    paths += [ROOT/name for name in json.loads((OUT/'FIGURE_INPUTS.json').read_text())['inputs']]
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.pdf','.tex','.csv','.json','.md','.png']
               and p.name!='PAPER_EVIDENCE_MANIFEST.json']
    manifest={'scope':'Editorial revision and computational metric verification; no independent clinical validation or new model fits',
        'evidence_baseline_commit':EVIDENCE_COMMIT,
        'input_hashes':{p.relative_to(ROOT).as_posix():sha256(p) for p in paths},
        'artifact_hashes':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(artifacts)}}
    (OUT/'PAPER_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def verify():
    manifest=json.loads((OUT/'PAPER_EVIDENCE_MANIFEST.json').read_text())
    for section in ['input_hashes','artifact_hashes']:
        for name,digest in manifest[section].items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe paper evidence path')
            target=(ROOT/name).resolve()
            if not target.is_relative_to(ROOT) or sha256(target)!=digest:raise ValueError('Paper evidence mismatch: '+name)
    actual=collect();saved=pd.read_csv(OUT/'NUMERIC_TRACEABILITY.csv')
    if actual.shape!=saved.shape:raise ValueError('Paper metric coverage changed')
    for col in actual.columns:
        if col=='value':
            if not np.allclose(actual[col],saved[col],atol=1e-12,rtol=0):raise ValueError('Paper metric reconstruction failed')
        elif not actual[col].equals(saved[col]):raise ValueError('Paper traceability differs: '+col)
    return {'reconstructed_metric_values':len(actual),'prediction_groups':len(actual[['analysis','repeat','model_id','condition']].drop_duplicates()),
            'artifacts_verified':len(manifest['artifact_hashes']),'new_model_fits':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true');parser.add_argument('--manifest',action='store_true')
    args=parser.parse_args()
    if args.write:collect().to_csv(OUT/'NUMERIC_TRACEABILITY.csv',index=False)
    elif args.manifest:write_manifest()
    else:print(json.dumps(verify(),indent=2))
