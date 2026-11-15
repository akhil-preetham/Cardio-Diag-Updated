"""Final report plots, using actual Times New Roman where installed."""
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_curve
from portable_release import discover_workspace, PROJECT, sha256


def main():
    root=discover_workspace();out=root/PROJECT/'paper/final';out.mkdir(exist_ok=True)
    font=os.environ.get('CARDIODIAG_TIMES_FONT','C:/Windows/Fonts/times.ttf')
    if not os.path.isfile(font): raise FileNotFoundError('Set CARDIODIAG_TIMES_FONT to licensed Times New Roman')
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':10,
                         'axes.labelsize':10,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42})
    result=root/PROJECT/'results/final_analyses_v3';old=root/PROJECT/'results/research_extension_v2'
    fig,axes=plt.subplots(3,2,figsize=(7,8.5))
    for col,name,models in [(0,'CALIBRATION',['raw','weak_v1','logit_c1']),
                           (1,'ADAPTATION',['frozen_raw','target_offset','target_logit_c1'])]:
        data=pd.read_csv(old/(name+'_PREDICTIONS.csv'))
        for model in models:
            g=data[(data.repeat==1)&(data.model_id==model)]
            fpr,tpr,_=roc_curve(g.y_true,g.probability)
            label={'raw':'Raw','weak_v1':'V1 sigmoid','logit_c1':'C1 sigmoid','frozen_raw':'Source raw',
                   'target_offset':'Target offset','target_logit_c1':'Target C1'}[model]
            axes[0,col].plot(fpr,tpr,label=label)
        axes[0,col].plot([0,1],[0,1],'--',color='gray')
        axes[0,col].set(title=('Source' if col==0 else 'Target')+' ROC: repeat 1',xlabel='False-positive rate',ylabel='True-positive rate')
        axes[0,col].legend(loc='lower right')
    data=pd.read_csv(result/'SOURCE_LEARNING_SUMMARY.csv')
    axes[1,0].plot(data.fraction,data.roc_auc,marker='o')
    axes[1,0].set(title='Source development size',xlabel='Fraction of outer training set',ylabel='Mean out-of-fold AUC')
    data=pd.read_csv(result/'ADAPTATION_LEARNING_SUMMARY.csv')
    for name,g in data.groupby('model_id'):
        axes[1,1].plot(g.fraction,g.brier,marker='o',label='Offset' if name=='target_offset' else 'Prevalence')
    axes[1,1].set(title='Target adaptation size',xlabel='Fraction of adaptation folds',ylabel='Mean held-out Brier')
    axes[1,1].legend()
    data=pd.read_csv(result/'DECISION_CURVES.csv')
    for col,cohort,models in [(0,'source',['raw','logit_c1']),(1,'target_adaptation',['frozen_raw','target_offset'])]:
        selected=data[data.cohort==cohort]
        for model in models:
            g=selected[selected.model_id==model].groupby('threshold').mean(numeric_only=True)
            axes[2,col].plot(g.index,g.net_benefit,label='Raw' if model in ['raw','frozen_raw'] else 'Updated')
        g=selected.groupby('threshold').mean(numeric_only=True)
        axes[2,col].plot(g.index,g.investigate_all,'--',label='All')
        axes[2,col].axhline(0,color='gray',linestyle=':',label='None')
        axes[2,col].set(title=('Source' if col==0 else 'Target')+' decision curves',xlabel='Research threshold',ylabel='Mean net benefit',ylim=(-.1,.85))
        axes[2,col].legend(ncol=2)
    fig.tight_layout(pad=.6)
    fig.savefig(out/'FINAL_ANALYSES.pdf');fig.savefig(out/'FINAL_ANALYSES.png',dpi=220)
    plt.close(fig)
    # Editorial package charts read reconstructed metrics, never pasted arrays.
    data=pd.read_csv(out/'NUMERIC_TRACEABILITY.csv')
    metrics=data.groupby(['analysis','model_id','condition','metric']).value.mean()
    charts=[('SOURCE_COMPARISON','source','roc_auc',[
        ('historical_v3_balanced','Five-field reference'),('l2_no_symptoms28_historical_calibration','No symptoms (28)'),
        ('l2_risk_symptoms17_historical_calibration','Risk + symptoms (17)'),
        ('historical_selection','Historical selection'),('l2_full38_historical_calibration','Equal-feature L2 (38)')]),
        ('CALIBRATION_LOSS','calibration','brier',[(x,label) for x,label in
            [('raw','Identical raw components'),('weak_v1','Previous sigmoid'),('offset','Intercept offset'),
             ('logit_c1','Fixed-C1 sigmoid'),('logit_cv','CV sigmoid')]]),
        ('TARGET_LOSS','adaptation','brier',[(x,label) for x,label in
            [('frozen_raw','Frozen source raw'),('frozen_source_calibrated','Frozen source calibrated'),
             ('adaptation_prevalence','Target prevalence'),('target_offset','Target offset'),
             ('target_logit_c1','Target C1'),('target_logit_cv','Target CV')]])]
    for filename,analysis,metric,models in charts:
        values=[metrics.loc[(analysis,model,'{}',metric)] for model,_ in models]
        figure,ax=plt.subplots(figsize=(6.2,2.7))
        ax.barh([label for _,label in models],values,color='0.65',edgecolor='0.2')
        ax.set(xlabel='Mean complete-repeat ROC-AUC' if metric=='roc_auc' else 'Mean complete-repeat Brier',
               xlim=(0,1) if metric=='roc_auc' else (0,max(values)*1.18))
        ax.invert_yaxis();figure.tight_layout();figure.savefig(out/(filename+'.pdf'));plt.close(figure)
    masked=data[(data.analysis=='mcar')&(data.metric=='roc_auc')].copy()
    masked['mask_rate']=masked.condition.map(lambda x:json.loads(x)['mask_rate'])
    means=masked.groupby('mask_rate').value.mean()
    figure,ax=plt.subplots(figsize=(6.2,2.6));ax.plot(means.index*100,means.values,marker='o')
    ax.set(xlabel='Held-out predictor cells removed (%)',ylabel='Mean complete-repeat ROC-AUC',ylim=(.6,.85))
    figure.tight_layout();figure.savefig(out/'MISSINGNESS_STRESS.pdf');plt.close(figure)
    inputs=[root/PROJECT/'results/research_extension_v2'/name for name in
            ['CALIBRATION_PREDICTIONS.csv','ADAPTATION_PREDICTIONS.csv']]
    inputs += [result/name for name in ['SOURCE_LEARNING_SUMMARY.csv','ADAPTATION_LEARNING_SUMMARY.csv','DECISION_CURVES.csv']]
    inputs += [out/'NUMERIC_TRACEABILITY.csv',Path(__file__)]
    (out/'FIGURE_INPUTS.json').write_text(json.dumps({'scope':'Plots of frozen evidence; no new model fits',
        'inputs':{p.relative_to(root).as_posix():sha256(p) for p in inputs}},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
