"""Generate the reference-format paper's own workflow and paired data plots."""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from sklearn.metrics import roc_curve
from portable_release import discover_workspace, PROJECT


def main():
    root=discover_workspace();out=root/PROJECT/'paper/reference_format';out.mkdir(exist_ok=True)
    font=Path(os.environ.get('CARDIODIAG_TIMES_FONT','C:/Windows/Fonts/times.ttf'))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
                         'axes.titlesize':10,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10})
    old=root/PROJECT/'results/research_extension_v2';final=root/PROJECT/'results/final_analyses_v3'
    width=(11906-1700)/1440
    figure,ax=plt.subplots(figsize=(width,1.9));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    rows=[['ZAS records\n303 source rows','15 outer partitions\ntraining and test','Selection on D\nCalibration on C','Held-out scores\nmatched raw and mapped'],
          ['Five-core source fit\nsource labels only','Frozen AngioCAD\n377 matched records','Separate target update\nother folds labels','Held-out evaluation\nupdating and stability']]
    for j,labels in enumerate(rows):
        y=2.05 if j==0 else .65
        for i,label in enumerate(labels):
            x=.05+i*2.52
            ax.add_patch(FancyBboxPatch((x,y-.35),2.25,.9,boxstyle='round,pad=0.03',facecolor='#eef1f4',edgecolor='#34495e'))
            ax.text(x+1.125,y+.1,label,ha='center',va='center',fontsize=10)
            if i<3:ax.annotate('',xy=(x+2.49,y+.1),xytext=(x+2.29,y+.1),arrowprops={'arrowstyle':'->','color':'#34495e'})
    figure.tight_layout(pad=.1);figure.savefig(out/'workflow.png',dpi=320);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(width,2.25))
    for axis,name,models in [(axes[0],'CALIBRATION',[('raw','Raw'),('weak_v1','Previous sigmoid'),('logit_c1','Fixed C1')]),
                            (axes[1],'ADAPTATION',[('frozen_raw','Source raw'),('target_offset','Target offset'),('target_logit_c1','Target C1')])]:
        frame=pd.read_csv(old/(name+'_PREDICTIONS.csv'))
        for model,label in models:
            g=frame[(frame.repeat==1)&(frame.model_id==model)];x,y,_=roc_curve(g.y_true,g.probability)
            axis.plot(x,y,label=label)
        axis.plot([0,1],[0,1],'--',color='gray');axis.legend(loc='lower right')
        axis.set(xlabel='False positive rate',ylabel='True positive rate',title='Source ROC' if name=='CALIBRATION' else 'Target ROC')
    figure.tight_layout(pad=.45);figure.savefig(out/'roc.png',dpi=320);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(width,2.25))
    frame=pd.read_csv(final/'SOURCE_LEARNING_SUMMARY.csv')
    axes[0].plot(frame.fraction,frame.roc_auc,marker='o');axes[0].set(title='Source development size',xlabel='Training fraction',ylabel='Mean out of fold AUC')
    frame=pd.read_csv(final/'ADAPTATION_LEARNING_SUMMARY.csv')
    for name,g in frame.groupby('model_id'):
        axes[1].plot(g.fraction,g.brier,marker='o',label='Offset' if name=='target_offset' else 'Prevalence')
    axes[1].set(title='Target updating size',xlabel='Adaptation fraction',ylabel='Mean held out Brier');axes[1].legend()
    figure.tight_layout(pad=.45);figure.savefig(out/'learning.png',dpi=320);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(width,2.25));frame=pd.read_csv(final/'DECISION_CURVES.csv')
    for axis,cohort,models in [(axes[0],'source',['raw','logit_c1']),(axes[1],'target_adaptation',['frozen_raw','target_offset'])]:
        selected=frame[frame.cohort==cohort]
        for model in models:
            g=selected[selected.model_id==model].groupby('threshold').mean(numeric_only=True)
            axis.plot(g.index,g.net_benefit,label='Raw' if model in ['raw','frozen_raw'] else 'Updated')
        g=selected.groupby('threshold').mean(numeric_only=True)
        axis.plot(g.index,g.investigate_all,'--',label='All');axis.axhline(0,color='gray',linestyle=':',label='None')
        axis.set(title='Source decision curves' if cohort=='source' else 'Target decision curves',xlabel='Research threshold',ylabel='Mean net benefit',ylim=(-.1,.85));axis.legend(ncol=2)
    figure.tight_layout(pad=.45);figure.savefig(out/'decision.png',dpi=320);plt.close(figure)


if __name__=='__main__':main()
