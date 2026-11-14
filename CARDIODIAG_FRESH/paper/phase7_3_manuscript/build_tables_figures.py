from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
root=Path.cwd(); base=root/'paper/phase7_3_manuscript'; tdir=base/'TABLES'; fdir=base/'FIGURES'
for d in [tdir,fdir]: d.mkdir(parents=True,exist_ok=True)
# Source tables
rep=pd.read_csv(root/'results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv')
paired=pd.read_csv(root/'results/phase6_zas_selection/PAIRED_REPEAT_DIFFERENCES.csv')
stage=pd.read_csv(root/'results/phase4_validation_audit/FINAL_STAGE_COMPARISON.csv')
cent=pd.read_csv(root/'results/phase4_validation_audit/INDEPENDENT_METRIC_REPRODUCTION.csv')
# cohort table
coh=pd.DataFrame([
['ZAS','Tehran Rajaei source dataset',303,216,87,'71.3%','Cath: Cad vs Normal; UCI metadata says CAD if diameter narrowing >=50%','No verified patient ID; local validated CSV has 56 columns, while repository source descriptions report 54 feature variables; schema discrepancy unresolved'],
['UCI Cleveland','Cleveland',303,139,164,'45.9%','num 0 vs 1–4','One of four centres; source rows keyed with centre'],
['UCI Hungary','Hungary',294,106,188,'36.1%','num 0 vs 1–4','One of four centres; source rows keyed with centre'],
['UCI Switzerland','Switzerland',123,115,8,'93.5%','num 0 vs 1–4','Only 8 negatives; centre metrics particularly unstable'],
['UCI VA Long Beach','VA Long Beach',200,149,51,'74.5%','num 0 vs 1–4','One of four centres; source rows keyed with centre'],
['UCI total','Four-centre total',920,509,411,'55.3%','num 0 vs 1–4','Four-centre cohort; not external validation of ZAS']
],columns=['Dataset/centre','Source','Records','Positive','Negative','Positive prevalence','Endpoint','Identity/schema note'])
coh.to_csv(tdir/'TABLE1_COHORTS.csv',index=False)
# Feature groups
features=pd.DataFrame([
['V3 reference','Age; Sex; Current Smoker; EX-Smoker; FH','Timing uncertain (recording time unverified)','Historical comparator, not timestamp-confirmed'],
['Basic risk','Age; Sex; DM; HTN; Current Smoker; EX-Smoker; FH','Timing uncertain','Phase 6 selection candidate'],
['Symptoms/examination','Typical Chest Pain; Dyspnea; Function Class; Atypical; Nonanginal; Exertional CP; LowTH Ang; BP; PR; Edema','Timing uncertain','Phase 6 selection candidate'],
['Resting ECG','Q Wave; St Elevation; St Depression; Tinversion; LVH; Poor R Progression; BBB','Timing uncertain','Phase 6 selection candidate'],
['Laboratory','FBS; CR; TG; LDL; HDL; BUN; ESR; HB; K; Na; WBC; Lymph; Neut; PLT','Timing/units incompletely documented','Phase 6 selection candidate'],
['Eligible combined','Union of basic risk, symptoms/examination, resting ECG and laboratory groups (38 fields)','Timing uncertain; “eligible” is a historical group label only','Selected in 15/15 Phase 6 outer fits'],
['Excluded','Cath; angiographic fields; EF-TTE; Region RWMA; VHD','Outcome/direct or potentially downstream','Not used as predictors']
],columns=['Group','Features','Timing classification','Role'])
features.to_csv(tdir/'TABLE2_FEATURE_GROUPS.csv',index=False)
# Phase6 performance wide table
wide=[]
for r in [1,2,3]:
 a=rep[(rep.repeat==r)&(rep.model_id=='selection_aware_procedure')].iloc[0]
 b=rep[(rep.repeat==r)&(rep.model_id=='frozen_v3_logistic_balanced')].iloc[0]
 d=paired[paired.repeat==r].iloc[0]
 wide.append([r,303,216,.712871,a.roc_auc,b.roc_auc,d.roc_auc_difference_selection_minus_v3,a.pr_auc_average_precision,b.pr_auc_average_precision,a.brier_score,b.brier_score,a.log_loss,b.log_loss,a.sensitivity,b.sensitivity,a.specificity,b.specificity])
p6=pd.DataFrame(wide,columns=['Repeat','Rows each','Events','Prevalence','Selection ROC-AUC','V3-style ROC-AUC','Paired delta ROC-AUC','Selection AP','V3-style AP','Selection Brier','V3-style Brier','Selection log loss','V3-style log loss','Selection sensitivity','V3-style sensitivity','Selection specificity','V3-style specificity'])
p6.to_csv(tdir/'TABLE3_PHASE6_PRIMARY.csv',index=False)
# UCI table metrics
uci=[]
for _,r in stage.iterrows():
 uci.append([r.stage,920,r.pooled_roc_auc, float('nan'),r.pooled_brier, float('nan'),r.centre_auc_min,r.centre_auc_max])
# use exact AP/logloss from audited report table, verified source
ap=[.729290,.824077,.819088,.864469]; ll=[.611668,.510644,.513922,.486086]
for i,row in enumerate(uci): row[3]=ap[i];row[5]=ll[i]
uci_df=pd.DataFrame(uci,columns=['UCI stage','Records','Pooled ROC-AUC','Average precision','Brier','Log loss','Centre ROC-AUC minimum','Centre ROC-AUC maximum'])
uci_df.to_csv(tdir/'TABLE4_UCI_STAGES.csv',index=False)
# Paired increments table
pairs=pd.read_csv(root/'results/phase4_validation_audit/PAIRED_STAGE_COMPARISONS.csv')
r=pairs[(pairs.stage_first=='stage3_resting_measurements')&(pairs.stage_second=='stage4_exercise_test')&(pairs.scope=='pooled')].iloc[0]
inc=pd.DataFrame([
['Stage 3 → Stage 4','ROC-AUC',r.roc_auc_difference,r.roc_auc_ci_low,r.roc_auc_ci_high,400,'Paired record bootstrap; centre/source-row paired'],
['Stage 3 → Stage 4','Average precision',r.pr_auc_difference,r.pr_auc_ci_low,r.pr_auc_ci_high,400,'Paired record bootstrap; centre/source-row paired'],
['Stage 3 → Stage 4','Brier',r.brier_difference,r.brier_ci_low,r.brier_ci_high,400,'Paired record bootstrap; centre/source-row paired'],
['Stage 3 → Stage 4','Log loss',r.log_loss_difference,r.log_loss_ci_low,r.log_loss_ci_high,400,'Paired record bootstrap; centre/source-row paired']
],columns=['Comparison','Metric','Difference (later−earlier)','95% interval low','95% interval high','Bootstrap draws','Uncertainty unit'])
inc.to_csv(tdir/'TABLE5_UCI_PAIRED.csv',index=False)
# markdown table renderer helper (avoid an undeclared optional tabulate dependency)
def fmt(v):
 if pd.isna(v): return ''
 if isinstance(v,(int,float)):
  return f'{v:.4f}' if not float(v).is_integer() else str(int(v))
 return str(v).replace('|','\\|')
def md_table(df):
 headers=list(df.columns); lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
 for _,row in df.iterrows(): lines.append('| '+' | '.join(fmt(v) for v in row.tolist())+' |')
 return '\n'.join(lines)+'\n'
for f in tdir.glob('*.csv'):
 df=pd.read_csv(f)
 (tdir/(f.stem+'.md')).write_text(md_table(df),encoding='utf-8')
# Style
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.dpi':180,'savefig.dpi':300})
# Fig 1 study design flow
fig,ax=plt.subplots(figsize=(11,6.2));ax.set_xlim(0,11);ax.set_ylim(0,7);ax.axis('off')
def box(x,y,w,h,text,fc,fs=10):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.12',facecolor=fc,edgecolor='#334155',linewidth=1.1);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#475569',linewidth=1.2))
box(.35,5.4,3.0,1.05,'ZAS cohort\n303 source rows; Cath outcome\n216 Cad / 87 Normal','#e0f2fe')
box(4.0,5.4,3.0,1.05,'Phase 6 nested selection\n6 groups × 5 unique model families\ninner selection; training-only preprocessing','#dcfce7',9)
box(7.65,5.4,3.0,1.05,'15 paired outer folds\n3 repeats × 5 folds\ncomplete repeat OOF predictions','#fef3c7')
arrow(3.35,5.92,4.0,5.92);arrow(7.0,5.92,7.65,5.92)
box(1.2,3.35,3.7,1.1,'Primary ZAS comparison\nSelection-aware procedure vs re-evaluated\nV3-style comparator on matched splits','#dbeafe',9)
box(6.1,3.35,3.7,1.1,'Separate UCI track\n920 records; four centres\nLeave-one-centre-out staged analyses','#f3e8ff',9)
arrow(9.15,5.4,8.6,4.45);arrow(8.2,5.4,3.2,4.45)
box(1.2,1.1,3.7,1.1,'Interpretation boundary\nRetrospective discrimination only\nTiming and patient identity unverified','#fee2e2',9)
box(6.1,1.1,3.7,1.1,'Not external validation of ZAS\nDifferent endpoint coding, features,\nmissingness and validation unit','#fee2e2',9)
arrow(3.05,3.35,3.05,2.2);arrow(7.95,3.35,7.95,2.2)
fig.tight_layout();fig.savefig(fdir/'FIGURE1_STUDY_WORKFLOW.png',bbox_inches='tight');fig.savefig(fdir/'FIGURE1_STUDY_WORKFLOW.pdf',bbox_inches='tight');plt.close(fig)
# Fig 2 paired repeat AUCs
fig,ax=plt.subplots(figsize=(7.8,4.8));x=[1,2,3]
for i,(label,col,color,marker) in enumerate([('Selection-aware procedure','Selection ROC-AUC','#176B87','o'),('V3-style comparator','V3-style ROC-AUC','#D97706','s')]):
 ax.plot(x,p6[col],marker=marker,color=color,lw=2.1,ms=7,label=label)
 for xx,yy in zip(x,p6[col]): ax.annotate(f'{yy:.3f}',(xx,yy),xytext=(0,9 if i==0 else -17),textcoords='offset points',ha='center',fontsize=8,color=color)
ax.set_xticks(x,['Repeat 1','Repeat 2','Repeat 3']);ax.set_ylabel('Complete-repeat out-of-fold ROC-AUC');ax.set_ylim(.65,1.0);ax.grid(axis='y',alpha=.25);ax.legend(frameon=False,loc='lower right');ax.set_title('Paired ZAS repeat-level discrimination');fig.text(.5,.01,'Each repeat predicts all 303 source rows once. Repeats overlap; points are not independent samples.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(fdir/'FIGURE2_PHASE6_REPEAT_AUC.png',bbox_inches='tight');fig.savefig(fdir/'FIGURE2_PHASE6_REPEAT_AUC.pdf',bbox_inches='tight');plt.close(fig)
# Fig 3 UCI centre AUC heatmap
center_auc=pd.read_csv(root/'results/phase4_validation_audit/INDEPENDENT_METRIC_REPRODUCTION.csv')
auc=center_auc[(center_auc.metric=='roc_auc')].pivot(index='held_out_centre',columns='stage',values='recomputed')
order=['stage1_basic_risk','stage2_symptoms','stage3_resting_measurements','stage4_exercise_test']; centers=['Cleveland','Hungary','Switzerland','VA Long Beach'];auc=auc.loc[centers,order]
fig,ax=plt.subplots(figsize=(9,4.5));im=ax.imshow(auc.values,vmin=.45,vmax=.95,cmap='YlGnBu',aspect='auto')
for i,c in enumerate(centers):
 for j,s in enumerate(order):ax.text(j,i,f'{auc.iloc[i,j]:.3f}',ha='center',va='center',color='white' if auc.iloc[i,j]>.78 else '#102a43',fontsize=9)
ax.set_xticks(range(4),['Stage 1','Stage 2','Stage 3','Stage 4']);ax.set_yticks(range(4),['Cleveland (n=303)','Hungary (n=294)','Switzerland (n=123; 8 negatives)','VA Long Beach (n=200)']);ax.set_title('UCI leave-one-centre-out ROC-AUC by held-out centre');fig.colorbar(im,ax=ax,label='ROC-AUC',shrink=.85);fig.text(.5,.01,'Centre estimates are descriptive; Switzerland has only eight negative records.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(fdir/'FIGURE3_UCI_CENTRE_AUC.png',bbox_inches='tight');fig.savefig(fdir/'FIGURE3_UCI_CENTRE_AUC.pdf',bbox_inches='tight');plt.close(fig)
# Fig 4 timing status diagram
fig,ax=plt.subplots(figsize=(9,4.7));ax.axis('off');ax.set_xlim(0,9);ax.set_ylim(0,5)
cols=[('#dcfce7','Confirmed before decision','None established from source timestamps'),('#fef3c7','Timing uncertain','Demographics, history, symptoms, ECG, labs'),('#ffedd5','Potentially downstream','TTE / RWMA / VHD; work-up timing not proven'),('#fee2e2','Outcome / direct outcome fields','Cath, angiographic fields; excluded from predictors')]
for i,(fc,title,sub) in enumerate(cols):
 y=3.95-i*1.05
 p=FancyBboxPatch((.35,y),8.3,.78,boxstyle='round,pad=.03,rounding_size=.08',facecolor=fc,edgecolor='#64748b');ax.add_patch(p);ax.text(.6,y+.49,title,ha='left',va='center',weight='bold',fontsize=10);ax.text(4.05,y+.49,sub,ha='left',va='center',fontsize=9)
ax.set_title('ZAS feature timing and leakage status',pad=14);fig.text(.5,.01,'Categories reflect available documentation; no collection time is inferred from feature names.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(fdir/'FIGURE4_FEATURE_TIMING.png',bbox_inches='tight');fig.savefig(fdir/'FIGURE4_FEATURE_TIMING.pdf',bbox_inches='tight');plt.close(fig)
print('tables and figures generated')

from pathlib import Path
import pandas as pd
r=Path('paper/phase7_3_manuscript/TABLES')
co=pd.read_csv(r/'TABLE1_COHORTS.csv'); c=co[['Dataset/centre','Records','Positive','Negative','Positive prevalence','Endpoint']]
(r/'TABLE1_COHORTS.md').write_text('# Table 1. Cohorts and endpoint definitions\n\n'+c.to_string(index=False)+'\n\n*Counts are source records/rows, not verified unique patients. ZAS local schema (56 columns including outcome) differs from UCI description (54 features); unresolved.*\n',encoding='utf-8')
fg=pd.read_csv(r/'TABLE2_FEATURE_GROUPS.csv')
(r/'TABLE2_FEATURE_GROUPS.md').write_text('# Table 2. ZAS feature groups, temporal classification and role\n\n'+fg.to_string(index=False)+'\n\n*The label “eligible combined” is a frozen feature-group name, not a statement that features are confirmed available before the diagnostic decision.*\n',encoding='utf-8')
p6=pd.read_csv(r/'TABLE3_PHASE6_PRIMARY.csv'); primary=p6[['Repeat','Rows each','Events','Prevalence','Selection ROC-AUC','V3-style ROC-AUC','Paired delta ROC-AUC']]
rep=pd.read_csv('results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv'); sec=[]
for _,x in rep.iterrows(): sec.append([int(x['repeat']),'Selection-aware' if x['model_id']=='selection_aware_procedure' else 'V3-style comparator',x['pr_auc_average_precision'],x['brier_score'],x['log_loss'],x['sensitivity'],x['specificity'],x['calibration_intercept'],x['calibration_slope']])
sd=pd.DataFrame(sec,columns=['Repeat','Procedure','Average precision','Brier','Log loss','Sensitivity','Specificity','Calibration intercept','Calibration slope'])
(r/'TABLE3_PHASE6_PRIMARY.md').write_text('# Table 3. Phase 6 primary ZAS results by complete-repeat out-of-fold set\n\n'+primary.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\nPaired mean AUC difference = 0.1740; descriptive split-variation t interval 0.1367–0.2113 (three overlapping repeat summaries; not patient/population uncertainty).\n\nSecondary performance and calibration summaries (each repeat includes all 303 source rows):\n\n'+sd.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\nAP reference prevalence = 0.7129 for each repeat. Calibration measures and thresholds are limited; thresholds are selected on training OOF data using Youden J and have no clinical meaning.\n',encoding='utf-8')
u=pd.read_csv(r/'TABLE4_UCI_STAGES.csv'); u=u[['UCI stage','Pooled ROC-AUC','Average precision','Brier','Log loss','Centre ROC-AUC minimum','Centre ROC-AUC maximum']]
(r/'TABLE4_UCI_STAGES.md').write_text('# Table 4. Separate UCI four-centre leave-one-centre-out stage results\n\n'+u.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\nPooled estimates use the 920-record four-centre cohort; centre ranges are minima/maxima across held-out centres. Stage 4 adds exercise-test features and represents a different later task.\n',encoding='utf-8')
i=pd.read_csv(r/'TABLE5_UCI_PAIRED.csv')
(r/'TABLE5_UCI_PAIRED.md').write_text('# Table 5. Paired UCI Stage 3-to-Stage 4 metric differences\n\n'+i.to_string(index=False,float_format=lambda x:f'{x:.5f}')+'\n\nIntervals are percentile intervals from 400 paired source-record bootstrap draws using the same sampled rows for both stages; they are not patient-cluster or centre-sampling intervals.\n',encoding='utf-8')

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
p=Path('paper/phase7_3_manuscript/FIGURES')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':180,'savefig.dpi':300})
fig,ax=plt.subplots(figsize=(10.2,6.3));ax.set_xlim(0,10);ax.set_ylim(0,7);ax.axis('off')
def bx(x,y,w,h,txt,fc):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.10',facecolor=fc,edgecolor='#334155',lw=1.1));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9,wrap=True)
def ar(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color='#475569',lw=1.1))
bx(.35,5.65,4.15,.85,'ZAS: 303 source rows; Cath endpoint\n216 Cad / 87 Normal','#e0f2fe')
bx(.35,4.15,4.15,.95,'Phase 6 nested selection\n6 groups × 5 unique model families\n3 inner folds; preprocessing within fit partitions','#dcfce7')
bx(.35,2.62,4.15,.95,'Matched outer evaluation\n3 repeats × 5 folds; complete-repeat OOF\nselection procedure vs V3-style comparator','#fef3c7')
bx(.35,.72,4.15,1.15,'ZAS interpretation\nRetrospective discrimination only\nTiming and patient identity unverified','#fee2e2')
ar(2.43,5.65,2.43,5.10);ar(2.43,4.15,2.43,3.57);ar(2.43,2.62,2.43,1.87)
bx(5.5,5.65,4.15,.85,'UCI Heart Disease: 920 records\nCleveland, Hungary, Switzerland, VA Long Beach','#f3e8ff')
bx(5.5,4.15,4.15,.95,'Separate staged logistic analyses\nLeave one centre out; tune in development centres\nStage 1–4 information sets','#e0e7ff')
bx(5.5,2.62,4.15,.95,'Four-centre results\nPooled and held-out-centre metrics\npaired record-level stage comparisons','#fef3c7')
bx(5.5,.72,4.15,1.15,'UCI interpretation\nSeparate multicentre evaluation\nNot external validation of ZAS','#fee2e2')
ar(7.58,5.65,7.58,5.10);ar(7.58,4.15,7.58,3.57);ar(7.58,2.62,7.58,1.87)
ax.set_title('Study workflow: independent evidence tracks',fontsize=13,pad=12)
fig.tight_layout();fig.savefig(p/'FIGURE1_STUDY_WORKFLOW.png',bbox_inches='tight');fig.savefig(p/'FIGURE1_STUDY_WORKFLOW.pdf',bbox_inches='tight');plt.close(fig)

from pathlib import Path
q=Path('paper/phase7_3_manuscript/TABLES/TABLE4_UCI_STAGES.md')
t=q.read_text(encoding='utf-8-sig')
for a,b in [('stage1_basic_risk','Stage 1 — basic risk'),('stage2_symptoms','Stage 2 — symptoms'),('stage3_resting_measurements','Stage 3 — resting measurements'),('stage4_exercise_test','Stage 4 — exercise test')]: t=t.replace(a,b)
q.write_text(t,encoding='utf-8')

from pathlib import Path
import pandas as pd
T=Path('paper/phase7_3_manuscript/TABLES')
def pipe(df):
 cols=[str(c) for c in df.columns]
 def cell(v):
  if pd.isna(v): return ''
  if isinstance(v,(int,float)):
   return f'{v:.4f}' if abs(v-round(v))>1e-9 else str(int(v))
  return str(v).replace('|','\\|').replace('\n','<br>')
 lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for _,row in df.iterrows(): lines.append('| '+' | '.join(cell(v) for v in row.tolist())+' |')
 return '\n'.join(lines)
co=pd.read_csv(T/'TABLE1_COHORTS.csv')[['Dataset/centre','Records','Positive','Negative','Positive prevalence','Endpoint']]
(T/'TABLE1_COHORTS.md').write_text('# Table 1. Cohorts and endpoint definitions\n\n'+pipe(co)+'\n\n*Counts are records/source rows; patient uniqueness is not verified. The local ZAS file has 56 columns including outcome, whereas UCI describes 54 features; the difference is unresolved.*\n',encoding='utf-8')
fg=pd.read_csv(T/'TABLE2_FEATURE_GROUPS.csv')
(T/'TABLE2_FEATURE_GROUPS.md').write_text('# Table 2. ZAS feature groups and timing classification\n\n'+pipe(fg)+'\n\n*“Eligible combined” is the candidate-group name from the historical project, not evidence of confirmed decision-time availability.*\n',encoding='utf-8')
p6=pd.read_csv(T/'TABLE3_PHASE6_PRIMARY.csv')
primary=p6[['Repeat','Rows each','Events','Prevalence','Selection ROC-AUC','V3-style ROC-AUC','Paired delta ROC-AUC']]
rep=pd.read_csv('results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv'); sec=[]
for _,x in rep.iterrows(): sec.append([int(x['repeat']),'Selection-aware' if x['model_id']=='selection_aware_procedure' else 'V3-style comparator',x['pr_auc_average_precision'],x['brier_score'],x['log_loss'],x['sensitivity'],x['specificity'],x['calibration_intercept'],x['calibration_slope']])
sd=pd.DataFrame(sec,columns=['Repeat','Procedure','Average precision','Brier','Log loss','Sensitivity','Specificity','Calibration intercept','Calibration slope'])
(T/'TABLE3_PHASE6_PRIMARY.md').write_text('# Table 3. Phase 6 primary ZAS results by complete-repeat out-of-fold set\n\n'+pipe(primary)+'\n\nPaired mean AUC difference = 0.1740; descriptive split-variation t interval 0.1367–0.2113 (three overlapping repeat summaries; not patient/population uncertainty).\n\nSecondary performance and calibration summaries (each repeat includes all 303 source rows):\n\n'+pipe(sd)+'\n\nAP reference prevalence = 0.7129 for each repeat. Calibration estimates are limited; thresholds were selected on training OOF predictions using Youden J and have no clinical interpretation.\n',encoding='utf-8')
u=pd.read_csv(T/'TABLE4_UCI_STAGES.csv');u=u[['UCI stage','Pooled ROC-AUC','Average precision','Brier','Log loss','Centre ROC-AUC minimum','Centre ROC-AUC maximum']]
for a,b in [('stage1_basic_risk','Stage 1 — basic risk'),('stage2_symptoms','Stage 2 — symptoms'),('stage3_resting_measurements','Stage 3 — resting measurements'),('stage4_exercise_test','Stage 4 — exercise test')]: u['UCI stage']=u['UCI stage'].str.replace(a,b,regex=False)
(T/'TABLE4_UCI_STAGES.md').write_text('# Table 4. Separate UCI four-centre leave-one-centre-out stage results\n\n'+pipe(u)+'\n\nPooled estimates use the 920-record four-centre cohort; centre ranges are minima/maxima across held-out centres. Stage 4 adds exercise-test features and is a distinct later task.\n',encoding='utf-8')
i=pd.read_csv(T/'TABLE5_UCI_PAIRED.csv')
(T/'TABLE5_UCI_PAIRED.md').write_text('# Table 5. Paired UCI Stage 3-to-Stage 4 metric differences\n\n'+pipe(i)+'\n\nIntervals are percentile intervals from 400 paired source-record bootstrap draws using the same sampled rows for both stages; they are not patient-cluster or centre-sampling intervals.\n',encoding='utf-8')

# Rebuild Table 6 from saved outer model selections only.
import pandas as pd
from pathlib import Path
root=Path.cwd(); out=Path('paper/phase7_3_manuscript/TABLES')
d=pd.read_csv(root/'results/phase6_zas_selection/OUTER_MODEL_SELECTIONS.csv')
d=d[d.model_id=='selection_aware_procedure']
rows=[]
for g in ['v3_five_predictor','basic_risk','symptoms_examination','resting_ecg','laboratory','eligible_combined']:
 n=int((d.selected_group==g).sum());rows.append(['Feature group',g,n,len(d),n/len(d)])
for m in ['logistic_l2','elastic_net','spline_logistic','random_forest','hist_gradient_boosting']:
 n=int((d.selected_model_family==m).sum());rows.append(['Model family',m,n,len(d),n/len(d)])
t=pd.DataFrame(rows,columns=['Selection dimension','Selected category','Outer fits selected','Outer fits total','Frequency'])
t.to_csv(out/'TABLE6_SELECTION_FREQUENCY.csv',index=False)
lines=['| '+' | '.join(t.columns)+' |','| --- | --- | ---: | ---: | ---: |']
for _,r in t.iterrows(): lines.append('| '+str(r['Selection dimension'])+' | '+str(r['Selected category'])+' | '+str(int(r['Outer fits selected']))+' | '+str(int(r['Outer fits total']))+' | '+f"{float(r['Frequency']):.4f}"+' |')
(out/'TABLE6_SELECTION_FREQUENCY.md').write_text('# Table 6. Phase 6 outer selection frequencies\n\n'+'\n'.join(lines)+'\n\n*Frequencies summarize 15 outer training fits. Feature-group selection does not identify individual feature necessity.*\n',encoding='utf-8')
