"""Adapt the user-supplied reference DOCX, retaining its six-section page system."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
import pandas as pd
from portable_release import discover_workspace, PROJECT, sha256

REFERENCE_SHA='a15fd5c3c51a476e2addf94e9387e07a879cece46f8507ec55d1c4425beb1734'
PRESERVE=['word/theme/theme1.xml','word/fontTable.xml','word/webSettings.xml',
          'word/numbering.xml','word/settings.xml','_rels/.rels']
ROLE={'title':0,'abstract':4,'keywords':5,'heading':6,'subheading':7,'body':8,
      'algorithm_title':60,'algorithm':62,'table_caption':84,'caption':87,'reference':100}


def clean(text):
    text=re.sub(r'\\cite\{([^}]+)\}',lambda m:'['+', '.join(x.lstrip('r') for x in m[1].split(','))+']',text)
    for _ in range(5):
        text=re.sub(r'\\(?:textbf|textit|emph|texttt|path|url|mathrm)\{([^{}]*)\}',r'\1',text)
    for a,b in [(r'\%','%'),(r'\_','_'),(r'\&','&'),(r'\geq','≥'),(r'\times','×'),('``','"'),("''",'"'),('$',''),('--','-')]:text=text.replace(a,b)
    return re.sub(r'\s+',' ',text).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path);args=parser.parse_args()
    if sha256(args.reference)!=REFERENCE_SHA:raise ValueError('Reference differs from the inspected template')
    root=discover_workspace();out=root/PROJECT/'paper/reference_format';out.mkdir(exist_ok=True)
    doc=Document(args.reference)
    paragraphs={name:deepcopy(doc.paragraphs[i]._p) for name,i in ROLE.items()}
    sections=[deepcopy(s._sectPr) for s in doc.sections]
    table_properties=deepcopy(doc.tables[0]._tbl.tblPr)
    for child in list(doc._element.body):doc._element.body.remove(child)
    for rid,rel in list(doc.part.rels.items()):
        if rel.reltype.endswith('/image') or rel.is_external:doc.part.drop_rel(rid)
    for style in doc.styles:
        if hasattr(style,'font'):
            style.font.name='Times New Roman';style.font.size=Pt(10)
    doc.styles['Title'].font.color.rgb=RGBColor(0,0,0)
    # Default fonts cannot silently fall back to the source's theme or Consolas.
    for fonts in doc.styles.element.xpath('.//w:rFonts'):
        for key in list(fonts.attrib):
            if key.endswith('Theme'):del fonts.attrib[key]
        for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Times New Roman')
    for size in doc.styles.element.xpath('.//w:sz | .//w:szCs'):size.set(qn('w:val'),'20')

    def font(run,bold=None,italic=None,color=None):
        run.font.name='Times New Roman';run.font.size=Pt(10)
        for key in ['ascii','hAnsi','eastAsia','cs']:run._r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:'+key),'Times New Roman')
        if bold is not None:run.bold=bold
        if italic is not None:run.italic=italic
        if color:run.font.color.rgb=RGBColor.from_string(color)
        return run

    def para(text='',role='body',bold=None,italic=None):
        element=deepcopy(paragraphs[role])
        for child in list(element):
            if child.tag!=qn('w:pPr'):element.remove(child)
        if element.pPr is not None:
            for child in list(element.pPr):
                if child.tag in [qn('w:sectPr'),qn('w:pBdr')]:element.pPr.remove(child)
        doc._element.body.append(element)
        from docx.text.paragraph import Paragraph
        p=Paragraph(element,doc._body)
        run=p.add_run(clean(text));font(run,bold,italic)
        if role=='title':p.style='Title';run.font.color.rgb=RGBColor(0,0,0)
        if role in ['heading','subheading','algorithm_title','table_caption']:p.paragraph_format.keep_with_next=True
        p.paragraph_format.widow_control=True
        return p

    def section_end(index):
        p=doc.paragraphs[-1]
        p._p.get_or_add_pPr().append(deepcopy(sections[index]))

    def image(name,caption):
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(0)
        p.add_run().add_picture(str(out/name),width=Inches((11906-1700)/1440))
        para(caption,'caption')

    def table(caption,headers,rows,widths):
        para(caption,'table_caption',bold=True)
        t=doc._body.add_table(rows=1,cols=len(headers),width=Inches(sum(widths)));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        t._tbl.remove(t._tbl.tblPr);t._tbl.insert(0,deepcopy(table_properties))
        t._tbl.tblPr.find(qn('w:tblW')).set(qn('w:w'),str(round(sum(widths)*1440)))
        for column,w in zip(t.columns,widths):column.width=Inches(w)
        for i,values in enumerate([headers,*rows]):
            cells=t.rows[0].cells if i==0 else t.add_row().cells
            for cell,value,w in zip(cells,values,widths):
                cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p=cell.paragraphs[0];p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(2)
                p.paragraph_format.keep_with_next=i<len(rows)
                font(p.add_run(str(value)),bold=i==0,color='FFFFFF' if i==0 else '000000')
                if i==0:
                    shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'1F3864');cell._tc.get_or_add_tcPr().append(shading)
            trpr=t.rows[i]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            if i==0:trpr.append(OxmlElement('w:tblHeader'))

    def equation(label,parts):
        p=para('',role='body');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        math=OxmlElement('m:oMath')
        def run(text):
            r=OxmlElement('m:r');pr=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'20');pr.append(sz);r.append(pr)
            node=OxmlElement('m:t');node.text=text;r.append(node);return r
        for part in parts:
            if isinstance(part,tuple):
                f=OxmlElement('m:f');a=OxmlElement('m:num');a.append(run(part[0]));b=OxmlElement('m:den');b.append(run(part[1]));f.extend([a,b]);math.append(f)
            else:math.append(run(part))
        p._p.append(math);font(p.add_run('  ('+str(label)+')'))

    authors=json.loads((root/'AUTHORS.json').read_text())['authors']
    source=(root/PROJECT/'paper/final/CARDIODIAG_FINAL_PAPER.tex').read_text(encoding='utf-8')
    title='CARDIODIAG Information Matched Evaluation and Probability Reliability in Retrospective Coronary Artery Disease Classification'
    para(title,'title',bold=True)
    author_table=doc._body.add_table(rows=1,cols=3,width=Inches((12240-1700)/1440));author_table.alignment=WD_TABLE_ALIGNMENT.CENTER;author_table.autofit=False
    for cell,a in zip(author_table.rows[0].cells,authors):
        cell.width=Inches((12240-1700)/1440/3)
        p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        lines=[a['name'],'Department of CSE','Prasad V. Potluri Siddhartha','Institute of Technology (PVPSIT)','Vijayawada India',a['email']]
        for i,line in enumerate(lines):
            r=p.add_run(line);font(r)
            if i<len(lines)-1:r.add_break()
    p=para('Project guide K. Naga Gopi','body');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    abstract=clean(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S)[1])
    p=para('',role='abstract');font(p.add_run('Abstract—'),True,True);font(p.add_run(abstract),False,False)
    p=para('',role='keywords');font(p.add_run('Index Terms—'),True,True)
    font(p.add_run('Coronary artery disease, information matched evaluation, nested validation, calibration, cohort transport, reproducibility.'),False,True)
    section_end(0)

    para('I. Introduction','heading',True)
    para('A. Problem Statement','subheading',True,True)
    para('A high coronary artery disease classification score does not establish that a complex learning procedure adds value beyond its predictors, that its probabilities are reliable, or that it will work in another cohort. Evaluation must include all model and preprocessing choices within training boundaries. When competing methods receive different information, an apparent algorithmic gain can instead reflect the available measurements.')
    para('B. Motivation','subheading',True,True)
    para('The original CardioDiag work combines explanation guided dimension reduction and optimized voting and stacking ensembles [1]. Those architectures and published results remain prior work. An earlier project comparison between rich feature selection and a five field reference on ZAS produced a large AUC gap, but also changed the information available to the procedures. This motivates a simple equal information control and calibration labels excluded from the entire base model search.')
    para('Retrospective angiographic classification is also distinct from early screening or future CAD risk. Predictor timing and unique patient linkage are unresolved in the public records used here. A compatibility constrained transfer experiment and explicit target updating can expose probability failure without making a prospective clinical claim.')
    para('C. Contributions','subheading',True,True)
    para('We contribute paired information matched controls, a calibration construction disjoint from complete selection, a source frozen shared core cohort test, and saved probability and split evidence. Follow-up studies examine calibration regularization, missing input robustness, target adaptation, sensitivity focused thresholds, learning curves and input stability. Positive and negative results are retained; no new classifier architecture or superiority to the original CardioDiag ensembles is claimed.')
    para('D. Paper Structure','subheading',True,True)
    para('Section II reviews related work. Section III defines the data and evaluation methodology; Section IV gives the training algorithm. Section V presents quantitative results, discussion and threats to validity. Section VI concludes. The supplementary methods retain detailed provenance and executable traceability.')
    para('II. Literature Survey','heading',True)
    para('Finite sample selection criteria can overfit when many configurations are compared [2]. Nested assessment includes specified choices inside training but does not undo earlier researcher exposure to a cohort. Probability calibration and ranking are different properties [5]. TRIPOD+AI and PROBAST+AI support transparent reporting and appraisal rather than certifying clinical validity [6, 7]. External evaluation requires compatible predictors, outcomes and populations, with precision driven sample size planning [8, 9].')
    para('Existing ZAS studies already compare machine and deep learning methods [11], combine feature selection and resampling with Bayesian SVM optimization [12], and use SHAP guided wrapper selection across public resources [13]. AngioCAD also has its own provider framework [14]. Their scores are not a head-to-head leaderboard because information sets, endpoints and validation rules differ. Our research object is the complete evaluation workflow, not another optimization or explanation module.')
    para('Four questions guide the analysis: whether rich feature search adds value over an equal information simple model; how calibration changes loss when labels are isolated from selection; how missing or perturbed inputs affect results; and how frozen source scores transport compared with later label assisted updating. All questions remain retrospectively motivated on previously examined records.')
    section_end(1)
    image('workflow.png','Fig. 1. Source calibration and five core transport are separate model tracks. Development D, calibration C and outer test records are disjoint; target updating uses other target folds only.')
    section_end(2)

    para('III. Methodology','heading',True)
    para('A. Data and Endpoints','subheading',True,True)
    para('Z-Alizadeh Sani contains 303 source rows with 216 Cad and 87 Normal labels, prevalence 0.7129 [3]. The validated table agrees with the provider workbook on 55 non-target fields plus Cath, despite descriptive prose mentioning 54 predictors. No exact full record duplicates were found, but row keys are not verified person identifiers. We evaluate retrospective discrimination rather than prospective screening.')
    para('AngioCAD has 413 label IDs, of which 377 match clinical feature records [4, 14]. The derived positive endpoint requires any of 15 segments to have a category above 50% stenosis: 298 positive and 79 non-positive. Ten non-positive records have a 26–50% category. This cannot recover exact equality at the source catalogue threshold of at least 50%. Excluding those ten cases or assigning them positive are sensitivity definitions, not recovered ground truth.')
    para('The matched target diagnoses comprise 186 STEMI, 20 NSTEMI, 55 unstable angina and 116 stable angina records. Non-positive stenosis is not a healthy population control. Six target ages and 46 smoking values are missing; 36 label-only records lack clinical features. Sampling, measurement timing and binary coding equivalence remain limitations.')
    para('B. Predictors and Matched Controls','subheading',True,True)
    para('The rich eligible source group uses 38 fields: seven risk, ten symptom and examination, seven resting ECG and fourteen laboratory variables. Cath, direct angiography, EF-TTE, Region RWMA and VHD are excluded. Their exclusion does not establish pre-decision timing for retained fields. The historical V3 reference uses Age, Sex, current and former smoking, and family history. The separately fitted transfer model uses Age, Sex, DM, HTN and current smoking; it is neither V3 nor a transported 38-field model.')
    para('We retain three repeats of five source outer folds, with three-fold inner AUC selection across six groups and five families: L2, elastic-net and spline logistic, random forest and histogram gradient boosting. This gives 30 candidates and 120 parameter settings. All transforms are fitted within their relevant training data. Every procedure supplies 303 out-of-fold predictions per repeat; overlapping repeats are not 909 independent patients.')
    para('The equal-feature control uses ordinary L2 logistic regression with C in {0.1, 1, 10} on all 38 fields. Additional controls remove ten symptom/examination fields or retain only 17 risk and symptom fields. Historical post-selection sigmoid and refit conventions are preserved for matching. Their calibration labels influence configuration selection, so they are not independently calibrated controls.')
    para('C. Selection Independent Calibration','subheading',True,True)
    para('For each outer training set and each of three components, let C be a calibration block and D its complementary development records. The complete 30-candidate search is rerun using D only. The selected base model fitted on D scores C, whose labels fit the probability map. That exact base and map score the outer test; a larger full-training refit is not substituted. Averaging the three component probabilities gives the ensemble. An identical-component raw mean isolates the calibration operation.')
    para('The initial weak sigmoid fits raw probabilities with C = 10^6. A frozen follow-up reuses all 45 components to compare a logit intercept offset, fixed-C1 logit sigmoid and a logit sigmoid whose C is selected from {0.1, 1, 10} by calibration-block inner log loss. Outer test labels do not select the map. Equation (1) describes the logit map; the intercept-only variant fixes b = 1.')
    equation(1,['f(s) = σ(a + b logit(s))'])
    para('D. Source Frozen Transfer and Target Updating','subheading',True,True)
    para('The five-core source model uses training-fold age median imputation and scaling, binary mode imputation and unbalanced logistic C tuning. A separate source calibration ensemble wraps the whole search in disjoint calibration folds. Both full-source procedures freeze target probabilities before scoring target outcomes. They differ in training size and averaging as well as calibration, so their target contrast is not a pure calibration effect.')
    para('Later target updating uses three repeats of five-fold adaptation/evaluation on previously frozen source scores. Other target folds fit a prevalence constant, offset or sigmoid; each evaluation record is excluded from its map. This is retrospective label-assisted target adaptation, not untouched independent validation or evidence from a future hospital.')
    para('E. Robustness Metrics and Uncertainty','subheading',True,True)
    para('Frozen source models are evaluated after 0%, 10%, 25% and 50% held-out random cell deletion. An optional estimator adds one 20%-masked training copy, half weighting original and copied rows and fitting indicators within each estimator fit. Copies never cross into validation. Typed encoding, whole-feature deletion and all negative comparisons remain recorded.')
    para('Complete-repeat AUC, average precision, Brier and log loss summarize discrimination and probability accuracy. Brier is squared probability error, not a skill score. Joint calibration intercept and slope are distinct from offset-only updating. Conditional record bootstraps use 2,000 target draws or 1,000 paired source draws shared across repeats. Fifty typed-L2 out-of-bag refits and fifty separate five-core coefficient refits have different scopes; neither is full-search or patient-cluster uncertainty.')
    equation(2,['Brier = ',('1','n'),' Σᵢ (pᵢ − yᵢ)²'])
    para('Final threshold development targets 90% sensitivity using nested training-only out-of-fold scores. Source learning curves retune C within training fractions; target learning curves use disjoint adaptation fractions. Decision curves retain a fixed 0.05–0.95 research grid with all and none strategies [10]. Equation (3) defines net benefit, but no intended action or clinical harm ratio was established.')
    equation(3,['NB = ',('TP','n'),' − ',('FP','n'),' × ',('t','1 − t')])
    para('IV. Algorithm','heading',True)
    para('Algorithm 1  Selection independent calibration','algorithm_title',True)
    steps=['Input: fixed outer splits and candidate grid.',
           '1: For each outer training and test partition:',
           '2: Form three D and C splits within training.',
           '3: Search all candidates using D labels only.',
           '4: Fit the selected base on D and score C.',
           '5: Fit the prespecified map on C scores and labels.',
           '6: Score the untouched test using that base and map.',
           '7: Average mapped and matched raw components.',
           'Output: one held-out score per source row per repeat.']
    for i,step in enumerate(steps):
        p=para(step,'algorithm');p.paragraph_format.keep_with_next=i<len(steps)-1
    section_end(3)

    trace=pd.read_csv(root/PROJECT/'paper/final/NUMERIC_TRACEABILITY.csv')
    means=trace.groupby(['analysis','model_id','condition','metric']).value.mean()
    def value(analysis,model,metric):return float(means.loc[(analysis,model,'{}',metric)])
    para('V. Results and Discussion','heading',True)
    para('A. Information Matched Source Comparisons','subheading',True,True)
    para('All results are complete-repeat metrics from saved held-out probabilities. Table I retains eight original source procedures. Historical selection averaged AUC 0.9103 versus 0.9174 for the equal-feature logistic control; selection-minus-control AUC was negative in all three repeats. This supports an information-availability explanation of the earlier large gain over V3, rather than a necessary complexity advantage in this workflow. Historical calibration limitations still apply.')
    models=[('historical_v3_balanced','Historical balanced V3'),('l2_v3_ordinary_historical_calibration','Ordinary L2 on V3 fields'),
            ('l2_risk_symptoms17_historical_calibration','L2 risk and symptoms 17'),('l2_no_symptoms28_historical_calibration','L2 without symptoms 28'),
            ('l2_full38_historical_calibration','Equal-feature L2 38'),('historical_selection','Historical selection'),
            ('same_components_uncalibrated_ensemble','Identical raw components'),('selection_independent_calibrated_ensemble','Corrected independent calibration')]
    rows=[[label,*[f'{value("source",model,m):.4f}' for m in ['roc_auc','ap','brier','log_loss']]] for model,label in models]
    table('TABLE I. COMPLETE REPEAT SOURCE METRICS', ['Procedure','AUC','AP','Brier','Log loss'],rows,[3.1,.95,.95,.95,.95])
    para('B. Probability Reliability and Cohort Transport','subheading',True,True)
    para('Fixed-C1 logit calibration improved AUC/Brier from 0.9096/0.1037 for the previous independent sigmoid to 0.9144/0.1029, while the identical raw components retained higher AUC 0.9164. The conditional source-record Brier difference interval against raw spans zero, approximately [−0.005329, 0.000685]. Thus the proper-loss improvement is descriptive, not established population superiority.')
    cal=[('raw','Identical raw ensemble'),('weak_v1','Previous independent sigmoid'),('offset','Logit intercept offset'),('logit_c1','Fixed-C1 logit sigmoid'),('logit_cv','Calibration-block CV sigmoid')]
    rows=[[label,*[f'{value("calibration",model,m):.4f}' for m in ['roc_auc','brier','log_loss']]] for model,label in cal]
    table('TABLE II. SAME COMPONENT CALIBRATION FOLLOW UP',['Mapping','AUC','Brier','Log loss'],rows,[3.5,1.1,1.1,1.1])
    para('The five-core source model averaged source AUC 0.8059 and achieved frozen AngioCAD AUC 0.7125, with conditional record-bootstrap interval 0.6474–0.7730. Target Brier was 0.1982; even a post hoc source-prevalence constant scored 0.1717. Boundary exclusion gave raw AUC 0.7293, and upper assignment gave 0.7259. These sensitivities do not resolve endpoint equivalence or identify a causal hospital-shift effect.')
    target=[('frozen_raw','Frozen source raw'),('frozen_source_calibrated','Frozen source calibrated'),('adaptation_prevalence','Target-fold prevalence'),('target_offset','Target intercept update'),('target_logit_c1','Target fixed-C1 update'),('target_logit_cv','Target CV sigmoid')]
    rows=[[label,*[f'{value("adaptation",model,m):.4f}' for m in ['roc_auc','brier','log_loss']]] for model,label in target]
    table('TABLE III. HELD OUT TARGET ADAPTATION METRICS',['Procedure','AUC','Brier','Log loss'],rows,[3.5,1.1,1.1,1.1])
    para('Target intercept updating reduced held-out Brier to 0.1507; target C1 improved loss further but slightly lowered pooled AUC. Fold-specific monotonic maps can change pooled out-of-fold ordering. Target labels were used only in the separate adaptation folds; this remains an internal evaluation of updating on an already examined cohort.')
    image('roc.png','Fig. 2. Repeat-one source calibration and target updating ROC curves. The raw and mapped source ensembles share fitted components. Target updates use disjoint adaptation labels.')
    para('C. Robustness Learning and Stability','subheading',True,True)
    para('At 50% random deletion, ordinary-core AUC fell to 0.6875. Training augmentation improved it to 0.7033 but reduced complete-input AUC from 0.8059 to 0.8045 and worsened some whole-block scenarios. Typed rich-feature encoding also regressed (AUC/Brier 0.8992/0.1129). Neither candidate replaced the released default.')
    para('A training-only 90%-sensitivity target produced held-out sensitivities 0.8796, 0.8889 and 0.8981, mean specificity 0.4904, PPV 0.8126 and NPV 0.6400. The threshold target was not guaranteed on new records; predictive values reflect source enrichment. Learning curves increased source AUC from 0.7690 at 25% training size to 0.8059 at 100%. Target-offset Brier was 0.1525 at 10% adaptation size and 0.1507 at 100%; one sample per size/fold cannot determine a required clinical sample size.')
    image('learning.png','Fig. 3. Source development and target adaptation learning curves. Points are mean complete-repeat scores; fractions use only training or adaptation records, with evaluation records held out.')
    para('Age, DM, HTN and smoking coefficient signs were positive in all fifty five-core bootstraps; the sex coefficient was positive in 98%. These signs are not causal effects or feature necessity. Five-percent binary-code flips reduced AUC to 0.7828 and increased Brier to 0.1613. Decision curves in Fig. 4 remain exploratory: no clinical action or acceptable harm ratio was specified.')
    image('decision.png','Fig. 4. Exploratory complete-repeat decision curves. All and none are reference strategies. The visible net-benefit range is limited to −0.1 through 0.85; full values remain in CSV. No clinical threshold is recommended.')
    section_end(4)

    para('D. Discussion','subheading',True,True)
    para('Three findings define the contribution. First, equal information reduced the evidence for a complex-search advantage over a simpler logistic procedure. This is not a universal model-family ranking: the source cohort is small, previously examined and assessed with deliberately retained historical calibration rules. Second, separating calibration labels from all selection choices removes one avoidable dependence but does not ensure precise small-block maps. Ranking and probability loss show different trade-offs. Third, source-only transport and labeled target updating require different evidence and must not be presented as interchangeable external validation.')
    para('The project contributes executable traceability of predictions, memberships and probability construction. The supplement supplies exact paths and the paper verifier reconstructs 732 metric values across 180 prediction groups without retraining. Windows and Linux checks validate the recorded artifacts; they are not an independent investigator replication, prospective study or clinical workflow evaluation. A matched OVEC/OSEC experiment would address architectural superiority, while a new compatible timing-verified cohort would address stronger clinical transport claims. Neither is claimed as completed.')
    para('E. Threats to Validity','subheading',True,True)
    para('ZAS has 303 records and substantial prior research exposure. Unique patients, acquisition times and repeated visits cannot be verified. Clinical spectrum, outcome thresholds, missing target clinical records and incomplete binary ascertainment constrain transfer. Retained historical controls have selection-dependent calibration and fit-size mismatch. Independent calibration blocks remain small; record-bootstrap intervals condition on saved models and unverified independence. Typed-L2 and five-core refit studies are not uncertainty for the full 30-candidate search.')
    para('Simulated cell deletion and coding perturbations are numerical scenarios, not measured clinical missingness or error mechanisms. Age and sex subgroup reports flag small classes but do not establish fairness. Target adaptation reuses a previously examined cohort. Thresholds, learning curves and decision curves do not establish screening utility, patient-specific reliable risk or deployment readiness. Source data and code permissions and institution-specific declarations still require author review before submission.')
    para('VI. Conclusion','heading',True)
    para('Within the evaluated retrospective workflow, information-matched logistic regression exceeded historical selection. Regularized calibration improved the previous independent-calibration ensemble, while labeled target updating lowered probability loss on AngioCAD. Typed encoding and complete-input augmentation regressed and were not promoted. Learning, threshold and perturbation analyses expose remaining sample-size and input-quality sensitivity. The defensible result is a reproducible empirical reliability study, not a new universally superior diagnostic algorithm. Independent patient-level, timing-verified clinical evaluation remains necessary for stronger intended-use claims.')
    para('Data Code and Author Statements','heading',True)
    para('Code, saved predictions, protocols and supplementary evidence are available at https://github.com/akhil-preetham/Cardio-Diag-Updated. Public source and target providers are cited in [3, 4, 14]. No prospective recruitment or clinical deployment was performed. Ethics determination, funding, conflicts, contribution roles and corresponding author require author confirmation. OpenAI ChatGPT/Codex assisted code and prose preparation; authors remain responsible for scientific accuracy and the selected venue disclosure requirements.')
    para('References','heading',True)
    for key,entry in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',source,re.S):
        para('['+key.lstrip('r')+'] '+clean(entry),'reference')
    doc._element.body.append(deepcopy(sections[5]))
    doc.core_properties.title=title;doc.core_properties.author='; '.join(a['name'] for a in authors)
    doc.core_properties.last_modified_by='CARDIODIAG project authors'
    doc.core_properties.subject='Retrospective coronary artery disease evaluation'
    doc.core_properties.keywords='CARDIODIAG; calibration; retrospective evaluation'
    doc.core_properties.comments='';doc.core_properties.created=datetime(2026,11,15,tzinfo=timezone.utc)
    doc.core_properties.modified=datetime(2026,11,15,tzinfo=timezone.utc)
    path=out/'CARDIODIAG_CONFERENCE_PAPER.docx';doc.save(path)
    # Restore opaque source parts byte-for-byte, but scrub stale extended metadata.
    import io
    from lxml import etree
    buf=io.BytesIO()
    with zipfile.ZipFile(args.reference) as original,zipfile.ZipFile(path) as built,zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for item in built.infolist():
            data=original.read(item.filename) if item.filename in PRESERVE else built.read(item.filename)
            if item.filename=='docProps/app.xml':
                xml=etree.fromstring(data)
                for child in list(xml):
                    if etree.QName(child).localname in ['Company','Manager','Pages','Words','Characters','Lines','Paragraphs','CharactersWithSpaces']:xml.remove(child)
                data=etree.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True)
            z.writestr(item,data)
    path.write_bytes(buf.getvalue())
    if sha256(args.reference)!=REFERENCE_SHA:raise ValueError('Reference file changed')
    record={'reference_sha256':REFERENCE_SHA,'reference_not_redistributed':True,'source_sections':6,
            'source_section_geometry_preserved':True,'column_sequence':[1,2,1,2,1,2],
            'intentional_deviations':['All text uses user-required Times New Roman 10 point','Three own authors replace four source authors','Own research and images replace every source content slot','Editable native Word equations replace source equation images','Title is black Word Title style','Author metadata scrubbed; no source media retained'],
            'preserved_parts':{n:hashlib.sha256(zipfile.ZipFile(args.reference).read(n)).hexdigest() for n in PRESERVE},
            'inputs':{p.relative_to(root).as_posix():sha256(p) for p in [Path(__file__),root/'tools/plot_conference_paper.py',root/'AUTHORS.json',root/PROJECT/'paper/final/CARDIODIAG_FINAL_PAPER.tex',root/PROJECT/'paper/final/NUMERIC_TRACEABILITY.csv',
                root/PROJECT/'results/research_extension_v2/CALIBRATION_PREDICTIONS.csv',root/PROJECT/'results/research_extension_v2/ADAPTATION_PREDICTIONS.csv',
                root/PROJECT/'results/final_analyses_v3/SOURCE_LEARNING_SUMMARY.csv',root/PROJECT/'results/final_analyses_v3/ADAPTATION_LEARNING_SUMMARY.csv',root/PROJECT/'results/final_analyses_v3/DECISION_CURVES.csv',
                out/'workflow.png',out/'roc.png',out/'learning.png',out/'decision.png']}}
    (out/'BUILD_PROVENANCE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
