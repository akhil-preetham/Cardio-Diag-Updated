"""Produce the final 10-point Times New Roman paper; prior papers preserved."""
import os
from pathlib import Path
import re
import shutil
import pandas as pd
from portable_release import discover_workspace, PROJECT
from paper_editorial import enhance


def main():
    root=discover_workspace();out=root/PROJECT/'paper/final';out.mkdir(exist_ok=True)
    text=(root/PROJECT/'paper/research_extension_v2/CARDIODIAG_RESEARCH_REPORT_V2.tex').read_text(encoding='utf-8')
    fonts=Path(os.environ.get('CARDIODIAG_TIMES_DIR','C:/Windows/Fonts')).resolve()
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        if not (fonts/name).is_file(): raise FileNotFoundError('Set CARDIODIAG_TIMES_DIR to licensed Times New Roman fonts')
    # XeTeX on Windows needs relative font paths. These local licensed font
    # copies are excluded from Git; only permitted PDF font subsets are shared.
    local_fonts=out/'fonts';local_fonts.mkdir(exist_ok=True)
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        shutil.copyfile(fonts/name,local_fonts/name)
    text=text.replace(r'\documentclass[conference]{IEEEtran}',r'\documentclass[conference,10pt]{IEEEtran}')
    text=text.replace(r'\usepackage{graphicx,booktabs,array,url}',r'\usepackage{graphicx,booktabs,array,url}'+'\n'+r'\urlstyle{same}')
    text=text.replace(r'\usepackage[T1]{fontenc}',r'''\usepackage{fontspec}
\setmainfont[Path={fonts/},BoldFont=timesbd.ttf,ItalicFont=timesi.ttf,BoldItalicFont=timesbi.ttf]{times.ttf}
\makeatletter
\renewcommand\normalsize{\@setfontsize\normalsize{10}{12}}
\makeatother
\let\small\normalsize\let\footnotesize\normalsize\let\scriptsize\normalsize
\let\large\normalsize\let\Large\normalsize\let\LARGE\normalsize
\let\huge\normalsize\let\Huge\normalsize
\setlength{\tabcolsep}{3pt}
\normalsize'''.replace('FONTDIR',fonts.as_posix()))
    title=re.search(r'\\title\{([^}]+)\}',text).group(1)
    titleblock=r'''\twocolumn[\begin{center}\normalsize\textbf{TITLE}\par\vspace{6pt}
Erukudinnela Akhil Preetham, Chunduri Hari Krishna, G Vennela\\
Department of Computer Science and Engineering\\
Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT), Vijayawada, Andhra Pradesh, India\\
Project guide: K. Naga Gopi\end{center}\vspace{8pt}]'''.replace('TITLE',title)
    text=text.replace(r'\maketitle',titleblock)
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'''\begin{abstract}
High cardiovascular classification scores do not establish fair model comparisons, calibrated probabilities or cross-cohort reliability. We extend a CardioDiag-derived project through information-matched controls, selection-independent calibration and compatibility-constrained public-cohort evaluation. Z-Alizadeh Sani contains 303 source rows, including 216 CAD labels. Fifteen paired outer partitions compare historical selection with a 38-feature logistic control and independently calibrated ensembles. Mean source ROC-AUC was 0.9103 for selection and 0.9174 for equal-feature logistic regression. Fixed-C1 logit calibration improved the existing independent-calibration ensemble from AUC/Brier 0.9096/0.1037 to 0.9144/0.1029, while retaining lower AUC than identical uncalibrated components. A separate five-core model achieved source AUC 0.8059 and frozen-target AUC 0.7125 on 377 matched AngioCAD cases. Retrospective target-offset adaptation reduced held-out Brier from 0.1982 to 0.1507. Missingness augmentation improved performance under heavy random deletion but slightly degraded complete-input performance; typed encoding also regressed and was not promoted. A training-only 90\%-sensitivity threshold yielded held-out sensitivity 0.8889 and specificity 0.4904. Learning curves, coefficient bootstraps, input perturbations and exploratory decision curves quantify remaining sensitivities. Predictions, partitions, maps, hashes and tests preserve both positive and negative results. Previously examined cohorts, unresolved predictor timing, unverified patient independence and endpoint/coding differences preclude prospective or clinical-deployment claims. The contribution is an auditable retrospective reliability study rather than a universally superior diagnostic algorithm.
\end{abstract}''',text,flags=re.S)
    text=re.sub(r'\\section\{Introduction\}.*?(?=\\section\{Related Work)',lambda _:r'''\section{Introduction}
High cardiovascular classification scores do not establish fair model comparison, calibrated probabilities or transport to another clinical setting. A defensible extension must distinguish information availability from algorithmic gains and evaluate selection, preprocessing and calibration inside the training procedure.

The original CardioDiag approach integrates preprocessing, shadow sampling, explanation-guided dimensionality reduction and optimized ensembles \cite{r1}. It already includes tuning, explanations, calibration and ablations. Our contribution is a retrospective empirical reliability study: information-matched simple controls, selection-independent calibration, compatibility-constrained public-cohort transport and prespecified missing-input stress tests. Follow-up analyses examine regularized calibration, target adaptation, sensitivity-focused thresholds, decision curves, learning curves and stability. All outcomes, including regressions, are preserved. No clinical deployment or algorithmic-priority claim is made.

''',text,flags=re.S)
    text=re.sub(r'\\section\{Related Work and Research Questions\}.*?(?=\\section\{Data,)',lambda _:r'''\section{Related Work and Research Questions}
Finite-sample model selection can overfit evaluation criteria \cite{r2}; nested evaluation does not erase earlier researcher adaptation to a dataset. Calibration and discrimination measure different properties \cite{r5}. External evaluation requires compatible predictor definitions, endpoints and target populations \cite{r8}, with precision-driven sample-size planning \cite{r9}. TRIPOD+AI and PROBAST+AI inform transparent reporting and risk-of-bias appraisal \cite{r6,r7}.

We ask whether richer information explains the historical selection gain, whether selection-independent calibration improves probability loss, how a shared-core model transports, and whether prespecified updating or robustness methods address measured failures. Previously examined cohorts and reused partitions make the work retrospective, not independent confirmation.

''',text,flags=re.S)
    text=re.sub(r'\\section\{Conclusion and Project Disposition\}.*?(?=\\section\*\{Author,)', '',text,flags=re.S)
    text=re.sub(r'\\subsection\{Source-Only Shared-Core Development and Transport\}.*?(?=\\subsection\{Missingness)',lambda _:r'''\subsection{Source-Only Shared-Core Development and Transport}
The five-field source-only logistic model uses training-fold age median imputation/scaling and binary mode imputation. Unbalanced logistic C in 0.1, 1 and 10 is selected by three-fold source-training AUC. The source-calibrated arm wraps the entire search in a three-fold calibration ensemble, keeping calibration labels outside model selection. All source models, mappings and target predictions are frozen before target evaluation. Target outcomes never enter this original transport fit. Ten boundary cases receive excluded/upper-assignment scoring sensitivities, without redefining the source outcome.

''',text,flags=re.S)
    text=re.sub(r'\\subsection\{Missingness Stress and Statistical Summaries\}.*?(?=\\section\{Results\})',lambda _:r'''\subsection{Missingness Stress and Statistical Summaries}
The original five-core stress test masks held-out source inputs at fixed rates 0, 10, 25 and 50\%, without refitting. Saved masks are reused for matched follow-up comparisons. Complete-repeat ROC-AUC, average precision, Brier and log loss summarize predictions; repeat ranges are descriptive because partitions overlap. Original target record-bootstrap intervals use 2,000 resamples conditional on the fitted model and unverified record independence. They are not patient-cluster or refitting intervals. No target labels select a source model or threshold.

''',text,flags=re.S)
    text=re.sub(r'\\section\*\{Author, Data-Use and Submission Statements\}.*?(?=\\setcounter\{table\})',lambda _:r'''\section*{Declarations}
The listed authors and institution were supplied by the project team. No ethics approval or exemption is asserted. Funding, conflicts, contributions and corresponding-author designation require author confirmation. Original methods and datasets remain attributed to their providers; AngioCAD is CC BY 4.0. AI-assisted coding and drafting were used. Authors must verify the content and venue disclosure requirements before submission.
Code, predictions and audit artifacts: \url{https://github.com/akhil-preetham/Cardio-Diag-Updated}.

''',text,flags=re.S)
    text=re.sub(r'\\noindent\\textit\{Integrated report, version 2.*?\\par',
                lambda _:r'\noindent\textit{Final retrospective research report. Times New Roman, 10 point. Historical results and all negative comparisons are retained.}\par',text)
    # Plot source remains available; replace its float with final study overview.
    text=re.sub(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', '',text,flags=re.S)
    text=re.sub(r'\\subsection\{Uncertainty, Subgroups and Integration Decisions\}.*?(?=\\begin\{thebibliography\})',lambda _:r'''\subsection{Uncertainty, Subgroups and Integration Decisions}
One thousand paired source-record bootstrap draws share resampled IDs across repeats; intervals condition on saved fits and unverified row independence. Fifty separate typed-L2 development bootstraps use grouped duplicate copies for C tuning and out-of-bag evaluation. These are not patient-level intervals for the full selection procedure. Prespecified sex and age groups receive repeat-one metrics and class counts; groups with fewer than ten cases in either class are flagged unstable, without fairness certification.

A descriptive gate requires nondecreased mean AUC and nonincreased mean Brier versus the stated comparator. Fixed-C1 and inner-CV source calibration pass against existing corrected calibration; target offset passes against frozen raw scores. Typed encoding and clean-input augmentation fail. No variant automatically replaces a released model, and passing does not establish significance. The version-two verifier reconstructed 84 repeat-metric groups and reproduced all 909 ordinary-core baseline predictions. Original hashes remain valid. The additional final analyses below retain these boundaries.

''',text,flags=re.S)
    result=root/PROJECT/'results/final_analyses_v3'
    sensitivity=pd.read_csv(result/'SENSITIVITY_REPEAT_METRICS.csv').mean(numeric_only=True)
    learning=pd.read_csv(result/'SOURCE_LEARNING_SUMMARY.csv').set_index('fraction')
    target=pd.read_csv(result/'ADAPTATION_LEARNING_SUMMARY.csv')
    small=target[(target.fraction==.1)&(target.model_id=='target_offset')].iloc[0]
    perturb=pd.read_csv(result/'PERTURBATION_SUMMARY.csv').set_index('scenario')
    addition=r'''
\section{Final Decision and Stability Analyses}
These analyses were frozen before execution as a further retrospective extension on the same examined records. No new model family, feature group or clinical cohort was introduced. All defaults remain unchanged. The final-analysis code and input hashes are retained separately from the original and version-two experiments.

\subsection{Sensitivity-Focused Thresholds and Decision Curves}
A sensitivity target of 90\% was prespecified for the five-core source model. Within each outer training set, three threshold-development folds generate out-of-fold probabilities; every probability comes from a complete three-fold C search fitted without its threshold-validation records. The largest threshold attaining at least 90\% sensitivity on these training-only probabilities is then applied to the unchanged outer-test probabilities. Achieving the training target does not guarantee the same sensitivity on new records.

Held-out repeat sensitivities were 0.879630, 0.888889 and 0.898148. Mean sensitivity was SENS and specificity SPEC; mean positive/negative predictive values were PPV/NPV. These predictive values reflect the enriched source prevalence, not community screening prevalence. The threshold study quantifies missed cases and false positives but does not establish a clinical screening policy.

Decision curves use the fixed threshold grid 0.05--0.95 in steps of 0.05 and the frozen complete-repeat source and target-adaptation predictions. Net benefit is the true-positive fraction minus the false-positive fraction multiplied by threshold odds, $NB=TP/n-(FP/n)\,t/(1-t)$, following Vickers and Elkin \cite{r10}. Investigate-all and investigate-none are explicit comparators. No threshold was chosen using these curves. The curves are exploratory because no clinical action, elicited harm ratio or intended-use population was established. Apparent positive benefit in these cohorts is not proof of clinical effectiveness.

\subsection{Development and Adaptation Learning Curves}
The five-core source model was fitted using stratified 25\%, 50\%, 75\% and 100\% subsets of each outer training set, keeping outer test records unchanged and repeating C tuning within the sampled development subset. Mean AUC increased from AUC25 at 25\% to AUC100 at 100\%; Brier decreased from BRIER25 to BRIER100. This indicates continued sample-size sensitivity within the observed range, not an extrapolated guarantee about larger future datasets.

Target intercept updating used stratified 10\%, 25\%, 50\%, 75\% and 100\% subsets of the existing adaptation folds, each disjoint from its evaluation fold. Mean Brier was SMALL at 10\% (approximately 30 labeled adaptation records per fold) and 0.150723 at 100\% (approximately 301--302). The prevalence-only comparator and every sampled case membership are retained. The target adaptation size result comes from one deterministic stratified sample per size/fold; it is not a sample-size sufficiency rule or independent confirmation.

\subsection{Coefficient and Input Stability}
Fifty five-core development bootstraps reran logistic C tuning using stratified group folds so duplicate source-row copies could not appear in both fitting and validation. Age, diabetes, hypertension and smoking coefficients were positive in all refits; the sex coefficient was positive in 98\%. Age coefficients refer to a training-standardized age variable, while the other fields use their recorded binary codes. Sign consistency is neither causal evidence nor stability of the 38-field selection procedure.

Prespecified held-out perturbations included age shifts of plus/minus five years, rounding age to five-year increments and independent 5\% binary code flips. Age rounding retained mean AUC ROUND, while binary flips reduced AUC to FLIP and increased Brier to FLIPBRIER. These are numerical error scenarios, not empirically measured clinical error distributions. They show input-quality sensitivity without asserting robustness to all coding or measurement errors.

The final verifier reconstructs 57 complete-repeat metric groups and 627 decision-curve points, checks threshold/adaptation separation and reproduces the full-size learning-curve baseline to tolerance $10^{-12}$. Together with the original and version-two checks, these artifacts provide executable research traceability. No new compatible cohort, verified patient grouping, predictor timing, ethics determination or prospective clinical validation is claimed. Those evidence limitations remain explicit rather than being replaced by additional algorithm searches.

\section{Conclusion}
Information-matched logistic regression outperformed the historical selection procedure. Regularized calibration improved the existing independently calibrated ensemble, while target updating reduced probability loss in a retrospective adaptation study. Typed encoding and clean-input augmentation regressed and were not promoted. Learning, threshold and perturbation analyses quantify remaining sample-size and input-quality sensitivity. The code and retrospective evidence are finalized within this scope; genuinely independent, timing-verified clinical evaluation remains future work.

\begin{figure*}[p]
\centering\includegraphics[width=\textwidth]{FINAL_ANALYSES.pdf}
\caption{Final retrospective study overview. Top: repeat-one ROC curves. Middle: development-size and target-adaptation learning curves, averaged across complete repeats. Bottom: exploratory decision curves; the displayed net-benefit axis is limited to $[-0.1,0.85]$ for readability, while all values are retained in CSV. No clinical threshold is recommended.}
\end{figure*}
'''
    replacements={'SENS':f'{sensitivity.sensitivity:.6f}','SPEC':f'{sensitivity.specificity:.6f}',
        'PPV':f'{sensitivity.ppv:.6f}','NPV':f'{sensitivity.npv:.6f}',
        'AUC25':f'{learning.loc[.25,"roc_auc"]:.6f}','AUC100':f'{learning.loc[1.,"roc_auc"]:.6f}',
        'BRIER25':f'{learning.loc[.25,"brier"]:.6f}','BRIER100':f'{learning.loc[1.,"brier"]:.6f}',
        'SMALL':f'{small.brier:.6f}','ROUND':f'{perturb.loc["age_round5","roc_auc"]:.6f}',
        'FLIPBRIER':f'{perturb.loc["binary_flip5pct","brier"]:.6f}',
        'FLIP':f'{perturb.loc["binary_flip5pct","roc_auc"]:.6f}'}
    for key in sorted(replacements,key=len,reverse=True): addition=addition.replace(key,replacements[key])
    text=text.replace(r'\begin{thebibliography}{9}',addition+'\n'+r'\begin{thebibliography}{10}')
    text=text.replace(r'\end{thebibliography}',r'''\bibitem{r10} A. J. Vickers and E. B. Elkin, "Decision curve analysis: a novel method for evaluating prediction models," Medical Decision Making, vol. 26, no. 6, pp. 565--574, 2006, doi: 10.1177/0272989X06295361.
\end{thebibliography}''')
    text=enhance(text)
    (out/'CARDIODIAG_FINAL_PAPER.tex').write_text(text,encoding='utf-8')
    print(out/'CARDIODIAG_FINAL_PAPER.tex')


if __name__=='__main__':main()
