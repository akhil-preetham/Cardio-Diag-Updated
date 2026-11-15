"""Merge reviewed editorial material without importing the stale package results."""
import re

EVIDENCE_COMMIT = '97ee42c4d79453ef346bc0cfcb62e87fdfb7223b'


def enhance(text):
    text = text.replace('CARDIODIAG: Selection-Aware Evaluation, Calibration and Cross-Cohort Robustness for Retrospective Coronary Artery Disease Discrimination',
        'CARDIODIAG: Information-Matched Evaluation and Probability Reliability in Retrospective Coronary Artery Disease Classification')
    text = re.sub(r'\\section\{Introduction\}.*?(?=\\section\{Related Work)',lambda _:r'''\section{Introduction}
Public clinical benchmarks often emphasize the highest classification score. Interpretation also depends on whether models receive the same measurements, whether selection is evaluated inside training boundaries and whether probabilities retain their meaning beyond the source sample. Richer information can appear to justify a complex algorithm when the contribution actually comes from available predictors.

The original CardioDiag study introduced explanation-guided hybrid dimensionality reduction and optimized voting/stacking ensembles \cite{r1}. Those architectures and published results belong to the original authors. This extension studies a complete evaluation workflow rather than renaming those methods or claiming a new classifier. Its contribution is a paired information-matched comparison, calibration disjoint from complete model selection, compatibility-constrained cohort transport and explicit probability-loss and robustness analyses. Every negative comparison is retained; no matched OVEC/OSEC superiority experiment is claimed.

''',text,flags=re.S)
    text = re.sub(r'\\section\{Related Work and Research Questions\}.*?(?=\\section\{Data,)',lambda _:r'''\section{Related Work and Research Questions}
Classifier optimization is established on ZAS: Mansoor et al. compared machine/deep-learning methods \cite{r11}; Baratpur et al. combined feature selection, resampling and Bayesian SVM optimization \cite{r12}; Elemam et al. proposed SHAP-guided wrapper selection across public CAD resources \cite{r13}. Their scores are not head-to-head comparisons with ours: information sets, endpoints and validation rules differ. AngioCAD's provider study also includes its own stenosis framework \cite{r14}. The literature comparison and primary records appear in Supplement S2.

Selection criteria can overfit finite data \cite{r2}; nested evaluation does not erase earlier researcher exposure. Calibration differs from ranking \cite{r5}. External evaluation requires compatible populations, predictors and endpoints \cite{r8,r9}; TRIPOD+AI and PROBAST+AI guide reporting and appraisal, not certification \cite{r6,r7}.

We ask: RQ1, does rich-feature search add value over a same-information simple control? RQ2, how does calibration change loss when its labels are excluded from all base-model selection? RQ3, how sensitive are results to missing or perturbed inputs? RQ4, how do frozen source probabilities transport, and how does subsequent target-label updating differ? These are retrospectively motivated questions on previously explored records; final threshold and learning-curve analyses retain that scope.

''',text,flags=re.S)
    text = re.sub(r'\\subsection\{ZAS Source Data\}.*?(?=\\subsection\{AngioCAD)',lambda _:r'''\subsection{ZAS Source Data}
The validated ZAS table contains 303 rows, 55 non-target fields and Cath: 216 Cad and 87 Normal records (prevalence 0.7129). UCI describes CAD as coronary narrowing of at least 50\% \cite{r3}. The validated CSV matches the provider workbook; provider prose refers to 54 predictors while its workbook contains 55. File reconciliation and exact provenance are in Supplement S1. No exact full-record/all-predictor duplicates were found, but row keys are not verified person IDs. Repeated encounters and predictor timing remain unresolved. We therefore evaluate retrospective source-row discrimination, not prospective screening.

''',text,flags=re.S)
    text = re.sub(r'\\subsection\{AngioCAD Acquisition and Cohort Flow\}.*?(?=\\begin\{center\})',lambda _:r'''\subsection{AngioCAD Acquisition and Cohort Flow}
The provider's clinical and label workbooks contain 377 and 413 unique release IDs; all 377 clinical IDs match labels, leaving 36 label-only cases. The matched endpoint is positive if any of 15 segments has a category above 50\%: 298 positive, 79 non-positive. Ten non-positive records have a 26--50\% category, which cannot resolve equality at 50\%. Exact endpoint harmonization with source $\geq50\%$ is therefore unavailable.

The matched diagnoses are 186 STEMI, 20 NSTEMI, 55 unstable-angina and 116 stable-angina records. Non-positive stenosis is not a healthy population control; patient independence and selection due to missing clinical records remain unverified. Provider checksums, joins and severity reconciliation are documented in Supplement S1 and the preserved provenance files \cite{r4,r14}.

''',text,flags=re.S)
    text = re.sub(r'\\subsection\{Frozen Source Evaluation\}.*?(?=\\subsection\{Equal-Feature)',lambda _:r'''\subsection{Frozen Source Evaluation}
Three repeats of the historical stratified five-fold assignments give 15 paired partitions and one out-of-fold score per source row per repeat. Each outer-training set has approximately 242 records. Its three-fold inner AUC search spans six groups and five families (L2, elastic-net and spline logistic, random forest and histogram gradient boosting), totaling 30 candidates and 120 settings. Transforms and choices are confined to training; test labels are excluded. Historical dtype-based preprocessing is retained for comparability. Seeds, tie order and exact package versions are recorded in code/manifests; Supplement S3 gives the evidence boundary.

''',text,flags=re.S)
    text = re.sub(r'\\section\{Discussion\}.*?(?=\\section\{Limitations)',lambda _:r'''\section{Discussion}
First, the same-information logistic control exceeded historical selection, weakening the evidence that complexity explains the original large gain over a five-field reference. This is a result within the evaluated workflow: historical calibration remains selection-dependent, and a small reused cohort cannot establish a universal family ranking.

Second, isolating calibration labels from complete selection removes one avoidable dependence, but small calibration blocks still yield uncertain maps. Regularized logit calibration improved loss relative to the previous independent sigmoid while identical raw components retained higher AUC. The result is a measured ranking--probability trade-off rather than a new algorithm or guaranteed calibration benefit.

Third, frozen source transport and target-label updating answer different questions. Updating improves proper loss only with labeled target adaptation records. Endpoint boundaries, acute referral spectrum, missing data and coding uncertainty prevent attribution of the observed source/target difference solely to hospital shift. Final sensitivity, learning and perturbation analyses quantify additional limitations without establishing clinical effectiveness. A matched original-architecture comparison or a new timing-verified cohort would answer stronger, different questions; neither is represented as completed.

''',text,flags=re.S)
    text = text.replace('AI-assisted coding and drafting were used. Authors must verify the content and venue disclosure requirements before submission.',
        'OpenAI ChatGPT/Codex assisted coding, literature discovery, drafting and chart preparation. Authors must verify the content, tool-use disclosure and venue requirements before submission.')
    text = text.replace('Code, predictions and audit artifacts:',
        'Evidence baseline: commit '+r'\path{'+EVIDENCE_COMMIT+r'}. Supplementary methods, literature comparison, figure inputs and executable numerical traceability accompany this report. No independent investigator or clinical replication is claimed. Code, predictions and audit artifacts:')
    refs = r'''
\bibitem{r11} C. M. M. Mansoor, S. K. Chettri, and H. M. M. Naleer, "Development of an efficient novel method for coronary artery disease prediction using machine learning and deep learning techniques," Technology and Health Care, vol. 32, no. 6, pp. 4545--4569, 2024, doi: 10.3233/THC-240740.
\bibitem{r12} A. Z. Baratpur et al., "Coronary artery disease prediction using Bayesian-optimized support vector machine with feature selection," Frontiers in Network Physiology, vol. 5, art. 1658470, 2025, doi: 10.3389/fnetp.2025.1658470.
\bibitem{r13} T. Elemam, H. Refaat, and M. Makhlouf, "An explainable AI-driven hybrid feature selection approach for coronary artery disease diagnosis," Scientific Reports, vol. 16, art. 10411, 2026, doi: 10.1038/s41598-026-41712-y.
\bibitem{r14} M. S. Hosseini et al., "AngioCAD: a public x-ray angiography dataset and an adaptive fusion framework for stenosis detection," Computer Methods and Programs in Biomedicine, vol. 282, art. 109368, 2026, doi: 10.1016/j.cmpb.2026.109368.
'''
    text = text.replace(r'\end{thebibliography}',refs+'\n'+r'\end{thebibliography}')
    return text
