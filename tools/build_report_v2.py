"""Build an additive IEEE report source; all v1 artifacts remain unchanged."""
from pathlib import Path
import re
from portable_release import PROJECT, discover_workspace

root = discover_workspace()
old = root / PROJECT / 'paper/research_extension_v1/CARDIODIAG_RESEARCH_REPORT.tex'
out = root / PROJECT / 'paper/research_extension_v2'
text = old.read_text(encoding='utf-8')
# Keep the historical numeric tables; replace four earlier figures with the new
# paired ROC/calibration figure to keep the integrated report concise.
text = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', text, flags=re.S)
text = re.sub(r'\\section\{Discussion\}.*?(?=\\section\{Limitations)',
              lambda _: r'''\section{Discussion}
Information-matched comparisons explain much of the apparent improvement over the earlier five-field reference: the simple 38-field logistic control outperformed historical selection. Corrected calibration separates the entire selection process from its calibration labels, yet ranking and probability loss show different trade-offs. Source-only transport loses discrimination and probability accuracy, while input deletion further reduces information. These findings support a reproducible retrospective methods study; they do not establish algorithmic superiority or clinical deployment readiness. The final follow-up section tests scoped remedies while retaining all negative findings and the established model defaults.

''', text, flags=re.S)
text = text.replace('We perform no AngioCAD label-based adaptation.',
                    'The original transport experiment performs no AngioCAD label-based adaptation; the separately frozen follow-up below evaluates retrospective target adaptation with disjoint adaptation and evaluation folds.')
text = text.replace('No target recalibration was undertaken to conceal these limitations.',
                    'The original transport scores remain unchanged. The separately reported retrospective target-adaptation follow-up does not resolve these endpoint or population limitations.')
text = text.replace('The contribution is a reproducible empirical assessment',
                    'A frozen follow-up improved source independent-calibration AUC/Brier from 0.9096/0.1037 to 0.9144/0.1029 and target Brier from 0.1982 to 0.1507 using held-out target intercept updating. Typed encoding and clean-input augmentation regressed and were not promoted. The contribution is a reproducible empirical assessment')
text = text.replace('The computational extension, audit artifacts and compiled report are complete within the stated retrospective scope.',
                    'The computational extension, audit artifacts and compiled report are complete within the stated retrospective scope; the version-two follow-up below adds probability updating, robustness comparisons and explicit nonregression decisions.')
text = text.replace('\\begin{IEEEkeywords}', '\\noindent\\textit{Integrated report, version 2, 14 November 2026. Historical results are retained; the final numbered section presents the additive reliability follow-up.}\\par\n\\begin{IEEEkeywords}', 1)
addition = r'''
\setcounter{table}{4}
\section{Additive Reliability Follow-Up: Methods and Results}
The follow-up protocol was frozen before its experiments, after examining the original results. Consequently this remains retrospective research on already explored cohorts. The development data, 45 independent calibration checkpoints, original outer assignments and target probabilities were hashed before execution. No historical predictions, model bundles or defaults were replaced. Every prespecified variant, including negative findings, is retained. This adds an empirical calibration and robustness study rather than a new clinical validation cohort.

\subsection{Explicit Typing and Calibration Stability}
A fixed schema partitions the 38 eligible fields into 17 continuous measurements, one ordinal field, 11 numeric binary indicators and nine nominal fields. Continuous and ordinal fields receive training-fold imputation and scaling; categories receive training-fold mode imputation and fixed-category encoding. A three-value logistic C grid is tuned within the original source partitions. The same historical calibration convention is retained solely for matching this comparison. Mean AUC/Brier is 0.899248/0.112920, below the historical equal-feature control's 0.917447/0.101124. Explicit typing improves transparency but this tested representation is not promoted.

For calibration, unchanged independent v1 components supply calibration-block labels and raw scores, plus outer-test raw predictions. Three alternatives are fitted: an intercept-only logit offset, a logit-slope sigmoid with fixed C=1, and a sigmoid whose C is selected from 0.1, 1 and 10 using three-fold negative log loss within the calibration block. Development, calibration and outer-test records are disjoint. Three component probabilities are averaged for each outer test. Base classifiers are not reselected using calibration or outer-test labels.

\begin{table}[htbp]
\centering\caption{Independent-component calibration follow-up: source repeat means}
\small\begin{tabular}{lrrr}\toprule
Procedure & AUC & Brier & Log loss\\\midrule
Identical raw ensemble & 0.916418 & 0.105460 & 0.341194\\
Existing weak sigmoid & 0.909607 & 0.103731 & 0.338996\\
Intercept-only offset & 0.915851 & 0.105450 & 0.341636\\
Fixed-C1 logit sigmoid & 0.914449 & 0.102922 & 0.333578\\
Calibration-block CV & 0.913598 & 0.103401 & 0.335545\\\bottomrule
\end{tabular}\end{table}

Fixed-C1 calibration improves AUC by 0.004842 and Brier by 0.000809 relative to the existing independently calibrated ensemble. It still does not exceed the identical raw ensemble's AUC. These comparisons concern distinct probability constructions, not evidence of a universally superior classifier. The source-record bootstrap interval for its Brier difference against raw is approximately [-0.005329, 0.000685], which crosses zero. Thus the observed proper-loss improvement is descriptive rather than established statistical superiority.

\subsection{Training-Only Missingness Augmentation}
The five-core estimator adds one independently 20\%-masked copy of its training inputs, assigns each original/copy half weight, and fits imputation, missingness indicators and logistic regression. Augmentation occurs inside estimator.fit after each inner split; no copied record crosses from training into validation. C is selected using clean inner-fold AUC. Both methods are evaluated on the exact previous four MCAR masks and on complete deletion of Age, smoking, or the combined DM/HTN/smoking block.

At 50\% random cell deletion, mean AUC improves from 0.687544 to 0.703322 and Brier decreases from 0.186714 to 0.184097. On complete records, AUC falls from 0.805884 to 0.804465 and Brier rises from 0.152801 to 0.153791. Whole-age and combined-risk deletion also worsen. Therefore this estimator remains an optional robustness experiment. Random masking is not a simulation of all clinical missingness mechanisms, and the approach cannot recover unobserved information.

\subsection{Retrospective Target Adaptation}
Frozen source predictions are updated using three repeats of stratified five-fold target adaptation/evaluation on the 377 matched AngioCAD cases. Each map is fitted only using the other target folds and evaluated on held-out cases. Comparators include frozen raw/source-calibrated predictions and an adaptation-fold prevalence constant. The target endpoint remains stenosis greater than 50\%; the ten boundary cases are also excluded or reassigned for scoring sensitivity without changing the calibration-training labels.

\begin{table}[htbp]
\centering\caption{Target adaptation: complete held-out repeat means}
\small\begin{tabular}{lrrr}\toprule
Procedure & AUC & Brier & Log loss\\\midrule
Frozen source raw & 0.712471 & 0.198249 & 0.578727\\
Frozen source calibrated & 0.707225 & 0.193659 & 0.562933\\
Adaptation prevalence & 0.492906 & 0.165652 & 0.513399\\
Target intercept offset & 0.712776 & 0.150723 & 0.485410\\
Target fixed-C1 sigmoid & 0.708542 & 0.148793 & 0.466986\\
Target inner-CV sigmoid & 0.709179 & 0.148911 & 0.466874\\\bottomrule
\end{tabular}\end{table}

Target updating improves probability loss over both frozen source scores and the fold-prevalence baseline. Fold-specific maps can alter pooled out-of-fold AUC despite preserving within-fold ranking when their slopes are positive. These results require labeled target adaptation records and do not establish transfer to a future hospital. The target cohort was already examined; calling this untouched external validation would be incorrect. The released source scoring interface does not silently apply target-updated maps.

\begin{figure*}[t]
\centering\includegraphics[width=0.94\textwidth]{ROC_CALIBRATION.pdf}
\caption{Repeat-one ROC and five-quantile calibration diagnostics for source calibration and target adaptation. These are descriptive plots; aggregate results use three complete repeats. The diagonal represents random ranking for ROC and perfect agreement for calibration.}
\end{figure*}

\subsection{Uncertainty, Subgroups and Integration Decisions}
One thousand paired source-record bootstrap draws share the same resampled source IDs across all three repeats, preserving pairing and repeated-record dependence. These intervals condition on saved fits and unverified row independence. Separately, 50 bootstrap-development typed-logistic refits tune C with three-fold stratified group validation: duplicate copies share a source-ID group. Only out-of-bag source IDs are evaluated. This assesses refitting sensitivity for that L2 pipeline, not uncertainty for the full 30-candidate selection procedure.

Prespecified sex and age groups (at most 60, greater than 60, missing) receive repeat-one class counts and performance summaries. Any subgroup with fewer than ten cases in either outcome class is flagged unstable. These diagnostics do not certify fairness. A fixed 0.5 threshold is reported descriptively without optimization and does not establish a clinical action threshold.

The descriptive acceptance gate requires nondecreased mean AUC and nonincreased mean Brier relative to the stated comparator. Fixed-C1 and inner-CV source calibration pass against existing corrected calibration. Intercept-only target adaptation passes against frozen raw probabilities. Typed encoding and clean-input augmentation fail and are not promoted. Other target maps improve loss while slightly reducing pooled AUC; their trade-off is explicitly retained. Passing the gate does not automatically change a released model or establish significance.

Verification recomputed 84 complete-repeat metric groups, checked calibration/adaptation separation and 50 bootstrap memberships, and reproduced all 909 ordinary-core baseline predictions to tolerance 1e-12. The release suite passed 45 tests, with one optional historical LoRAS test skipped. No experiment warnings were captured. Scoring now checks the source reference in the requested checkout, validates its hash before model loading, enforces Python 3.11 and creates output exclusively. All 163 original primary manifest entries remain valid.

Acquisition timing, verified patient IDs, exact boundary stenosis values, authoritative coding equivalence and institutional declarations remain unresolved because the released tables cannot supply them. Completion here means the prespecified software and retrospective analyses are implemented and verified; prospective reliability and clinical deployment remain unestablished.
'''
# Append after the original conclusion/declarations and before references.
text = text.replace('\\newpage\n\\begin{thebibliography}', addition + '\n\\begin{thebibliography}')
(out / 'CARDIODIAG_RESEARCH_REPORT_V2.tex').write_text(text, encoding='utf-8')
print(out / 'CARDIODIAG_RESEARCH_REPORT_V2.tex')
