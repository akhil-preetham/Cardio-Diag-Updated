# Selection-Aware Retrospective Discrimination of Angiography-Defined Coronary Artery Disease: A Reproducible Reanalysis

Author names, order, affiliations and corresponding author require confirmation.

Working IEEE-style research draft, 14 November 2026. This draft reports existing saved experiments; proposed follow-up experiments are not reported as completed. It has not been submitted or accepted.

## Abstract

Cardiovascular machine-learning benchmarks often compare many feature sets and classifiers, making the evaluation of the entire selection procedure important. We present a reproducible retrospective reanalysis developed from the CardioDiag research workflow. On 303 Z-Alizadeh Sani source records, with 216 coronary artery disease labels and 87 normal labels, we evaluated a feature-group/model-selection procedure against a matched five-feature reference procedure. Three repeats of five outer folds were used; within outer training, three-fold validation selected among six feature groups and five model families, including hyperparameters. Complete-repeat out-of-fold ROC-AUC values were 0.9134, 0.8999 and 0.9176 for the selection procedure, versus 0.7370, 0.7420 and 0.7300 for the reference. The mean paired difference was 0.1740. Mean Brier scores were 0.1076 and 0.1746, respectively. Independent saved-artifact checks reconciled prediction coverage, labels, metric aggregation and listed hashes. Calibration scores were estimator-out-of-fold but selection-dependent, and their training size differed from the final refit. These limitations, uncertain predictor timing, unverified patient independence and retrospective protocol development constrain interpretation. The findings support higher observed discrimination on the evaluated source-row partitions, not clinical readiness, calibrated individual risk or external generalization. Equal-feature controls and independently constructed calibration are prioritized for further evaluation.

## Index Terms

Coronary artery disease, nested cross-validation, model selection, calibration, reproducibility, retrospective discrimination.

## I. Introduction

Public cardiovascular datasets make it possible to investigate classification methods with limited computational resources. However, a high benchmark score does not establish performance before a diagnostic decision, and selecting the best result from many feature/model combinations can introduce optimism. The evaluated object should include the selection process itself rather than only the winning fitted configuration [2].

The original CardioDiag study combines percentile-based preprocessing, localized random affine shadow sampling, explanation-guided feature engineering and optimized voting/stacking ensembles [1]. It reports results on Five States and Mendeley datasets, together with calibration, decision-curve, subgroup and ablation analyses. The present study derives from that project but has a different primary task: retrospective discrimination of an angiography-defined label in the Z-Alizadeh Sani (ZAS) resource. Its results are not a direct performance comparison with the original paper because datasets and protocols differ.

Our contribution is an applied methodological reanalysis: a bounded selection procedure is evaluated on matched outer partitions against a fixed-feature reference; saved predictions and metric definitions are independently reconciled; and limitations concerning calibration, timing and the unit of validation are made explicit. We do not propose a new classifier or claim algorithmic priority. The research question is whether the recorded selection procedure had higher retrospective discrimination than the matched five-feature reference on the evaluated partitions. A separate historical UCI centre-held-out investigation is retained as contextual evidence, without treating it as external validation of ZAS.

## II. Background and Related Work

CardioDiag's OVEC and OSEC architectures and XAIH-DRC feature engineering are attributed to Khan et al. [1]. Adding another explanation plot, tuning method or calibration curve is therefore insufficient by itself to distinguish a subsequent study. The relevant advance must be supported by a well-defined evaluation question and evidence beyond an unqualified winning score.

Cawley and Talbot describe overfitting of finite-sample model-selection criteria and consequent bias in performance evaluation [2]. Nested validation addresses selection occurring within the evaluated training procedure. It does not independently validate researchers' earlier decisions made after exploring the same dataset. Calibration is also distinct from discrimination: accurate ordering does not establish accurate probabilities, and calibration can vary across settings [5].

TRIPOD+AI provides reporting guidance, while PROBAST+AI supports assessment of quality, risk of bias and applicability [6], [7]. We use them as frameworks for transparent reporting, without asserting checklist compliance or low risk of bias. Our analysis is a retrospective development and evidence-reconciliation study rather than prospective clinical validation.

## III. Materials and Methods

### A. Data, Outcome and Evaluation Unit

The local validated ZAS file contains 303 records and 56 columns including Cath. Cath was mapped as Cad = 1 and Normal = 0, yielding 216 positive and 87 negative records; prevalence was 0.7129. The authoritative repository describes angiographic CAD using a diameter-narrowing threshold of at least 50% [3]. Its listed 54-feature description differs from the local 55 non-target columns; exact source-version reconciliation remains unresolved. We do not silently remove or relabel a column to force agreement.

Unique patient identity and row-level predictor acquisition times are not established. Accordingly, records are described as source rows, and row-based partitioning does not imply independently verified patients. The study cannot establish predictor availability before the decision to perform angiography. Public-data reanalysis does not by itself settle the authors' institutional ethics or data-use obligations.

### B. Candidate Information Sets and Reference Procedure

The reference uses Age, Sex, Current Smoker, EX-Smoker and FH. Its model family is class-balanced logistic regression; regularization C is selected inside outer training. It is a matched re-evaluation of a V3-style training procedure, not an unchanged historical fitted model, calibrator or threshold.

The selection pool contains six groups: the five reference fields, basic risk factors, symptoms/examination, resting ECG, laboratory measurements and their 38-field eligible_combined union. The combined group includes seven risk fields, ten symptom/examination fields, seven ECG fields and fourteen laboratory fields. The group name is historical and does not establish decision-time eligibility. EF-TTE, Region RWMA and VHD were excluded as potentially downstream/diagnostic fields; Cath and direct outcome information were not candidate predictors. All retained fields remain timing uncertain.

Five unique family labels were evaluated: L2 logistic regression, elastic-net logistic regression, spline logistic regression, random forest and histogram gradient boosting. Logistic and Ridge aliases from earlier benchmarking were not counted as separate families. This gives 30 group-family candidates; hyperparameter grids additionally vary within each candidate.

### C. Training and Outer Evaluation

The completed protocol used the first three repeats of frozen five-fold outer assignments, giving 15 paired outer partitions. A larger five-repeat plan was reduced for computational reasons, as documented in the saved protocol. Operational inner splits were three-fold stratified partitions generated within outer training using seed 61051 + 100 x repeat + fold. Older inner assignments in the source manifest were not the operational Phase 6 inner folds.

Each candidate's hyperparameters were selected by inner ROC-AUC, then the highest-scoring group-family candidate was chosen using a fixed tie order. Imputation, encoding, scaling and optional spline transformations were fitted within pipelines on the relevant training partitions. Outer-test labels did not select feature groups, families, parameters, calibration maps or thresholds. Post-fit permutation importance used outer-test data descriptively and did not feed back into saved predictions.

The candidate universe followed earlier dataset benchmarking, and outer assignments were reused from earlier work. Thus, freezing choices before this particular run does not make the overall research process a prospectively registered confirmation. The result is reported as a retrospective methodological reanalysis. New random seeds on the same data would not resolve this limitation.

### D. Calibration and Threshold Construction

After selecting a configuration using all inner folds, the historical implementation refitted that selected configuration on each inner training block to obtain held-block scores. A sigmoid map was fitted to raw base-model probabilities and labels, and a Youden-J threshold was chosen on the resulting training probabilities. The selected full-outer-training pipeline retained from GridSearchCV's refit was used with the learned map to obtain outer-test probabilities.

Although each score excluded its row from the corresponding estimator fit, that row's label had already influenced configuration selection. The scores are therefore estimator-out-of-fold but selection-dependent. The inner scoring models also used approximately two-thirds of outer training, whereas the final refit used all of it. This construction does not leak outer-test labels, but limits claims about calibration independence and transport of the learned map. A separate audit corrected a reporting display containing four calibration-fold labels for an operational three-fold design; it did not generate a corrected predictive experiment.

### E. Metrics, Aggregation and Uncertainty

For each repeat, metrics were computed over its complete 303-row out-of-fold prediction set. AUC over this set is not the arithmetic mean of its five fold AUCs. We report ROC-AUC, average precision (AP), Brier score, log loss and descriptive sensitivity/specificity at training-selected thresholds. AP is interpreted alongside prevalence. Calibration intercept/slope are jointly estimated logistic coefficients using the logit of outer probabilities; the intercept is not a calibration-in-the-large estimate with slope fixed at one. Both can be unstable in this sample.

Three repeats reuse the same rows and overlapping training sets. Their variation and paired differences are descriptive; no population confidence interval, independent-replicate significance test or patient-level superiority claim is made. These metrics describe a complete learning procedure assembled from fold-specific fitted models, not one fixed deployed model.

### F. Evidence Reconciliation and Reproducibility

The Phase 6.1 audit compared saved predictions with source labels and outer assignments, reconstructed fold and complete-repeat metrics, checked candidate-ranking metadata and verified enumerated hashes. It did not rerun the full benchmark. The current handover additionally ran eight focused Phase 6/6.1 tests, which passed, and checked 190 direct path-to-hash entries across selected manifests, with no mismatches. These are engineering and traceability checks; they do not resolve patient independence or establish scientific correctness by hashing.

Source code, split assignments, prediction tables and historical audit reports are retained. Working requirements and a later pinned environment snapshot serve different purposes. The historically documented Phase 6 environment uses Python 3.11.16 and scikit-learn 1.9.1; exact environment provenance should accompany a public release. No Git worktree was available for a commit-based provenance claim.

## IV. Results

### A. Primary Retrospective Discrimination

Table I shows complete-repeat AUCs. Mean repeat AUC was 0.910316 for selection and 0.736315 for the V3-style reference; the mean paired difference was 0.174001. All three repeat estimates favored the selection procedure. These observations are specific to the evaluated source-row partitions and do not establish general population superiority.

Table I. Complete-repeat OOF ROC-AUC; 303 rows per procedure per repeat.

| Repeat | Selection | V3-style | Difference |
|---|---|---|---|
| 1 | 0.9134 | 0.7370 | 0.1764 |
| 2 | 0.8999 | 0.7420 | 0.1579 |
| 3 | 0.9176 | 0.7300 | 0.1877 |
| Mean | 0.9103 | 0.7363 | 0.1740 |

### B. Probabilistic and Threshold-Based Summaries

Table II gives means of the three complete-repeat summaries. Observed Brier and log loss were lower for selection. This is performance of the historical probability-producing procedure on outer predictions, not evidence that calibration construction was independent of selection or that probabilities are clinically transferable. Threshold metrics have no established screening interpretation.

Table II. Means across three repeats; AP prevalence reference = 0.7129.

| Metric | Selection | V3-style |
|---|---|---|
| AP | 0.9573 | 0.8469 |
| Brier | 0.1076 | 0.1746 |
| Log loss | 0.3467 | 0.5286 |
| Sensitivity | 0.8318 | 0.6898 |
| Specificity | 0.8314 | 0.7088 |
| Calibration intercept | 0.173 | -0.007 |
| Calibration slope | 0.868 | 1.005 |

### C. Selection Behavior and Historical Context

The 38-field combined group was selected in all 15 outer fits. L2 logistic was chosen nine times, elastic-net four times and spline logistic twice; neither tree family was selected. Group-level stability does not show that all 38 fields are necessary. It also suggests that a fixed rich-feature regularized logistic procedure is an essential control before attributing the difference to automated model selection.

An earlier feature-group benchmark reported a winning AUC near 0.928, selected post hoc from 36 rows. Its score is retained as exploratory historical context, not a primary unbiased estimate or a valid direct estimate of the benefit of nesting. Differences in procedure and evaluation preclude interpreting the difference between 0.928 and 0.910 as a measured selection-bias correction.

The separate UCI investigation reconstructed 920 records from Cleveland, Hungary, Switzerland and VA Long Beach using num = 0 versus num = 1-4 [4]. Historical pooled centre-held-out AUCs were 0.7179, 0.8225, 0.8215 and 0.8450 across successive information stages. The final stage includes exercise-test information and addresses a later information task. These summaries provide context for site/information sensitivity; they are not external validation of ZAS, and the current focused tests did not rerun that entire UCI reconstruction.

## V. Discussion

The evaluated selection procedure showed higher retrospective discrimination than a sparse five-feature reference. The comparison supports the observed difference between procedures; it does not isolate the effect of a model-selection algorithm from the effect of adding symptom, examination, ECG and laboratory information. Since regularized logistic families were selected throughout, additional model complexity may be unnecessary. A result showing similar performance with a simpler rich-feature model would be valuable even without increasing AUC.

The calibration audit is a substantive limitation. Honest outer evaluation can assess a procedure whose internal calibration is imperfect, but that does not make its calibration inputs independent. A future comparison should select the entire base procedure without each calibration block, retain each fitted base/calibrator pair, and compare calibrated versus uncalibrated ensembles built from identical base components. This design has not been executed here and no improvement is assumed.

Predictor timing and referral spectrum also constrain clinical interpretation. Recorded symptoms may encode diagnostic work-up among people referred for angiography. Without timestamps and a defined decision point, a high AUC cannot be interpreted as early detection or a substitute for diagnostic testing. Simulated missingness tests and centre-held-out evaluations can examine bounded robustness, but neither guarantees reliable answers under arbitrary clinical distribution shift.

## VI. Limitations and Future Work

This is a small retrospective source-row study with substantial preceding dataset exploration. Patient uniqueness, acquisition timing and exact schema lineage remain unverified. The procedure used only three outer repeats. Selection-dependent calibration, model-size mismatch and data-selected Youden thresholds limit probability and decision claims. Repeated OOF summaries are not independent samples, and saved-artifact reconciliation is not a fresh reproduction of all model fits.

There is no independently compatible validation cohort for the exact current ZAS models, no demonstrated clinical utility, and no established fairness or prospective performance claim. UCI is a separate task. Public dataset availability does not establish interchangeable outcomes or predictors. Future work should prioritize equal-feature controls, symptom/laboratory/ECG ablations, selection-independent calibration and independently sourced, harmonized evaluation before broader tuning. New experiments must be versioned separately and reported even if performance decreases.

## VII. Conclusion

A selection-aware procedure had higher complete-repeat out-of-fold discrimination than a matched V3-style reference on three evaluated ZAS source-row repeats. Independent artifact checks supported the recorded results and exposed limits that high AUC alone would conceal. The contribution is a reproducible, bounded retrospective reanalysis. Clinical reliability, patient-level generalization and calibrated individual risk remain unestablished; equal-information comparisons and properly separated calibration are the most meaningful immediate extensions.

## Author and Submission Notes

Author names/order, affiliations, contributions, correspondence, ethics determination, data-use permissions, funding and conflicts require author confirmation. Do not copy these declarations from the base paper. AI-assisted drafting was used to prepare this working document; authors must verify its content and complete the target venue's current disclosure requirements. This document is not submission-ready until declarations, citations, venue formatting and final evidence review are completed.

## References

[1] M. J. Khan, B. Raza, and A. Nadeem, "CardioDiag: a machine learning approach for cardiovascular disease diagnosis using XAI-guided hybrid dimensionality reduction and optimized-based ensemble classifiers," PeerJ Computer Science, vol. 12, e4059, 2026, doi: 10.7717/peerj-cs.4059.

[2] G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," Journal of Machine Learning Research, vol. 11, pp. 2079-2107, 2010.

[3] R. Alizadehsani, M. Roshanzamir, and Z. Sani, "Z-Alizadeh Sani," UCI Machine Learning Repository, dataset, doi: 10.24432/C5Q31T.

[4] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, dataset, doi: 10.24432/C52P4X.

[5] B. Van Calster et al., "Calibration: the Achilles heel of predictive analytics," BMC Medicine, vol. 17, art. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[6] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, e078378, 2024, doi: 10.1136/bmj-2023-078378.

[7] K. G. M. Moons et al., "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," BMJ, vol. 388, e082505, 2025, doi: 10.1136/bmj-2024-082505.
