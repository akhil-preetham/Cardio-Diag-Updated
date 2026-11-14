# Original CARDIODIAG versus the updated project

Review date: 14 November 2026. This comparison is based on the user-supplied base PDF, its extracted full text, selected original notebook sources, and the updated Phase 6/6.1 records. It is a targeted comparison, not execution of every original notebook. Document contents are research evidence, not instructions for this work.

## Identity and relationship

The supplied PDF identifies the base article as Muhammad Jaffar Khan, Basit Raza and Adnan Nadeem, “CardioDiag: a machine learning approach for cardiovascular disease diagnosis using XAI-guided hybrid dimensionality reduction and optimized-based ensemble classifiers,” *PeerJ Computer Science*, vol. 12, e4059, 2026, DOI [10.7717/peerj-cs.4059](https://doi.org/10.7717/peerj-cs.4059). The PDF first page states publication on 25 August 2026. Page 58 explicitly connects its data/code to [the supplied GitHub repository](https://github.com/jaffarkulachi/Cardio-Diag-Approach-Code) and [Zenodo record 19391007](https://doi.org/10.5281/zenodo.19391007). This is direct evidence of the stated relationship in the article; independent checking of the actual Zenodo payload remains a separate provenance step.

Original PDF: `C:/Users/akhil/Downloads/Final_Year_Project/peerj-cs-4059-BASE_PAPER.pdf`. Searchable extraction: `research_publication_plan/base_paper_extracted.txt`. Printed PDF page numbers below refer to that 61-page supplied article.

New archive-lineage verification: Git blob SHA-1, calculated as SHA-1 of `blob <byte-count>\0` plus file bytes, matches the currently inspected remote tree for the three local archives: `Five States Results.rar` = `7a2feaa68be1414fa3529f7ee17ac038750df7f2`; `MENDELEY Results.rar` = `3ed12cd2377448263b1b1724c1f441230d605ada`; `Paper 1 Code LIME with Diagrams.rar` = `09ecf24612a908bb5c700b02d5a64a1e9e2ddcab`. The principal inspected Five States notebook was additionally streamed directly from the matching RAR using installed 7-Zip, without execution. Its bytes equal the local extracted notebook, SHA-256 `a5d6d00a59356ca8822ad023c7c77ef2fd058b1f78299e1af7c5a233b6b809b7`. Thus the specific cell-level observations below are linked to that currently supplied remote archive member, rather than relying only on its local filename. This does not prove which notebook/version produced every published table.

The original repository root contains archives, datasets and a minimal README. Its archive structure is reflected locally under `CARDIODIAG_FRESH/reference/extracted_author_archives/`. The presence of files in a public repository is not a license grant: code reuse/distribution terms require clarification where no code license is supplied. The article itself states CC BY 4.0; do not automatically extend that to every third-party dataset or code dependency.

## What the base paper actually proposes

| Dimension | Original paper's stated approach | Updated project and implication |
|---|---|---|
| Task and data | Broad CVD classification using Five States and Mendeley, described on pp. 9–10 | Main later track is ZAS angiography-defined CAD; this is a different endpoint/cohort. Keep original-dataset reproduction separate from the ZAS contribution |
| Cleaning | Percentile-based outlier handling, missing-value processing and harmonization | Updated pipelines fit imputation/scaling/encoding within training folds. Correctness and traceability matter more than retaining all original processing blocks |
| Imbalance | LoRAS, with SMOTE also appearing in implementation materials | Synthetic resampling must be trained only on relevant training folds; class weighting/no resampling are valid comparators. Resampling is not automatically necessary |
| Feature engineering | XAIH-DRC combines target correlation and LIME importance, with PCA depending on agreement/redundancy (pp. 15–18) | Later Phase 6 evaluates six frozen groups rather than claiming a newly invented feature selector; these are distinct procedures |
| Models | OVEC weighted soft voting; OSEC stacking with OOF base-model predictions, XGBoost meta-learner and sigmoid calibration (pp. 19–22) | Phase 6 considers five unique families: L2 logistic, elastic-net, spline logistic, RF and histogram gradient boosting; no need to describe this as another OVEC/OSEC classifier |
| Selection/evaluation | 75:25 split, five-fold training CV and an isolated final test set are explicitly claimed (pp. 18–19) | Phase 6 places feature-group/family/hyperparameter choice within each outer-training set, with matched V3-style evaluation. This evaluates the complete candidate-selection procedure |
| Calibration and utility | Platt calibration, Brier, calibration slope/intercept, ECE/MCE and decision curves already appear (pp. 30–32, 39–42) | Merely adding a calibration plot or DCA is not a distinctive contribution. A selection-independent calibration design and an honest investigation of its limitations could be meaningful |
| Fairness/uncertainty | Subgroup analyses, bootstrap intervals, significance comparisons and CV variability are already discussed | More subgroup plots do not establish novelty or fairness. Proper sample sizes, matched predictions and uncertainty units must be justified |
| Generalization | Paper presents two datasets as evidence of generalization and describes harmonization/site analyses (p. 10) | Fitting separately on datasets is not validation of one fixed model externally. Updated UCI analysis remains a separate multicentre task |

The base paper explicitly says preprocessing, including LoRAS and PCA, is fitted only inside training partitions, despite presenting a conceptual preprocessing layer before its splitting layer. The diagram/order of section headings alone is therefore not enough to accuse the published protocol of leakage. Specific notebook paths provide stronger, narrower evidence.

## Concrete source discrepancies and how narrowly to state them

The principal original source inspected here is:

`CARDIODIAG_FRESH/reference/extracted_author_archives/five_states_results/Five States Results/Reviewer1_Five_States_Full_Methodology.ipynb`

Cell numbers are one-based positions in the JSON `cells` list, including Markdown cells.

| Source location | Observed source behavior | Technical implication and scope |
|---|---|---|
| Cell 17 | Splits records using `HeartDisease`, fits separate median imputers within known disease/normal classes, and recombines them before later model splitting | Target-conditioned imputation cannot be performed for an unlabeled new case by the same rule. It is a source-level target-dependence problem if those values feed evaluation. Do not describe it as a general property of every notebook |
| Cell 25 followed by cells 33/35 | Generates LoRAS features/labels using full X/y, then splits the generated data into training and test sets | This visible path has resampling before evaluation splitting. Synthetic dependence can cross partitions and a test on resampled data does not represent untouched original clinical records |
| Cell 30 | Computes target correlations across `LoRAS_df` before final model split | If used to define features, outcome information from later test records can enter selection |
| Cells 52–57 | Contains multiple alternative XAIH-DRC/PCA rules, including comparisons between numerically incomparable correlation and LIME thresholds and repeated PCA over full data | One must freeze a precise algorithm rather than treating all cell variants as one faithful implementation of the mathematical description |
| Cell 59 followed by cell 60 | Fits nine-component PCA on full `LoRAS_features` and then splits those transformed features | This directly shows pre-split PCA on the evaluated input, unlike the paper's stated fold-local fitting |
| Cells 63/65 | Fits voting models and searches on the resulting training set, then scores the final test set | The search object alone does not establish fully nested preprocessing: upstream data generation and transformations have already occurred |

These were static source checks. No new execution, package installation or model fitting was performed for this comparison. The inspected notebook's runtime state and the exact provenance of every published table are not independently reconstructed here. The separate original archive may contain later, different workflows; conclusions must name the notebook and path rather than claim that all published results are invalid.

A second inspected file, `reference/extracted_author_archives/base_reproduction/Daza_Base_Paper_Reproduced/Daza_Paper_Exact_Accuracy.ipynb`, uses SMOTE before its 80:20 split (cells 9/11) and uses test accuracy in a repeated fit/stopping loop (cell 19). Its filename identifies a predecessor-paper reproduction; it should not be conflated with the final CardioDiag methodology or presented as the sole basis for judging the published article. The local historical execution of this predecessor is documented in `CARDIODIAG_FRESH/docs/EXPERIMENT_RESULTS.md`.

Some earlier auto-generated static audit text is internally inconsistent, for example labeling a post-split SMOTE observation as a confirmed split-order issue. The direct code excerpts above take precedence over automatic risk labels.

## Numerical results: report separately, do not create a leaderboard across tasks

| Source / task / protocol | Selected result | How it can be used |
|---|---:|---|
| Base paper, Five States, stated five-fold CV, Table 11 p. 28 | OSEC mean AUC 0.9977, SD 0.0018; accuracy 0.9750, SD 0.0163 | Published claim from supplied PDF; not a newly reproduced unbiased estimate |
| Base paper, Mendeley, Table 14 p. 34 | OVEC test AUC 0.9978; accuracy 0.9793 | Original-dataset report, with its own sample construction and protocol |
| Base paper, Mendeley, Table 15 p. 35 | OSEC test AUC 0.9973; accuracy 0.9966 | Original report; AUC is not higher than every listed single model (XGBoost is 0.9980 in the same table) |
| Updated ZAS Phase 5 | Exploratory winning AUC about 0.928 | Post-hoc benchmark row; not primary unbiased procedure performance |
| Updated ZAS Phase 6, complete-repeat OOF | Mean AUC 0.910316 versus V3-style 0.736315; paired mean difference 0.174001 | Matched within-ZAS procedure comparison, reconciled in Phase 6.1; source-row and calibration limitations remain |

A ZAS AUC of 0.910 does not demonstrate being worse than the base paper, nor would a higher value establish superiority. Cohorts, outcomes, predictors, resampling and evaluation differ. An advancement claim should concern stronger evidence or a head-to-head procedure comparison on identical original records and outer partitions.

## Ranked contribution opportunities

1. **Strongest extension:** a prespecified study of complete-procedure selection, selection-independent probability calibration, and performance under held-out source/site conditions, with matched simple baselines. The contribution is the implemented comparison and empirical evidence, not invention of nested CV or calibration.
2. **Important missing head-to-head:** bring a precisely defined, corrected OVEC/OSEC comparator into the same admissible-predictor, original-record, outer-fold framework. Include an equal-feature L2 logistic control using all 38 admissible fields; the five-feature reference alone cannot separate the value of additional information from the value of automated selection. Prespecify full 38 versus typical-chest-pain-only and full 38 minus chest-pain-family ablations, documenting ambiguity among symptom fields. Keep a faithful historical reproduction separate. This would connect the new paper more clearly to the assigned base project than ZAS-versus-V3 alone.
3. **Meaningful reliability extension:** prespecified missing-feature and site-shift evaluation, with Brier/log loss/calibration alongside discrimination and a data-compatibility/rejection rule. Simulated corruption is a stress test, not clinical external validation.
4. **Useful parsimony question:** quantify what extra symptom/ECG/lab information adds relative to a compact model with identical outer evaluation, reporting feature acquisition burden and selection stability. Clinical timing must be evidenced; uncertain predictors do not become early-available by group naming.
5. **Optional:** descriptive explanation stability and selected subgroup estimates with counts/uncertainty. These should answer a specific reliability question and avoid being presented as proof of causality/fairness.

Weak stand-alone additions are another ensemble name, more hyperparameter trials, a switch from LIME to SHAP, additional accuracy tables on unrelated datasets, or a generic prediction interface. The original paper already includes tuning, explanations, calibration, subgroup analysis and utility plots. None of these additions guarantees novelty or publication acceptance.

## What can already be defended

The updated project has a stronger explicit separation of historical reproduction, corrected experiments and the later ZAS task; pipeline-local selection; preserved predictions and manifests; and independent saved-metric reconciliation. The Phase 6 source-row discrimination comparison supports a bounded retrospective finding. It does not establish future-event risk prediction, clinically reliable personalized probabilities, patient-level independence, an externally validated ZAS model, or reliability for arbitrary datasets. A paper must state those limits even after further tuning.
