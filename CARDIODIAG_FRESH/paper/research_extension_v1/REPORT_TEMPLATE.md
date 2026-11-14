# CARDIODIAG: Selection-Aware Evaluation, Calibration and Cross-Cohort Robustness for Retrospective Coronary Artery Disease Discrimination

## Abstract

High cardiovascular classification scores do not establish that model selection is evaluated fairly, that probabilities are calibrated, or that performance transports to another clinical setting. We extend an existing CardioDiag-derived research project through information-matched controls, selection-independent calibration and a new public-cohort transport experiment. The source Z-Alizadeh Sani (ZAS) resource contains 303 source rows with 216 CAD labels. Fifteen paired outer partitions evaluate the historical selection procedure, a 38-feature logistic control, information-group ablations and a corrected three-component calibration ensemble. @ABSTRACT_ZAS@ A separately frozen model using Age, Sex, diabetes, hypertension and current smoking averaged source ROC-AUC 0.8059 and achieved 0.7125 on 377 matched AngioCAD cases; its record-bootstrap interval was 0.6474-0.7730. Target probability accuracy deteriorated, and simulated source-feature deletion reduced mean AUC from 0.8059 to 0.6875 at a 50% cell-mask rate. Endpoint, timing, coding and patient-independence limitations preclude a clinical-generalization claim. The contribution is a reproducible empirical assessment of what drives discrimination and where reliability fails, with preserved historical evidence and independently checked outputs.

## Index Terms

Coronary artery disease, nested validation, calibration, external transport, information availability, reproducibility.

## I. Introduction

Cardiovascular classification projects often begin with a public table, a collection of classifiers and a search for a high accuracy or area under the receiver-operating-characteristic curve (ROC-AUC). A useful research question requires more: the outcome must be defined, the predictors must correspond to a stated information set, selection must be evaluated as part of the training procedure, and the interpretation must match the population studied. A model can rank cases reasonably while providing inaccurate probabilities or losing performance in another setting.

The original CardioDiag paper by Khan, Raza and Nadeem integrates percentile-based preprocessing, localized random affine shadow sampling, explanation-guided hybrid dimensionality reduction and optimized voting/stacking ensembles [1]. It already reports tuning, explainability, calibration, decision curves, subgroup analyses and ablations. Adding another classifier or explanation figure is therefore a weak basis for a subsequent novelty claim. This project instead examines the evaluation and reliability of a CardioDiag-derived research workflow, attributing the original architectures to their authors.

Earlier local work evaluated a feature-group/model-selection procedure on ZAS against a five-feature V3-style reference. Mean repeat AUC was approximately 0.910 versus 0.736. However, the procedures differed in both available information and model selection, while calibration inputs were selected using labels from their later held-out calibration blocks. These observations motivate two distinct controls: a simple model with the same rich feature set, and a comparison in which calibration is separated from the entire selection process.

The present extension also asks whether a compact shared-feature model transports to AngioCAD, a separately released clinical/angiographic resource [4]. The experiment deliberately reports endpoint and spectrum discrepancies rather than treating all public heart-disease labels as equivalent. A further stress test deletes held-out predictor values without refitting the model. These studies evaluate a reusable research procedure; they do not demonstrate a universal cardiovascular diagnostic system.

The contributions are: (i) a paired information-matched logistic control and fixed information-group ablations; (ii) a fully rerun selection procedure within independent calibration-training blocks, with an identical-component raw ensemble control; (iii) source-only development followed by frozen, compatibility-constrained AngioCAD evaluation and endpoint sensitivities; and (iv) versioned predictions, input/output hashes, split traces, tests and a reproducible compiled report. These are applied methodological and empirical contributions, not inventions of nested validation, sigmoid calibration or logistic regression.

## II. Related Work and Research Questions

Cawley and Talbot describe overfitting of finite-sample selection criteria and the resulting bias in performance evaluation [2]. Evaluating the winning configuration after comparing many alternatives on the same data can underestimate selection uncertainty. Nested validation evaluates choices made within training, but it cannot erase earlier researcher adaptation after exploring a dataset. Accordingly, all new ZAS analyses here are explicitly retrospectively motivated follow-ups on previously studied source rows and outer assignments.

Calibration concerns the agreement between predicted probabilities and observed event frequencies, while discrimination concerns ordering [5]. A high AUC is therefore insufficient for reliable individual risk estimates. Model updating may be needed when populations change, but updating with target labels must be distinguished from untouched transport evaluation. We perform no AngioCAD label-based adaptation. External-evaluation guidance also requires attention to the target population, predictor definitions and outcome ascertainment [8]. Validation sample size should reflect the desired precision of performance measures rather than a universal case-count rule [9]. TRIPOD+AI and PROBAST+AI provide reporting and appraisal frameworks [6], [7]; completing an artifact or checklist does not imply low bias or clinical applicability.

Four research questions structure the extension. RQ1 asks whether rich-feature automated selection improves on a fixed 38-feature regularized logistic procedure. RQ2 asks how the calibration operation changes probability accuracy when applied to the same selected base components. RQ3 asks how removing symptom/examination information, or restricting the information set to risk plus symptoms, changes discrimination. RQ4 asks how a new shared-core source model behaves under cohort change and simulated missing inputs. No acceptance criterion requires an AUC increase, and the full locked comparisons are retained regardless of direction.

## III. Data, Provenance and Clinical Scope

### A. ZAS Source Data

The validated ZAS CSV contains 303 source rows and 56 columns, comprising 55 non-target fields and Cath. Cath is mapped as Cad = 1 and Normal = 0, producing 216 positive and 87 negative rows, with prevalence 0.7129. UCI describes CAD using coronary diameter narrowing of at least 50% [3]. The current provider workbook has the same 303-by-56 schema. A fresh official archive download matches the project's raw archive byte for byte; its workbook is also byte-identical, and the first worksheet equals the validated CSV in column order, inferred dtypes and values.

This resolves the question of an unexplained local extra field. The remaining discrepancy is between provider prose describing 54 features and the provider workbook containing 55 non-target fields. An empty second worksheet with 100 header columns is a template, not additional clinical observations. No full-record or all-predictor exact duplicate rows were found. These checks do not establish unique-patient independence: verified identifiers and visit linkage are unavailable. The evaluation unit remains a source row.

### B. AngioCAD Acquisition and Cohort Flow

Only the two small clinical-feature and label workbooks from the official open CC BY 4.0 release were acquired; image archives were unnecessary [4]. Provider byte sizes and MD5 checksums matched, and local SHA-256 hashes were retained. The label table contains 413 unique case IDs, whereas the clinical table contains 377. All 377 clinical IDs match labels; the remaining 36 label cases lack clinical records. The release does not explain that absence in the inspected metadata, so selection associated with missing clinical records cannot be excluded.

The operational positive target is any of 15 coronary segments labelled 51-75, 76-90, 91-99 or 100 percent stenosis. This yields 298 positive and 79 nonpositive matched cases. The available 26-50 category cannot identify exact 50% narrowing, preventing reconstruction of the exact source >=50% boundary. Ten of the 79 nonpositive cases contain that ambiguous category. They are handled by prespecified exclusion and upper-assignment sensitivity analyses; neither assignment is asserted to recover their true source-equivalent labels.

Matched clinical diagnoses comprise 186 STEMI, 20 NSTEMI, 55 unstable-angina and 116 stable-angina records. Nonpositive stenosis cases are therefore not necessarily normal or healthy controls. They represent a different coronary-referral spectrum from a population-screening sample. Case IDs are release keys, not independent verification that no individual has multiple visits. Independent review used a separate spreadsheet parser and confirmed the joins, severity vocabulary and agreement between vessel labels embedded in the feature workbook and the separate label workbook.

@COHORT_TABLE@

### C. Predictor Eligibility and Harmonization

The historical 38-field group contains seven risk fields, ten symptom/examination fields, seven resting-ECG fields and fourteen laboratory fields. The original V3-style comparator uses Age, Sex, Current Smoker, EX-Smoker and FH. EF-TTE, Region RWMA and VHD remain excluded as potentially downstream or diagnostic; Cath and direct angiographic outcome fields are not predictors. None of the retained source fields has source-confirmed collection time before the diagnostic decision. The task is retrospective discrimination rather than prospective early prediction.

AngioCAD lacks the complete V3 and rich symptom/ECG fields. Its new allowlisted core is Age, Sex, DM, HTN and Current Smoker, which replaces V3's former-smoking and family-history fields with diabetes and hypertension. It is a new model, not a transported historical V3 or 38-field fitted model. Age is used as numeric years; Male/Female are explicitly mapped, including the literal ZAS spelling Fmale. Binary 0/1 codes are retained directly without outcome-guided inversion. Accessible target materials lack a complete ascertainment/timing and binary-direction dictionary, so semantic equivalence remains an assumption.

No laboratory units were inferred merely from similar column names. Diagnostic labels, vessel severity, Gensini score, LVEF, procedure-series and medication fields are excluded from the shared model. Age has six missing target entries, and current smoking has 46; the corresponding source fields have none missing. Source-training medians/modes are applied unchanged at evaluation. Three target ages fall outside the observed source age range of 30-86 years. No outcome-dependent complete-case exclusion or target-based imputation is performed.

## IV. Experimental Design

### A. Frozen Source Evaluation

The extension retains the first three repeats of the historical five-fold outer assignments, giving 15 paired partitions and 303 out-of-fold predictions per procedure per repeat. Each outer-training set has approximately 242 rows. Historical three-fold inner partitions use seed 61051 + 100 times repeat + fold. All transformations and hyperparameter selection are confined to their relevant training subsets. Outer labels are excluded from training-function arguments; the new runners drop Cath from input frames before fitting.

The historical pool consists of six feature groups and five unique model families: L2 logistic, elastic-net logistic, spline logistic, random forest and histogram gradient boosting. There are 30 group-family candidates and 120 parameter settings, with deterministic tie order. The installed scikit-learn API and package versions are recorded; an interior l1_ratio with saga is interpreted as elastic net in that API. The original dtype-based preprocessing is retained for comparability. Numeric-coded categories can enter numeric/spline branches, an acknowledged limitation rather than a silently revised algorithm.

### B. Equal-Feature and Information Controls

The information-matched control uses ordinary L2 logistic regression with all 38 fields and C in {0.1, 1, 10}. The exact historical pipeline, inner partitions, post-selection cross-fitted sigmoid and Youden-J threshold construction are retained. This preserves the same calibration limitations deliberately, allowing a more interpretable comparison with saved historical selection results. It is not the independently calibrated arm.

Three additional fixed-family controls use 28 fields after dropping all ten symptom/examination fields; 17 fields comprising risk plus symptoms/examination, excluding ECG and laboratory information; and the five V3 fields with ordinary unweighted L2. Full38 is reused rather than fitted twice. The ordinary five-field ablation is distinguished from historical class-balanced V3. These comparisons address information content under common modeling rules, and do not identify causal effects of individual clinical measurements.

### C. Selection-Independent Calibration Ensemble

For each outer-training set T, the three inner blocks are reused as calibration sets C_k, and D_k is the remaining development subset. The entire 30-candidate search is rerun using new three-fold validation only within D_k. Neither the base estimator nor its group/family/parameter choice uses C_k labels. The selected pipeline refitted on D_k scores C_k, and a fixed weakly regularized sigmoid with C = 1,000,000 is fitted to raw base probabilities there.

The exact fitted D_k base model is retained with its C_k calibrator; it is never swapped for a model subsequently refitted on all of T. Each pair predicts the outer-test rows, and the three calibrated probabilities are averaged. A paired raw ensemble averages the probabilities from those identical base components. Only this latter contrast isolates the calibration operation. Comparing the new ensemble with the historical single-refit procedure also changes development sample size, averaging and repeated selection, so it cannot be described as a pure calibration-effect experiment.

Calibration blocks contain roughly 80 rows. Independence of their labels from component selection does not guarantee precise sigmoid estimation or good probability accuracy. The fixed C is large and supplies weak regularization; it is not evidence of adequate small-sample regularization. New corrected arms use a fixed 0.5 threshold for descriptive sensitivity/specificity, without a screening or clinical-cost interpretation. No calibrator, threshold or family was added after observing outer performance.

@WORKFLOW_FIGURE@

### D. Source-Only Shared-Core Development and Transport

The five-core transport study has its own frozen protocol and preprocessing dictionary. Unbalanced L2 logistic C in {0.1, 1, 10} is selected by three-fold source-training AUC. The same 15 outer source partitions estimate its within-ZAS performance. A second source arm wraps the entire search inside each training block of a three-fold sigmoid calibration ensemble. Each fitted search and its held-block calibrator remain paired. The two transport arms differ in calibration, ensemble averaging and training size, so their contrast is not solely attributable to calibration.

After source evaluation, both procedures are fitted using source rows only. Model files and target probabilities are hashed before target outcomes are scored. Target labels do not select predictors, C, preprocessing, calibration or thresholds, and are not used to update either model. The principal target definition is the recorded >50% rule. Prespecified sensitivities exclude ten boundary-ambiguous cases or assign those ten positive as an upper assignment. All scenarios use the same frozen probabilities.

### E. Missingness Stress and Statistical Summaries

The raw shared-core source pipeline is also evaluated after independent per-cell deletion at rates 0, 0.10, 0.25 and 0.50 in held-out inputs. Masks have deterministic seeds, are independent of outcomes and are saved. The fitted pipeline is not refitted, and training-derived imputers handle the removed values. One fixed mask realization per rate/fold is used; this is a simulated MCAR stress test, not evidence of robustness to actual clinical missing-not-at-random processes.

ROC-AUC, average precision (AP), Brier score, log loss and calibration diagnostics are reported. AP is interpreted relative to event prevalence. Calibration intercept/slope are jointly estimated logistic-regression coefficients on logit probabilities, not an intercept with unit slope fixed. Source metrics are calculated from complete-repeat out-of-fold predictions rather than averages of fold AUCs. All three repeat results and their descriptive means/ranges are retained; overlapping repeats do not provide independent population replicates.

For the fixed transported models, 2,000 record-resampling percentile bootstrap draws summarize target metric uncertainty. They condition on the source model and available target records, excluding source-refitting, patient-cluster and hospital-sampling uncertainty. Bootstrap convergence and usable draw counts are saved. No population-level significance test or independent-patient inference is based on repeated folds. A separately labelled post hoc diagnostic compares target probability loss with a constant equal to source prevalence; it does not select or update any fitted model.

## V. Results

### A. Information-Matched Source Comparisons

@SOURCE_RESULTS_TEXT@

@SOURCE_TABLE@

@SOURCE_FIGURE@

The historical rich-feature selection versus sparse-reference improvement cannot by itself establish a benefit from algorithmic selection. The new rich-feature logistic comparison holds information substantially closer, while the fixed-family ablations isolate larger information-group restrictions. Every reported mean is calculated from three complete 303-row OOF sets. The same rows recur across repeats and cannot be counted as 909 independent participants.

### B. Calibration Construction and Selection Behavior

@CALIBRATION_RESULTS_TEXT@

@SELECTION_RESULTS_TEXT@

The corrected design removes the identified calibration-label dependence on component selection and avoids applying a small-fit map to a larger refit model. Any remaining calibration error is an empirical outcome, not a contradiction of those implementation checks. The new raw/calibrated comparison retains identical selected components, making a negative probability result scientifically informative. No result was substituted into the historical Phase 6 files.

### C. Cross-Cohort Performance and Endpoint Sensitivity

The raw five-core source procedure averaged AUC 0.8059 and Brier 0.1528; the source calibration ensemble averaged AUC 0.8039 and Brier 0.1537. On the principal AngioCAD target, raw-model AUC was 0.7125 with a 0.6474-0.7730 record-bootstrap interval. The calibration ensemble gave AUC 0.7072 with a 0.6453-0.7680 interval. These are different source/target case mixes and partly mismatched endpoints, not paired randomized estimates of a causal setting effect.

@TARGET_TABLE@

Target raw Brier/log loss were 0.1982/0.5787; ensemble values were 0.1937/0.5629. Joint target calibration intercept/slope were 1.105/0.553 for raw and 1.076/0.627 for the ensemble, indicating substantial calibration departure. As post hoc descriptive context, a constant probability of 216/303, fixed by source prevalence, yielded target Brier 0.1717 and log loss 0.5290. Both transported procedures therefore had worse probabilistic loss than that source-constant reference on these cases, despite nontrivial discrimination. This observation must not be hidden by reporting AUC alone.

Excluding the ten ambiguous records raised raw/ensemble AUC to 0.7293/0.7227 on 367 records; assigning all ten positive yielded 0.7259/0.7195 on 377. Those changes concern altered endpoint assignments and sample composition. They neither validate true >=50% labels nor eliminate spectrum, missingness and coding differences. Target AP near 0.894-0.897 must be read against the high target prevalence of 0.7905.

@CALIBRATION_FIGURE@

### D. Missing-Input Stress

@STRESS_TABLE@

Mean AUC declined from 0.8059 without deletion to 0.7758, 0.7522 and 0.6875 at 10%, 25% and 50% masking. Mean Brier increased from 0.1528 to 0.1626, 0.1698 and 0.1867. Imputation enabled a numerical output but did not preserve information or establish reliability. These outcomes apply to the frozen random masks and source cohort; different clinical missingness mechanisms can behave differently.

@STRESS_FIGURE@

## VI. Discussion

### A. What the Extension Establishes

The project advances beyond selecting a visually impressive classifier score by measuring three separate issues: the contribution of richer information, the effect of a properly separated calibration operation, and performance under a new clinical source. The distinction matters because a procedure that performs well on rich retrospective features may not be useful when only a compact set is available. Likewise, a well-separated calibration implementation may still estimate a weak map from a small block, and source calibration need not transport.

@DISCUSSION_SOURCE@

The new cross-cohort experiment is stronger evidence than treating another random seed as external validation. It uses a genuinely separate public resource and source-only fits, while retaining compatibility constraints explicitly. However, it validates neither the historical V3 weights nor the rich-feature ZAS model. Its particular value is empirical evidence of degradation: AUC falls, calibration changes and probability loss is poor relative to a simple post hoc source-prevalence diagnostic. A reliable scientific report includes this failure boundary rather than modifying models until the target score improves.

### B. Relationship to the Base Project

The supplied base paper explicitly states training-fold-only preprocessing and a 75:25 final holdout. Static review of a traceable released notebook identifies specific paths with full-data LoRAS and PCA before a later split, and target-conditioned imputation. Local archive Git-blob hashes match the public release and the inspected notebook matches its archive member. These are concrete source/report discrepancies, but the notebook's execution state and provenance of every published table were not reconstructed; they do not justify declaring every original published result invalid.

The present analysis does not claim to beat the original paper's approximately 0.997 AUC on its different Five States or Mendeley tasks. Nor does it claim to have invented the original OVEC, OSEC or XAIH-DRC components. Corrected architecture-specific head-to-head experiments on identical admissible records would be needed for an original-ensemble superiority claim. The contribution reported here is selection-aware information and reliability evaluation with new clinical-cohort evidence.

The historical four-centre UCI study is also retained as a separate task. Reconstructing a heart-disease cohort or using a new file name does not make it an independent ZAS validator. Existing Heart5/Statlog/UCI amalgams can overlap in provenance. By keeping these tracks separate and acquiring provider-released AngioCAD tables, the extension avoids a misleading combined leaderboard across incompatible labels and reused records.

### C. Research Interface and Reproducibility

A small research scoring interface accompanies the frozen five-core source model. It requires exactly the five declared fields, checks adult age bounds and categorical/binary codes, rejects empty or entirely missing records, and preserves missing values for the source-trained imputers. It reports missing-field counts and age-range flags alongside probabilities, without a diagnosis label. The trusted local model bundle is checked against its prediction-freeze hash before loading. These controls prevent schema mistakes; they do not certify a patient-specific prediction or detect every form of distribution shift.

The full candidate experiment uses one BLAS/OpenMP thread and one search job to control local resources. The first outer partition took 85.07 seconds, projecting roughly 21.3 minutes for the locked 15-partition design. The complete procedure makes 45 corrected component selections, 1,350 group-family searches and approximately 17,550 search/refit fits before logistic controls. @RUNTIME_TEXT@ Immutable component checkpoints retain scores, parameters, calibration labels/scores, model probabilities, traces, warnings and runtimes so that audit reproduction need not refit the entire benchmark.

@QA_TEXT@

The report and its tables/figures are generated from saved metrics, not manually chosen favorable runs. Input/output manifests and protocol hashes link data, implementation and predictions. A pinned environment snapshot and recorded package versions support computational reproducibility; matching hashes do not establish scientific validity, patient identity or clinical permissions. Historical results and manuscript files remain preserved under their original namespaces.

## VII. Limitations and Responsible Interpretation

First, ZAS is small relative to searching many groups and families. The follow-up reuses previously explored rows and outer assignments, so it is not prospectively independent confirmation. Second, source-row independence is unverified in both resources. Unique release IDs and zero exact duplicates cannot establish independent patients or support patient-cluster uncertainty. Third, predictor acquisition timing is unresolved; retrospective symptoms and measurements may encode diagnostic work-up or referral rather than information available at an earlier clinical decision.

Fourth, the historical information controls retain selection-dependent calibration and training-size mismatch for comparability, while the corrected sigmoid maps use small calibration samples and weak regularization. Fifth, the target endpoint is >50% by coarse categories, whereas source documentation states >=50%. Sensitivity assignments expose but do not fix that discrepancy. Sixth, absence of 36 target clinical records, acute/stable case mix, missing smoking/age and incompletely documented binary ascertainment constrain cross-cohort interpretation. No target recalibration was undertaken to conceal these limitations.

Seventh, three overlapping source repeats and fixed-model target record bootstraps cover limited uncertainty components. They do not estimate arbitrary-hospital transport or full development uncertainty. Eighth, simulated MCAR deletion is not a model of every clinical missingness mechanism. Ninth, thresholds have no clinician-defined decision cost, and no clinical-utility, fairness certification or prospective benefit is established. Finally, no novel-classifier priority or superiority to original ensembles is claimed without the relevant matched experiments.

The appropriate intended use is methodological research and retrospective cohort analysis. A deployed clinical system would require a well-defined decision point, source-confirmed predictor timing, independent patient-level evaluation, compatible outcome adjudication, local calibration and a separately justified clinical-use study. The completed computational work provides evidence and failure boundaries, rather than a guarantee of reliable answers for arbitrary datasets.

## VIII. Conclusion and Project Disposition

@CONCLUSION_TEXT@

The new public-cohort experiment demonstrates the limits of source-only transport, and the missingness stress test shows that generating an output after imputation does not preserve predictive information. This makes information-matched controls, probability loss and explicit data compatibility central to the contribution. The computational extension, audit artifacts and compiled report are complete within the stated retrospective scope. Clinical deployment, prospective validation and universal reliability remain unestablished.

## Author, Data-Use and Submission Statements

@AUTHOR_STATEMENT@

The base CardioDiag methods and provider datasets are attributed to their original authors. Source data were used according to the inspected provider records, including AngioCAD CC BY 4.0; institutional ethics determination and any broader release permissions must be confirmed by the project authors. Funding, conflicts and contribution statements must not be inferred from the base paper. AI-assisted coding and drafting were used in preparing this extension; authors must verify the content and complete the chosen venue's current disclosure requirements. This research report is not a claim of acceptance or submission readiness.

## References

[1] M. J. Khan, B. Raza, and A. Nadeem, "CardioDiag: a machine learning approach for cardiovascular disease diagnosis using XAI-guided hybrid dimensionality reduction and optimized-based ensemble classifiers," PeerJ Computer Science, vol. 12, e4059, 2026, doi: 10.7717/peerj-cs.4059.

[2] G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," Journal of Machine Learning Research, vol. 11, pp. 2079-2107, 2010.

[3] R. Alizadehsani, M. Roshanzamir, and Z. Sani, "Z-Alizadeh Sani," UCI Machine Learning Repository, dataset, doi: 10.24432/C5Q31T.

[4] M. S. Hosseini et al., "AngioCAD: A Publicly Available X-ray Angiography Dataset for Coronary Artery Stenosis and CAD Detection," Zenodo, dataset, 2025, doi: 10.5281/zenodo.15826856.

[5] B. Van Calster et al., "Calibration: the Achilles heel of predictive analytics," BMC Medicine, vol. 17, art. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[6] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, e078378, 2024, doi: 10.1136/bmj-2023-078378.

[7] K. G. M. Moons et al., "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," BMJ, vol. 388, e082505, 2025, doi: 10.1136/bmj-2024-082505.

[8] R. D. Riley et al., "Evaluation of clinical prediction models (part 2): how to undertake an external validation study," BMJ, vol. 384, e074820, 2024, doi: 10.1136/bmj-2023-074820.

[9] R. D. Riley et al., "Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study," BMJ, vol. 384, e074821, 2024, doi: 10.1136/bmj-2023-074821.
