# CARDIODIAG: fastest defensible route to a paper

Prepared 14 November 2026. Based on the supplied 61-page Khan et al. base paper, original GitHub inventory, current CARDIODIAG_FRESH code/results, and current official dataset sources. This is a research plan, not a claim that proposed experiments have been completed. Deadline preference: as soon as possible; no hospital collaborator is available.

## 1. Recommended decision

Finish a bounded **retrospective methodological reanalysis** using the existing ZAS evidence. Add one focused, budgeted experiment if time permits: an equal-feature simple-model control. A scientifically stronger second version can add selection-independent calibration and information-group ablation. Keep UCI as a separate centre-held-out study. Do not delay the immediate manuscript for an unverified external cohort or a large model search.

The present results support better retrospective discrimination than the matched five-feature V3-style procedure on evaluated source-row splits. They do not establish that an ensemble, selection algorithm or a new classifier is responsible for the entire improvement. Thirty-eight features versus five is a major information difference. The most valuable comparison is therefore with a simple regularized logistic procedure using those same 38 features and the same evaluation rules.

No amount of tuning can guarantee a reliable answer for every dataset or patient. A reusable training workflow is realistic; a universally calibrated cardiovascular model is not supported. Define population, endpoint, information availability and validation separately for each study.

## 2. Original work versus your actual advance

| Dimension | Original base work | Updated work and remaining question |
|---|---|---|
| Task/data | Five States and Mendeley CVD benchmarks | Primary focus moved to ZAS Cath retrospective CAD; UCI separate |
| Methods | Percentile handling, LoRAS, correlation/LIME-guided feature engineering, optional PCA, OVEC/OSEC, tuning | Pipeline-local preprocessing and selection-aware group/family/C tuning |
| Validation claim | Paper pp.18-19 states 75:25 holdout and five-fold training CV with fold-local preprocessing | Phase 6 evaluates the complete candidate procedure inside 15 paired outer partitions |
| Reliability analyses | Original already reports calibration, decision curves, subgroup analysis and ablations | These alone are not a new contribution; more rigorous construction/validation and information-availability analysis can add value |
| Reproducibility | GitHub archive/notebook release linked on paper p.58 | Structured package, saved predictions, manifests, independent metric reconciliation and documented limitations |
| Novel contribution justified now | Original named architecture belongs to original authors | Your defensible contribution is the selection-aware, traceable retrospective reanalysis and explicit evidence boundaries |

The original paper explicitly claims leakage prevention. Specific supplied notebook cells can be inconsistent with that description; BASE_VS_UPDATED.md distinguishes inspected code from paper assertions. Do not describe the entire original publication as disproven merely from a pipeline diagram. Do not compare its approximately 0.997 AUC on a different task with your 0.910 ZAS AUC as a win/loss experiment.

## 3. Objectives and testable questions

| Objective | Research question | Primary measure | Status |
|---|---|---|---|
| O1: selection-aware retrospective evaluation | How does the whole selection procedure compare with the V3-style procedure on matched source-row partitions? | Complete-repeat OOF ROC-AUC, descriptive paired differences | Completed saved-artifact evidence |
| O2: explain source of improvement | Does selection offer benefit beyond a fixed 38-feature regularized logistic procedure? | AUC and Brier; fit time; feature count | Highest-value proposed follow-up |
| O3: probability reliability | How does a selection-independent calibrated ensemble compare with its identical uncalibrated base ensemble? | Brier primary; log loss, calibration plots/intercept/slope secondary | Proposed; requires new fits |
| O4: information requirements | How much performance is lost without symptoms, laboratories or ECG? | Paired AUC/Brier changes and feature count | Proposed; freeze a small set of ablations |
| O5: setting sensitivity | How does a separately trained procedure behave across UCI centres and missingness patterns? | Centre-specific AUC/Brier/log loss; pooled metrics secondary | Historical UCI exists; follow-up optional |

These are applied methodological objectives, not first-ever algorithm claims. Nesting, sigmoid calibration and ablation are established techniques. Novelty must come from a well-specified question, an informative result and a reproducible artifact, supported by a wider literature review before submission.

## 4. Clinical datasets: what to use now

Use authoritative downloads and preserve original files with access date, license and checksum. See dataset_and_literature_review.md for the verified online search and source links.

| Priority | Resource | Immediate use | Important boundary |
|---|---|---|---|
| 1 | UCI Z-Alizadeh Sani | Main retrospective CAD analysis; existing 303-row local file | Source schema/row lineage reconciliation still required; no predictor timestamps or verified unique IDs |
| 1 | Official UCI Heart Disease four-centre records | Separate setting-variation/centre-held-out evaluation | Not external validation of a fitted ZAS model; old clinical data and missingness |
| 2 | Mendeley Cardiovascular Disease Dataset dzz48mvjht/1 | Reproduce original task or independent workflow replication with its own target | Source-defined CVD label is not automatically angiographic Cath; downstream vessel information must be excluded for an early-information analysis |
| 2: acquired and screened | AngioCAD, paper 2026, DOI 10.1016/j.cmpb.2026.109368; data DOI 10.5281/zenodo.15826856 | Two small open clinical/label workbooks downloaded and checksummed; 377 matched case IDs, 298 >50% cases and 79 without >50% stenosis | Exact current ZAS model cannot transfer: missing predictors, threshold boundary and different referral spectrum; no predictive evaluation yet |
| Not a new external cohort | ZAS extension, Statlog or a merged Heart5 CSV | Supplementary same-source work only after overlap audit | Potential same patients/source overlap; a new filename is not independence |
| Reject for this immediate task | All-positive 100-person stenosis image dataset, image segmentation sets, Framingham teaching data | Do not use as CAD-vs-normal tabular validation | Incompatible controls/modalities or explicit publication restrictions |

Finding real clinical data online is feasible. Finding an independent dataset with all 38 compatible ZAS predictors, identical Cath ascertainment and timing is much harder. If none passes screening, write that external validation was not available; do not substitute retraining on a different outcome and call it external validation.

**Concrete new opportunity:** AngioCAD's official release has 413 label IDs but only 377 clinical-feature IDs, all of which join. Its categories yield 298 cases with a segment >50% and 79 without one. Ten of those 79 contain a 26-50 segment, so exact >=50% status is unknown. These are not 79 healthy controls. The feature sheet includes 186 STEMI, 20 NSTEMI, 55 unstable-angina and 116 stable-angina diagnoses: a very different referral spectrum. It lacks FH, EX-Smoker and the ZAS symptom/ECG groups. A **new fixed shared-feature cross-cohort study** is possible after units/timing/endpoint review, but not unchanged validation of the historical V3/38-field models. Raw workbooks, provider checksums, screening code and aggregate findings are preserved in data_screening/angiocad/. No image archive or new model result was produced.

For a stronger subsequent paper, the highest-value option is to freeze a compact shared clinical feature set using source definitions, develop it on ZAS, and assess it on AngioCAD with the endpoint mismatch explicit and boundary cases handled by prespecified sensitivity assignments. Keep AngioCAD model outcomes out of feature/model selection. A reciprocal cohort experiment is optional and separate; each direction needs its own locked development procedure. Existing exposure to aggregate metadata/label counts must be disclosed; it is not a prospectively untouched study.

### Dataset acceptance checklist

Record the originating institution/cohort, dates, source/version, permitted use, patient/study identifier, repeated visits, outcome definition and adjudication, positive/negative counts, recruitment/referral setting, predictor units/codes, missingness, collection timing, and exclusions. Compare raw predictor fingerprints across public resources before assigning independent cohort labels. A whole-row duplicate check including IDs or labels alone is insufficient.

For true external validation, first freeze a source model and transformations; verify every required field and target independently; apply it without tuning on target outcomes. If features must change, it is a newly trained harmonized model. If target labels are used for adaptation, reserve disjoint adaptation/evaluation subsets and report adapted performance separately. No pooled leaderboard across incompatible endpoints.

## 5. Modeling and fine-tuning plan

### Immediate version: no new experiment required

Retain the existing Phase 6 numbers and all limitations. Lead with matched repeated OOF discrimination and the reproducibility audit. The strongest conclusion is descriptive and dataset-specific. The supplied IEEE-style draft uses only saved evidence; new experiments appear only in this plan/future work.

### Minimal useful follow-up: equal-feature control

Freeze a new dated protocol under a new namespace. Compare the historical selection procedure with L2 logistic regression using all 38 candidate fields, fitted with the same fold-local preprocessing and a prespecified C grid. Use matched outer partitions and explicitly label this as a retrospectively motivated follow-up. Report uncalibrated scores/probabilities and, if calibration is compared, give both arms the same valid construction. Do not mix calibrated and uncalibrated arms and attribute changes solely to model selection. Explain whether reproducing a historical arm is required for a genuinely matched implementation.

The current selection winner was L2 logistic in 9/15 outer fits. If the fixed rich-feature logistic control performs similarly, that is an informative simplification result. AUC gain is not required for a useful paper. It would change the interpretation from sophisticated algorithm benefit to information availability plus regularization.

If claiming an advance over the original architecture, add a precisely specified **corrected OVEC/OSEC comparator on the same admissible records, features and outer partitions**. Keep resampling/feature selection within training and construct stacking meta-features out of fold. A faithful historical reproduction belongs in a separate track. This comparator connects the paper directly to the assigned base work, but is secondary to the simple equal-feature control under the urgent deadline. Do not claim that ZAS-versus-V3 alone proves superiority to original OVEC/OSEC.

### Stronger probability study: selection-independent ensemble

Within each outer training set T, make three stratified calibration blocks C_k. For each k:

1. Let D_k = T minus C_k. Use only D_k to select feature group, family and hyperparameters with inner CV. Fit all transformations there.
2. Fit the winning base model on D_k and obtain scores for C_k. Fit a prespecified regularized sigmoid using only those held-block scores/labels.
3. Retain that fitted model/calibrator pair; do not refit its base model on C_k after calibrator fitting.
4. Apply each pair to the outer test set and average the three calibrated probabilities. Compare with the average raw probabilities from the same base components to isolate calibration's effect.

This makes each component's calibration labels independent of its model selection and preserves the model/calibrator pairing. It evaluates an ensemble training procedure, not a single final full-data model. About 80 records per calibration block is small; report instability rather than promise precise individual risk.

Freeze sigmoid regularization without choosing it from outer results. Prefer Brier as the primary probability endpoint. Avoid isotonic calibrator hunting on this sample. Either omit operational cutoffs or use a fixed 0.5 rule for descriptive reporting only. A data-selected cutoff needs a further training-only selection design; Youden J does not establish a clinically useful screening threshold.

### Small, meaningful ablation menu

Use the same fitting/tuning/calibration procedure for fixed information sets: all 38; without symptom fields; risk plus symptoms/examination without ECG/laboratory; and the five-feature comparator. Keep the number fixed before fitting. Do not call these prediagnostic groups without source timing evidence. Measure loss of information, not just score maximization. Typical Chest Pain importance motivates this investigation but is not itself evidence of causation.

### Dataset-specific tuning policy

Use a semantic dictionary for continuous, binary, nominal and ordinal fields; numeric dtype is not a clinical definition. Fit imputation/encoding/scaling/splines inside training. Remove IDs, target-derived fields and post-decision variables according to the stated task. Start with logistic L2, elastic net and one nonlinear model only if justified. Add CatBoost or a resampling method as a prespecified comparator rather than an unlimited search. Do not presume SMOTE/LoRAS improves probability estimates; preserve test prevalence and evaluate unresampled baselines first. Pin the actual scikit-learn API and log convergence warnings.

## 6. Validation, uncertainty and reliability

Patient-group splitting is mandatory when reliable patient IDs exist; otherwise explicitly report source-row splitting. Site-held-out evaluation and random row CV answer different questions. Newly generated seeds on a previously analyzed dataset do not restore an untouched validation cohort.

Report complete-repeat OOF AUC with its prevalence, AP, Brier, log loss, calibration plots, intercept/slope, feature count and runtime. Report each UCI centre separately; pooled AUC can reflect between-centre differences. The three-repeat t interval is descriptive split variation, not a population confidence interval or significance test. Do not treat 15 overlapping folds as 15 independent trials. Bootstrapping saved OOF rows conditions on fitted predictions and does not include full training uncertainty; patient-cluster intervals require known IDs, and four-centre intervals cannot support arbitrary-hospital transport claims.

Missingness stress tests should use frozen masks/rates and training-fold-only imputers. They are simulated perturbations, not new clinical validation. A small sex/age subgroup table may be descriptive where class counts permit; sparse strata do not justify fairness certification. Explainability is descriptive, not causal, and its stability should be evaluated on development data if used to select features.

A research prediction interface should reject missing required fields, unknown units, invalid categories and a mismatched task/schema; display the study scope and evidence limits. A calibrated probability is not a confidence guarantee for one person. Optional conformal sets/abstention need independent calibration and an exchangeability argument; standard coverage does not automatically survive hospital shift or guarantee clinical safety.

## 7. Execution sequence and compute limits

Times below are planning ranges, not measured promises; source access and author responses can extend them.

| Stage | Planning effort | Deliverable and stopping rule |
|---|---|---|
| A: evidence freeze and source reconciliation | 0.5-1 working day | Column/source map, endpoint definitions, duplicate/overlap checks, manuscript result register; retain unresolved facts explicitly |
| B: paper from current evidence | 1-2 working days including author review | IEEE draft, traceable tables/figures, declarations; no invented new results |
| C: optional equal-feature control | Pilot then budget decision | One-partition runtime/RAM log, frozen full-run cost; run all planned partitions regardless of score |
| D: optional calibration + ablation | Schedule only after pilot | New versioned predictions, paired metrics, negative findings, manifest; no changes to Phase 6 outputs |
| E: final formatting and checks | 0.5-1 working day after content freeze | Publication-specific template, citations, author approvals, fresh page inspection, release manifest |

The full current candidate pool has 120 hyperparameter settings across six groups; three-fold searches plus per-candidate refits cost about 390 fits per selection. Three calibration components times 15 outer partitions is approximately 17,550 search/refit operations before comparators or ablations. This is a counting estimate, not a wall-clock measurement. Do not launch it merely because it sounds more rigorous. Pilot a representative partition; if too expensive, freeze a reduced logistic-focused follow-up and label it as retrospectively motivated. Log a runtime/RAM cutoff before seeing outcomes. Reaching a score target must never be the stopping rule.

## 8. Paper structure and claims

Working title: **Selection-Aware Retrospective Discrimination of Angiography-Defined Coronary Artery Disease: A Reproducible Reanalysis**.

The new IEEE-style draft has abstract/index terms; introduction and explicit relation to Khan et al.; related methodological work; source data/feature policy; nested evaluation/calibration limitation; saved results; discussion; limitations; conclusion; and numbered references. Keep UCI secondary or supplementary if space is tight. Do not mix its results with ZAS in an unexplained ranking table.

| Permitted now | Requires new evidence |
|---|---|
| Higher observed repeat-level discrimination than matched V3-style comparator | A superior new classifier; superiority over the base paper on the same task |
| Saved-prediction and hash reconciliation | Independent reproduction of the entire fitted benchmark |
| Retrospective source-row discrimination | Prospective early detection, verified patient-level performance |
| Calibration construction is limited | Reliably calibrated individual probabilities |
| Separate UCI centre-held-out analysis | External validation of fixed ZAS weights |
| An IEEE-style review draft | IEEE acceptance, publication readiness or publication guarantee |

The updated project is an advance derived from CardioDiag, not the original invention of OVEC/OSEC/XAIH-DRC. Cite the base work and clearly attribute reused methods. Public GitHub metadata has no detected license; public visibility alone does not grant an unrestricted code-relicensing right. Resolve reuse/distribution permissions before release. Base-paper ethics/funding/author declarations are not yours and must not be copied.

## 9. Final human items

Confirm all author names and order (including Akhil), affiliations, corresponding author, contributions, ethics determination, data-use compliance, funding, conflicts and any institutional wording. Select the actual venue before final template export. Complete the TRIPOD+AI checklist and PROBAST+AI assessment item by item; completion of a checklist is not proof of low bias. For an IEEE submission, review its current AI-assisted-text disclosure requirements and accurately disclose assistance used in manuscript preparation.

## 10. Package and evidence boundaries

This directory contains the source comparison, dataset search, methodology review, a numbered-reference IEEE-style manuscript draft, LaTeX source, review PDF, numeric traceability and build/QA records. New source discovery does not retroactively create experiments. All proposed results remain absent until generated and checked. Historical CARDIODIAG_FRESH outputs and manuscripts remain unchanged.

Primary methodology: Cawley and Talbot, JMLR 2010, https://jmlr.org/papers/v11/cawley10a.html. Reporting: https://doi.org/10.1136/bmj-2023-078378 and https://doi.org/10.1136/bmj-2024-082505. IEEE tools/templates: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/.
