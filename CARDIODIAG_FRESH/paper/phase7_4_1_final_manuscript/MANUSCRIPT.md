# Selection-aware retrospective discrimination of angiography-defined coronary artery disease in the Z-Alizadeh Sani dataset: a methodological reanalysis with separate UCI centre-held-out analyses

**Author order and full names to be confirmed:** Ch Hari Krishna; G Vennela  
**Affiliations:** [To be completed]  
**Corresponding author:** [To be confirmed]

## Structured abstract

**Background:** Small public coronary artery disease (CAD) datasets are frequently used to benchmark classifiers. When a feature group and model are selected after comparing candidates, performance of the winning row is optimistic unless selection is included in the evaluation. In diagnostic datasets, the recorded predictors may also have been collected at different points in the diagnostic pathway.

**Objective:** To estimate, on the Z-Alizadeh Sani (ZAS) dataset, the retrospective discrimination of a prespecified feature-group/model-selection procedure against a matched re-evaluation of the frozen V3 five-feature reference; and to summarize a separate four-centre UCI Heart Disease staged evaluation without treating it as external validation of ZAS.

**Methods:** We analyzed 303 ZAS source rows with an angiography-defined binary Cath outcome (216 Cad, 87 Normal). Three repeats of five frozen outer folds were used. Within each outer training set, three-fold inner validation selected among six feature groups and five unique model families; hyperparameter tuning and all preprocessing were contained in training partitions. A paired V3-style comparator used the same outer partitions, fixed V3 features and balanced logistic-regression family, with its regularization parameter selected inside outer training. Repeat-level metrics were computed from complete 303-row out-of-fold predictions. UCI analyses were kept separate and summarized from four-centre leave-one-centre-out saved predictions.

**Results:** The selection-aware procedure had repeat-level ROC-AUCs of 0.9134, 0.8999 and 0.9176, compared with 0.7370, 0.7420 and 0.7300 for the V3-style comparator. Paired differences were +0.1764, +0.1579 and +0.1877 (mean +0.1740; descriptive split-variation t interval +0.1367 to +0.2113). These three repeats reuse source rows and are not independent patient samples. The selected group was eligible_combined in 15/15 outer fits; selected model families were L2 logistic 9/15, elastic-net logistic 4/15 and spline logistic 2/15. Calibration cross-fit scores were selection-dependent. In the separate UCI cohort (920 records), pooled leave-one-centre-out ROC-AUC ranged from 0.7179 at Stage 1 to 0.8450 at Stage 4; Stage 2 was 0.8225. Stage 4 includes exercise-test features and is a distinct later task. Switzerland contributed 115 positive and only 8 negative records.

**Conclusions:** On the evaluated ZAS source-row partitions, the selection-aware procedure showed higher retrospective discrimination than the matched V3-style comparator. This is not patient-level or population-level inference. Predictor availability before the diagnostic decision is unconfirmed; the data therefore support retrospective discrimination only, not prospective prediagnostic prediction. UCI results are a separate multicentre task and do not validate ZAS externally.

**Keywords:** coronary artery disease; retrospective discrimination; model selection; nested cross-validation; feature timing; calibration; reproducibility

## Introduction

Coronary artery disease (CAD) classification has been studied using public tabular datasets that combine demographic, symptom, examination, electrocardiographic, laboratory and diagnostic information. The ZAS resource contains 303 records and an angiography-defined CAD label; its original report described multiple data-mining approaches and highlighted features that include typical chest pain, age and angiographic or echocardiographic measurements [1,2]. The public data are useful for method comparison, but a high score on such a small dataset does not by itself establish performance at an earlier clinical decision point. A dataset label may be confirmed by a diagnostic procedure while predictor collection times are undocumented.

A second problem is model-selection optimism. Optimizing a validation score over model families, feature groups or hyperparameters can select a candidate that benefited from noise in the same finite validation data [3]. Nested evaluation is intended to evaluate the complete selection procedure by reserving outer partitions for evaluation. A related reporting issue is that discrimination and calibration answer different questions: rank ordering can be good while probabilities are poorly calibrated, and calibration estimates are themselves sample-size sensitive [4]. TRIPOD+AI and PROBAST+AI provide current frameworks for reporting and appraising prediction-model studies; they improve transparency but do not make a design valid by themselves [5,6].

This study is not a new classifier proposal and does not claim to be the first machine-learning study on ZAS. Its contribution is a traceable reanalysis of a historical public-data benchmark: (1) it separates an exploratory, post-hoc Phase 5 winner from the selection-aware Phase 6 estimate; (2) it evaluates a frozen candidate-selection procedure on the same outer source-row partitions as a re-evaluated V3-style five-feature comparator; (3) it independently reconciles the saved predictions and metrics; and (4) it treats feature timing and calibration construction as central interpretive constraints. A separate UCI Heart Disease track is reported only as a four-centre staged evaluation, not as validation of ZAS. The central question is: **On the evaluated ZAS source-row partitions, did the prespecified selection-aware procedure discriminate angiography-defined CAD better than a matched re-evaluation of the V3-style reference?**

## Related work

The original ZAS publication introduced a 303-patient CAD dataset and reported classification experiments, including feature relevance analyses [1]. The current UCI repository describes the endpoint as CAD when coronary diameter narrowing is at least 50% and lists 54 feature variables, whereas the local validated project file contains 56 columns including the outcome; this schema difference is retained as unresolved rather than silently harmonized [2]. Prior high-accuracy reports using the dataset are not directly comparable unless they share the endpoint, feature timing, splits, model-selection process and metric definitions.

The UCI Heart Disease collection comprises Cleveland, Hungary, Switzerland and VA Long Beach sources. The repository documents 76 original attributes, a commonly used 14-attribute subset, missing values, and `num` values 0–4, often dichotomized as absence versus presence [7,9]. A four-centre evaluation can describe performance across those assembled centres, but it does not become external validation of another dataset with different patients, predictors, missingness and endpoint coding.

Methodologically, Cawley and Talbot describe the selection bias that arises from optimizing finite-sample model-selection criteria [3]. Calibration is a distinct performance dimension and should not be inferred from discrimination alone [4]. TRIPOD+AI recommends complete, transparent reporting of model development/evaluation and open-science declarations; PROBAST+AI distinguishes methodological quality from risk of bias and applicability [5,6]. We use these sources to motivate transparent evaluation and restrained interpretation, not to claim that checklist compliance validates the model. The novelty is therefore a specific reproducibility and selection-aware reanalysis of the project’s own ZAS benchmark, not a claim of algorithmic novelty or a comprehensive review of all CAD machine-learning literature.

## Materials and methods

### Design and research question

This is a retrospective, repository-based analysis of saved machine-learning predictions and a selection-aware nested validation experiment already completed in Phase 6. The present manuscript reconstructs and interprets those results; no new model fits are reported. The question is retrospective discrimination of angiography-defined CAD using recorded clinical features. The unit of evaluation is a source row because unique patient identifiers and patient-level independence cannot be verified.

### Data sources and outcomes

**ZAS.** The local validated file contains 303 rows. `Cath` was mapped as Cad=1 and Normal=0, yielding 216 positive and 87 negative source rows (prevalence 0.7129). The UCI dataset description defines CAD as diameter narrowing ≥50% [2]. The local dataset has 56 columns including outcome, while UCI describes 54 features; the project did not resolve this discrepancy. The data contain no verified patient ID or row-level predictor acquisition timestamps. Phase 6 split by source-row index, not by confirmed patient.

**UCI Heart Disease.** A separately assembled cohort included 920 records from Cleveland (n=303), Hungary (n=294), Switzerland (n=123) and VA Long Beach (n=200). The outcome was `num=0` versus `num=1–4`; these codes represent no disease versus disease in the original collection [7,9]. There were 509 positive and 411 negative records overall. Positive counts by centre were Cleveland 139, Hungary 106, Switzerland 115 and VA Long Beach 149. UCI is not pooled with ZAS and is not external validation of the ZAS analysis.

### Feature sets and timing

The frozen V3 comparator uses Age, Sex, Current Smoker, EX-Smoker and FH. The Phase 6 candidate universe included six fixed groups: V3 five predictors; basic risk (Age, Sex, DM, HTN, smoking, family history); symptoms/examination; resting ECG; laboratory; and eligible_combined (the 38-field union of basic risk, symptoms/examination, ECG and laboratory). Exact field lists are in Table 2 and the Phase 6 protocol. Logistic and Ridge labels in Phase 5 were aliases for the same L2 logistic implementation; Phase 6 retained one L2 logistic family, alongside elastic-net logistic, spline logistic, random forest and histogram gradient boosting.

No ZAS feature has source-confirmed collection time before the diagnostic decision. Although age and sex are inherently knowable and the earlier Phase 5.1 audit called them available before the Cath outcome, the record time is undocumented, so this manuscript classifies them as timing uncertain for the stricter decision-time question. Symptoms, examination, ECG and laboratory measurements are also timing uncertain. EF-TTE, Region RWMA and VHD were excluded as potentially downstream/diagnostic fields; Cath and angiographic fields were outcome/direct outcome information and were excluded. “Potentially downstream” denotes risk, not proof of the actual time of collection.

UCI stages were: Stage 1 age/sex; Stage 2 plus chest-pain category; Stage 3 plus resting blood pressure, cholesterol, fasting-blood-sugar indicator and resting ECG; Stage 4 plus exercise maximum heart rate, exercise-induced angina, exercise ST depression and slope. UCI `ca`, `thal` and `num` were excluded from predictor sets. Stage names encode intended information sets but do not establish measurement timestamps.

### Phase 6 selection-aware ZAS evaluation

The candidate pool was fixed before outer evaluation: six feature groups × five unique model families = 30 group-family candidates. Hyperparameter grids were those in `runner/phase6_zas_selection/run.py`; the previously observed Phase 5 AUC of 0.928 was not used to define or expand the pool. Six group names included V3, basic risk, symptom/examination, resting ECG, laboratory and eligible_combined. Model families were L2 logistic regression, elastic-net logistic regression, spline logistic regression, random forest and histogram gradient boosting.

The full planned five repeats of five outer folds was reduced to the first three repeats of the frozen V3/Stage 6 split manifest after the full plan exceeded 35 CPU minutes without producing artifacts. This yielded 15 outer partitions. The same outer partitions were used for both procedures. Within each outer-training partition, deterministic stratified three-fold inner validation selected each candidate’s hyperparameters by ROC-AUC and then selected the group/family with the highest inner score; ties followed fixed candidate order. The V3-style comparator retained its five features, balanced logistic family and class weighting, while regularization C was selected within outer training. Thus this is a re-evaluated procedure comparator, not the untouched original fitted V3 model, calibrator or threshold.

All imputation, one-hot encoding, spline expansion and scaling were included in sklearn pipelines fit only on training partitions. After candidate selection, selected-model inner cross-fit raw scores were generated, sigmoid calibration was fit on those scores, and a Youden-J threshold was chosen on the calibrated training scores. The selected model was then refit on all outer-training records and evaluated on the outer test fold. Outer-test labels did not drive feature-group/model/hyperparameter selection, calibration fitting or threshold selection. After predictions were frozen, outer-test features/labels were used for descriptive permutation importance.

Calibration scores are limited: hyperparameters and feature group had already been selected using labels from all inner folds, including those later used as calibration cross-fit blocks. Scores are out-of-fold for estimator fitting but selection-dependent. In addition, inner score models used about two-thirds of outer training, while the final model used all outer-training rows. This limitation affects calibration interpretation; it does not create outer-test label leakage or invalidate the outer discrimination evaluation. Phase 6.1 found four calibration-fold metadata entries where the actual design had three; this was reporting-only and did not affect predictions.

### UCI four-centre analysis

UCI was reconstructed from four official source files. The staged procedure fit a logistic-regression pipeline separately for each stage and held-out centre; regularization and class weight were selected by four-fold stratified validation in the other three centres. Imputation, missingness indicators, encoding and scaling were fitted within development-centre training folds. The Phase 4.3 independent audit reconstructed 920 records and checked 3,680 saved stage predictions, keys, centre assignment, labels and metrics. We report pooled saved prediction metrics and centre-specific ROC-AUC. Stage changes are paired by `(held_out_centre, source_row)`. The reported paired intervals use 400 paired record bootstrap draws with replacement; they do not resample centres or patients and do not include model refitting uncertainty.

### Metrics and uncertainty

We report ROC-AUC for ranking, average precision (AP) for the precision-recall summary, Brier score and log loss for probabilistic accuracy, plus sensitivity and specificity at training-selected thresholds for ZAS. ZAS AP is interpreted against prevalence 0.7129; AP is prevalence-sensitive and must not be compared across different cohorts without context [8]. Calibration intercept and slope are descriptive regressions on outer-test predictions and are unstable in small folds. A higher AUC does not establish clinical benefit.

For each Phase 6 repeat, metrics were recalculated from its complete 303-row out-of-fold prediction set, not by averaging fold AUCs. The mean and range over three repeats describe only the evaluated split variation. The paired t interval over three repeat-level differences is labeled descriptive split-variation; repeats reuse rows and overlapping training sets, so this is not a patient-level or population confidence interval. No significance test is reported. UCI paired percentile intervals are explicitly record-bootstrap intervals conditional on the four observed centres.

### Reproducibility

The Phase 6 and Phase 6.1 source code, protocol, saved split manifest, predictions, metric tables and manifests are retained in the repository. Phase 6.1 independently reconstructed all 1,818 predictions, fold/repeat metrics and paired repeat differences and verified listed manifest hashes. The documented reproduction environment is the project-local `.venv` (Python 3.11.16, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1; pytest 9.2.0, Ruff 0.17.4). The Phase 6 model run is not repeated for this manuscript. Exact non-training audit commands are in the supplement/manuscript package; unpinned requirement files and absent Git metadata limit environment provenance.

## Results

### Cohort and feature eligibility

ZAS included 303 source rows (216 Cad, 87 Normal; prevalence 71.3%). Patient-level independence is unverified. The local file schema differs from the UCI feature count description and was not silently reconciled. UCI included 920 records (509 positive) across four centres; Switzerland was highly enriched for the positive label (115/123, 93.5%; eight negatives). Table 1 summarizes cohorts. No predictor was confirmed available before the diagnostic decision. Table 2 distinguishes uncertain predictors from excluded diagnostic/downstream-risk fields.

### Primary Phase 6 ZAS comparison

Across three complete-repeat OOF sets, selection-aware ROC-AUC values were 0.9134, 0.8999 and 0.9176; the V3-style comparator values were 0.7370, 0.7420 and 0.7300. Paired differences were +0.1764, +0.1579 and +0.1877. Their mean was +0.1740; the descriptive split-variation 95% t interval was +0.1367 to +0.2113. All three repeat estimates favored the selection-aware procedure, but these repeats are not independent patient samples and do not support a significance claim.

Other complete-repeat metrics were also better for the selection-aware procedure in this evaluation: mean AP 0.9573 vs 0.8469; mean Brier 0.1076 vs 0.1746; mean log loss 0.3467 vs 0.5286. ZAS event prevalence was 0.7129, which is the AP prevalence reference. Threshold-based mean sensitivity/specificity were 0.8318/0.8314 vs 0.6898/0.7088; thresholds were selected within outer training and have no clinical interpretation. Calibration intercept/slope means were 0.173/0.868 for selection-aware and −0.007/1.005 for comparator; these small-sample descriptive metrics do not overcome selection-dependent calibration construction. Full repeat values are in Table 3.

The selected feature group was eligible_combined in 15/15 outer fits. Family selection was L2 logistic 9/15, elastic-net 4/15, spline logistic 2/15. The group count does not establish that every included variable is necessary. Typical Chest Pain had the highest mean marginal permutation AUC decrease (0.125; SD 0.050), followed by Age (0.042; SD 0.017) and DM (0.019; SD 0.011). These outer-test permutation results are descriptive, not causal, and may be affected by correlated features or symptom ascertainment. An Age/Sex-only timing-uncertain sensitivity had mean repeat AUC about 0.729 (range 0.726–0.731), below the selected procedure; it is not a timing-confirmed comparator.

### Exploratory Phase 5 result

The Phase 5 eligible_combined Logistic/Ridge result reconstructed to ROC-AUC 0.928214, AP 0.966830, Brier 0.096424 and log loss 0.314108 on 303 source rows (216 events). It was selected post hoc from 36 benchmark rows and Logistic/Ridge were identical estimator aliases. This result is exploratory, not an unbiased final performance estimate; it is separate from and was not used to form the Phase 6 candidate pool.

### Separate UCI four-centre results

Pooled UCI leave-one-centre-out ROC-AUC/AP/Brier/log loss were: Stage 1 0.7179/0.7293/0.2112/0.6117; Stage 2 0.8225/0.8241/0.1681/0.5106; Stage 3 0.8215/0.8191/0.1684/0.5139; Stage 4 0.8450/0.8645/0.1584/0.4861. These are UCI cohort metrics and are not compared numerically as validation of ZAS. Stage 2 and Stage 4 are distinct intended information-time tasks.

Paired Stage 3-to-Stage 4 differences were: ROC-AUC +0.02353 (95% paired record-bootstrap interval +0.00608 to +0.04426); AP +0.04538 (+0.01960 to +0.07062); Brier −0.00994 (−0.01861 to −0.00151); log loss −0.02784 (−0.05139 to −0.00562). These intervals resample records from the four observed centres, not independent patients or centres. Stage 2-to-3 AUC changed by approximately −0.001; no clinical significance interpretation is attached. UCI centre-specific AUCs varied: Stage 1 ranged 0.501–0.710, Stage 2 0.710–0.838, Stage 3 0.679–0.819, and Stage 4 0.726–0.875. Switzerland’s eight negative records make its metrics particularly unstable (Figure 3).

## Discussion

### Principal findings

The Phase 6 selection-aware procedure had greater complete-repeat OOF discrimination than the matched V3-style comparator on all three evaluated source-row repeats. The procedure selected the same broad combined feature group at all 15 outer fits, although model family varied. Phase 6.1 independently reconciled the predictions and metrics. This is stronger evidence than reporting the Phase 5 winning row alone because the feature-group/model choice was repeated within outer training folds. It remains an internal, small-sample reanalysis of one 303-row dataset.

The apparent advantage should be interpreted as a property of the entire candidate-selection procedure on these partitions, not as a validated final 38-feature model. The observed high AP is partly conditioned on the high ZAS event prevalence (71.3%). Although Brier and log loss were lower in the selection procedure, calibration construction is selection-dependent and fit-size-mismatched. Better probability scores here do not establish reliable individual risk estimates.

### Competing explanations and clinical meaning

The selected group contains symptoms, examination, ECG and laboratory variables whose acquisition times relative to referral and angiography are undocumented. The feature with the largest marginal permutation importance was Typical Chest Pain. This could represent clinically relevant symptom information, but may also reflect symptom ascertainment among patients selected for angiography, referral spectrum or diagnostic work-up. Because timestamps and source workflow are absent, this analysis cannot separate those explanations. No feature is asserted to have been available before the decision to perform diagnostic testing.

The V3 comparator is matched by outer partitions and uses the same Phase 6 evaluation structure, but it is a re-evaluated V3-style procedure with inner C selection and recalibration, not the original V3 fitted equation. The paired difference therefore compares training procedures, not only feature count or a frozen deployed model. Three repeat differences all favor the selection-aware procedure, but source rows recur across repeats and no patient-level independent sample is established. The reported t interval is not a conventional population inference interval.

### Separate UCI interpretation

UCI Stage 2 has higher pooled ROC-AUC than Stage 1, while Stage 3 is essentially unchanged relative to Stage 2; Stage 4 has higher pooled discrimination but includes exercise-test information. These stages should not be presented as equivalent alternatives at one decision time. Centre variation and missingness matter, and the high-positive Switzerland centre contains only eight negatives. Leave-one-centre-out evaluation demonstrates performance on held-out centres within this assembled historical dataset; four centres do not estimate transport to arbitrary hospitals. The UCI endpoint, predictors and collection process differ from ZAS and cannot externally validate the ZAS procedure.

### Strengths

The analysis retains a prespecified candidate pool, evaluates group/model selection within outer training partitions, keeps preprocessing in pipelines, compares procedures on matched outer partitions, reports complete-repeat OOF metrics rather than conflating them with fold averages, and independently reconciles saved predictions. Separate explicit timing and leakage audits prevent downstream diagnostic fields from being treated as eligible. UCI is kept as a separate track.

## Limitations

First, 303 ZAS rows are small for selecting among 30 group/family combinations and several hyperparameters. Although outer evaluation addresses optimism from this fixed selection procedure, estimates remain variable and were completed over only three repeats after the full run exceeded the available computational budget. Second, identifiers cannot verify unique-patient independence or permit patient-level uncertainty. Third, no acquisition times are available; retrospective features may encode work-up or referral. Fourth, Phase 6 calibration crossfit scores are selection-dependent, inner model training size differs from final refit, and thresholds use Youden J without clinical costs. Fifth, the split-variation interval uses three overlapping repeats and is descriptive only. Sixth, Phase 5’s selected 0.928 result remains post-hoc. Seventh, UCI record bootstraps do not resample centres or verified patients; Switzerland has eight negative cases. Finally, the local ZAS column count conflicts with repository documentation, code requirements files are not fully pinned, Git metadata is unavailable, and independent external ZAS validation is absent.

## Conclusion

On the evaluated ZAS source-row partitions, a nested, selection-aware feature-group/model procedure had higher repeat-level out-of-fold ROC-AUC than a matched re-evaluated V3-style comparator in each of three repeats. The result supports only a cautious retrospective discrimination statement. It does not establish patient-level superiority, prospective or prediagnostic prediction, calibrated individual risk, clinical utility, transportability or external validation. Predictor timing remains unresolved. UCI results are a separate four-centre staged evaluation and must not be described as external validation of ZAS.

## Ethics statement

**[Author action required]** This work reanalyzes public, de-identified repository datasets and saved predictions. The authors must confirm institutional ethics review or exemption status, applicable data-use terms, and whether any additional institutional determination is required. No approval or exemption is asserted here.

## Funding

**[To be completed by authors. Do not infer “no funding.”]**

## Conflict of interest

**[To be completed by all authors.]**

## Author contributions

**[Author order and contributions to be confirmed by all authors; names known from project context are Ch Hari Krishna and G Vennela.]**

## Data and code availability

The source ZAS and UCI datasets are available through the UCI Machine Learning Repository [2,7], subject to their posted terms. The derived local data, exact versions, code, reproducibility artifact location and stable public repository URL must be reviewed and supplied by authors before publication. The present project directory is not a public archival link. Patient-level data or identifiers are not claimed as available.

## References

1. Alizadehsani R, Habibi J, Hosseini MJ, et al. A data mining approach for diagnosis of coronary artery disease. *Comput Methods Programs Biomed*. 2013;111(1):52–61. doi:10.1016/j.cmpb.2013.03.004.
2. Alizadehsani R, Roshanzamir M, Sani Z. Z-Alizadeh Sani [dataset]. UCI Machine Learning Repository. doi:10.24432/C5Q31T.
3. Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. *J Mach Learn Res*. 2010;11:2079–2107.
4. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17:230. doi:10.1186/s12916-019-1466-7.
5. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378.
6. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ*. 2025;388:e082505. doi:10.1136/bmj-2024-082505.
7. Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease [dataset]. UCI Machine Learning Repository. doi:10.24432/C52P4X.
8. Davis J, Goadrich M. The relationship between Precision-Recall and ROC curves. In: *Proceedings of ICML*. 2006:233–240. doi:10.1145/1143844.1143874.
9. Detrano R, Janosi A, Steinbrunn W, et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. *Am J Cardiol*. 1989;64(5):304–310. doi:10.1016/0002-9149(89)90524-9.

Full metadata and authoritative links are provided in `REFERENCES.md` and `references.bib`.

## Tables

### Table 1. Cohorts and endpoint definitions

| Dataset/centre | Records | Positive | Negative | Positive prevalence | Endpoint |
| --- | --- | --- | --- | --- | --- |
| ZAS | 303 | 216 | 87 | 71.3% | Cath: Cad vs Normal; UCI metadata says CAD if diameter narrowing >=50% |
| UCI Cleveland | 303 | 139 | 164 | 45.9% | num 0 vs 1–4 |
| UCI Hungary | 294 | 106 | 188 | 36.1% | num 0 vs 1–4 |
| UCI Switzerland | 123 | 115 | 8 | 93.5% | num 0 vs 1–4 |
| UCI VA Long Beach | 200 | 149 | 51 | 74.5% | num 0 vs 1–4 |
| UCI total | 920 | 509 | 411 | 55.3% | num 0 vs 1–4 |

*Counts are records/source rows; patient uniqueness is not verified. The local ZAS file has 56 columns including outcome, whereas UCI describes 54 features; the difference is unresolved.*

### Table 2. ZAS feature groups and timing categories

| Group | Features | Timing classification | Role |
| --- | --- | --- | --- |
| V3 reference | Age; Sex; Current Smoker; EX-Smoker; FH | Timing uncertain (recording time unverified) | Historical comparator, not timestamp-confirmed |
| Basic risk | Age; Sex; DM; HTN; Current Smoker; EX-Smoker; FH | Timing uncertain | Phase 6 selection candidate |
| Symptoms/examination | Typical Chest Pain; Dyspnea; Function Class; Atypical; Nonanginal; Exertional CP; LowTH Ang; BP; PR; Edema | Timing uncertain | Phase 6 selection candidate |
| Resting ECG | Q Wave; St Elevation; St Depression; Tinversion; LVH; Poor R Progression; BBB | Timing uncertain | Phase 6 selection candidate |
| Laboratory | FBS; CR; TG; LDL; HDL; BUN; ESR; HB; K; Na; WBC; Lymph; Neut; PLT | Timing/units incompletely documented | Phase 6 selection candidate |
| Eligible combined | Union of basic risk, symptoms/examination, resting ECG and laboratory groups (38 fields) | Timing uncertain; “eligible” is a historical group label only | Selected in 15/15 Phase 6 outer fits |
| Excluded | Cath; angiographic fields; EF-TTE; Region RWMA; VHD | Outcome/direct or potentially downstream | Not used as predictors |

*“Eligible combined” is the candidate-group name from the historical project, not evidence of confirmed decision-time availability.*

### Table 3. Phase 6 primary ZAS results by complete-repeat out-of-fold set

| Repeat | Rows each | Events | Prevalence | Selection ROC-AUC | V3-style ROC-AUC | Paired delta ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 303 | 216 | 0.7129 | 0.9134 | 0.7370 | 0.1764 |
| 2 | 303 | 216 | 0.7129 | 0.8999 | 0.7420 | 0.1579 |
| 3 | 303 | 216 | 0.7129 | 0.9176 | 0.7300 | 0.1877 |

Paired mean AUC difference = 0.1740; descriptive split-variation t interval 0.1367–0.2113 (three overlapping repeat summaries; not patient/population uncertainty).

Secondary performance and calibration summaries (each repeat includes all 303 source rows):

| Repeat | Procedure | Average precision | Brier | Log loss | Sensitivity | Specificity | Calibration intercept | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | V3-style comparator | 0.8528 | 0.1741 | 0.5257 | 0.6713 | 0.7126 | -0.0616 | 1.0781 |
| 1 | Selection-aware | 0.9602 | 0.1075 | 0.3447 | 0.8657 | 0.7931 | 0.1218 | 0.8693 |
| 2 | V3-style comparator | 0.8519 | 0.1734 | 0.5251 | 0.6852 | 0.7126 | 0.0222 | 0.9781 |
| 2 | Selection-aware | 0.9537 | 0.1137 | 0.3666 | 0.8148 | 0.8161 | 0.2055 | 0.8097 |
| 3 | V3-style comparator | 0.8359 | 0.1763 | 0.5349 | 0.7130 | 0.7011 | 0.0173 | 0.9575 |
| 3 | Selection-aware | 0.9580 | 0.1016 | 0.3289 | 0.8148 | 0.8851 | 0.1927 | 0.9247 |

AP reference prevalence = 0.7129 for each repeat. Calibration estimates are limited; thresholds were selected on training OOF predictions using Youden J and have no clinical interpretation.

### Table 4. Separate UCI pooled leave-one-centre-out results

| UCI stage | Pooled ROC-AUC | Average precision | Brier | Log loss | Centre ROC-AUC minimum | Centre ROC-AUC maximum |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 1 — basic risk | 0.7179 | 0.7293 | 0.2112 | 0.6117 | 0.5011 | 0.7099 |
| Stage 2 — symptoms | 0.8225 | 0.8241 | 0.1681 | 0.5106 | 0.7102 | 0.8380 |
| Stage 3 — resting measurements | 0.8215 | 0.8191 | 0.1684 | 0.5139 | 0.6790 | 0.8188 |
| Stage 4 — exercise test | 0.8450 | 0.8645 | 0.1584 | 0.4861 | 0.7261 | 0.8747 |

Pooled estimates use the 920-record four-centre cohort; centre ranges are minima/maxima across held-out centres. Stage 4 adds exercise-test features and is a distinct later task.

### Table 5. UCI paired Stage 3-to-Stage 4 changes

| Comparison | Metric | Difference (later−earlier) | 95% interval low | 95% interval high | Bootstrap draws | Uncertainty unit |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 3 → Stage 4 | ROC-AUC | 0.0235 | 0.0061 | 0.0443 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Average precision | 0.0454 | 0.0196 | 0.0706 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Brier | -0.0099 | -0.0186 | -0.0015 | 400 | Paired record bootstrap; centre/source-row paired |
| Stage 3 → Stage 4 | Log loss | -0.0278 | -0.0514 | -0.0056 | 400 | Paired record bootstrap; centre/source-row paired |

Intervals are percentile intervals from 400 paired source-record bootstrap draws using the same sampled rows for both stages; they are not patient-cluster or centre-sampling intervals.

### Table 6. Phase 6 outer model-selection frequencies

| Selection dimension | Selected category | Outer fits selected | Outer fits total | Frequency |
| --- | --- | ---: | ---: | ---: |
| Feature group | v3_five_predictor | 0 | 15 | 0.0000 |
| Feature group | basic_risk | 0 | 15 | 0.0000 |
| Feature group | symptoms_examination | 0 | 15 | 0.0000 |
| Feature group | resting_ecg | 0 | 15 | 0.0000 |
| Feature group | laboratory | 0 | 15 | 0.0000 |
| Feature group | eligible_combined | 15 | 15 | 1.0000 |
| Model family | logistic_l2 | 9 | 15 | 0.6000 |
| Model family | elastic_net | 4 | 15 | 0.2667 |
| Model family | spline_logistic | 2 | 15 | 0.1333 |
| Model family | random_forest | 0 | 15 | 0.0000 |
| Model family | hist_gradient_boosting | 0 | 15 | 0.0000 |

*Frequencies summarize 15 outer training fits. Feature-group selection does not identify individual feature necessity.*

## Figures

- **Figure 1.** Study workflow and separation of ZAS and UCI tracks.

![Figure 1. Study workflow](FIGURES/FIGURE1_STUDY_WORKFLOW.png)

[PDF](FIGURES/FIGURE1_STUDY_WORKFLOW.pdf).
- **Figure 2.** Complete-repeat Phase 6 OOF AUC values. Repeats reuse source rows and are not independent samples.

![Figure 2. Repeat-level AUCs](FIGURES/FIGURE2_PHASE6_REPEAT_AUC.png)

[PDF](FIGURES/FIGURE2_PHASE6_REPEAT_AUC.pdf).
- **Figure 3.** UCI held-out-centre AUC matrix. Switzerland has eight negative records.

![Figure 3. UCI centre AUCs](FIGURES/FIGURE3_UCI_CENTRE_AUC.png)

[PDF](FIGURES/FIGURE3_UCI_CENTRE_AUC.pdf).
- **Figure 4.** ZAS feature timing and leakage categories; categories represent documentation status, not assumed measurement time.

![Figure 4. Feature timing categories](FIGURES/FIGURE4_FEATURE_TIMING.png)

[PDF](FIGURES/FIGURE4_FEATURE_TIMING.pdf).

### Figure captions

**Figure 1.** Separate retrospective analyses: Phase 6 selection-aware ZAS discrimination on matched source-row partitions and a separate UCI four-centre leave-one-centre-out staged evaluation. UCI is not used as validation of ZAS.

**Figure 2.** Repeat-level AUC is calculated from all 303 source rows predicted once out-of-fold within each repeat. The lines pair selection-aware and V3-style comparator estimates under the same outer split repeat. The y-axis spans the complete ROC-AUC range (0–1). Repeat points are not independent patient samples; no population confidence interval is shown.

**Figure 3.** ROC-AUC by UCI stage and held-out centre. Estimates are descriptive. Switzerland has 115 positive and 8 negative records, limiting stability of centre-specific estimates.

**Figure 4.** Feature timing classifications based on current source documentation. No feature is confirmed by source timestamps as available before the diagnostic decision. EF-TTE, Region RWMA and VHD are excluded as potentially downstream/diagnostic; Cath and angiographic fields are outcome/direct outcome information.





