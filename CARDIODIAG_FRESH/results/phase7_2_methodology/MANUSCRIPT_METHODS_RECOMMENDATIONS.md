# Manuscript methods and reporting recommendations

## Primary analysis

- **Primary endpoint/task:** ZAS `Cath` (Cad vs Normal) and retrospective discrimination from recorded clinical features. State the source-row evaluation unit because patient identifiers are unverified.
- **Primary comparison:** complete-repeat out-of-fold ROC-AUC for the Phase 6 frozen selection procedure versus the matched, re-evaluated V3-style comparator. Define the latter accurately: same outer partitions; fixed V3 five features and balanced logistic family; C reselected within outer training; same Phase 6 calibration/threshold operation. Do not call it an untouched frozen fitted V3 model.
- **Primary reporting:** list the three repeat-level AUCs and paired differences, their range and mean. Call the t interval split-variation only. Do not claim statistical significance or patient-level inference.

## Secondary and exploratory analyses

- Secondary ZAS metrics: average precision with event prevalence (0.7129), Brier, log loss, fold calibration intercept/slope, and inner-selected-threshold sensitivity/specificity. Explicitly state calibration scores are selection-dependent and thresholds have no clinical interpretation.
- Feature/model/group frequencies and permutation importance: descriptive/exploratory; selection group frequency is not individual feature necessity, permutation is not causal.
- Demographic-only sensitivity: present only as timing-uncertain sensitivity, not as a confirmed prediagnostic baseline.
- Phase 5 AUC 0.928214: omit from primary results or place in a clearly labeled historical exploratory analysis; state selected post hoc among 36 displayed rows and that Logistic/Ridge alias.
- UCI: separate secondary track/table. Report four stages and centre-specific performance. Stage 2 and Stage 4 are separate intended information-time tasks. Report Stage 3→4 paired record-bootstrap differences as exploratory.

## Statistical conventions

1. Give n, event/nonevent counts, prevalence, validation unit and partition design next to every result.
2. Keep repeat-level complete OOF AUC separate from mean outer-fold AUC. AUC of pooled predictions within a repeat is the repeat estimate; an unweighted mean of fold AUCs is a different summary.
3. Show fold-level and repeat-level results rather than only pooled scores. Repeat/fold observations overlap in training and across repeats; no t-test that treats these as independent patients.
4. For UCI bootstrap, write “400-resample paired record bootstrap” and identify the key `(held_out_centre, source_row)`. It is conditional on four observed centres and not patient/centre-cluster uncertainty.
5. Report AP with prevalence. Avoid comparing AP between the different cohorts/tasks without explaining baseline prevalence and endpoints.
6. Use “higher in all three evaluated repeats,” not “significantly better.” Do not treat ROC-AUC difference alone as clinical benefit.

## Calibration and thresholds

Describe calibration as outer-test assessment of probabilities from the specified procedure, with selection-dependent inner cross-fit scores and inner-to-final training-size mismatch. Calibration intercept/slope estimates are unstable on small folds; Brier/log loss are limited probability metrics, not proof of transportable risk. The Youden threshold is training-selected and equally weights sensitivity/specificity mathematically; report thresholded measures as exploratory. A new nested-calibration experiment is needed only if the paper intends strong final-model calibration or clinical operating-point claims, not for the narrow retrospective discrimination comparison.

## Feature timing and datasets

State no feature has source-confirmed availability before diagnostic decision. Age/Sex may be conceptually knowable, but the existing audit only asserts availability before outcome, with recording time undocumented. List all selected predictors as timing uncertain; identify TTE/RWMA/VHD and angiographic/outcome fields as excluded/downstream-risk fields. The evidence required is source codebook, workflow/provenance and encounter-relative timestamps.

Describe ZAS and UCI as separate datasets, endpoints and analyses. UCI held-out centres support within-cohort centre-held-out evidence only and not external validation of ZAS. Explain centre heterogeneity and Switzerland’s 8 negatives.

## Recommended tables and figures

- Table 1: cohort, endpoint, counts/prevalence, schema and missingness by dataset/centre.
- Table 2: ZAS V3-style comparator and selection-aware complete-repeat OOF metrics, with each repeat visible and paired AUC difference.
- Figure 1: Phase 6 paired repeat and outer-fold AUC distributions; annotate descriptive split variation and avoid patient-level CI whiskers.
- Table 3 or supplement: selected group/family frequencies and timing categories; avoid presenting group inclusion as individual feature selection.
- Table 4: UCI four-stage pooled plus held-out-centre metrics with per-centre denominators and prevalence.
- Figure 2: centre-level UCI ROC/calibration summaries with sparse-negative caveat and separate stage panels.
- Supplement: Phase 5 post-hoc benchmark, Phase 6.1 reconciliation, feature timing dictionary, exact protocol and artifact manifest.

## Manuscript statements that must not appear

Do not claim prospective prediction, prediagnostic availability, patient-level confidence, statistical significance from repeated splits, clinical utility/readiness/safe triage, externally validated ZAS performance, UCI external validation of ZAS, or that Phase 5 AUC 0.928 is an unbiased final estimate. Do not call Phase 6 calibration independently cross-fitted or clinically calibrated.

## Human decisions before submission

Confirm author list/affiliations/contributions; ethics or exemption determination and data-use permissions; funding/conflicts; target journal/reporting checklist; source-timing evidence availability; and whether intended manuscript claims remain retrospective discrimination only. These are author/governance choices, not technical findings of this review.
