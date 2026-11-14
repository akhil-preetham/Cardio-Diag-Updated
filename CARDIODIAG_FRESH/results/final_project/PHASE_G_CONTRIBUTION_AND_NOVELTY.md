# Phase G — Research contribution and novelty assessment

## Research gap

The repository tests whether headline performance on small, commonly reused CAD benchmarks survives leakage-aware evaluation, predictor-timing scrutiny, calibration assessment and centre-held-out analysis. This is a reproducibility and validity question, not a new classifier invention. Prior ZAS work already uses the same small angiography-referred cohort and reports high accuracy; UCI Heart Disease has long been reused as a centre dataset. The local contribution is to preserve the frozen V3 baseline, make feature timing and outcome compatibility explicit, reconcile the Phase 5 0.928 result to predictions, identify model duplication/post-selection, and report centre variation with paired stages.

## Contribution statement

**Engineering contribution:** auditable, phase-isolated data and prediction artifacts with saved protocols/manifests and separate ZAS/UCI research tracks.

**Empirical finding:** the ZAS Phase 5 `eligible_combined` result reconstructs to ROC-AUC 0.9282, but it is post hoc, duplicated across Logistic/Ridge labels and based on features with unresolved acquisition timing; it is not promoted as a primary prediagnostic estimate. On the UCI four-centre predictions, Stage 2 improves over Stage 1, Stage 3 contributes little over Stage 2 on pooled discrimination, and Stage 4 improves pooled metrics while answering a later exercise-test task; centre results vary and Switzerland is highly imbalanced.

**Methodological contribution:** the strongest claim is a transparent audit and correction of evaluation interpretation: nested hyperparameter selection does not correct later candidate-set selection, and compatibility of outcome, predictors and timing governs what can be called external validation.

## Claims not supported

No novel algorithm, prospective screening benefit, clinical utility, causal feature effect, patient-level safety, broad generalization or external validation is established. Explainability and high AUC do not create these claims.

## Evidence and literature position

The original ZAS publication describes a 303-record angiography dataset and feature-engineering/classification results; the UCI dataset cards describe the ZAS >=50% threshold and the four-centre Heart Disease collection. Model-selection bias is a known risk when choosing among candidates on finite validation results (Cawley & Talbot). TRIPOD+AI provides reporting guidance, while PROBAST+AI addresses risk of bias/applicability; neither is evidence that this project is clinically valid.

Verified sources: Alizadehsani et al., *Computer Methods and Programs in Biomedicine* 111(1), 52–61 (2013), DOI 10.1016/j.cmpb.2013.03.004; UCI Z-Alizadeh Sani DOI 10.24432/C5Q31T; UCI Heart Disease DOI 10.24432/C52P4X; Cawley & Talbot, *JMLR* 11, 2079–2107 (2010); Collins et al., *BMJ* 385:e078378 (2024), DOI 10.1136/bmj-2023-078378; Moons et al., *BMJ* 388:e082505 (2025), DOI 10.1136/bmj-2024-082505.
