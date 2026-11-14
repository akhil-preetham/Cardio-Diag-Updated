# Phase B — Prediction tasks and feature eligibility decisions

## ZAS target task

The dataset target `Cath` encodes CAD versus Normal, with UCI metadata describing CAD as at least 50% diameter narrowing. The collection is described as for CAD diagnosis. Local files contain 303 rows and 54 predictors plus the target, but no patient ID or feature acquisition timestamps. Therefore no candidate has been confirmed as available before the angiographic outcome from source-timing records.

Keep V3's frozen Age/Sex/Current Smoker/EX-Smoker/FH configuration as the historical reference, not as proof that these five variables are timestamp-verified. The rest of the feature set remains exploratory until documentation identifies a specific prediction time and verifies ascertainment before diagnostic decisions. EF-TTE, Region RWMA and VHD are excluded from early prediction (C); Cath is target only (D). The row-by-row dictionary is `ZAS_FEATURE_ELIGIBILITY_MATRIX.csv`.

## UCI task ladder

The outcome is binary `num` (0 versus 1–4) in the four-centre angiography-referred cohort. Keep the stages separate:

- Stage 1: age, sex.
- Stage 2: Stage 1 plus chest-pain category.
- Stage 3: Stage 2 plus admission resting blood pressure, serum cholesterol, fasting blood sugar indicator, resting ECG.
- Stage 4: Stage 3 plus exercise maximum heart rate, exercise-induced angina, exercise ST depression and ST-segment slope. This is a later exercise-test task, not an early-presentation substitute.

Exact temporal timestamps are absent. Stages reflect documented variable/task definitions rather than patient-specific proof of acquisition before angiography. The fluoroscopy vessel count `ca`, thallium test result `thal`, and target `num` are not candidate predictors in these stage definitions (C/C/D). No feature selection or clinical prediction time is inferred from the label “available in this dataset.” See `UCI_FEATURE_ELIGIBILITY_MATRIX.csv`.

## Frozen eligibility policy

Category A requires direct evidence that the variable is available at the named prediction time. Category B means plausible or stage-consistent but insufficient acquisition documentation. Category C means downstream, diagnostic, or unsuitable for the task. Category D is the target or target-derived field. In this audit, none of the ZAS or UCI clinical predictors reaches A because the source has no row-level acquisition timestamps. The category describes evidence sufficiency; it does not claim that every B variable was actually acquired after outcome.

## Leakage/exclusion decisions

- Never use ZAS `Cath` or UCI `num` as predictors.
- Exclude ZAS EF-TTE, Region RWMA and VHD from prediagnostic models.
- Exclude UCI `ca` (angiography-derived) and `thal` (diagnostic stress imaging) from the defined stages.
- Preserve exercise variables only in the explicitly later Stage 4 task.
- Do not use patient/site/source identifiers as features.
- Stage 2 is the most defensible expanded early-presentation UCI research task; it remains an internal-external transportability experiment with uncertain original acquisition timing, not a clinically validated model.
