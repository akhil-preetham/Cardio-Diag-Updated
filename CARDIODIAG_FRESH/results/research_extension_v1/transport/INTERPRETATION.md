# Shared-core transport interpretation and limitations

This is a new five-core model using Age, Sex, DM, HTN and Current Smoker. It is
neither historical V3 (which uses FH/EX-Smoker) nor the historical 38-feature model.
No AngioCAD label selected features, C, preprocessing, calibration or thresholds.
No target model updating was done. All source fitting precedes target scoring;
PREDICTION_FREEZE.json hashes the trained models and target probabilities.

The raw arm refits on all source rows. The sigmoid ensemble retains three matched
source search/calibrator pairs, each search independent of its calibration labels.
They differ in training sizes, model averaging and calibration; a difference in
performance cannot be attributed to calibration alone. Raw probabilities are not
explicitly calibrated. Neither arm is a clinically validated risk estimate.

Core field codes are mapped directly from observed named source/target columns;
binary presence direction follows conventional0/1, with no label-guided inversion.
The accessible target source has no explicit binary coding or timing dictionary.
The resulting analysis is exploratory and conditional on this semantic assumption.
Source timing and patient independence remain unverified. Target case keys do not
prove distinct patients. Shared coarse-core fingerprints are not duplicate people.

Target >50% endpoint differs from source catalogue >=50%; ten matched nonpositive
cases have 26-50 segments. Exclusion and upper assignment are sensitivity scenarios,
not validated true >=50 labels. Nonobstructive controls can have other coronary
conditions. The acute/stable referral spectrum differs from ZAS. Thirty-six label
cases lack clinical records; exclusion may produce selection. No prediagnostic,
population screening, any-dataset, clinical benefit or external validation of the
original fitted models is claimed.

Target intervals use2,000 record-resampling percentile bootstrap draws conditional
on fixed source fits and the available target cohort. They omit source fitting and
centre-sampling uncertainty; no patient-independence assumption is verified. The
three overlapping source repeat summaries describe split variation, not three
independent studies. Reused development data and known historical performance make
this a methodological sensitivity analysis rather than a newly untouched trial.

Joint calibration intercept/slope regressions are evaluation diagnostics, jointly
estimated on logit probabilities. They are not calibration-in-the-large with an
offset and are never applied to update target predictions. The0.5 threshold is
fixed without clinical justification. Model artifacts are research-only; load only
trusted local files in the recorded environment, never arbitrary external joblib.
