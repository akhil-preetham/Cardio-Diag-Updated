# Missingness and site audit

## UCI

Rows by site are Cleveland 303, Hungary 294, Switzerland 123 and VA Long Beach 200. Missingness is concentrated in `slope` (309), `ca` (611), `thal` (486), `fbs` (90), `oldpeak` (62), `trestbps` (59), `thalach` (55), `exang` (55) and `chol` (30). Site-aware evaluation must report these patterns and must not impute using held-out-centre information.

The processed UCI files do not include the full-schema `smoke`, `famhist`, `htn` and `dm` fields. The presence of those names in the 76-attribute documentation does not establish availability in the 14-column processed files.

## AngioCAD

The clinical table has 377 patients. Missing counts include Age 6, Current Smoker 46, BMI 33, LVEF 84 and LVEF dysfunction 85. Thirty-six label-only patients have no matching clinical row. Post-diagnostic fields (Gensini Score and artery stenosis labels) are present and are excluded from any prediagnostic model.
