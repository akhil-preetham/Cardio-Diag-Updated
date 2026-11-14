# Public dataset acquisition and feasibility report

The UCI four-centre dataset was reproduced from the official source files at 920 rows, not the landing-page processed count of 303. Its target is angiographic disease: `num=0` indicates less than 50% diameter narrowing and `num=1–4` indicates greater than 50% narrowing. The cohort has site labels, substantial missingness and two within-site exact duplicate rows. It is a valid candidate for a separate site-aware benchmark, not an unquestioned independent validation cohort.

AngioCAD is publicly accessible under CC BY 4.0. Its two patient-level spreadsheets were downloaded and audited. The clinical table contains 377 patients and the label table 413; it contains clinical/laboratory variables and artery-level stenosis labels. The image archives are 16.4 GB and were not downloaded. Gensini score and artery stenosis are post-diagnostic/target-derived and excluded from prediagnostic modeling. AngioCAD is conditionally suitable for a separate patient-level clinical or imaging experiment, not yet for frozen V3 validation.

The 100-patient Mendeley angiographic dataset is all-positive and image-selected, so it cannot supply a binary CAD validation cohort. No direct external-validation cohort is established by this acquisition phase.
