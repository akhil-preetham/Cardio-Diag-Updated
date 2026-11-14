# Clinical Feature Policy v2

## Strict primary features

`Age` is recorded age in years; `Sex` is recorded sex; `Current Smoker` and `EX-Smoker`
are binary smoking-history indicators; and `FH` is a binary family-history indicator.
Observed coding is preserved from the source. Smoking indicators were mutually exclusive
in the audited records, with no row having both indicators. Their source timing remains
a limitation, but they are baseline demographic/risk-factor variables.

## Exploratory pending evidence

Weight, Length, BMI, DM, HTN, Obesity, CRF, CVA, Airway disease, Thyroid Disease, DLP,
BP, PR, FBS, CR, TG, LDL, HDL, BUN, ESR, HB, K, Na, WBC, Lymph, Neut, and PLT require
unit, definition, or timing confirmation. Symptoms, examination findings, and ECG fields
also require prediction-time documentation before inclusion.

## Excluded

`Cath` is the target. `EF-TTE`, `Region RWMA`, and `VHD` are downstream/diagnostic
variables. `CHF`, edema, rales, functional class, and related clinical findings require
clinical review because they may overlap with established cardiac disease. Constant
`Exertional CP` is non-informative.

No exploratory feature may be added solely because it improves performance.
