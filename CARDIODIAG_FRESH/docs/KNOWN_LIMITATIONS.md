# Known limitations

- The historical author environment was not recovered.
- The author notebook has leakage, non-nested selection, invalid/legacy candidates, and test-set-driven stopping.
- Corrected repeated holdouts overlap and are not independent external cohorts.
- The generated corrected reports do not contain formal confidence intervals; conditional bootstrap intervals are reported separately and must not be interpreted as external-cohort uncertainty.
- Paired DeLong/Wilcoxon/bootstrap analyses are available for the persisted OVEC/OSEC holdouts; they remain conditional on a single fixed holdout and overlapping corrected resamples.
- The approximately 70,000-row benchmark dataset was not present in the workspace or author release.
- The supplementary author notebooks were not executed as notebooks; equivalent controlled Python analyses were executed separately and are not historical notebook reproduction.
- The available paper PDF could not be text-extracted with installed local tools. Exact published table values and figure captions therefore remain unverified.
- No clinical effectiveness, deployment, fairness, causal, or publication-readiness claim is supported.
