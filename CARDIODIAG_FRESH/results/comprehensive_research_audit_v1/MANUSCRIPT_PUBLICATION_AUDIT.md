# Manuscript and publication audit

## Supported claims

`paper/manuscript.tex` appropriately distinguishes author-faithful execution from corrected validation, discloses repeated-holdout dependence, excludes Mendeley `patientid` in corrected validation, and avoids clinical deployment claims. Its reported author-faithful and corrected summary values agree with the documented artifacts reviewed here.

## Claims requiring correction or reconciliation

1. Resolve the contradiction over whether paired bootstrap/Wilcoxon/DeLong analyses were performed. The manuscript must cite one authoritative artifact set.
2. Do not call corrected repeated holdouts external validation.
3. Do not call compatibility-adapted OVEC exact reproduction.
4. Do not imply that LIME establishes causality or that subgroup metrics establish fairness.
5. State that historical environment equivalence, the missing benchmark dataset, and exact paper-table extraction remain unresolved.

## Novelty assessment

The strongest defensible contribution is a transparent, reproducible audit and leakage-controlled comparator of a published cardiovascular ensemble workflow, with preserved source-faithful failures and explicit compatibility adaptations. A claim of a novel clinically superior algorithm is not established by the current evidence.

## Publication readiness

**Not ready for final scientific claims.** The project is suitable for a reproducibility/audit manuscript after provenance reconciliation and stronger independent validation. External calibration, patient-level independence, and the missing benchmark dataset are prerequisites for broader generalization claims.
