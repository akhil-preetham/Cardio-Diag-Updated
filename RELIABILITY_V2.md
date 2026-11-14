# Additive reliability follow-up

The original release remains intact. This follow-up addresses probability calibration, missing-input robustness, feature typing, uncertainty, subgroup reporting and portable scoring. Its protocol and executed code were hashed before experiments. It uses previously examined public records and is a retrospective follow-up, not new independent clinical validation. No default model is replaced.

## Results and decisions

Values below are means of complete out-of-fold metrics across three repeats; overlapping repeats are not independent replications. AUC measures ranking; lower Brier and log loss indicate better probabilistic performance.

| Source procedure | AUC | Brier | Log loss |
|---|---:|---:|---:|
| Historical 30-candidate selection | 0.910316 | 0.107616 | 0.346740 |
| Historical equal-feature L2 control | 0.917447 | 0.101124 | 0.328824 |
| Explicitly typed 38-field control | 0.899248 | 0.112920 | 0.362604 |
| Identical independent components, raw ensemble | 0.916418 | 0.105460 | 0.341194 |
| Existing independently calibrated ensemble | 0.909607 | 0.103731 | 0.338996 |
| New fixed-C1 logit calibration | 0.914449 | 0.102922 | 0.333578 |
| New calibration-block CV logit calibration | 0.913598 | 0.103401 | 0.335545 |
| New intercept-only calibration | 0.915851 | 0.105450 | 0.341636 |

The fixed-C1 calibration improves both AUC and Brier relative to the existing independently calibrated ensemble, using the same base predictions and disjoint held-out calibration records. It still ranks below the identical uncalibrated ensemble and does not establish superiority to the historically calibrated L2 control, whose calibration dependence differs. The latter remains a historical comparison rather than an independently validated replacement. All calibration maps and selection scores are published.

Explicit typing is integrated as a validated optional pipeline and documented schema, but its tested variant worsens AUC and Brier. It is not promoted. Accurate semantic coding remains valuable for auditability; it is not evidence that every encoding change improves discrimination.

The missing-input estimator adds one 20%-masked training copy, half weighting original and masked records, with missingness indicators. This operation occurs inside each estimator fit, including each tuning fold. On the exact existing 50% MCAR masks, AUC improves from 0.687544 to 0.703322 and Brier from 0.186714 to 0.184097. On complete inputs it slightly worsens AUC (0.805884 to 0.804465) and Brier (0.152801 to 0.153791). Whole-age and combined-risk-field deletion also worsen. Consequently it is an optional robustness experiment, not a default replacement or a general missing-data solution.

| AngioCAD procedure | AUC | Brier | Log loss |
|---|---:|---:|---:|
| Frozen raw source model | 0.712471 | 0.198249 | 0.578727 |
| Frozen source-calibrated ensemble | 0.707225 | 0.193659 | 0.562933 |
| Target adaptation-fold prevalence | 0.492906 | 0.165652 | 0.513399 |
| Target intercept-only adaptation | 0.712776 | 0.150723 | 0.485410 |
| Target fixed-C1 logit adaptation | 0.708542 | 0.148793 | 0.466986 |
| Target inner-CV logit adaptation | 0.709179 | 0.148911 | 0.466874 |

Target adaptation fits only on the other target folds; every target record receives held-out predictions once per repeat. The proper-loss improvement over both frozen source probabilities and a prevalence-only baseline supports a scoped calibration result. Fold-specific monotonic maps can slightly alter pooled out-of-fold AUC, despite preserving ranking within a fold. This is **retrospective target adaptation on a previously explored cohort**, not untouched external validation, direct transfer of the 38-field model, or clinical diagnosis. Target labels are necessary to fit these maps; the existing scoring command does not silently apply them to new populations.

## What is solved and what remains

| Limitation | Action and disposition |
|---|---|
| Unstable small-block calibration | Three prespecified alternatives evaluated on identical frozen base components; fixed-C1 improves over existing corrected calibration. |
| Missing-input sensitivity | Training-only augmentation and seven exact stress scenarios integrated; mixed results retained without promotion. |
| Implicit feature types | Explicit fixed schema, code validation and typed comparator integrated; measured degradation documented. |
| Target probability shift | Disjoint adaptation/evaluation folds, prevalence baseline and endpoint sensitivity integrated. Requires target labels and further validation. |
| Incomplete uncertainty assessment | 1,000 paired record draws shared across repeats plus 50 development-bootstrap L2 refits with grouped duplicate copies and out-of-bag evaluation. These do not establish patient-level independence. |
| Limited subgroup transparency | Prespecified age and sex diagnostics with class counts and small-class warnings. Descriptive only. |
| Portability and scoring integrity | Explicit checkout-relative reference data, source-data hash gate, Python 3.11 gate, exclusive output creation and automated verification. |
| No verified patient IDs or acquisition times | Cannot resolve from these files. Patient grouping, prospective prediction and temporal leakage exclusion remain unsupported. |
| Endpoint boundary mismatch | Existing ten-case boundary sensitivities retained and repeated for adapted predictions; exact stenosis values are unavailable. |
| Unconfirmed binary ascertainment and units | Code validates recorded categories; clinical equivalence cannot be established without authoritative documentation. |
| Small, selected historical cohorts | No synthetic increase in evidence. New genuinely compatible independent cohorts or prospective collection are still required. |
| Ethics and institutional declarations | Author/institution supplied; no exemption, approval or funding statement invented. |

The recursive audit is bounded by available evidence. Completion means all prespecified software improvements and experiments are implemented and checked; it does not mean no future improvement is possible or that clinical reliability has been established.

## Evidence and reproduction

Executed code: `CARDIODIAG_FRESH/runner/research_extension_v2/run.py`. Frozen protocol, hashes, predictions, maps, splits, uncertainty outputs, warnings and decisions: `CARDIODIAG_FRESH/results/research_extension_v2/`. Read `DECISION_REGISTER.csv` for every nonregression decision. No outcome-driven extra grids were added.

```sh
python tools/portable_release.py --root . --check-runtime
python tools/verify_reliability_v2.py
python -m pytest -q
python tools/plot_reliability_v2.py
```

The experiment runner refuses to overwrite a completed namespace. Reproduction of new fits requires a separately frozen output version. The existing eight-page IEEE report remains the v1 report; this document is its explicitly versioned follow-up, and new figures and threshold diagnostics are in `CARDIODIAG_FRESH/paper/research_extension_v2/`.
