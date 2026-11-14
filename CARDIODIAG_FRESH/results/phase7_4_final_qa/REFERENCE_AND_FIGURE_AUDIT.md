# Phase 7.4 — Reference and figure audit

## References

In-text citations [1]–[9] all have matching records in the manuscript, `REFERENCES.md` and `references.bib`; every bibliography item is cited. No fabricated entry or unsupported novelty/priority claim was found. Metadata was checked against authoritative records previously captured in Phase 7.3.

| Ref | Metadata/source checked | Result |
|---|---|---|
| 1 | Alizadehsani et al., CMPB 2013; DOI 10.1016/j.cmpb.2013.03.004; PubMed PMID 23537611 | Consistent; supports original ZAS context, not local predictor timing. |
| 2 | UCI Z-Alizadeh Sani, DOI 10.24432/C5Q31T | Official catalog supports 303 instances, endpoint and 54-feature description; local schema discrepancy flagged. |
| 3 | Cawley & Talbot, JMLR 2010; 11:2079–2107 | Publisher record consistent; supports general selection-optimism rationale. |
| 4 | Van Calster et al., BMC Medicine 2019; DOI 10.1186/s12916-019-1466-7 | Consistent; used for calibration distinction. |
| 5 | Collins et al., BMJ 2024;385:e078378; DOI 10.1136/bmj-2023-078378 | Official BMJ record supports TRIPOD+AI reporting guidance. |
| 6 | Moons et al., BMJ 2025;388:e082505; DOI 10.1136/bmj-2024-082505 | Official BMJ record supports PROBAST+AI appraisal framework. |
| 7 | UCI Heart Disease, DOI 10.24432/C52P4X | Official catalog supports source collection/missingness description. |
| 8 | Davis & Goadrich, ICML 2006; DOI 10.1145/1143844.1143874 | Consistent; supports PR/AP context, not local result values. |
| 9 | Detrano et al., Am J Cardiol 1989;64(5):304–310; DOI 10.1016/0002-9149(89)90524-9 | Bibliographic record consistent; appropriate historical UCI citation. |

## Tables and figures

- Tables are numbered 1–6 with captions; all have CSV and Markdown versions. Values were checked in `MANUSCRIPT_NUMERICAL_AUDIT.md`.
- Each of four figures has a resolvable PNG and same-basename PDF. PNG sizes range 2301–3039 px wide with 300-dpi metadata.
- Figure 1 correctly separates ZAS and UCI. Figure 3 has numeric labels, stated color scale and Switzerland's sparse negatives. Figure 4 distinguishes documented timing categories without inferring collection time.
- Figure 2 starts its vertical scale at .65, which emphasizes the difference visually. Values and ticks are visible; recommend a zero baseline or clearer note for journal production.
- PDF presence was checked; independent raster rendering/pixel comparison was not performed. Final journal-specific layout inspection remains pending.
