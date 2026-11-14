# Reproducibility audit

## Integrity

- Original extracted notebook SHA-256: `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`.
- Prepared working-copy SHA-256 before/after run: `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`.
- Five States dataset SHA-256: `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2`.
- The nested author repository was clean at commit `e63b02c8a2aba6344a48025f657696fd99ac3a5f`.
- Runner execution gates are disabled and the allowlist contains only the prepared working copy.

## Traceability

The 18-notebook inventory, cell-level evidence audit, execution manifest, executed notebook, corrected JSON reports, prediction CSVs, and final status inventory are preserved under `reference/reproduction_runner/`.

The author-faithful metrics are traceable to the executed notebook output and successful manifest entry. Corrected metrics are traceable to the two JSON reports and 20 persisted prediction files. No static-only notebook is described as reproduced.

## Methodological audit

The author source applies SMOTE before splitting, fits label encoders before splitting, uses non-nested model selection, and uses held-out test accuracy for stopping. The corrected runner moves preprocessing and SMOTE into fold-local pipelines and evaluates the test partition once after selection, but repeated holdouts overlap. It is therefore a corrected comparator rather than independent external validation.

## Quality

Focused implementation validation passes: 8 tests, Python compilation, and
Ruff for new implementation/experiment files. The broader legacy runner set
retains 38 pre-existing findings, primarily formatting/import style, regex-flag
style, and audit-utility findings. The exact-count compatibility OVEC rerun was
written to a new output root; previous completed outputs were preserved. No
original notebook or dataset was modified.
