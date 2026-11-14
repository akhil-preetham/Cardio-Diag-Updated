# Figure 2 axis review and correction

## Finding

The Phase 7.3 plot source set `ax.set_ylim(.65,1.0)` in `paper/phase7_3_manuscript/build_tables_figures.py` (line 97). This was an explicit zoom, not an accidental autoscale. The plotted quantity is ROC-AUC, whose complete scale is 0–1. The narrow 0.35-unit display magnified the vertical separation by about 2.86 times relative to a full-scale display and could overstate the visual size of the observed difference.

## Correction

The corrected manuscript copy now uses `paper/phase7_4_1_final_manuscript/regenerate_figure2.py`. The script reads only the frozen `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv` and writes the PNG and PDF together. It sets the y-axis to 0–1, labels that range, uses ticks at 0.2 increments and saves the PNG at 300 dpi. No values, repeats, labels, or model outputs were changed.

The three paired repeat values remain selection-aware procedure AUC 0.9134, 0.8999 and 0.9176 and V3-style comparator AUC 0.7370, 0.7420 and 0.7300 (displayed to three decimals). The figure identifies repeat overlap and does not imply independent patients or show a confidence interval. The complete manuscript caption retains these qualifications.

The original Phase 7.3 manuscript, source and figure files were not overwritten. Byte-for-byte originals are retained in `paper/phase7_3_manuscript/`; the Phase 7.4.1 package also contains `FIGURE2_PHASE6_REPEAT_AUC_ORIGINAL.png` and `.pdf` as convenient before/after references. Their checksums are listed in `PHASE7_4_1_MANIFEST.json`.

## Visual review

Both corrected formats were regenerated from the same script. The PNG was inspected at full size and embedded in the final PDF. Axis range, metric label, repeat labels, values, legend and footnote are readable. The zero baseline gives a visually conservative comparison. The PDF sidecar is present alongside the PNG in the corrected manuscript package.

## Reproduction

```powershell
.\.venv-phase7_4_1\Scripts\python.exe paper\phase7_4_1_final_manuscript\regenerate_figure2.py
```
