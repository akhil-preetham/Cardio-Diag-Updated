# Manuscript rendering and visual QA

## Rendering result

The complete corrected Markdown manuscript rendered successfully to `paper/phase7_4_1_final_manuscript/MANUSCRIPT.pdf`. The renderer is a local ReportLab workflow, not a LaTeX compile. No `pdflatex`/`xelatex`/`lualatex`, Pandoc, Typst or LibreOffice executable was available on the inspected command path. Therefore this report does not claim successful LaTeX compilation or journal-template typesetting.

The final PDF has 12 US Letter portrait pages (612 × 792 points; 921,116 bytes). It was generated with the Codex desktop bundled Python 3.12.14 runtime, ReportLab 4.4.9, and Windows Arial font files. Page text and images were checked with pypdf 6.21.4 and rasterized with pypdfium2 5.13.0/Pillow 13.0.0. The exact bundled executable used on this machine was:

```text
C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe
```

From the project root, the exact local commands were:

```powershell
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" results\phase7_4_1_qa\render_markdown_manuscript.py
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" results\phase7_4_1_qa\verify_render.py
```

This runtime path is tied to the current Codex desktop installation. It is not included in, or pinned by, the clean project dependency snapshot. For a different machine, the same renderer needs ReportLab and compatible font files; the verifier additionally needs pypdf, pypdfium2 and Pillow.

## Inspection performed

- Rasterized all 12 final pages and reviewed the contact sheet plus full-size pages containing the results tables, figures and final figure captions. Rendered page images and the contact sheet are under `results/phase7_4_1_qa/rendered_pages_v2/`.
- Text extraction found no blank pages, replacement-glyph character, or missing expected manuscript section/table/figure markers.
- Confirmed all four manuscript figure PNG paths resolve and render. Figure 2 shows the corrected 0–1 scale. Its adjacent PDF companion paths are printed visibly in the PDF; the renderer does not emit fragile local-file PDF links. The companion PDFs are included in the manuscript package.
- Tables fit within the page frames; wide tables use landscape pages and repeat their header rows. No clipping or text overflow was observed in visual review. Table text is smaller than body text and should be checked again against the target journal’s typesetting requirements.
- The final “Figure captions” block begins on its own page and all four captions remain together. Page breaks, page numbers and headers were reviewed.
- The final PDF contains zero link annotations; no clickable local file links are claimed. Four figure PNGs are embedded; their source paths and sidecar PDF paths exist in the package.
- The manuscript source contains no standalone displayed equation blocks requiring equation rendering.

## Limitations

This verifies legibility and layout for a locally rendered review PDF. It does not verify journal-specific margins, reference style, template compliance, accessibility tagging, PDF/A, or submission-system acceptance. A journal-specific export should be rendered and checked after journal selection. The earlier rejected draft rendering (with excessive landscape page breaks) is not the final PDF; the final 12-page rendering was regenerated after the layout correction.
