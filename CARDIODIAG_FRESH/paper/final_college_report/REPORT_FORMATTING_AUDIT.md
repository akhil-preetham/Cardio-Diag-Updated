# Paper-style report formatting and rendering audit

## Applied layout

- 6 US Letter portrait pages; full-width centered title/author block followed by two-column paper text, including Roman-numeral sections, figures, tables and references.
- Main body Times New Roman 10 pt, black, single-spaced. Compact tables and references use smaller explicitly sized Times New Roman text. All document text runs and table runs were checked for font, size and black color.
- Two figures: a full-width workflow schematic and a repeat-level AUC comparison. Nine numbered tables cover timing, candidate groups, search grids, repeat-level performance, selection/feature results and the UCI track.
- Table headers repeat across continuations; rows are protected against splitting. The workflow and AUC figure captions are present in the rendered PDF.
- The supplied Downloads PDF was used for formatting only. It is not the project's results source.

## Render verification

Microsoft Word for Microsoft 365 exported the editable DOCX to PDF. Poppler reports US Letter pages. All 6 pages were rasterized at 120 dpi to `qa/word_pages_final_paper/` and visually inspected for column flow, figure paths, labels, table continuation, glyphs, clipping and references. The report has no intentionally blank page. The final references/declarations page is shorter than the body pages because author declarations and the verified source list are concise; it remains readable and within the requested six-to-eight-page range.

## Reproduction commands (PowerShell from repository root)

```powershell
$py = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
& $py 'paper/final_college_report/build_report.py'
$docx = (Resolve-Path 'paper/final_college_report/CARDIODIAG_FINAL_PROJECT_REPORT.docx').Path
$pdf = Join-Path (Split-Path $docx -Parent) 'CARDIODIAG_FINAL_PROJECT_REPORT.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$document = $word.Documents.Open($docx, $false, $true)
$document.ExportAsFixedFormat($pdf, 17)
$document.Close($false)
$word.Quit()
& 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\native\poppler\Library\bin\pdfinfo.exe' $pdf
& $py 'paper/final_college_report/build_final_package.py'
```

The package audit is read-only with respect to historical experiment namespaces; no model fitting or benchmark was run for this report revision.
