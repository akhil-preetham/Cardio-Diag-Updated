# Reference format conference paper

`CARDIODIAG_CONFERENCE_PAPER.docx` is the editable conference-style paper following the user-supplied Word reference. `CARDIODIAG_CONFERENCE_PAPER.pdf` is the matching native Word export. The current draft has six pages, with Times New Roman 10-point text throughout title, author block, body, tables, algorithm and references. Three equations are native editable Word math. Mathematical glyphs use Word's math fonts.

The structure follows the reference: full-width title/authors/abstract/keywords, two-column introduction and literature survey, full-width workflow, two-column methodology and algorithm, wider results/tables/figures, then two-column discussion, conclusion and references. The corrected authors/emails and all core findings, negative comparisons and clinical limitations are retained. The original eight-page detailed report and five-page supplement remain under `paper/final`.

## Fidelity and intentional differences

The reference has six sections: Letter size for the opening section, A4 for subsequent sections, approximately 0.5903-inch margins and a 0.25-inch two-column gap. These source properties are preserved rather than silently treated as an official venue template. A venue may require a uniform page size; confirm its requirements before submission. Matching a selected paper's layout does not establish acceptance.

The user's fixed Times New Roman 10-point requirement replaces the source's smaller fonts and larger title. The title is black and uses Word's Title style. Three current authors replace four source authors. All scientific text, original figures, tables and personal metadata have been replaced; no reference scientific claims or results are copied. Tables keep rows together and repeat headers; their columns expand to accommodate 10-point metrics without shrinking type. The same held-out results are displayed to four decimals; exact values remain in the numerical trace.

`REFERENCE_LAYOUT.json` records source measurements and its SHA-256. `BUILD_PROVENANCE.json` records input hashes and preserved opaque package parts. The reference DOCX is not redistributed. `CONFERENCE_QA.json` records rendered page, font, author, equation and metric-table checks. `REFERENCE_FORMAT_MANIFEST.json` hashes the reporting artifacts and their inputs. All charts are generated from frozen CSV evidence and the original experiments are unchanged.

## Rebuild

Use the pinned research runtime for `tools/plot_conference_paper.py`. Use the bundled document runtime with python-docx for `tools/build_conference_word.py --reference PATH_TO_THE_RETAINED_DOCX`. The retained reference must match the inspected hash; it is opened unchanged, and a working copy is edited. The builder requires the local reference and never substitutes a generic template.

Render with the bundled document runtime using `tools/render_conference_word.py INPUT_DOCX OUTPUT_RENDER_DIR --renderer PATH_TO_BUNDLED_RENDER_DOCX_PY`. On this Windows system it uses installed Microsoft Word as a read-only PDF backend because the runtime has no LibreOffice executable. Word does not save the reference or working DOCX during rendering. Then run `tools/qa_conference_paper.py --render-dir OUTPUT_RENDER_DIR --reference PATH_TO_THE_RETAINED_DOCX`, inspect every PNG, record the visual review, and refresh `tools/verify_conference_paper.py --manifest`.

Portable verification without Word or the private reference: `python tools/verify_conference_paper.py`. This checks hashes, six section geometries, font/size properties, preserved parts, three editable equations, four own images, author contacts and all 65 displayed result-table metric cells against the frozen numerical trace. It does not refit models or supply clinical validation.
