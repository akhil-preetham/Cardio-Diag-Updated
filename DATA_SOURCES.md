# Data sources and release scope

The default release verifies the ZAS and AngioCAD studies. It is not a single merged cardiovascular dataset.

| Resource | Original provider | Included in this release |
|---|---|---|
| Z-Alizadeh Sani | [UCI](https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani), DOI 10.24432/C5Q31T, posted CC BY 4.0 | Small official source workbook, validated 303-row table and provenance checks |
| AngioCAD | [Zenodo](https://zenodo.org/records/15826856), DOI 10.5281/zenodo.15826856, CC BY 4.0 | Only clinical/label workbooks and official metadata, provider checksums, local hashes and screening code |
| UCI Heart Disease | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI 10.24432/C52P4X | Historical source code and derived research records; obtain official raw files for a full reconstruction |
| Mendeley cardiovascular dataset | [Mendeley](https://data.mendeley.com/datasets/dzz48mvjht/1) | Historical source code and evidence; source/version/outcome constraints are documented |
| Other historical cardiovascular, ECG and survey tracks | See project data-acquisition code and catalogues | Code/documentation retained; raw clinical/imaging archives omitted |

ZAS source workbook: 303 rows, 55 fields besides Cath; the provider prose's 54-feature count differs from its workbook. Exact local-to-official workbook comparison is recorded in the provenance evidence. Patient uniqueness and predictor timestamps are not verified.

AngioCAD has 413 label IDs but 377 matching clinical-feature IDs. The operational >50% segment rule yields 298 positives and 79 without >50% stenosis. Ten of the latter contain a 26-50 segment, so true >=50% labels cannot be recovered. Clinical codes/ascertainment, timing, case mix and source/target equivalence remain limitations. No AngioCAD image archive was downloaded for this extension.

Provider attribution and original posted terms remain with the respective resources. Public availability, a case key or a source-row checksum does not establish ethics exemption, patient independence or clinical applicability. The release contains no private hospital cohort. Notebook outputs were cleared so optional notebooks do not silently redistribute their embedded data/plots as source code.
