# Dataset Provenance Audit v3

All v2-acquired inputs were audited locally without additional downloads. UCI Z-Alizadeh Sani is version 412, DOI 10.24432/C5Q31T, CC BY 4.0, 303 rows and 56 columns. PTB-XL is version 1.0.3, DOI 10.13026/kfzx-aw45, CC BY 4.0, 21,799 records and 18,869 patients. PTB Diagnostic is version 1.0.0, DOI 10.13026/C28C71, ODC-By 1.0, 549 records; 290 subject directories are exposed in the local RECORDS file while the official page documents 294 subjects. Auricle is version 1.0.0, DOI 10.13026/swme-vp69, ODC-By 1.0, metadata-only locally. NHANES is the 2017-2018 public-use cycle with ten components.

PTB-XL, PTB Diagnostic, and Auricle metadata checksums were independently compared with their respective official SHA256SUMS files. Both ECG archives passed ZIP structural validation. The Auricle checksum file was re-downloaded from the Auricle endpoint and all acquired Auricle metadata checksums pass; it is not the PTB Diagnostic checksum file.

Existing author-release datasets and prior manifests remain hash-consistent. Kaggle licensing remains unresolved, and Five States provenance remains unresolved.
