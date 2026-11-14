# ECG Integrity and Overlap Audit v3

- PTB-XL: 21799 records, 18869 patients; patient multi-fold assignments: 0.
- PTB Diagnostic: 549 records and 290 visible subject directories.
- PTB-XL `strat_fold` is suitable for patient-level protocol construction because each indexed patient remains in one fold. Do not randomly split records.
- PTB-XL and PTB Diagnostic share PTB provenance, but no validated identifier crosswalk exists. Their independence is unresolved and must not be claimed.
- PTB-XL SCP labels are hierarchical and multi-label; diagnostic superclass, subclass, rhythm, and form labels must not be collapsed without a prespecified task definition.
