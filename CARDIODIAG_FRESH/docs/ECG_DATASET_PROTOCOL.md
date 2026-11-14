# ECG Dataset Protocol

PTB-XL and PTB Diagnostic remain separate ECG tracks. The pipeline indexes metadata, identifiers, patient/record counts, labels, and official checksums without loading waveforms into memory. Patient-level splits are required; random record-level splitting is prohibited when multiple records belong to one patient.
