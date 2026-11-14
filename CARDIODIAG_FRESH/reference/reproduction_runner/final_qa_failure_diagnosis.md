# Final QA failure diagnosis

## Finding

The previous verification command returned exit code 1 after all substantive checks had printed successfully. Its final probe was:

```powershell
Get-Process python,jupyter -ErrorAction SilentlyContinue | Select-Object ProcessName,Id
```

Because no matching `python` or `jupyter` process was present, the PowerShell process returned a nonzero status. This was a process-observation probe, not notebook execution.

## Impact

- Artifact generation was not affected; the requested planning artifacts had already been written.
- Hash, `pip check`, configuration, and author-repository checks ran before the failing probe.
- The command as a whole cannot be called a successful aggregate verification because its final process check failed.
- The probe did not prove historical environment immutability or prove the absence of all possible execution activity; it only found no matching processes at that moment.

## Skipped or incomplete evidence

No later command in that shell invocation ran after the process probe. A robust process check should treat “no matching process” as a successful negative observation, not as a command failure. The command also did not independently compare the readiness matrix with the source inventory or validate every report.

## Executive-summary assessment

The earlier summary correctly reported the printed positive checks, but it overstated the overall final verification because the command ended nonzero and the process probe was not normalized. This QA pass replaces that conclusion with the findings in `artifact_integrity_report.md` and `final_qa_and_reproduction_readiness.md`.

## Corrective action

This QA stage used read-only inspection and a validation command that explicitly reports missing processes as an observed safe state. No notebook execution, kernel startup, model training, package operation, or source modification was performed.
