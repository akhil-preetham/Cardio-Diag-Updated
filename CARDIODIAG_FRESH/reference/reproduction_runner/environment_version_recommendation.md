# Environment version recommendation

## Recommendation

Do not alter `.venv` or `.venv-repro`. Use a separate, disposable environment only after explicit approval.

## Options

| Strategy | Version position | Evidence status | Use |
|---|---|---|---|
| Current `.venv-repro` | Python 3.11.16 with installed current packages | Installed and recorded; source only partially compatible | Static checks and corrected validation; not exact author execution |
| Historical candidate | Python 3.8/3.9 with older NumPy, pandas, scikit-learn, imbalanced-learn, XGBoost, and matplotlib | Unverified hypothesis; no installation performed | Track A only if independently pinned and API-checked |
| Compatibility-adapted | Current or separately isolated modern environment with documented B changes | Technically plausible but not exact | Track B |

No exact historical package combination is included by the author artifacts. Older dependencies may preserve legacy Decision Tree or XGBoost syntax, but they cannot be assumed to accept the misspelled KNN metric or make a regressor/classification scoring pairing scientifically valid.

## Windows considerations

Use a Windows-compatible Python interpreter, local dataset paths, isolated output directories, and one process at a time. Avoid relying on Colab mounts, `/content` paths, or shell commands embedded in notebooks. Native build requirements for older packages are an additional unverified risk.

## Memory

The base notebook is estimated below 1 GB. Full methodology and fairness notebooks are estimated at 2–4 GB; LIME notebooks at 4–8 GB. These are static estimates, not measurements.
