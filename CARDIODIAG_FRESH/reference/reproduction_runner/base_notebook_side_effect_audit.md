# Base notebook side-effect audit

This audit is source-only. The original and prepared notebooks were read but not executed.

## Cell findings

| Notebook/cell | Operation | Classification | Assessment |
|---|---|---|---|
| Original cell 2 | `from google.colab import drive`; `drive.mount('/content/drive')` | Block execution until resolved | Requires Google Colab and Drive access. The prepared copy omits the mount only; the original is unchanged. |
| Original cell 4 | `pd.read_csv('/content/drive/MyDrive/heart5.csv')` | Requires controlled working directory | Reads an external dataset. The prepared copy points to the verified local `heart5.csv`; dataset identity and hash are recorded. |
| Prepared cell 4 | `pd.read_csv(<local heart5.csv>)` | Safe with input-hash gate | Read-only input access, provided the pre-run hash matches the run manifest. |
| Cell 17 | `os.environ['PYTHONHASHSEED'] = str(SEED)` | Requires manual approval | Mutates the child notebook process environment and affects hash randomization. It does not modify the repository, but it should be recorded as part of the run environment. |
| Notebook outputs/embedded Colab metadata | Rendered HTML/JavaScript and existing output metadata | Requires manual approval | These are pre-existing notebook content, not evidence that this QA stage executed the notebook. They may depend on Colab rendering. |

## Not detected in executable cell sources

No shell commands, package-install commands, network requests, subprocess calls, deletion calls, pickle/joblib loads, file writes, `to_csv`, `to_excel`, or `savefig` calls were found in the base notebook's executable cell source. The original notebook does import `os`, `random`, and scientific packages, but no package manager invocation was found.

## Controls required

- Execute only the prepared copy.
- Recompute notebook and dataset hashes immediately before any approved run.
- Use a dedicated working directory and output directory.
- Preserve the original and prepared notebooks.
- Treat the environment mutation as an explicit run event.
- Do not infer safety for the other 17 notebooks from this base-notebook audit.
