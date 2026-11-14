# Base notebook preparation report

## Scope

This package is preparation-only. No notebook kernel was started, no model was trained, and notebook execution remains disabled.

- Original: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb`
- Working copy: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\reproduction_runner\work\author_faithful\base_reproduction\Daza_Paper_Exact_Accuracy.ipynb`
- Original SHA-256: `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`
- Working-copy SHA-256: `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`
- Verified dataset: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\author_release\heart5.csv`
- Dataset SHA-256: `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2`

## Cell inspection

The notebook contains 20 cells: 16 code and 4 markdown.

- Cell 2 imports `google.colab` and calls `drive.mount`; changed only in the working copy.
- Cell 4 reads `/content/drive/MyDrive/heart5.csv`; changed only to the verified local path.
- Cells 1, 5–20 are unchanged in operation, including SMOTE placement, 80:20 split, grids, CV, calibration, evaluation, and test-set early-stopping behavior.

Cell-by-cell environment review:

| Cells | Review |
|---|---|
| 1 | Python/package imports only; no local-path modification needed. |
| 2 | Google Colab Drive import and mount; changed in the working copy. |
| 3 | Markdown only. |
| 4 | Dataset load from a Colab Drive path; changed in the working copy. |
| 5 | Missing-value removal and categorical encoding; preserved exactly. |
| 6–8 | Data inspection and feature/target assignment; preserved exactly. |
| 9–11 | SMOTE, class-count reporting, and 80:20 split; preserved exactly. |
| 12, 14, 16 | Markdown only. |
| 13 | Hyperparameter grids; preserved exactly, including legacy/invalid candidates. |
| 15 | Five-fold grid searches and XGBRegressor classification scoring; preserved exactly. |
| 17–18 | Stacking and calibration; preserved exactly. |
| 19 | Repeated test-set evaluation labelled early stopping; preserved exactly. |
| 20 | Final metrics and ROC plot; preserved exactly. |

## Required packages and compatibility

The pinned environment is recorded in `requirements-author-reproduction.txt`. Core imports are available in `.venv-repro` at the pinned versions recorded in `base_notebook_environment.txt`. `google.colab` is intentionally not required after the working-copy-only cell 2 adaptation.

Compatibility is not a guarantee of successful execution. The source contains invalid or legacy configurations listed below.

## Expected failures or warnings

- Cell 13: **DecisionTreeClassifier grid includes max_features='auto', which is incompatible with current scikit-learn.** Likely failed candidate(s); GridSearchCV may continue with other candidates depending on error handling.
- Cell 13: **KNeighborsClassifier grid includes the misspelled metric 'minkowskii'.** Likely failed candidate(s); valid metric candidates may still run.
- Cell 13: **LogisticRegression grid combines penalty/solver values that are incompatible and includes penalty='none'.** Some grid candidates are expected to fail under current scikit-learn.
- Cell 13: **XGBRegressor grid uses objective='reg:linear' and silent, both legacy parameters.** Likely incompatible or ignored under current XGBoost.
- Cell 15: **XGBRegressor is scored with scoring='accuracy'.** Expected scoring failure because the regressor produces continuous outputs for a classification metric; this may abort GridSearchCV.
- Cell 19: **The loop repeatedly refits and evaluates on X_test while calling the behavior early stopping.** Methodological defect: test performance controls stopping; preserve for author-faithful reproduction.

## Preservation verification

The original extracted notebook was read-only during preparation. The original SHA-256 is recorded in `SHA256SUMS.txt` and must be rechecked immediately before any future approved execution.

## Readiness decision

The working copy is technically prepared for an explicitly approved execution attempt, but it is not methodologically corrected. Execution still requires a reviewed allowlist, isolated output directory, and confirmation of the expected compatibility failures.
