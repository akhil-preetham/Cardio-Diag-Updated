# Complete notebook inventory and risk classification

Runtime and memory values are conservative static estimates, not measurements.
No notebook was executed during this audit.

## 10: `base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb`
- Phase: **base reproduction**
- Dependencies: `google, imblearn, matplotlib, numpy, pandas, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **5-15 minutes**
- Memory requirement: **<1 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Requires local dataset path adaptation; preserve original Colab path in the audit.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.

## 20: `five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb`
- Phase: **full methodology**
- Dependencies: `google, imblearn, lime, loras, matplotlib, matplotlib_venn, numpy, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/correlation_heatmap.pdf, /content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 20: `five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb`
- Phase: **full methodology**
- Dependencies: `google, imblearn, lime, loras, matplotlib, matplotlib_venn, numpy, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 30: `five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb`
- Phase: **XAIH-DRC**
- Dependencies: `google, imblearn, lime, loras, matplotlib, matplotlib_venn, numpy, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **10-30 minutes**
- Memory requirement: **1-2 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 40: `five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb`
- Phase: **OVEC**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, openpyxl, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/heart_dataset.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 40: `five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb`
- Phase: **OVEC**
- Dependencies: `google, imblearn, lime, loras, matplotlib, numpy, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.

## 40: `mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb`
- Phase: **OVEC**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, openpyxl, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `MendeleyCardiovascular_Disease_Dataset.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv, /content/drive/MyDrive/heart_mendeleyset.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Patient identifier appears in source; exclude unless a defensible policy is documented.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb`
- Phase: **OSEC**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, openpyxl, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/heart_dataset.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb`
- Phase: **OSEC**
- Dependencies: `google, imblearn, lightgbm, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb`
- Phase: **OSEC**
- Dependencies: `google, imblearn, lightgbm, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb`
- Phase: **OSEC**
- Dependencies: `catboost, google, imblearn, lightgbm, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/DCA_OVEC_300dpi.pdf, /content/calibration_curve_300dpi.pdf, /content/confusion_matrix_ys66.pdf, /content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/heart5.csv, /content/novec_confusion_matrix.pdf, /content/pr_curve_300dpi.pdf, /content/voting_confusion_matrix.pdf`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb`
- Phase: **OSEC**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, openpyxl, pandas, seaborn, sklearn, xgboost`
- Dataset requirements: `MendeleyCardiovascular_Disease_Dataset.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv, /content/drive/MyDrive/heart_mendeleyset.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Patient identifier appears in source; exclude unless a defensible policy is documented.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 50: `mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb`
- Phase: **OSEC**
- Dependencies: `google, imblearn, lightgbm, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `MendeleyCardiovascular_Disease_Dataset.csv, heart5.csv`
- Expected runtime: **15-45 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/MendeleyCardiovascular_Disease_Dataset.csv, /content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 60: `five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb`
- Phase: **LIME**
- Dependencies: `PIL, google, imblearn, imgkit, joblib, lime, loras, matplotlib, numpy, pandas, reportlab, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **30-120 minutes**
- Memory requirement: **4-8 GB**
- Colab-specific paths: `/content/LIME_directionality_plot.pdf, /content/LIME_directionality_plot.pdf, /content/LIME_directionality_plot.pdf, /content/NOVEM_CVD_All_Plots.pdf, /content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.

## 60: `lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb`
- Phase: **LIME**
- Dependencies: `PIL, google, imblearn, imgkit, joblib, lime, loras, matplotlib, numpy, pandas, reportlab, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **30-120 minutes**
- Memory requirement: **4-8 GB**
- Colab-specific paths: `/content/LIME_directionality_plot.pdf, /content/LIME_directionality_plot.pdf, /content/LIME_directionality_plot.pdf, /content/NOVEM_CVD_All_Plots.pdf, /content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/LoRAS_train_test_data.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl, /content/drive/MyDrive/models/voting_pipeline.pkl`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.

## 80: `five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb`
- Phase: **statistics/clinical utility**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, pandas, scipy, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **10-60 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 80: `five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb`
- Phase: **statistics/clinical utility**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `heart5.csv`
- Expected runtime: **10-60 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/drive/MyDrive/heart5.csv, /content/drive/MyDrive/heart5.csv`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.

## 80: `mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb`
- Phase: **statistics/clinical utility**
- Dependencies: `google, imblearn, loras, matplotlib, numpy, pandas, scipy, seaborn, sklearn, xgboost`
- Dataset requirements: `MendeleyCardiovascular_Disease_Dataset.csv, heart5.csv`
- Expected runtime: **10-60 minutes**
- Memory requirement: **2-4 GB**
- Colab-specific paths: `/content/DCA_OVEC_300dpi.pdf, /content/DCA_OVEC_300dpi.pdf, /content/DCA_OVEC_300dpi.pdf, /content/calibration_curve_300dpi.pdf, /content/calibration_curve_300dpi.pdf, /content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv, /content/drive/MyDrive/heart5.csv, /content/pr_curve_300dpi.pdf, /content/pr_curve_300dpi.pdf`
- Prerequisites: Run only after prior phase outputs are verified; use a copied notebook and isolated output directory.
- Methodological risks:
  - Verify SMOTE is applied after splitting; source contains both operations.
  - Verify test data are not used for model selection or repeated tuning.
  - XGBRegressor appears alongside classification accuracy; verify estimator/metric compatibility.
  - LabelEncoder on feature categories may impose arbitrary ordinal structure.
  - Patient identifier appears in source; exclude unless a defensible policy is documented.
  - Row deletion may change the analysis population and requires missingness justification.
  - Specialized method is source-detected; correctness and executed code path require manual review.
  - Statistical inference requires independent predictions and a prespecified comparison.
