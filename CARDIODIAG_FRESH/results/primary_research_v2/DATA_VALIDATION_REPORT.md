# Data Validation Report

{
  "source": "data\\interim\\validated\\z_alizadeh_sani.csv",
  "rows": 303,
  "columns": 56,
  "target": "Cath",
  "class_counts": {
    "Cad": 216,
    "Normal": 87
  },
  "strict_features": [
    "Age",
    "Sex",
    "Current Smoker",
    "EX-Smoker",
    "FH"
  ],
  "feature_definitions": {
    "Age": "Age in years",
    "Sex": "Recorded sex",
    "Current Smoker": "Current smoking indicator",
    "EX-Smoker": "Former smoking indicator",
    "FH": "Family history indicator"
  },
  "missing_values": {
    "Age": 0,
    "Sex": 0,
    "Current Smoker": 0,
    "EX-Smoker": 0,
    "FH": 0
  },
  "exact_duplicate_rows": 0,
  "strict_predictor_duplicate_rows": 147,
  "raw_data_modified": false
}
