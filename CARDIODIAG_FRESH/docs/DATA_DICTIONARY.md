# CARDIODIAG data dictionary

Machine-readable per-dataset dictionaries are under `data/metadata/data_dictionaries/`. Original target columns and codings are retained. The UCI Cleveland binary target is derived from original `num` and is stored separately.

- `five_states`: target `HeartDisease`, columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease
- `mendeley`: target `target`, columns: patientid, age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels, target
- `kaggle_cvd_70000`: target `cardio`, columns: id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
- `uci_heart_disease_cleveland`: target `num`, columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
- `statlog_heart`: target `target`, columns: age, sex, chest_pain, rest_bp, serum_chol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, oldpeak, slope, major_vessels, thal, target
- `uci_heart_failure`: target `DEATH_EVENT`, columns: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, DEATH_EVENT
