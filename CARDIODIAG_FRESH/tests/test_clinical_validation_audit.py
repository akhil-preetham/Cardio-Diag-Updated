import json

from runner.clinical_validation_audit import audit_ecg, audit_existing_overlap, audit_nhanes, audit_z_alizadeh


def test_z_feature_audit_excludes_target_and_downstream_fields():
    audit = audit_z_alizadeh().set_index("original_feature_name")
    assert audit.loc["Cath", "eligibility_decision"] == "exclude_target"
    assert audit.loc["EF-TTE", "eligibility_decision"] == "exclude"
    assert audit.loc["Region RWMA", "eligibility_decision"] == "exclude"
    assert audit.loc["VHD", "eligibility_decision"] == "exclude"
    assert audit.loc["Exertional CP", "unique_value_count"] == 1


def test_nhanes_audit_preserves_merge_and_weight_rules():
    audit, merge_steps, summary = audit_nhanes()
    by_variable = audit.set_index("variable")
    assert summary["merged_rows"] == 9254
    assert summary["duplicate_seqn"] == 0
    assert all(step["merged_rows"] == 9254 for step in merge_steps)
    assert by_variable.loc["LBXTR", "recommended_weight"] == "WTSAF2YR"
    assert by_variable.loc["WTMEC2YR", "eligibility_decision"] == "required_design_variable"


def test_ecg_audit_enforces_patient_level_ptb_xl_structure():
    summary = audit_ecg()
    assert summary["ptb_xl_records"] == 21799
    assert summary["ptb_xl_patients"] == 18869
    assert summary["ptb_xl_patient_multi_fold_count"] == 0
    assert summary["ptb_diagnostic_record_count"] == 549


def test_existing_overlap_confirms_cleveland_statlog_correspondence():
    overlap = audit_existing_overlap()
    pair = next(
        item for item in overlap["pairwise_checks"]
        if item["left"] == "cleveland" and item["right"] == "statlog"
    )
    assert pair["classification"] == "confirmed_record_level_overlap"
    assert pair["matched_right_rows"] == 270
    assert pair["target_agreement_after_standard_mapping"] == 270


def test_v3_manifest_records_outputs_and_no_model_activity():
    manifest_path = "results/clinical_validation_audit_v3/audit_manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["prohibited_actions"]["model_training"] is False
    assert manifest["prohibited_actions"]["model_tuning"] is False
    assert manifest["prohibited_actions"]["model_evaluation"] is False
    assert manifest["outputs"]
