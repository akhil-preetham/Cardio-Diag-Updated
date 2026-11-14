import json
from pathlib import Path

import pytest

from runner.common import load_config, resolve
from runner.dataset_validation import inspect_dataset


def test_config_resolves_project_root():
    config_path = Path(__file__).parents[1] / "config" / "runner_config.json"
    config, root = load_config(config_path)
    assert root == config_path.parents[1]
    assert resolve(root, config["datasets"]["five_states"]).exists()


def test_dataset_validation_excludes_patient_identifier(tmp_path):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("patientid,target,value\n1,0,2\n2,1,3\n", encoding="utf-8")
    result = inspect_dataset("sample", dataset, "target")
    assert result["status"] == "validated"
    assert result["identifier_columns"] == ["patientid"]
    assert result["target_values"] == {0: 1, 1: 1}
