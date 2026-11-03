import pandas as pd

from twincure.db import initialize_database
from twincure.preprocessing import (
    load_patient_histories,
    load_preprocessed_histories,
    preprocess_patient_histories,
    standardize_symptom_text,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def initialize(path):
    initialize_database(
        path,
        ROOT / "database/schema.sql",
        ROOT / "data/synthetic_patients.csv",
        ROOT / "data/synthetic_steps.csv",
        ROOT / "data/synthetic_rules.json",
    )


def test_standardizes_case_punctuation_and_whitespace():
    assert standardize_symptom_text("  CHEST,  Discomfort!!  fatigue ") == "chest discomfort fatigue"


def test_null_and_empty_symptoms_become_empty_string():
    assert standardize_symptom_text(None) == ""
    assert standardize_symptom_text(float("nan")) == ""
    assert standardize_symptom_text("   ") == ""


def test_preprocessing_preserves_original_fields_and_adds_normalized_text():
    source = pd.DataFrame([{"patient_id": "SYN-X", "symptoms_text": "Chest, Fatigue", "journey_outcome": 1}])
    result = preprocess_patient_histories(source)
    assert result.loc[0, "symptoms_text"] == "Chest, Fatigue"
    assert result.loc[0, "normalized_symptoms"] == "chest fatigue"
    assert result.loc[0, "patient_id"] == "SYN-X"
    assert list(source.columns) == ["patient_id", "symptoms_text", "journey_outcome"]


def test_loads_histories_from_database(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    raw = load_patient_histories(db)
    processed = load_preprocessed_histories(db)
    assert len(raw) == 8
    assert "dataset_label" in processed.columns
    assert processed.loc[processed.patient_id == "SYN001", "normalized_symptoms"].iloc[0] == "chest discomfort fatigue"
