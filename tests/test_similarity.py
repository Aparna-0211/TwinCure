import pandas as pd

from twincure.db import initialize_database
from twincure.preprocessing import load_preprocessed_histories
from twincure.similarity import find_similar_patients, rank_similar_patients

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def initialize(path):
    initialize_database(
        path,
        ROOT / "database/schema.sql",
        ROOT / "data/synthetic_patients.csv",
        ROOT / "data/synthetic_steps.csv",
        ROOT / "data/synthetic_rules.json",
    )


def write_settings(path, threshold):
    path.write_text(f"similarity_threshold = {threshold}\nmax_segment_length = 2\n", encoding="utf-8")


def test_expected_similar_journeys_are_ranked_and_preserve_original_fields(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    result = find_similar_patients(db, "CHEST discomfort, fatigue")
    assert result.status == "ok"
    assert result.threshold == 0.20
    assert "SYN001" in set(result.matches.patient_id)
    assert result.matches.similarity_score.is_monotonic_decreasing
    assert (result.matches.similarity_score >= result.threshold).all()
    for column in ("age_group", "disease", "symptoms_text", "journey_outcome", "dataset_label", "normalized_symptoms"):
        assert column in result.matches.columns
    row = result.matches.loc[result.matches.patient_id == "SYN001"].iloc[0]
    assert row.symptoms_text == "chest discomfort fatigue"
    assert row.dataset_label == "SYNTHETIC_EDUCATIONAL_ONLY"


def test_dissimilar_case_returns_no_matches(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    result = find_similar_patients(db, "volcano orchestra telescope")
    assert result.status == "insufficient_similarity"
    assert result.matches.empty
    assert "threshold" in result.message


def test_threshold_is_loaded_from_config_and_applied_inclusively(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    histories = load_preprocessed_histories(db)
    settings = tmp_path / "settings.toml"
    write_settings(settings, 1.0)
    strict = rank_similar_patients("chest discomfort fatigue", histories, settings)
    assert strict.status == "ok"
    assert strict.threshold == 1.0
    assert strict.matches.patient_id.tolist() == ["SYN001"]
    write_settings(settings, 0.20)
    relaxed = rank_similar_patients("chest discomfort fatigue", histories, settings)
    assert relaxed.threshold == 0.20
    assert len(relaxed.matches) > len(strict.matches)


def test_empty_query_is_handled_without_vectorizer_error(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    for empty in ("", "   ", None, float("nan")):
        result = find_similar_patients(db, empty)
        assert result.status == "empty_query"
        assert result.matches.empty


def test_missing_candidate_symptoms_are_ignored_safely(tmp_path):
    histories = pd.DataFrame([
        {"patient_id": "SYN-A", "symptoms_text": None, "journey_outcome": 0, "dataset_label": "SYNTHETIC_EDUCATIONAL_ONLY"},
        {"patient_id": "SYN-B", "symptoms_text": "chest discomfort", "journey_outcome": 1, "dataset_label": "SYNTHETIC_EDUCATIONAL_ONLY"},
    ])
    settings = tmp_path / "settings.toml"
    write_settings(settings, 0.20)
    result = rank_similar_patients("chest discomfort", histories, settings)
    assert result.status == "ok"
    assert result.matches.patient_id.tolist() == ["SYN-B"]
