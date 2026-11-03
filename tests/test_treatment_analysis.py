from types import SimpleNamespace

import pandas as pd

from twincure.db import initialize_database
from twincure.similarity import find_similar_patients
from twincure.treatment_analysis import (
    analyze_treatment_paths,
    generate_treatment_segments,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def initialized_db(tmp_path):
    db = tmp_path / "test.db"
    initialize_database(
        db,
        ROOT / "database/schema.sql",
        ROOT / "data/synthetic_patients.csv",
        ROOT / "data/synthetic_steps.csv",
        ROOT / "data/synthetic_rules.json",
    )
    return db


def one_patient_match(patient_id, score=0.73):
    return SimpleNamespace(matches=pd.DataFrame([{"patient_id": patient_id, "similarity_score": score}]))


def test_single_positive_step_keeps_source_and_score(tmp_path):
    db = initialized_db(tmp_path)
    segments = analyze_treatment_paths(one_patient_match("SYN001", 0.87), db)
    single = segments[(segments.source_patient_id == "SYN001") & (segments.source_step_ids.map(len) == 1)]
    first = single[single.start_step_order == 1].iloc[0]
    assert first.classification == "Positive"
    assert first.source_step_ids == ["ST001A"]
    assert first.step_outcomes == [1.0]
    assert first.source_similarity_score == 0.87


def test_positive_segment_of_consecutive_steps(tmp_path):
    db = initialized_db(tmp_path)
    segments = analyze_treatment_paths(one_patient_match("SYN001"), db)
    pair = segments[segments.source_step_ids.map(lambda ids: ids == ["ST001A", "ST001B"])].iloc[0]
    assert pair.classification == "Positive"
    assert pair.mean_outcome == 1.0
    assert pair.step_orders == [1, 2]
    assert pair.treatment_descriptions == [
        "Example initial observation recorded in a synthetic journey",
        "Example follow-up review recorded in a synthetic journey",
    ]


def test_negative_segment_is_retained_and_labelled(tmp_path):
    db = initialized_db(tmp_path)
    segments = analyze_treatment_paths(one_patient_match("SYN003"), db)
    negative = segments[segments.source_step_ids.map(lambda ids: ids == ["ST003A"])].iloc[0]
    assert negative.classification == "Negative"
    assert negative.mean_outcome == -1.0


def test_mixed_segment_is_retained_and_labelled(tmp_path):
    db = initialized_db(tmp_path)
    segments = analyze_treatment_paths(one_patient_match("SYN003"), db)
    mixed = segments[segments.source_step_ids.map(lambda ids: ids == ["ST003A", "ST003B"])].iloc[0]
    assert mixed.classification == "Mixed/Unclear"
    assert mixed.mean_outcome == 0.0


def test_segments_never_cross_patient_boundaries():
    steps = pd.DataFrame([
        {"patient_id": "SYN-A", "step_id": "A1", "step_order": 1, "step_description": "first", "step_outcome": 1},
        {"patient_id": "SYN-B", "step_id": "B1", "step_order": 1, "step_description": "other", "step_outcome": -1},
    ])
    segments = generate_treatment_segments(steps, {"SYN-A": 0.9, "SYN-B": 0.8}, 2)
    assert all(len(set(segment_ids)) == 1 for segment_ids in segments.source_step_ids)
    assert set(segments.source_patient_id) == {"SYN-A", "SYN-B"}


def test_original_order_is_preserved_even_if_input_is_reversed(tmp_path):
    db = initialized_db(tmp_path)
    result = analyze_treatment_paths(one_patient_match("SYN001"), db)
    pair = result[result.source_step_ids.map(lambda ids: ids == ["ST001A", "ST001B"])].iloc[0]
    assert pair.step_orders == [1, 2]
    assert pair.source_step_ids == ["ST001A", "ST001B"]
    assert pair.treatment_descriptions[0].startswith("Example initial")


def test_configured_max_segment_length_is_respected(tmp_path):
    db = initialized_db(tmp_path)
    settings = tmp_path / "settings.toml"
    settings.write_text("similarity_threshold = 0.2\nmax_segment_length = 2\n", encoding="utf-8")
    segments = analyze_treatment_paths(one_patient_match("SYN001"), db, settings)
    assert max(len(ids) for ids in segments.source_step_ids) == 2
    assert len(segments) == 5  # three one-step windows and two adjacent pairs


def test_empty_input_returns_empty_table_without_error(tmp_path):
    db = initialized_db(tmp_path)
    empty = SimpleNamespace(matches=pd.DataFrame(columns=["patient_id", "similarity_score"]))
    result = analyze_treatment_paths(empty, db)
    assert result.empty
    assert "source_patient_id" in result.columns


def test_missing_outcome_or_description_is_not_invented():
    steps = pd.DataFrame([
        {"patient_id": "SYN-M", "step_id": "M1", "step_order": 1, "step_description": None, "step_outcome": None},
        {"patient_id": "SYN-M", "step_id": "M2", "step_order": 2, "step_description": "known synthetic note", "step_outcome": 1},
    ])
    segments = generate_treatment_segments(steps, {"SYN-M": 0.5}, 2)
    first = segments[segments.source_step_ids.map(lambda ids: ids == ["M1"])].iloc[0]
    pair = segments[segments.source_step_ids.map(lambda ids: ids == ["M1", "M2"])].iloc[0]
    assert first.treatment_descriptions == [None]
    assert first.step_outcomes == [None]
    assert pd.isna(first.mean_outcome)
    assert first.classification == "Mixed/Unclear"
    assert pair.step_outcomes == [None, 1.0]
    assert pd.isna(pair.mean_outcome)
    assert pair.classification == "Mixed/Unclear"
    assert not bool(pair.outcomes_complete)


def test_segment_windows_do_not_cross_a_gap_in_step_order():
    steps = pd.DataFrame([
        {"patient_id": "SYN-G", "step_id": "G1", "step_order": 1, "step_description": "one", "step_outcome": 1},
        {"patient_id": "SYN-G", "step_id": "G3", "step_order": 3, "step_description": "three", "step_outcome": 1},
    ])
    segments = generate_treatment_segments(steps, {"SYN-G": 0.4}, 2)
    assert len(segments) == 2
    assert all(len(ids) == 1 for ids in segments.source_step_ids)
