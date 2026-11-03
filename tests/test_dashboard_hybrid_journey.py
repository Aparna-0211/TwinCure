import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import twincure.dashboard.hybrid_journey as hybrid_page
from twincure.db import initialize_database
from twincure.hybrid_generation import generate_hybrid_journeys
from twincure.segment_selection import select_candidate_segments
from twincure.similarity import find_similar_patients
from twincure.treatment_analysis import analyze_treatment_paths

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"
QUERY = "irregular heartbeat dizziness"


def initialized_database(tmp_path):
    db_path = tmp_path / "hybrid-dashboard.sqlite"
    initialize_database(
        db_path,
        ROOT / "database" / "schema.sql",
        ROOT / "data" / "synthetic_patients.csv",
        ROOT / "data" / "synthetic_steps.csv",
        ROOT / "data" / "synthetic_rules.json",
    )
    return db_path


def open_page(db_path, monkeypatch):
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio[0].set_value("Hybrid Journey").run(timeout=30)
    return app


def submit_query(app, query=QUERY):
    app.text_area[0].set_value(query)
    app.button[0].click().run(timeout=30)
    return app


def page_text(app):
    groups = (app.title, app.subheader, app.markdown, app.info, app.error, app.warning, app.success, app.caption)
    values = [str(item.value) for group in groups for item in group]
    for item in app.dataframe:
        frame = item.value
        if isinstance(frame, pd.DataFrame):
            values.extend(str(value) for value in frame.astype(str).to_numpy().ravel())
    return values


def run_existing_pipeline(db_path, query=QUERY):
    matches = find_similar_patients(db_path, query)
    segments = analyze_treatment_paths(matches, db_path)
    selection = select_candidate_segments(segments, db_path, ROOT / "data" / "synthetic_rules.json")
    generation = generate_hybrid_journeys(selection)
    return matches, segments, selection, generation


def test_successful_pipeline_displays_validated_hybrid_order_and_provenance(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    matches, segments, selection, expected = run_existing_pipeline(db_path)
    assert matches.status == "ok"
    assert selection["eligible_combinations"]
    assert expected["status"] == "generated"
    assert expected["hybrid_journeys"]

    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert app.success
    text = page_text(app)
    first = expected["hybrid_journeys"][0]
    assert any(first["hybrid_journey_id"] in value for value in text)
    assert any("Generation status:" in value for value in text)
    assert any("Stage 4A compatibility explanation" in value for value in text)
    assert any("Stage 4A sequence explanation" in value for value in text)
    assert any(first["compatibility_rule_support"] in value for value in text)
    assert any(first["sequence_rule_support"] in value for value in text)
    assert any("Synthetic educational prototype — not for clinical use." in value for value in text)
    assert any("not clinically validated" in value for value in text)

    displayed = next(
        item.value for item in app.dataframe
        if isinstance(item.value, pd.DataFrame)
        and "hybrid_step_position" in item.value.columns
    )
    expected_steps = pd.DataFrame([{
        "hybrid_step_position": row["hybrid_position"],
        "source_patient_id": row["source_patient_id"],
        "source_segment_id": row["source_segment_id"],
        "source_step_id": row["source_step_id"],
        "original_step_order": row["original_step_order"],
        "description": "Not recorded" if row["treatment_description"] is None else str(row["treatment_description"]),
        "synthetic_outcome": "Not recorded" if row["original_outcome"] is None else str(row["original_outcome"]),
    } for row in first["ordered_steps"]])
    pd.testing.assert_frame_equal(displayed.reset_index(drop=True), expected_steps.reset_index(drop=True))
    assert first["ordered_segment_ids"] == [segment["source_segment_id"] for segment in first["included_segments"]]


def test_only_stage4a_approved_combinations_reach_stage4b(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    matches, segments, selection, expected = run_existing_pipeline(db_path)
    expected_ids = {tuple(hybrid["ordered_segment_ids"]) for hybrid in expected["hybrid_journeys"]}
    assert expected_ids
    assert all(tuple(h["ordered_segment_ids"]) in {
        tuple(pair["ordered_segment_ids"]) for pair in selection["eligible_combinations"]
    } for h in expected["hybrid_journeys"])

    seen = {}
    real_generator = hybrid_page.generate_hybrid_journeys
    def capture_and_generate(stage4a):
        seen["eligible"] = list(stage4a["eligible_combinations"])
        return real_generator(stage4a)
    monkeypatch.setattr(hybrid_page, "generate_hybrid_journeys", capture_and_generate)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert len(seen["eligible"]) == len(selection["eligible_combinations"])
    assert {tuple(item["ordered_segment_ids"]) for item in seen["eligible"]} == {
        tuple(item["ordered_segment_ids"]) for item in selection["eligible_combinations"]
    }


def test_empty_and_no_similar_patient_queries_are_clear(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    empty = open_page(db_path, monkeypatch)
    submit_query(empty, "   ")
    assert any("Enter a synthetic symptom description" in value for value in page_text(empty))
    assert not empty.exception

    no_match = open_page(db_path, monkeypatch)
    submit_query(no_match, "zzzxxyy qqqvvp unrelatedtoken")
    assert any("No similar synthetic patient journeys" in value for value in page_text(no_match))
    assert not no_match.exception


def test_no_treatment_segments_is_reported(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(hybrid_page, "analyze_treatment_paths", lambda *_args: pd.DataFrame())
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert any("No treatment segments were generated" in value for value in page_text(app))
    assert not app.exception


def test_no_eligible_combinations_and_no_valid_hybrid_are_reported(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(
        hybrid_page,
        "select_candidate_segments",
        lambda *_args: {"eligible_combinations": [], "rejected_segments": [], "positive_candidates": pd.DataFrame()},
    )
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    text = page_text(app)
    assert any("Generation status: no_valid_hybrid" in value for value in text)
    assert any("No Stage 4A-approved combination could produce a valid hybrid" in value for value in text)
    assert any("no hybrid was created" in value for value in text)
    assert not app.exception


def test_duplicate_invalid_stage4a_candidate_is_rejected_by_existing_stage4b(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    invalid_selection = {
        "eligible_combinations": [{
            "pair_id": "BAD-PAIR",
            "ordered_segment_ids": ["SEG-A", "SEG-A"],
            "segments": [
                {"segment_id": "SEG-A", "source_patient_id": "SYN001", "classification": "Positive",
                 "source_step_ids": ["ST001A"], "step_orders": [1], "treatment_descriptions": ["synthetic"], "step_outcomes": [1]},
                {"segment_id": "SEG-A", "source_patient_id": "SYN002", "classification": "Positive",
                 "source_step_ids": ["ST002A"], "step_orders": [1], "treatment_descriptions": ["synthetic"], "step_outcomes": [1]},
            ],
            "compatibility_reason": "synthetic test only",
            "sequence_reason": "synthetic test only",
        }],
        "rejected_segments": [],
    }
    generation = generate_hybrid_journeys(invalid_selection)
    assert generation["status"] == "no_valid_hybrid"
    assert generation["rejected_combinations"]
    monkeypatch.setattr(hybrid_page, "select_candidate_segments", lambda *_args: invalid_selection)
    monkeypatch.setattr(hybrid_page, "generate_hybrid_journeys", generate_hybrid_journeys)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    text = page_text(app)
    assert any("Stage 4B rejected or duplicate combinations (1)" in value for value in text)
    assert any("same source segment appears more than once" in value.lower() for value in text)
    assert any("Generation status: no_valid_hybrid" in value for value in text)
    assert not app.exception


def test_database_unavailable_does_not_create_file(tmp_path, monkeypatch):
    missing = tmp_path / "missing.sqlite"
    app = open_page(missing, monkeypatch)
    submit_query(app)
    assert any("Database unavailable" in value for value in page_text(app))
    assert not missing.exists()
    assert not app.exception


def test_rule_file_unavailable_is_reported(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(hybrid_page, "SYNTHETIC_RULES_PATH", tmp_path / "missing-rules.json")
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert any("Synthetic rule file unavailable" in value for value in page_text(app))
    assert not app.exception


def test_unexpected_processing_error_is_reported(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    def fail_generation(_selection):
        raise RuntimeError("synthetic test failure")
    monkeypatch.setattr(hybrid_page, "generate_hybrid_journeys", fail_generation)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert any("Unexpected hybrid-processing error" in value for value in page_text(app))
    assert not app.exception


def test_hybrid_navigation_preserves_previous_pages_and_evidence_placeholder(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    app = open_page(db_path, monkeypatch)
    assert "Hybrid Journey" in [item.value for item in app.title]
    app.sidebar.radio[0].set_value("Home").run(timeout=30)
    assert "TwinCure" in [item.value for item in app.title]
    app.sidebar.radio[0].set_value("Patient Similarity").run(timeout=30)
    assert "Patient Similarity" in [item.value for item in app.title]
    app.sidebar.radio[0].set_value("Treatment Analysis").run(timeout=30)
    assert "Treatment Analysis" in [item.value for item in app.title]
    app.sidebar.radio[0].set_value("Compatibility Analysis").run(timeout=30)
    assert "Compatibility Analysis" in [item.value for item in app.title]
    app.sidebar.radio[0].set_value("Evidence Scoring").run(timeout=30)
    assert "Evidence Scoring" in [item.value for item in app.title]
    assert app.text_area
    assert any("Score = 100 × (0.4118S + 0.3529O + 0.2353A)" in item.value for item in app.markdown)
    assert not app.exception
