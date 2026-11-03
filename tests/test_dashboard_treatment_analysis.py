from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

import twincure.dashboard.treatment_analysis as treatment_page
from twincure.db import initialize_database, connect
from twincure.similarity import find_similar_patients
from twincure.treatment_analysis import analyze_treatment_paths

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"
QUERY = "chest discomfort fatigue"


def initialized_database(tmp_path):
    db_path = tmp_path / "treatment-dashboard.sqlite"
    initialize_database(
        db_path,
        ROOT / "database" / "schema.sql",
        ROOT / "data" / "synthetic_patients.csv",
        ROOT / "data" / "synthetic_steps.csv",
        ROOT / "data" / "synthetic_rules.json",
    )
    return db_path


def open_treatment_page(db_path, monkeypatch):
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Treatment Analysis").run()
    return app


def submit_query(app, query=QUERY):
    app.text_area[0].set_value(query)
    app.button[0].click().run()
    return app


def test_successful_treatment_analysis_groups_categories_and_preserves_provenance(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    expected_matches = find_similar_patients(db_path, QUERY)
    assert expected_matches.status == "ok"
    expected_segments = analyze_treatment_paths(expected_matches, db_path)
    counts = expected_segments.groupby("classification").size().to_dict()
    assert set(counts) == {"Positive", "Negative", "Mixed/Unclear"}

    app = open_treatment_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("Analyzed" in item.value and f"{len(expected_segments)} segment(s)" in item.value for item in app.success)
    assert "Synthetic educational prototype — not for clinical use." in [item.value for item in app.warning]
    assert any("artificial educational examples" in item.value for item in app.info)
    assert any("+0.5" in item.value and "−0.5" in item.value and "Mixed / Unclear" in item.value for item in app.markdown)

    rendered_subheaders = [item.value for item in app.subheader]
    assert f"Positive ({counts['Positive']})" in rendered_subheaders
    assert f"Negative ({counts['Negative']})" in rendered_subheaders
    assert f"Mixed / Unclear ({counts['Mixed/Unclear']})" in rendered_subheaders

    rendered = [item.value for item in app.markdown]
    sample = expected_segments.iloc[0]
    assert any(f"`{sample['source_patient_id']}`" in value for value in rendered)
    assert any(f"`{sample['segment_id']}`" in value for value in rendered)
    for source_step_id, order, description, outcome in zip(
        sample["source_step_ids"], sample["step_orders"],
        sample["treatment_descriptions"], sample["step_outcomes"],
    ):
        assert any(f"Original step order {order}" in value and f"`{source_step_id}`" in value for value in rendered)
        assert any(str(description) in value for value in rendered)
        assert any(f"Original outcome: {float(outcome):+g} (synthetic outcome)" in value for value in rendered)
    score_metrics = [metric.value for metric in app.metric if metric.label == "Stage 2 similarity score"]
    mean_metrics = [metric.value for metric in app.metric if metric.label == "Mean synthetic outcome"]
    assert f"{float(sample['source_similarity_score']):.3f}" in score_metrics
    assert f"{float(sample['mean_outcome']):.2f}" in mean_metrics


def test_empty_query_has_clear_message(tmp_path, monkeypatch):
    app = open_treatment_page(initialized_database(tmp_path), monkeypatch)
    submit_query(app, "  ")
    assert not app.exception
    assert any("Enter a synthetic symptom description" in item.value for item in app.warning)


def test_no_similar_patients_message(tmp_path, monkeypatch):
    app = open_treatment_page(initialized_database(tmp_path), monkeypatch)
    submit_query(app, "volcano orchestra telescope")
    assert not app.exception
    assert any("No similar synthetic patient journeys were found" in item.value for item in app.info)
    assert not any("Analyzed" in item.value for item in app.success)


def test_no_treatment_segments_message_when_matched_journeys_have_no_steps(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM journey_steps")
    app = open_treatment_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("No treatment segments were generated" in item.value for item in app.info)


def test_missing_and_unclear_outcomes_are_shown_without_fabrication(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    fake_similarity = SimpleNamespace(
        status="ok",
        message="synthetic test match",
        threshold=0.2,
        matches=pd.DataFrame([{"patient_id": "SYN-TEST", "similarity_score": 0.7}]),
    )
    synthetic_missing_segment = pd.DataFrame([{
        "segment_id": "SEG-SYN-TEST-0001-0001",
        "source_patient_id": "SYN-TEST",
        "source_step_ids": ["STEP-SYN-1"],
        "step_orders": [1],
        "treatment_descriptions": [None],
        "step_outcomes": [None],
        "mean_outcome": None,
        "classification": "Mixed/Unclear",
        "source_similarity_score": 0.7,
    }])
    monkeypatch.setattr(treatment_page, "find_similar_patients", lambda *_args: fake_similarity)
    monkeypatch.setattr(treatment_page, "analyze_treatment_paths", lambda *_args: synthetic_missing_segment)
    app = open_treatment_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = [item.value for item in app.markdown]
    assert any("Mixed / Unclear (1)" == item for item in [x.value for x in app.subheader])
    assert any("Missing / invalid (not recorded)" in value for value in text)
    assert any("Unavailable (incomplete or invalid outcomes)" in metric.value for metric in app.metric)


def test_missing_database_is_reported_without_creation(tmp_path, monkeypatch):
    missing_db = tmp_path / "missing.sqlite"
    app = open_treatment_page(missing_db, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("Database unavailable" in item.value for item in app.error)
    assert not missing_db.exists()


def test_unexpected_stage3_error_is_presented_clearly(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(treatment_page, "analyze_treatment_paths", lambda *_args: (_ for _ in ()).throw(RuntimeError("test failure")))
    app = open_treatment_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("Unexpected processing error" in item.value for item in app.error)
