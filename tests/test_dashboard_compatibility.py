import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import twincure.dashboard.compatibility_analysis as compatibility_page
from twincure.db import initialize_database
from twincure.segment_selection import select_candidate_segments
from twincure.similarity import find_similar_patients
from twincure.treatment_analysis import analyze_treatment_paths

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"
QUERY = "chest discomfort fatigue"


def initialized_database(tmp_path):
    db_path = tmp_path / "compatibility-dashboard.sqlite"
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
    app.sidebar.radio[0].set_value("Compatibility Analysis").run(timeout=30)
    return app


def submit_query(app, query=QUERY):
    app.text_area[0].set_value(query)
    app.button[0].click().run(timeout=30)
    return app


def page_text(app):
    collections = (app.markdown, app.info, app.error, app.warning, app.success, app.caption, app.subheader)
    values = [str(item.value) for collection in collections for item in collection]
    for element in app.dataframe:
        frame = element.value
        if isinstance(frame, pd.DataFrame):
            values.extend(str(value) for value in frame.astype(str).to_numpy().ravel())
    return values


def controlled_segments():
    """Test-only segment data using source step IDs in the synthetic SQLite database."""
    rows = []
    for segment_id, patient_id, step_id, classification, outcome in (
        ("SEG-OBS", "SYN001", "ST001A", "Positive", 1),
        ("SEG-REV", "SYN002", "ST002B", "Positive", 1),
        ("SEG-NEG", "SYN003", "ST003A", "Negative", -1),
    ):
        rows.append({
            "segment_id": segment_id,
            "source_patient_id": patient_id,
            "source_step_ids": [step_id],
            "step_orders": [1],
            "treatment_descriptions": [f"Synthetic test description for {step_id}"],
            "step_outcomes": [outcome],
            "mean_outcome": float(outcome),
            "classification": classification,
            "source_similarity_score": 0.8,
            "start_step_order": 1,
            "end_step_order": 1,
            "outcomes_complete": True,
        })
    return pd.DataFrame(rows)


def write_test_rules(tmp_path, compatibility_rules, sequence_rules):
    """Write isolated test-only rules; the project synthetic_rules.json is untouched."""
    path = tmp_path / "scenario-rules.json"
    path.write_text(json.dumps({
        "dataset_label": "SYNTHETIC_EDUCATIONAL_ONLY",
        "description": "Test-only rule fixture; not an application rule set.",
        "compatibility_rules": compatibility_rules,
        "sequence_rules": sequence_rules,
    }), encoding="utf-8")
    return path


def use_controlled_pipeline(db_path, monkeypatch, rules_path):
    matches = pd.DataFrame([
        {"patient_id": "SYN001", "similarity_score": 0.9},
        {"patient_id": "SYN002", "similarity_score": 0.8},
        {"patient_id": "SYN003", "similarity_score": 0.7},
    ])
    result = SimpleNamespace(status="ok", matches=matches, threshold=0.2, message="synthetic test match")
    monkeypatch.setattr(compatibility_page, "find_similar_patients", lambda *_args: result)
    monkeypatch.setattr(compatibility_page, "analyze_treatment_paths", lambda *_args: controlled_segments())
    monkeypatch.setattr(compatibility_page, "SYNTHETIC_RULES_PATH", rules_path)
    return select_candidate_segments(
        controlled_segments(), db_path, rules_path
    )


def test_successful_pipeline_displays_eligible_pairs_unknowns_and_nonpositive_segments(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    matches = find_similar_patients(db_path, QUERY)
    assert matches.status == "ok"
    segments = analyze_treatment_paths(matches, db_path)
    selection = select_candidate_segments(segments, db_path, ROOT / "data" / "synthetic_rules.json")
    assert selection["eligible_combinations"]
    assert any(item["status"] == "unknown" for item in selection["compatibility_results"])
    assert any(item["segment"]["classification"] != "Positive" for item in selection["rejected_segments"])

    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any("passed Stage 4A validation" in item.value for item in app.success)
    assert f"B. Eligible combinations ({len(selection['eligible_combinations'])})" in text
    assert f"A. Eligible Positive candidate segments ({len(selection['eligible_segments'])})" in text
    first_pair = selection["eligible_combinations"][0]
    assert any(str(first_pair["pair_id"]) in value for value in text)
    for source_patient_id in first_pair["source_patient_ids"]:
        assert any(str(source_patient_id) in value for value in text)
    assert any("Every cross-segment category pair has an explicit compatible rule." in value for value in text)
    assert any("Exactly one direction is established by an explicit allowed sequence rule." in value for value in text)
    assert any("unknown (not treated as compatible)" in value for value in text)
    assert any("Rejected segment (not Positive)" in value for value in text)
    assert any("does not establish clinical safety or medical approval" in value for value in text)


@pytest.mark.parametrize(
    "case,compatibility_rules,sequence_rules,expected_heading,expected_text",
    [
        (
            "unknown compatibility",
            [],
            [{"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "allowed", "rule_note": "test-only forward allowance"}],
            "Unresolved combinations",
            "unknown (not treated as compatible)",
        ),
        (
            "explicit incompatibility",
            [{"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "incompatible", "rule_note": "test-only explicit incompatibility"}],
            [{"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "allowed", "rule_note": "test-only forward allowance"}],
            "Rejected combinations",
            "explicitly incompatible",
        ),
        (
            "prohibited ordering",
            [{"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "compatible", "rule_note": "test-only compatibility"}],
            [
                {"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "prohibited", "rule_note": "test-only prohibition"},
                {"from_category": "REVIEW", "to_category": "OBSERVATION", "rule_type": "prohibited", "rule_note": "test-only reverse prohibition"},
            ],
            "Rejected combinations",
            "prohibited sequence",
        ),
        (
            "no applicable ordering",
            [{"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "compatible", "rule_note": "test-only compatibility"}],
            [],
            "Unresolved combinations",
            "no applicable ordering established",
        ),
        (
            "multiple possible orderings",
            [{"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "compatible", "rule_note": "test-only compatibility"}],
            [
                {"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "allowed", "rule_note": "test-only forward allowance"},
                {"from_category": "REVIEW", "to_category": "OBSERVATION", "rule_type": "allowed", "rule_note": "test-only reverse allowance"},
            ],
            "Unresolved combinations",
            "multiple possible orderings",
        ),
    ],
)
def test_stage4a_outcomes_are_displayed_without_recalculation(
    tmp_path, monkeypatch, case, compatibility_rules, sequence_rules, expected_heading, expected_text
):
    db_path = initialized_database(tmp_path)
    rules_path = write_test_rules(tmp_path, compatibility_rules, sequence_rules)
    expected = use_controlled_pipeline(db_path, monkeypatch, rules_path)
    assert not expected["eligible_combinations"]
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any(expected_heading in value for value in text)
    assert any(expected_text in value for value in text), case
    assert "Rejected segment (not Positive)" in text

    if case == "multiple possible orderings":
        assert any("SEG-OBS" in value and "SEG-REV" in value and "→" in value for value in text)


def test_database_unavailable_is_reported_without_creating_file(tmp_path, monkeypatch):
    missing = tmp_path / "missing.sqlite"
    app = open_page(missing, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("Database unavailable" in value for value in page_text(app))
    assert not missing.exists()


def test_rule_loading_error_is_reported_without_accepting_combinations(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    missing_rules = tmp_path / "missing-rules.json"
    monkeypatch.setattr(compatibility_page, "SYNTHETIC_RULES_PATH", missing_rules)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    assert any("Synthetic rule file unavailable" in value for value in page_text(app))
    assert not any("passed Stage 4A validation" in value for value in page_text(app))
