from pathlib import Path

from streamlit.testing.v1 import AppTest

from twincure.db import initialize_database
from twincure.similarity import find_similar_patients

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"


def initialized_database(tmp_path):
    db_path = tmp_path / "similarity-dashboard.sqlite"
    initialize_database(
        db_path,
        ROOT / "database" / "schema.sql",
        ROOT / "data" / "synthetic_patients.csv",
        ROOT / "data" / "synthetic_steps.csv",
        ROOT / "data" / "synthetic_rules.json",
    )
    return db_path


def open_similarity_page(db_path, monkeypatch):
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.radio[0].set_value("Patient Similarity").run()
    return app


def submit_query(app, query):
    app.text_area[0].set_value(query)
    app.button[0].click().run()
    return app


def test_successful_search_renders_stage2_matches_and_scores(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    query = "CHEST discomfort, fatigue"
    expected = find_similar_patients(db_path, query)
    assert expected.status == "ok"

    app = open_similarity_page(db_path, monkeypatch)
    submit_query(app, query)
    assert not app.exception
    assert any("Found" in item.value for item in app.success)
    assert any(f"{expected.threshold:.2f}" in item.value for item in app.caption)
    assert "Synthetic educational prototype — not for clinical use." in [item.value for item in app.warning]
    assert any("does not establish medical equivalence" in item.value for item in app.info)

    displayed_scores = [item.value for item in app.metric if item.label == "TF-IDF cosine similarity"]
    expected_scores = [f"{float(score):.3f}" for score in expected.matches["similarity_score"].tolist()]
    assert displayed_scores == expected_scores
    rendered_markdown = [item.value for item in app.markdown]
    for _, patient in expected.matches.iterrows():
        assert any(f"`{patient['patient_id']}`" in value for value in rendered_markdown)
        assert any(str(patient["symptoms_text"]) in value for value in rendered_markdown)


def test_empty_input_gets_clear_message(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    app = open_similarity_page(db_path, monkeypatch)
    submit_query(app, "   ")
    assert not app.exception
    assert any("Enter a synthetic symptom description" in item.value for item in app.warning)


def test_nontextual_punctuation_input_is_rejected_as_invalid(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    app = open_similarity_page(db_path, monkeypatch)
    submit_query(app, "!!! ???")
    assert not app.exception
    assert any("Invalid input" in item.value and "no usable words" in item.value for item in app.error)


def test_no_matches_message_comes_from_stage2_result(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    app = open_similarity_page(db_path, monkeypatch)
    submit_query(app, "volcano orchestra telescope")
    assert not app.exception
    assert any("No matching synthetic patient journeys were found" in item.value for item in app.info)
    assert not [item for item in app.metric if item.label == "TF-IDF cosine similarity"]


def test_missing_database_is_reported_without_creating_file(tmp_path, monkeypatch):
    missing = tmp_path / "missing.sqlite"
    app = open_similarity_page(missing, monkeypatch)
    submit_query(app, "chest discomfort fatigue")
    assert not app.exception
    assert any("Database unavailable" in item.value for item in app.error)
    assert not missing.exists()


def test_home_and_other_sections_remain_unchanged(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    app = AppTest.from_file(str(APP)).run()
    assert "Optimizing Treatments from Similar Patient Journeys" in [item.value for item in app.subheader]
    assert "Database status: accessible" in [item.value for item in app.success]

    app.sidebar.radio[0].set_value("Treatment Analysis").run()
    assert not app.exception
    assert "Treatment Analysis" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Compatibility Analysis").run()
    assert not app.exception
    assert "Compatibility Analysis" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Hybrid Journey").run()
    assert not app.exception
    assert "Hybrid Journey" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Evidence Scoring").run()
    assert not app.exception
    assert "Evidence Scoring" in [item.value for item in app.title]
    assert app.text_area
    assert any("Score = 100 × (0.4118S + 0.3529O + 0.2353A)" in item.value for item in app.markdown)
