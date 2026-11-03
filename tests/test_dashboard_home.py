from pathlib import Path

from streamlit.testing.v1 import AppTest

from twincure.db import initialize_database
from twincure.dashboard.home import load_home_stats

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"


def initialized_database(tmp_path):
    db_path = tmp_path / "dashboard.sqlite"
    initialize_database(
        db_path,
        ROOT / "database" / "schema.sql",
        ROOT / "data" / "synthetic_patients.csv",
        ROOT / "data" / "synthetic_steps.csv",
        ROOT / "data" / "synthetic_rules.json",
    )
    return db_path


def test_home_page_shows_live_stats_from_existing_database(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    stats = load_home_stats(db_path)
    assert stats["accessible"] is True
    assert stats["patients"] > 0
    assert stats["journey_steps"] > 0
    assert stats["sequence_rules"] > 0
    assert stats["compatibility_rules"] > 0

    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "TwinCure" in [item.value for item in app.title]
    assert "Optimizing Treatments from Similar Patient Journeys" in [item.value for item in app.subheader]
    assert "Synthetic educational prototype — not for clinical use." in [item.value for item in app.warning]
    assert "Database status: accessible" in [item.value for item in app.success]
    metrics = {item.label: item.value for item in app.metric}
    assert metrics == {
        "Synthetic patients": str(stats["patients"]),
        "Journey steps": str(stats["journey_steps"]),
        "Sequence rules": str(stats["sequence_rules"]),
        "Compatibility rules": str(stats["compatibility_rules"]),
    }


def test_unavailable_database_is_reported_without_creating_a_file(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.sqlite"
    monkeypatch.setenv("TWINCURE_DB_PATH", str(missing))
    stats = load_home_stats(missing)
    assert stats["accessible"] is False
    assert "not found" in stats["error"]
    assert not missing.exists()

    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    errors = [item.value for item in app.error]
    assert any("Database status: unavailable" in message for message in errors)
    metrics = {item.label: item.value for item in app.metric}
    assert all(value == "—" for value in metrics.values())
    assert not missing.exists()


def test_navigation_keeps_all_six_dashboard_sections(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setenv("TWINCURE_DB_PATH", str(db_path))
    app = AppTest.from_file(str(APP)).run()
    radio = app.sidebar.radio[0]
    assert list(radio.options) == [
        "Home",
        "Patient Similarity",
        "Treatment Analysis",
        "Compatibility Analysis",
        "Hybrid Journey",
        "Evidence Scoring",
    ]
    app.sidebar.radio[0].set_value("Patient Similarity").run()
    assert not app.exception
    assert "Patient Similarity" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Treatment Analysis").run()
    assert not app.exception
    assert "Treatment Analysis" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Compatibility Analysis").run()
    assert not app.exception
    assert "Compatibility Analysis" in [item.value for item in app.title]
    assert app.text_area
    assert any("An unknown rule is not treated as compatible." in item.value for item in app.markdown)

    app.sidebar.radio[0].set_value("Hybrid Journey").run()
    assert not app.exception
    assert "Hybrid Journey" in [item.value for item in app.title]
    assert app.text_area

    app.sidebar.radio[0].set_value("Evidence Scoring").run()
    assert not app.exception
    assert "Evidence Scoring" in [item.value for item in app.title]
    assert app.text_area
    assert any("Score = 100 × (0.4118S + 0.3529O + 0.2353A)" in item.value for item in app.markdown)
