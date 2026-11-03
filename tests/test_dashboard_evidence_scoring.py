from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

import twincure.dashboard.evidence_scoring as evidence_page
from twincure.db import initialize_database
from twincure.evidence_scoring import score_hybrid_evidence
from twincure.hybrid_generation import generate_hybrid_journeys
from twincure.segment_selection import select_candidate_segments
from twincure.similarity import find_similar_patients
from twincure.treatment_analysis import analyze_treatment_paths

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "twincure" / "app.py"
QUERY = "irregular heartbeat dizziness"


def initialized_database(tmp_path):
    db_path = tmp_path / "evidence-dashboard.sqlite"
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
    app.sidebar.radio[0].set_value("Evidence Scoring").run(timeout=30)
    return app


def submit_query(app, query=QUERY):
    app.text_area[0].set_value(query)
    app.button[0].click().run(timeout=30)
    return app


def page_text(app):
    groups = (app.title, app.subheader, app.markdown, app.info, app.error, app.warning, app.success, app.caption, app.metric)
    values = [str(item.value) for group in groups for item in group]
    values.extend(str(item.label) for item in app.metric)
    for item in app.dataframe:
        frame = item.value
        if isinstance(frame, pd.DataFrame):
            values.extend(str(value) for value in frame.astype(str).to_numpy().ravel())
    return values


def existing_pipeline(db_path, query=QUERY):
    similarity = find_similar_patients(db_path, query)
    segments = analyze_treatment_paths(similarity, db_path)
    selection = select_candidate_segments(segments, db_path, ROOT / "data" / "synthetic_rules.json")
    generation = generate_hybrid_journeys(selection)
    scoring = score_hybrid_evidence(generation)
    return similarity, segments, selection, generation, scoring


def step(patient, segment, step_id, order, outcome, similarity, classification="Positive"):
    return {
        "hybrid_journey_id": "HYB-CONTROLLED",
        "hybrid_position": order,
        "source_patient_id": patient,
        "source_segment_id": segment,
        "source_step_id": step_id,
        "original_step_order": order,
        "treatment_description": f"Synthetic test description {step_id}",
        "original_outcome": outcome,
        "original_segment_classification": classification,
        "source_similarity_score": similarity,
    }


def rejected(segment_id, patient, step_ids, outcomes, classification):
    return {"segment": {
        "segment_id": segment_id,
        "source_patient_id": patient,
        "source_step_ids": step_ids,
        "step_outcomes": outcomes,
        "classification": classification,
    }, "reason": "synthetic test-only Stage 4A rejection"}


def install_controlled_pipeline(monkeypatch, generation):
    similarity = SimpleNamespace(
        status="ok",
        matches=pd.DataFrame([{"patient_id": "A", "similarity_score": 0.8}, {"patient_id": "B", "similarity_score": 0.6}]),
        threshold=0.2,
        message="synthetic test match",
    )
    monkeypatch.setattr(evidence_page, "find_similar_patients", lambda *_args: similarity)
    monkeypatch.setattr(evidence_page, "analyze_treatment_paths", lambda *_args: pd.DataFrame([{"test_segment": True}]))
    monkeypatch.setattr(evidence_page, "select_candidate_segments", lambda *_args: {
        "eligible_combinations": [{"pair_id": "synthetic-test-pair"}],
        "rejected_segments": generation.get("rejected_segments", []),
    })
    monkeypatch.setattr(evidence_page, "generate_hybrid_journeys", lambda _selection: generation)


def controlled_generation(unrelated_rejection=True):
    rejections = [
        rejected("B-NEG-A", "B", ["B2", "B3"], [-1, -1], "Negative"),
        rejected("B-NEG-B", "B", ["B3", "B4", "B5"], [-1, -1, -1], "Negative"),
    ]
    if unrelated_rejection:
        rejections.append(rejected("U-NEG", "UNRELATED-PATIENT", ["U1", "U2"], [-1, -1], "Negative"))
    return {
        "status": "generated",
        "message": "Generated one controlled synthetic test hybrid.",
        "hybrid_journeys": [{
            "hybrid_journey_id": "HYB-CONTROLLED",
            "status": "generated",
            "ordered_steps": [
                step("A", "A-POS", "A1", 1, 1, 0.8),
                step("B", "B-POS", "B1", 2, 1, 0.6),
            ],
        }],
        "rejected_segments": rejections,
        "rejected_combinations": [],
    }


def test_real_stages_2_to_5_pipeline_displays_exact_stage5_components_and_formula(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    similarity, segments, selection, generation, expected = existing_pipeline(db_path)
    assert similarity.status == "ok"
    assert not segments.empty
    assert selection["eligible_combinations"]
    assert generation["hybrid_journeys"]
    assert expected["results"]
    result = expected["results"][0]
    assert result["status"] == "scored"

    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any(result["hybrid_journey_id"] in value for value in text)
    assert any("Score = 100 × (0.4118S + 0.3529O + 0.2353A)" in value for value in text)
    assert any(
        "The evidence-support score measures how strongly the synthetic example data supports a generated hybrid journey under TwinCure's prototype scoring rules. It is not a probability of treatment success, clinical efficacy score, or medical recommendation."
        in value for value in text
    )

    metrics = app.metric
    first_metrics = metrics[:4]
    assert [item.label for item in first_metrics] == [
        "S — Similarity support", "O — Outcome support", "A — Agreement support", "Final evidence-support score (0–100)"
    ]
    assert first_metrics[0].value == f"{result['S']:.4f}"
    assert first_metrics[1].value == f"{result['O']:.4f}"
    assert first_metrics[2].value == f"{result['A']:.4f}"
    assert first_metrics[3].value == f"{result['evidence_support_score']:.2f}"

    frames = [item.value for item in app.dataframe if isinstance(item.value, pd.DataFrame)]
    similarity_frame = next(frame for frame in frames if "similarity_score" in frame.columns)
    assert similarity_frame["source_patient_id"].tolist() == [row["source_patient_id"] for row in result["components"]["S"]["calculations"]]
    outcome_frame = next(frame for frame in frames if "individual_outcome_support" in frame.columns)
    assert outcome_frame["source_patient_id"].tolist() == [row["source_patient_id"] for row in result["components"]["O"]["calculations"]]
    votes_frame = next(frame for frame in frames if "journey_vote" in frame.columns)
    expected_votes = {row["source_patient_id"]: row["vote"] for row in result["components"]["A"]["calculations"]["votes"]}
    assert dict(zip(votes_frame["source_patient_id"], votes_frame["journey_vote"])) == expected_votes


def test_hybrid_specific_agreement_includes_contributor_rejections_but_excludes_unrelated_patient(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    with_unrelated = controlled_generation(unrelated_rejection=True)
    without_unrelated = controlled_generation(unrelated_rejection=False)
    with_score = score_hybrid_evidence(with_unrelated)["results"][0]
    without_score = score_hybrid_evidence(without_unrelated)["results"][0]
    assert with_score["A"] == without_score["A"] == 0.5
    assert with_score["evidence_support_score"] == without_score["evidence_support_score"]
    votes = with_score["components"]["A"]["calculations"]["votes"]
    vote_by_patient = {row["source_patient_id"]: row for row in votes}
    assert vote_by_patient["A"]["vote"] == "positive"
    assert vote_by_patient["B"]["vote"] == "negative"
    assert vote_by_patient["B"]["unique_source_step_ids"] == ["B1", "B2", "B3", "B4", "B5"]
    assert "UNRELATED-PATIENT" not in with_score["components"]["A"]["calculations"]["candidate_patient_ids"]

    install_controlled_pipeline(monkeypatch, with_unrelated)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any("only patients contributing to this hybrid" in value.lower() for value in text)
    assert "UNRELATED-PATIENT" not in text
    votes_frame = next(
        item.value for item in app.dataframe
        if isinstance(item.value, pd.DataFrame) and "journey_vote" in item.value.columns
    )
    assert dict(zip(votes_frame["source_patient_id"], votes_frame["journey_vote"])) == {"A": "positive", "B": "negative"}
    assert "B2 · B3 · B4 · B5" in votes_frame.loc[votes_frame["source_patient_id"] == "B", "unique_source_step_ids"].iloc[0]


def test_insufficient_evidence_is_explained_and_score_is_withheld(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    generation = {
        "status": "generated",
        "hybrid_journeys": [{
            "hybrid_journey_id": "HYB-INCOMPLETE",
            "ordered_steps": [
                step("A", "A-POS", "A1", 1, 1, None),
                step("B", "B-POS", "B1", 2, 1, 0.6),
            ],
        }],
        "rejected_segments": [],
    }
    expected = score_hybrid_evidence(generation)["results"][0]
    assert expected["status"] == "insufficient_evidence"
    assert expected["S"] is None
    assert expected["evidence_support_score"] is None
    install_controlled_pipeline(monkeypatch, generation)
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any("insufficient_evidence" in value for value in text)
    assert any("S is undefined because source journey A has a missing or invalid similarity score." in value for value in text)
    assert any(item.label == "S — Similarity support" and item.value == "Undefined" for item in app.metric)
    assert any(item.label == "Final evidence-support score (0–100)" and item.value == "Withheld" for item in app.metric)
    assert not any(item.label == "Final evidence-support score (0–100)" and item.value not in ("Withheld", None) for item in app.metric)


def test_no_eligible_combination_reports_no_hybrid_and_no_score(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(evidence_page, "select_candidate_segments", lambda *_args: {
        "eligible_combinations": [], "rejected_segments": [], "positive_candidates": pd.DataFrame()
    })
    app = open_page(db_path, monkeypatch)
    submit_query(app)
    assert not app.exception
    text = page_text(app)
    assert any("Stage 4B status: no_valid_hybrid" in value for value in text)
    assert any("No eligible combinations were approved by Stage 4A" in value for value in text)
    assert not app.metric


def test_empty_and_no_similar_case_are_reported(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    empty = open_page(db_path, monkeypatch)
    submit_query(empty, " ")
    assert any("Enter a synthetic symptom description" in value for value in page_text(empty))
    assert not empty.exception

    no_match = open_page(db_path, monkeypatch)
    submit_query(no_match, "zzzxxyy qqqvvp unrelatedtoken")
    assert any("No similar synthetic patient journeys" in value for value in page_text(no_match))
    assert not no_match.exception


def test_database_and_rule_loading_errors_are_reported(tmp_path, monkeypatch):
    missing_db = tmp_path / "missing.sqlite"
    app = open_page(missing_db, monkeypatch)
    submit_query(app)
    assert any("Database unavailable" in value for value in page_text(app))
    assert not missing_db.exists()

    db_path = initialized_database(tmp_path)
    monkeypatch.setattr(evidence_page, "SYNTHETIC_RULES_PATH", tmp_path / "missing-rules.json")
    rules_missing = open_page(db_path, monkeypatch)
    submit_query(rules_missing)
    assert any("Synthetic rule file unavailable" in value for value in page_text(rules_missing))
    assert not rules_missing.exception


def test_evidence_page_and_all_prior_dashboard_sections_remain_routable(tmp_path, monkeypatch):
    db_path = initialized_database(tmp_path)
    app = open_page(db_path, monkeypatch)
    assert "Evidence Scoring" in [item.value for item in app.title]
    for section, expected_title in (
        ("Home", "TwinCure"),
        ("Patient Similarity", "Patient Similarity"),
        ("Treatment Analysis", "Treatment Analysis"),
        ("Compatibility Analysis", "Compatibility Analysis"),
        ("Hybrid Journey", "Hybrid Journey"),
    ):
        app.sidebar.radio[0].set_value(section).run(timeout=30)
        assert not app.exception
        assert expected_title in [item.value for item in app.title]
    assert app.sidebar.radio[0].options == [
        "Home", "Patient Similarity", "Treatment Analysis", "Compatibility Analysis", "Hybrid Journey", "Evidence Scoring"
    ]
