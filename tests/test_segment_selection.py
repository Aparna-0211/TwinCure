import json

import pandas as pd

from twincure.db import initialize_database
from twincure.segment_selection import select_candidate_segments

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


def segment(segment_id, patient, step_id, order=1, classification="Positive", score=0.77):
    return {
        "segment_id": segment_id,
        "source_patient_id": patient,
        "source_step_ids": [step_id],
        "step_orders": [order],
        "treatment_descriptions": [f"synthetic {step_id}"],
        "step_outcomes": [1 if classification == "Positive" else -1],
        "mean_outcome": 1.0 if classification == "Positive" else -1.0,
        "classification": classification,
        "source_similarity_score": score,
        "start_step_order": order,
        "end_step_order": order,
        "outcomes_complete": True,
    }


def rules_file(tmp_path, compatibility=None, sequence=None):
    source = json.loads((ROOT / "data/synthetic_rules.json").read_text(encoding="utf-8"))
    if compatibility is not None:
        source["compatibility_rules"] = compatibility
    if sequence is not None:
        source["sequence_rules"] = sequence
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def test_positive_candidates_and_nonpositive_rejections_keep_provenance(tmp_path):
    db = initialized_db(tmp_path)
    frame = pd.DataFrame([
        segment("positive-obs", "SYN001", "ST001A", score=0.91),
        segment("negative-review", "SYN002", "ST002B", classification="Negative"),
        {**segment("mixed-support", "SYN003", "ST003B"), "classification": "Mixed/Unclear"},
    ])
    result = select_candidate_segments(frame, db, ROOT / "data/synthetic_rules.json")
    assert result["positive_candidates"].segment_id.tolist() == ["positive-obs"]
    rejected_ids = {item["segment"]["segment_id"] for item in result["rejected_segments"]}
    assert {"negative-review", "mixed-support"}.issubset(rejected_ids)
    assert any("Not selected" in item["reason"] for item in result["rejected_segments"] if item["segment"]["segment_id"] == "negative-review")
    assert "positive-obs" in rejected_ids  # positive, but no distinct journey is available to pair with
    candidate = result["positive_candidates"].iloc[0]
    assert candidate.source_patient_id == "SYN001"
    assert candidate.source_step_ids == ["ST001A"]
    assert candidate.step_orders == [1]
    assert candidate.step_outcomes == [1]
    assert candidate.source_similarity_score == 0.91


def test_explicitly_compatible_and_allowed_order_is_eligible(tmp_path):
    db = initialized_db(tmp_path)
    frame = pd.DataFrame([
        segment("obs-from-one", "SYN001", "ST001A", score=0.9),
        segment("review-from-two", "SYN002", "ST002B", score=0.8),
    ])
    result = select_candidate_segments(frame, db, ROOT / "data/synthetic_rules.json")
    assert result["compatibility_results"][0]["status"] == "compatible"
    sequence = result["sequence_results"][0]
    assert sequence["status"] == "ordered"
    assert sequence["ordered_segment_ids"] == ["obs-from-one", "review-from-two"]
    assert len(result["eligible_combinations"]) == 1
    pair = result["eligible_combinations"][0]
    assert pair["source_patient_ids"] == ["SYN001", "SYN002"]
    assert pair["segments"][0]["source_step_ids"] == ["ST001A"]


def test_unknown_compatibility_is_unconfirmed_and_not_eligible(tmp_path):
    db = initialized_db(tmp_path)
    frame = pd.DataFrame([
        segment("obs-one", "SYN001", "ST001A"),
        segment("support-two", "SYN002", "ST002C"),
    ])
    result = select_candidate_segments(frame, db, ROOT / "data/synthetic_rules.json")
    assert result["compatibility_results"][0]["status"] == "unknown"
    assert "unconfirmed" in result["compatibility_results"][0]["reason"]
    assert result["eligible_combinations"] == []


def test_explicit_incompatibility_is_rejected(tmp_path):
    db = initialized_db(tmp_path)
    rules = rules_file(
        tmp_path,
        compatibility=[{"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "incompatible", "rule_note": "Test-only explicit incompatibility."}],
    )
    frame = pd.DataFrame([
        segment("obs-one", "SYN001", "ST001A"),
        segment("review-two", "SYN002", "ST002B"),
    ])
    result = select_candidate_segments(frame, db, rules)
    assert result["compatibility_results"][0]["status"] == "incompatible"
    assert result["eligible_combinations"] == []
    assert "explicitly incompatible" in result["compatibility_results"][0]["reason"]


def test_unresolved_sequence_is_flagged_not_arbitrarily_ordered(tmp_path):
    db = initialized_db(tmp_path)
    rules = rules_file(tmp_path, sequence=[])
    frame = pd.DataFrame([
        segment("obs-one", "SYN001", "ST001A"),
        segment("review-two", "SYN002", "ST002B"),
    ])
    result = select_candidate_segments(frame, db, rules)
    seq = result["sequence_results"][0]
    assert seq["status"] == "unresolved"
    assert seq["ordered_segment_ids"] is None
    assert "further review" in seq["reason"]
    assert result["eligible_combinations"] == []


def test_ambiguous_sequence_reports_both_orders(tmp_path):
    db = initialized_db(tmp_path)
    rules = rules_file(tmp_path, sequence=[
        {"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "allowed", "rule_note": "forward"},
        {"from_category": "REVIEW", "to_category": "OBSERVATION", "rule_type": "allowed", "rule_note": "reverse"},
    ])
    frame = pd.DataFrame([
        segment("obs-one", "SYN001", "ST001A"),
        segment("review-two", "SYN002", "ST002B"),
    ])
    result = select_candidate_segments(frame, db, rules)
    seq = result["sequence_results"][0]
    assert seq["status"] == "ambiguous"
    assert len(seq["valid_orderings"]) == 2
    assert seq["ordered_segment_ids"] is None
    assert result["eligible_combinations"] == []


def test_same_journey_segments_are_not_combined_and_duplicate_ids_are_removed(tmp_path):
    db = initialized_db(tmp_path)
    first = segment("same-obs", "SYN001", "ST001A")
    second = segment("same-review", "SYN001", "ST001B")
    frame = pd.DataFrame([first, second, first])
    result = select_candidate_segments(frame, db, ROOT / "data/synthetic_rules.json")
    assert result["positive_candidates"].segment_id.tolist() == ["same-obs", "same-review"]
    assert result["compatibility_results"] == []
    assert result["eligible_combinations"] == []
    assert any(d["decision"] == "not_compared_same_journey" for d in result["selection_decisions"])


def test_empty_input_returns_structured_empty_result(tmp_path):
    db = initialized_db(tmp_path)
    result = select_candidate_segments(pd.DataFrame(), db, ROOT / "data/synthetic_rules.json")
    assert result["positive_candidates"].empty
    assert result["eligible_segments"].empty
    assert result["rejected_segments"] == []
    assert result["eligible_combinations"] == []


def test_explicitly_prohibited_ordering_is_invalid(tmp_path):
    db = initialized_db(tmp_path)
    rules = rules_file(tmp_path, sequence=[
        {"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "prohibited", "rule_note": "test prohibition"},
        {"from_category": "REVIEW", "to_category": "OBSERVATION", "rule_type": "prohibited", "rule_note": "reverse test prohibition"},
    ])
    frame = pd.DataFrame([
        segment("obs-one", "SYN001", "ST001A"),
        segment("review-two", "SYN002", "ST002B"),
    ])
    result = select_candidate_segments(frame, db, rules)
    assert result["sequence_results"][0]["status"] == "invalid"
    assert result["sequence_results"][0]["ordered_segment_ids"] is None
    assert result["eligible_combinations"] == []
