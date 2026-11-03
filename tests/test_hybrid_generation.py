import copy
import json

import pandas as pd

from twincure.db import initialize_database
from twincure.segment_selection import select_candidate_segments
from twincure.hybrid_generation import generate_hybrid_journeys

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


def make_segment(segment_id, patient, step_ids, orders, descriptions, outcomes=None, classification="Positive", score=0.8):
    if outcomes is None:
        outcomes = [1] * len(step_ids)
    return {
        "segment_id": segment_id,
        "source_patient_id": patient,
        "source_step_ids": step_ids,
        "step_orders": orders,
        "treatment_descriptions": descriptions,
        "step_outcomes": outcomes,
        "mean_outcome": 1.0 if classification == "Positive" else -1.0,
        "classification": classification,
        "source_similarity_score": score,
        "start_step_order": orders[0],
        "end_step_order": orders[-1],
        "outcomes_complete": True,
    }


def selection(db, segments, rules_path=None):
    return select_candidate_segments(
        pd.DataFrame(segments), db, rules_path or ROOT / "data/synthetic_rules.json"
    )


def custom_rules(tmp_path, sequence=None, compatibility=None):
    rules = json.loads((ROOT / "data/synthetic_rules.json").read_text(encoding="utf-8"))
    if sequence is not None:
        rules["sequence_rules"] = sequence
    if compatibility is not None:
        rules["compatibility_rules"] = compatibility
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path


def basic_pair(db, rules_path=None):
    return selection(db, [
        make_segment("obs-source", "SYN001", ["ST001A"], [1], ["synthetic initial note"], score=0.91),
        make_segment("review-source", "SYN002", ["ST002B"], [2], ["synthetic review note"], score=0.82),
    ], rules_path)


def test_successful_hybrid_uses_stage4a_approved_pair_and_provenance(tmp_path):
    db = initialized_db(tmp_path)
    selected = basic_pair(db)
    assert len(selected["eligible_combinations"]) == 1
    result = generate_hybrid_journeys(selected)
    assert result["status"] == "generated"
    assert len(result["hybrid_journeys"]) == 1
    journey = result["hybrid_journeys"][0]
    assert journey["ordered_segment_ids"] == ["obs-source", "review-source"]
    assert journey["compatibility_rule_support"]
    assert journey["sequence_rule_support"]


def test_validated_sequence_orders_segments_and_positions_steps(tmp_path):
    db = initialized_db(tmp_path)
    selected = basic_pair(db)
    journey = generate_hybrid_journeys(selected)["hybrid_journeys"][0]
    assert [row["source_segment_id"] for row in journey["ordered_steps"]] == ["obs-source", "review-source"]
    assert [row["hybrid_position"] for row in journey["ordered_steps"]] == [1, 2]


def test_original_order_within_multistep_source_segment_is_preserved(tmp_path):
    db = initialized_db(tmp_path)
    compatibility = [
        {"category_a": "OBSERVATION", "category_b": "REVIEW", "status": "compatible", "rule_note": "explicit test compatible"},
        {"category_a": "REVIEW", "category_b": "SUPPORT", "status": "compatible", "rule_note": "explicit test compatible"},
        {"category_a": "OBSERVATION", "category_b": "SUPPORT", "status": "compatible", "rule_note": "explicit test compatible"},
    ]
    rules = custom_rules(tmp_path, compatibility=compatibility)
    segments = [
        make_segment("multi-source", "SYN001", ["ST001A", "ST001B"], [1, 2], ["first original", "second original"]),
        make_segment("support-source", "SYN002", ["ST002C"], [3], ["third original"]),
    ]
    selected = selection(db, segments, rules)
    assert len(selected["eligible_combinations"]) == 1
    journey = generate_hybrid_journeys(selected)["hybrid_journeys"][0]
    assert [row["source_step_id"] for row in journey["ordered_steps"]] == ["ST001A", "ST001B", "ST002C"]
    assert [row["original_step_order"] for row in journey["ordered_steps"]] == [1, 2, 3]


def test_each_merged_step_retains_required_provenance(tmp_path):
    db = initialized_db(tmp_path)
    journey = generate_hybrid_journeys(basic_pair(db))["hybrid_journeys"][0]
    first = journey["ordered_steps"][0]
    assert first == {
        "hybrid_journey_id": journey["hybrid_journey_id"],
        "hybrid_position": 1,
        "source_patient_id": "SYN001",
        "source_segment_id": "obs-source",
        "source_step_id": "ST001A",
        "original_step_order": 1,
        "treatment_description": "synthetic initial note",
        "original_outcome": 1,
        "original_segment_classification": "Positive",
        "source_similarity_score": 0.91,
    }
    assert journey["included_segments"][0]["source_patient_id"] == "SYN001"


def test_duplicate_segment_and_duplicate_step_inputs_are_rejected(tmp_path):
    db = initialized_db(tmp_path)
    result = basic_pair(db)
    duplicate_segment = copy.deepcopy(result)
    pair = duplicate_segment["eligible_combinations"][0]
    pair["segments"].append(copy.deepcopy(pair["segments"][0]))
    pair["segment_ids"].append(pair["segments"][0]["segment_id"])
    pair["ordered_segment_ids"].append(pair["segments"][0]["segment_id"])
    rejected = generate_hybrid_journeys(duplicate_segment)
    assert rejected["status"] == "no_valid_hybrid"
    assert "same source segment" in rejected["rejected_combinations"][0]["reason"]

    duplicate_step = copy.deepcopy(result)
    pair = duplicate_step["eligible_combinations"][0]
    pair["segments"][1]["source_step_ids"] = [pair["segments"][0]["source_step_ids"][0]]
    rejected = generate_hybrid_journeys(duplicate_step)
    assert rejected["status"] == "no_valid_hybrid"
    assert "duplicated" in rejected["rejected_combinations"][0]["reason"]


def test_negative_and_mixed_segments_cannot_be_merged(tmp_path):
    db = initialized_db(tmp_path)
    segments = [
        make_segment("positive-obs", "SYN001", ["ST001A"], [1], ["synthetic obs"]),
        make_segment("negative-review", "SYN002", ["ST002B"], [2], ["synthetic review"], classification="Negative", outcomes=[-1]),
        make_segment("mixed-support", "SYN003", ["ST003B"], [2], ["synthetic support"], classification="Mixed/Unclear", outcomes=[1]),
    ]
    selected = selection(db, segments)
    result = generate_hybrid_journeys(selected)
    assert result["status"] == "no_valid_hybrid"
    assert result["hybrid_journeys"] == []
    assert {row["segment"]["segment_id"] for row in result["rejected_segments"]} >= {"negative-review", "mixed-support"}


def test_unknown_compatibility_never_reaches_hybrid_generation(tmp_path):
    db = initialized_db(tmp_path)
    segments = [
        make_segment("obs-source", "SYN001", ["ST001A"], [1], ["obs"]),
        make_segment("support-source", "SYN002", ["ST002C"], [3], ["support"]),
    ]
    selected = selection(db, segments)
    assert selected["eligible_combinations"] == []
    result = generate_hybrid_journeys(selected)
    assert result["status"] == "no_valid_hybrid"


def test_unresolved_or_ambiguous_sequence_never_generates_hybrid(tmp_path):
    db = initialized_db(tmp_path)
    segments = [
        make_segment("obs-source", "SYN001", ["ST001A"], [1], ["obs"]),
        make_segment("review-source", "SYN002", ["ST002B"], [2], ["review"]),
    ]
    unresolved_rules = custom_rules(tmp_path, sequence=[])
    unresolved = selection(db, segments, unresolved_rules)
    assert unresolved["eligible_combinations"] == []
    assert generate_hybrid_journeys(unresolved)["status"] == "no_valid_hybrid"

    ambiguous_rules = custom_rules(tmp_path, sequence=[
        {"from_category": "OBSERVATION", "to_category": "REVIEW", "rule_type": "allowed", "rule_note": "forward"},
        {"from_category": "REVIEW", "to_category": "OBSERVATION", "rule_type": "allowed", "rule_note": "reverse"},
    ])
    ambiguous = selection(db, segments, ambiguous_rules)
    assert ambiguous["eligible_combinations"] == []
    assert generate_hybrid_journeys(ambiguous)["status"] == "no_valid_hybrid"


def test_no_valid_hybrid_returns_clear_status_and_reasons(tmp_path):
    result = generate_hybrid_journeys({"eligible_combinations": [], "rejected_segments": [{"reason": "not positive"}]})
    assert result["status"] == "no_valid_hybrid"
    assert "no path was forced" in result["message"]
    assert result["rejected_segments"] == [{"reason": "not positive"}]


def test_multiple_valid_combinations_remain_separate(tmp_path):
    db = initialized_db(tmp_path)
    segments = [
        make_segment("obs-one", "SYN001", ["ST001A"], [1], ["obs one"]),
        make_segment("obs-two", "SYN008", ["ST008A"], [1], ["obs two"]),
        make_segment("review-three", "SYN002", ["ST002B"], [2], ["review three"]),
    ]
    selected = selection(db, segments)
    assert len(selected["eligible_combinations"]) == 2
    result = generate_hybrid_journeys(selected)
    assert len(result["hybrid_journeys"]) == 2
    assert result["hybrid_journeys"][0]["hybrid_journey_id"] != result["hybrid_journeys"][1]["hybrid_journey_id"]
    assert all(len(journey["ordered_steps"]) == 2 for journey in result["hybrid_journeys"])


def test_generation_is_deterministic_for_identical_candidates(tmp_path):
    db = initialized_db(tmp_path)
    segments = [
        make_segment("obs-one", "SYN001", ["ST001A"], [1], ["obs one"]),
        make_segment("obs-two", "SYN008", ["ST008A"], [1], ["obs two"]),
        make_segment("review-three", "SYN002", ["ST002B"], [2], ["review three"]),
    ]
    selected = selection(db, segments)
    shuffled = copy.deepcopy(selected)
    shuffled["eligible_combinations"].reverse()
    first = generate_hybrid_journeys(selected)
    second = generate_hybrid_journeys(shuffled)
    assert [row["hybrid_journey_id"] for row in first["hybrid_journeys"]] == [row["hybrid_journey_id"] for row in second["hybrid_journeys"]]
    assert [row["ordered_steps"] for row in first["hybrid_journeys"]] == [row["ordered_steps"] for row in second["hybrid_journeys"]]
