import copy

import pytest

from twincure.evidence_scoring import score_hybrid_evidence


def step(patient, segment, step_id, order, outcome, similarity, classification="Positive"):
    return {
        "hybrid_journey_id": "HYB-TEST",
        "hybrid_position": order,
        "source_patient_id": patient,
        "source_segment_id": segment,
        "source_step_id": step_id,
        "original_step_order": order,
        "treatment_description": f"synthetic description {step_id}",
        "original_outcome": outcome,
        "original_segment_classification": classification,
        "source_similarity_score": similarity,
    }


def result(steps, rejected_segments=None, hybrid_id="HYB-TEST"):
    for row in steps:
        row["hybrid_journey_id"] = hybrid_id
    return {
        "status": "generated",
        "hybrid_journeys": [{"hybrid_journey_id": hybrid_id, "ordered_steps": steps}],
        "rejected_segments": rejected_segments or [],
    }


def rejected_segment(segment_id, patient, step_ids, outcomes, classification):
    return {"segment": {
        "segment_id": segment_id,
        "source_patient_id": patient,
        "source_step_ids": step_ids,
        "step_outcomes": outcomes,
        "classification": classification,
    }, "reason": "synthetic Stage 4A rejection"}


def base_result():
    steps = [
        step("SYN001", "SEG-A", "ST001A", 1, 1, 0.9),
        step("SYN002", "SEG-B", "ST002B", 1, 1, 0.7),
    ]
    rejected = [
        rejected_segment("SEG-NEG", "SYN003", ["ST003A"], [-1], "Negative"),
        rejected_segment("SEG-NEUTRAL", "SYN004", ["ST004A"], [0], "Mixed/Unclear"),
    ]
    return result(steps, rejected)


def test_similarity_support_averages_unique_source_journeys():
    scored = score_hybrid_evidence(base_result())["results"][0]
    assert scored["S"] == pytest.approx(0.8)
    details = scored["components"]["S"]["calculations"]
    assert [row["source_patient_id"] for row in details] == ["SYN001", "SYN002"]
    assert all(row["counted_once"] for row in details)


def test_outcome_support_maps_journey_means_and_averages_patients():
    rows = [
        step("A", "AS", "A1", 1, 1, 0.8),
        step("A", "AS", "A2", 2, 0, 0.8),  # journey mean .5 -> support .75
        step("B", "BS", "B1", 1, 1, 0.6),  # support 1.0
    ]
    scored = score_hybrid_evidence(result(rows))["results"][0]
    assert scored["O"] == pytest.approx(0.875)
    detail = scored["components"]["O"]["calculations"]
    assert [item["individual_outcome_support"] for item in detail] == [0.75, 1.0]


def test_agreement_counts_unique_positive_and_negative_journeys_and_excludes_neutral():
    scored = score_hybrid_evidence(base_result())["results"][0]
    assert scored["A"] == pytest.approx(1.0)
    details = scored["components"]["A"]["calculations"]
    assert details["positive_vote_count"] == 2
    assert details["negative_vote_count"] == 0
    assert details["denominator"] == 2
    assert details["candidate_patient_ids"] == ["SYN001", "SYN002"]


def test_final_weighted_score_uses_approved_formula():
    scored = score_hybrid_evidence(base_result())["results"][0]
    expected = round(100 * (0.4118 * 0.8 + 0.3529 * 1.0 + 0.2353 * 1.0), 2)
    assert scored["evidence_support_score"] == expected
    assert scored["status"] == "scored"


def test_repeated_steps_from_one_patient_do_not_multiply_similarity_or_vote():
    rows = [
        step("A", "A1", "STEP-1", 1, 1, 0.8),
        step("A", "A2", "STEP-1", 1, 1, 0.8),  # same source step repeated in another segment
        step("B", "B1", "STEP-2", 1, 1, 0.4),
    ]
    # An unrelated negative source must not contribute to this hybrid's A.
    scored = score_hybrid_evidence(result(rows, [rejected_segment("NEG", "C", ["STEP-3"], [-1], "Negative")]))["results"][0]
    assert scored["S"] == pytest.approx(0.6)  # A=.8 and B=.4; repeated A only once
    assert scored["O"] == pytest.approx(1.0)  # STEP-1 appears once in journey A
    agreement = scored["components"]["A"]["calculations"]
    assert agreement["positive_vote_count"] == 2
    assert agreement["negative_vote_count"] == 0
    assert agreement["denominator"] == 2


def test_multiple_segments_and_overlapping_steps_contribute_once_to_agreement():
    rows = [
        step("A", "A-SEG1", "A1", 1, 1, 0.8),
        step("A", "A-SEG2", "A2", 2, 0, 0.8),
        step("B", "B-SEG", "B1", 1, 1, 0.7),
    ]
    rejected = [rejected_segment("A-OVERLAP", "A", ["A1"], [1], "Positive")]
    scored = score_hybrid_evidence(result(rows, rejected))["results"][0]
    votes = scored["components"]["A"]["calculations"]["votes"]
    vote_a = next(v for v in votes if v["source_patient_id"] == "A")
    assert vote_a["unique_source_step_ids"] == ["A1", "A2"]
    assert vote_a["mean_outcome"] == pytest.approx(0.5)
    assert vote_a["vote"] == "positive"
    assert scored["components"]["A"]["calculations"]["positive_vote_count"] == 2


def test_neutral_journeys_are_not_in_agreement_denominator():
    evidence = base_result()
    evidence["rejected_segments"].append(
        rejected_segment("SEG-P2-NEG", "SYN002", ["SYN002-NEG"], [-1], "Negative")
    )
    scored = score_hybrid_evidence(evidence)["results"][0]
    details = scored["components"]["A"]["calculations"]
    assert details["positive_journey_ids"] == ["SYN001"]
    assert details["negative_journey_ids"] == []
    assert details["neutral_or_unclear_journey_ids"] == ["SYN002"]
    assert details["denominator"] == 1


def test_unrelated_patient_cannot_change_hybrid_agreement_or_score():
    baseline = result([
        step("A", "A-SEG", "A1", 1, 1, 0.9),
        step("B", "B-SEG", "B1", 1, 1, 0.7),
    ])
    with_unrelated = copy.deepcopy(baseline)
    with_unrelated["rejected_segments"] = [
        rejected_segment("UNRELATED-NEG", "UNRELATED", ["U1"], [-1], "Negative"),
        rejected_segment("UNRELATED-POS", "UNRELATED2", ["U2"], [1], "Positive"),
    ]
    before = score_hybrid_evidence(baseline)["results"][0]
    after = score_hybrid_evidence(with_unrelated)["results"][0]
    assert before["A"] == after["A"] == 1.0
    assert before["evidence_support_score"] == after["evidence_support_score"]
    assert after["components"]["A"]["calculations"]["candidate_patient_ids"] == ["A", "B"]


def test_rejected_candidate_evidence_for_contributing_journey_affects_its_vote():
    evidence = result([
        step("A", "A-SEG", "A1", 1, 1, 0.9),
        step("B", "B-SEG", "B1", 1, 1, 0.7),
    ])
    # This rejected negative segment belongs to B, who contributes to this hybrid.
    evidence["rejected_segments"] = [
        rejected_segment("B-NEG1", "B", ["B-NEG1", "B-NEG2"], [-1, -1], "Negative"),
        rejected_segment("B-NEG2", "B", ["B-NEG2", "B-NEG3"], [-1, -1], "Negative"),
    ]
    scored = score_hybrid_evidence(evidence)["results"][0]
    votes = scored["components"]["A"]["calculations"]["votes"]
    vote_b = next(item for item in votes if item["source_patient_id"] == "B")
    assert vote_b["unique_source_step_ids"] == ["B-NEG1", "B-NEG2", "B-NEG3", "B1"]
    assert vote_b["vote"] == "negative"
    assert scored["A"] == pytest.approx(0.5)


def test_zero_agreement_denominator_is_insufficient_evidence():
    rows = [
        step("A", "A1", "A1", 1, 1, 0.8),
        step("B", "B1", "B1", 1, 1, 0.6),
    ]
    rejected = [
        rejected_segment("A-NEG", "A", ["A-NEG-STEP"], [-1], "Negative"),
        rejected_segment("B-NEG", "B", ["B-NEG-STEP"], [-1], "Negative"),
    ]
    scored = score_hybrid_evidence(result(rows, rejected))["results"][0]
    assert scored["A"] is None
    assert scored["evidence_support_score"] is None
    assert scored["status"] == "insufficient_evidence"
    assert any("denominator is zero" in reason for reason in scored["missing_evidence"])


def test_missing_similarity_is_not_filled_and_withholds_final_score():
    rows = [
        step("A", "A1", "A1", 1, 1, None),
        step("B", "B1", "B1", 1, 1, 0.6),
    ]
    scored = score_hybrid_evidence(result(rows))["results"][0]
    assert scored["S"] is None
    assert scored["evidence_support_score"] is None
    assert scored["status"] == "insufficient_evidence"
    assert any("missing or invalid similarity" in reason for reason in scored["missing_evidence"])


def test_missing_selected_outcome_is_not_fabricated():
    rows = [
        step("A", "A1", "A1", 1, None, 0.8),
        step("B", "B1", "B1", 1, 1, 0.6),
    ]
    scored = score_hybrid_evidence(result(rows))["results"][0]
    assert scored["O"] is None
    assert scored["evidence_support_score"] is None
    assert scored["status"] == "insufficient_evidence"
    assert any("missing outcome" in reason for reason in scored["missing_evidence"])


def test_all_required_components_must_be_available():
    scored = score_hybrid_evidence({"hybrid_journeys": [{"hybrid_journey_id": "EMPTY", "ordered_steps": []}], "rejected_segments": []})["results"][0]
    assert scored["status"] == "insufficient_evidence"
    assert scored["evidence_support_score"] is None
    assert scored["S"] is None and scored["O"] is None and scored["A"] is None


def test_scores_stay_between_zero_and_one_hundred():
    result_max = result([
        step("A", "A1", "A1", 1, 1, 1.0),
        step("B", "B1", "B1", 1, 1, 1.0),
    ])
    scored = score_hybrid_evidence(result_max)["results"][0]
    assert scored["evidence_support_score"] == 100.0
    ordinary = score_hybrid_evidence(base_result())["results"][0]
    assert 0.0 <= ordinary["evidence_support_score"] <= 100.0


def test_identical_inputs_produce_identical_scores_and_details():
    original = base_result()
    first = score_hybrid_evidence(original)
    second = score_hybrid_evidence(copy.deepcopy(original))
    assert first == second
