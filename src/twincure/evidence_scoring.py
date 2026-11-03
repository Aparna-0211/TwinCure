"""Transparent synthetic evidence-support scoring for generated hybrid journeys.

The score summarizes synthetic evidence only. It is not a clinical success
probability, effectiveness prediction, or recommendation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

S_WEIGHT = 0.4118
O_WEIGHT = 0.3529
A_WEIGHT = 0.2353


def score_hybrid_evidence(hybrid_generation_result: dict[str, Any] | None) -> dict[str, Any]:
    """Score every hybrid in a Stage 4B result, reporting undefined evidence.

    S uses each source patient in the current hybrid once. O uses the unique
    source steps in positive segments of that hybrid, grouped once per patient.
    A is calculated separately for each hybrid, using only journeys represented
    in that hybrid. For those journeys, available candidate segments from the
    generation set and Stage 4A rejected-segment report are deduplicated before
    producing one positive/negative/neutral vote per journey.
    """
    if hybrid_generation_result is None or not isinstance(hybrid_generation_result, dict):
        return {
            "status": "no_hybrid_journeys",
            "results": [],
            "message": "No Stage 4B generation result was provided; evidence cannot be scored.",
        }
    hybrids = hybrid_generation_result.get("hybrid_journeys") or []
    if not hybrids:
        return {
            "status": "no_hybrid_journeys",
            "results": [],
            "message": "Stage 4B produced no hybrid journeys to score.",
        }

    candidate_segments, segment_conflicts = _collect_candidate_segments(hybrid_generation_result)
    results = [
        _score_one_hybrid(hybrid, candidate_segments, segment_conflicts)
        for hybrid in hybrids
    ]
    if all(result["status"] == "scored" for result in results):
        status = "scored"
        message = f"Calculated synthetic evidence-support scores for {len(results)} hybrid journey(s)."
    elif any(result["status"] == "scored" for result in results):
        status = "completed_with_insufficient_evidence"
        message = "Some hybrid journeys were scored; others lacked required evidence. See each journey result."
    else:
        status = "insufficient_evidence"
        message = "No hybrid journey had all required evidence components. See component-specific reasons."
    return {"status": status, "results": results, "message": message}


def _score_one_hybrid(
    hybrid: dict[str, Any],
    candidate_segments: dict[str, dict[str, Any]],
    segment_conflicts: set[str],
) -> dict[str, Any]:
    hybrid_id = hybrid.get("hybrid_journey_id")
    steps = hybrid.get("ordered_steps")
    errors: list[str] = []
    if not isinstance(steps, list) or not steps:
        steps = []
        errors.append("S and O are undefined because the hybrid has no ordered source steps.")
    similarity, similarity_details, similarity_error = _calculate_similarity(steps)
    if similarity_error:
        errors.append(similarity_error)
    outcome, outcome_details, outcome_error = _calculate_outcome_support(steps)
    if outcome_error:
        errors.append(outcome_error)
    contributing_patients = {
        str(step["source_patient_id"])
        for step in steps
        if step.get("source_patient_id") is not None
    }
    relevant_segments = {
        segment_id: segment
        for segment_id, segment in candidate_segments.items()
        if segment.get("source_patient_id") is not None
        and str(segment["source_patient_id"]) in contributing_patients
    }
    relevant_conflicts = segment_conflicts.intersection(contributing_patients)
    agreement = _calculate_agreement(relevant_segments, relevant_conflicts)
    agreement_value = agreement["value"]
    if agreement_value is None:
        errors.append(agreement["reason"])

    components = {
        "S": {"value": similarity, "weight": S_WEIGHT, "meaning": "Mean similarity across unique source journeys in this hybrid.", "calculations": similarity_details},
        "O": {"value": outcome, "weight": O_WEIGHT, "meaning": "Mean mapped outcome support across unique journeys contributing positive segments in this hybrid.", "calculations": outcome_details},
        "A": {"value": agreement_value, "weight": A_WEIGHT, "meaning": "Positive unique journey votes divided by positive plus negative unique journey votes, using only source journeys contributing to this hybrid.", "calculations": agreement["details"]},
    }
    complete = similarity is not None and outcome is not None and agreement_value is not None and not errors
    score = None
    if complete:
        score = round(100.0 * (S_WEIGHT * similarity + O_WEIGHT * outcome + A_WEIGHT * agreement_value), 2)
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            complete = False
            score = None
            errors.append("The calculated score was outside its valid 0–100 range and was withheld.")

    if complete:
        status = "scored"
        explanation = (
            f"Synthetic evidence support = 100 × ({S_WEIGHT:.4f}×S + {O_WEIGHT:.4f}×O + "
            f"{A_WEIGHT:.4f}×A) = {score:.2f}. Each source journey and source step was counted at most once "
            "within its component. This is not a clinical success probability or a claim that the hybrid works."
        )
    else:
        status = "insufficient_evidence"
        explanation = "No final score was calculated because required evidence is incomplete: " + " ".join(errors)
    return {
        "hybrid_journey_id": hybrid_id,
        "unique_contributing_patient_ids": sorted({str(step.get("source_patient_id")) for step in steps if step.get("source_patient_id") is not None}),
        "S": similarity,
        "O": outcome,
        "A": agreement_value,
        "components": components,
        "evidence_support_score": score,
        "status": status,
        "missing_evidence": errors,
        "explanation": explanation,
    }


def _calculate_similarity(steps: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]], str | None]:
    """Average the original Stage 2 similarity once per source patient."""
    scores_by_patient: dict[str, list[float | None]] = defaultdict(list)
    for step in steps:
        patient = step.get("source_patient_id")
        if patient is None:
            return None, [], "S is undefined because a source patient ID is missing."
        score = _number_or_none(step.get("source_similarity_score"))
        scores_by_patient[str(patient)].append(score)
    if not scores_by_patient:
        return None, [], "S is undefined because the hybrid has no source patient journeys."

    details: list[dict[str, Any]] = []
    unique_scores: list[float] = []
    for patient in sorted(scores_by_patient):
        values = scores_by_patient[patient]
        if any(value is None for value in values):
            return None, details, f"S is undefined because source journey {patient} has a missing or invalid similarity score."
        distinct = {float(value) for value in values if value is not None}
        if len(distinct) != 1:
            return None, details, f"S is undefined because source journey {patient} has conflicting similarity values across its steps."
        value = next(iter(distinct))
        if not 0.0 <= value <= 1.0:
            return None, details, f"S is undefined because source journey {patient} has a similarity score outside 0–1."
        unique_scores.append(value)
        details.append({"source_patient_id": patient, "similarity_score": value, "counted_once": True})
    return sum(unique_scores) / len(unique_scores), details, None


def _calculate_outcome_support(steps: list[dict[str, Any]]) -> tuple[float | None, list[dict[str, Any]], str | None]:
    """Map unique-step outcomes to support once per contributing journey."""
    by_patient: dict[str, dict[str, float | None]] = defaultdict(dict)
    segment_classifications: dict[str, set[str]] = defaultdict(set)
    for step in steps:
        patient = step.get("source_patient_id")
        step_id = step.get("source_step_id")
        if patient is None or step_id is None:
            return None, [], "O is undefined because selected positive-segment provenance is missing a patient or step ID."
        classification = step.get("original_segment_classification")
        if classification != "Positive":
            return None, [], f"O is undefined because hybrid step {step_id} is not from a Positive segment."
        patient_id, original_step_id = str(patient), str(step_id)
        segment_classifications[patient_id].add(str(classification))
        outcome = _number_or_none(step.get("original_outcome"))
        if outcome is not None and outcome not in (-1.0, 0.0, 1.0):
            return None, [], f"O is undefined because source step {original_step_id} has an invalid outcome value."
        if original_step_id in by_patient[patient_id]:
            previous = by_patient[patient_id][original_step_id]
            if previous != outcome:
                return None, [], f"O is undefined because repeated source step {original_step_id} has conflicting outcome values."
        else:
            by_patient[patient_id][original_step_id] = outcome
    if not by_patient:
        return None, [], "O is undefined because no selected positive source journeys contribute to the hybrid."

    details: list[dict[str, Any]] = []
    individual_supports: list[float] = []
    for patient in sorted(by_patient):
        outcomes = by_patient[patient]
        if any(value is None for value in outcomes.values()):
            return None, details, f"O is undefined because a selected positive segment from journey {patient} has a missing outcome."
        mean = sum(float(value) for value in outcomes.values() if value is not None) / len(outcomes)
        support = (mean + 1.0) / 2.0
        individual_supports.append(support)
        details.append({
            "source_patient_id": patient,
            "unique_source_step_ids": sorted(outcomes),
            "unique_step_outcomes": [outcomes[step_id] for step_id in sorted(outcomes)],
            "mean_outcome": mean,
            "individual_outcome_support": support,
            "each_step_counted_once": True,
        })
    return sum(individual_supports) / len(individual_supports), details, None


def _collect_candidate_segments(
    generation_result: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Collect unique candidate segments for agreement, including Stage 4A rejections.

    Positive eligible segments are recovered from all alternative hybrids and
    deduplicated by segment ID. Rejected negative/mixed/unpaired segments are
    supplied unchanged by Stage 4B from Stage 4A. Unique source steps then drive
    one journey-level vote, regardless of overlapping candidate segments.
    """
    segments: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for hybrid in generation_result.get("hybrid_journeys", []) or []:
        if not isinstance(hybrid, dict):
            continue
        grouped: dict[str, dict[str, Any]] = {}
        for step in hybrid.get("ordered_steps", []) or []:
            segment_id = step.get("source_segment_id")
            if segment_id is None:
                continue
            sid = str(segment_id)
            item = grouped.setdefault(sid, {
                "segment_id": sid,
                "source_patient_id": step.get("source_patient_id"),
                "classification": step.get("original_segment_classification"),
                "source_step_ids": [],
                "step_outcomes": [],
            })
            item["source_step_ids"].append(step.get("source_step_id"))
            item["step_outcomes"].append(step.get("original_outcome"))
        for sid, item in grouped.items():
            _add_segment(segments, conflicts, sid, item)

    for rejected in generation_result.get("rejected_segments", []) or []:
        if not isinstance(rejected, dict) or not isinstance(rejected.get("segment"), dict):
            continue
        item = rejected["segment"]
        sid = item.get("segment_id")
        if sid is None:
            continue
        _add_segment(segments, conflicts, str(sid), item)
    return segments, conflicts


def _add_segment(segments: dict[str, dict[str, Any]], conflicts: set[str], sid: str, item: dict[str, Any]) -> None:
    """Store one segment once; flag inconsistent duplicate provenance."""
    normalized = {
        "segment_id": sid,
        "source_patient_id": item.get("source_patient_id"),
        "classification": item.get("classification", item.get("original_segment_classification")),
        "source_step_ids": list(item.get("source_step_ids") or []),
        "step_outcomes": list(item.get("step_outcomes") or []),
    }
    prior = segments.get(sid)
    if prior is None:
        segments[sid] = normalized
    elif prior != normalized:
        conflicts.add(str(normalized.get("source_patient_id") or prior.get("source_patient_id") or ""))


def _calculate_agreement(
    segments: dict[str, dict[str, Any]], conflicts: set[str]
) -> dict[str, Any]:
    """Give each candidate source journey one positive/negative/neutral vote."""
    journey_steps: dict[str, dict[str, float | None]] = defaultdict(dict)
    for segment in segments.values():
        patient = segment.get("source_patient_id")
        if patient is None:
            continue
        patient_id = str(patient)
        step_ids = segment.get("source_step_ids") or []
        outcomes = segment.get("step_outcomes") or []
        if len(step_ids) != len(outcomes):
            conflicts.add(patient_id)
            continue
        for step_id, raw_outcome in zip(step_ids, outcomes):
            if step_id is None:
                conflicts.add(patient_id)
                continue
            key = str(step_id)
            outcome = _number_or_none(raw_outcome)
            if outcome is not None and outcome not in (-1.0, 0.0, 1.0):
                outcome = None
            if key in journey_steps[patient_id] and journey_steps[patient_id][key] != outcome:
                conflicts.add(patient_id)
            else:
                journey_steps[patient_id][key] = outcome

    votes: list[dict[str, Any]] = []
    positive: list[str] = []
    negative: list[str] = []
    neutral: list[str] = []
    for patient in sorted(journey_steps):
        outcomes = journey_steps[patient]
        if patient in conflicts or not outcomes or any(value is None for value in outcomes.values()):
            vote = "neutral/unclear"
            mean = None
        else:
            numeric = [float(value) for value in outcomes.values() if value is not None]
            mean = sum(numeric) / len(numeric)
            if mean >= 0.5 and 1.0 in numeric:
                vote = "positive"
                positive.append(patient)
            elif mean <= -0.5 and -1.0 in numeric:
                vote = "negative"
                negative.append(patient)
            else:
                vote = "neutral/unclear"
                neutral.append(patient)
        votes.append({"source_patient_id": patient, "unique_source_step_ids": sorted(outcomes), "mean_outcome": mean, "vote": vote})
    denominator = len(positive) + len(negative)
    if denominator == 0:
        value = None
        reason = "A is undefined because the candidate journeys contain no positive or negative unique-journey votes (the denominator is zero)."
    else:
        value = len(positive) / denominator
        reason = None
    return {
        "value": value,
        "reason": reason,
        "details": {
            "candidate_patient_ids": sorted(journey_steps),
            "positive_journey_ids": positive,
            "negative_journey_ids": negative,
            "neutral_or_unclear_journey_ids": neutral,
            "votes": votes,
            "positive_vote_count": len(positive),
            "negative_vote_count": len(negative),
            "denominator": denominator,
            "each_journey_counted_once": True,
            "each_source_step_counted_once": True,
        },
    }


def _number_or_none(value: Any) -> float | None:
    """Return finite numeric values, leaving null or invalid data unavailable."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
