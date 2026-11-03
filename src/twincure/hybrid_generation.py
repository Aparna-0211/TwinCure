"""Create traceable hybrid candidates from Stage 4A-approved combinations.

This module trusts Stage 4A's explicit compatibility and sequence validation and
never derives a new ordering or compatibility decision. Each approved combination
becomes a separate proposed synthetic journey; no confidence score is calculated.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd

STEP_FIELDS = (
    "source_step_ids",
    "step_orders",
    "treatment_descriptions",
    "step_outcomes",
)


def generate_hybrid_journeys(selection_result: dict[str, Any] | None) -> dict[str, Any]:
    """Generate one separate hybrid candidate per Stage 4A-eligible combination.

    The function uses only `eligible_combinations` from Stage 4A. Each input
    combination must contain positive segments, distinct source journeys, and
    Stage 4A's explicit order. Invalid shapes, duplicate segments/steps, and
    duplicate combinations are rejected with reasons. No segments are merged
    across alternative candidates.
    """
    if selection_result is None:
        return _empty_result("No Stage 4A selection result was provided.")
    if not isinstance(selection_result, dict):
        raise ValueError("selection_result must be the dictionary returned by Stage 4A")

    candidates = selection_result.get("eligible_combinations") or []
    inherited_rejections = list(selection_result.get("rejected_segments") or [])
    rejected_combinations: list[dict[str, Any]] = []
    hybrids: list[dict[str, Any]] = []
    seen_combinations: set[tuple[str, ...]] = set()

    # Sorting by pair ID makes generation stable even if Stage 4A candidates
    # arrive in a different list order.
    ordered_candidates = sorted(
        candidates,
        key=lambda item: str(item.get("pair_id", "")) if isinstance(item, dict) else "",
    )
    for candidate in ordered_candidates:
        if not isinstance(candidate, dict):
            rejected_combinations.append({"pair_id": None, "reason": "Stage 4A candidate is not a structured object."})
            continue
        reason = _validate_candidate(candidate)
        if reason:
            rejected_combinations.append({"pair_id": candidate.get("pair_id"), "segment_ids": candidate.get("segment_ids"), "reason": reason})
            continue

        segments = candidate["segments"]
        by_id = {str(segment["segment_id"]): segment for segment in segments}
        ordered_ids = tuple(str(segment_id) for segment_id in candidate["ordered_segment_ids"])
        if ordered_ids in seen_combinations:
            rejected_combinations.append({
                "pair_id": candidate.get("pair_id"),
                "segment_ids": list(ordered_ids),
                "reason": "Duplicate eligible combination omitted; the same source segments already form a candidate journey.",
            })
            continue
        seen_combinations.add(ordered_ids)

        hybrid_id = _hybrid_id(ordered_ids)
        step_rows: list[dict[str, Any]] = []
        segment_explanations: list[dict[str, Any]] = []
        position = 1
        for segment_id in ordered_ids:
            segment = by_id[segment_id]
            segment_explanations.append({
                "source_segment_id": segment_id,
                "source_patient_id": str(segment["source_patient_id"]),
                "reason": (
                    "Included because Stage 4A selected this Positive segment, confirmed explicit compatibility, "
                    "and established the unique sequence order."
                ),
            })
            values = {field: segment[field] for field in STEP_FIELDS}
            for index, (step_id, step_order, description, outcome) in enumerate(zip(
                values["source_step_ids"],
                values["step_orders"],
                values["treatment_descriptions"],
                values["step_outcomes"],
            )):
                step_rows.append({
                    "hybrid_journey_id": hybrid_id,
                    "hybrid_position": position,
                    "source_patient_id": str(segment["source_patient_id"]),
                    "source_segment_id": segment_id,
                    "source_step_id": step_id,
                    "original_step_order": step_order,
                    "treatment_description": _preserve_missing(description),
                    "original_outcome": _preserve_missing(outcome),
                    "original_segment_classification": segment["classification"],
                    "source_similarity_score": _preserve_missing(segment.get("source_similarity_score")),
                })
                position += 1

        hybrids.append({
            "hybrid_journey_id": hybrid_id,
            "status": "generated",
            "ordered_segment_ids": list(ordered_ids),
            "source_patient_ids": [str(by_id[sid]["source_patient_id"]) for sid in ordered_ids],
            "ordered_steps": step_rows,
            "included_segments": segment_explanations,
            "compatibility_rule_support": candidate.get("compatibility_reason"),
            "sequence_rule_support": candidate.get("sequence_reason"),
            "pair_id": candidate.get("pair_id"),
        })

    if hybrids:
        status = "generated"
        message = f"Generated {len(hybrids)} separate hybrid candidate journey(s) from Stage 4A-approved combinations."
    else:
        status = "no_valid_hybrid"
        message = "No Stage 4A-approved combination could produce a valid hybrid journey; no path was forced."
    return {
        "status": status,
        "message": message,
        "hybrid_journeys": hybrids,
        "rejected_combinations": rejected_combinations,
        "rejected_segments": inherited_rejections,
    }


def _validate_candidate(candidate: dict[str, Any]) -> str | None:
    """Check structural safety without recalculating Stage 4A's rule decisions."""
    segments = candidate.get("segments")
    ordered_ids = candidate.get("ordered_segment_ids")
    if not isinstance(segments, list) or len(segments) < 2:
        return "Candidate must contain at least two Stage 4A-approved source segments."
    if not isinstance(ordered_ids, list) or len(ordered_ids) != len(segments):
        return "Stage 4A did not provide a complete, explicit segment order."
    segment_ids: list[str] = []
    patient_ids: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            return "Candidate contains a malformed source segment."
        if segment.get("classification") != "Positive":
            return "Only Positive segments can be included in a hybrid candidate."
        if segment.get("segment_id") is None or segment.get("source_patient_id") is None:
            return "Source segment ID or patient ID is missing."
        segment_ids.append(str(segment["segment_id"]))
        patient_ids.append(str(segment["source_patient_id"]))
        for field in STEP_FIELDS:
            if not isinstance(segment.get(field), (list, tuple)):
                return f"Source segment {segment['segment_id']} is missing its {field} provenance list."
        lengths = {len(segment[field]) for field in STEP_FIELDS}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            return f"Source segment {segment['segment_id']} has incomplete or empty step provenance."
        step_ids = segment["source_step_ids"]
        orders = segment["step_orders"]
        if any(step_id is None or _is_missing(step_id) for step_id in step_ids):
            return f"Source segment {segment['segment_id']} has a missing step ID."
        if len({str(step_id) for step_id in step_ids}) != len(step_ids):
            return f"Source segment {segment['segment_id']} repeats a source step ID."
        try:
            if any(float(orders[i]) >= float(orders[i + 1]) for i in range(len(orders) - 1)):
                return f"Source segment {segment['segment_id']} does not preserve strictly increasing original step order."
        except (TypeError, ValueError):
            return f"Source segment {segment['segment_id']} has an invalid original step order."

    if len(set(segment_ids)) != len(segment_ids):
        return "The same source segment appears more than once in this candidate."
    if len(set(patient_ids)) != len(patient_ids):
        return "A source patient journey contributes more than once to this hybrid component."
    if set(segment_ids) != {str(item) for item in ordered_ids} or len(set(map(str, ordered_ids))) != len(ordered_ids):
        return "Stage 4A's explicit order does not match the candidate's unique source segments."
    if not candidate.get("compatibility_reason") or not candidate.get("sequence_reason"):
        return "Stage 4A compatibility or sequence rule support is missing."
    all_steps = [str(step_id) for segment in segments for step_id in segment["source_step_ids"]]
    if len(set(all_steps)) != len(all_steps):
        return "A source treatment step would be duplicated in the proposed hybrid."
    return None


def _hybrid_id(ordered_segment_ids: tuple[str, ...]) -> str:
    """Create a stable unique ID from the exact, validated ordered segment list."""
    canonical = "\x1f".join(ordered_segment_ids).encode("utf-8")
    return "HYB-" + hashlib.sha256(canonical).hexdigest()[:16].upper()


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except (TypeError, ValueError):
        return False


def _preserve_missing(value: Any) -> Any:
    """Return missing scalar data as None; never synthesize a replacement value."""
    return None if value is None or _is_missing(value) else value


def _empty_result(message: str) -> dict[str, Any]:
    return {
        "status": "no_valid_hybrid",
        "message": message,
        "hybrid_journeys": [],
        "rejected_combinations": [],
        "rejected_segments": [],
    }
