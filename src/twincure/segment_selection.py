"""Select positive synthetic segments and validate pairwise compatibility/order.

This stage reports candidate combinations only. It does not merge segments or
create a treatment journey. Rules are read from the project's explicit JSON.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .db import connect

SELECTION_COLUMNS = [
    "segment_id", "source_patient_id", "source_step_ids", "step_orders",
    "treatment_descriptions", "step_outcomes", "mean_outcome", "classification",
    "source_similarity_score", "start_step_order", "end_step_order",
    "outcomes_complete",
]


def select_candidate_segments(
    segments: pd.DataFrame,
    db_path: str | Path,
    rules_path: str | Path,
) -> dict[str, Any]:
    """Separate positive candidates and validate distinct-journey pairs.

    Returns candidate segments, reasons for non-positive rejection, all pairwise
    compatibility and sequence results, candidate pairs eligible for a later
    stage, and per-segment decisions. The pair is a validation component only;
    no steps are concatenated or reordered here.
    """
    if segments is None or segments.empty:
        empty = pd.DataFrame(columns=SELECTION_COLUMNS)
        return {
            "positive_candidates": empty.copy(),
            "rejected_segments": [],
            "compatibility_results": [],
            "sequence_results": [],
            "eligible_combinations": [],
            "eligible_segments": empty.copy(),
            "selection_decisions": [],
        }

    required = {"segment_id", "source_patient_id", "source_step_ids", "step_orders", "classification"}
    missing = required.difference(segments.columns)
    if missing:
        raise ValueError(f"Segments are missing required columns: {', '.join(sorted(missing))}")

    # One source segment can only be one candidate, even if duplicated in input.
    data = segments.drop_duplicates(subset=["segment_id"], keep="first").copy()
    data["source_patient_id"] = data["source_patient_id"].astype(str)
    positive = data[data["classification"] == "Positive"].copy().reset_index(drop=True)
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in data[data["classification"] != "Positive"].to_dict("records"):
        reason = f"Not selected: segment classification is {row.get('classification')!r}; only Positive segments are candidates."
        rejected.append({"segment": row, "reason": reason})
        decisions.append({"segment_id": row["segment_id"], "decision": "rejected", "reason": reason})

    if positive.empty:
        return {
            "positive_candidates": positive,
            "rejected_segments": rejected,
            "compatibility_results": [],
            "sequence_results": [],
            "eligible_combinations": [],
            "eligible_segments": positive.copy(),
            "selection_decisions": decisions,
        }

    rules = _load_rules(rules_path)
    categories = _load_step_categories(db_path, positive)
    compatibility_results: list[dict[str, Any]] = []
    sequence_results: list[dict[str, Any]] = []
    eligible_combinations: list[dict[str, Any]] = []
    eligible_ids: set[str] = set()

    # Candidate components are pairwise comparisons between distinct journeys.
    # Segments from one patient are never combined or used to multiply that
    # patient's contribution inside a single pair/component.
    records = positive.to_dict("records")
    for first, second in combinations(records, 2):
        if first["source_patient_id"] == second["source_patient_id"]:
            decisions.append({
                "segment_ids": [first["segment_id"], second["segment_id"]],
                "decision": "not_compared_same_journey",
                "reason": "Segments share a source patient journey; this pair is not a cross-journey candidate component.",
            })
            continue
        pair_id = "PAIR::" + "::".join(sorted([str(first["segment_id"]), str(second["segment_id"])]))
        compat = _check_compatibility(first, second, categories, rules)
        compat["pair_id"] = pair_id
        compatibility_results.append(compat)
        sequence = _check_sequence(first, second, categories, rules)
        sequence["pair_id"] = pair_id
        sequence_results.append(sequence)

        eligible = compat["status"] == "compatible" and sequence["status"] == "ordered"
        if eligible:
            eligible_combinations.append({
                "pair_id": pair_id,
                "segment_ids": [first["segment_id"], second["segment_id"]],
                "source_patient_ids": [first["source_patient_id"], second["source_patient_id"]],
                "ordered_segment_ids": sequence["ordered_segment_ids"],
                "compatibility_reason": compat["reason"],
                "sequence_reason": sequence["reason"],
                "segments": [first, second],
            })
            eligible_ids.update([str(first["segment_id"]), str(second["segment_id"])])
            decisions.append({"pair_id": pair_id, "decision": "eligible_for_later_merging", "reason": "Explicit compatibility and a unique allowed sequence are both present."})
        else:
            why = []
            if compat["status"] != "compatible":
                why.append(f"compatibility is {compat['status']}: {compat['reason']}")
            if sequence["status"] != "ordered":
                why.append(f"sequence is {sequence['status']}: {sequence['reason']}")
            decisions.append({"pair_id": pair_id, "decision": "not_eligible", "reason": "; ".join(why)})

    eligible_segments = positive[positive["segment_id"].astype(str).isin(eligible_ids)].copy().reset_index(drop=True)
    used_ids = eligible_ids
    for row in records:
        sid = str(row["segment_id"])
        if sid not in used_ids:
            reason = "Positive segment retained as a candidate, but it has no pair with both explicitly compatible categories and one rule-established order."
            rejected.append({"segment": row, "reason": reason})
            decisions.append({"segment_id": sid, "decision": "not_eligible_for_later_merging", "reason": reason})

    return {
        "positive_candidates": positive,
        "rejected_segments": rejected,
        "compatibility_results": compatibility_results,
        "sequence_results": sequence_results,
        "eligible_combinations": eligible_combinations,
        "eligible_segments": eligible_segments,
        "selection_decisions": decisions,
    }


def _load_rules(path: str | Path) -> dict[str, Any]:
    """Load the provided explicit synthetic compatibility/sequence rules."""
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise ValueError("Rules JSON must contain an object")
    return rules


def _load_step_categories(db_path: str | Path, segments: pd.DataFrame) -> dict[str, str | None]:
    """Fetch category labels for the source step IDs referenced by segments."""
    step_ids: set[str] = set()
    for values in segments["source_step_ids"]:
        if isinstance(values, (list, tuple)):
            step_ids.update(str(value) for value in values if value is not None and not pd.isna(value))
    if not step_ids:
        return {}
    identifiers = sorted(step_ids)
    placeholders = ",".join("?" for _ in identifiers)
    with connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT step_id, step_category FROM journey_steps WHERE step_id IN ({placeholders})",
            identifiers,
        ).fetchall()
    return {str(row["step_id"]): row["step_category"] for row in rows}


def _categories(segment: dict[str, Any], category_map: dict[str, str | None]) -> list[str | None]:
    """Resolve ordered categories using the source step IDs, preserving boundaries."""
    step_ids = segment.get("source_step_ids")
    if not isinstance(step_ids, (list, tuple)):
        return []
    return [category_map.get(str(step_id)) for step_id in step_ids]


def _check_compatibility(
    first: dict[str, Any], second: dict[str, Any], category_map: dict[str, str | None], rules: dict[str, Any]
) -> dict[str, Any]:
    """Require explicit compatible rules for every cross-segment category pair."""
    first_categories = _categories(first, category_map)
    second_categories = _categories(second, category_map)
    rule_index: dict[frozenset[str], dict[str, Any]] = {}
    for rule in rules.get("compatibility_rules", []):
        a, b = rule.get("category_a"), rule.get("category_b")
        if a is not None and b is not None:
            rule_index[frozenset((str(a), str(b)))] = rule

    checks: list[dict[str, Any]] = []
    for a in first_categories:
        for b in second_categories:
            rule = rule_index.get(frozenset((str(a), str(b)))) if a is not None and b is not None else None
            status = rule.get("status") if rule else "unknown"
            reason = rule.get("rule_note", "No explicit compatibility rule exists for this category pair.") if rule else "No explicit compatibility rule exists for this category pair."
            checks.append({"category_a": a, "category_b": b, "status": status, "reason": reason})
    if not checks:
        status, reason = "unknown", "Source step categories are unavailable; compatibility cannot be confirmed."
    elif any(item["status"] == "incompatible" for item in checks):
        status = "incompatible"
        reason = "At least one explicitly incompatible category pair was found."
    elif all(item["status"] == "compatible" for item in checks):
        status = "compatible"
        reason = "Every cross-segment category pair has an explicit compatible rule."
    else:
        status = "unknown"
        reason = "At least one category pair is unknown or has no explicit rule; compatibility is unconfirmed."
    return {"segment_ids": [first["segment_id"], second["segment_id"]], "status": status, "reason": reason, "checks": checks}


def _check_sequence(
    first: dict[str, Any], second: dict[str, Any], category_map: dict[str, str | None], rules: dict[str, Any]
) -> dict[str, Any]:
    """Find whether exactly one segment order is supported by explicit directed rules."""
    left_categories = _categories(first, category_map)
    right_categories = _categories(second, category_map)
    if not left_categories or not right_categories or any(c is None for c in left_categories + right_categories):
        return {"segment_ids": [first["segment_id"], second["segment_id"]], "status": "unresolved", "ordered_segment_ids": None, "valid_orderings": [], "reason": "Source categories are missing; no order can be established."}

    allowed = {(r.get("from_category"), r.get("to_category")) for r in rules.get("sequence_rules", []) if r.get("rule_type") == "allowed"}
    prohibited = {(r.get("from_category"), r.get("to_category")) for r in rules.get("sequence_rules", []) if r.get("rule_type") == "prohibited"}

    def internal_status(categories: list[str]) -> str:
        """Require each original adjacent step pair to have an allowed rule."""
        edges = list(zip(categories, categories[1:]))
        if any(edge in prohibited for edge in edges):
            return "invalid"
        if any(edge not in allowed for edge in edges):
            return "unresolved"
        return "valid"

    internal_first = internal_status(left_categories)
    internal_second = internal_status(right_categories)
    if "invalid" in (internal_first, internal_second):
        return {"segment_ids": [first["segment_id"], second["segment_id"]], "status": "invalid", "ordered_segment_ids": None, "valid_orderings": [], "reason": "At least one original within-segment transition is explicitly prohibited by the sequence rules."}
    if "unresolved" in (internal_first, internal_second):
        return {"segment_ids": [first["segment_id"], second["segment_id"]], "status": "unresolved", "ordered_segment_ids": None, "valid_orderings": [], "reason": "At least one original within-segment transition lacks an explicit allowed sequence rule."}

    def boundary_status(before: list[str], after: list[str]) -> str:
        # Original order within each source segment is retained; explicitly
        # validate the new boundary needed to place unrelated journeys together.
        edge = (before[-1], after[0])
        if edge in prohibited:
            return "invalid"
        return "valid" if edge in allowed else "unresolved"

    valid: list[list[str]] = []
    forward_status = boundary_status(left_categories, right_categories)
    reverse_status = boundary_status(right_categories, left_categories)
    if forward_status == "valid":
        valid.append([str(first["segment_id"]), str(second["segment_id"])])
    if reverse_status == "valid":
        valid.append([str(second["segment_id"]), str(first["segment_id"])])
    if len(valid) == 1:
        status, ordered, reason = "ordered", valid[0], "Exactly one direction is established by an explicit allowed sequence rule."
    elif len(valid) > 1:
        status, ordered, reason = "ambiguous", None, "More than one direction is allowed; no arbitrary order was chosen."
    elif forward_status == "invalid" and reverse_status == "invalid":
        status, ordered, reason = "invalid", None, "Both possible cross-journey orders are explicitly prohibited by the sequence rules."
    else:
        status, ordered, reason = "unresolved", None, "No explicit allowed rule establishes either cross-journey order; further review is required."
    return {"segment_ids": [first["segment_id"], second["segment_id"]], "status": status, "ordered_segment_ids": ordered, "valid_orderings": valid, "reason": reason}
