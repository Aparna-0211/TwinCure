"""Compatibility Analysis page powered by the existing Stages 2–4A."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import DatabaseError

from ..preprocessing import standardize_symptom_text
from ..segment_selection import select_candidate_segments
from ..similarity import find_similar_patients
from ..treatment_analysis import analyze_treatment_paths

SYNTHETIC_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "synthetic_rules.json"


def render_compatibility_analysis(st: Any, db_path: str | Path) -> None:
    """Render rule-validated combinations from matched synthetic journeys."""
    st.title("Compatibility Analysis")
    st.write(
        "Enter a synthetic symptom description. TwinCure will run the existing "
        "Stage 2 similarity search, analyze matched journeys with Stage 3, and "
        "submit those segments to Stage 4A's explicit compatibility and sequence checks."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")
    st.info(
        "Compatibility here means only that the provided synthetic prototype rules "
        "allow a pair. It does not establish clinical safety or medical approval. "
        "This page does not provide medical recommendations."
    )
    st.markdown(
        "**How the prototype rules are applied**  \n"
        "- Compatibility rules decide whether source step categories may be paired. "
        "An unknown rule is not treated as compatible.  \n"
        "- Sequence rules decide the permitted order between source segments. The "
        "page accepts an order only when Stage 4A finds exactly one explicitly allowed direction.  \n"
        "- Missing or ambiguous order is unresolved, not guessed.  \n"
        "- Only Stage 4A's explicitly validated combinations can proceed to the later Hybrid Journey stage."
    )

    with st.form("compatibility_analysis_form"):
        query = st.text_area(
            "Synthetic symptom description",
            key="compatibility_analysis_query",
            help="Use synthetic educational text only. Do not enter real patient information.",
            placeholder="For example: chest discomfort and fatigue",
        )
        submitted = st.form_submit_button("Analyze compatibility")

    st.caption("Every journey, segment, outcome, and rule displayed here is synthetic educational data.")
    if not submitted:
        return
    if not isinstance(query, str):
        st.error("Invalid input. Enter a text description using words or letters.")
        return
    if not query.strip():
        st.warning("Enter a synthetic symptom description before analysis.")
        return
    if not standardize_symptom_text(query):
        st.error("Invalid input. The description contains no usable words or letters; please revise it.")
        return

    database = Path(db_path).expanduser()
    if not database.is_file():
        st.error(
            "Database unavailable. The configured TwinCure SQLite database was not found. "
            "Initialize it using the project README instructions."
        )
        return
    if not SYNTHETIC_RULES_PATH.is_file():
        st.error(
            "Synthetic rule file unavailable. The explicit compatibility and sequence "
            "rules could not be found; no validation was performed."
        )
        return

    try:
        similarity_result = find_similar_patients(database, query)
        if similarity_result.status != "ok":
            st.info(
                "No similar synthetic patient journeys were found. "
                f"{similarity_result.message}"
            )
            st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
            return
        segments = analyze_treatment_paths(similarity_result, database)
        if segments.empty:
            st.info(
                "No treatment segments were generated for the matching synthetic journeys; "
                "compatibility analysis cannot continue."
            )
            return
        selection = select_candidate_segments(segments, database, SYNTHETIC_RULES_PATH)
    except (sqlite3.Error, DatabaseError) as exc:
        st.error(
            "Database unavailable. TwinCure could not read the configured synthetic "
            f"journey database ({type(exc).__name__}). Check its path and contents."
        )
        return
    except (OSError, json.JSONDecodeError) as exc:
        st.error(
            "Database, configuration, or synthetic rule-loading error. No compatibility "
            f"decision was made ({type(exc).__name__}). Check the project files."
        )
        return
    except Exception as exc:
        st.error(
            "Unexpected compatibility-processing error. No combination was accepted "
            f"({type(exc).__name__}). Check the query, database, and rules."
        )
        return

    st.success(
        f"Compared {len(selection['positive_candidates'])} positive candidate segment(s); "
        f"{len(selection['eligible_combinations'])} combination(s) passed Stage 4A validation."
    )
    st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
    _render_selection(st, selection)


def _render_selection(st: Any, selection: dict[str, Any]) -> None:
    """Render Stage 4A result fields without recomputing compatibility or order."""
    eligible_segments = selection.get("eligible_segments", pd.DataFrame())
    st.subheader(f"A. Eligible Positive candidate segments ({len(eligible_segments)})")
    if eligible_segments.empty:
        positive = selection.get("positive_candidates", pd.DataFrame())
        if positive.empty:
            st.info("No Positive treatment segments were selected from the matching journeys.")
        else:
            st.info(
                "Positive segments were found, but none belongs to a pair that passed "
                "both explicit compatibility and unique sequence validation. See the decisions below."
            )
    else:
        st.dataframe(
            pd.DataFrame([_segment_summary(row) for row in eligible_segments.to_dict("records")]),
            width="stretch",
            hide_index=True,
        )

    combinations = selection.get("eligible_combinations", [])
    st.subheader(f"B. Eligible combinations ({len(combinations)})")
    if not combinations:
        st.info("No segment pair passed both compatibility and sequence validation.")
    else:
        st.caption(
            "Each row preserves both source journeys. The validated order and acceptance reasons "
            "are copied from Stage 4A; the dashboard does not choose or recalculate an order."
        )
        st.dataframe(
            pd.DataFrame([_eligible_pair_summary(item) for item in combinations]),
            width="stretch",
            hide_index=True,
        )

    _render_noneligible(st, selection)


def _render_noneligible(st: Any, selection: dict[str, Any]) -> None:
    """Show Stage 4A rejections and unresolved pairs with their original reasons."""
    st.subheader("C. Rejected or unresolved decisions")
    rejected_segments = selection.get("rejected_segments", [])
    segment_map: dict[str, dict[str, Any]] = {}
    candidates = selection.get("positive_candidates", pd.DataFrame())
    if isinstance(candidates, pd.DataFrame):
        for row in candidates.to_dict("records"):
            segment_map[str(row.get("segment_id"))] = row
    for item in rejected_segments:
        segment = item.get("segment", {})
        if isinstance(segment, dict) and segment.get("segment_id") is not None:
            segment_map[str(segment["segment_id"])] = segment

    eligible_pair_ids = {item.get("pair_id") for item in selection.get("eligible_combinations", [])}
    sequence_by_pair = {item.get("pair_id"): item for item in selection.get("sequence_results", [])}
    compatibility_by_pair = {item.get("pair_id"): item for item in selection.get("compatibility_results", [])}
    pair_ids = list(dict.fromkeys([*compatibility_by_pair.keys(), *sequence_by_pair.keys()]))
    rejected_pairs: list[dict[str, Any]] = []
    unresolved_pairs: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        if pair_id in eligible_pair_ids:
            continue
        compatibility = compatibility_by_pair.get(pair_id)
        sequence = sequence_by_pair.get(pair_id)
        hard_rejection = (
            (compatibility is not None and compatibility.get("status") == "incompatible")
            or (sequence is not None and sequence.get("status") == "invalid")
        )
        row = _pair_summary(pair_id, compatibility, sequence, segment_map)
        (rejected_pairs if hard_rejection else unresolved_pairs).append(row)

    if rejected_segments:
        st.markdown(f"**Rejected or unpaired source segments ({len(rejected_segments)})**")
        segment_rows = []
        for item in rejected_segments:
            segment = item.get("segment", {})
            classification = segment.get("classification")
            disposition = (
                "Rejected segment (not Positive)"
                if classification != "Positive"
                else "Positive segment not eligible for a validated pair"
            )
            segment_rows.append({"decision": disposition, **_segment_summary(segment), "reason": item.get("reason")})
        st.dataframe(pd.DataFrame(segment_rows), width="stretch", hide_index=True)

    if rejected_pairs:
        st.markdown(f"**Rejected combinations ({len(rejected_pairs)})**")
        st.dataframe(pd.DataFrame(rejected_pairs), width="stretch", hide_index=True)
    if unresolved_pairs:
        st.markdown(f"**Unresolved combinations ({len(unresolved_pairs)})**")
        st.dataframe(pd.DataFrame(unresolved_pairs), width="stretch", hide_index=True)

    not_compared = [
        item for item in selection.get("selection_decisions", [])
        if item.get("decision") == "not_compared_same_journey"
    ]
    if not_compared:
        st.markdown(f"**Not compared (same source journey; {len(not_compared)})**")
        st.dataframe(
            pd.DataFrame([
                {"segment_ids": " · ".join(map(str, item.get("segment_ids", []))), "reason": item.get("reason")}
                for item in not_compared
            ]),
            width="stretch",
            hide_index=True,
        )

    if not rejected_segments and not rejected_pairs and not unresolved_pairs and not not_compared:
        st.info("There are no rejected or unresolved segment decisions for these results.")


def _segment_summary(segment: dict[str, Any]) -> dict[str, Any]:
    step_ids = _as_list(segment.get("source_step_ids"))
    orders = _as_list(segment.get("step_orders"))
    descriptions = _as_list(segment.get("treatment_descriptions"))
    steps = [
        f"order {order} · {step_id} · {_display(description)}"
        for step_id, order, description in zip(step_ids, orders, descriptions)
    ]
    return {
        "source_patient_id": segment.get("source_patient_id"),
        "segment_id": segment.get("segment_id"),
        "classification": segment.get("classification"),
        "source_step_ids": " · ".join(map(str, step_ids)),
        "original_step_order_and_descriptions": " | ".join(steps),
    }


def _eligible_pair_summary(combination: dict[str, Any]) -> dict[str, Any]:
    segment_map = {
        str(segment.get("segment_id")): segment
        for segment in combination.get("segments", [])
    }
    ordered_ids = [str(value) for value in combination.get("ordered_segment_ids", [])]
    ordered_sources = [
        f"{segment_map.get(segment_id, {}).get('source_patient_id', 'unknown')} / {segment_id}"
        for segment_id in ordered_ids
    ]
    return {
        "pair_id": combination.get("pair_id"),
        "source_journeys": " | ".join(ordered_sources),
        "validated_order_from_stage4a": " → ".join(ordered_sources),
        "compatibility_reason": combination.get("compatibility_reason"),
        "sequence_reason": combination.get("sequence_reason"),
    }


def _pair_summary(
    pair_id: Any,
    compatibility: dict[str, Any] | None,
    sequence: dict[str, Any] | None,
    segment_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = compatibility or sequence or {}
    segment_ids = source.get("segment_ids", [])
    sources = [
        f"{segment_map.get(str(segment_id), {}).get('source_patient_id', 'unknown')} / {segment_id}"
        for segment_id in segment_ids
    ]
    compatibility_status = compatibility.get("status") if compatibility else "not evaluated"
    sequence_status = sequence.get("status") if sequence else "not evaluated"
    if compatibility_status == "incompatible":
        compatibility_label = "explicitly incompatible"
    elif compatibility_status == "unknown":
        compatibility_label = "unknown (not treated as compatible)"
    else:
        compatibility_label = str(compatibility_status)
    if sequence_status == "invalid":
        sequence_label = "prohibited sequence"
    elif sequence_status == "unresolved":
        sequence_label = "no applicable ordering established"
    elif sequence_status == "ambiguous":
        sequence_label = "multiple possible orderings"
    else:
        sequence_label = str(sequence_status)
    valid_orderings = sequence.get("valid_orderings", []) if sequence else []
    return {
        "pair_id": pair_id,
        "source_journeys": " | ".join(sources),
        "compatibility_status": compatibility_label,
        "compatibility_reason": compatibility.get("reason") if compatibility else "No compatibility result was returned.",
        "sequence_status": sequence_label,
        "sequence_reason": sequence.get("reason") if sequence else "No sequence result was returned.",
        "valid_orderings_if_ambiguous": " | ".join(" → ".join(map(str, order)) for order in valid_orderings),
    }


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _display(value: Any) -> str:
    if value is None:
        return "Not recorded"
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return "Not recorded"
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text.strip() else "Not recorded"
