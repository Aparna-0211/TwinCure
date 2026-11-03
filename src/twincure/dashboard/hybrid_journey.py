"""Hybrid Journey page powered by the existing Stages 2–4B."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import DatabaseError

from ..hybrid_generation import generate_hybrid_journeys
from ..preprocessing import standardize_symptom_text
from ..segment_selection import select_candidate_segments
from ..similarity import find_similar_patients
from ..treatment_analysis import analyze_treatment_paths

SYNTHETIC_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "synthetic_rules.json"


def render_hybrid_journey(st: Any, db_path: str | Path) -> None:
    """Render separate Stage 4B hybrids from Stage 4A-approved combinations."""
    st.title("Hybrid Journey")
    st.write(
        "Enter a synthetic symptom description to view example hybrid journeys. "
        "The page reuses the existing Stage 2 similarity, Stage 3 segment analysis, "
        "Stage 4A compatibility/sequence validation, and Stage 4B hybrid generation."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")
    st.info(
        "Hybrid journeys are assembled from synthetic examples through prototype rules and "
        "artificial outcomes. They are not clinically validated and are not claims of safety, "
        "effectiveness, or suitability for any real patient. No diagnoses, prescriptions, "
        "or clinical recommendations are provided."
    )
    st.markdown(
        "**How to read the result**  \n"
        "- Stage 4B creates a separate candidate only from a combination Stage 4A marked eligible.  \n"
        "- Original order is retained inside each source segment; the cross-journey order is the "
        "unique order explicitly validated by Stage 4A.  \n"
        "- Every displayed step identifies its source patient, segment, source step, and original order."
    )
    with st.form("hybrid_journey_form"):
        query = st.text_area(
            "Synthetic symptom description",
            key="hybrid_journey_query",
            help="Use synthetic educational text only. Do not enter real patient information.",
            placeholder="For example: chest discomfort and fatigue",
        )
        submitted = st.form_submit_button("Generate hybrid examples")
    st.caption("All journeys, segments, descriptions, and outcomes shown are synthetic educational examples.")
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
        st.error("Synthetic rule file unavailable. Stage 4A validation was not performed; no hybrid was generated.")
        return

    try:
        similarity_result = find_similar_patients(database, query)
        if similarity_result.status != "ok":
            st.info(f"No similar synthetic patient journeys were found. {similarity_result.message}")
            st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
            return
        segments = analyze_treatment_paths(similarity_result, database)
        if segments.empty:
            st.info("No treatment segments were generated for the matching synthetic journeys; no hybrid can be created.")
            return
        selection = select_candidate_segments(segments, database, SYNTHETIC_RULES_PATH)
        generation = generate_hybrid_journeys(selection)
    except (sqlite3.Error, DatabaseError) as exc:
        st.error(f"Database unavailable. TwinCure could not read the synthetic journey database ({type(exc).__name__}). Check its path and contents.")
        return
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Database, configuration, or synthetic rule-loading error. No hybrid was generated ({type(exc).__name__}). Check the project files.")
        return
    except Exception as exc:
        st.error(f"Unexpected hybrid-processing error. No hybrid was generated ({type(exc).__name__}). Check the query, database, and rules.")
        return

    st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
    st.success(
        f"Stage 4A approved {len(selection.get('eligible_combinations', []))} combination(s); "
        f"Stage 4B generated {len(generation.get('hybrid_journeys', []))} separate hybrid example(s)."
    )
    if generation.get("status") == "no_valid_hybrid":
        st.warning("Generation status: no_valid_hybrid")
        st.info(generation.get("message", "No valid hybrid was generated; no path was forced."))
    elif generation.get("status") != "generated":
        st.info(f"Generation status: {generation.get('status', 'unknown')}. {generation.get('message', '')}")

    _render_rejections(st, generation)
    hybrids = generation.get("hybrid_journeys", [])
    if not hybrids:
        if not selection.get("eligible_combinations"):
            st.info("No eligible combinations were approved by Stage 4A, so Stage 4B had no input and no hybrid was created.")
        elif generation.get("status") != "no_valid_hybrid":
            st.info("No valid hybrid journeys were generated.")
        return
    for hybrid in hybrids:
        _render_hybrid(st, hybrid)


def _render_hybrid(st: Any, hybrid: dict[str, Any]) -> None:
    """Render one Stage 4B hybrid with overview, ordered steps, and provenance."""
    steps = hybrid.get("ordered_steps", [])
    segments = hybrid.get("included_segments", [])
    patients = list(dict.fromkeys(str(item.get("source_patient_id")) for item in segments))
    journey_id = hybrid.get("hybrid_journey_id", "unknown")
    st.subheader(f"Hybrid journey {journey_id}")
    st.markdown(
        f"**Hybrid journey ID:** `{journey_id}`  \n"
        f"**Contributing source patients:** {len(patients)}  \n"
        f"**Included treatment steps:** {len(steps)}  \n"
        f"**Generation status:** `{hybrid.get('status', 'unknown')}`"
    )
    if steps:
        frame = pd.DataFrame([{
            "hybrid_step_position": row.get("hybrid_position"),
            "source_patient_id": row.get("source_patient_id"),
            "source_segment_id": row.get("source_segment_id"),
            "source_step_id": row.get("source_step_id"),
            "original_step_order": row.get("original_step_order"),
            "description": _display(row.get("treatment_description")),
            "synthetic_outcome": _display(row.get("original_outcome")),
        } for row in steps])
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.info("This generated hybrid has no steps; inspect the generation record before use.")

    st.markdown("**Provenance and segment contribution**")
    for segment in segments:
        st.markdown(
            f"- Source patient **{segment.get('source_patient_id', 'unknown')}** contributed "
            f"segment **{segment.get('source_segment_id', 'unknown')}**. "
            f"{segment.get('reason', 'No segment explanation was recorded.')}"
        )
    st.markdown("**Stage 4A compatibility explanation**")
    st.write(hybrid.get("compatibility_rule_support") or "No compatibility explanation was recorded.")
    st.markdown("**Stage 4A sequence explanation**")
    st.write(hybrid.get("sequence_rule_support") or "No sequence explanation was recorded.")
    st.caption(
        "The segment order shown above is Stage 4A's validated order. Within each segment, "
        "source steps retain their original order; different approved combinations remain separate examples."
    )


def _render_rejections(st: Any, generation: dict[str, Any]) -> None:
    """Expose Stage 4B's invalid/duplicate rejections and Stage 4A exclusions."""
    rejected = generation.get("rejected_combinations", [])
    if rejected:
        st.markdown(f"**Stage 4B rejected or duplicate combinations ({len(rejected)})**")
        st.dataframe(pd.DataFrame([{
            "pair_id": item.get("pair_id"),
            "segment_ids": " · ".join(map(str, item.get("segment_ids", []))) if isinstance(item.get("segment_ids"), list) else _display(item.get("segment_ids")),
            "reason": item.get("reason"),
        } for item in rejected]), width="stretch", hide_index=True)
    excluded = generation.get("rejected_segments", [])
    if excluded:
        rows = []
        for item in excluded:
            segment = item.get("segment", {})
            rows.append({
                "source_patient_id": segment.get("source_patient_id"),
                "segment_id": segment.get("segment_id"),
                "classification": segment.get("classification"),
                "reason": item.get("reason"),
            })
        st.markdown(f"**Stage 4A segments excluded from hybrid generation ({len(rows)})**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _display(value: Any) -> str:
    """Render missing scalar source data transparently rather than filling it in."""
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
