"""Evidence Scoring page powered by the existing Stages 2–5."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import DatabaseError

from ..evidence_scoring import score_hybrid_evidence
from ..hybrid_generation import generate_hybrid_journeys
from ..preprocessing import standardize_symptom_text
from ..segment_selection import select_candidate_segments
from ..similarity import find_similar_patients
from ..treatment_analysis import analyze_treatment_paths

SYNTHETIC_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "synthetic_rules.json"
SCORE_FORMULA = "Score = 100 × (0.4118S + 0.3529O + 0.2353A)"
MEANING_LIMITATION = (
    "The evidence-support score measures how strongly the synthetic example data supports a generated hybrid journey "
    "under TwinCure's prototype scoring rules. It is not a probability of treatment success, clinical efficacy score, "
    "or medical recommendation."
)


def render_evidence_scoring(st: Any, db_path: str | Path) -> None:
    """Run existing Stages 2–5 and display each hybrid's evidence components."""
    st.title("Evidence Scoring")
    st.write(
        "Enter a synthetic symptom description to review Stage 5 evidence-support calculations "
        "for hybrid examples from the existing Stage 2→3→4A→4B pipeline."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")
    st.info(MEANING_LIMITATION)
    st.markdown(
        "**Formula used by Stage 5**  \n"
        f"`{SCORE_FORMULA}`  \n"
        "The dashboard displays the values returned by Stage 5; it does not recalculate the weights, "
        "round component inputs, or substitute zero for missing evidence."
    )
    with st.expander("What S, O, and A mean", expanded=True):
        st.markdown(
            "- **S — Similarity support:** mean cosine similarity across unique source journeys in this hybrid.  \n"
            "- **O — Outcome support:** for each contributing journey, unique source-step outcomes in its selected Positive "
            "segments are converted to support as `(mean outcome + 1) / 2`; journey supports are then averaged.  \n"
            "- **A — Agreement support:** positive journey votes divided by positive plus negative journey votes. "
            "Neutral/unclear votes are excluded from the denominator. Votes are calculated once per unique source journey "
            "and unique source step. Only journeys contributing to this particular hybrid are considered. Available "
            "rejected/non-positive candidate evidence from those journeys may affect their vote; unrelated journeys do not."
        )
    with st.form("evidence_scoring_form"):
        query = st.text_area(
            "Synthetic symptom description",
            key="evidence_scoring_query",
            help="Use synthetic educational text only. Do not enter real patient information.",
            placeholder="For example: irregular heartbeat and dizziness",
        )
        submitted = st.form_submit_button("Calculate evidence support")
    st.caption("All patient journeys, segments, outcomes, and scores shown are derived from synthetic educational examples.")
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
        st.error("Database unavailable. The configured SQLite database was not found. Initialize it using the README instructions.")
        return
    if not SYNTHETIC_RULES_PATH.is_file():
        st.error("Synthetic rule file unavailable. Stage 4A could not validate combinations; no scores were calculated.")
        return

    try:
        similarity_result = find_similar_patients(database, query)
        if similarity_result.status != "ok":
            st.info(f"No similar synthetic patient journeys were found. {similarity_result.message}")
            st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
            return
        segments = analyze_treatment_paths(similarity_result, database)
        if segments.empty:
            st.info("No treatment segments were generated for matching journeys; evidence scoring cannot continue.")
            return
        selection = select_candidate_segments(segments, database, SYNTHETIC_RULES_PATH)
        generation = generate_hybrid_journeys(selection)
        scoring = score_hybrid_evidence(generation)
    except (sqlite3.Error, DatabaseError) as exc:
        st.error(f"Database unavailable. TwinCure could not read the synthetic journey database ({type(exc).__name__}). Check its path and contents.")
        return
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Database, configuration, or synthetic rule-loading error. No score was calculated ({type(exc).__name__}). Check the project files.")
        return
    except Exception as exc:
        st.error(f"Unexpected evidence-scoring processing error. No score was calculated ({type(exc).__name__}). Check the query, database, and rules.")
        return

    st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
    if generation.get("status") == "no_valid_hybrid":
        st.warning("Stage 4B status: no_valid_hybrid")
        st.info(generation.get("message", "No valid hybrid was generated; no path was forced."))
    if not selection.get("eligible_combinations"):
        st.info("No eligible combinations were approved by Stage 4A; there is no hybrid to score.")
    if scoring.get("status") == "no_hybrid_journeys" or not scoring.get("results"):
        st.info(scoring.get("message", "No hybrid journeys were available for evidence scoring."))
        return

    st.write(scoring.get("message", "Stage 5 completed its evidence analysis."))
    for result in scoring["results"]:
        _render_result(st, result)


def _render_result(st: Any, result: dict[str, Any]) -> None:
    """Display Stage 5 values and calculation details for one hybrid only."""
    hybrid_id = result.get("hybrid_journey_id") or "Unknown hybrid"
    st.subheader(f"Evidence support — {hybrid_id}")
    status = result.get("status", "unknown")
    if status == "insufficient_evidence":
        st.warning("Status: insufficient_evidence. The final score is withheld because required evidence is incomplete.")
    else:
        st.caption(f"Status: {status}")

    components = result.get("components") or {}
    columns = st.columns(3)
    for column, key, label in zip(columns, ("S", "O", "A"), ("S — Similarity support", "O — Outcome support", "A — Agreement support")):
        value = components.get(key, {}).get("value", result.get(key))
        column.metric(label, _format_component(value))
    final_score = result.get("evidence_support_score")
    if status == "scored" and final_score is not None:
        st.metric("Final evidence-support score (0–100)", f"{final_score:.2f}")
        st.caption(SCORE_FORMULA)
    else:
        st.metric("Final evidence-support score (0–100)", "Withheld")
        st.caption("No score is generated when one or more required components is undefined.")

    st.markdown("**Similarity support (S)**")
    st.write("Stage 5's mean cosine similarity across unique source journeys contributing to this hybrid.")
    similarity_rows = components.get("S", {}).get("calculations", [])
    if similarity_rows:
        st.dataframe(pd.DataFrame(similarity_rows), width="stretch", hide_index=True)
    elif components.get("S", {}).get("value") is None:
        st.info(_component_reason("S", result))

    st.markdown("**Outcome support (O)**")
    st.write(
        "Stage 5 uses unique source steps in the selected Positive segments. It computes each contributing "
        "journey's mean outcome, maps it to support with `(mean + 1) / 2`, and averages journey supports."
    )
    outcome_rows = components.get("O", {}).get("calculations", [])
    if outcome_rows:
        outcome_frame = pd.DataFrame([{
            "source_patient_id": item.get("source_patient_id"),
            "unique_source_step_ids": _join(item.get("unique_source_step_ids")),
            "unique_step_outcomes": _join(item.get("unique_step_outcomes")),
            "mean_outcome": item.get("mean_outcome"),
            "individual_outcome_support": item.get("individual_outcome_support"),
        } for item in outcome_rows])
        st.dataframe(outcome_frame, width="stretch", hide_index=True)
    elif components.get("O", {}).get("value") is None:
        st.info(_component_reason("O", result))

    st.markdown("**Agreement support (A)**")
    st.write(
        "Stage 5 gives one positive, negative, or neutral/unclear vote per unique source journey. "
        "Each journey's overlapping source steps are deduplicated. Only patients contributing to this hybrid "
        "are in this calculation. Rejected/non-positive candidate evidence from those contributing journeys "
        "can change their vote; patients unrelated to this hybrid cannot. Neutral/unclear votes are not in the denominator."
    )
    agreement = components.get("A", {}).get("calculations", {})
    vote_rows = agreement.get("votes", []) if isinstance(agreement, dict) else []
    if vote_rows:
        vote_frame = pd.DataFrame([{
            "source_patient_id": item.get("source_patient_id"),
            "unique_source_step_ids": _join(item.get("unique_source_step_ids")),
            "mean_outcome": item.get("mean_outcome"),
            "journey_vote": item.get("vote"),
        } for item in vote_rows])
        st.dataframe(vote_frame, width="stretch", hide_index=True)
        st.caption(
            f"Contributing journey IDs: {', '.join(map(str, agreement.get('candidate_patient_ids', []))) or 'none'}. "
            f"Positive votes: {', '.join(map(str, agreement.get('positive_journey_ids', []))) or 'none'}; "
            f"negative votes: {', '.join(map(str, agreement.get('negative_journey_ids', []))) or 'none'}; "
            f"neutral/unclear: {', '.join(map(str, agreement.get('neutral_or_unclear_journey_ids', []))) or 'none'}."
        )
    elif components.get("A", {}).get("value") is None:
        st.info(_component_reason("A", result))

    missing = result.get("missing_evidence") or []
    if missing:
        st.markdown("**Why this score is insufficient**")
        for reason in missing:
            st.write(f"- {reason}")
    if result.get("explanation"):
        st.caption(result["explanation"])


def _component_reason(key: str, result: dict[str, Any]) -> str:
    """Find Stage 5's explicit reason for an undefined component."""
    for reason in result.get("missing_evidence", []) or []:
        if str(reason).startswith(f"{key} is undefined"):
            return str(reason)
    return f"{key} is undefined; Stage 5 did not provide a component value."


def _format_component(value: Any) -> str:
    """Format returned components for display without changing the score calculation."""
    if value is None:
        return "Undefined"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "Undefined"


def _join(values: Any) -> str:
    """Make lists in Stage 5 details readable in a table cell."""
    return " · ".join(map(str, values)) if isinstance(values, (list, tuple)) else str(values or "")
