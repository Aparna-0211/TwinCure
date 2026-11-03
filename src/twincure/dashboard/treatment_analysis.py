"""Treatment Analysis dashboard page backed by the existing Stage 2 and 3 code."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import DatabaseError

from ..preprocessing import standardize_symptom_text
from ..similarity import find_similar_patients
from ..treatment_analysis import analyze_treatment_paths

CATEGORY_LABELS = (
    ("Positive", "Positive"),
    ("Negative", "Negative"),
    ("Mixed/Unclear", "Mixed / Unclear"),
)


def render_treatment_analysis(st: Any, db_path: str | Path) -> None:
    """Render a symptom query and Stage 3 segments for Stage 2 matches."""
    st.title("Treatment Analysis")
    st.write(
        "Enter a synthetic symptom description. TwinCure will use the existing "
        "Stage 2 similarity workflow, then analyze the matched journeys with the "
        "existing Stage 3 contiguous-segment function."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")
    st.info(
        "All displayed journeys and outcomes are artificial educational examples, "
        "not real clinical evidence. Segment labels do not indicate that a treatment "
        "is medically effective. This page does not provide diagnoses, prescriptions, "
        "or clinical recommendations."
    )

    with st.form("treatment_analysis_form"):
        query = st.text_area(
            "Synthetic symptom description",
            key="treatment_analysis_query",
            help="Use synthetic educational text only. Do not enter real patient information.",
            placeholder="For example: chest discomfort and fatigue",
        )
        submitted = st.form_submit_button("Analyze matching synthetic journeys")

    st.markdown(
        "**Stage 3 classification criteria**  \n"
        "- **Positive:** mean outcome ≥ +0.5, with at least one +1.  \n"
        "- **Negative:** mean outcome ≤ −0.5, with at least one −1.  \n"
        "- **Mixed / Unclear:** all other cases, including missing or invalid outcomes."
    )
    st.caption("These rules classify constructed example outcomes only; they are not clinical evidence.")
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
    except (sqlite3.Error, DatabaseError, OSError) as exc:
        st.error(
            "Database unavailable. TwinCure could not read the synthetic journey "
            f"database ({type(exc).__name__}). Check its path and contents."
        )
        return
    except Exception as exc:
        st.error(
            "Unexpected processing error. Treatment analysis could not be completed "
            f"({type(exc).__name__}). Check the query, database, and project configuration."
        )
        return

    if not isinstance(segments, pd.DataFrame) or segments.empty:
        st.info(
            "No treatment segments were generated for the matching synthetic journeys. "
            "The matched journeys may not contain usable ordered step records."
        )
        return

    st.success(
        f"Analyzed {len(similarity_result.matches)} matching synthetic journey(s) "
        f"and generated {len(segments)} segment(s)."
    )
    st.caption(f"Configured similarity threshold: {similarity_result.threshold:.2f}")
    st.caption("Source IDs, original step order, descriptions, outcomes, means, and similarity scores are retained below.")

    for classification, heading in CATEGORY_LABELS:
        category = segments[segments["classification"] == classification]
        st.subheader(f"{heading} ({len(category)})")
        if category.empty:
            st.info(f"No {heading.lower()} segments were generated for this query.")
            continue
        for _, segment in category.iterrows():
            _render_segment(st, segment)


def _render_segment(st: Any, segment: pd.Series) -> None:
    """Display one source segment without dropping its original provenance."""
    with st.container(border=True):
        st.markdown(f"**Source patient ID:** `{segment.get('source_patient_id')}`")
        st.markdown(f"**Segment ID:** `{segment.get('segment_id')}`")
        st.markdown(f"**Classification:** {segment.get('classification')}")

        metrics = st.columns(2)
        metrics[0].metric("Mean synthetic outcome", _format_mean(segment.get("mean_outcome")))
        metrics[1].metric("Stage 2 similarity score", _format_similarity(segment.get("source_similarity_score")))

        step_ids = _as_list(segment.get("source_step_ids"))
        step_orders = _as_list(segment.get("step_orders"))
        descriptions = _as_list(segment.get("treatment_descriptions"))
        outcomes = _as_list(segment.get("step_outcomes"))
        st.markdown("**Original source steps**")
        if not step_ids:
            st.write("No source step IDs are available.")
            return
        for step_id, order, description, outcome in zip(step_ids, step_orders, descriptions, outcomes):
            st.markdown(f"**Original step order {order} · source step ID `{step_id}`**")
            st.write(f"Original step description: {_format_description(description)}")
            st.write(f"Original outcome: {_format_outcome(outcome)}")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except (TypeError, ValueError):
        return False


def _format_description(value: Any) -> str:
    return "Not recorded" if _is_missing(value) or not str(value).strip() else str(value)


def _format_outcome(value: Any) -> str:
    if _is_missing(value):
        return "Missing / invalid (not recorded)"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Missing / invalid (not recorded)"
    if number not in (-1.0, 0.0, 1.0):
        return "Missing / invalid (not recorded)"
    return f"{number:+g} (synthetic outcome)"


def _format_mean(value: Any) -> str:
    if _is_missing(value):
        return "Unavailable (incomplete or invalid outcomes)"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "Unavailable (incomplete or invalid outcomes)"


def _format_similarity(value: Any) -> str:
    if _is_missing(value):
        return "Not available"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "Not available"
