"""Patient Similarity page powered by the existing Stage 2 workflow."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import DatabaseError

from ..preprocessing import standardize_symptom_text
from ..similarity import find_similar_patients


def render_patient_similarity(st: Any, db_path: str | Path) -> None:
    """Render symptom input and Stage 2 TF-IDF/cosine matching results."""
    st.title("Patient Similarity")
    st.write(
        "Enter a synthetic symptom description to find textually similar journeys "
        "in TwinCure's educational dataset. The existing Stage 2 workflow applies "
        "the similarity threshold from `config/settings.toml`."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")
    st.info(
        "TF-IDF and cosine similarity measure text overlap only. Similarity does "
        "not establish medical equivalence, diagnosis, or treatment effectiveness. "
        "TwinCure does not provide diagnoses, prescriptions, or clinical recommendations."
    )

    with st.form("patient_similarity_form"):
        query = st.text_area(
            "Synthetic symptom description",
            key="patient_similarity_query",
            help="Use synthetic educational text only. Do not enter real patient information.",
            placeholder="For example: chest discomfort and fatigue",
        )
        submitted = st.form_submit_button("Find similar synthetic journeys")

    st.caption("All returned patient journeys and outcomes are synthetic educational records.")
    if not submitted:
        return

    if not isinstance(query, str):
        st.error("Invalid input. Enter a text description using words or letters.")
        return
    if not query.strip():
        st.warning("Enter a synthetic symptom description before searching.")
        return
    if not standardize_symptom_text(query):
        st.error("Invalid input. The description contains no usable words or letters; please revise it.")
        return

    database = Path(db_path).expanduser()
    if not database.is_file():
        st.error(
            "Database unavailable. The configured TwinCure SQLite database was not found. "
            "Initialize it using the instructions in the project README."
        )
        return

    try:
        result = find_similar_patients(database, query)
    except (sqlite3.Error, DatabaseError, OSError) as exc:
        st.error(
            "Database unavailable. TwinCure could not read the configured synthetic "
            f"journey database ({type(exc).__name__}). Check its path and contents."
        )
        return
    except (ValueError, TypeError) as exc:
        st.error(
            "The search could not be processed. Check that the description is valid "
            f"and the configured similarity settings can be read ({type(exc).__name__})."
        )
        return
    except Exception as exc:  # Keep the dashboard usable for unexpected I/O/library failures.
        st.error(
            "The similarity search could not be completed. Check the database and "
            f"project configuration ({type(exc).__name__})."
        )
        return

    if result.status == "empty_query":
        st.warning("Enter a synthetic symptom description before searching.")
        return
    if result.status == "ok":
        st.success(result.message)
        st.caption(f"Configured similarity threshold: {result.threshold:.2f}")
        st.subheader("Matching synthetic patient journeys")
        for _, patient in result.matches.iterrows():
            with st.container(border=True):
                st.markdown(f"**Synthetic patient ID:** `{patient['patient_id']}`")
                st.markdown("**Original symptom description**")
                symptoms = patient.get("symptoms_text")
                st.write("Not recorded" if pd.isna(symptoms) else str(symptoms))
                score = float(patient["similarity_score"])
                st.metric("TF-IDF cosine similarity", f"{score:.3f}")
                st.caption(
                    "This score is textual similarity calculated with TF-IDF and "
                    "cosine similarity; it does not establish medical equivalence "
                    "or treatment effectiveness."
                )
        return

    if result.status in {"no_candidates", "insufficient_similarity"}:
        st.info(f"No matching synthetic patient journeys were found. {result.message}")
        st.caption(f"Configured similarity threshold: {result.threshold:.2f}")
        return

    st.info(result.message)
    st.caption(f"Search status: {result.status}; configured threshold: {result.threshold:.2f}")
