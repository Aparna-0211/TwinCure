"""TwinCure Streamlit dashboard entry point (Stages 6A–6F)."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from twincure.dashboard.home import render_home
from twincure.dashboard.patient_similarity import render_patient_similarity
from twincure.dashboard.treatment_analysis import render_treatment_analysis
from twincure.dashboard.compatibility_analysis import render_compatibility_analysis
from twincure.dashboard.hybrid_journey import render_hybrid_journey
from twincure.dashboard.evidence_scoring import render_evidence_scoring

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "twincure.db"
NAVIGATION_ITEMS = (
    "Home",
    "Patient Similarity",
    "Treatment Analysis",
    "Compatibility Analysis",
    "Hybrid Journey",
    "Evidence Scoring",
)


def get_database_path() -> Path:
    """Resolve the project database, allowing an optional path override for testing."""
    configured_path = os.environ.get("TWINCURE_DB_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_DATABASE_PATH


def main() -> None:
    """Render implemented pages or a clear notice for later-stage pages."""
    st.set_page_config(page_title="TwinCure", page_icon=None, layout="wide")
    st.sidebar.title("TwinCure")
    section = st.sidebar.radio("Navigate", NAVIGATION_ITEMS, index=0)
    st.caption("Synthetic educational prototype — not for clinical use.")

    if section == "Home":
        render_home(st, get_database_path())
        return
    if section == "Patient Similarity":
        render_patient_similarity(st, get_database_path())
        return
    if section == "Treatment Analysis":
        render_treatment_analysis(st, get_database_path())
        return
    if section == "Compatibility Analysis":
        render_compatibility_analysis(st, get_database_path())
        return
    if section == "Hybrid Journey":
        render_hybrid_journey(st, get_database_path())
        return
    if section == "Evidence Scoring":
        render_evidence_scoring(st, get_database_path())
        return

    st.title(section)
    st.info("Coming in the next stage.")
    st.warning("Synthetic educational prototype — not for clinical use.")


if __name__ == "__main__":
    main()
