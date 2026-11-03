"""Home-page UI and database status for the TwinCure Streamlit dashboard."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..db import connect, list_patients, list_steps


def load_home_stats(db_path: str | Path) -> dict[str, Any]:
    """Read live database counts using TwinCure's existing SQLite helpers.

    A missing database is reported without opening SQLite, since sqlite3 would
    otherwise create a new empty file and make the unavailable state misleading.
    Query/schema errors are returned as a short user-readable status message.
    """
    path = Path(db_path).expanduser()
    if not path.is_file():
        return {
            "accessible": False,
            "patients": None,
            "journey_steps": None,
            "sequence_rules": None,
            "compatibility_rules": None,
            "error": f"Database file not found: {path}",
        }

    try:
        patients = list_patients(path)
        steps = list_steps(path)
        with connect(path) as connection:
            sequence_rules = connection.execute(
                "SELECT COUNT(*) AS total FROM sequence_rules"
            ).fetchone()["total"]
            compatibility_rules = connection.execute(
                "SELECT COUNT(*) AS total FROM compatibility_rules"
            ).fetchone()["total"]
        return {
            "accessible": True,
            "patients": len(patients),
            "journey_steps": len(steps),
            "sequence_rules": int(sequence_rules),
            "compatibility_rules": int(compatibility_rules),
            "error": None,
        }
    except (OSError, sqlite3.Error, KeyError, TypeError) as exc:
        return {
            "accessible": False,
            "patients": None,
            "journey_steps": None,
            "sequence_rules": None,
            "compatibility_rules": None,
            "error": f"The database could not be read ({type(exc).__name__}: {exc}).",
        }


def render_home(st: Any, db_path: str | Path) -> None:
    """Render the project overview, safety notice, database health, and counts."""
    st.title("TwinCure")
    st.subheader("Optimizing Treatments from Similar Patient Journeys")
    st.write(
        "TwinCure is a modular educational project that compares synthetic patient "
        "journey descriptions, examines their recorded step sequences and outcomes, "
        "and builds traceable example hybrid journeys. Its similarity and evidence "
        "scores summarize this synthetic dataset only."
    )
    st.warning("Synthetic educational prototype — not for clinical use.")

    stats = load_home_stats(db_path)
    st.subheader("System status")
    if stats["accessible"]:
        st.success("Database status: accessible")
    else:
        st.error(f"Database status: unavailable. {stats['error']}")
        st.info(
            "Check that the project database exists and is readable. If it has not "
            "been initialized, follow the README section ‘Initialize or rebuild the database’."
        )

    st.subheader("Synthetic dataset overview")
    columns = st.columns(4)
    metrics = (
        ("Synthetic patients", "patients"),
        ("Journey steps", "journey_steps"),
        ("Sequence rules", "sequence_rules"),
        ("Compatibility rules", "compatibility_rules"),
    )
    for column, (label, key) in zip(columns, metrics):
        value = stats[key] if stats["accessible"] else "—"
        column.metric(label, value)
    st.caption("All records and rules shown here are synthetic educational data.")
