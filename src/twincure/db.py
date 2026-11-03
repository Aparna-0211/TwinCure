"""SQLite setup and data access for synthetic educational TwinCure records.

This module intentionally accepts only rows labelled SYNTHETIC_EDUCATIONAL_ONLY.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable

SYNTHETIC_LABEL = "SYNTHETIC_EDUCATIONAL_ONLY"


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database with foreign-key checks enabled."""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(
    db_path: str | Path,
    schema_path: str | Path,
    patients_csv: str | Path,
    steps_csv: str | Path,
    rules_json: str | Path,
) -> None:
    """Create tables and import the bundled synthetic example data.

    Existing rows with matching primary keys are left unchanged, making this safe
    to rerun after initial setup. It does not accept unlabeled or real records.
    """
    connection = connect(db_path)
    try:
        connection.executescript(Path(schema_path).read_text(encoding="utf-8"))
        patients = _read_csv(patients_csv)
        steps = _read_csv(steps_csv)
        rules = json.loads(Path(rules_json).read_text(encoding="utf-8"))
        if rules.get("dataset_label") != SYNTHETIC_LABEL:
            raise ValueError("Rules file must identify itself as synthetic educational data")

        for row in patients:
            _require_synthetic(row)
            connection.execute(
                """INSERT OR IGNORE INTO patients
                   (patient_id, age_group, disease, symptoms_text, journey_outcome, dataset_label)
                   VALUES (:patient_id, :age_group, :disease, :symptoms_text, :journey_outcome, :dataset_label)""",
                _typed_patient(row),
            )
        for row in steps:
            patient_id = row["patient_id"]
            parent = connection.execute(
                "SELECT dataset_label FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            if parent is None or parent["dataset_label"] != SYNTHETIC_LABEL:
                raise ValueError(f"Step {row['step_id']} has no synthetic parent journey")
            connection.execute(
                """INSERT OR IGNORE INTO journey_steps
                   (step_id, patient_id, step_order, step_category, step_description, step_outcome, outcome_notes)
                   VALUES (:step_id, :patient_id, :step_order, :step_category, :step_description, :step_outcome, :outcome_notes)""",
                _typed_step(row),
            )
        for rule in rules.get("sequence_rules", []):
            connection.execute(
                "INSERT OR IGNORE INTO sequence_rules (from_category, to_category, rule_type, rule_note) VALUES (?, ?, ?, ?)",
                (rule["from_category"], rule["to_category"], rule["rule_type"], rule["rule_note"]),
            )
        for rule in rules.get("compatibility_rules", []):
            connection.execute(
                "INSERT OR IGNORE INTO compatibility_rules (category_a, category_b, status, rule_note) VALUES (?, ?, ?, ?)",
                (rule["category_a"], rule["category_b"], rule["status"], rule["rule_note"]),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def list_patients(db_path: str | Path) -> list[sqlite3.Row]:
    """Return synthetic journeys in stable ID order."""
    with connect(db_path) as connection:
        return list(connection.execute("SELECT * FROM patients ORDER BY patient_id"))


def list_steps(db_path: str | Path, patient_id: str | None = None) -> list[sqlite3.Row]:
    """Return ordered journey steps, optionally for one source patient."""
    with connect(db_path) as connection:
        if patient_id is None:
            query = "SELECT * FROM journey_steps ORDER BY patient_id, step_order"
            return list(connection.execute(query))
        query = "SELECT * FROM journey_steps WHERE patient_id = ? ORDER BY step_order"
        return list(connection.execute(query, (patient_id,)))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _require_synthetic(row: dict[str, str]) -> None:
    if row.get("dataset_label") != SYNTHETIC_LABEL:
        raise ValueError("Patient rows must be labelled SYNTHETIC_EDUCATIONAL_ONLY")


def _typed_patient(row: dict[str, str]) -> dict[str, object]:
    _require_synthetic(row)
    outcome = row.get("journey_outcome", "")
    return {**row, "journey_outcome": int(outcome) if outcome else None}


def _typed_step(row: dict[str, str]) -> dict[str, object]:
    return {**row, "step_order": int(row["step_order"]), "step_outcome": int(row["step_outcome"])}
