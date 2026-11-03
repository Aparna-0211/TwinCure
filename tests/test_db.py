import csv
import json
import sqlite3

import pytest

from twincure.db import initialize_database, list_patients, list_steps

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def initialize(path):
    initialize_database(
        path,
        ROOT / "database/schema.sql",
        ROOT / "data/synthetic_patients.csv",
        ROOT / "data/synthetic_steps.csv",
        ROOT / "data/synthetic_rules.json",
    )


def test_import_preserves_synthetic_journeys_and_order(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    patients = list_patients(db)
    steps = list_steps(db, "SYN001")
    assert len(patients) == 8
    assert all(row["dataset_label"] == "SYNTHETIC_EDUCATIONAL_ONLY" for row in patients)
    assert [row["step_id"] for row in steps] == ["ST001A", "ST001B", "ST001C"]
    assert [row["step_order"] for row in steps] == [1, 2, 3]
    assert [row["step_outcome"] for row in steps] == [1, 1, 0]


def test_import_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    initialize(db)
    assert len(list_patients(db)) == 8
    assert len(list_steps(db)) == 24


def test_database_has_explicit_rules(tmp_path):
    db = tmp_path / "test.db"
    initialize(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequence_rules").fetchone()[0] == 2
        statuses = {row[0] for row in connection.execute("SELECT status FROM compatibility_rules")}
    assert {"compatible", "unknown"}.issubset(statuses)


def test_unlabelled_patient_data_is_rejected(tmp_path):
    db = tmp_path / "test.db"
    bad_csv = tmp_path / "patients.csv"
    source = ROOT / "data/synthetic_patients.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["dataset_label"] = "REAL_PATIENT_DATA"
    with bad_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="SYNTHETIC_EDUCATIONAL_ONLY"):
        initialize_database(
            db,
            ROOT / "database/schema.sql",
            bad_csv,
            ROOT / "data/synthetic_steps.csv",
            ROOT / "data/synthetic_rules.json",
        )
