"""Load and standardize patient-history text for similarity analysis.

The original database fields remain available; normalized text is added in a
separate column so preprocessing never overwrites the source symptom description.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .db import connect


def load_patient_histories(db_path: str | Path) -> pd.DataFrame:
    """Load all histories from SQLite into a Pandas DataFrame.

    Every original patient column is retained so later analysis results can
    include the original ID, demographics, diagnosis context, and outcome.
    """
    with connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM patients ORDER BY patient_id", connection
        )


def standardize_symptom_text(value: object) -> str:
    """Lowercase symptom text, normalize punctuation/spacing, and handle nulls.

    Null, NaN, and whitespace-only inputs become an empty string. Punctuation
    becomes spaces so neighboring words do not accidentally join together.
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_patient_histories(histories: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized symptom column added.

    The `symptoms_text` source column is not modified. Empty or missing source
    symptom fields are represented by an empty `normalized_symptoms` value.
    """
    if "symptoms_text" not in histories.columns:
        raise ValueError("Patient histories must contain a symptoms_text column")
    result = histories.copy()
    result["normalized_symptoms"] = result["symptoms_text"].map(standardize_symptom_text)
    return result


def load_preprocessed_histories(db_path: str | Path) -> pd.DataFrame:
    """Convenience function to load database histories and normalize symptom text."""
    return preprocess_patient_histories(load_patient_histories(db_path))
