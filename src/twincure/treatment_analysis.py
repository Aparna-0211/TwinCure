"""Analyze ordered synthetic journey steps and label contiguous segments.

This module only describes synthetic historical segments. It does not merge
journeys, generate recommendations, or imply real clinical effectiveness.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
try:  # tomllib is built in from Python 3.11; tomli supports Python 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

import pandas as pd

from .db import connect

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"
SEGMENT_COLUMNS = [
    "segment_id",
    "source_patient_id",
    "source_step_ids",
    "step_orders",
    "treatment_descriptions",
    "step_outcomes",
    "mean_outcome",
    "classification",
    "source_similarity_score",
    "start_step_order",
    "end_step_order",
    "outcomes_complete",
]


def load_max_segment_length(settings_path: str | Path | None = None) -> int:
    """Read and validate the maximum contiguous segment length from TOML."""
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    with path.open("rb") as settings_file:
        settings = tomllib.load(settings_file)
    max_length = int(settings["max_segment_length"])
    if max_length < 1:
        raise ValueError("max_segment_length in settings must be at least 1")
    return max_length


def analyze_treatment_paths(
    similarity_result: object,
    db_path: str | Path,
    settings_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load matched journeys' steps from SQLite and return their segments.

    The Stage 2 result supplies the matching patient IDs and similarity scores.
    Only those patients' steps are analyzed. A patient ID is never combined with
    another journey, and each output segment retains its source and step details.
    """
    matches = getattr(similarity_result, "matches", None)
    if not isinstance(matches, pd.DataFrame):
        raise ValueError("similarity_result must contain a Pandas matches DataFrame")
    if matches.empty or "patient_id" not in matches.columns:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    max_length = load_max_segment_length(settings_path)
    selected = matches.dropna(subset=["patient_id"]).drop_duplicates(subset=["patient_id"])
    patient_ids = selected["patient_id"].astype(str).tolist()
    if not patient_ids:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    score_by_patient: dict[str, object] = {}
    if "similarity_score" in selected.columns:
        score_by_patient = dict(zip(selected["patient_id"].astype(str), selected["similarity_score"]))

    placeholders = ",".join("?" for _ in patient_ids)
    query = (
        "SELECT patient_id, step_id, step_order, step_category, step_description, "
        "step_outcome, outcome_notes FROM journey_steps "
        f"WHERE patient_id IN ({placeholders}) ORDER BY patient_id, step_order"
    )
    with connect(db_path) as connection:
        steps = pd.read_sql_query(query, connection, params=patient_ids)
    return generate_treatment_segments(steps, score_by_patient, max_length)


def generate_treatment_segments(
    steps: pd.DataFrame,
    similarity_by_patient: Mapping[str, object] | None,
    max_segment_length: int,
) -> pd.DataFrame:
    """Generate all contiguous windows up to the maximum length, per journey.

    Each individual step is a segment. Longer segments are consecutive windows
    from the same patient's ordered steps, never cross patient boundaries, and
    never cross a gap in step order. Missing outcomes are retained as missing;
    such segments receive no calculated mean and are labelled Mixed/Unclear.
    """
    if max_segment_length < 1:
        raise ValueError("max_segment_length must be at least 1")
    if steps.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    required = {"patient_id", "step_id", "step_order"}
    missing_columns = required.difference(steps.columns)
    if missing_columns:
        raise ValueError(f"Step data is missing required columns: {', '.join(sorted(missing_columns))}")
    if similarity_by_patient is None:
        similarity_by_patient = {}

    # Rows without journey identity or sequence position cannot safely form a segment.
    usable = steps.dropna(subset=["patient_id", "step_order"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    usable["step_order"] = pd.to_numeric(usable["step_order"], errors="coerce")
    usable = usable.dropna(subset=["step_order"])
    usable = usable.sort_values(["patient_id", "step_order"], kind="mergesort")

    segments: list[dict[str, object]] = []
    for patient_id, patient_steps in usable.groupby("patient_id", sort=True):
        rows = patient_steps.to_dict("records")
        # Split at sequence gaps so a window cannot pretend missing steps are contiguous.
        runs: list[list[dict[str, object]]] = []
        for row in rows:
            if not runs or float(row["step_order"]) != float(runs[-1][-1]["step_order"]) + 1:
                runs.append([row])
            else:
                runs[-1].append(row)

        for run in runs:
            for start in range(len(run)):
                max_end = min(len(run), start + max_segment_length)
                for end in range(start + 1, max_end + 1):
                    window = run[start:end]
                    segments.append(_build_segment(
                        str(patient_id), window, similarity_by_patient.get(str(patient_id))
                    ))
    return pd.DataFrame(segments, columns=SEGMENT_COLUMNS)


def _build_segment(
    patient_id: str,
    rows: list[dict[str, object]],
    similarity_score: object,
) -> dict[str, object]:
    """Calculate one segment's outcome summary without filling missing values."""
    step_ids = [row.get("step_id") for row in rows]
    orders = [int(row["step_order"]) for row in rows]
    descriptions = [_optional_value(row.get("step_description")) for row in rows]
    outcomes = [_optional_outcome(row.get("step_outcome")) for row in rows]
    complete = all(value is not None for value in outcomes)
    mean_outcome: float | None = sum(outcomes) / len(outcomes) if complete else None

    if not complete:
        classification = "Mixed/Unclear"
    elif mean_outcome >= 0.5 and 1 in outcomes:
        classification = "Positive"
    elif mean_outcome <= -0.5 and -1 in outcomes:
        classification = "Negative"
    else:
        classification = "Mixed/Unclear"

    score = _optional_value(similarity_score)
    start, end = orders[0], orders[-1]
    return {
        "segment_id": f"SEG-{patient_id}-{start:04d}-{end:04d}",
        "source_patient_id": patient_id,
        "source_step_ids": step_ids,
        "step_orders": orders,
        "treatment_descriptions": descriptions,
        "step_outcomes": outcomes,
        "mean_outcome": mean_outcome,
        "classification": classification,
        "source_similarity_score": score,
        "start_step_order": start,
        "end_step_order": end,
        "outcomes_complete": complete,
    }


def _optional_outcome(value: object) -> float | None:
    """Convert a known synthetic outcome to a number; retain missing as None."""
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric not in (-1.0, 0.0, 1.0):
        return None
    return numeric


def _optional_value(value: object) -> object:
    """Represent null-like values as None without inventing replacement text."""
    if value is None or pd.isna(value):
        return None
    return value
