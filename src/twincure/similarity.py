"""Find synthetic patient journeys with similar symptom text.

TF-IDF turns words into weighted numeric features, and cosine similarity
compares the direction of those vectors. Results are educational similarity
matches only, not clinical recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
try:  # tomllib is built in from Python 3.11; tomli supports Python 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import load_preprocessed_histories, preprocess_patient_histories, standardize_symptom_text

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"


@dataclass
class SimilarityResult:
    """Ranked threshold-matching journeys and a plain-language result status."""

    matches: pd.DataFrame
    threshold: float
    status: str
    message: str


def load_similarity_threshold(settings_path: str | Path | None = None) -> float:
    """Read the similarity cutoff from TOML settings and validate its range."""
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    with path.open("rb") as settings_file:
        settings = tomllib.load(settings_file)
    threshold = float(settings["similarity_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold in settings must be between 0 and 1")
    return threshold


def rank_similar_patients(
    query_symptoms: object,
    histories: pd.DataFrame,
    settings_path: str | Path | None = None,
) -> SimilarityResult:
    """Rank existing histories against one synthetic case and apply config cutoff.

    The query is transformed together with the stored symptom documents, so
    words in the new case are represented consistently. Empty candidate symptom
    descriptions receive no match. Returned rows contain all original columns
    plus normalized text and a similarity score, sorted highest-first with IDs
    as a deterministic tie-breaker.
    """
    threshold = load_similarity_threshold(settings_path)
    if "patient_id" not in histories.columns:
        raise ValueError("Patient histories must contain a patient_id column")
    prepared = histories if "normalized_symptoms" in histories.columns else preprocess_patient_histories(histories)
    query_text = standardize_symptom_text(query_symptoms)
    empty_matches = prepared.iloc[0:0].copy()
    empty_matches["similarity_score"] = pd.Series(dtype="float64")

    if not query_text:
        return SimilarityResult(empty_matches, threshold, "empty_query", "The new synthetic case has no usable symptom text.")
    if prepared.empty:
        return SimilarityResult(empty_matches, threshold, "no_candidates", "There are no stored patient journeys to compare.")

    candidate_texts = prepared["normalized_symptoms"].fillna("").astype(str).tolist()
    valid_positions = [i for i, text in enumerate(candidate_texts) if text.strip()]
    if not valid_positions:
        return SimilarityResult(empty_matches, threshold, "insufficient_similarity", "Stored journeys have no usable symptom descriptions.")

    valid_texts = [candidate_texts[i] for i in valid_positions]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectors = vectorizer.fit_transform([query_text, *valid_texts])
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()

    scored = prepared.copy()
    scored["similarity_score"] = 0.0
    for position, score in zip(valid_positions, scores):
        scored.iloc[position, scored.columns.get_loc("similarity_score")] = float(score)

    matches = scored.loc[scored["similarity_score"] >= threshold].copy()
    matches = matches.sort_values(
        by=["similarity_score", "patient_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if matches.empty:
        return SimilarityResult(matches, threshold, "insufficient_similarity", f"No stored journey met the configured threshold of {threshold:.2f}.")
    return SimilarityResult(matches, threshold, "ok", f"Found {len(matches)} journey(s) meeting the configured threshold of {threshold:.2f}.")


def find_similar_patients(
    db_path: str | Path,
    query_symptoms: object,
    settings_path: str | Path | None = None,
) -> SimilarityResult:
    """Load synthetic journeys from SQLite and return configured similarity matches."""
    histories = load_preprocessed_histories(db_path)
    return rank_similar_patients(query_symptoms, histories, settings_path)
