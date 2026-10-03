"""Semantic scores for a Commission proposal, from local sentence embeddings (no LLM).

left_right             0 = fully right-wing, 1 = fully left-wing.
                       Similarity of title + summary to two sets of reference statements;
                       0.5 means equally close to both (most technical proposals).
disruptive_acceptable  0 = radical change, 1 = fully in line with what was already proposed.
                       Mean cosine similarity of the title to the 5 closest Commission proposals
                       filed in the 5 years strictly BEFORE this one: known on the filing date.

Both are heuristics for exploration and the demo, not ground truth.
Model: sentence-transformers/all-MiniLM-L6-v2 (English, runs on CPU, deterministic).
"""
import re
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS_FROM = "2016-01-01"
LOOKBACK_YEARS = 5
TOP_K = 5
TEMPERATURE = 0.05  # spread of the left/right sigmoid

LEFT = [
    "strengthening workers' rights, collective bargaining and adequate minimum wages",
    "expanding social protection, public services and welfare",
    "taxing large corporations and wealthy individuals to reduce inequality",
    "ambitious climate action and strict environmental protection",
    "stricter regulation of corporations and financial markets",
    "protecting the rights of migrants, refugees and asylum seekers",
    "gender equality, anti-discrimination and LGBTIQ rights",
    "stronger consumer protection and corporate accountability",
]
RIGHT = [
    "cutting red tape and reducing the administrative burden on businesses",
    "free markets, competitiveness and lower taxes",
    "strengthening external border control and returning irregular migrants",
    "more powers for police, law enforcement and security",
    "increasing defence spending and military capabilities",
    "protecting national sovereignty and subsidiarity",
    "fiscal discipline and reducing public debt",
    "supporting farmers, traditional industries and energy security",
]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


@lru_cache(maxsize=1)
def _anchors() -> tuple[np.ndarray, np.ndarray]:
    return embed(LEFT), embed(RIGHT)


ACT_TYPE = re.compile(
    r"^\s*(proposal for an?\s+)?(amended proposal for an?\s+)?(joint\s+)?(council\s+)?(implementing\s+|delegated\s+)?"
    r"(regulation|directive|decision)\s*(of the european parliament and of the council)?\s*",
    re.I,
)


def clean_title(title: str) -> str:
    """Drop the act-type boilerplate that every title shares, keep the subject."""
    return ACT_TYPE.sub("", title or "").strip() or (title or "")


def left_right(text: str) -> float:
    v = embed([text])[0]
    left, right = _anchors()
    top = lambda m: float(np.sort(m @ v)[-3:].mean())  # noqa: E731
    diff = top(left) - top(right)
    return round(1 / (1 + np.exp(-diff / TEMPERATURE)), 3)


def nearest_chapter(text: str, chapters: dict[str, str]) -> str:
    """Code of the directory chapter whose label is closest to the text."""
    codes = list(chapters)
    sims = embed([chapters[c] for c in codes]) @ embed([text])[0]
    return codes[int(np.argmax(sims))]


@lru_cache(maxsize=1)
def _corpus():
    """All Commission proposals since CORPUS_FROM with their title embeddings (built once)."""
    from .eurlex_api import list_range

    # Skip corrigenda (CELEX ending in R(01), ...): same proposal, not a separate precedent
    rows = [r for r in list_range(CORPUS_FROM, date.today().isoformat()) if r["title"] and "R(" not in r["initiative_id"]]
    vecs = embed([clean_title(r["title"]) for r in rows])
    return rows, vecs


def disruptive_acceptable(celex: str, title: str, filing_date: str) -> tuple[float | None, list[dict]]:
    rows, vecs = _corpus()
    end = date.fromisoformat(filing_date)
    start = (end - timedelta(days=365 * LOOKBACK_YEARS)).isoformat()
    idx = [i for i, r in enumerate(rows) if start <= r["filing_date"] < filing_date and r["initiative_id"] != celex]
    if not idx:
        return None, []
    sims = vecs[idx] @ embed([clean_title(title)])[0]
    order = np.argsort(sims)[::-1][:TOP_K]
    score = round(float(np.clip(sims[order].mean(), 0, 1)), 3)
    nearest = [
        {
            "initiative_id": rows[idx[j]]["initiative_id"],
            "filing_date": rows[idx[j]]["filing_date"],
            "title": rows[idx[j]]["title"],
            "similarity": round(float(sims[j]), 3),
        }
        for j in order[:3]
    ]
    return score, nearest
