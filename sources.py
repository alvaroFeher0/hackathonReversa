"""
One interface over every data source (apis/), for the training set and for predict.py.

Each source is a function  proposal -> {feature: value}  that may only use information dated
strictly BEFORE the proposal's filing date. Feature names are prefixed with the source name
("economy__brent_crude_13w_momentum").

    from sources import Proposal, get_features, SOURCES
    p = Proposal("52021PC0206", date(2021, 4, 21), "Proposal for a REGULATION ... (Artificial Intelligence Act) ...")
    get_features(p)                                   # fast sources only
    get_features(p, ["economy", "news"])              # pick sources
    python sources.py 52021PC0206 --sources all       # same, from the shell

Sources (cost = time per proposal on a cache miss):
    economy        free    13-week price momentum of the last full week before filing (parlament_economy)
    parliament     free    EP term, left/right score, seats, sector acceptance of the law (parlament_economy)
    eurlex         medium  text length, EUR-Lex directory category (eurlex_api; downloads the text)
    scores         medium  left/right of title+summary, similarity to earlier proposals and how many of
                           those were adopted before filing (eurlex_scores; ~30 s setup on first call)
    consultations  slow    public consultations published before filing (law_history, Have Your Say)
    news           slow    European headlines before filing and their sentiment (getNewsFile, GDELT 1 req / 6 s)

Deliberately NOT a source: law_history.get_eu_votings. Every EP vote happens after filing, so with
the cutoff it is always empty; use it for analysis and the demo only.

Caching: every non-free source caches its answer per proposal in data/features/<source>/<celex>.json,
so building the training set again is instant and slow sources can be filled in over several runs.
Permanent misses (nothing found) are cached as empty; network failures and rate limits are not,
so the next run retries them.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import requests

DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "features"


@dataclass(frozen=True)
class Proposal:
    celex: str          # proposal CELEX, e.g. 52021PC0206
    filing_date: date   # the cutoff: only information from before this day may be used
    title: str


@dataclass(frozen=True)
class Source:
    name: str
    fetch: Callable[[Proposal], dict]
    cost: str           # "free" | "medium" | "slow"
    cached: bool = True


class Transient(Exception):
    """Temporary failure (network, rate limit): not cached, retried next time."""


# --------------------------------------------------------------------------- sources

def economy(p: Proposal) -> dict:
    from apis import parlament_economy as pe
    rows = pe._read_dataset()
    # Last FULL week before the filing date: the filing week's average includes later days.
    # Raw momentum only: the *_z and score columns are scaled with statistics of the whole
    # 2021-2026 series, i.e. future data.
    past = [r for r in rows if date.fromisoformat(r["week_start"]) <= p.filing_date - timedelta(days=7)]
    if not past:
        return {}  # the dataset starts in 2021
    row = past[-1]
    return {f"{i.column}_13w_momentum": pe._to_float(row.get(f"{i.column}_13w_momentum"))
            for i in pe.INSTRUMENTS}


def parliament(p: Proposal) -> dict:
    from apis import parlament_economy as pe
    rows = [r for r in pe._read_parliament_context()
            if date.fromisoformat(r["valid_from"]) <= p.filing_date
            and (not r.get("valid_to") or p.filing_date <= date.fromisoformat(r["valid_to"]))]
    if not rows:
        return {}  # the context table starts in 2021
    row = rows[-1]
    # Which Parliament was sitting is known on the filing date. Caveat: the sector stances in this
    # table were written by hand recently, with hindsight.
    stance = pe._law_stance_payload(row, p.title)
    return {"term": row["parliament_term"], "left_right_score": pe._to_float(row.get("left_right_score")),
            "pro_eu_majority_seats": pe._to_int(row.get("pro_eu_majority_seats")),
            "right_populist_seats": pe._to_int(row.get("right_populist_or_hard_right_seats")),
            "law_sector": pe._detect_law_sector(p.title),
            "sector_acceptance": stance.get("sector_acceptance_score")}


def eurlex(p: Proposal) -> dict:
    from apis import eurlex_api
    d = eurlex_api.get_one(p.celex)  # text of the proposal itself: exists on the filing date
    if d is None:
        raise LookupError(f"{p.celex} not on EUR-Lex")
    return {"text_chars_act": d.get("text_chars_act"), "category": d.get("category"),
            "category_code": d.get("category_code"), "category_from_embedding": d.get("category_source") == "embedding"}


def scores(p: Proposal) -> dict:
    from apis import eurlex_api, eurlex_scores as es
    d = eurlex_api.get_one(p.celex)
    summary = (d or {}).get("summary") or ""
    score, nearest = es.disruptive_acceptable(p.celex, p.title, p.filing_date.isoformat())
    known = eurlex_api.outcomes([n["initiative_id"] for n in nearest])
    adopted_before = sum(1 for n in nearest
                         if (o := known.get(n["initiative_id"])) and o["adopted"] and o["date_adopted"]
                         and o["date_adopted"] < p.filing_date.isoformat())
    return {"left_right": es.left_right(f"{es.clean_title(p.title)}. {summary}"),
            "disruptive_acceptable": score,
            "nearest_similarity": nearest[0]["similarity"] if nearest else None,
            "nearest_adopted_before_filing": adopted_before if nearest else None}


def consultations(p: Proposal) -> dict:
    from apis import law_history
    try:
        n = law_history.get_consultation_count(p.celex, up_to=p.filing_date - timedelta(days=1))
    except LookupError:
        return {"n_before_filing": None, "found": 0}  # no Have Your Say initiative matched
    return {"n_before_filing": n, "found": 1}


def news(p: Proposal) -> dict:
    import pandas as pd
    from apis import getNewsFile as gn
    try:
        arts = gn.get_headlines(p.celex, p.title, pd.Timestamp(p.filing_date), retries=3)
    except gn.RateLimited as e:
        raise Transient(str(e))
    if arts.empty:
        return {"n_headlines": 0}
    s = gn.score_headlines(arts["title"].tolist())
    return {"n_headlines": len(s), "avg_sentiment": sum(x["sentiment"] for x in s) / len(s),
            "share_negative": sum(x["sentiment_label"] == "negative" for x in s) / len(s)}


SOURCES: dict[str, Source] = {s.name: s for s in [
    Source("economy", economy, "free", cached=False),
    Source("parliament", parliament, "free", cached=False),
    Source("eurlex", eurlex, "medium"),
    Source("scores", scores, "medium"),
    Source("consultations", consultations, "slow"),
    Source("news", news, "slow"),
]}
FAST = [n for n, s in SOURCES.items() if s.cost == "free"]


# ------------------------------------------------------------------------ interface

def _cache_path(source: str, celex: str) -> Path:
    return CACHE / source / f"{celex}.json"


def fetch(source: str, p: Proposal, refresh: bool = False) -> dict | None:
    """Features from one source, prefixed; None if it failed temporarily (retry later)."""
    src = SOURCES[source]
    path = _cache_path(source, p.celex)
    if src.cached and not refresh and path.exists():
        cached = json.loads(path.read_text())
        if cached.get("filing_date") == p.filing_date.isoformat():
            return cached["features"]
    try:
        raw = src.fetch(p)
    except (Transient, requests.RequestException, urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"   {source} {p.celex}: temporary failure ({e.__class__.__name__}: {str(e)[:80]}), not cached",
              file=sys.stderr)
        return None
    except (LookupError, ValueError) as e:
        print(f"   {source} {p.celex}: nothing found ({str(e)[:80]})", file=sys.stderr)
        raw = {}
    feats = {f"{source}__{k}": v for k, v in raw.items()}
    if src.cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"filing_date": p.filing_date.isoformat(), "features": feats}, default=str))
        tmp.rename(path)
    return feats


def get_features(p: Proposal, sources: list[str] | None = None, refresh: bool = False) -> dict:
    """All features for one proposal from the chosen sources (default: the free ones)."""
    out: dict = {}
    for name in sources or FAST:
        feats = fetch(name, p, refresh)
        out.update(feats or {})
    return out


def parse_sources(arg: str) -> list[str]:
    if arg == "all":
        return list(SOURCES)
    if arg == "fast":
        return FAST
    names = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources {sorted(unknown)}; choose from {list(SOURCES)}, 'fast' or 'all'")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("celex", help="proposal CELEX, e.g. 52021PC0206")
    ap.add_argument("--sources", default="fast", help="comma list, 'fast' (default) or 'all'")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    from apis import eurlex_api
    d = eurlex_api.get_one(args.celex.upper())
    if d is None:
        raise SystemExit(f"{args.celex} not found on EUR-Lex")
    p = Proposal(d["initiative_id"], date.fromisoformat(d["filing_date"][:10]), d["title"])
    print(json.dumps({"proposal": p.__dict__, "features": get_features(p, parse_sources(args.sources), args.refresh)},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
