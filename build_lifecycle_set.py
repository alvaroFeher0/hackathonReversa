"""
Lifecycle training set: 4 rows per proposal, one per query date, with features as known on that date.

    .venv/bin/python build_lifecycle_set.py                  # all proposals in eu_lifecycle_balanced.csv
    .venv/bin/python build_lifecycle_set.py --limit 5        # first 5 proposals (test)

Input: eu_lifecycle_balanced.csv (celex, filing_date, status, status_date; other columns are ignored).
Query dates: the filing date plus 3 random days strictly between the filing date and the end of the
lifecycle (status_date). Seeded per CELEX, so every run gives the same dates.

Per row:
    1) eurlex_api (celex)                author, author_dg, law_left_right, law_disruptive_acceptable,
                                         similar_laws_approval (share of the 3 closest earlier proposals
                                         adopted strictly before the query date: 1 adopted, 0 not)
    2) parlament_economy (query date, title from 1)
                                         parliament_left_right, sector_acceptance, econ_<category>
    3) law_history (celex, query date)   n_consultations, n_votings, in_favor_increase, in_favor_variability
Labels: status (passed / withdrawn / stuck), end_date (lifecycle end, only for passed).

Output: data/lifecycle_training_set.csv. Each proposal's rows are cached in
data/features/lifecycle/<celex>.json; a re-run only fetches what is missing. Network failures
are not cached, so the next run retries them.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from fastapi import HTTPException

from apis import eurlex_api, eurlex_scores, law_history, parlament_economy

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "eu_lifecycle_balanced.csv"
OUT = ROOT / "data" / "lifecycle_training_set.csv"
CACHE = ROOT / "data" / "features" / "lifecycle"
SEED = 42
N_RANDOM_DATES = 3
TRANSIENT = (requests.RequestException, urllib.error.URLError, TimeoutError, ConnectionError)


class Incomplete(Exception):
    """A source failed temporarily: the proposal is not cached and is retried next run."""


def query_dates(celex: str, filing: date, end: date) -> list[date]:
    """Filing date + N_RANDOM_DATES distinct days in (filing, end), sorted. Deterministic per CELEX."""
    rng = random.Random(f"{SEED}-{celex}")
    span = (end - filing).days
    if span <= 1:
        offsets = [0] * N_RANDOM_DATES
    elif span - 1 >= N_RANDOM_DATES:
        offsets = rng.sample(range(1, span), N_RANDOM_DATES)
    else:
        offsets = [rng.randrange(1, span) for _ in range(N_RANDOM_DATES)]
    return [filing] + sorted(filing + timedelta(days=o) for o in offsets)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def law_info(celex: str) -> dict | None:
    """eurlex_api analysis of the proposal; None if EUR-Lex does not have it."""
    try:
        return eurlex_api.analysis(celex)
    except HTTPException as e:
        print(f"   {celex}: eurlex {e.detail}", file=sys.stderr)
        return None


def similar_laws_approval(info: dict, qd: date) -> float | None:
    """Mean over the closest earlier proposals: 1 if adopted strictly before the query date, else 0."""
    vals = [1.0 if n["post_outcome"] == "adopted" and n["post_date_adopted"]
            and n["post_date_adopted"][:10] < qd.isoformat() else 0.0
            for n in info["closest_previous_proposals"] if n["post_outcome"] != "unknown"]
    return sum(vals) / len(vals) if vals else None


def economy_features(title: str, qd: date) -> dict:
    try:
        c = parlament_economy.combined(qd, title)
    except HTTPException as e:  # no context for that date
        print(f"   economy {qd}: {e.detail}", file=sys.stderr)
        return {}
    out = {"parliament_left_right": c["law_stance"]["left_right"],
           "sector_acceptance": c["law_stance"]["sector_acceptance_score"]}
    out.update({f"econ_{_slug(e['name'])}": e["score"] for e in c["economy"]})
    return out


def history_features(celex: str, qd: date) -> dict:
    out = {}
    try:
        out["n_consultations"] = law_history.get_consultation_count(celex, up_to=qd)
    except LookupError:  # no Have Your Say initiative matched
        out["n_consultations"] = None
    try:
        v = law_history.get_eu_votings(celex, up_to=qd)
        out.update(n_votings=v["n_votings"], in_favor_increase=v["in_favor_increase"],
                   in_favor_variability=v["in_favor_variability"])
    except LookupError:  # no procedure linked in Cellar
        out.update(n_votings=None, in_favor_increase=None, in_favor_variability=None)
    return out


def proposal_rows(celex: str, filing: date, dates: list[date]) -> list[dict]:
    """Feature rows of one proposal, one per query date. Raises Incomplete on network failures."""
    try:
        info = law_info(celex)
        title = info["title"] if info else ""
        rows = []
        for qd in dates:
            row = {"celex": celex, "filing_date": filing.isoformat(), "query_date": qd.isoformat()}
            if info:
                row.update(author=info["author"], author_dg=info["author_dg"],
                           law_left_right=info["left_right"],
                           law_disruptive_acceptable=info["disruptive_acceptable"],
                           similar_laws_approval=similar_laws_approval(info, qd))
            if title:
                row.update(economy_features(title, qd))
            row.update(history_features(celex, qd))
            rows.append(row)
        return rows
    except TRANSIENT as e:
        raise Incomplete(f"{e.__class__.__name__}: {str(e)[:80]}") from e


def cached_rows(celex: str, filing: date, dates: list[date], refresh: bool) -> list[dict]:
    path = CACHE / f"{celex}.json"
    key = [d.isoformat() for d in dates]
    if not refresh and path.exists():
        cached = json.loads(path.read_text())
        if cached["query_dates"] == key:
            return cached["rows"]
    rows = proposal_rows(celex, filing, dates)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"query_dates": key, "rows": rows}, default=str))
    tmp.rename(path)
    return rows


def build(limit: int | None = None, workers: int = 2, refresh: bool = False) -> pd.DataFrame:
    src = pd.read_csv(INPUT, dtype=str)
    if limit:
        src = src.head(limit)
    eurlex_scores._corpus()  # load the embedding model and corpus once, before the threads start
    eurlex_scores._anchors()

    jobs = {}
    for r in src.itertuples(index=False):
        filing, end = date.fromisoformat(r.filing_date), date.fromisoformat(r.status_date)
        jobs[r.celex] = (filing, query_dates(r.celex, filing, end))

    rows, failed, t0 = [], [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cached_rows, c, f, d, refresh): c for c, (f, d) in jobs.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows += fut.result()
            except Incomplete as e:
                failed.append(futs[fut])
                print(f"   {futs[fut]}: temporary failure ({e}), not cached", file=sys.stderr)
            except Exception as e:  # a bug for one proposal must not lose the whole run
                failed.append(futs[fut])
                print(f"   {futs[fut]}: ERROR {e.__class__.__name__}: {e}, not cached", file=sys.stderr)
            print(f"   {i}/{len(futs)} proposals, {time.time() - t0:.0f}s", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    if failed:
        print(f"{len(failed)} proposals failed, run again to retry: {failed}", file=sys.stderr)

    df = pd.DataFrame(rows)
    labels = src[["celex", "status", "status_date"]].rename(columns={"status_date": "end_date"})
    labels["status"] = labels["status"].str.lower()
    labels.loc[labels["status"] != "passed", "end_date"] = None
    df = df.merge(labels, on="celex", how="left")
    return df.sort_values(["filing_date", "celex", "query_date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="first N proposals only (for testing)")
    ap.add_argument("--workers", type=int, default=2, help="proposals fetched in parallel")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    df = build(args.limit, args.workers, args.refresh)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {df['celex'].nunique()} proposals, {df.shape[1]} columns")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
