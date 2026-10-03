"""
News headlines published before each EU proposal, from the GDELT DOC 2.0 API
(public, no key, news in 65+ languages from 2017 onwards).

Leakage rule: every query ends the day BEFORE the proposal date, so only news that
existed before the proposal is collected.

Input:  data/eu_laws_metadata.csv, proposal rows only (batch mode). The API does not need it: a CELEX
        missing from that file (or a missing file) is looked up live on EUR-Lex.
Output: data/news_headlines.csv, one row per article:
        initiative_id, proposal_date, news_query, seendate, title, url, domain, language, sourcecountry

Usage (batch):
  python -m apis.getNewsFile --smoke                       # 5 most recent proposals, checks everything works
  python -m apis.getNewsFile                               # all proposals from 2017 (~1 request per proposal)
  python -m apis.getNewsFile --limit 200 --window-days 180

Usage (API):
  python -m apis.getNewsFile --serve                       # http://127.0.0.1:8000, docs at /docs
  python -m apis.getNewsFile --serve --host 0.0.0.0        # reachable from other machines on the network
  curl "http://127.0.0.1:8000/headlines?celex=52021PC0206"
  curl "http://127.0.0.1:8000/headlines?celex=52021PC0206&lang=english&limit=20"
  curl "http://127.0.0.1:8000/headlines?title=Artificial%20Intelligence%20Act&date=2021-04-21"
  curl "http://127.0.0.1:8000/sentiment?celex=52021PC0206"   # same headlines + sentiment per headline and average

  The cutoff is the document's own date (or `date`). For a proposal that is its filing date, which is
  what model features need. For an adopted law it is the law's date, so the headlines include news
  from after the proposal: fine for analysis or the demo, not as a model feature.

Only news from European outlets is kept (EUROPE below). GDELT refuses a query with all 40 countries
("too long"), so articles are filtered on their `sourcecountry` after download.

Rate limit: GDELT allows one request every 5 s per IP and blocks for minutes after a burst.
Every process and thread on this machine (batch job, API, several API requests at once) shares
one throttle file, data/raw/news/.gdelt_throttle: one call every PAUSE seconds, and after a 429
everybody stays silent for COOLDOWN seconds. Teammates on the same network share the IP, so
their calls count too. Every successful response is cached under data/raw/news/, so repeats are
free and a re-run resumes; a proposal that keeps failing is skipped and retried on the next run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
EURLEX_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
CDM = "http://publications.europa.eu/ontology/cdm#"
GDELT_START = pd.Timestamp("2017-01-01")  # DOC 2.0 full-text coverage starts here
PAUSE = 6  # seconds between requests (GDELT asks for at most one every 5 s)
COOLDOWN = 120  # seconds of silence for everybody after a 429
MAX_RECORDS = 250  # GDELT maximum per call

DATA = Path(__file__).resolve().parent.parent / "data"  # repo-root data/, from any working directory
RAW = DATA / "raw" / "news"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "eu-laws-news-headlines (hackathon, research use)"
THROTTLE = RAW / ".gdelt_throttle"  # holds the earliest time the next request may go out

# European source countries, as GDELT names them in an article's `sourcecountry`.
# Russia, Belarus and Turkey are left out on purpose; edit this list to change what counts as European.
EUROPE = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
    "Spain", "Sweden",                                                    # EU 27
    "United Kingdom", "Norway", "Switzerland", "Iceland", "Liechtenstein",  # UK, EFTA
    "Albania", "Bosnia and Herzegovina", "Kosovo", "Macedonia", "Montenegro", "Serbia",
    "Moldova", "Ukraine",                                                 # candidates
    "Czechia", "Bosnia-Herzegovina", "North Macedonia",                   # other spellings
]


def _norm(country: str) -> str:
    return re.sub(r"[^a-z]", "", country.lower())


EUROPE_NORM = {_norm(c) for c in EUROPE}

# Words that appear in almost every EU title and say nothing about the topic.
STOP = set("""
proposal regulation directive decision council european parliament union commission
amending amendment amend laying down establishing concerning regards respect behalf position
taken adopted within application applications certain rules framework member states state
community agreement protocol conclusion signing provisional between relating related other
their which with from into under that this these those than also such well following
implementing delegated annex annexes article articles period extension mobilisation
providing provision promotion partial form exchange letters republic session carried field
displaced workers equivalence appointing appointment appoint alternate member members proposed
replacing replacement federal kingdom
january february march april june july august september october november december
""".split())

EU_CONTEXT = '("European Union" OR "European Commission" OR Brussels)'


class RateLimited(Exception):
    """GDELT did not answer: rate limited (429) or the connection kept failing."""
    def __init__(self, retry_after: float, reason: str = "rate limit"):
        super().__init__(f"GDELT {reason}, retry in {retry_after:.0f}s")
        self.retry_after, self.reason = retry_after, reason


def build_query(title: str, max_terms: int = 4) -> str:
    """Turn a long EU proposal title into a short GDELT query.

    A short name in brackets, e.g. "(Artificial Intelligence Act)", is used as an exact phrase.
    Otherwise the first distinctive words of the subject are ANDed, plus an EU context term.
    """
    short = re.search(r"\(([A-Za-z][A-Za-z\- ]{6,60}?\b(?:act|regulation|directive|package|strategy))\)",
                      title, flags=re.I)
    if short:
        name = short.group(1).strip()
        return f'"{name.title() if name.isupper() else name}"'
    # Drop the boilerplate head: "Proposal for a REGULATION OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL on ..."
    # Titles come in any case (some are all caps), so work in lower case; GDELT ignores case anyway.
    subject = re.sub(r"^.*?\b(?:council|commission)\b\s*", "", title.lower(), count=1)
    subject = re.sub(r"\bof \d{1,2} \w+ \d{4}\b", " ", subject)  # dates like "of 27 april 2026"
    subject = re.sub(r"\([^)]*\)|\b\d[\w/.\-]*", " ", subject)  # bracketed refs, numbers like 2003/17/EC
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ\-]+", subject)
             if len(w) >= 4 and w.lower() not in STOP]
    seen, terms = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            terms.append(w)
        if len(terms) == max_terms:
            break
    if not terms:
        return ""
    return " ".join(terms) + " " + EU_CONTEXT


def _throttle(max_wait: float | None) -> None:
    """Wait for our turn in the machine-wide queue, then book the next slot PAUSE seconds later.

    Raises RateLimited instead of waiting when the turn is more than max_wait seconds away.
    """
    THROTTLE.parent.mkdir(parents=True, exist_ok=True)
    with open(THROTTLE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # one process/thread at a time; released when the file closes
        f.seek(0)
        text = f.read().strip()
        wait = (float(text) if text else 0.0) - time.time()
        if max_wait is not None and wait > max_wait:
            raise RateLimited(wait)
        if wait > 0:
            time.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(time.time() + PAUSE))


def _cool_down() -> None:
    """After a 429: nobody on this machine calls GDELT for COOLDOWN seconds."""
    with open(THROTTLE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read().strip()
        until = max(float(text) if text else 0.0, time.time() + COOLDOWN)
        f.seek(0)
        f.truncate()
        f.write(str(until))


def fetch_articles(query: str, start: pd.Timestamp, end: pd.Timestamp, cache: Path,
                   retries: int = 4, max_wait: float | None = None) -> list[dict]:
    """Article list for one query and window, cached as JSON. [] when GDELT has nothing or rejects the query.

    max_wait=None waits as long as the throttle says (batch); a number fails fast with RateLimited (API).
    """
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    params = {
        "query": query, "mode": "artlist", "format": "json",
        "maxrecords": MAX_RECORDS, "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    reason = "rate limit"
    for attempt in range(retries):
        _throttle(max_wait)
        try:
            r = SESSION.get(ENDPOINT, params=params, timeout=30)
        except requests.RequestException as e:
            # Dropped connections are common with GDELT; the throttle already spaces the retry.
            print(f"   {e.__class__.__name__}, retrying", file=sys.stderr)
            reason = "connection error"
            continue
        text = r.text.strip()
        if r.status_code == 429 or "limit requests" in text.lower() or r.status_code >= 500:
            print(f"   HTTP {r.status_code}: pausing all GDELT calls for {COOLDOWN}s", file=sys.stderr)
            _cool_down()
            reason = "rate limit"
            continue
        try:
            articles = json.loads(text).get("articles", []) if text else []
        except json.JSONDecodeError:
            # GDELT answers bad queries ("too short or too long"...) with plain text: cache as empty.
            print(f"   GDELT rejected query {query!r}: {text[:100]}", file=sys.stderr)
            articles = []
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(articles), encoding="utf-8")
        return articles
    raise RateLimited(COOLDOWN if reason == "rate limit" else PAUSE, reason)


def get_headlines(celex: str, title: str, filed: pd.Timestamp, window_days: int = 90,
                  retries: int = 4, query: str | None = None, lang: str | None = None,
                  max_wait: float | None = None) -> pd.DataFrame:
    """Headlines in the window before `filed`. Raises RateLimited if GDELT kept refusing."""
    end = filed - timedelta(days=1)  # strictly before the document date
    start = max(end - timedelta(days=window_days - 1), GDELT_START)
    query = query or build_query(title)
    if not query or end < GDELT_START:
        return pd.DataFrame()
    if lang:
        query += f" sourcelang:{lang}"
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cache = RAW / f"{re.sub(r'[^A-Za-z0-9]+', '_', celex)}_{start:%Y%m%d}_{end:%Y%m%d}_{qhash}.art.json"
    arts = pd.DataFrame(fetch_articles(query, start, end, cache, retries, max_wait))
    if arts.empty:
        return arts
    arts = arts[arts["sourcecountry"].fillna("").map(_norm).isin(EUROPE_NORM)]  # European outlets only
    seen = pd.to_datetime(arts["seendate"], format="%Y%m%dT%H%M%SZ")
    arts = arts[seen < filed].copy()  # belt and braces: nothing on or after the document date
    arts.insert(0, "initiative_id", celex)
    arts.insert(1, "proposal_date", filed.date())
    arts.insert(2, "news_query", query)
    return arts


def headlines_for(row: pd.Series, window_days: int) -> pd.DataFrame | None:
    """Batch helper: headlines before one proposal row; None if GDELT kept refusing."""
    try:
        return get_headlines(row["celex"], row["title"], row["proposal_date"], window_days)
    except RateLimited:
        return None


def eurlex_lookup(celex: str) -> dict | None:
    """Title (English) and document date of one CELEX, straight from EUR-Lex Cellar. None if unknown."""
    q = f"""
        SELECT ?date ?title WHERE {{
          ?w <{CDM}resource_legal_id_celex> "{celex}"^^<http://www.w3.org/2001/XMLSchema#string> ;
             <{CDM}work_date_document> ?date .
          OPTIONAL {{
            ?e <{CDM}expression_belongs_to_work> ?w ;
               <{CDM}expression_uses_language> <http://publications.europa.eu/resource/authority/language/ENG> ;
               <{CDM}expression_title> ?title .
          }}
        }} LIMIT 1"""
    r = SESSION.post(EURLEX_SPARQL, data={"query": q}, headers={"Accept": "application/sparql-results+json"},
                     timeout=30)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    if not rows:
        return None
    return {"celex": celex, "kind": {"5": "proposal", "3": "law"}.get(celex[0], "other"),
            "title": rows[0].get("title", {}).get("value", ""), "work_date_document": rows[0]["date"]["value"]}


# ------------------------------------------------------------------- sentiment

# Multilingual (XLM-RoBERTa) negative/neutral/positive classifier: European headlines come in many languages.
# Downloaded from Hugging Face on first use (~1 GB, cached in ~/.cache/huggingface).
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_classifier = None
_classifier_lock = threading.Lock()


def score_headlines(titles: list[str]) -> list[dict]:
    """Per headline: label, probabilities, and sentiment = P(positive) - P(negative), from -1 to +1."""
    global _classifier
    if not titles:
        return []
    with _classifier_lock:  # load once; the model is not safe to call from several threads at once
        if _classifier is None:
            from transformers import pipeline
            _classifier = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None)
        results = _classifier(titles, batch_size=32, truncation=True, max_length=128)
    scored = []
    for res in results:
        probs = {d["label"].lower(): d["score"] for d in res}
        scored.append({
            "sentiment_label": max(probs, key=probs.get),
            "sentiment": round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4),
            **{f"p_{k}": round(v, 4) for k, v in sorted(probs.items())},
        })
    return scored


# ------------------------------------------------------------------------- API

def create_app(metadata_path: Path = DATA / "eu_laws_metadata.csv"):
    from fastapi import FastAPI, HTTPException, Query

    app = FastAPI(title="EU law news headlines",
                  description="News headlines from European outlets (GDELT) published before an EU proposal "
                              "or law, and their sentiment.")
    docs: dict[str, dict] = {}
    docs_lock = threading.Lock()

    def lookup(celex: str) -> dict:
        celex = celex.strip().upper()
        if not re.fullmatch(r"[0-9A-Z()_\-]{6,30}", celex):
            raise HTTPException(400, f"{celex!r} does not look like a CELEX id (e.g. 52021PC0206)")
        with docs_lock:
            if not docs and metadata_path.exists():  # local metadata, loaded once
                meta = pd.read_csv(metadata_path, dtype=str, keep_default_na=False,
                                   usecols=["celex", "kind", "title", "work_date_document"])
                docs.update({r["celex"]: r for r in meta.to_dict("records")})
            if celex in docs:
                return docs[celex]
        try:
            doc = eurlex_lookup(celex)  # not in the local file (or no file): ask EUR-Lex
        except requests.RequestException as e:
            raise HTTPException(503, f"EUR-Lex lookup failed ({e.__class__.__name__}); pass title and date instead")
        if doc is None:
            raise HTTPException(404, f"CELEX {celex} not found on EUR-Lex")
        with docs_lock:
            docs[celex] = doc
        return doc

    def find_headlines(celex, title, date, query, lang, window_days, limit) -> dict:
        """Shared by both endpoints: resolve the law, fetch its headlines, build the response."""
        kind = None
        if celex:
            doc = lookup(celex)
            celex = doc["celex"]
            title, kind = title or doc["title"], doc["kind"]
            date = date or doc["work_date_document"]
        if not title and not query:
            raise HTTPException(400, "Give a celex, or a title (or query) and a date")
        if not date:
            raise HTTPException(400, "No date known for this document: pass date=YYYY-MM-DD")
        try:
            filed = pd.Timestamp(date).tz_localize(None).normalize()
        except (ValueError, TypeError):
            raise HTTPException(400, f"Bad date {date!r}: use YYYY-MM-DD")
        try:
            arts = get_headlines(celex or "adhoc", title or "", filed, window_days, retries=2,
                                 query=query, lang=lang, max_wait=20)
        except RateLimited as e:
            secs = max(int(e.retry_after) + 1, 1)
            raise HTTPException(503, f"GDELT {e.reason}: try again in {secs}s (cached laws still answer instantly)",
                                headers={"Retry-After": str(secs)})
        end = filed - timedelta(days=1)
        cols = ["seendate", "title", "url", "domain", "language", "sourcecountry"]
        items = arts[[c for c in cols if c in arts.columns]].head(limit).to_dict("records") if len(arts) else []
        return {
            "celex": celex, "kind": kind, "title": title, "cutoff_date": str(filed.date()),
            "window_start": str(max(end - timedelta(days=window_days - 1), GDELT_START).date()),
            "window_end": str(end.date()),
            "query": arts["news_query"].iloc[0] if len(arts) else query or build_query(title or ""),
            "n_headlines": len(items), "headlines": items,
        }

    params = dict(
        celex=Query(None, description="Proposal or law CELEX, e.g. 52021PC0206"),
        title=Query(None, description="Title or short name, when there is no CELEX"),
        date=Query(None, description="Cutoff YYYY-MM-DD: only news before this day. Defaults to the document's own date"),
        query=Query(None, description="Override the GDELT query built from the title"),
        lang=Query(None, description="Only articles in this source language, e.g. english, spanish"),
        window_days=Query(90, ge=1, le=3650),
        limit=Query(MAX_RECORDS, ge=1, le=MAX_RECORDS),
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/headlines")
    def headlines(celex: str | None = params["celex"], title: str | None = params["title"],
                  date: str | None = params["date"], query: str | None = params["query"],
                  lang: str | None = params["lang"], window_days: int = params["window_days"],
                  limit: int = params["limit"]):
        return find_headlines(celex, title, date, query, lang, window_days, limit)

    @app.get("/sentiment")
    def sentiment(celex: str | None = params["celex"], title: str | None = params["title"],
                  date: str | None = params["date"], query: str | None = params["query"],
                  lang: str | None = params["lang"], window_days: int = params["window_days"],
                  limit: int = params["limit"]):
        """Same headlines as /headlines, each scored with SENTIMENT_MODEL, plus the averages."""
        out = find_headlines(celex, title, date, query, lang, window_days, limit)
        scores = score_headlines([h["title"] for h in out["headlines"]])
        for h, sc in zip(out["headlines"], scores):
            h.update(sc)
        n = len(scores)
        out["sentiment_model"] = SENTIMENT_MODEL
        out["average"] = None if not n else {
            "sentiment": round(sum(sc["sentiment"] for sc in scores) / n, 4),  # -1 negative … +1 positive
            **{k: round(sum(sc[k] for sc in scores) / n, 4) for k in ("p_negative", "p_neutral", "p_positive")},
            "label_counts": {lab: sum(sc["sentiment_label"] == lab for sc in scores)
                             for lab in ("negative", "neutral", "positive")},
        }
        return out

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DATA / "eu_laws_metadata.csv"))
    ap.add_argument("--from-year", type=int, default=2017)
    ap.add_argument("--to-year", type=int, default=2100)
    ap.add_argument("--window-days", type=int, default=90, help="days of news before the proposal date")
    ap.add_argument("--limit", type=int, help="only the first N proposals (most recent first)")
    ap.add_argument("--smoke", action="store_true", help="quick test: 5 proposals")
    ap.add_argument("--serve", action="store_true", help="run the HTTP API instead of the batch job")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.serve:
        import uvicorn
        uvicorn.run(create_app(Path(args.input)), host=args.host, port=args.port)
        return
    if args.smoke:
        args.limit = 5

    meta = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    props = meta[meta["kind"] == "proposal"].copy()
    props["proposal_date"] = pd.to_datetime(props["work_date_document"], errors="coerce")
    props = props.dropna(subset=["proposal_date"])
    props = props[(props["proposal_date"] > GDELT_START + timedelta(days=1))
                  & props["proposal_date"].dt.year.between(args.from_year, args.to_year)]
    props = props.sort_values("proposal_date", ascending=False).reset_index(drop=True)
    if args.limit:
        props = props.head(args.limit)
    print(f"{len(props)} proposals, {args.window_days}-day window before each proposal date")

    found, skipped = [], []
    for n, (_, row) in enumerate(props.iterrows(), 1):
        arts = headlines_for(row, args.window_days)
        if arts is None:
            skipped.append(row["celex"])
            print(f"   {n}/{len(props)} {row['celex']}: skipped (rate limited, re-run to retry)")
            continue
        found.append(arts)
        first = arts["title"].iloc[0][:70] if len(arts) else ""
        print(f"   {n}/{len(props)} {row['celex']}: {len(arts)} headlines  {first}")

    out = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "news_headlines.csv", index=False)
    print(f"wrote data/news_headlines.csv: {len(out)} headlines for "
          f"{out['initiative_id'].nunique() if len(out) else 0} of {len(props)} proposals"
          + (f"; {len(skipped)} skipped, re-run to retry" if skipped else ""))


if __name__ == "__main__":
    main()
