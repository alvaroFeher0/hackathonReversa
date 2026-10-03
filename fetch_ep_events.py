"""
European Parliament procedure events (dated milestones) for every proposal since 2018.

    .venv/bin/python fetch_ep_events.py            # all proposals in data/eu_laws_metadata.csv since 2018
    .venv/bin/python fetch_ep_events.py --limit 20

Source: the "Key events" table of the EP Legislative Observatory (OEIL) procedure file, public HTML
(the EP Open Data API has the same events but rate-limits to a few requests per minute). The procedure
comes from the proposal's interinstitutional reference (2021/0106/COD -> 2021/0106(COD)). Each event has
a date and a label ("Vote in committee, 1st reading", "Act adopted by Council after Parliament's 1st
reading", ...), Parliament and Council steps alike. The events happen after the filing date: features
built from them must only count events dated on or before the query date (train_timing_model.ep_features).

Cache: data/raw/oeil/<yyyy-nnnn>.json, one file per procedure ([] when OEIL has no file, e.g. most
NLE). Only procedures where the EP has a role are fetched. A re-run only fetches what is missing;
network failures are not cached.
Output: data/ep_events.csv (procedure, date, type), all cached procedures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "raw" / "oeil"
OUT = ROOT / "data" / "ep_events.csv"
OEIL = "https://oeil.secure.europarl.europa.eu/oeil/en/procedure-file?reference={}/{}({})"
HEADERS = {"User-Agent": "Mozilla/5.0 (hackathon-reversa)"}
PAUSE = 0.5  # seconds between requests per worker: be polite to the endpoint
REF = "dossier_procedure_code_interinstitutional_reference_procedure"


EP_ROLE = ("COD", "CNS", "APP", "BUD", "ACI", "INI")  # NLE files have no EP record: not fetched (no events)


def procedure_id(ref: str) -> str | None:
    """'2021/0106/COD' -> '2021-0106-COD' (first reference if several); None when the EP has no role."""
    m = re.match(r"\s*(\d{4})/(\d+)/([A-Z]+)", str(ref))
    return f"{m.group(1)}-{int(m.group(2)):04d}-{m.group(3)}" if m and m.group(3) in EP_ROLE else None


def parse_key_events(html: str) -> list[dict]:
    """(date, type) rows of the OEIL "Key events" table; type is the event label."""
    start, end = html.find("Key events"), html.find("Technical information")
    if start < 0:
        return []
    events = []
    for tr in BeautifulSoup(html[start:end if end > start else None], "html.parser").find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) >= 2 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", cells[0]):
            d, m, y = cells[0].split("/")
            events.append({"date": f"{y}-{m}-{d}", "type": cells[1]})
    return events


def proposal_procedures(since: str = "2018-01-01") -> pd.DataFrame:
    """celex -> EP procedure id for every Commission proposal filed since `since`."""
    m = pd.read_csv(ROOT / "data" / "eu_laws_metadata.csv", dtype=str, keep_default_na=False,
                    usecols=["celex", "kind", "work_date_document", REF])
    p = m[(m["kind"] == "proposal") & m["celex"].str.match(r"^5\d{4}PC\d+$") & (m["work_date_document"] >= since)]
    return pd.DataFrame({"celex": p["celex"], "procedure": p[REF].map(procedure_id)}).dropna()


def fetch(pid: str) -> list[dict]:
    path = CACHE / f"{pid}.json"
    if path.exists():
        return json.loads(path.read_text())
    year, num, kind = pid.split("-")
    for attempt in range(6):
        r = requests.get(OEIL.format(year, num, kind), headers=HEADERS, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(5 * 2 ** attempt)
            continue
        break
    if r.status_code == 404:
        events = []
    else:
        r.raise_for_status()
        events = parse_key_events(r.text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events))
    time.sleep(PAUSE)
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pids = sorted(proposal_procedures()["procedure"].unique())[: args.limit]
    rows, failed, t0 = [], [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, p): p for p in pids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += [{"procedure": futs[f], **e} for e in f.result()]
            except Exception as e:  # not cached: retried next run
                failed.append(futs[f])
                print(f"   {futs[f]}: {e.__class__.__name__} {str(e)[:80]}", file=sys.stderr)
            print(f"   {i}/{len(pids)} procedures, {time.time() - t0:.0f}s", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    pd.DataFrame(rows, columns=["procedure", "date", "type"]).sort_values(["procedure", "date"]).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} events, {len(pids) - len(failed)} procedures"
          + (f", {len(failed)} failed (run again)" if failed else ""))


if __name__ == "__main__":
    main()
