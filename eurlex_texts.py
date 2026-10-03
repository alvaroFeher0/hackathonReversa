"""Commission proposals (EUR-Lex, document type PC) in a date range, with full English text.

    python eurlex_texts.py [--from 2021-01-01] [--to 2026-10-03] [--limit N]

Same set as the EUR-Lex advanced search "Preparatory acts / Commission proposals" filtered on
"Date of document". One row per proposal -> data/eurlex_proposals.csv and .json

Steps
    1. CELLAR SPARQL: list of proposals (celex, date, title, DG, procedure)  -> data/raw/eurlex/sparql
    2. Every HTML part of the English text (act + annexes), cached on disk    -> data/raw/eurlex/texts/{celex}/
    3. Plain text -> length in characters + extractive summary

Join keys
    initiative_id  CELEX number, same key as data/ep_bills.csv
    com_ref        "COM(2021) 206", as shown on EUR-Lex
    procedure_ref  "2021/0106(COD)", key of the EP Legislative Observatory and of EP votes

The summary is extractive (first sentences of "Reasons for and objectives of the proposal"),
not generated: it comes only from the filing text, so it is known on the filing date.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"
CELEX_URL = "http://publications.europa.eu/resource/celex/{celex}"
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
RAW = Path("data/raw/eurlex")
OUT = Path("data/eurlex_proposals")
PAUSE = 1.0  # seconds between requests
SUMMARY_CHARS = 700

_last_call = 0.0


def _wait():
    global _last_call
    gap = PAUSE - (time.time() - _last_call)
    if gap > 0:
        time.sleep(gap)
    _last_call = time.time()


def sparql(query: str) -> list[dict]:
    """Run a SPARQL query; cache the raw JSON answer on disk."""
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    path = RAW / "sparql" / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        _wait()
        r = requests.get(
            SPARQL_URL,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return [{k: v["value"] for k, v in row.items()} for row in data["results"]["bindings"]]


def proposals_query(date_from: str, date_to: str) -> str:
    return f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT ?celex ?date ?title ?svc ?proc WHERE {{
  ?w cdm:resource_legal_id_celex ?celex ;
     cdm:resource_legal_type "PC"^^xsd:string ;
     cdm:work_date_document ?date .
  FILTER(?date >= "{date_from}"^^xsd:date && ?date <= "{date_to}"^^xsd:date)
  OPTIONAL {{ ?w cdm:resource_legal_service_responsible ?svc }}
  OPTIONAL {{ ?e cdm:expression_belongs_to_work ?w ;
                 cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ;
                 cdm:expression_title ?title }}
  OPTIONAL {{ ?w cdm:work_part_of_dossier ?d .
              ?d a cdm:procedure_code_interinstitutional ;
                 owl:sameAs ?proc .
              FILTER(CONTAINS(STR(?proc), "/procedure/")) }}
}}"""


def list_proposals(date_from: str, date_to: str) -> pd.DataFrame:
    df = pd.DataFrame(sparql(proposals_query(date_from, date_to)))
    df = df.sort_values(["celex", "title"], na_position="last").drop_duplicates("celex")
    df["svc"] = df["svc"].str.rsplit("/", n=1).str[-1]
    # .../procedure/2021_106 -> 2021/0106 (the procedure type, e.g. COD, is read from the text)
    yn = df["proc"].str.extract(r"/procedure/(\d{4})_(\d+)$")
    df["procedure_ref"] = (yn[0] + "/" + yn[1].str.zfill(4)).where(yn[0].notna())
    return df.drop(columns="proc").reset_index(drop=True)


def com_ref(celex: str) -> str:
    """52021PC0206 -> COM(2021) 206"""
    m = re.fullmatch(r"5(\d{4})PC(\d+)(.*)", celex)
    return f"COM({m[1]}) {int(m[2])}{m[3]}" if m else ""


def _html_parts(listing_html: str) -> list[str]:
    """Links of every HTML file in a CELLAR 300 'multiple choices' listing (act first, then annexes)."""
    soup = BeautifulSoup(listing_html, "html.parser")
    parts = []
    for item in soup.select("li[title=item]"):
        name = item.select_one("li[title=stream_name]")
        a = item.find("a", href=True)
        if a and name and name.get_text().lower().endswith((".html", ".xhtml", ".htm")):
            parts.append((name.get_text(), a["href"]))
    return [href for _, href in sorted(parts, key=lambda p: ("annex" in p[0].lower(), p[0]))]


def download_text(celex: str) -> list[Path]:
    """Download (once) all English HTML parts of a proposal; return the cached files."""
    folder = RAW / "texts" / celex
    done = folder / ".done"
    if done.exists():
        return sorted(folder.glob("*.html"))
    folder.mkdir(parents=True, exist_ok=True)
    _wait()
    r = requests.get(
        CELEX_URL.format(celex=celex),
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        timeout=120,
    )
    if r.status_code == 300:
        links = _html_parts(r.text)
    elif r.status_code == 200:
        links = []
        (folder / "part01.html").write_bytes(r.content)
    else:
        links = []
        print(f"{celex}: no English HTML text (HTTP {r.status_code})")
    for i, link in enumerate(links, 1):
        _wait()
        p = requests.get(link, timeout=120)
        if p.status_code == 200:
            (folder / f"part{i:02d}.html").write_bytes(p.content)
    done.touch()
    return sorted(folder.glob("*.html"))


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("IMMC.")]
    return "\n".join(lines)


def _paragraphs(text: str) -> list[str]:
    # Drop footnote markers (lines that are only a number) and section numbers like "1.1."
    return [ln for ln in text.split("\n") if not re.fullmatch(r"[\d.()]+", ln)]


def summarise(text: str, title: str) -> str:
    """First sentences of 'Reasons for and objectives', else of the explanatory memorandum."""
    paras = _paragraphs(text)
    start = None
    for pattern in (r"reasons for and objectives", r"context of the proposal", r"explanatory memorandum"):
        start = next((i + 1 for i, p in enumerate(paras) if re.search(pattern, p, re.I)), None)
        if start is not None:
            break
    if start is None:
        return title or ""
    body = []
    for p in paras[start:]:
        if len(p) < 60 and not body:  # still on headings
            continue
        if len(p) < 60 and p.isupper():  # next heading
            break
        body.append(p)
        if sum(map(len, body)) >= SUMMARY_CHARS:
            break
    summary = re.sub(r"\s+", " ", " ".join(body)).strip()
    if len(summary) > SUMMARY_CHARS:
        cut = summary[:SUMMARY_CHARS]
        end = cut.rfind(". ")
        summary = cut[: end + 1] if end > SUMMARY_CHARS // 3 else cut.rstrip() + "…"
    return summary or (title or "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", default="2021-01-01")
    p.add_argument("--to", dest="date_to", default="2026-10-03")
    p.add_argument("--limit", type=int, default=0, help="only the first N proposals (for testing)")
    args = p.parse_args()

    df = list_proposals(args.date_from, args.date_to).sort_values(["date", "celex"])
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} proposals between {args.date_from} and {args.date_to}")

    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        files = download_text(r.celex)
        texts = [html_to_text(f) for f in files]
        full = "\n".join(texts)
        title = r.title if isinstance(r.title, str) else ""
        proc = r.procedure_ref if isinstance(r.procedure_ref, str) else ""
        m = re.search(r"\b(\d{4}/\d{4})\s*\((\w+)\)", full[:3000])
        if m and (not proc or m[1] == proc):
            proc = f"{m[1]}({m[2]})"
        rows.append({
            "initiative_id": r.celex,
            "com_ref": com_ref(r.celex),
            "procedure_ref": proc,
            "filing_date": r.date,
            "title": title,
            "author": r.svc if isinstance(r.svc, str) else "",
            "is_legislative": bool(proc),
            "n_parts": len(files),
            "text_chars": len(full) if files else None,
            "text_chars_act": len(texts[0]) if files else None,
            "summary": summarise(texts[0], title) if files else title,
            "url": EURLEX_URL.format(celex=r.celex),
        })
        if i % 25 == 0:
            print(f"{i}/{len(df)}")

    out = pd.DataFrame(rows)
    out["text_chars"] = out["text_chars"].astype("Int64")
    out["text_chars_act"] = out["text_chars_act"].astype("Int64")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT.with_suffix(".csv"), index=False)
    out.to_json(OUT.with_suffix(".json"), orient="records", force_ascii=False, indent=1)
    print(f"{len(out)} rows -> {OUT}.csv / .json")
    print(f"with text: {out['text_chars'].notna().sum()}  legislative: {out['is_legislative'].sum()}")


if __name__ == "__main__":
    main()
