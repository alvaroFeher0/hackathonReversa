"""Live API on EUR-Lex: every request reads the Commission proposal directly from the site.

    pip install fastapi uvicorn
    uvicorn eurlex_api:app --reload
    open http://127.0.0.1:8000/docs

Endpoints
    GET /proposals?date_from=2021-01-01&date_to=2026-10-03
        List of proposals in the range (metadata only, one SPARQL query: fast).
    GET /proposals/{celex}            e.g. /proposals/52021PC0206
        One proposal: dates, author, length of the act text (annexes excluded), summary.

Nothing is written to disk. Answers are kept in memory while the server runs,
so asking twice for the same proposal does not hit EUR-Lex again.
"""
import re
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup

from eurlex_texts import CELEX_URL, EURLEX_URL, SPARQL_URL, _html_parts, com_ref, summarise

app = FastAPI(title="EUR-Lex proposals")

PREFIXES = """PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
"""
ENG = "<http://publications.europa.eu/resource/authority/language/ENG>"


def run_sparql(query: str) -> list[dict]:
    r = requests.get(
        SPARQL_URL,
        params={"query": PREFIXES + query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=300,
    )
    r.raise_for_status()
    return [{k: v["value"] for k, v in row.items()} for row in r.json()["results"]["bindings"]]


def _metadata_where(celex_filter: str) -> str:
    return f"""
  ?w cdm:resource_legal_id_celex ?celex ;
     cdm:resource_legal_type "PC"^^xsd:string ;
     cdm:work_date_document ?date .
  {celex_filter}
  OPTIONAL {{ ?w cdm:resource_legal_service_responsible ?svc }}
  OPTIONAL {{ ?e cdm:expression_belongs_to_work ?w ;
                 cdm:expression_uses_language {ENG} ;
                 cdm:expression_title ?title }}
  OPTIONAL {{ ?w cdm:work_part_of_dossier ?d .
              ?d a cdm:procedure_code_interinstitutional ;
                 owl:sameAs ?proc .
              FILTER(CONTAINS(STR(?proc), "/procedure/")) }}"""


def _row(b: dict) -> dict:
    m = re.search(r"/procedure/(\d{4})_(\d+)$", b.get("proc", ""))
    return {
        "initiative_id": b["celex"],
        "com_ref": com_ref(b["celex"]),
        "procedure_ref": f"{m[1]}/{int(m[2]):04d}" if m else "",
        "filing_date": b["date"],
        "title": b.get("title", ""),
        "author": "European Commission",
        "author_dg": b.get("svc", "").rsplit("/", 1)[-1],
        "url": EURLEX_URL.format(celex=b["celex"]),
    }


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = {}
    for b in rows:
        seen.setdefault(b["celex"], b)
    return [_row(b) for b in sorted(seen.values(), key=lambda b: (b["date"], b["celex"]))]


@lru_cache(maxsize=32)
def list_range(date_from: str, date_to: str) -> list[dict]:
    flt = f'FILTER(?date >= "{date_from}"^^xsd:date && ?date <= "{date_to}"^^xsd:date)'
    return _dedupe(run_sparql(f"SELECT ?celex ?date ?title ?svc ?proc WHERE {{{_metadata_where(flt)}\n}}"))


def act_text(celex: str) -> str:
    """Plain text of the act only (first HTML part, annexes excluded)."""
    r = requests.get(
        CELEX_URL.format(celex=celex),
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        timeout=120,
    )
    if r.status_code == 300:
        links = _html_parts(r.text)
        if not links:
            return ""
        r = requests.get(links[0], timeout=120)
    if r.status_code != 200:
        return ""
    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    return "\n".join(ln for ln in lines if ln and not ln.startswith("IMMC."))


def outcomes(celexes: list[str]) -> dict[str, dict]:
    """Outcome today of each proposal, from its procedure: adopted 1/0, adoption date, final act.

    Missing from the result = EUR-Lex has no outcome data for it (unknown, not 'rejected').
    """
    if not celexes:
        return {}
    vals = " ".join(f'"{c}"^^xsd:string' for c in celexes)
    rows = run_sparql(f"""SELECT ?celex ?adopted ?date (MIN(?fc) AS ?final) WHERE {{
  VALUES ?celex {{ {vals} }}
  ?w cdm:resource_legal_id_celex ?celex ; cdm:work_part_of_dossier ?d .
  ?d a cdm:procedure_code_interinstitutional ; cdm:dossier_adopted-proposal ?adopted .
  OPTIONAL {{ ?d cdm:dossier_date_adopted ?date }}
  OPTIONAL {{ ?d cdm:dossier_contains_work ?f . ?f cdm:resource_legal_id_celex ?fc .
              FILTER(STRSTARTS(?fc, "3")) }}
}} GROUP BY ?celex ?adopted ?date""")
    return {
        r["celex"]: {"adopted": r["adopted"] == "1", "date_adopted": r.get("date"), "final_celex": r.get("final")}
        for r in rows
    }


DIR_URI = "http://publications.europa.eu/resource/authority/dir-eu-legal-act/"
# Chapters of the EUR-Lex directory of legislation (first two digits of the directory code)
CHAPTERS = {
    "01": "Institutions & budget",
    "02": "Customs",
    "03": "Agriculture",
    "04": "Fisheries",
    "05": "Employment & social policy",
    "06": "Services & establishment",
    "07": "Transport",
    "08": "Competition",
    "09": "Taxation",
    "10": "Economic & monetary policy",
    "11": "External relations & trade",
    "12": "Energy",
    "13": "Industry & internal market",
    "14": "Regional policy",
    "15": "Environment, consumers & health",
    "16": "Science, education & culture",
    "17": "Company law",
    "18": "Foreign & security policy",
    "19": "Justice, home affairs & migration",
    "20": "Citizens' Europe",
}


def category(celex: str) -> dict:
    """Official EUR-Lex directory code of the proposal -> chapter and sub-chapter labels."""
    rows = run_sparql(f"""SELECT ?c ?sub WHERE {{
  ?w cdm:resource_legal_id_celex "{celex}"^^xsd:string ;
     cdm:resource_legal_is_about_concept_directory-code ?c .
  BIND(IRI(CONCAT("{DIR_URI}", SUBSTR(STRAFTER(STR(?c), "{DIR_URI}"), 1, 4))) AS ?p)
  OPTIONAL {{ ?p <http://www.w3.org/2004/02/skos/core#prefLabel> ?sub FILTER(LANG(?sub) = "en") }}
}} ORDER BY ?c""")
    codes = [(r["c"].removeprefix(DIR_URI), r.get("sub", "")) for r in rows]
    if not codes:
        return {}
    code, sub = codes[0]
    return {
        "category": CHAPTERS.get(code[:2], code[:2]),
        "subcategory": sub,
        "category_code": code,
        "all_categories": sorted({CHAPTERS.get(c[:2], c[:2]) for c, _ in codes}),
        "category_source": "eurlex",
    }


@lru_cache(maxsize=512)
def get_one(celex: str) -> dict | None:
    flt = f'FILTER(?celex = "{celex}"^^xsd:string)'
    rows = _dedupe(run_sparql(f"SELECT ?celex ?date ?title ?svc ?proc WHERE {{{_metadata_where(flt)}\n}}"))
    if not rows:
        return None
    out = rows[0]
    text = act_text(celex)
    m = re.search(r"\b(\d{4}/\d{4})\s*\((\w+)\)", text[:3000])
    if m and (not out["procedure_ref"] or m[1] == out["procedure_ref"]):
        out["procedure_ref"] = f"{m[1]}({m[2]})"
    out["text_chars_act"] = len(text) if text else None
    out["summary"] = summarise(text, out["title"]) if text else out["title"]
    cat = category(celex)
    if not cat:  # ~6% of proposals have no directory code: nearest chapter by embeddings
        from eurlex_scores import clean_title, nearest_chapter

        code = nearest_chapter(f"{clean_title(out['title'])}. {out['summary']}", CHAPTERS)
        cat = {"category": CHAPTERS[code], "subcategory": "", "category_code": code,
               "all_categories": [CHAPTERS[code]], "category_source": "embedding"}
    out.update(cat)
    return out


@app.get("/proposals")
def proposals(date_from: str = "2021-01-01", date_to: str = "2026-10-03"):
    for d in (date_from, date_to):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            raise HTTPException(422, "dates must be YYYY-MM-DD")
    rows = list_range(date_from, date_to)
    return {"count": len(rows), "proposals": rows}


def _get_or_404(celex: str) -> dict:
    celex = celex.upper()
    if not re.fullmatch(r"5\d{4}PC\d{4}(\(\d+\))?", celex):
        raise HTTPException(422, "expected a CELEX number like 52021PC0206")
    out = get_one(celex)
    if out is None:
        raise HTTPException(404, f"{celex} not found on EUR-Lex")
    return out


@app.get("/proposals/{celex}")
def proposal(celex: str):
    return _get_or_404(celex)


@app.get("/proposals/{celex}/analysis")
def analysis(celex: str):
    """Title, summary, author and two semantic scores (see eurlex_scores.py).

    The first call loads the embedding model and embeds every proposal since 2016 (~30 s).
    """
    from eurlex_scores import clean_title, disruptive_acceptable, left_right

    p = _get_or_404(celex)
    score, nearest = disruptive_acceptable(p["initiative_id"], p["title"], p["filing_date"])
    # adopted_before_filing: known on the filing date, usable as a feature.
    # post_*: outcome as known today, analysis only (may be after the filing date).
    known = outcomes([n["initiative_id"] for n in nearest])
    for n in nearest:
        o = known.get(n["initiative_id"])
        if o is None:
            n.update(adopted_before_filing=None, post_outcome="unknown", post_date_adopted=None, post_final_celex=None)
            continue
        before = o["adopted"] and bool(o["date_adopted"]) and o["date_adopted"] < p["filing_date"]
        n.update(
            adopted_before_filing=before,
            post_outcome="adopted" if o["adopted"] else "not_adopted",
            post_date_adopted=o["date_adopted"],
            post_final_celex=o["final_celex"],
        )
    return {
        "initiative_id": p["initiative_id"],
        "title": p["title"],
        "summary": p["summary"],
        "author": p["author"],
        "author_dg": p["author_dg"],
        "filing_date": p["filing_date"],
        "category": p["category"],
        "subcategory": p["subcategory"],
        "category_source": p["category_source"],
        "left_right": left_right(f"{clean_title(p['title'])}. {p['summary']}"),
        "disruptive_acceptable": score,
        "closest_previous_proposals": nearest,
    }
