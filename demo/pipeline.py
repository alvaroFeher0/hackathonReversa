"""
Demo pipeline: CELEX + query date -> model inputs -> p_law and days_to_law.

Uses the same code that built the training sets, so the models see exactly the features they were
trained on:
    approval model  build_lifecycle_set.proposal_rows (features as known on the query date)
                    + procedure_type and days_since_filing (as in train_lifecycle_model.load)
    timing model    train_timing_model.predict_days: proposal metadata (the build_training_set.META_COLUMNS
                    fields) + EP milestones dated on or before the query date (fetch_ep_events)

The metadata is read live from the Cellar SPARQL endpoint (left column of the leakage map only), so
proposals missing from data/eu_laws_metadata.csv work too.
"""
from __future__ import annotations

import os

# torch (embeddings) and LightGBM each bring their own OpenMP runtime; on macOS a LightGBM predict
# after torch has run segfaults unless both stay single-threaded. Must be set before either loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_lifecycle_set  # noqa: E402
import fetch_ep_events  # noqa: E402
import train_timing_model  # noqa: E402  (the pickled TimingModel lives in this module)
from apis import eurlex_api, eurlex_scores  # noqa: E402

LIFECYCLE_MODEL = ROOT / "models" / "lifecycle_model.joblib"

# Cellar property -> build_training_set.META_COLUMNS name. All known on the filing date.
META_PROPERTIES = {
    "resource_legal_service_responsible": "dg",
    "resource_legal_is_about_concept_directory-code": "directory_code",
    "resource_legal_proposes_to_amend_resource_legal": "proposes_to_amend",
    "resource_legal_eea": "eea",
    "resource_legal_based_on_resource_legal": "legal_bases",
    "work_cites_work": "cited_works",
    "work_is_about_concept_eurovoc": "eurovoc",
    "work_has_resource-type": "resource_type",
}
DOSSIER_PROPERTIES = {
    "procedure_code_interinstitutional_has_type_concept_type_procedure_code_interinstitutional": "procedure_type",
    "procedure_code_interinstitutional_reference_procedure": "procedure_ref",  # e.g. 2021/0106/COD
}


class PipelineError(Exception):
    """Input the pipeline cannot handle (unknown CELEX, query date before filing...)."""


def warm_up() -> None:
    """Load the embedding model and embed the proposal corpus once (the slow part of a first run)."""
    eurlex_scores._corpus()
    eurlex_scores._anchors()


def proposal_metadata(celex: str) -> dict[str, str]:
    """meta__* fields (formatted like data/eu_laws_metadata.csv) and the procedure reference."""
    props = ", ".join(f"cdm:{p}" for p in META_PROPERTIES)
    dossier = ", ".join(f"cdm:{p}" for p in DOSSIER_PROPERTIES)
    rows = eurlex_api.run_sparql(f"""SELECT ?p ?o WHERE {{
  ?w cdm:resource_legal_id_celex "{celex}"^^xsd:string .
  {{ ?w ?p ?o . FILTER(?p IN ({props})) }}
  UNION {{ ?w cdm:work_part_of_dossier ?d . ?d a cdm:procedure_code_interinstitutional ; ?p ?o .
           FILTER(?p IN ({dossier})) }}
}}""")
    names = META_PROPERTIES | DOSSIER_PROPERTIES
    values: dict[str, list[str]] = {}
    for r in rows:
        name = names[r["p"].rsplit("#", 1)[-1]]
        o = r["o"]
        if name == "eurovoc":
            o = "eurovoc:" + o.rsplit("/", 1)[-1]
        elif name in ("dg", "directory_code", "resource_type", "procedure_type"):
            o = o.rsplit("/", 1)[-1]
        values.setdefault(name, []).append(o)
    out = {f"meta__{n}": " | ".join(sorted(set(values.get(n, [])))) for n in names.values()}
    out["procedure_ref"] = out.pop("meta__procedure_ref")
    return out


def ep_milestones(pid: str | None, query_date: date) -> list[dict]:
    """EP / Council key events on or before the query date (nothing later is read)."""
    if not pid:
        return []
    ev = train_timing_model.load_events([pid], fetch_missing=True)[pid]
    if ev is None or ev.empty:
        return []
    ev = ev[ev["date"] <= pd.Timestamp(query_date)]
    return [{"date": d.date(), "event": t, "stage": s} for d, t, s in ev[["date", "type", "stage"]].itertuples(index=False)]


def run(celex: str, query_date: date) -> dict:
    """Everything the demo shows: proposal info, model inputs, both predictions."""
    celex = celex.strip().upper()
    try:
        info = eurlex_api.get_one(celex)
    except Exception as e:  # network / SPARQL failure
        raise PipelineError(f"EUR-Lex lookup failed: {e}") from e
    if info is None:
        raise PipelineError(f"{celex} is not a Commission proposal on EUR-Lex")
    filing = date.fromisoformat(info["filing_date"][:10])
    if query_date < filing:
        raise PipelineError(f"query date {query_date} is before the proposal date {filing}")

    meta = proposal_metadata(celex)
    procedure_ref = meta.pop("procedure_ref")
    pid = fetch_ep_events.procedure_id(procedure_ref)

    # Approval model: the lifecycle training rows, built by the same function
    row = build_lifecycle_set.proposal_rows(celex, filing, [query_date])[0]
    row["procedure_type"] = meta["meta__procedure_type"] or None
    row["days_since_filing"] = (query_date - filing).days
    bundle = joblib.load(LIFECYCLE_MODEL)
    x = pd.DataFrame([row]).reindex(columns=bundle["features"])
    p_law = float(bundle["model"].predict_proba(x)[:, 1][0])

    # Timing model: days from filing to law, given what is known on the query date
    milestones = ep_milestones(pid, query_date)  # also caches the events read by predict_days
    timing_in = pd.DataFrame([{"initiative_id": celex, "filing_date": pd.Timestamp(filing), "title": info["title"],
                               "query_date": pd.Timestamp(query_date), "procedure": pid, **meta}])
    days_to_law = int(train_timing_model.predict_days(timing_in)[0])
    law_date = (pd.Timestamp(filing) + pd.Timedelta(days=days_to_law)).date()

    return {
        "celex": celex, "query_date": query_date, "filing_date": filing,
        "title": info["title"], "summary": info.get("summary", ""), "url": info["url"],
        "procedure_ref": procedure_ref or info.get("procedure_ref", ""), "category": info.get("category", ""),
        "dg": info.get("author_dg", ""), "meta": meta, "milestones": milestones,
        "features": row, "model_features": bundle["features"],
        "p_law": p_law, "days_to_law": days_to_law, "law_date": law_date,
        "days_left": (law_date - query_date).days,
    }
