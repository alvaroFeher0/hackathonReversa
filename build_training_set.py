"""
Training set: one row per Commission proposal, its labels, and features from every source.

    python build_training_set.py                               # free sources, all proposals since 2018
    python build_training_set.py --sources all --limit 50      # every source, first 50 (test)
    python build_training_set.py --sources fast,eurlex,scores  # add sources; cached ones cost nothing

    from build_training_set import build_training_set
    df = build_training_set(sources=["economy", "parliament"])

Output: data/training_set.parquet with
    initiative_id, filing_date, title           key and raw proposal fields
    meta__*                                     filing-day metadata from eu_laws_metadata.csv
    <source>__*                                 features from sources.py (known before filing)
    post_status, post_date_adopted              outcome today: labels only, NEVER model inputs
    is_law, days_to_law                         labels (see label_status)

Labels come from the proposal's dossier (its status today): adopted, withdrawn or pending.
Cellar has no dossier outcome before 2018, so the set starts in 2018.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sources

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "training_set.parquet"

PROCEDURE = "dossier_procedure_code_interinstitutional_has_type_concept_type_procedure_code_interinstitutional"
# Metadata fields that exist on the filing date (Claude.md leakage map). Everything else in the
# metadata file (dossier status and dates, end-of-validity, internal numbers...) is post-filing.
META_COLUMNS = {
    "resource_type": "resource_type",
    PROCEDURE: "procedure_type",
    "resource_legal_service_responsible": "dg",
    "resource_legal_is_about_concept_directory-code": "directory_code",
    "resource_legal_proposes_to_amend_resource_legal": "proposes_to_amend",
    "resource_legal_eea": "eea",
    "resource_legal_based_on_resource_legal": "legal_bases",
    "work_cites_work": "cited_works",
    "resource_legal_is_about_subject-matter": "subjects",
    "work_is_about_concept_eurovoc": "eurovoc",
}


def _flag(series: pd.Series) -> pd.Series:
    """Dossier flags can be multi-valued ("0 | 1"): true if any value is 1."""
    return series.map(lambda v: "1" in [t.strip() for t in str(v).split("|")])


def load_proposals(path: Path = DATA / "eu_laws_metadata.csv", since: str = "2018-01-01") -> pd.DataFrame:
    """Base table: proposals with filing-day metadata and their outcome today."""
    meta = pd.read_csv(path, dtype=str, keep_default_na=False)
    p = meta[(meta["kind"] == "proposal") & meta["celex"].str.match(r"^5\d{4}PC\d+$")].copy()
    p["filing_date"] = pd.to_datetime(p["work_date_document"], errors="coerce")
    p = p[p["filing_date"] >= since].dropna(subset=["filing_date"])

    out = pd.DataFrame({"initiative_id": p["celex"], "filing_date": p["filing_date"], "title": p["title"]})
    for src, name in META_COLUMNS.items():
        out[f"meta__{name}"] = p[src]
    status = np.select([_flag(p["dossier_dossier_adopted-proposal"]), _flag(p["dossier_dossier_withdrawn-proposal"]),
                        _flag(p["dossier_dossier_pending-proposal"])], ["adopted", "withdrawn", "pending"], "unknown")
    out["post_status"] = status
    out["post_date_adopted"] = pd.to_datetime(
        p["dossier_dossier_date_adopted"].str.split("|").str[0].str.strip(), errors="coerce")
    return out.sort_values(["filing_date", "initiative_id"]).reset_index(drop=True)


def add_labels(df: pd.DataFrame, stale_pending_years: float = 3.0, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """is_law: 1 adopted, 0 withdrawn or pending longer than stale_pending_years (stalled), NaN otherwise.
    days_to_law: filing -> adoption, for adopted proposals only."""
    snapshot = snapshot if snapshot is not None else df["filing_date"].max()  # data is as of its newest proposal
    stale_before = snapshot - pd.DateOffset(days=int(365.25 * stale_pending_years))
    df = df.copy()
    df["is_law"] = np.nan
    if stale_pending_years > 0:
        df.loc[(df["post_status"] == "pending") & (df["filing_date"] < stale_before), "is_law"] = 0
    df.loc[df["post_status"] == "withdrawn", "is_law"] = 0
    df.loc[df["post_status"] == "adopted", "is_law"] = 1
    df["label_status"] = np.where(df["is_law"].notna(), "labelled", "unlabelled")
    days = (df["post_date_adopted"] - df["filing_date"]).dt.days
    df["days_to_law"] = days.where((df["post_status"] == "adopted") & (days >= 0))
    return df


def add_source_features(df: pd.DataFrame, names: list[str], refresh: bool = False) -> pd.DataFrame:
    """One call to sources.get_features per proposal; slow sources print progress."""
    rows, t0 = [], time.time()
    slow = any(sources.SOURCES[n].cost != "free" for n in names)
    for i, r in enumerate(df.itertuples(index=False), 1):
        p = sources.Proposal(r.initiative_id, r.filing_date.date(), r.title)
        rows.append(sources.get_features(p, names, refresh))
        if slow and (i % 10 == 0 or i == len(df)):
            print(f"   {i}/{len(df)} proposals, {time.time() - t0:.0f}s", end="\r")
    if slow:
        print()
    feats = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, feats], axis=1)


def build_training_set(sources_list: list[str] | None = None, since: str = "2018-01-01", limit: int | None = None,
                       stale_pending_years: float = 3.0, refresh: bool = False) -> pd.DataFrame:
    df = load_proposals(since=since)
    if limit:
        df = df.head(limit)
    df = add_labels(df, stale_pending_years)
    return add_source_features(df, sources_list or sources.FAST, refresh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="fast", help=f"comma list of {list(sources.SOURCES)}, 'fast' or 'all'; "
                                                      "'fast,eurlex' adds to the free ones")
    ap.add_argument("--since", default="2018-01-01")
    ap.add_argument("--limit", type=int, help="first N proposals only (for testing slow sources)")
    ap.add_argument("--stale-pending-years", type=float, default=3.0,
                    help="pending longer than this counts as failed (is_law = 0); 0 = never")
    ap.add_argument("--refresh", action="store_true", help="ignore the source caches")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    names = []
    for part in args.sources.split(","):
        names += sources.parse_sources(part.strip())
    names = list(dict.fromkeys(names))
    print(f"sources: {names}")
    df = build_training_set(names, args.since, args.limit, args.stale_pending_years, args.refresh)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    lab = df[df["label_status"] == "labelled"]
    print(f"wrote {args.out}: {len(df)} proposals ({len(lab)} labelled: {int((lab.is_law == 1).sum())} law, "
          f"{int((lab.is_law == 0).sum())} not), {df.shape[1]} columns")
    feat_cols = [c for c in df.columns if "__" in c and not c.startswith("meta__")]
    if feat_cols:
        cover = df[feat_cols].notna().mean().groupby(lambda c: c.split("__")[0]).mean()
        print("source coverage (share of proposals with values): "
              + ", ".join(f"{k} {v:.0%}" for k, v in cover.items()))


if __name__ == "__main__":
    main()
