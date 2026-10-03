"""
Timing model: given a proposal and a query date anywhere in its lifecycle, predict when it becomes law.
Learns the days still to go (adoption date - query date) from what is known on the query date;
days_to_law = days_since_filing + predicted remaining days, so it is never earlier than the query date.

    .venv/bin/python fetch_ep_events.py                       # first: EP milestones (data/raw/ep_events/)
    .venv/bin/python train_timing_model.py                    # evaluate (train <= 2021, test 2022-2023), save
    .venv/bin/python train_timing_model.py --test-years 2023  # other test years (train = years before)
    .venv/bin/python train_timing_model.py --train-until 2025-12-31   # final fit also on recent filings

    from train_timing_model import proposal_inputs, predict_days       # predict.py
    df = proposal_inputs(["52021PC0206"], query_dates=["2023-01-15"])   # metadata + procedure by CELEX
    predict_days(df)                                                     # -> days_to_law (from filing)
    # minimal inputs also work: initiative_id, filing_date, title [, query_date, procedure, meta__*]

Rows: every adopted proposal since 2018 in data/eu_laws_metadata.csv (~1,700, all procedure types),
5 query dates each: the filing date, one early date (within the first 90 days or first 15% of the
lifecycle) and 3 uniform random dates between filing and adoption. Seeded per CELEX (deterministic).
Label: dossier date_adopted (build_training_set.add_labels).

Features, all known on the query date:
    filing-date: procedure type, resource type, DG, directory code, amends flag, counts of legal bases /
        cited works / EuroVoc terms, EEA flag (proposal metadata, Claude.md leakage map left column;
        EuroVoc describes the content, note 1), title keywords and title_ridge (TF-IDF ridge on log
        days_to_law, fitted out-of-fold by proposal on training rows only)
    query-date: days_since_filing, days to the next / since the last scheduled EP election, month
    EP and Council milestones (fetch_ep_events.py, OEIL key events): per stage, count and days since first/last event dated on or
        before the query date; events after the query date are never read (ep_features)
Pending / withdrawn outcomes, dossier dates other than the filing date and post_* columns are not read.

Model: LightGBM, absolute-error objective on log1p(remaining days) = conditional median (the scored
metric is mean absolute error in days). Leaves chosen on the last training year.
Censoring: recent proposals that already passed are the fast ones, so the final fit uses filings up to
--train-until (default end of 2023, the last year with < 15% still pending).

Output: data/timing_test_predictions.csv, models/timing_model.joblib
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import fetch_ep_events

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "models" / "timing_model.joblib"
SEED = 42
N_RANDOM_DATES = 3

# European Parliament elections, scheduled years in advance (as in pass_model.py).
EP_ELECTIONS = pd.to_datetime(["2014-05-25", "2019-05-26", "2024-06-09", "2029-06-07"])

KEYWORDS = {
    "ep_council": "european parliament and of the council", "council_dec": "council decision",
    "council_reg": "council regulation", "implementing": "implementing", "amend": "amend",
    "position": "position to be (?:adopted|taken)", "conclusion": "conclusion", "signing": "signing",
    "agreement": "agreement", "protocol": "protocol",
    "mobilisation": "mobilisation|globalisation adjustment|solidarity fund", "codif": "codification|recast",
    "repeal": "repeal", "temporary": "temporary|extension|prolong", "covid": "covid", "ukraine": "ukraine|russia",
    "budget": "budget|draft amending", "tariff": "tariff|duty|customs", "fish": "fish",
    "vat": "value added tax|vat", "appoint": "appoint|member of the", "derog": "derogat|authoris",
    "establish": "establishing|laying down", "framework": "framework",
}
# OEIL key-event labels grouped into lifecycle stages (fetch_ep_events.py): first matching pattern wins.
EP_STAGES = {
    "second_reading": r"2nd reading",
    "urgent": r"urgent procedure",
    "budget": r"budgetary report|draft budget",
    "reconsultation": r"reconsultation",
    "agreement": r"text agreed",
    "trilogue": r"interinstitutional negotiations",
    "referral": r"committee referral announced",
    "committee_vote": r"vote in committee",
    "report": r"committee report tabled",
    "ep_debate": r"debate in parliament",
    "ep_decision": r"decision by parliament",
    "council_position": r"council position|council's .*position|position of the council",
    "council_debate": r"debate in council",
    "council_adopted": r"act adopted by council",
    "signed": r"final act signed",
    "ep_done": r"end of procedure in parliament",
    "resumption": r"resumption of business",  # carried over to a new parliamentary term
}


def stage_of(label: str) -> str | None:
    label = str(label).lower()
    return next((s for s, pattern in EP_STAGES.items() if re.search(pattern, label)), None)


CATEGORICAL = ["procedure_type", "resource_type", "dg", "directory_1", "directory_2"]
STATIC = (["amends", "n_legal_bases", "n_cited_works", "n_eurovoc", "eea", "title_len"]
          + [f"kw_{k}" for k in KEYWORDS])
DYNAMIC = ["days_since_filing", "days_to_election", "days_since_election", "filing_days_to_election", "month",
           "ep_has_record", "ep_n_events", "ep_days_since_last"]
EP_FEATURES = [f"ep_{s}_{k}" for s in EP_STAGES for k in ("n", "since_first", "since_last")]
FEATURES = CATEGORICAL + STATIC + DYNAMIC + EP_FEATURES + ["title_ridge"]
PROPOSAL_TYPES = {"regulation": "PROP_REG", "directive": "PROP_DIR", "decision": "PROP_DEC"}


# ---------------------------------------------------------------- query dates and EP events

def query_dates(celex: str, filing: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Filing date, one early date and N_RANDOM_DATES uniform dates in (filing, end). Seeded per CELEX."""
    rng = random.Random(f"{SEED}-{celex}")
    span = (end - filing).days
    if span <= 1:
        return [filing]
    early = rng.randrange(1, max(2, min(90, int(0.15 * span))))
    offsets = {early} | {rng.randrange(1, span) for _ in range(N_RANDOM_DATES)}
    return [filing] + [filing + pd.Timedelta(days=o) for o in sorted(offsets)]


def load_events(procedures, fetch_missing: bool = False) -> dict[str, pd.DataFrame]:
    """procedure -> events (date, stage), from the fetch_ep_events.py cache. Procedures without a cache
    file (and no fetch) get None: unknown, as opposed to an empty frame = the EP has no record."""
    out = {}
    for pid in set(p for p in procedures if isinstance(p, str)):
        path = fetch_ep_events.CACHE / f"{pid}.json"
        if not path.exists() and not fetch_missing:
            out[pid] = None
            continue
        ev = pd.DataFrame(fetch_ep_events.fetch(pid) if fetch_missing else json.loads(path.read_text()),
                          columns=["date", "type"])
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
        ev["stage"] = ev["type"].map(stage_of)
        out[pid] = ev.dropna(subset=["date"]).sort_values("date")
    return out


def ep_features(procedure: pd.Series, query: pd.Series, events: dict) -> pd.DataFrame:
    """EP milestone features per row, counting only events dated on or before the query date."""
    rows = []
    for pid, qd in zip(procedure, query):
        ev = events.get(pid) if isinstance(pid, str) else None
        if ev is None:  # NLE (no EP role) or not fetched
            rows.append({"ep_has_record": 0, "ep_n_events": 0})
            continue
        ev = ev[ev["date"] <= qd]  # nothing after the query date
        r = {"ep_has_record": 1, "ep_n_events": len(ev),
             "ep_days_since_last": (qd - ev["date"].max()).days if len(ev) else np.nan}
        for stage in EP_STAGES:
            s = ev.loc[ev["stage"] == stage, "date"]
            r[f"ep_{stage}_n"] = len(s)
            if len(s):
                r[f"ep_{stage}_since_first"] = (qd - s.min()).days
                r[f"ep_{stage}_since_last"] = (qd - s.max()).days
        rows.append(r)
    return pd.DataFrame(rows, columns=["ep_has_record", "ep_n_events", "ep_days_since_last"] + EP_FEATURES)


# ---------------------------------------------------------------- features

def _count(s: pd.Series) -> pd.Series:
    return s.map(lambda v: 0 if v == "" else str(v).count("|") + 1)


def _election_days(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    nxt = np.searchsorted(EP_ELECTIONS.values, dates.values, side="left")
    to_next = (EP_ELECTIONS.values[nxt] - dates.values) / np.timedelta64(1, "D")
    since_last = (dates.values - EP_ELECTIONS.values[nxt - 1]) / np.timedelta64(1, "D")
    return to_next, since_last


def features(df: pd.DataFrame, events: dict | None = None) -> pd.DataFrame:
    """Features on the query date from initiative_id, filing_date, title [, query_date (default: filing
    date), procedure (EP id, e.g. 2021-0106), meta__* columns]; missing ones become NaN / 0.
    Shared by training and prediction. title_ridge is added by TimingModel."""
    df = df.reset_index(drop=True)
    meta = lambda name: df.get(f"meta__{name}", pd.Series("", index=df.index)).fillna("").astype(str)
    title = df["title"].fillna("").str.lower()
    filing = pd.to_datetime(df["filing_date"])
    query = pd.to_datetime(df["query_date"]) if "query_date" in df else filing
    from_title = title.str.extract(r"proposal for an? (?:council )?(regulation|directive|decision)")[0].map(PROPOSAL_TYPES)

    x = pd.DataFrame(index=df.index)
    x["procedure_type"] = meta("procedure_type").replace("", np.nan)
    x["resource_type"] = meta("resource_type").replace("", np.nan).fillna(from_title)
    x["dg"] = meta("dg").str.split("|").str[0].str.strip().replace("", np.nan)
    x["directory_1"] = meta("directory_code").str[:2].replace("", np.nan)
    x["directory_2"] = meta("directory_code").str[:4].replace("", np.nan)
    x["amends"] = (meta("proposes_to_amend") != "").astype(int)
    x["n_legal_bases"] = _count(meta("legal_bases"))
    x["n_cited_works"] = _count(meta("cited_works"))
    x["n_eurovoc"] = _count(meta("eurovoc"))
    x["eea"] = (meta("eea").str.strip() == "1").astype(int)
    x["title_len"] = title.str.len()
    for k, pattern in KEYWORDS.items():
        x[f"kw_{k}"] = title.str.contains(pattern).astype(int)

    x["days_since_filing"] = (query - filing).dt.days
    x["days_to_election"], x["days_since_election"] = _election_days(query)
    x["filing_days_to_election"], _ = _election_days(filing)
    x["month"] = query.dt.month
    procedure = df["procedure"] if "procedure" in df else pd.Series(None, index=df.index, dtype=object)
    ep = ep_features(procedure, query, events if events is not None else load_events(procedure))
    return pd.concat([x, ep.set_index(x.index)], axis=1)


class TimingModel:
    """Title TF-IDF ridge (stacked out-of-fold by proposal) + LightGBM median regressor on
    log1p(remaining days)."""

    def __init__(self, num_leaves: int = 16):
        self.num_leaves = num_leaves

    @staticmethod
    def _ridge(titles: np.ndarray, log_y: np.ndarray):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        return vec, Ridge(alpha=3.0).fit(vec.fit_transform(titles), log_y)

    def _x(self, x: pd.DataFrame, title_ridge: np.ndarray) -> pd.DataFrame:
        x = x.assign(title_ridge=title_ridge)[FEATURES].copy()
        for c in CATEGORICAL:
            known = x[c].where(x[c].isin(self.categories_[c]))  # unseen category (e.g. a new DG) -> NaN
            x[c] = pd.Categorical(known, categories=self.categories_[c])
        return x

    def fit(self, df: pd.DataFrame, remaining: np.ndarray, x: pd.DataFrame | None = None) -> "TimingModel":
        """df: one row per (proposal, query date) with days_to_law (the ridge's target, per proposal)."""
        x = features(df) if x is None else x
        titles = df["title"].fillna("").to_numpy()
        log_total = np.log1p(df["days_to_law"].to_numpy(dtype=float))
        oof = np.zeros(len(df))
        for tr, va in GroupKFold(5).split(titles, groups=df["initiative_id"]):
            first = pd.Series(tr).groupby(df["initiative_id"].to_numpy()[tr]).first().to_numpy()  # 1 row/proposal
            vec, ridge = self._ridge(titles[first], log_total[first])
            oof[va] = ridge.predict(vec.transform(titles[va]))
        first = df.reset_index(drop=True).groupby("initiative_id").head(1).index.to_numpy()
        self.vec_, self.ridge_ = self._ridge(titles[first], log_total[first])
        self.categories_ = {c: sorted(x[c].dropna().unique()) for c in CATEGORICAL}
        self.lgbm_ = LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=self.num_leaves,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                   reg_lambda=5.0, cat_smooth=20, min_data_per_group=20,
                                   random_state=SEED, deterministic=True, verbose=-1)
        self.lgbm_.fit(self._x(x, oof), np.log1p(np.asarray(remaining, dtype=float)))
        return self

    def predict_remaining(self, df: pd.DataFrame, x: pd.DataFrame | None = None) -> np.ndarray:
        x = features(df) if x is None else x
        ridge = self.ridge_.predict(self.vec_.transform(df["title"].fillna("").to_numpy()))
        return np.maximum(np.expm1(self.lgbm_.predict(self._x(x, ridge))), 0)

    def importance(self) -> pd.Series:
        return pd.Series(self.lgbm_.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)


# ---------------------------------------------------------------- data

def _metadata(since: str = "2018-01-01") -> pd.DataFrame:
    import build_training_set as bts  # reads the 180 MB metadata CSV
    p = bts.add_labels(bts.load_proposals(since=since))
    proc = fetch_ep_events.proposal_procedures(since).drop_duplicates("celex")
    return p.merge(proc.rename(columns={"celex": "initiative_id"}), on="initiative_id", how="left")


def load() -> pd.DataFrame:
    """One row per (adopted proposal, query date), with labels days_to_law and remaining."""
    p = _metadata()
    p = p[p["days_to_law"].notna()].reset_index(drop=True)
    p["law_date"] = p["filing_date"] + pd.to_timedelta(p["days_to_law"], unit="D")
    qd = [(i, q) for i, r in p.iterrows() for q in query_dates(r["initiative_id"], r["filing_date"], r["law_date"])]
    d = p.loc[[i for i, _ in qd]].reset_index(drop=True)
    d["query_date"] = [q for _, q in qd]
    d["remaining"] = (d["law_date"] - d["query_date"]).dt.days
    d["days_since_filing"] = (d["query_date"] - d["filing_date"]).dt.days
    d["lifecycle_share"] = (d["query_date"] - d["filing_date"]).dt.days / d["days_to_law"].clip(lower=1)  # report only
    assert (d["remaining"] >= 0).all(), "a query date after adoption"
    return d


def proposal_inputs(celexes, query_dates=None, fetch_missing: bool = True) -> pd.DataFrame:
    """Model inputs for proposals by CELEX (from data/eu_laws_metadata.csv); fetches EP events not
    cached yet. query_dates: one per CELEX (default: the filing date)."""
    p = _metadata(since="1990-01-01").drop_duplicates("initiative_id").set_index("initiative_id")
    df = p.reindex(list(celexes)).reset_index()
    if query_dates is not None:
        df["query_date"] = pd.to_datetime(list(query_dates))
    if fetch_missing:
        load_events(df["procedure"].dropna(), fetch_missing=True)
    return df


def predict_days(df: pd.DataFrame, path: Path = MODEL_PATH) -> np.ndarray:
    """days_to_law (filing -> adoption) on each row's query date, rounded to whole days."""
    x = features(df)
    remaining = joblib.load(path)["model"].predict_remaining(df, x)
    return np.round(x["days_since_filing"].to_numpy() + remaining).astype(int)


# ---------------------------------------------------------------- evaluation

def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {"MAE days": np.abs(err).mean(), "median AE": np.median(np.abs(err)), "bias (pred - true)": err.mean(),
            "within 90d": (np.abs(err) <= 90).mean(), "within 45d": (np.abs(err) <= 45).mean()}


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Remaining days from the median total duration per procedure type (floored at 0), and the median
    remaining days of training rows with the same procedure type and a similar days_since_filing."""
    first = train.drop_duplicates("initiative_id")
    overall = first["days_to_law"].median()
    total = test["meta__procedure_type"].map(first.groupby("meta__procedure_type")["days_to_law"].median()).fillna(overall)
    bins = [-1, 0, 30, 90, 180, 365, 540, 730, 1095, 100000]
    key = lambda d: d["meta__procedure_type"] + pd.cut(d["days_since_filing"], bins).astype(str)
    cond = train.groupby(key(train))["remaining"].median()
    return {"baseline: median total per procedure type": np.maximum(total.to_numpy() - test["days_since_filing"], 0),
            "baseline: median remaining per procedure x elapsed": key(test).map(cond).fillna(
                train["remaining"].median()).to_numpy()}


def report(test: pd.DataFrame, pred: np.ndarray, base: dict[str, np.ndarray]) -> None:
    truth = test["remaining"].to_numpy()
    rows = {"MODEL": metrics(truth, pred)} | {n: metrics(truth, b) for n, b in base.items()}
    print("== TEST, all query dates (error on days_to_law; scoring: <90 good, <45 excellent)")
    print(pd.DataFrame(rows).T.round(2).to_string())
    b = base["baseline: median remaining per procedure x elapsed"]
    stage = pd.cut(test["lifecycle_share"], [-0.01, 0, 0.25, 0.5, 0.75, 1.0],
                   labels=["filing date", "0-25%", "25-50%", "50-75%", "75-100%"])
    t = test.assign(stage=stage, model=np.abs(pred - truth), baseline=np.abs(b - truth))
    print("\n== MAE by position in the lifecycle (share of the true duration elapsed; baseline = procedure x elapsed)")
    print(t.groupby("stage", observed=True).agg(n=("model", "size"), model=("model", "mean"),
                                                baseline=("baseline", "mean")).round(0).to_string())
    print("\n== MAE by procedure type")
    print(t.groupby(t["meta__procedure_type"].replace("", "?")).agg(
        n=("model", "size"), model=("model", "mean"), baseline=("baseline", "mean")).round(0).to_string())


def choose_leaves(train: pd.DataFrame, x: pd.DataFrame) -> tuple[int, pd.Series]:
    """Validate on the last training year, fitting on the years before it."""
    year = train["filing_date"].dt.year
    fit, val = (year < year.max()).to_numpy(), (year == year.max()).to_numpy()
    mae = pd.Series({n: metrics(train["remaining"][val].to_numpy(),
                                TimingModel(n).fit(train[fit], train["remaining"][fit], x[fit])
                                .predict_remaining(train[val], x[val]))["MAE days"]
                     for n in (8, 16, 32)}, name=f"MAE on {year.max()}")
    return int(mae.idxmin()), mae


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-years", type=int, nargs="+", default=[2022, 2023])
    ap.add_argument("--train-until", default="2023-12-31", help="last filing date used in the final fit")
    args = ap.parse_args()
    from train_timing_model import TimingModel as Model  # so the pickle points to this module, not __main__

    d = load()
    x = features(d)
    covered = d["procedure"].map(lambda p: isinstance(p, str) and (fetch_ep_events.CACHE / f"{p}.json").exists())
    ep_role = d["procedure"].notna()
    print(f"{d['initiative_id'].nunique()} adopted proposals, {len(d)} rows; EP events cached for "
          f"{covered[ep_role].mean():.0%} of rows whose procedure involves the EP")

    year = d["filing_date"].dt.year
    tr, te = (year < min(args.test_years)).to_numpy(), year.isin(args.test_years).to_numpy()
    train, test = d[tr], d[te]
    for name, s in (("train", train), ("test", test)):
        f = s.drop_duplicates("initiative_id")
        print(f"{name:5s} {len(f):4d} proposals {s['filing_date'].dt.year.min()}-{s['filing_date'].dt.year.max()} "
              f"({len(s)} rows), days_to_law median {f['days_to_law'].median():.0f}, "
              f"{f['meta__procedure_type'].value_counts().to_dict()}")

    leaves, val = choose_leaves(train, x[tr])
    print(f"\n== leaves chosen on the last training year: {leaves}\n{val.round(1).to_string()}\n")
    model = Model(leaves).fit(train, train["remaining"], x[tr])
    pred = model.predict_remaining(test, x[te])
    report(test, pred, baselines(train, test))
    print("\n== top features (LightGBM gain share)")
    print((model.importance() / model.importance().sum()).head(15).round(3).to_string())

    fin = (d["filing_date"] <= args.train_until).to_numpy()
    final = Model(leaves).fit(d[fin], d["remaining"][fin], x[fin])
    test[["initiative_id", "meta__procedure_type", "filing_date", "query_date", "days_to_law"]].assign(
        pred_days_to_law=np.round(test["query_date"].sub(test["filing_date"]).dt.days + pred).astype(int)
    ).to_csv(DATA / "timing_test_predictions.csv", index=False)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "num_leaves": leaves, "train_until": args.train_until,
                 "target": "remaining days from query_date; days_to_law = days_since_filing + prediction"},
                MODEL_PATH)
    print(f"\nwrote data/timing_test_predictions.csv, {MODEL_PATH.relative_to(ROOT)} "
          f"(refit on {d[fin]['initiative_id'].nunique()} proposals filed up to {args.train_until})")


if __name__ == "__main__":
    main()
