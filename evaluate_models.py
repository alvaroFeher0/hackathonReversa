"""
Training and test metrics of both models, next to their baselines, in one CSV.

    .venv/bin/python evaluate_models.py          # -> reports/model_metrics.csv

Approval model (models/lifecycle_model.joblib): scored as saved, on the split it was trained with
(data/lifecycle_train.csv / data/lifecycle_test.csv, written by train_lifecycle_model.py).

Timing model (models/timing_model.joblib): scored as saved. Train = the rows it was fit on (adopted
proposals filed up to its train_until date); test = adopted proposals filed after that date, which
it never saw. The test set is biased towards fast files: recent proposals that already became law
are the quick ones, the slow ones are still pending.

Baselines are computed from each model's own training rows only.

Output columns: model, predictor, split, rows, proposals, metric, value.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_lifecycle_model as L  # noqa: E402
import train_timing_model as T  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "model_metrics.csv"


def rows(model: str, predictor: str, split: str, df: pd.DataFrame, metrics: dict, id_col: str) -> list[dict]:
    return [{"model": model, "predictor": predictor, "split": split, "rows": len(df),
             "proposals": df[id_col].nunique(), "metric": k, "value": round(float(v), 4)}
            for k, v in metrics.items()]


def approval() -> list[dict]:
    bundle = joblib.load(ROOT / "models" / "lifecycle_model.joblib")
    model, features = bundle["model"], bundle["features"]
    parse = ["filing_date", "query_date"]
    train = pd.read_csv(L.DATA / "lifecycle_train.csv", parse_dates=parse)
    test = pd.read_csv(L.DATA / "lifecycle_test.csv", parse_dates=parse)
    assert set(train["celex"]) == set(bundle["trained_on"]), "lifecycle_train.csv is not the saved model's split"

    name, out = "approval (lifecycle_model)", []
    rate = train["y"].mean()
    per_proc = train.groupby("procedure_type")["y"].mean()
    for split, d in (("train", train), ("test", test)):
        p = model.predict_proba(d[features])[:, 1]
        first = (d["query_date"] == d["filing_date"]).to_numpy()
        out += rows(name, "model", split, d, L.metrics(d["y"], p), "celex")
        out += rows(name, "model, filing-date rows only", split, d[first], L.metrics(d["y"][first], p[first]), "celex")
        out += rows(name, "baseline: train approval rate", split, d, L.metrics(d["y"], np.full(len(d), rate)), "celex")
        out += rows(name, "baseline: approval rate per procedure type", split, d,
                    L.metrics(d["y"], d["procedure_type"].map(per_proc).fillna(rate).to_numpy()), "celex")
    return out


def timing() -> list[dict]:
    bundle = joblib.load(T.MODEL_PATH)
    model = bundle["model"]
    d = T.load()
    x = T.features(d)
    tr = (d["filing_date"] <= bundle["train_until"]).to_numpy()
    train, test = d[tr], d[~tr]

    name, out = "timing (timing_model)", []
    for split, s, xs in (("train", train, x[tr]), ("test", test, x[~tr])):
        truth = s["remaining"].to_numpy()
        out += rows(name, "model", split, s, T.metrics(truth, model.predict_remaining(s, xs)), "initiative_id")
        for b, pred in T.baselines(train, s).items():
            out += rows(name, b, split, s, T.metrics(truth, pred), "initiative_id")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    df = pd.DataFrame(approval() + timing())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.pivot_table(index=["model", "predictor", "split"], columns="metric", values="value", sort=False)
          .round(3).to_string())
    print(f"\nwrote {args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
