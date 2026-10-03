"""
Approval model on the lifecycle set: probability (0-1) that a proposal is approved, given what is
known on a query date.

    .venv/bin/python build_lifecycle_set.py       # first: data/lifecycle_training_set.csv
    .venv/bin/python train_lifecycle_model.py     # split, tune, fit, evaluate, save
    .venv/bin/python train_lifecycle_model.py --with-economy   # also use the econ_* columns
    .venv/bin/python train_lifecycle_model.py --rounds 5       # 5 rotating 80/20 splits, averaged
    .venv/bin/python train_lifecycle_model.py --compare 5      # every candidate model on the same 5 splits

Labels: passed = 1; withdrawn = 0; stuck = 0 only if pending for >= --min-stuck-years at SNAPSHOT.
Younger stuck proposals are still open (they may pass), so they are dropped, not labelled 0.

Split: by proposal. All rows of a proposal (its 4 query dates) are either in train or in test,
stratified by label, fixed seed. Hyper-parameters are chosen with grouped cross-validation on the
training set only; the test set is scored once, at the end.
--rounds N: the proposals are shuffled and cut into N groups; each group is the test set once
(80/20 for N=5), with model selection redone on the other groups only. Metrics are averaged.
A split by time is not usable here: after the 2-year rule there are almost no failed proposals
filed after 2024Q1, so a "latest years" test set would be nearly all passes.

econ_* columns are left out by default: the economy scores are scaled with statistics of the whole
2021-2026 series (future data). author is constant (European Commission) and not used.

Output: data/lifecycle_train.csv, data/lifecycle_test.csv, data/lifecycle_test_predictions.csv,
        models/lifecycle_model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 42
SNAPSHOT = pd.Timestamp("2026-10-03")  # date the lifecycle statuses were taken

NUMERIC = ["law_left_right", "law_disruptive_acceptable", "similar_laws_approval", "parliament_left_right",
           "sector_acceptance", "n_consultations", "n_votings", "in_favor_increase", "in_favor_variability",
           "days_since_filing"]
CATEGORICAL = ["author_dg", "procedure_type"]


def load(min_stuck_years: float) -> pd.DataFrame:
    d = pd.read_csv(DATA / "lifecycle_training_set.csv", parse_dates=["filing_date", "query_date"])
    src = pd.read_csv(ROOT / "eu_lifecycle_balanced.csv", usecols=["celex", "procedure_type"])
    d = d.merge(src, on="celex", how="left").drop(columns=["author"])
    d = d.drop_duplicates(["celex", "query_date"])  # very short lifecycles repeat the same date
    pending_days = (SNAPSHOT - d["filing_date"]).dt.days
    d = d[(d["status"] != "stuck") | (pending_days >= 365.25 * min_stuck_years)].copy()
    d["y"] = (d["status"] == "passed").astype(int)
    d["days_since_filing"] = (d["query_date"] - d["filing_date"]).dt.days  # known on the query date
    return d.reset_index(drop=True)


def split(d: pd.DataFrame, test_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole proposals, stratified by label."""
    k = round(1 / test_share)
    folds = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    tr, te = next(folds.split(d, d["y"], groups=d["celex"]))
    train, test = d.iloc[tr].copy(), d.iloc[te].copy()
    assert not set(train["celex"]) & set(test["celex"]), "a proposal is on both sides"
    return train, test


def preprocessor(numeric: list[str], scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), CATEGORICAL),
    ])


def candidates(numeric: list[str]) -> dict[str, Pipeline]:
    """Small, regularised models: few proposals (~200 in train), so variance is the main risk.
    No class weights: they would distort the probabilities, and calibration is what Brier scores."""
    out = {}
    for c in (0.01, 0.03, 0.1, 0.3, 1.0):
        out[f"logreg C={c}"] = Pipeline([
            ("prep", preprocessor(numeric, scale=True)),
            ("model", LogisticRegression(C=c, max_iter=2000))])
    for leaves, n in ((4, 100), (8, 100), (4, 300)):
        out[f"lgbm leaves={leaves} trees={n}"] = Pipeline([
            ("prep", preprocessor(numeric, scale=False)),
            ("model", LGBMClassifier(n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=30,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                     random_state=SEED, verbose=-1))])
    return out


def grouped_cv(model: Pipeline, train: pd.DataFrame, features: list[str]) -> dict:
    """Out-of-fold predictions on the training set, folds by proposal."""
    oof = np.zeros(len(train))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in folds.split(train, train["y"], groups=train["celex"]):
        m = model.fit(train.iloc[tr][features], train.iloc[tr]["y"])
        oof[va] = m.predict_proba(train.iloc[va][features])[:, 1]
    return {"auc": roc_auc_score(train["y"], oof), "log_loss": log_loss(train["y"], oof),
            "brier": brier_score_loss(train["y"], oof)}


def metrics(y, p) -> dict:
    return {"AUC": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, "Brier": brier_score_loss(y, p),
            "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "Accuracy@0.5": accuracy_score(y, p >= 0.5)}


def bootstrap_auc(test: pd.DataFrame, p: np.ndarray, n: int = 1000) -> tuple[float, float]:
    """95% interval for the test AUC, resampling whole proposals."""
    rng = np.random.default_rng(SEED)
    t = test.assign(p=p)
    groups = [g for _, g in t.groupby("celex")]
    aucs = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if s["y"].nunique() == 2:
            aucs.append(roc_auc_score(s["y"], s["p"]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def select(train: pd.DataFrame, numeric: list[str], features: list[str]) -> tuple[str, pd.DataFrame]:
    """Best candidate by grouped-CV log loss on the training set only."""
    results = {name: grouped_cv(m, train, features) for name, m in candidates(numeric).items()}
    cv = pd.DataFrame(results).T.sort_values("log_loss")
    return cv.index[0], cv


def rounds(d: pd.DataFrame, numeric: list[str], features: list[str], n: int) -> None:
    """Each of n groups of proposals is the test set once; model selection on the rest only."""
    folds = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    rows = []
    for i, (tr, te) in enumerate(folds.split(d, d["y"], groups=d["celex"]), 1):
        train, test = d.iloc[tr], d.iloc[te]
        assert not set(train["celex"]) & set(test["celex"]), "a proposal is on both sides"
        best, _ = select(train, numeric, features)
        model = candidates(numeric)[best].fit(train[features], train["y"])
        p = model.predict_proba(test[features])[:, 1]
        first = (test["query_date"] == test["filing_date"]).to_numpy()
        base = np.full(len(test), train["y"].mean())
        cm = confusion_matrix(test["y"], p >= 0.5, labels=[0, 1])
        r = {"round": i, "model": best, "test proposals": test["celex"].nunique(),
             "test not approved": int((test.drop_duplicates("celex")["y"] == 0).sum())}
        r.update({f"model {k}": v for k, v in metrics(test["y"], p).items()})
        r.update({f"baseline {k}": v for k, v in metrics(test["y"], base).items()})
        r.update({f"filing-date {k}": v for k, v in metrics(test["y"][first], p[first]).items()})
        r.update({"recall not approved": cm[0, 0] / cm[0].sum(), "recall approved": cm[1, 1] / cm[1].sum()})
        rows.append(r)
        print(f"   round {i}: {best:22s} test AUC {r['model AUC']:.3f}  accuracy {r['model Accuracy@0.5']:.3f}")

    res = pd.DataFrame(rows).set_index("round")
    print("\n== per round (test set of each round)")
    cols = ["model", "test proposals", "test not approved", "model AUC", "model Brier", "model LogLoss",
            "model Accuracy@0.5", "baseline Brier", "baseline Accuracy@0.5"]
    print(res[cols].round(3).to_string())

    num = res.drop(columns=["model"])
    summary = pd.DataFrame({"mean": num.mean(), "std": num.std(), "min": num.min(), "max": num.max()})
    print(f"\n== average over {n} rounds")
    print(summary.drop(index=["test proposals", "test not approved"]).round(3).to_string())


def compare(d: pd.DataFrame, numeric: list[str], features: list[str], n: int) -> None:
    """Every candidate on the same n rotating splits (no selection inside rounds), averaged."""
    folds = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
                 .split(d, d["y"], groups=d["celex"]))
    rows = []
    for name, model in candidates(numeric).items():
        for i, (tr, te) in enumerate(folds, 1):
            train, test = d.iloc[tr], d.iloc[te]
            p = model.fit(train[features], train["y"]).predict_proba(test[features])[:, 1]
            first = (test["query_date"] == test["filing_date"]).to_numpy()
            cm = confusion_matrix(test["y"], p >= 0.5, labels=[0, 1])
            r = {"model": name, "round": i, **metrics(test["y"], p),
                 "recall not approved": cm[0, 0] / cm[0].sum(),
                 "filing-date AUC": roc_auc_score(test["y"][first], p[first])}
            rows.append(r)
    for i, (tr, te) in enumerate(folds, 1):
        train, test = d.iloc[tr], d.iloc[te]
        b = np.full(len(te), train["y"].mean())
        rows.append({"model": "baseline (train approval rate)", "round": i, **metrics(test["y"], b),
                     "recall not approved": 0.0, "filing-date AUC": 0.5})

    res = pd.DataFrame(rows)
    mean = res.groupby("model").mean(numeric_only=True).drop(columns="round")
    std = res.groupby("model").std(numeric_only=True)
    table = mean.copy()
    for c in ["AUC", "Brier", "LogLoss"]:
        table[f"{c} std"] = std[c]
    table = table[["AUC", "AUC std", "Brier", "Brier std", "LogLoss", "LogLoss std", "Accuracy@0.5",
                   "recall not approved", "filing-date AUC"]].sort_values("LogLoss")
    print(f"== average over the same {n} splits (sorted by log loss, lower is better)")
    print(table.round(3).to_string())
    print("\n== AUC per round")
    print(res.pivot(index="model", columns="round", values="AUC").loc[table.index].round(3).to_string())
    family = table.drop(index="baseline (train approval rate)")
    family = family.groupby(family.index.str.split().str[0]).apply(lambda g: g.sort_values("LogLoss").iloc[0])
    print("\n== best setting per model family")
    print(family[["AUC", "Brier", "LogLoss", "Accuracy@0.5", "filing-date AUC"]].round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-stuck-years", type=float, default=2.0)
    ap.add_argument("--test-share", type=float, default=0.25)
    ap.add_argument("--with-economy", action="store_true", help="also use the econ_* columns (future-scaled)")
    ap.add_argument("--rounds", type=int, help="evaluate with N rotating splits (each 1/N of proposals is "
                                               "the test set once) instead of one split; saves nothing")
    ap.add_argument("--compare", type=int, help="run every candidate model on the same N rotating splits and "
                                                "average; saves nothing")
    args = ap.parse_args()

    d = load(args.min_stuck_years)
    numeric = NUMERIC + ([c for c in d.columns if c.startswith("econ_")] if args.with_economy else [])
    features = numeric + CATEGORICAL
    if args.compare:
        print(f"== {args.compare} rounds, {d['celex'].nunique()} proposals ({len(d)} rows), split by proposal")
        compare(d, numeric, features, args.compare)
        return
    if args.rounds:
        print(f"== {args.rounds} rounds, {d['celex'].nunique()} proposals ({len(d)} rows), "
              f"split by proposal, model selection inside each round's training part")
        rounds(d, numeric, features, args.rounds)
        return
    train, test = split(d, args.test_share)
    train.to_csv(DATA / "lifecycle_train.csv", index=False)
    test.to_csv(DATA / "lifecycle_test.csv", index=False)

    def describe(name, x):
        p = x.drop_duplicates("celex")
        return (f"{name:5s} {len(p):3d} proposals ({p['y'].sum()} approved, {(1 - p['y']).sum()} not: "
                f"{p['status'].value_counts().to_dict()}), {len(x)} rows, approved share {x['y'].mean():.2f}")
    print("== data (stuck kept only if pending >= %.1f years)" % args.min_stuck_years)
    print(describe("train", train))
    print(describe("test", test))
    print("   proposals in both sets: 0 (checked)\n")

    print("== model selection: 5-fold cross-validation on TRAIN only, folds by proposal")
    best, cv = select(train, numeric, features)
    print(cv.round(3).to_string())
    print(f"   chosen (lowest log loss): {best}\n")

    model = candidates(numeric)[best].fit(train[features], train["y"])
    p_train = model.predict_proba(train[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]

    base_rate = train["y"].mean()
    proc_rate = train.groupby("procedure_type")["y"].mean()
    p_proc = test["procedure_type"].map(proc_rate).fillna(base_rate).to_numpy()
    first = (test["query_date"] == test["filing_date"]).to_numpy()

    rows = {
        "model on train (fit data)": metrics(train["y"], p_train),
        "model on train (CV, unseen folds)": {"AUC": cv.loc[best, "auc"], "Brier": cv.loc[best, "brier"],
                                              "LogLoss": cv.loc[best, "log_loss"], "Accuracy@0.5": np.nan},
        "MODEL on TEST": metrics(test["y"], p_test),
        "baseline: train approval rate": metrics(test["y"], np.full(len(test), base_rate)),
        "baseline: rate per procedure type": metrics(test["y"], p_proc),
        "MODEL on TEST, filing-date rows only": metrics(test["y"][first], p_test[first]),
    }
    print("== evaluation")
    print(pd.DataFrame(rows).T.round(3).to_string())
    lo, hi = bootstrap_auc(test, p_test)
    print(f"   test AUC 95% interval (bootstrap over proposals): {lo:.3f} - {hi:.3f}\n")

    print("== confusion matrix on TEST (threshold 0.5), rows = truth")
    cm = confusion_matrix(test["y"], p_test >= 0.5, labels=[0, 1])
    print(pd.DataFrame(cm, index=["not approved", "approved"], columns=["pred not", "pred approved"]).to_string())

    print("\n== calibration on TEST (predicted vs observed approval rate)")
    bins = pd.cut(p_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    print(pd.DataFrame({"rows": test["y"].groupby(bins, observed=False).size(),
                        "mean predicted": pd.Series(p_test).groupby(bins, observed=False).mean().round(3).values,
                        "observed": test["y"].groupby(bins, observed=False).mean().round(3).values}).to_string())

    names = model.named_steps["prep"].get_feature_names_out()
    est = model.named_steps["model"]
    weights = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
    imp = pd.Series(weights, index=[n.split("__", 1)[1] for n in names])
    print("\n== " + ("logistic coefficients (standardised, + = more likely approved)" if hasattr(est, "coef_")
                     else "LightGBM split importance"))
    print(imp.reindex(imp.abs().sort_values(ascending=False).index).head(15).round(3).to_string())

    out = test[["celex", "filing_date", "query_date", "status"]].assign(y=test["y"], p_approved=p_test.round(4))
    out.to_csv(DATA / "lifecycle_test_predictions.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features, "trained_on": sorted(train["celex"]),
                 "min_stuck_years": args.min_stuck_years}, ROOT / "models" / "lifecycle_model.joblib")
    print("\nwrote data/lifecycle_train.csv, data/lifecycle_test.csv, data/lifecycle_test_predictions.csv, "
          "models/lifecycle_model.joblib")


if __name__ == "__main__":
    main()
