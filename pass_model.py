"""
Pass/fail model: probability that an EU Commission proposal becomes law (p_law),
using only what was known on its filing date.

    python build_training_set.py                  # first: data/training_set.parquet
    python pass_model.py                          # train on 2018-2021, test on 2022, save the model
    python pass_model.py --stale-pending-years 0  # never count pending proposals as failed
    python pass_model.py --test-year 2023

Input:  data/training_set.parquet (build_training_set.py): meta__* columns, every <source>__* column
        it contains (added automatically), and the post_* outcome columns used for the labels only.
Output: models/pass_model.joblib             fitted model + feature builder, for predict.py
        data/pass_model_test_predictions.csv  test-year predictions next to the truth

Labels (from the proposal's dossier, i.e. its status today):
    adopted = 1, withdrawn = 0,
    pending for more than --stale-pending-years at the data snapshot = 0 (stalled, de facto failed),
    more recent pending = unlabelled, left out.
Cellar has no dossier outcome before 2018, so training starts in 2018.

Features: filing-day metadata (FeatureBuilder) plus whatever source features the training set has.
post_* columns and labels are refused as inputs (leakage guard in FeatureBuilder).
Subject-matter and EuroVoc codes are assigned when the proposal is catalogued and describe its
content, so they are used (Claude.md, leakage map note 1).
"""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"

NOT_FEATURES = {"initiative_id", "filing_date", "title", "is_law", "days_to_law", "label_status", "y"}

# Commission mandates and European Parliament elections, as scheduled at the time (known in advance).
COMMISSION_STARTS = [date(2014, 11, 1), date(2019, 12, 1), date(2024, 12, 1)]
EP_ELECTIONS = [date(2014, 5, 25), date(2019, 5, 26), date(2024, 6, 9), date(2029, 6, 7)]

TITLE_FLAGS = {
    "title_position": r"position to be (?:taken|adopted)",
    "title_conclusion": r"\bconclusion\b",
    "title_signing": r"\bsigning\b",
    "title_amending": r"\bamending\b",
    "title_repealing": r"\brepealing\b",
    "title_amended_proposal": r"^amended proposal",
    "title_mobilisation": r"mobilisation of the european globalisation",
    "title_agreement": r"\bagreement\b",
    "title_extension": r"\bextension\b|\bprolong",
}


def _tokens(value: str) -> list[str]:
    return [t.strip() for t in value.split("|") if t.strip()]


# ------------------------------------------------------------------------ features

class FeatureBuilder:
    """Filing-day features from a training-set row. fit() learns vocabularies and categories on the
    training rows only; transform() is shared by training and prediction (predict.py)."""

    CATEGORICAL = ["resource_type", "procedure_type", "dg", "directory_chapter", "commission"]

    def __init__(self, top_subjects: int = 30, top_eurovoc: int = 40, min_count: int = 10):
        self.top_subjects, self.top_eurovoc, self.min_count = top_subjects, top_eurovoc, min_count

    @staticmethod
    def source_columns(df: pd.DataFrame) -> list[str]:
        """Features from sources.py: <source>__<name>. Leakage guard: post_* and labels never pass."""
        cols = [c for c in df.columns if "__" in c and not c.startswith("meta__")]
        bad = [c for c in cols if c.startswith("post_") or c in NOT_FEATURES]
        assert not bad, f"post-filing or label columns among features: {bad}"
        return cols

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        def vocab(col: str, k: int) -> list[str]:
            counts = df[col].map(_tokens).explode().value_counts()
            return counts[counts >= self.min_count].head(k).index.tolist()
        self.subjects_ = vocab("meta__subjects", self.top_subjects)
        self.eurovoc_ = vocab("meta__eurovoc", self.top_eurovoc)
        self.sources_ = self.source_columns(df)
        self.source_categorical_ = [c for c in self.sources_  # text columns (pandas 3 uses a string dtype)
                                    if not pd.api.types.is_numeric_dtype(df[c])
                                    and not pd.api.types.is_bool_dtype(df[c])
                                    and not df[c].dropna().map(lambda v: isinstance(v, (bool, int, float))).all()]
        base = self._base(df)
        self.categories_ = {}
        for c in self.CATEGORICAL + self.source_categorical_:
            col = base[c] if c in base else df[c].astype("string").fillna("none")
            counts = col.value_counts()
            # Rare categories -> "other", so the model never sees a category with 1-2 examples.
            self.categories_[c] = counts[counts >= self.min_count].index.tolist()
        return self

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["resource_type"] = df["meta__resource_type"]
        f["procedure_type"] = df["meta__procedure_type"].map(lambda v: (_tokens(v) or ["none"])[0])
        f["dg"] = df["meta__dg"].map(lambda v: (_tokens(v) or ["none"])[0])
        f["directory_chapter"] = df["meta__directory_code"].map(lambda v: (_tokens(v) or ["00"])[0][:2])
        f["commission"] = df["filing_date"].map(
            lambda d: str(max(s for s in COMMISSION_STARTS if s <= d.date()).year))
        return f

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        f = self._base(df)
        for c in self.CATEGORICAL:
            f[c] = pd.Categorical(f[c].where(f[c].isin(self.categories_[c]), "other"),
                                  categories=self.categories_[c] + ["other"])
        d = df["filing_date"].dt.date
        f["days_into_commission"] = d.map(lambda x: (x - max(s for s in COMMISSION_STARTS if s <= x)).days)
        f["days_to_next_election"] = d.map(lambda x: (min(e for e in EP_ELECTIONS if e > x) - x).days)
        f["filing_month"] = df["filing_date"].dt.month
        f["amends_existing"] = (df["meta__proposes_to_amend"] != "").astype(int)
        f["eea_relevant"] = (df["meta__eea"] == "1").astype(int)
        f["n_legal_bases"] = df["meta__legal_bases"].map(lambda v: len(_tokens(v)))
        f["n_cited_works"] = df["meta__cited_works"].map(lambda v: len(_tokens(v)))
        f["n_subjects"] = df["meta__subjects"].map(lambda v: len(_tokens(v)))
        f["n_eurovoc"] = df["meta__eurovoc"].map(lambda v: len(_tokens(v)))
        title = df["title"].str.lower()
        f["title_words"] = title.str.split().str.len()
        for name, pattern in TITLE_FLAGS.items():
            f[name] = title.str.contains(pattern, regex=True).astype(int)
        subjects = df["meta__subjects"].map(lambda v: set(_tokens(v)))
        for s in self.subjects_:
            f[f"subject_{s}"] = subjects.map(lambda x: int(s in x))
        eurovoc = df["meta__eurovoc"].map(lambda v: set(_tokens(v)))
        for e in self.eurovoc_:
            f[f"eurovoc_{e.split(':')[-1]}"] = eurovoc.map(lambda x: int(e in x))
        # Source features: numbers as they are (NaN = source had nothing), text as categories.
        src = {}
        for c in self.sources_:
            col = df[c] if c in df else pd.Series(np.nan, index=df.index)
            if c in self.source_categorical_:
                col = col.astype("string").fillna("none")
                src[c] = pd.Categorical(col.where(col.isin(self.categories_[c]), "other"),
                                        categories=self.categories_[c] + ["other"])
            else:
                src[c] = pd.to_numeric(col.map(lambda v: float(v) if isinstance(v, bool) else v), errors="coerce")
        return pd.concat([f, pd.DataFrame(src, index=df.index)], axis=1)


# -------------------------------------------------------------------------- models

def baseline_rates(f: pd.DataFrame, y: pd.Series) -> dict:
    """Pass rate per procedure type on the training rows (the baseline to beat)."""
    rates = y.groupby(f["procedure_type"].astype(str)).mean().to_dict()
    rates["_all"] = float(y.mean())
    return rates


def baseline_predict(rates: dict, f: pd.DataFrame) -> np.ndarray:
    return f["procedure_type"].astype(str).map(rates).fillna(rates["_all"]).to_numpy()


def logistic_model(f: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    cat = [c for c in f.columns if isinstance(f[c].dtype, pd.CategoricalDtype)]
    num = [c for c in f.columns if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                   StandardScaler()), num)])
    return make_pipeline(pre, LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000,
                                                 random_state=SEED))


def lgbm_model():
    """LightGBM, then sigmoid calibration (Brier rewards calibrated probabilities).
    Sigmoid rather than isotonic: there are only ~100 negatives, isotonic would overfit."""
    base = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          class_weight="balanced", random_state=SEED, deterministic=True,
                          force_row_wise=True, verbose=-1)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def scores(y: pd.Series, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"AUC": roc_auc_score(y, p) if y.nunique() > 1 else float("nan"),
            "Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p, labels=[0, 1]),
            "Accuracy@0.5": accuracy_score(y, p >= 0.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DATA / "training_set.parquet"))
    ap.add_argument("--train-from", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2022, help="train on the years before, test on this one")
    ap.add_argument("--stale-pending-years", type=float, default=3.0,
                    help="pending for longer than this at the snapshot counts as failed; 0 = never")
    args = ap.parse_args()

    from build_training_set import add_labels
    p = pd.read_parquet(args.input)
    p = p[p["filing_date"] >= f"{args.train_from}-01-01"].reset_index(drop=True)
    snapshot = p["filing_date"].max()  # the data is as of its newest proposal
    p["y"] = add_labels(p, args.stale_pending_years, snapshot)["is_law"]
    srcs = sorted({c.split("__")[0] for c in FeatureBuilder.source_columns(p)})
    print(f"source features in the training set: {srcs or 'none'}")
    year = p["filing_date"].dt.year
    train = p[(year < args.test_year) & p["y"].notna()]
    test = p[(year == args.test_year) & p["y"].notna()]
    print(f"snapshot {snapshot.date()}, stale pending > {args.stale_pending_years} years = failed")
    print(f"train {args.train_from}-{args.test_year - 1}: {len(train)} proposals, "
          f"{int((train.y == 0).sum())} failed ({(train.y == 0).mean():.1%})")
    print(f"test  {args.test_year}: {len(test)} proposals, {int((test.y == 0).sum())} failed "
          f"({(test.y == 0).mean():.1%})")
    if (test.y == 0).sum() < 10:
        print("   warning: fewer than 10 failures in the test year, AUC is very noisy")

    fb = FeatureBuilder().fit(train)
    Xtr, Xte = fb.transform(train), fb.transform(test)
    ytr, yte = train["y"].astype(int), test["y"].astype(int)
    print(f"{Xtr.shape[1]} features")

    rates = baseline_rates(Xtr, ytr)
    results = {"baseline (pass rate per procedure)": baseline_predict(rates, Xte),
               "always pass (train rate)": np.full(len(Xte), ytr.mean())}
    logit = logistic_model(Xtr).fit(Xtr, ytr)
    results["logistic regression"] = logit.predict_proba(Xte)[:, 1]
    lgbm = lgbm_model().fit(Xtr, ytr)
    results["LightGBM (calibrated)"] = lgbm.predict_proba(Xte)[:, 1]

    table = pd.DataFrame({name: scores(yte, pr) for name, pr in results.items()}).T
    print(f"\nTest year {args.test_year} (AUC higher is better; Brier and LogLoss lower is better)")
    print(table.round(4).to_string())

    # Which features LightGBM leans on (average over the calibration folds).
    imp = np.mean([c.estimator.booster_.feature_importance("gain") for c in lgbm.calibrated_classifiers_], axis=0)
    top = pd.Series(imp, index=Xtr.columns).sort_values(ascending=False).head(15)
    print("\nTop features (LightGBM gain):")
    print((top / top.sum()).round(3).to_string())

    out = test[["initiative_id", "filing_date", "title", "y"]].copy()
    for name, pr in results.items():
        out[name] = pr
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "pass_model_test_predictions.csv", index=False)

    # Final model for predict.py: refit on every labelled year, test year included.
    full = p[p["y"].notna() & (year <= args.test_year)]
    fb_full = FeatureBuilder().fit(full)
    final = lgbm_model().fit(fb_full.transform(full), full["y"].astype(int))
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": fb_full, "trained_on": f"{args.train_from}-{args.test_year}",
                 "stale_pending_years": args.stale_pending_years, "test_scores": table.to_dict()},
                MODELS / "pass_model.joblib")
    print(f"\nwrote data/pass_model_test_predictions.csv and models/pass_model.joblib "
          f"(refit on {args.train_from}-{args.test_year}, {len(full)} proposals)")


if __name__ == "__main__":
    main()
