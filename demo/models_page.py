"""Models tab: what the two models are, why we chose them, and their metrics (reports/model_metrics.csv)."""
from __future__ import annotations

import re
import textwrap

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline

METRICS = pipeline.ROOT / "reports" / "model_metrics.csv"
APPROVAL, TIMING = "approval (lifecycle_model)", "timing (timing_model)"
GREEN, RED, CYAN, AMBER, VIOLET, MUTED = "#34f5a4", "#ff4d6d", "#3ad7ff", "#ffb547", "#a78bfa", "#8b93a7"
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color="#cfd6e6"), margin=dict(l=10, r=10, t=40, b=10), height=320,
              legend=dict(orientation="h", y=-0.15))


@st.cache_data(show_spinner=False)
def load_metrics(mtime: float) -> pd.DataFrame:
    return pd.read_csv(METRICS)


@st.cache_resource(show_spinner=False)
def top_features() -> tuple[pd.Series, pd.Series]:
    """Approval: standardised logistic coefficients. Timing: LightGBM gain share."""
    pipe = joblib.load(pipeline.LIFECYCLE_MODEL)["model"]
    est = pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    weights = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
    approval = pd.Series(weights, index=names)
    approval = approval.reindex(approval.abs().sort_values(ascending=False).index).head(8)
    imp = joblib.load(pipeline.train_timing_model.MODEL_PATH)["model"].importance()
    return approval, (imp / imp.sum()).head(8)


def value(m: pd.DataFrame, model: str, predictor: str, split: str, metric: str) -> float:
    s = m[(m["model"] == model) & (m["predictor"] == predictor) & (m["split"] == split) & (m["metric"] == metric)]
    return float(s["value"].iloc[0]) if len(s) else float("nan")


def kpi(label: str, val: str, sub: str, color: str) -> str:
    return (f'<div class="sig" style="--c:{color};min-height:112px"><div class="name">{label}</div>'
            f'<div class="val">{val}</div><div class="nature">{sub}</div></div>')


def compare_chart(m: pd.DataFrame, model: str, metric: str, title: str, fmt: str, better: str) -> go.Figure:
    """Test score of the model and each baseline."""
    t = m[(m["model"] == model) & (m["split"] == "test") & (m["metric"] == metric)]
    t = t[~t["predictor"].str.contains("filing-date")]
    names = ["<br>".join(textwrap.wrap(p, 22)) for p in t["predictor"]]
    colors = [GREEN if p == "model" else MUTED for p in t["predictor"]]
    fig = go.Figure(go.Bar(x=names, y=t["value"], marker=dict(color=colors, line=dict(color=colors, width=2)),
                           text=[fmt.format(v) for v in t["value"]], textposition="outside"))
    fig.update_layout(**LAYOUT, title=f"{title} on the test set ({better})", showlegend=False,
                      xaxis=dict(tickangle=0),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.06)", range=[0, t["value"].max() * 1.25]))
    return fig


def train_test_chart(m: pd.DataFrame, model: str, metrics: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for split, color in (("train", VIOLET), ("test", CYAN)):
        vals = [value(m, model, "model", split, k) for k in metrics]
        fig.add_trace(go.Bar(name=split, x=metrics, y=vals, marker_color=color,
                             text=[f"{v:.2f}" if v < 10 else f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(**LAYOUT, title=title, barmode="group", yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    return fig


def pretty(name: str) -> str:
    """Model column names -> readable labels."""
    name = re.sub(r"^author_dg_infrequent_sklearn$", "DG: other (rare) DGs", name)
    name = re.sub(r"^(author_)?dg_(.+)$", r"DG: \2", name)
    name = re.sub(r"^procedure_type_(.+)$", r"procedure: \1", name)
    name = re.sub(r"^missingindicator_(.+)$", r"\1 missing", name)
    return {"title_ridge": "title text model", "n_cited_works": "acts cited", "dg": "DG",
            "directory_2": "EUR-Lex subject (sub-chapter)", "directory_1": "EUR-Lex subject (chapter)",
            "filing_days_to_election": "days from proposal to EP election",
            "days_to_election": "days to next EP election"}.get(name, name.replace("_", " "))


def feature_chart(s: pd.Series, title: str, signed: bool) -> go.Figure:
    s = s.iloc[::-1].rename(pretty)
    colors = [(GREEN if v > 0 else RED) if signed else CYAN for v in s]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors))
    fig.update_layout(**LAYOUT, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    return fig


def split_sizes(m: pd.DataFrame, model: str) -> dict:
    s = m[(m["model"] == model) & (m["predictor"] == "model")].groupby("split")[["rows", "proposals"]].first()
    return s.to_dict("index")


def render() -> None:
    if not METRICS.exists():
        st.warning("reports/model_metrics.csv is missing: run `.venv/bin/python evaluate_models.py`.")
        return
    m = load_metrics(METRICS.stat().st_mtime)
    approval_w, timing_w = top_features()
    cfg = {"displayModeBar": False}

    st.subheader("Our two models")
    st.markdown("Both answer a question **as of a query date**: every input is cut at that date, so a model "
                "never sees a vote, milestone or adoption that happened later. Each is scored against simple "
                "baselines computed on its own training data. Metrics come from `reports/model_metrics.csv` "
                "(`evaluate_models.py`) and describe the models exactly as saved and used on the Predict tab.")

    # ------------------------------------------------------------------ approval
    st.markdown("### 1 · Will it become law? · `lifecycle_model`")
    sizes = split_sizes(m, APPROVAL)
    auc, brier = value(m, APPROVAL, "model", "test", "AUC"), value(m, APPROVAL, "model", "test", "Brier")
    base_brier = value(m, APPROVAL, "baseline: train approval rate", "test", "Brier")
    acc, base_acc = (value(m, APPROVAL, "model", "test", "Accuracy@0.5"),
                     value(m, APPROVAL, "baseline: train approval rate", "test", "Accuracy@0.5"))
    c = st.columns(4)
    c[0].markdown(kpi("Test AUC", f"{auc:.2f}", "ranking passed vs. failed (0.5 = coin flip)",
                      GREEN if auc > 0.85 else CYAN if auc > 0.7 else AMBER), unsafe_allow_html=True)
    c[1].markdown(kpi("Test Brier", f"{brier:.3f}", f"baseline {base_brier:.3f} · lower is better",
                      GREEN if brier < base_brier else RED), unsafe_allow_html=True)
    c[2].markdown(kpi("Test accuracy", f"{acc:.0%}", f"baseline {base_acc:.0%} (always 'passes')", CYAN),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("Test set", f"{sizes['test']['proposals']}", f"proposals never seen in training "
                      f"({sizes['test']['rows']} query dates)", VIOLET), unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            "**What it is.** A regularised **logistic regression** that outputs the probability that a "
            "Commission proposal is adopted, given what is known on the query date: the responsible DG, the "
            "procedure type (OLP, CNS…), how similar earlier proposals fared, Parliament's stance on the "
            "sector, public consultations, EP votes so far and the days already spent in the procedure.\n\n"
            "**Why this model.** We trained it on ~260 labelled proposals (4 query dates each), so variance "
            "is the main risk. We compared logistic regressions and small LightGBM models with 5-fold "
            "cross-validation grouped by proposal; the logistic regression had the lowest log loss. It gives "
            "well-calibrated probabilities (what the Brier score rewards), and its coefficients are readable.\n\n"
            "**How we tested it.** Whole proposals are held out (stratified by outcome), so none of a test "
            "proposal's query dates is seen in training. Withdrawn proposals and those stuck for 2+ years "
            "count as not adopted; younger pending ones are left out, never labelled as failures.")
    with right:
        st.plotly_chart(compare_chart(m, APPROVAL, "Brier", "Brier score", "{:.3f}", "lower is better"),
                        width="stretch", config=cfg)
    a, b = st.columns(2)
    a.plotly_chart(train_test_chart(m, APPROVAL, ["AUC", "Brier", "LogLoss", "Accuracy@0.5"], "Train vs. test"),
                   width="stretch", config=cfg)
    b.plotly_chart(feature_chart(approval_w, "Strongest coefficients (green = more likely to pass)", True),
                   width="stretch", config=cfg)
    filing_auc = value(m, APPROVAL, "model, filing-date rows only", "test", "AUC")
    st.info(f"**Limitation.** The model leans on what happens after the proposal (votes, time in procedure). "
            f"Scored only on the proposal date itself, its test AUC is {filing_auc:.2f}, and the gap between "
            f"train AUC ({value(m, APPROVAL, 'model', 'train', 'AUC'):.2f}) and test AUC ({auc:.2f}) shows "
            "some overfitting on a small dataset.")

    # ------------------------------------------------------------------ timing
    st.markdown("### 2 · When will it become law? · `timing_model`")
    sizes = split_sizes(m, TIMING)
    mae = value(m, TIMING, "model", "test", "MAE days")
    base = m[(m["model"] == TIMING) & (m["split"] == "test") & (m["metric"] == "MAE days")
             & m["predictor"].str.startswith("baseline")]["value"].min()
    c = st.columns(4)
    c[0].markdown(kpi("Test mean error", f"{mae:.0f} days", "good < 90 · excellent < 45",
                      GREEN if mae < 45 else CYAN if mae < 90 else AMBER), unsafe_allow_html=True)
    c[1].markdown(kpi("Best baseline", f"{base:.0f} days", f"model is {base - mae:.0f} days closer on average",
                      GREEN if mae < base else RED), unsafe_allow_html=True)
    c[2].markdown(kpi("Median error", f"{value(m, TIMING, 'model', 'test', 'median AE'):.0f} days",
                      f"{value(m, TIMING, 'model', 'test', 'within 90d'):.0%} of predictions within 90 days", CYAN),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("Test set", f"{sizes['test']['proposals']}", f"adopted proposals filed after training "
                      f"({sizes['test']['rows']} query dates)", VIOLET), unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            "**What it is.** A **LightGBM** gradient-boosted model that predicts the days still to go until "
            "adoption from the query date; the predicted date is the query date plus that. Inputs: proposal "
            "metadata (procedure type, DG, legal bases, cited acts, EuroVoc terms, title keywords), the "
            "European Parliament election calendar as scheduled, and the EP / Council milestones (committee "
            "vote, trilogues, agreement…) reached by the query date. A small text model on the title "
            "(TF-IDF + ridge) is fed in as one more input.\n\n"
            "**Why this model.** We are scored on the mean error in days, so it is trained with an absolute "
            "error objective on log-days, i.e. it predicts the *median* duration, robust to the few files "
            "that take many years. Trees handle the mix of categories, counts and missing EP data without "
            "manual tuning, and it trains on ~1,700 adopted proposals of every procedure type.\n\n"
            "**How we tested it.** By time: trained on proposals filed up to the end of 2023, tested on "
            "adopted proposals filed after that, which it never saw.")
    with right:
        st.plotly_chart(compare_chart(m, TIMING, "MAE days", "Mean error (days)", "{:.0f}", "lower is better"),
                        width="stretch", config=cfg)
    a, b = st.columns(2)
    a.plotly_chart(train_test_chart(m, TIMING, ["MAE days", "median AE"], "Train vs. test error (days)"),
                   width="stretch", config=cfg)
    b.plotly_chart(feature_chart(timing_w, "Most used inputs (share of LightGBM gain)", False),
                   width="stretch", config=cfg)
    st.info("**Limitation.** The test set is biased towards fast files: proposals filed since 2024 that are "
            "already law are the quick ones, while the slow ones are still pending. Expect larger errors on "
            "older or harder files; our earlier check (train to 2021, test on 2022–2023) gave a mean error "
            "of about 95 days.")

    with st.expander("All metrics (reports/model_metrics.csv)"):
        st.dataframe(m, hide_index=True, width="stretch")
