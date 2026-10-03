"""
Bill-to-Law demo: CELEX + query date -> pipeline -> both models -> LLM reasons -> dashboard.

    .venv/bin/streamlit run demo/app.py

LLM settings: see demo/llm.py (GROK_API_KEY in .env, or LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL).
"""
from __future__ import annotations

import html
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import architecture  # noqa: E402
import llm  # noqa: E402
import models_page  # noqa: E402
import pipeline  # noqa: E402

st.set_page_config(page_title="Bill-to-Law Predictor", page_icon="⚖️", layout="wide")

GREEN, RED, CYAN, AMBER, VIOLET, BLUE = "#34f5a4", "#ff4d6d", "#3ad7ff", "#ffb547", "#a78bfa", "#4d8dff"
MUTED = "#8b93a7"

st.markdown(f"""<style>
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #1b1f3b 0%, transparent 60%),
          radial-gradient(900px 500px at 110% 10%, #0f2a33 0%, transparent 55%), #07080d; }}
.block-container {{ padding-top: 3.5rem; max-width: 1250px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.brand {{ font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1.2;
          background: linear-gradient(90deg, {CYAN}, {VIOLET} 55%, {GREEN});
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.tag {{ color: {MUTED}; margin-top: -4px; }}
.card {{ background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
         border-radius: 18px; padding: 18px 22px; backdrop-filter: blur(6px); }}
.law-title {{ font-size: 1.05rem; font-weight: 600; line-height: 1.45; }}
.chips span {{ display: inline-block; margin: 8px 6px 0 0; padding: 3px 11px; border-radius: 999px;
               font-size: .78rem; color: #cfd6e6; background: rgba(255,255,255,0.06);
               border: 1px solid rgba(255,255,255,0.1); }}
.hero {{ display: flex; align-items: center; gap: 26px; min-height: 190px;
         border-color: var(--c); box-shadow: 0 0 40px -14px var(--c), inset 0 0 40px -30px var(--c); }}
.ring {{ width: 150px; height: 150px; border-radius: 50%; flex: none; display: grid; place-items: center;
         background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0);
         filter: drop-shadow(0 0 14px var(--c)); }}
.ring > div {{ width: 118px; height: 118px; border-radius: 50%; background: #0b0d16; display: grid;
               place-items: center; font-size: 2.1rem; font-weight: 800; color: var(--c); }}
.hero-label {{ color: {MUTED}; text-transform: uppercase; font-size: .75rem; letter-spacing: .12em; }}
.hero-big {{ font-size: 3rem; font-weight: 800; color: var(--c); text-shadow: 0 0 22px var(--c); line-height: 1.1; }}
.hero-sub {{ color: #cfd6e6; font-size: .92rem; margin-top: 6px; }}
.sig {{ --c: {CYAN}; border-radius: 16px; padding: 14px 16px; margin-bottom: 14px; min-height: 128px;
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid var(--c); box-shadow: 0 0 16px -6px var(--c); }}
.sig.rising  {{ animation: pulse var(--speed) ease-in-out infinite; }}
.sig.falling {{ animation: pulse var(--speed) ease-in-out infinite; }}
.sig.volatile {{ animation: flicker 1.6s steps(1) infinite; }}
.sig .name {{ color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; }}
.sig .val {{ font-size: 1.75rem; font-weight: 700; color: var(--c); text-shadow: 0 0 14px var(--c); }}
.sig .nature {{ font-size: .8rem; color: #e6e9f2; }}
.sig .bar {{ height: 4px; border-radius: 4px; margin-top: 10px; background: rgba(255,255,255,0.08); }}
.sig .bar > div {{ height: 100%; border-radius: 4px; background: var(--c); box-shadow: 0 0 10px var(--c); }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 8px -6px var(--c); }} 50% {{ box-shadow: 0 0 30px 0 var(--c); }} }}
@keyframes flicker {{ 0% {{ box-shadow: 0 0 24px -2px var(--c); }} 30% {{ box-shadow: 0 0 6px -6px var(--c); }}
                      45% {{ box-shadow: 0 0 30px 0 var(--c); }} 70% {{ box-shadow: 0 0 12px -4px var(--c); }} }}
.reason {{ border-left: 3px solid var(--c); padding: 10px 14px; margin: 8px 0; border-radius: 0 12px 12px 0;
           background: linear-gradient(90deg, rgba(255,255,255,0.06), transparent);
           box-shadow: -8px 0 18px -12px var(--c); color: #e6e9f2; font-size: .93rem; }}
.reason-head {{ font-weight: 700; color: var(--c); text-shadow: 0 0 12px var(--c); margin-top: 6px; }}
.spectrum {{ position: relative; height: 10px; border-radius: 6px; margin: 30px 4px 34px;
             background: linear-gradient(90deg, {BLUE}, #6b7280 50%, {RED}); opacity: .9; }}
.dot {{ position: absolute; top: -7px; width: 24px; height: 24px; margin-left: -12px; border-radius: 50%;
        border: 3px solid #0b0d16; }}
.dot span {{ position: absolute; top: -24px; left: 50%; transform: translateX(-50%); white-space: nowrap;
             font-size: .75rem; color: #e6e9f2; }}
.ends {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .75rem; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data

@st.cache_resource(show_spinner="Loading the embedding model and the proposal corpus (once per server)…")
def warm_up() -> bool:
    """Retried: the EUR-Lex endpoint often answers 502 for a few seconds. A failure is not cached."""
    for attempt in range(3):
        try:
            pipeline.warm_up()
            return True
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


@st.cache_data(show_spinner=False)
def predict(celex: str, query_date: date) -> dict:
    return pipeline.run(celex, query_date)


@st.cache_data(show_spinner=False)
def reasons(payload: dict, query_date: str, cfg: dict) -> dict:
    return llm.explain(payload, query_date, cfg)


def llm_payload(r: dict) -> dict:
    f = r["features"]
    return {
        "proposal": {"celex": r["celex"], "title": r["title"], "proposal_date": r["filing_date"],
                     "procedure": r["procedure_ref"], "procedure_type": f.get("procedure_type"),
                     "responsible_dg": r["dg"], "category": r["category"]},
        "query_date": r["query_date"],
        "ep_council_milestones_so_far": [{"date": m["date"], "event": m["event"]} for m in r["milestones"]],
        "model_inputs": {k: f.get(k) for k in [*r["model_features"]] + [c for c in f if c.startswith("econ_")]},
        "prediction": {"probability_adopted": round(r["p_law"], 3),
                       "predicted_days_from_proposal_to_law": r["days_to_law"],
                       "predicted_adoption_date": r["law_date"]},
    }


def secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:  # no secrets.toml
        return {}


# --------------------------------------------------------------------------- widgets

def glow_color(value: float | None, good: float, bad: float, higher_is_better: bool = True) -> str:
    if value is None:
        return MUTED
    if not higher_is_better:
        value, good, bad = -value, -good, -bad
    return GREEN if value >= good else RED if value <= bad else CYAN


def signal_card(name: str, value: str, nature: str, color: str, fill: float | None = None,
                motion: str = "", speed: float = 2.4) -> str:
    bar = f'<div class="bar"><div style="width:{max(0, min(1, fill)) * 100:.0f}%"></div></div>' if fill is not None else ""
    return (f'<div class="sig {motion}" style="--c:{color};--speed:{speed:.1f}s">'
            f'<div class="name">{name}</div><div class="val">{value}</div>'
            f'<div class="nature">{nature}</div>{bar}</div>')


def num(v, fmt="{:.2f}", none="n/a"):
    return none if v is None or v != v else fmt.format(v)


def signals(f: dict, milestones: list[dict], query_date: date) -> list[str]:
    cards = []
    if milestones:
        idle = (query_date - milestones[-1]["date"]).days
        recent = sum((query_date - m["date"]).days <= 180 for m in milestones)
        if idle <= 90:
            cards.append(signal_card("Procedure activity", f"⚡ {recent} in 6 mo",
                                     f"last step {idle} days ago: moving", GREEN, min(1, recent / 6), "rising", 1.2))
        elif idle <= 365:
            cards.append(signal_card("Procedure activity", f"{recent} in 6 mo", f"last step {idle} days ago: slowing",
                                     AMBER, min(1, recent / 6), "volatile"))
        else:
            cards.append(signal_card("Procedure activity", "❄ stalled", f"no step for {idle} days", RED, 0.05,
                                     "falling", 3.2))
    else:
        cards.append(signal_card("Procedure activity", "—", "no EP / Council milestones before the query date",
                                 CYAN, 0))
    inc, var, n_votes = f.get("in_favor_increase"), f.get("in_favor_variability"), f.get("n_votings") or 0
    if n_votes and inc is not None and abs(inc) >= 0.02:
        rising = inc > 0
        cards.append(signal_card(
            "Vote momentum", f"{'▲' if rising else '▼'} {inc * 100:+.1f} pp",
            "MEPs in favour " + ("climbing between votes" if rising else "eroding between votes"),
            GREEN if rising else RED, 0.5 + inc / 2, "rising" if rising else "falling",
            speed=max(0.8, 2.6 - 4 * abs(inc))))
    else:
        cards.append(signal_card("Vote momentum", "— flat", "no shift in support yet" if n_votes else
                                 "no plenary votes before the query date", CYAN, 0.5))
    if var is not None and var > 0.05:
        cards.append(signal_card("Vote volatility", num(var), "support swings between votes: unpredictable",
                                 AMBER, var, "volatile"))
    else:
        cards.append(signal_card("Vote volatility", num(var), "steady voting pattern", CYAN, var or 0))
    cards.append(signal_card("EP votes so far", str(int(n_votes)), "plenary votes up to the query date", VIOLET,
                             min(1, n_votes / 20)))
    s = f.get("similar_laws_approval")
    cards.append(signal_card("Similar laws track record", num(s, "{:.0%}"),
                             "share of the closest earlier proposals adopted before the query date",
                             glow_color(s, 0.6, 0.4), s, "rising" if s is not None and s >= 0.6 else ""))
    a = f.get("sector_acceptance")
    cards.append(signal_card("Sector acceptance", num(a, "{:.0%}"), "Parliament's openness to this policy sector",
                             glow_color(a, 0.75, 0.5), a))
    d = f.get("law_disruptive_acceptable")
    cards.append(signal_card("Familiarity", num(d), "similarity to earlier proposals (low = disruptive)",
                             glow_color(d, 0.65, 0.5), d))
    c = f.get("n_consultations")
    cards.append(signal_card("Public consultations", num(c, "{:.0f}"), "Have Your Say rounds before the query date",
                             VIOLET, None if c is None else min(1, c / 6)))
    days = f.get("days_since_filing") or 0
    cards.append(signal_card("Days in procedure", f"{days:,}", "since the Commission adopted the proposal",
                             AMBER if days > 730 else CYAN, min(1, days / 1460), "falling" if days > 730 else ""))
    return cards


def spectrum(f: dict) -> str:
    """Law (1 = left) and Parliament (-1 left .. +1 right) on one left-right axis."""
    law = f.get("law_left_right")
    parl = f.get("parliament_left_right")
    dots = ""
    if law is not None:
        dots += (f'<div class="dot" style="left:{(1 - law) * 100:.0f}%;background:{CYAN};'
                 f'box-shadow:0 0 18px {CYAN}"><span style="top:26px">This law</span></div>')
    if parl is not None:
        dots += (f'<div class="dot" style="left:{(parl + 1) * 50:.0f}%;background:{VIOLET};'
                 f'box-shadow:0 0 18px {VIOLET}"><span>Parliament</span></div>')
    return (f'<div class="card"><div class="hero-label">Political alignment</div>'
            f'<div class="spectrum">{dots}</div><div class="ends"><span>◀ left</span><span>right ▶</span></div></div>')


def economy_chart(f: dict) -> go.Figure | None:
    econ = {k.removeprefix("econ_").replace("_", " ").title().replace("Euro Stoxx 50", "EURO STOXX 50"): v
            for k, v in f.items() if k.startswith("econ_") and v is not None}
    if not econ:
        return None
    names, vals = list(econ), list(econ.values())
    colors = [GREEN if v > 0.35 else RED if v < -0.35 else CYAN for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors, line=dict(color=colors, width=2)),
        text=[f"{v:+.1f}" for v in vals], textposition="outside", textfont=dict(color="#e6e9f2"),
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.35)", width=1))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=290,
        margin=dict(l=10, r=30, t=10, b=30), bargap=0.45, font=dict(color="#cfd6e6"),
        xaxis=dict(range=[-5.8, 5.8], tickvals=[-5, -2.5, 0, 2.5, 5], gridcolor="rgba(255,255,255,0.06)",
                   title="worse ◀  13-week momentum score  ▶ better"),
        yaxis=dict(autorange="reversed"))
    return fig


def timeline(r: dict) -> go.Figure:
    points = [("Proposal", r["filing_date"], VIOLET), ("Query date", r["query_date"], CYAN),
              ("Predicted law", r["law_date"], BLUE)]
    fig = go.Figure()
    if r["milestones"]:
        fig.add_trace(go.Scatter(
            x=[m["date"] for m in r["milestones"]], y=[0] * len(r["milestones"]), mode="markers",
            marker=dict(size=8, color=AMBER, symbol="diamond", line=dict(color="#07080d", width=1)),
            customdata=[m["event"] for m in r["milestones"]],
            hovertemplate="%{x|%d %b %Y}<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p[1] for p in points], y=[0] * 3, mode="lines",
                             line=dict(color="rgba(255,255,255,0.25)", width=3), hoverinfo="skip"))
    for size, alpha in ((34, 0.15), (22, 0.3)):  # halo
        fig.add_trace(go.Scatter(x=[p[1] for p in points], y=[0] * 3, mode="markers", hoverinfo="skip",
                                 marker=dict(size=size, color=[p[2] for p in points], opacity=alpha)))
    fig.add_trace(go.Scatter(
        x=[p[1] for p in points], y=[0] * 3, mode="markers+text", marker=dict(size=13, color=[p[2] for p in points]),
        text=[f"<b>{p[0]}</b><br>{p[1]:%d %b %Y}" for p in points], textposition="top center",
        textfont=dict(color="#e6e9f2", size=12), hovertemplate="%{x|%d %b %Y}<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=150, showlegend=False, margin=dict(l=40, r=40, t=10, b=10),
                      yaxis=dict(visible=False, range=[-0.6, 1.4]), xaxis=dict(showgrid=False, visible=False))
    return fig


def hero_probability(p: float) -> str:
    c = GREEN if p >= 0.65 else RED if p < 0.4 else AMBER
    verdict = "likely to become law" if p >= 0.65 else "unlikely to become law" if p < 0.4 else "toss-up"
    return (f'<div class="card hero" style="--c:{c};--p:{p * 100:.0f}"><div class="ring"><div>{p:.0%}</div></div>'
            f'<div><div class="hero-label">Probability of adoption</div>'
            f'<div class="hero-big" style="font-size:1.6rem">{verdict}</div>'
            f'<div class="hero-sub">approval model · features as known on the query date</div></div></div>')


def hero_days(r: dict) -> str:
    left = r["days_left"]
    sub = (f"about {left / 30.44:.0f} months after the query date" if left >= 0
           else "predicted date already passed: running late")
    return (f'<div class="card hero" style="--c:{BLUE}"><div>'
            f'<div class="hero-label">Predicted date of law</div>'
            f'<div class="hero-big">{r["law_date"]:%d %B %Y}</div>'
            f'<div class="hero-sub">{sub}</div>'
            f'<div class="hero-sub" style="color:{MUTED}">timing model · features as known on the query date</div></div></div>')


def reason_block(title: str, items: list[str], color: str) -> str:
    body = "".join(f'<div class="reason">{html.escape(s)}</div>' for s in items)
    return f'<div style="--c:{color}"><div class="reason-head">{title}</div>{body}</div>'


# --------------------------------------------------------------------------- pages

def predict_page() -> None:
    with st.form("query"):
        c1, c2, c3 = st.columns([3, 2, 1.2], vertical_alignment="bottom")
        celex = c1.text_input("CELEX of the Commission proposal", value="52021PC0206",
                              help="Sector 5 preparatory act, e.g. 52021PC0206 (AI Act)")
        qd = c2.date_input("Query date (nothing after it is used)", value=date(2023, 6, 1),
                           min_value=date(2018, 1, 1), max_value=date.today())
        go_ = c3.form_submit_button("Predict ⚡", width="stretch", type="primary")
    if not go_ and "last" not in st.session_state:
        st.info("Enter a proposal CELEX and a query date. The first run loads the embedding model and "
                "queries EUR-Lex, the EP and Have Your Say: it can take a few minutes; repeats are cached.")
        return
    if go_:
        st.session_state["last"] = (celex.strip().upper(), qd)
    celex, qd = st.session_state["last"]

    try:
        warm_up()
    except Exception as e:
        st.error(f"EUR-Lex is not answering right now ({e.__class__.__name__}). Press Predict again in a minute.")
        return
    with st.status(f"Running the pipeline for {celex} as of {qd:%d %b %Y}…", expanded=False) as status:
        try:
            r = predict(celex, qd)
        except pipeline.PipelineError as e:
            status.update(label=str(e), state="error")
            return
        except Exception as e:  # network failure, stale model file...
            status.update(label=f"Pipeline failed: {e.__class__.__name__}: {str(e)[:200]}", state="error")
            return
        status.update(label="Predictions ready", state="complete")
    f = r["features"]

    chips = "".join(f"<span>{html.escape(str(v))}</span>" for v in
                    (r["celex"], r["procedure_ref"] or "no procedure", f.get("procedure_type") or "?",
                     f"DG {r['dg']}", r["category"], f"proposed {r['filing_date']:%d %b %Y}") if v)
    st.markdown(f'<div class="card"><div class="law-title">{html.escape(r["title"])}</div>'
                f'<div class="chips">{chips} <span><a href="{r["url"]}" target="_blank">EUR-Lex ↗</a></span></div>'
                f'</div>', unsafe_allow_html=True)
    st.write("")

    a, b = st.columns(2)
    a.markdown(hero_probability(r["p_law"]), unsafe_allow_html=True)
    b.markdown(hero_days(r), unsafe_allow_html=True)
    st.plotly_chart(timeline(r), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Economy at the query date")
        fig = economy_chart(f)
        if fig:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.caption("No economy data before 2021.")
        st.markdown(spectrum(f), unsafe_allow_html=True)
    with right:
        st.subheader("Momentum & signals")
        cards = signals(f, r["milestones"], r["query_date"])
        cols = st.columns(2)
        for i, card in enumerate(cards):
            cols[i % 2].markdown(card, unsafe_allow_html=True)

    st.subheader("Why? · LLM reading of the inputs")
    cfg = llm.config(secrets())
    if not cfg["key"]:
        st.warning("No LLM configured: set GROK_API_KEY in .env (or LLM_API_KEY + LLM_PROVIDER) to get the six reasons.")
    else:
        try:
            with st.spinner(f"Asking {cfg['model']}…"):
                why = reasons(llm_payload(r), qd.isoformat(), cfg)
            c1, c2, c3 = st.columns(3)
            c1.markdown(reason_block("Why it could pass", why["pass"], GREEN), unsafe_allow_html=True)
            c2.markdown(reason_block("Why it might not", why["fail"], RED), unsafe_allow_html=True)
            c3.markdown(reason_block(f"Why around {r['law_date']:%B %Y}", why["timing"], BLUE), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"LLM call failed: {e}")

    if r["milestones"]:
        with st.expander(f"EP / Council milestones up to the query date ({len(r['milestones'])})"):
            st.dataframe(pd.DataFrame(r["milestones"]), hide_index=True, width="stretch")
    with st.expander("Model inputs (exactly what the models received)"):
        st.json({"approval_model": {k: f.get(k) for k in r["model_features"]},
                 "timing_model": {"filing_date": str(r["filing_date"]), "title": r["title"], **r["meta"]},
                 "display_only": {k: v for k, v in f.items() if k.startswith("econ_")}})


st.markdown('<div class="brand">Bill-to-Law Predictor</div>'
            '<div class="tag">Will an EU Commission proposal become law, and when? Using only what was known on the query date.</div>',
            unsafe_allow_html=True)
tab_predict, tab_models, tab_arch = st.tabs(["⚡ Predict", "📊 Models", "🧭 Architecture"])
# The static tabs are drawn first: Streamlit fills tabs in code order, so they would otherwise stay
# blank while a prediction runs.
with tab_models:
    models_page.render()
with tab_arch:
    architecture.render()
with tab_predict:
    predict_page()
