"""Live status of the pipeline: which steps are done, partial, blocked or missing.

    streamlit run pipeline_status.py

Everything is read from the repo on each refresh (scripts, outputs, caches, the saved model),
so the page stays current as lanes push their work. Nothing here writes to data/.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

ICON = {"done": "✅", "partial": "🟡", "blocked": "⛔", "missing": "⬜"}
ORDER = ["done", "partial", "blocked", "missing"]

# Timeline on the day (Claude.md).
MILESTONES = [
    ("09:30", "Find and load data"),
    ("11:00", "Build"),
    ("14:00", "First model beating the baseline + CSV end to end"),
    ("17:00", "Feature freeze"),
    ("18:30", "Full dry run on a fake 40-bill test"),
    ("19:00", "Test inputs published"),
    ("20:00", "CSV submitted"),
]


# ------------------------------------------------------------------------ helpers

def path(rel: str) -> Path:
    return BASE / rel


def exists(rel: str) -> bool:
    return path(rel).exists()


@st.cache_data(show_spinner=False)
def _read_table(p: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, low_memory=False)


def table(rel: str) -> pd.DataFrame | None:
    p = path(rel)
    if not p.exists():
        return None
    return _read_table(str(p), p.stat().st_mtime)


def n_files(rel: str, pattern: str = "*") -> int:
    p = path(rel)
    return sum(1 for f in p.rglob(pattern) if f.is_file()) if p.exists() else 0


def grep(text: str, files: str = "*.py") -> list[str]:
    """Repo .py files (outside .venv) that contain text."""
    hits = []
    for f in BASE.rglob(files):
        if ".venv" in f.parts or f.name == Path(__file__).name:
            continue
        try:
            if text in f.read_text(errors="ignore"):
                hits.append(str(f.relative_to(BASE)))
        except OSError:
            pass
    return sorted(hits)


def age(rel: str) -> str:
    p = path(rel)
    if not p.exists():
        return ""
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M %d/%m")


def output_line(rel: str) -> str:
    df = table(rel) if rel.endswith((".csv", ".parquet")) else None
    rows = f", {len(df):,} rows" if df is not None else ""
    return f"`{rel}` ({age(rel)}{rows})" if exists(rel) else f"`{rel}` — missing"


# -------------------------------------------------------------------------- steps

@dataclass
class Step:
    lane: str
    name: str
    check: Callable[[], tuple[str, list[str]]]  # -> (status, detail lines)
    command: str = ""
    result: tuple[str, list[str]] = field(default=("missing", []), init=False)


def files_step(script: str | None, outputs: list[str], needs: list[str] = ()) -> Callable:
    """done = script and every output exist; blocked = an input it needs is missing."""
    def check():
        lines = []
        if script:
            lines.append(f"script `{script}` " + ("present" if exists(script) else "**missing**"))
        lines += [f"needs {output_line(n)}" for n in needs]
        lines += [f"output {output_line(o)}" for o in outputs]
        have_out = [exists(o) for o in outputs]
        if all(have_out) and (not script or exists(script)):
            return "done", lines
        if any(not exists(n) for n in needs):
            return "blocked", lines
        if any(have_out) or (script and exists(script)):
            return "partial", lines
        return "missing", lines
    return check


def check_texts():
    bills = table("data/ep_bills.csv")
    got = n_files("data/raw/ep/texts", "*.html")
    total = len(bills) if bills is not None else 0
    lines = [f"{got:,} / {total:,} proposal texts in `data/raw/ep/texts/`",
             "~30% of proposals have no English HTML (PDF only), so 100% is not expected"]
    if got == 0:
        return "missing", lines
    return ("done" if total and got >= 0.6 * total else "partial"), lines


def check_context():
    outs = ["data/weekly_europe_economy.csv", "data/eu_parliament_context.csv"]
    ok = all(exists(o) for o in outs)
    return ("done" if ok else "partial"), [f"output {output_line(o)}" for o in outs] + [
        "served by `apis/parlament_economy.py`, used by the `economy` and `parliament` sources"]


def check_sources():
    if not exists("sources.py"):
        return "missing", ["`sources.py` missing"]
    lines = ["`sources.py`: economy, parliament (free) · eurlex, scores (medium) · consultations, news (slow)"]
    cached = {d.name: n_files(f"data/features/{d.name}", "*.json")
              for d in sorted(path("data/features").glob("*")) if d.is_dir()} if exists("data/features") else {}
    lines.append("feature caches: " + (", ".join(f"{k} {v}" for k, v in cached.items())
                                       if cached else "none yet (slow sources never run)"))
    return ("done" if cached else "partial"), lines


def check_leakage():
    tests = sorted(str(p.relative_to(BASE)) for p in BASE.rglob("test*leak*.py") if ".venv" not in p.parts)
    inline = grep("post-filing or label columns among features")
    lines = [f"test file: {', '.join(tests) or '**none**'}",
             f"inline guard (assert in FeatureBuilder): {', '.join(inline) or 'none'}"]
    return ("done" if tests else "partial" if inline else "missing"), lines


def check_pass_model():
    st_, lines = files_step("pass_model.py", ["models/pass_model.joblib", "data/pass_model_test_predictions.csv"],
                            needs=["data/training_set.parquet"])()
    if exists("models/pass_model.joblib") and not exists("data/training_set.parquet"):
        lines.append("⚠️ the saved model exists but cannot be rebuilt here: its input is missing")
        st_ = "partial"
    return st_, lines


def check_timing():
    hits = [f for f in grep('"quantile"') + grep("'quantile'") if "pipeline_status" not in f]
    lines = [f"quantile model code: {', '.join(sorted(set(hits))) or '**none**'}",
             "labels ready: `days_to_law` (build_training_set.py), `days_to_status` (eu_lifecycle.csv)"]
    return ("partial" if hits else "missing"), lines


def check_bonus():
    hits = [f for f in grep("survives") if f not in ("pipeline_status.py",)]
    return ("partial" if hits else "missing"), [f"article-level code: {', '.join(hits) or '**none**'}",
                                                "needs proposal + final-act texts (download_ep.py texts)"]


def check_one_command():
    main = path("main.py")
    size = main.stat().st_size if main.exists() else 0
    return ("done" if size > 200 else "missing"), [
        f"`main.py` is {size} bytes" + (" (empty)" if size == 0 else ""),
        "top three teams re-run the pipeline live: must be one command with fixed seeds"]


STEPS = [
    Step("A · Data", "EUR-Lex metadata of every proposal and act",
         files_step("downloadLaws.py", ["data/eu_laws_metadata.csv"]),
         ".venv/bin/python downloadLaws.py"),
    Step("A · Data", "EP proposals with outcome (2021–2026)",
         files_step("download_ep.py", ["data/ep_bills.csv"]), "python download_ep.py proposals"),
    Step("A · Data", "Proposal full texts", check_texts, "python download_ep.py texts"),
    Step("A · Data", "Lifecycle table (filing → EP vote → adoption)",
         files_step("build_eu_lifecycle.py", ["data/eu_lifecycle.csv", "data/eu_lifecycle_balanced.csv"]),
         "python build_eu_lifecycle.py"),
    Step("A · Data", "Excel export for the team",
         files_step("export_xlsx.py", ["data/ep_bills.xlsx"], needs=["data/ep_bills.csv"]), "python export_xlsx.py"),
    Step("A · Data", "Economy and Parliament context", check_context),
    Step("B · Features", "Sources interface (filing-day features)", check_sources,
         "python sources.py 52021PC0206 --sources all"),
    Step("B · Features", "Training set (meta + sources + labels)",
         files_step("build_training_set.py", ["data/training_set.parquet"], needs=["data/eu_laws_metadata.csv"]),
         "python build_training_set.py"),
    Step("B · Features", "Leakage guard test", check_leakage),
    Step("C · Models", "Pass model + baseline (time split)", check_pass_model, "python pass_model.py"),
    Step("C · Models", "Timing model (quantile, days_to_law)", check_timing),
    Step("D · Delivery", "predict.py: 40 test bills → submission CSV",
         files_step("predict.py", [], needs=["models/pass_model.joblib"]), "python predict.py"),
    Step("D · Delivery", "APIs server for the demo", files_step("apis/server.py", []), "uvicorn apis.server:app"),
    Step("D · Delivery", "Streamlit demo", lambda: (
        ("done", ["`app.py` present"]) if exists("app.py") else
        ("missing", ["no Streamlit demo yet (this status page does not count)"]))),
    Step("D · Delivery", "Bonus: article survives / changed / removed", check_bonus),
    Step("D · Delivery", "Whole pipeline from one command", check_one_command),
]


def warnings() -> list[str]:
    out = []
    if exists("eu_lifecycle_balanced.csv"):
        out.append("`eu_lifecycle_balanced.csv` is also in the repo root and tracked by git; "
                   "the script writes `data/eu_lifecycle_balanced.csv`. The root copy can go stale.")
    req = path("requirements.txt").read_text().lower() if exists("requirements.txt") else ""
    missing = [p for p in ("scikit-learn", "lightgbm", "joblib", "beautifulsoup4", "openpyxl", "streamlit", "pyarrow")
               if p not in req]
    if missing:
        out.append(f"`requirements.txt` lacks packages the scripts import: {', '.join(missing)}.")
    return out


def git_untracked() -> list[str]:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=BASE, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [l[3:] for l in r.stdout.splitlines() if l.startswith(("??", " M", "M ", "A "))]


# ----------------------------------------------------------------------------- page

st.set_page_config(page_title="Pipeline status", page_icon="🛠️", layout="wide")
st.title("Bill-to-Law predictor · pipeline status")
st.caption(f"Read from the repo at {datetime.now():%H:%M:%S}. Refresh the page (R) to re-check.")

for s in STEPS:
    s.result = s.check()

counts = {k: sum(s.result[0] == k for s in STEPS) for k in ORDER}
cols = st.columns(5)
cols[0].metric("Steps", len(STEPS))
for c, k in zip(cols[1:], ORDER):
    c.metric(f"{ICON[k]} {k}", counts[k])
st.progress((counts["done"] + 0.5 * counts["partial"]) / len(STEPS),
            text=f"{counts['done']} done, {counts['partial']} partial (counted as half)")

# Timeline: next milestone.
now = datetime.now().strftime("%H:%M")
upcoming = [(t, m) for t, m in MILESTONES if t > now]
if upcoming:
    st.info(f"Next milestone **{upcoming[0][0]}**: {upcoming[0][1]}")

tab_steps, tab_data, tab_model, tab_repo = st.tabs(["Steps", "Data", "Model", "Repo"])

with tab_steps:
    for lane in dict.fromkeys(s.lane for s in STEPS):
        st.subheader(lane)
        for s in [s for s in STEPS if s.lane == lane]:
            status, lines = s.result
            with st.expander(f"{ICON[status]}  {s.name}", expanded=status in ("blocked", "partial")):
                st.markdown("\n".join(f"- {l}" for l in lines) or "-")
                if s.command:
                    st.code(s.command, language="bash")
    st.caption("✅ done · 🟡 partial (started, or output present but not reproducible) · "
               "⛔ blocked (an input is missing) · ⬜ missing")

with tab_data:
    lc = table("data/eu_lifecycle.csv")
    if lc is not None:
        st.subheader("Lifecycle table · status by filing year")
        lc["year"] = pd.to_datetime(lc["filing_date"]).dt.year
        ct = pd.crosstab(lc["year"], lc["status"])
        st.bar_chart(ct)
        c1, c2 = st.columns(2)
        c1.dataframe(ct, use_container_width=True)
        passed = lc[lc["status"] == "Passed"]
        c2.dataframe(passed.groupby("procedure_type")["days_to_status"].describe()[["count", "50%", "mean"]]
                     .rename(columns={"50%": "median days"}).round(0), use_container_width=True)
        st.caption("Recent years: Stuck means still pending, not failed. Closed recent proposals are the fast ones.")
    bills = table("data/ep_bills.csv")
    if bills is not None:
        st.subheader("EP proposals · published as law by year")
        g = bills.groupby("year").agg(proposals=("initiative_id", "size"), law=("is_law", "sum"))
        g["share law"] = (g["law"] / g["proposals"]).round(2)
        st.dataframe(g, use_container_width=True)
        st.caption("`is_law` empty = no published act found (pending, withdrawn or lapsed): not a 0.")

with tab_model:
    if not exists("models/pass_model.joblib"):
        st.warning("No saved pass model yet.")
    else:
        try:
            import joblib
            m = joblib.load(path("models/pass_model.joblib"))
            st.write(f"Trained on **{m['trained_on']}**, stale pending > {m['stale_pending_years']} years = failed")
            scores = pd.DataFrame(m["test_scores"])
            st.dataframe(scores.round(4), use_container_width=True)
            st.caption("Held-out test year. AUC higher is better; Brier and LogLoss lower. "
                       "Targets: beat the baseline clearly, AUC > 0.85 is excellent.")
        except ModuleNotFoundError as e:
            st.warning(f"Cannot read `models/pass_model.joblib` here: `{e.name}` is not installed "
                       f"(`pip install {e.name}`).")
    preds = table("data/pass_model_test_predictions.csv")
    if preds is not None:
        st.subheader("Test-year predictions")
        st.dataframe(preds, use_container_width=True)

with tab_repo:
    for w in warnings():
        st.warning(w)
    changed = git_untracked()
    if changed:
        st.subheader("Not committed yet")
        st.markdown("\n".join(f"- `{f}`" for f in changed))
