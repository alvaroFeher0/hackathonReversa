"""Architecture tab: three APIs build one feature vector per proposal, then the models and the dashboard."""
import streamlit as st

DOT = """
digraph G {
  graph [rankdir=TB, bgcolor="transparent", pad=0.3, nodesep=0.3, ranksep=0.6, fontname="Helvetica", fontsize=16,
         fontcolor="#cfd6e6", compound=true];
  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14, color="#3ad7ff",
         fillcolor="#0f1424", fontcolor="#e6e9f2", penwidth=1.4];
  edge  [color="#5b6478", arrowsize=0.7, penwidth=1.2, fontcolor="#8b93a7", fontsize=10];

  ui [label="Input\\nCELEX 5yyyyPC#### + query date\\nwindow 01/01/2021 – 03/10/2026", color="#ffb547"];

  subgraph cluster_src {
    label="Public sources"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    cellar [label="EUR-Lex Cellar\\nSPARQL + full text", color="#a78bfa"];
    ctx    [label="EP composition\\n(by term)", color="#a78bfa"];
    yahoo  [label="Market prices\\n(weekly)", color="#a78bfa"];
    ep     [label="EP Open Data\\n(votes)", color="#a78bfa"];
    hys    [label="Have Your Say\\n(consultations)", color="#a78bfa"];
  }

  subgraph cluster_apis {
    label="Feature APIs (everything cut at the query date)"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    api1 [label="API 1 · Proposal metadata & history\\ntitle, summary, author / DG, filing date\\ncategory · subcategory · source\\nleft/right (0/1) · disruptive/acceptable (0/1)\\nclose_prev_proposals (top 3) → avg_approved"];
    api2 [label="API 2 · Political & economic context\\nParliament left/right balance\\nsector acceptance · consensus\\nstocks · energy · metals · oil (name, score)"];
    api3 [label="API 3 · Consultations & votes\\n# public consultations\\n# votes · support trend · support variability"];
  }

  fv [label="Feature vector\\nintrinsic + historical prior (1)\\ncontext (2) · procedural progress (3)", color="#ffb547"];

  subgraph cluster_models {
    label="Prediction (trained on 2021–2026 proposals with known outcomes)"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    m1 [label="Classifier\\np_law", shape=cylinder, color="#34f5a4"];
    m2 [label="Timing model\\ndays_to_law", shape=cylinder, color="#34f5a4"];
  }

  llmn [label="LLM\\n2 reasons pass · 2 fail · 2 timing\\nno info after the query date", color="#ffb547"];
  dash [label="Dashboard", color="#ffb547"];

  ui -> api1 [label=" CELEX"]; ui -> api3 [label=" CELEX + up_to"];
  cellar -> api1; ctx -> api2; yahoo -> api2; ep -> api3; hys -> api3;
  api1 -> api2 [label=" title + date"];
  api1 -> fv; api2 -> fv; api3 -> fv;
  fv -> m1; fv -> m2; m1 -> llmn; m2 -> llmn; fv -> llmn [style=dashed];
  m1 -> dash; m2 -> dash; llmn -> dash;
}
"""


def render() -> None:
    st.subheader("How a prediction is made")
    st.graphviz_chart(DOT, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**API 1 · Proposal metadata & history**  \nThe entry point. From the CELEX it returns the title "
                "(passed to API 2), summary, author / DG, filing date and a category / subcategory / source "
                "classification, plus two binary scores: left/right orientation and disruptive/acceptable. "
                "Its key feature is `close_prev_proposals`: the three most similar earlier proposals "
                "(title, similarity, approved, adoption date), averaged into `avg_approved`, a strong prior.")
    c2.markdown("**API 2 · Political & economic context**  \nFrom the title and proposal date: the left/right "
                "balance of Parliament, how well the affected sector accepts the proposal, a consensus measure, "
                "and market indicators (stocks, energy, metals, oil) as name–score pairs. The same proposal can "
                "have very different chances in a different Parliament or economy.")
    c3.markdown("**API 3 · Consultations & votes**  \nFrom the CELEX and an `up_to` date: the number of public "
                "consultations, and a summary of EU votes: how many, whether support is rising, and how much it "
                "varies. Stable, growing support points to adoption; erratic support means uncertainty.")
    st.markdown("**Prediction**  \nThe three outputs are merged into one feature vector per proposal: intrinsic "
                "characteristics and historical priors (API 1), context (API 2) and procedural progress (API 3). "
                "A classifier trained on 2021–2026 proposals with known outcomes predicts whether it will be "
                "adopted; a timing model predicts the days to law. The LLM then explains the result with two "
                "reasons for, two against and two on timing. It never feeds the models.")
