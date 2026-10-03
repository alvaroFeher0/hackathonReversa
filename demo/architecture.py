"""Architecture tab: how data flows from the public sources to the dashboard."""
import streamlit as st

DOT = """
digraph G {
  graph [rankdir=TB, bgcolor="transparent", pad=0.3, nodesep=0.25, ranksep=0.55, fontname="Helvetica", fontsize=16,
         fontcolor="#cfd6e6", compound=true];
  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, color="#3ad7ff",
         fillcolor="#0f1424", fontcolor="#e6e9f2", penwidth=1.4];
  edge  [color="#5b6478", arrowsize=0.7, penwidth=1.2];

  subgraph cluster_src {
    label="Public sources"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    cellar   [label="EUR-Lex Cellar\\nSPARQL + full text", color="#a78bfa"];
    ep       [label="European Parliament\\nOpen Data API (votes)", color="#a78bfa"];
    oeil     [label="OEIL Legislative\\nObservatory (key events)", color="#a78bfa"];
    hys      [label="Have Your Say\\n(consultations)", color="#a78bfa"];
    yahoo    [label="Market prices\\n(weekly, 2021+)", color="#a78bfa"];
    ctx      [label="EP context table\\n(hand-made, by term)", color="#a78bfa"];
  }

  subgraph cluster_apis {
    label="apis/"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    eurlex   [label="eurlex_api + eurlex_scores\\nmetadata, summary,\\nleft/right, similar laws"];
    hist     [label="law_history\\nconsultations, EP votes\\n(up to query date)"];
    econ     [label="parlament_economy\\neconomy scores -5..+5,\\nsector acceptance"];
  }

  subgraph cluster_train {
    label="Training (offline)"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    bal      [label="eu_lifecycle_balanced.csv\\nproposals + outcomes"];
    life     [label="build_lifecycle_set.py\\n4 query dates / proposal"];
    meta     [label="eu_laws_metadata.csv\\n→ build_training_set.py"];
    evts     [label="fetch_ep_events.py\\nEP / Council milestones"];
    tl       [label="train_lifecycle_model.py\\nlogreg / LightGBM", color="#34f5a4"];
    tt       [label="train_timing_model.py\\nLightGBM median (L1)\\n5 query dates / proposal", color="#34f5a4"];
  }

  subgraph cluster_models {
    label="models/"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    m1 [label="lifecycle_model.joblib\\np_law", shape=cylinder, color="#34f5a4"];
    m2 [label="timing_model.joblib\\ndays_to_law", shape=cylinder, color="#34f5a4"];
  }

  subgraph cluster_demo {
    label="demo/ (Streamlit)"; color="#2a3145"; style="rounded,dashed"; fontcolor="#8b93a7";
    ui    [label="CELEX + query date", color="#ffb547"];
    pipe  [label="pipeline.py\\nsame feature code\\nas training", color="#ffb547"];
    llmn  [label="llm.py\\n6 reasons, no info\\nafter the query date", color="#ffb547"];
    dash  [label="Dashboard\\nprobability, timeline,\\neconomy, momentum", color="#ffb547"];
  }

  cellar -> eurlex; ep -> hist; hys -> hist; yahoo -> econ; ctx -> econ;
  bal -> life; eurlex -> life; hist -> life; econ -> life; life -> tl -> m1;
  cellar -> meta -> tt -> m2; oeil -> evts -> tt; evts -> pipe [style=dashed];
  ui -> pipe; eurlex -> pipe [style=dashed]; hist -> pipe [style=dashed]; econ -> pipe [style=dashed];
  cellar -> pipe [style=dashed, label=" metadata", fontcolor="#8b93a7", fontsize=9];
  m1 -> pipe; m2 -> pipe; pipe -> llmn -> dash; pipe -> dash;
}
"""


def render() -> None:
    st.subheader("How a prediction is made")
    st.graphviz_chart(DOT, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Features as of the query date**  \nEvery source is cut at the query date "
                "(votes, consultations, similar laws adopted before it, last full economy week). "
                "The demo calls the exact functions that built the training set.")
    c2.markdown("**2 · Two models**  \n`lifecycle_model` gives the probability of adoption from features "
                "known on the query date. `timing_model` predicts days from proposal to law from the "
                "proposal metadata (procedure type, DG, legal bases, title) and the EP / Council "
                "milestones reached by the query date.")
    c3.markdown("**3 · Explanation**  \nThe LLM gets the model inputs and outputs and returns two reasons for "
                "passing, two against, two for the timing. It explains; it never feeds the models.")
