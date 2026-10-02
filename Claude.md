# Bill-to-Law Predictor

Hackathon project: Madrid Open, Reversa track, Challenge 02. Saturday 3 October 2026, 09:00–21:00, team of four.

## Goal

For a Spanish bill, predict **using only what was known on the day it was filed**:

1. `p_law`: probability (0–1) that it is published as law in the BOE.
2. `days_to_boe`: days from filing to BOE publication.
3. Bonus: per article, whether it `survives`, is `changed` or is `removed` between the first text and the final law.

## The one rule that disqualifies us

**No information dated after the filing date may reach the model.** No later stages, votes, committee assignment, news, or outcome-derived fields.

- Every feature must be computable from the filing record plus context that existed on that date.
- Columns prefixed `post_` are post-filing. They exist for analysis only and must never be model inputs.
- The dissolution date of a legislature is future information. `days_since_legislature_start` is allowed; days until dissolution is not.
- Do not ask an LLM about a bill's fate or content beyond its filing text: it may recall the outcome. LLM use is limited to classifying the topic from the title.
- Votes may only be used as history: aggregates over votes held strictly before the filing date.

When adding a feature, state in the PR or commit which source field it comes from and why it is known at filing.

## How we are scored

Hidden test: at 19:00 the organisers publish 40 past bills; we submit a CSV by 20:00.

| Metric | Bad | Good | Excellent |
|---|---|---|---|
| Pass/fail (Brier, AUC) | Does not beat baseline | Beats it clearly | AUC > 0.85 |
| Timing (mean error, days) | > 120 | < 90 | < 45 |
| Bonus (F1 on articles) | < 0.3 | 0.5 | ≥ 0.7 |

Overall weight: hidden test 60%, difficulty and bonus layers 20%, live demo 20%. The top three teams re-run their pipeline live, so everything must run from one command and be deterministic (fixed seeds).

**Baseline to beat:** Government bill passes, group bill fails. On the 174 closed bills of legislature XV it is 91% accurate, so it is strong.

## Submission format

Main CSV: `initiative_id, p_law, days_to_boe`
Bonus CSV: `initiative_id, article, survives / changed / removed`

The exact format of the 40 test bills is unknown until 19:00. Feature building must therefore be one function that takes minimal filing fields (id, filing date, type, author, title) and is shared by training and prediction.

## Domain primer

- Two kinds of bills: `121` = *proyecto de ley* (sent by the Government); `122` = *proposición de ley* from parliamentary groups; also `124` (Senate) and `125` (regional parliaments).
- Path: filing → (for group bills) plenary vote on whether to debate it → amendments → committee → Congress vote → Senate → publication in the BOE.
- 350 deputies, 176 is an absolute majority, no party has it alone: every law is a coalition deal.
- When parliament is dissolved, every bill still in progress dies (*caducado*).
- Outcomes seen in the data: `Aprobado con/sin modificaciones` (law), `Rechazado`, `Retirado`, `Decaído`, `Subsumido en otra iniciativa`, `Inadmitido a trámite`. Everything except *Aprobado* is `is_law = 0`. Bills still open have no label; never fill them with 0.
- The Congreso result date is the file-closing date, not the publication date. `days_to_boe` uses the BOE date.

## Data sources (public only; none handed out)

| Data | Source | Notes |
|---|---|---|
| Current legislature (XV) bills | `https://www.congreso.es/es/opendata/iniciativas` — `ProyectosDeLey`, `ProposicionesDeLey`, `IniciativasLegislativasAprobadas` (JSON/CSV/XML) | File names carry a daily timestamp; read the links from the page |
| Past legislatures (X–XIV) | `https://www.congreso.es/es/busqueda-de-iniciativas` | No bulk export; crawl one detail page per file number. URL parameters in `congreso_download.py` are unverified |
| Law text and publication | BOE API, `https://boe.es/datosabiertos/api/legislacion-consolidada` | |
| Votes | `https://www.congreso.es/webpublica/opendata/votaciones/Leg{N}/Sesion{S}/{YYYYMMDD}/` | One zip per plenary session; optional |
| Senate initiatives | `https://www.senado.es/web/relacionesciudadanos/datosabiertos/catalogodatos/iniciativas/index.html` | Fallback; biased towards bills that advanced |

Official JSON fields per bill: `LEGISLATURA, TIPO, OBJETO, NUMEXPEDIENTE, FECHAPRESENTACION, FECHACALIFICACION, AUTOR, TIPOTRAMITACION, RESULTADOTRAMITACION, SITUACIONACTUAL, COMISIONCOMPETENTE, PLAZOS, PONENTES, TRAMITACIONSEGUIDA, INICIATIVASRELACIONADAS, ENLACESBOCG, ENLACESDS`. Values contain embedded newlines; dates are `dd/mm/yyyy`.

## Current state

- `congreso_download.py`: downloader with three commands (`initiatives`, `history`, `votes`). Not yet tested against the live site.
- `build_dataset.py`: builds the master table from the three official JSON files, including the fuzzy title match between bills and published laws.
- `bills_leg15.csv`: legislature XV, 484 bills: 32 laws, 142 failed, 310 still open (unlabelled). Days to BOE: 37–934, median 243.

**Still missing:** legislatures X–XIV, which are the real training set, and a parser that turns their crawled HTML pages into the same columns as `bills_leg15.csv`.

## Master table schema

One row per bill.

| Column | Role |
|---|---|
| `initiative_id`, `legislature` | Key |
| `type`, `title`, `author`, `filing_date` | Raw filing fields |
| `author_kind`, `is_government_bill`, `author_in_government`, `n_authors`, `is_organic`, `amends_existing_law`, `days_since_legislature_start` | Features known at filing |
| `status`, `result`, `result_date` | Outcome |
| `is_law`, `days_to_boe`, `law_title`, `boe_date`, `match_score` | Labels |
| `post_procedure`, `post_committee`, `post_current_stage` | Post-filing; never model inputs |

`author_in_government` is hardcoded to PSOE and Sumar, valid for XV only. Other legislatures need a hand-made context table: start date, governing parties, seats per group.

## Modelling decisions

- Train on closed legislatures X–XIV. Do not train on XV: its closed bills are the fast ones, which biases pass rate and timing.
- Validate by time: train on older legislatures, evaluate on the most recent closed one. Never a random split.
- Pass model: logistic regression first, then LightGBM; calibrate probabilities (Brier rewards calibration).
- Timing model: predict the conditional median (quantile objective), trained on bills that became law.
- Always report the baseline next to any model score.
- Legislatures XI and XIII lasted months and almost everything lapsed; test with and without them.

## Stack and conventions

- Python 3.11+, `pandas`, `scikit-learn`, `lightgbm`, `shap`, `requests`, `beautifulsoup4`; Streamlit for the demo.
- Storage: Parquet/CSV files under `data/`. No database.
- `data/raw/` is never edited; everything downstream is rebuilt from it by script.
- Pipeline code lives in scripts, not notebooks. Notebooks are for exploration only.
- Crawling: keep the one-second pause between requests and cache every page to disk.

## Team lanes

| Lane | Owns |
|---|---|
| A: Data | Download, cleaning, BOE match, master table |
| B: Features | Feature function and the leakage guard test |
| C: Models | Baseline, pass model, timing model, time-based validation |
| D: Delivery | `predict.py` (test input → CSV), demo, bonus |

## Timeline on the day

09:30 find and load data · 11:00 build · 14:00 first model beating the baseline and a CSV generated end to end · 17:00 feature freeze · 18:30 full dry run on a fake 40-bill test · 19:00 test inputs published · 20:00 CSV submitted · 20:00–21:00 five-minute demos.