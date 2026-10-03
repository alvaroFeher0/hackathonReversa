# Bill-to-Law Predictor (EU)

Hackathon project: Madrid Open, Reversa track, Challenge 02. Saturday 3 October 2026, 09:00–21:00, team of four.

On the day, the team switched from Spanish bills (Congreso/BOE) to **EU legislation from EUR-Lex**. A European Commission *proposal* is the "bill"; an adopted regulation, directive or decision is the "law".

## Goal

For an EU legislative proposal, predict **using only what was known on the day it was adopted by the Commission**:

1. `p_law`: probability (0–1) that it is adopted and published as an act in the Official Journal (OJ).
2. `days_to_law`: days from the proposal date to the adopted act.
3. Bonus: per article, whether it `survives`, is `changed` or is `removed` between the proposal text and the final act.

## The one rule that disqualifies us

**No information dated after the proposal date may reach the model.** No later events, votes, Council positions, adoption, withdrawal, news, or outcome-derived fields.

- Every feature must be computable from the proposal record plus context that existed on that date.
- Columns prefixed `post_` are post-proposal. They exist for analysis only and must never be model inputs.
- Many Cellar fields are filled in after the fact. See "Leakage map" below; when in doubt, treat a field as post-proposal.
- The end date of a Parliament term or Commission mandate is allowed only as it was scheduled at the time. Never use a known actual end or reshuffle.
- Do not ask an LLM about a proposal's fate or content beyond its proposal text: it may recall the outcome. LLM use is limited to classifying the topic from the title.
- Aggregates over other dossiers (for example, the pass rate per DG or per procedure type) may only use dossiers closed strictly before the proposal date.

When adding a feature, state in the PR or commit which source field it comes from and why it is known at the proposal date.

## How we are scored

Hidden test: at 19:00 the organisers publish 40 past bills; we submit a CSV by 20:00. **TODO: confirm with the organisers that EU proposals are accepted, and in what input format.**

| Metric | Bad | Good | Excellent |
|---|---|---|---|
| Pass/fail (Brier, AUC) | Does not beat baseline | Beats it clearly | AUC > 0.85 |
| Timing (mean error, days) | > 120 | < 90 | < 45 |
| Bonus (F1 on articles) | < 0.3 | 0.5 | ≥ 0.7 |

Overall weight: hidden test 60%, difficulty and bonus layers 20%, live demo 20%. The top three teams re-run their pipeline live, so everything must run from one command and be deterministic (fixed seeds).

**Baseline:** the Spanish baseline (Government bill passes, group bill fails) does not carry over. Use a simple EU baseline, such as the majority class or the pass rate per procedure type (`OLP`, `NLE`, `CNS`…), computed on the training years. Always report it next to any model score.

## Submission format

Main CSV: `initiative_id, p_law, days_to_law` (`initiative_id` = proposal CELEX, e.g. `52021PC0206`)
Bonus CSV: `initiative_id, article, survives / changed / removed`

The exact format of the 40 test bills is unknown until 19:00. Feature building must therefore be one function that takes minimal proposal fields (CELEX, date, type, author/DG, title) and is shared by training and prediction.

## Domain primer

- **CELEX ids.** Sector `5` = preparatory acts, so `5yyyyPC####` is a Commission proposal (`PROP_REG`, `PROP_DIR`, `PROP_DEC`). Sector `3` = adopted acts: `3yyyyR####` regulation, `L` directive, `D` decision (`REG`, `DIR`, `DEC`).
- **Dossier / procedure.** Each proposal belongs to an interinstitutional procedure such as `2021/0395/COD`. The dossier links the proposal to the act it produced.
- **Procedure types** (`...type_procedure_code_interinstitutional`): `OLP`/`COD` ordinary legislative procedure (Parliament + Council co-decide); `NLE` non-legislative; `CNS` consultation; `APP` consent; `BUD` budget. Procedure type strongly affects both the pass rate and the timing.
- **Outcomes** (dossier flags): `adopted-proposal = 1` → law; `withdrawn-proposal = 1` → not law; `pending-proposal = 1` → still open, so **unlabelled: never fill it with 0**. `file-status` gives the same information (`PROP_ADOPTED`, `PROP_PENDING`, …) but is often empty.
- Commission withdrawals cluster at the start of a new Commission mandate (its work programme withdraws old pending proposals).

## Data

Source: EUR-Lex Cellar SPARQL endpoint `https://publications.europa.eu/webapi/rdf/sparql` (public, no key). Full text is fetched from `http://publications.europa.eu/resource/celex/{CELEX}`.

`downloadLaws.py` builds it:

```
.venv/bin/python downloadLaws.py --smoke   # 2022–2023, ~300 docs, ~1 min
.venv/bin/python downloadLaws.py           # all proposals + laws: metadata and titles (~172k docs, slow)
.venv/bin/python downloadLaws.py --text    # also English full text (slower, large)
```

- Output: `data/eu_laws_metadata.csv` (one row per CELEX, ~107 columns: every `cdm:` property, the English title and `dossier_*` fields); `data/eu_laws.csv` adds `text`.
- Everything is cached under `data/raw/` (`ids/`, `meta/`, `text/`); a re-run resumes. Multi-valued fields are joined with ` | `; authority URIs are shortened to codes (`COM`, `eurovoc:1234`).
- About 30% of documents have no English HTML text (PDF only): `text` is empty.

## Leakage map (Cellar fields on a proposal row)

| Known at the proposal date: candidate features | Post-proposal: labels/analysis only |
|---|---|
| `celex`, `resource_type`, `year`, `title`, `work_date_document` (proposal date) | `dossier_dossier_adopted-proposal`, `...withdrawn-proposal`, `...pending-proposal`, `...not-adopted-proposal` |
| `work_created_by_agent`, `resource_legal_service_responsible` (DG) | `dossier_procedure_code_interinstitutional_has_status_file-status` |
| `dossier_procedure_code_interinstitutional_has_type_concept_type_procedure_code_interinstitutional` (OLP/NLE/…), `..._european_union_competence_type` | `dossier_dossier_date_adopted`, `dossier_dossier_date_withdrawn`, `dossier_dossier_produces_resource_legal` |
| `resource_legal_based_on_concept_treaty`, `resource_legal_based_on_resource_legal` (legal basis) | `dossier_dossier_contains_event`, `dossier_dossier_contains_work`, `work_part_of_event*` |
| `resource_legal_proposes_to_amend_resource_legal`, `resource_legal_eea` | `resource_legal_date_vote`, `..._date_signature`, `..._date_entry-into-force`, `..._in-force`, `..._date_end-of-validity` |
| `resource_legal_is_about_subject-matter`, `..._directory-code`, `work_is_about_concept_eurovoc`¹ | `official-journal-act_*`, `resource_legal_published_in_official-journal`, `work_version`, `work_date_creation_legacy` |

¹ Subject and EuroVoc codes are assigned when the document is catalogued. They describe the proposal's content, so they are acceptable, but say so in the commit when you use them.

## Master table (to build)

One row per **proposal**, joined to the act its dossier produced.

| Column | Role |
|---|---|
| `initiative_id` (proposal CELEX), `procedure_ref` | Key |
| `resource_type`, `title`, `author`, `dg`, `proposal_date` | Raw proposal fields |
| procedure type, legal basis, amends-existing-act flag, subject/EuroVoc, days since the start of the Commission mandate | Features known at the proposal date |
| `is_law` (adopted = 1, withdrawn = 0, pending = NaN) | Label |
| `law_celex`, `law_date`, `days_to_law` | Labels: `law_date` is the adopted act's OJ publication date if present, otherwise its `work_date_document`. **Verify which is better before freezing.** |
| `post_*` | Everything in the right-hand column of the leakage map |

Mandate context (Commission start dates, Parliament terms) needs a small hand-made table.

## Modelling decisions

- Validate by time: train on older years, evaluate on the most recent fully closed years. Never a random split.
- Recent proposals are biased: the closed ones are the fast ones. Exclude years where a large share is still pending, or report results with and without them.
- Pass model: logistic regression first, then LightGBM; calibrate probabilities (Brier rewards calibration).
- Timing model: predict the conditional median (quantile objective), trained on proposals that became law.
- Always report the baseline next to any model score.

## Stack and conventions

- Python 3.11+ (venv: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`; on macOS LightGBM also needs `brew install libomp`).
- `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `shap`, `requests`, `beautifulsoup4`; Streamlit for the demo.
- Storage: Parquet/CSV files under `data/`. No database.
- `data/raw/` is never edited; everything downstream is rebuilt from it by script.
- Pipeline code lives in scripts, not notebooks. Notebooks are for exploration only.
- Be polite to the endpoint: keep the pauses in `downloadLaws.py` and cache every response to disk.

## Team lanes

| Lane | Owns |
|---|---|
| A: Data | `downloadLaws.py`, proposal → dossier → act linking, master table |
| B: Features | Feature function and the leakage guard test |
| C: Models | Baseline, pass model, timing model, time-based validation |
| D: Delivery | `predict.py` (test input → CSV), demo, bonus |

## Timeline on the day

09:30 find and load data · 11:00 build · 14:00 first model beating the baseline and a CSV generated end to end · 17:00 feature freeze · 18:30 full dry run on a fake 40-bill test · 19:00 test inputs published · 20:00 CSV submitted · 20:00–21:00 five-minute demos.
