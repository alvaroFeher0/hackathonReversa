from __future__ import annotations
import csv
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from fastapi import FastAPI, HTTPException, Query
START_DATE = date(2021, 1, 1)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "weekly_europe_economy.csv"
PARLIAMENT_CONTEXT_PATH = DATA_DIR / "eu_parliament_context.csv"
LAW_SECTOR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ai", r"\b(ai|artificial intelligence|artificial-intelligence|algorithm|algorithmic|machine learning|automated decision|umela inteligence|umělá inteligence)\b"),
    ("ecological", r"\b(climate|green deal|emission|emissions|carbon|co2|biodiversity|nature restoration|environment|ecolog|sustainab|renewable|circular economy|klima|emise|ekolog|zivotni prostredi|životní prostředí)\b"),
    ("economic", r"\b(econom|budget|tax|fiscal|market|competition|competitiveness|investment|finance|bank|banking|trade|customs|subsid|single market|hospodar|ekonom|dan|dan[eě]|rozpocet|rozpočet)\b"),
    ("digital_market", r"\b(digital|data|platform|online|cyber|cybersecurity|privacy|gdpr|internet|telecom|semiconductor|chip|cloud|digit|kyber)\b"),
    ("migration_security", r"\b(migration|asylum|border|security|defence|defense|police|terror|crime|visa|schengen|preparedness|migrace|azyl|bezpecnost|bezpečnost|obrana)\b"),
    ("energy", r"\b(energy|electricity|gas|oil|nuclear|hydrogen|grid|power market|renewables|energet|plyn|ropa|jad(e|é)rn|vodik)\b"),
    ("social", r"\b(social|worker|workers|labour|labor|employment|wage|minimum wage|health|education|housing|pension|equality|socialn|pracovn|mzda|zdravotnictvi|bydleni)\b"),
    ("agriculture", r"\b(agriculture|farm|farmer|food|fisheries|fishery|rural|crop|livestock|pesticide|fertili[sz]er|zemedel|zeměděl|potravin|rybolov)\b"),
)
@dataclass(frozen=True)
class Instrument:
    column: str
    symbol: str
    display_name: str
    role: str
    weight: float

@dataclass(frozen=True)
class Sector:
    key: str
    display_name: str
    instrument_columns: tuple[str, ...]

INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("euro_stoxx_50", "^STOXX50E", "EURO STOXX 50", "risk", 0.35),
    Instrument("brent_crude", "BZ=F", "Brent crude oil futures", "risk", 0.12),
    Instrument("wti_crude", "CL=F", "WTI crude oil futures", "risk", 0.08),
    Instrument("natural_gas", "NG=F", "Natural gas futures", "risk", 0.07),
    Instrument("copper", "HG=F", "Copper futures", "risk", 0.16),
    Instrument("wheat", "ZW=F", "Wheat futures", "risk", 0.06),
    Instrument("corn", "ZC=F", "Corn futures", "risk", 0.05),
    Instrument("soybeans", "ZS=F", "Soybean futures", "risk", 0.05),
    Instrument("gold", "GC=F", "Gold futures", "safe_haven", 0.04),
    Instrument("silver", "SI=F", "Silver futures", "risk", 0.02),
)

INSTRUMENT_BY_COLUMN = {instrument.column: instrument for instrument in INSTRUMENTS}
COMMODITY_SECTORS: tuple[Sector, ...] = (
    Sector("energy", "Energy", ("brent_crude", "wti_crude", "natural_gas")),
    Sector("industrial_metals", "Industrial metals", ("copper", "silver")),
    Sector("agriculture", "Agriculture", ("wheat", "corn", "soybeans")),
    Sector("precious_metals", "Precious metals", ("gold", "silver")),
)

app = FastAPI(
    title="European Economy Condition API",
    description=(
        "Weekly average EURO STOXX 50 and commodity prices from 2021 onward, "
        "with a date-based economy condition classifier."
    ),
    version="1.0.0",
)
def _unix_seconds(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())
def _fetch_yahoo_daily_prices(symbol: str, start: date, end: date) -> list[tuple[date, float]]:
    period1 = _unix_seconds(start)
    period2 = _unix_seconds(end + timedelta(days=1))
    encoded_symbol = quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {symbol}: {exc}") from exc
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}: {error}")
    series = result[0]
    timestamps = series.get("timestamp") or []
    closes = (
        series.get("indicators", {})
        .get("quote", [{}])[0]
        .get("close", [])
    )
    prices: list[tuple[date, float]] = []
    for timestamp, close in zip(timestamps, closes):
        if close is None:
            continue
        day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        prices.append((day, float(close)))
    return prices
def _average_by_week(daily_prices: Iterable[tuple[date, float]]) -> dict[date, float]:
    buckets: dict[date, list[float]] = {}
    for day, price in daily_prices:
        buckets.setdefault(_week_start(day), []).append(price)
    return {week: sum(values) / len(values) for week, values in buckets.items()}
def _read_dataset(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
def _read_parliament_context(path: Path = PARLIAMENT_CONTEXT_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
def _to_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
def _parse_json_field(row: dict[str, str], field: str) -> object:
    value = row.get(field)
    if not value:
        return None
    return json.loads(value)
def _nearest_parliament_context(target: date, rows: list[dict[str, str]]) -> dict[str, str]:
    dated_rows = []
    for row in rows:
        start = date.fromisoformat(row["valid_from"])
        end = date.max if not row.get("valid_to") else date.fromisoformat(row["valid_to"])
        dated_rows.append((start, end, row))
    matches = [row for start, end, row in dated_rows if start <= target <= end]
    if not matches:
        first = min(start for start, _, _ in dated_rows).isoformat() if dated_rows else "unknown"
        raise HTTPException(status_code=404, detail=f"No EU Parliament context available before {first}.")
    return matches[-1]
def _parliament_payload(row: dict[str, str], requested_date: date | None = None) -> dict[str, object]:
    return {
        "requested_date": requested_date.isoformat() if requested_date else None,
        "valid_from": row["valid_from"],
        "valid_to": row.get("valid_to") or None,
        "parliament_term": row["parliament_term"],
        "seats_total": _to_int(row.get("seats_total")),
        "political_group_seats": _parse_json_field(row, "political_group_seats"),
        "eu_executive_consensus": row["eu_executive_consensus"],
        "commission_orientation": row["commission_orientation"],
        "parliament_orientation": row["parliament_orientation"],
        "left_right_score": _to_float(row.get("left_right_score")),
        "pro_eu_majority_seats": _to_int(row.get("pro_eu_majority_seats")),
        "right_populist_or_hard_right_seats": _to_int(row.get("right_populist_or_hard_right_seats")),
        "sector_stances": _parse_json_field(row, "sector_stances"),
        "notes": row["notes"],
        "sources": _parse_json_field(row, "sources"),
    }
def _parse_query_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Date must use YYYY-MM-DD format.") from exc
def _detect_law_sector(law_name: str) -> str:
    normalized = law_name.lower()
    for sector, pattern in LAW_SECTOR_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return sector
    return "economic"
def _law_stance_payload(row: dict[str, str], law_name: str) -> dict[str, object]:
    sector = _detect_law_sector(law_name)
    sector_stances = _parse_json_field(row, "sector_stances")
    if not isinstance(sector_stances, dict):
        raise HTTPException(status_code=500, detail="Sector stances are malformed.")
    sector_context = sector_stances.get(sector)
    if not isinstance(sector_context, dict):
        raise HTTPException(status_code=404, detail=f"No stance configured for sector '{sector}'.")
    return {
        "left_right": _to_float(row.get("left_right_score")),
        "sector_acceptance_score": sector_context.get("acceptance_score"),
        "executive_consensus": row["eu_executive_consensus"],
        "commission_orientation": row["commission_orientation"],
        "parliament_orientation": row["parliament_orientation"],
    }
def _normalize_score(value: float | None) -> float:
    if value is None:
        return 0.0
    scaled = value * 2.5
    return round(max(-5.0, min(5.0, scaled)), 3)
def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)
def _stdev(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 1.0
    value = statistics.stdev(clean)
    return value if value else 1.0
def _label_from_score(score: float) -> str:
    if score >= 1.25:
        return "abnormally getting better"
    if score >= 0.35:
        return "getting better"
    if score > -0.35:
        return "stagnating"
    if score > -1.25:
        return "getting worse"
    return "abnormally getting worse"

def _trend_from_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.35:
        return "getting better"
    if score <= -0.35:
        return "lowering"
    return "stagnating"

def _signed_z_score(row: dict[str, str], instrument: Instrument) -> float | None:
    z_score = _to_float(row.get(f"{instrument.column}_z"))
    if z_score is None:
        return None
    return -z_score if instrument.role == "safe_haven" else z_score

def _average_signal(row: dict[str, str], columns: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for column in columns:
        instrument = INSTRUMENT_BY_COLUMN[column]
        signal = _signed_z_score(row, instrument)
        if signal is not None:
            values.append(signal)
    return _mean(values)

def _index_signal(row: dict[str, str]) -> dict[str, object]:
    score = _average_signal(row, ("euro_stoxx_50",))
    return {
        "name": "EURO STOXX 50",
        "score": score,
        "trend": _trend_from_score(score),
        "weekly_average_price": _to_float(row.get("euro_stoxx_50")),
        "thirteen_week_momentum": _to_float(row.get("euro_stoxx_50_13w_momentum")),
    }

def _sector_signals(row: dict[str, str]) -> dict[str, dict[str, object]]:
    sectors: dict[str, dict[str, object]] = {}
    for sector in COMMODITY_SECTORS:
        score = _average_signal(row, sector.instrument_columns)
        sectors[sector.key] = {
            "name": sector.display_name,
            "score": score,
            "trend": _trend_from_score(score),
            "commodities": list(sector.instrument_columns),
        }
    return sectors

def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>European Economy Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f3;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #697077;
      --line: #dad7cb;
      --good: #157f51;
      --flat: #766400;
      --bad: #b33b2e;
      --accent: #2458a7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 660px;
      line-height: 1.45;
    }
    form {
      display: flex;
      gap: 8px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 8px;
      border-radius: 8px;
    }
    input, button {
      height: 40px;
      border-radius: 6px;
      font: inherit;
    }
    input {
      border: 1px solid var(--line);
      padding: 0 10px;
      min-width: 152px;
    }
    button {
      border: 0;
      background: var(--accent);
      color: white;
      padding: 0 14px;
      cursor: pointer;
    }
    .summary {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr);
      gap: 16px;
      margin-bottom: 16px;
    }
    .panel, .tile {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .label {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .condition {
      margin-top: 8px;
      font-size: clamp(26px, 5vw, 44px);
      font-weight: 760;
      line-height: 1.05;
    }
    .score {
      font-variant-numeric: tabular-nums;
      font-size: 32px;
      font-weight: 760;
      white-space: nowrap;
    }
    .meta {
      color: var(--muted);
      margin-top: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }
    .tile h2 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    .trend {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      border-radius: 6px;
      padding: 4px 8px;
      font-weight: 700;
      background: #ecebe4;
    }
    .trend.good { color: var(--good); }
    .trend.flat { color: var(--flat); }
    .trend.bad { color: var(--bad); }
    .metric {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
    }
    .metric strong {
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }
    th, td {
      border-top: 1px solid var(--line);
      padding: 9px 6px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 650; }
    .error {
      margin-top: 12px;
      color: var(--bad);
      min-height: 22px;
    }
    @media (max-width: 860px) {
      header, .status { align-items: stretch; flex-direction: column; }
      form { width: 100%; }
      input { flex: 1; min-width: 0; }
      .summary { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 520px) {
      main { width: min(100% - 20px, 1120px); padding-top: 18px; }
      .grid { grid-template-columns: 1fr; }
      .panel, .tile { padding: 14px; }
      table { font-size: 12px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>European Economy Monitor</h1>
        <p class="subtitle">Weekly EURO STOXX 50 and commodity-sector signals from 2021 onward.</p>
      </div>
      <form id="dateForm">
        <input id="dateInput" type="date" required>
        <button type="submit">Check</button>
      </form>
    </header>

    <section class="summary">
      <div class="panel">
        <div class="status">
          <div>
            <div class="label">Overall condition</div>
            <div id="condition" class="condition">Loading</div>
            <div id="matchedWeek" class="meta"></div>
          </div>
          <div>
            <div class="label">Economy score</div>
            <div id="overallScore" class="score">0.00</div>
          </div>
        </div>
        <div id="error" class="error"></div>
      </div>
      <div class="panel">
        <div class="label">Index signal</div>
        <div id="indexName" class="condition" style="font-size: 28px;">EURO STOXX 50</div>
        <div id="indexTrend" class="trend flat">stagnating</div>
        <div class="metric"><span>score</span><strong id="indexScore">0.00</strong></div>
        <div class="metric"><span>13-week momentum</span><strong id="indexMomentum">0.00%</strong></div>
      </div>
    </section>

    <section id="sectorGrid" class="grid"></section>

    <section class="panel" style="margin-top: 16px;">
      <div class="label">Commodity details</div>
      <table>
        <thead>
          <tr>
            <th>Commodity</th>
            <th>Weekly avg</th>
            <th>13w momentum</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody id="commodityRows"></tbody>
      </table>
    </section>
  </main>

  <script>
    const names = {
      brent_crude: "Brent crude",
      wti_crude: "WTI crude",
      natural_gas: "Natural gas",
      copper: "Copper",
      wheat: "Wheat",
      corn: "Corn",
      soybeans: "Soybeans",
      gold: "Gold",
      silver: "Silver"
    };
    const dateInput = document.querySelector("#dateInput");
    const form = document.querySelector("#dateForm");
    const errorBox = document.querySelector("#error");

    function trendClass(trend) {
      if (trend === "getting better") return "good";
      if (trend === "lowering") return "bad";
      return "flat";
    }
    function number(value) {
      return value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    }
    function percent(value) {
      return value === null || value === undefined ? "n/a" : `${(Number(value) * 100).toFixed(2)}%`;
    }
    function price(value) {
      return value === null || value === undefined ? "n/a" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    function todayIso() {
      return new Date().toISOString().slice(0, 10);
    }
    async function loadEconomy(day) {
      errorBox.textContent = "";
      const response = await fetch(`/economy?date=${encodeURIComponent(day)}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not load economy data");
      }
      return response.json();
    }
    function render(payload) {
      document.querySelector("#condition").textContent = payload.condition;
      document.querySelector("#overallScore").textContent = number(payload.economy_score);
      document.querySelector("#matchedWeek").textContent = `Requested ${payload.requested_date}; matched week ${payload.matched_week_start}`;
      document.querySelector("#indexName").textContent = payload.index.name;
      document.querySelector("#indexTrend").textContent = payload.index.trend;
      document.querySelector("#indexTrend").className = `trend ${trendClass(payload.index.trend)}`;
      document.querySelector("#indexScore").textContent = number(payload.index.score);
      document.querySelector("#indexMomentum").textContent = percent(payload.index.thirteen_week_momentum);

      const sectorGrid = document.querySelector("#sectorGrid");
      sectorGrid.innerHTML = Object.entries(payload.commodity_sectors).map(([key, sector]) => `
        <article class="tile">
          <h2>${sector.name}</h2>
          <div class="trend ${trendClass(sector.trend)}">${sector.trend}</div>
          <div class="metric"><span>score</span><strong>${number(sector.score)}</strong></div>
          <div class="metric"><span>commodities</span><strong>${sector.commodities.map(item => names[item] || item).join(", ")}</strong></div>
        </article>
      `).join("");

      const rows = document.querySelector("#commodityRows");
      const commodities = Object.keys(names);
      rows.innerHTML = commodities.map(key => `
        <tr>
          <td>${names[key]}</td>
          <td>${price(payload.weekly_average_prices[key])}</td>
          <td>${percent(payload.thirteen_week_momentum[key])}</td>
          <td>${number(payload.signed_signals[key])}</td>
        </tr>
      `).join("");
    }
    async function refresh(day) {
      try {
        render(await loadEconomy(day));
      } catch (error) {
        errorBox.textContent = error.message;
      }
    }
    form.addEventListener("submit", event => {
      event.preventDefault();
      refresh(dateInput.value);
    });
    dateInput.value = todayIso();
    refresh(dateInput.value);
  </script>
</body>
</html>"""
def build_dataset(
    start: date = START_DATE,
    end: date | None = None,
    output_path: Path = DATASET_PATH,
) -> list[dict[str, str]]:
    """Download market data and build the weekly economy-condition dataset."""
    end = end or date.today()
    weekly_prices: dict[str, dict[date, float]] = {}
    for instrument in INSTRUMENTS:
        daily_prices = _fetch_yahoo_daily_prices(instrument.symbol, start, end)
        weekly_prices[instrument.column] = _average_by_week(daily_prices)
        time.sleep(0.2)
    all_weeks = sorted(set().union(*[set(values) for values in weekly_prices.values()]))
    rows: list[dict[str, str]] = []
    raw_rows: list[dict[str, float | date | None]] = []
    for week in all_weeks:
        row: dict[str, float | date | None] = {"week_start": week}
        for instrument in INSTRUMENTS:
            row[instrument.column] = weekly_prices[instrument.column].get(week)
        raw_rows.append(row)
    momentum_by_column: dict[str, list[float]] = {instrument.column: [] for instrument in INSTRUMENTS}
    for index, row in enumerate(raw_rows):
        if index < 13:
            continue
        previous = raw_rows[index - 13]
        for instrument in INSTRUMENTS:
            current_price = row[instrument.column]
            previous_price = previous[instrument.column]
            if isinstance(current_price, float) and isinstance(previous_price, float) and previous_price:
                momentum_by_column[instrument.column].append((current_price / previous_price) - 1)
    stats: dict[str, tuple[float, float]] = {}
    for instrument in INSTRUMENTS:
        values = momentum_by_column[instrument.column]
        stats[instrument.column] = (_mean(values) or 0.0, _stdev(values))
    for index, row in enumerate(raw_rows):
        output: dict[str, str] = {"week_start": row["week_start"].isoformat()}  # type: ignore[union-attr]
        score_parts: list[float] = []
        for instrument in INSTRUMENTS:
            price = row[instrument.column]
            output[instrument.column] = "" if price is None else f"{price:.6f}"
            momentum: float | None = None
            z_score: float | None = None
            if index >= 13:
                previous_price = raw_rows[index - 13][instrument.column]
                if isinstance(price, float) and isinstance(previous_price, float) and previous_price:
                    momentum = (price / previous_price) - 1
                    center, spread = stats[instrument.column]
                    z_score = (momentum - center) / spread
                    direction = -1 if instrument.role == "safe_haven" else 1
                    score_parts.append(direction * instrument.weight * z_score)
            output[f"{instrument.column}_13w_momentum"] = "" if momentum is None else f"{momentum:.6f}"
            output[f"{instrument.column}_z"] = "" if z_score is None else f"{z_score:.6f}"
        economy_score = sum(score_parts) if score_parts else 0.0
        output["economy_score"] = f"{economy_score:.6f}"
        output["condition"] = _label_from_score(economy_score)
        index = _index_signal(output)
        output["index_score"] = "" if index["score"] is None else f"{index['score']:.6f}"
        output["index_trend"] = str(index["trend"])
        for key, sector in _sector_signals(output).items():
            output[f"{key}_score"] = "" if sector["score"] is None else f"{sector['score']:.6f}"
            output[f"{key}_trend"] = str(sector["trend"])
        rows.append(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["week_start"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
def _nearest_available_row(target: date, rows: list[dict[str, str]]) -> dict[str, str]:
    dated_rows = [
        (date.fromisoformat(row["week_start"]), row)
        for row in rows
        if row.get("week_start")
    ]
    if not dated_rows:
        raise HTTPException(status_code=404, detail="Dataset is empty. Run /refresh first.")
    past_rows = [(week, row) for week, row in dated_rows if week <= target]
    if not past_rows:
        first_week = dated_rows[0][0].isoformat()
        raise HTTPException(status_code=404, detail=f"No data available before {first_week}.")
    return past_rows[-1][1]
@app.get("/eu-parliament")
def eu_parliament(query_date: date | str = Query(alias="date")) -> dict[str, object]:
    query_date = _parse_query_date(query_date)
    rows = _read_parliament_context()
    if not rows:
        raise HTTPException(status_code=404, detail="EU Parliament context dataset is missing.")
    row = _nearest_parliament_context(query_date, rows)
    return _parliament_payload(row, query_date)
@app.get("/eu-parliament/law-stance")
def eu_parliament_law_stance(
    query_date: date | str = Query(alias="date"),
    law_name: str = Query(alias="law", min_length=1),
) -> dict[str, object]:
    query_date = _parse_query_date(query_date)
    rows = _read_parliament_context()
    if not rows:
        raise HTTPException(status_code=404, detail="EU Parliament context dataset is missing.")
    row = _nearest_parliament_context(query_date, rows)
    return _law_stance_payload(row, law_name)
@app.get("/economy")
def economy(query_date: date | str = Query(alias="date")) -> list[dict[str, object]]:
    query_date = _parse_query_date(query_date)
    rows = _read_dataset()
    if not rows:
        if os.getenv("AUTO_BUILD_DATASET", "1") == "1":
            rows = build_dataset()
        else:
            raise HTTPException(status_code=404, detail="Dataset is missing. Run /refresh first.")
    row = _nearest_available_row(query_date, rows)
    index = _index_signal(row)
    output = [{"name": index["name"], "score": _normalize_score(index["score"])}]
    for sector in _sector_signals(row).values():
        output.append({"name": sector["name"], "score": _normalize_score(sector["score"])})
    return output
@app.get("/combined")
def combined(
    query_date: date | str = Query(alias="date"),
    law_name: str = Query(alias="law", min_length=1),
) -> dict[str, object]:
    query_date = _parse_query_date(query_date)
    return {
        "law_stance": eu_parliament_law_stance(query_date, law_name),
        "economy": economy(query_date),
    }
if __name__ == "__main__":
    rows = build_dataset()
    print(f"Wrote {len(rows)} weekly rows to {DATASET_PATH}")
