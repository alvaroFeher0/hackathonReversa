"""EU law data from a proposal CELEX. You only need the CELEX (and optionally up_to).

HOW TO CALL EACH FUNCTION (AI Act proposal CELEX is "52021PC0206"):

    from apis.law_history import get_consultation_count, get_eu_votings

    # 1) Number of public consultations (Have Your Say portal)
    get_consultation_count("52021PC0206")                  # -> 3
    get_consultation_count("52021PC0206", up_to="2021-01-01")  # -> 2 (only rounds published up to that date)

    # 2) European Parliament votings summary
    get_eu_votings("52021PC0206")                          # -> dict with n_votings, in_favor_increase,
                                                          #    in_favor_variability and final_vote
    get_eu_votings("52021PC0206", up_to="2023-12-31")      # -> same, only votes up to that date

INPUTS (both functions):
    celex : str  - Commission proposal CELEX (sector 5), e.g. "52021PC0206". URLs containing it also work.
            Adopted-act CELEX ("3yyyy...") are rejected.
    up_to : optional cutoff ("YYYY-MM-DD" or datetime). Only data with date <= up_to counts.
            None (default) = all data.

OUTPUTS:
    get_consultation_count -> int (number of consultations)
    get_eu_votings        -> dict {n_votings, in_favor_increase, in_favor_variability, final_vote}
"""

import json
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime

_BR_API = "https://ec.europa.eu/info/law/better-regulation/brpapi"
_EP_API = "https://data.europarl.europa.eu/api/v2"
_EP_HEADERS = {"User-Agent": "hackathon-test", "Accept": "application/ld+json"}
_STOP = {"regulation", "of", "the", "european", "parliament", "and", "council",
        "laying", "down", "proposal", "for", "on", "amending", "certain",
        "union", "legislative", "acts", "act", "with", "text", "eea",
        "relevance", "directive", "decision"}


# ---------------------------------------------------------------------------
# Shared helpers (used by both functions)
# ---------------------------------------------------------------------------

def _normalize_celex(celex: str) -> str:
    """Extract a clean proposal CELEX ("52021PC0206") from a plain id or a URL."""
    m = re.search(r"5\d{4}PC\d{4}", str(celex).upper())
    if not m:
        raise ValueError(f"Invalid proposal CELEX {celex!r}. E.g. '52021PC0206' (AI Act proposal).")
    return m.group(0)


def _to_datetime(value) -> datetime:
    """Accept datetime, unix timestamp or strings like "2024-03-13", "13/03/2024"."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    s = str(value).strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d", "%Y-%m-%d %H:%M", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:19] if "%H" in fmt else s[:10], fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(s[:10])


def _cellar_notice(celex: str) -> str:
    """Raw CELLAR metadata (XML) for a CELEX: title, procedure link, proposal COMs."""
    req = urllib.request.Request(f"http://publications.europa.eu/resource/celex/{celex}",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/xml;notice=object"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# FUNCTION 1: number of public consultations
# ---------------------------------------------------------------------------
# INPUT:  celex (str, proposal CELEX e.g. "52021PC0206"), up_to (optional date, default all)
# OUTPUT: int, e.g. AI Act -> 3 (2 when up_to="2021-01-01")
# HOW:    proposal CELEX -> CELLAR title + COM -> Have Your Say search ->
#         initiative whose publication references that COM -> count publications.

def _norm_com(s: str) -> str:
    s = str(s).upper().replace(" ", "")
    m = re.search(r"COM\((\d{4})\)0*(\d+)", s)
    if m:
        return f"COM({m.group(1)}){m.group(2)}"
    m = re.search(r"COM_(\d{4})_0*(\d+)", s)
    if m:
        return f"COM({m.group(1)}){m.group(2)}"
    return s


def _hys_search(text: str, size: int = 8) -> list:
    qs = urllib.parse.urlencode({"language": "EN", "page": 0, "size": size, "text": text})
    req = urllib.request.Request(_BR_API + "/searchInitiatives?" + qs,
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8")).get("initiativeResultDtoPage", {}).get("content", [])


def _hys_group(initiative_id) -> dict:
    req = urllib.request.Request(f"{_BR_API}/groupInitiatives/{int(float(initiative_id))}",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _resolve_initiative_id(celex: str) -> str:
    """Find the Have Your Say initiative behind a CELEX (via its proposal COM)."""
    celex = _normalize_celex(celex)
    notice = _cellar_notice(celex)
    tm = re.search(r"<TITLE[^>]*>\s*<VALUE>(.*?)</VALUE>", notice, flags=re.S)
    title = tm.group(1).strip() if tm else ""
    coms = Counter(_norm_com(m.group(0)) for m in re.finditer(r"COM\(\d{4}\)\d+", notice))
    coms.update(_norm_com(f"COM({m.group(1)}){m.group(2)}")
                for m in re.finditer(r"COM_(\d{4})_0*(\d+)", notice))
    queries = [q for q in re.findall(r"\(([^()]{4,80})\)", title)]
    core = re.sub(r"^(PROPOSAL FOR AN? )?(COUNCIL )?(REGULATION|DIRECTIVE|DECISION)\b.*?(COUNCIL|PARLIAMENT)\s*",
                  "", title, flags=re.I)
    core = core.split(" AND AMENDING")[0].strip()
    if core:
        queries.append(core[:120])
    cands = {}
    for q in queries[:3]:
        try:
            hits = _hys_search(q)
        except urllib.error.HTTPError:  # the portal rejects some long/odd queries (406): try the next one
            continue
        for c in hits:
            cands.setdefault(c["id"], c.get("shortTitle", ""))
    if not cands:
        raise LookupError(f"No Have Your Say initiative found for CELEX {celex}.")
    tokens = {w for w in re.findall(r"[a-z]{4,}", title.lower()) if w not in _STOP}
    best, best_key = None, (-1, -1, 0)
    for iid in list(cands)[:12]:
        g = _hys_group(iid)
        refs = {_norm_com(p.get("reference", "")) for p in g.get("publications", [])}
        com_score = sum(coms[r] for r in refs & set(coms))
        blob = g.get("shortTitle", "") + " " + " ".join(p.get("title", "") for p in g.get("publications", []))
        score = len(tokens & {w for w in re.findall(r"[a-z]{4,}", blob.lower()) if w not in _STOP})
        key = (com_score, score, -float(iid))
        if key > best_key:
            best, best_key = str(int(float(iid))), key
    if best is None or (best_key[0] == 0 and best_key[1] < 2):
        raise LookupError(f"No Have Your Say initiative found for CELEX {celex}.")
    return best


def get_consultation_count(celex: str, up_to=None) -> int:
    """Number of public consultations for a CELEX (e.g. "52021PC0206" = AI Act -> 3).

    INPUT:  celex (str), up_to (optional "YYYY-MM-DD"/datetime; only rounds
            with publishedDate <= up_to count).
    OUTPUT: int.
    """
    pubs = _hys_group(_resolve_initiative_id(celex)).get("publications", [])
    if up_to is None:
        return len(pubs)
    cutoff = _to_datetime(up_to)
    return sum(1 for p in pubs if _to_datetime(p.get("publishedDate")) <= cutoff)


# ---------------------------------------------------------------------------
# FUNCTION 2: European Parliament votings summary
# ---------------------------------------------------------------------------
# INPUT:  celex (str, proposal CELEX e.g. "52021PC0206"), up_to (optional date, default all)
# OUTPUT: dict {n_votings, in_favor_increase, in_favor_variability, final_vote}
#   n_votings: total EP plenary votes found (AI Act -> 34)
#   in_favor_increase: last minus first in-favor share as a fraction
#     (10% -> 80% gives 0.7; negative if support fell)
#   in_favor_variability: 0-1 unpredictability of in-favor shares
#     (std / 0.5; 0 = stable, 1 = swinging between extremes)
#   final_vote: {voting_date, in_favor_pct, against_pct, abstentions_pct,
#     attendees, outcome} of the last vote (None if no votings)
# HOW:    proposal CELEX -> CELLAR procedure link -> EP Open Data procedure events +
#         plenary decisions (vote counts).

def _ep_json(url: str):
    req = urllib.request.Request(url, headers=_EP_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        if r.status == 204:
            return None
        txt = r.read().decode("utf-8").strip()
        return json.loads(txt) if txt else None


def _resolve_process_id(celex: str) -> str:
    """Find the EP procedure (e.g. "2021-0106") behind a CELEX (via CELLAR)."""
    celex = _normalize_celex(celex)
    m = re.search(r"/resource/procedure/(\d+)_(\d+)", _cellar_notice(celex))
    if not m:
        raise LookupError(f"No legislative procedure found for CELEX {celex}.")
    return f"{m.group(1)}-{int(m.group(2)):04d}"


def get_eu_votings(celex: str, up_to=None) -> dict:
    """EP votings summary for a CELEX (e.g. "52021PC0206" = AI Act).

    INPUT:  celex (str), up_to (optional "YYYY-MM-DD"/datetime; only votes
            with voting_date <= up_to count).
    OUTPUT: dict {n_votings, in_favor_increase, in_favor_variability, final_vote}.
    """
    pid = _resolve_process_id(celex)
    cutoff = _to_datetime(up_to) if up_to is not None else None

    events = (_ep_json(f"{_EP_API}/procedures/{pid}/events") or {}).get("data", [])
    sittings, reports = set(), set()
    for e in events:
        aid = e.get("activity_id", "")
        if aid.startswith("MTG-PL-"):
            sittings.add("-".join(aid.split("-")[:5]))
        for k in ("based_on_a_realization_of", "decided_on_a_realization_of"):
            for d in e.get(k) or []:
                if "/doc/A-" in d:
                    base = d.split("/")[-1]
                    reports.add(base)
                    p = base.split("-")
                    if len(p) == 4:
                        reports.add(f"{p[0]}{p[1]}-{p[3]}/{p[2]}")
    reports = [r for r in reports if r.startswith("A-") or r.startswith("A9")]

    votes = []
    for sitting in sorted(sittings):
        offset = 0
        while True:
            j = _ep_json(f"{_EP_API}/meetings/{sitting}/decisions?limit=50&offset={offset}")
            batch = (j or {}).get("data", [])
            if not batch:
                break
            for d in batch:
                labels = d.get("activity_label") or {}
                docs = json.dumps(d.get("decided_on_a_realization_of") or [])
                if not any(r in json.dumps(labels, default=str) or r in docs for r in reports):
                    continue
                try:
                    vdate = datetime.strptime(d.get("activity_date"), "%Y-%m-%d")
                except Exception:
                    continue
                if cutoff and vdate > cutoff:
                    continue
                fav = d.get("number_of_votes_favor") or 0
                ag = d.get("number_of_votes_against") or 0
                ab = d.get("number_of_votes_abstention") or 0
                total = fav + ag + ab
                votes.append({
                    "voting_date": vdate.strftime("%Y-%m-%d"),
                    "in_favor_pct": round(100 * fav / total, 2) if total else 0.0,
                    "against_pct": round(100 * ag / total, 2) if total else 0.0,
                    "abstentions_pct": round(100 * ab / total, 2) if total else 0.0,
                    "attendees": d.get("number_of_attendees"),
                    "outcome": (d.get("decision_outcome") or "").split("/")[-1] or None,
                })
            if len(batch) < 50:
                break
            offset += 50

    votes.sort(key=lambda x: x["voting_date"])
    if not votes:
        return {"n_votings": 0, "in_favor_increase": 0.0,
                "in_favor_variability": 0.0, "final_vote": None}
    shares = [v["in_favor_pct"] / 100 for v in votes]
    return {"n_votings": len(votes),
            "in_favor_increase": round(shares[-1] - shares[0], 4),
            "in_favor_variability": round(min(statistics.pstdev(shares), 0.5) / 0.5, 4) if len(shares) > 1 else 0.0,
            "final_vote": votes[-1]}


if __name__ == "__main__":
    print("consultations:", get_consultation_count("52021PC0206"))
    print("consultations up to 2021-01-01:", get_consultation_count("52021PC0206", "2021-01-01"))
    print(json.dumps(get_eu_votings("52021PC0206"), indent=2))
