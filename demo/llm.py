"""
LLM explanation of a prediction: 2 reasons it may pass, 2 it may not, 2 for the timing.

The LLM only explains predictions already made; nothing it says feeds the models.

Configuration (environment variables, or the same keys in .streamlit/secrets.toml):
    GROK_API_KEY   xAI key; if set (and LLM_API_KEY is not), the provider defaults to "grok"
    LLM_PROVIDER   "grok", "anthropic" or "openai" (any OpenAI-compatible chat completions API)
    LLM_API_KEY    the provider's API key; without any key the demo shows the predictions without reasons
    LLM_MODEL      model id (default: grok-4 / claude-sonnet-5-5 / gpt-4o-mini)
    LLM_BASE_URL   optional, for OpenAI-compatible providers (default https://api.x.ai/v1 for grok)
Variables in the repo's .env file are loaded too (the real environment wins).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

PROMPT = (
    "These are the values of the current state of an eu law and the values of our prediction of how an EU law "
    "is going to end up, please dont use any information newer than {query_date} and give me only 2 short "
    "reasons (1 sentence) why the law could pass and give me 2 short reasons why its possible that it wont "
    "pass and 2 reasons why it will take this long, thats the only thing you should output"
)
FORMAT = ('\n\nAnswer as JSON only: {"pass": ["...", "..."], "fail": ["...", "..."], "timing": ["...", "..."]}'
          "\n\nData:\n")
DEFAULT_MODELS = {"grok": "grok-4", "anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
BASE_URLS = {"grok": "https://api.x.ai/v1", "openai": "https://api.openai.com/v1"}


def _dotenv() -> dict:
    """KEY=value lines of the repo's .env (no dependency on python-dotenv)."""
    if not ENV_FILE.exists():
        return {}
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def config(secrets: dict | None = None) -> dict:
    env = _dotenv()
    get = lambda k, d="": os.environ.get(k) or (secrets or {}).get(k) or env.get(k) or d
    provider = get("LLM_PROVIDER", "grok" if get("GROK_API_KEY") and not get("LLM_API_KEY") else "anthropic").lower()
    key = get("LLM_API_KEY") or (get("GROK_API_KEY") if provider == "grok" else "")
    return {"provider": provider, "key": key, "model": get("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            "base_url": get("LLM_BASE_URL", BASE_URLS.get(provider, BASE_URLS["openai"])).rstrip("/")}


def _call(cfg: dict, prompt: str) -> str:
    if cfg["provider"] == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=90, headers={
            "x-api-key": cfg["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": cfg["model"], "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    r = requests.post(f"{cfg['base_url']}/chat/completions", timeout=90,
                      headers={"Authorization": f"Bearer {cfg['key']}"},
                      json={"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def explain(data: dict, query_date: str, cfg: dict) -> dict[str, list[str]]:
    """{"pass": [2], "fail": [2], "timing": [2]}. Raises on API errors or an unreadable answer."""
    prompt = PROMPT.format(query_date=query_date) + FORMAT + json.dumps(data, indent=1, default=str)
    text = _call(cfg, prompt)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM answer is not JSON: {text[:200]}")
    out = json.loads(m.group(0))
    return {k: [str(s) for s in out.get(k, [])][:2] for k in ("pass", "fail", "timing")}
