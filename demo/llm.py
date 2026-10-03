"""
LLM explanation of a prediction: 2 reasons it may pass, 2 it may not, 2 for the timing.

The LLM only explains predictions already made; nothing it says feeds the models.

Configuration (environment variables, or the same keys in .streamlit/secrets.toml):
    LLM_PROVIDER   "anthropic" (default) or "openai" (any OpenAI-compatible chat completions API)
    LLM_API_KEY    the provider's API key; without it the demo shows the predictions without reasons
    LLM_MODEL      model id (default: claude-sonnet-5-5 / gpt-4o-mini)
    LLM_BASE_URL   optional, for OpenAI-compatible providers (default https://api.openai.com/v1)
"""
from __future__ import annotations

import json
import os
import re

import requests

PROMPT = (
    "These are the values of the current state of an eu law and the values of our prediction of how an EU law "
    "is going to end up, please dont use any information newer than {query_date} and give me only 2 short "
    "reasons (1 sentence) why the law could pass and give me 2 short reasons why its possible that it wont "
    "pass and 2 reasons why it will take this long, thats the only thing you should output"
)
FORMAT = ('\n\nAnswer as JSON only: {"pass": ["...", "..."], "fail": ["...", "..."], "timing": ["...", "..."]}'
          "\n\nData:\n")
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


def config(secrets: dict | None = None) -> dict:
    get = lambda k, d="": os.environ.get(k) or (secrets or {}).get(k) or d
    provider = get("LLM_PROVIDER", "anthropic").lower()
    return {"provider": provider, "key": get("LLM_API_KEY"), "model": get("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            "base_url": get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")}


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
