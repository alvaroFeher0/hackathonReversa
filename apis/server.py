"""All the team's APIs behind one server, for the demo.

    uvicorn apis.server:app                 (from the repo root)
    open http://127.0.0.1:8000/docs         index; each API keeps its own docs:

    /eurlex/docs    Commission proposals from EUR-Lex (eurlex_api.py)
    /economy/docs   weekly European economy + EU Parliament context (parlament_economy.py)
    /news/docs      news headlines before a law + sentiment (getNewsFile.py)

law_history.py has no HTTP API: it is plain functions (consultations, EP votes), used by sources.py.
"""
from fastapi import FastAPI

from . import eurlex_api, parlament_economy
from .getNewsFile import create_app as news_app

app = FastAPI(title="Bill-to-Law APIs", description=__doc__)
app.mount("/eurlex", eurlex_api.app)
app.mount("/economy", parlament_economy.app)
app.mount("/news", news_app())


@app.get("/")
def index():
    return {name: f"/{name}/docs" for name in ("eurlex", "economy", "news")}
