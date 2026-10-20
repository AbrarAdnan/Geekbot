"""The Streamlit user interface (step S4.4): a client of the API server, and nothing else.

**The rule.** The server is the one process that owns the storage (PLAN.md section 6.2). The UI is
another process, so everything it knows comes over HTTP from ``geekbot serve`` and it never opens
SQLite, Chroma or Ollama itself. ``import-linter`` enforces that in CI (``pyproject.toml``,
``[tool.importlinter]``): nothing under ``geekbot.ui`` may import the storage, ingest, retrieval,
agent, llm, server or CSV packages. The two Geekbot modules the UI may use are the ones written for
it: ``geekbot.apikey`` (find the local password) and ``geekbot.sse`` (read the answer stream), plus
``geekbot.config`` for the settings.

**Why the pages are buttons.** Streamlit runs the whole script again on every interaction. v1 put its
file uploader and its indexing in that script, so every click, a chat message included, indexed every
uploaded file again (v1 bug #1, PLAN.md section 5). Here nothing that costs anything runs on a plain
rerun: an upload, an ingest, a delete and a question each happen inside ``if st.button(...)`` or the
chat input's own branch, and what the user sees is read back from the server.

**The layout** (``streamlit run`` puts this folder first on ``sys.path``, so a module here would
shadow a library of the same name; the names are chosen to be unlikely ones)::

    app.py           the entry script: connect to the server, then the navigation
    client.py        GeekbotClient: HTTP, the key, the answer stream, wait for a job
    display.py       text and folding helpers with no Streamlit in them (so tests can run them)
    launch.py        the command line ``geekbot up`` starts Streamlit with
    views/           chat.py, knowledge.py, settings.py, evals.py: one function ``render`` each
"""

from pathlib import Path
from typing import Final

__all__ = ["APP_SCRIPT"]

# Where the entry script is, for ``geekbot up`` (it must not import ``app``: that imports Streamlit).
APP_SCRIPT: Final = Path(__file__).with_name("app.py")
