"""The settings page (step S4.4): what the server and this UI are running with. Read-only.

Nothing here changes a setting. Geekbot's settings are environment variables and the ``.env`` file
(``geekbot.config``); a running server has read them once, at start-up, and a model change needs a
restart anyway (the chat model is loaded onto the GPU before the server listens). So the page shows
the truth and says where to change it, and does not pretend that a form could.

Two sources, and the page keeps them apart because they can differ: **the server's own report**
(``/v1/health``: the model it loaded, the embedder it opened, what it holds) and **this UI process's
settings** (read the same way, from the same ``.env``, at the UI's start). Under ``geekbot up`` they
are the same; a server started by hand with other variables would show the difference here.

The API key is never shown, only where it came from.
"""

import streamlit as st

from geekbot.config import Settings, mask_settings
from geekbot.ui.client import ClientError, GeekbotClient
from geekbot.ui.display import health_rows

__all__ = ["render"]


def render(client: GeekbotClient) -> None:
    st.title("Settings")
    try:
        health = client.health()
    except ClientError as error:
        st.error(str(error))
        return
    st.subheader("The server")
    st.table(health_rows(health))
    st.subheader("This UI")
    st.table(
        [
            {"Setting": "Server address", "Value": client.base_url},
            {"Setting": "API key from", "Value": client.key_origin or "(not known)"},
        ]
    )
    st.subheader("Settings this UI read")
    st.json(mask_settings(Settings()))
    st.caption(
        "Read-only. To change a setting, edit `.env` (see `.env.example`) and restart `geekbot up`. "
        "`GEEKBOT_PROFILE=target` switches to the target models; the chat model is loaded again "
        "on restart."
    )
