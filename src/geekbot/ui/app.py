"""The entry script of the UI (step S4.4): connect to the server, then show the pages.

Started by ``geekbot up`` as ``python -m streamlit run src/geekbot/ui/app.py``; it can also be started
by hand while ``geekbot serve`` runs in another terminal. Streamlit runs this file from the top on
every interaction, so everything here is cheap: one ``/healthz`` call to see the server is still
there, and the page the user chose.

**Connecting** (``client.connect``): the key is read from the place the server keeps it, the server
is pinged, and one authenticated call proves the key is accepted. If any of that fails the page shows
one sentence that says what to do and a "Try again" button, and nothing else: no page can run without
a server, because a UI with no storage of its own has nothing to show.

``main`` is guarded by ``__name__ == "__main__"``, which Streamlit sets when it runs the script, so
a test can import this module without starting anything.
"""

import streamlit as st

from geekbot.config import Settings
from geekbot.ui.client import ClientError, GeekbotClient, connect
from geekbot.ui.views import chat, evals, knowledge, labels, settings

CLIENT_KEY = "geekbot_client"


def _client() -> GeekbotClient | None:
    """This session's client if its server still answers, else a new one, else ``None``.

    On ``None`` the reason has already been drawn on the page.
    """
    cached = st.session_state.get(CLIENT_KEY)
    if isinstance(cached, GeekbotClient) and cached.ping():
        return cached
    st.session_state.pop(CLIENT_KEY, None)
    try:
        client = connect(Settings())
    except ClientError as error:
        st.title("Geekbot")
        st.error(str(error))
        st.button("Try again")
        return None
    st.session_state[CLIENT_KEY] = client
    return client


def _run(client: GeekbotClient) -> None:
    def chat_page() -> None:
        chat.render(client)

    def knowledge_page() -> None:
        knowledge.render(client)

    def settings_page() -> None:
        settings.render(client)

    def evals_page() -> None:
        evals.render(client)

    def labels_page() -> None:
        labels.render(client)

    st.navigation(
        [
            st.Page(chat_page, title="Chat", url_path="chat", icon=":material/chat:", default=True),
            st.Page(
                knowledge_page, title="Knowledge", url_path="knowledge", icon=":material/folder:"
            ),
            st.Page(
                settings_page, title="Settings", url_path="settings", icon=":material/settings:"
            ),
            st.Page(evals_page, title="Evals", url_path="evals", icon=":material/monitoring:"),
            st.Page(labels_page, title="Labels", url_path="labels", icon=":material/rule:"),
        ]
    ).run()


def main() -> None:
    st.set_page_config(page_title="Geekbot", page_icon=":material/forum:", layout="wide")
    client = _client()
    if client is not None:
        _run(client)


if __name__ == "__main__":
    main()
