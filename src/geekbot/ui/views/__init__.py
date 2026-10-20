"""The pages of the UI (step S4.4). Each module has one function, ``render(client)``.

A page is handed the ``GeekbotClient`` and reads and changes everything through it. Pages import
Streamlit; the decisions they make (what to show for a job, what a stream adds up to) are functions in
``geekbot.ui.display`` so that tests can call them without running Streamlit.
"""
