import os
import time
import shutil
import sqlite3
from pathlib import Path
from typing import List

import streamlit as st
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredHTMLLoader, NotebookLoader,
    UnstructuredMarkdownLoader, JSONLoader, UnstructuredExcelLoader, PythonLoader,
)
from langchain_community.vectorstores import Chroma
from storage_utils import (
    normalize_source_path, METADATA_SOURCE_KEY, split_documents,
    calculate_chunk_ids, add_to_chroma, delete_from_chroma_by_source,
    list_sources_in_chroma
)
from rag_pipeline import query_rag
from get_embedding_function import get_embedding_function

APP_TITLE = "Geekbot"
DB_FILE = "rag_ui.sqlite"
CHROMA_PATH = Path("chroma") / "default_chat"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

RESET_MARKER = CHROMA_PATH / ".reset_required"

def auto_reset_chroma():
    if RESET_MARKER.exists():
        print("🧹 Reset marker found. Forcing Chroma DB cleanup...")
        try:
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            print("✅ Chroma DB folder removed.")
        except Exception as e:
            print(f"⚠️ Failed to remove Chroma DB: {e}")
        finally:
            RESET_MARKER.unlink(missing_ok=True)
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        print("✅ Fresh Chroma DB folder created.")

# Run auto-reset check on startup
auto_reset_chroma()

SCHEMA = {
    "messages": """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          ts REAL NOT NULL
        )
    """,
    "settings": """
        CREATE TABLE IF NOT EXISTS settings (
          key TEXT PRIMARY KEY,
          value TEXT
        )
    """
}


def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

with get_conn() as c:
    cur = c.cursor()
    for sql in SCHEMA.values():
        cur.execute(sql)
    c.commit()

SUPPORTED_LOADERS = {
    ".py": PythonLoader,
    ".pdf": PyPDFLoader,
    ".txt": lambda f: TextLoader(f, encoding="utf-8"),
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".ipynb": NotebookLoader,
    ".json": JSONLoader,
}


def load_docs_for_path(path: Path) -> List[Document]:
    docs = []
    if path.is_file():
        loader_class = SUPPORTED_LOADERS.get(path.suffix.lower())
        if not loader_class:
            return []
        loader = loader_class(str(path)) if not callable(loader_class) else loader_class(str(path))
        file_docs = loader.load()
        normalized = normalize_source_path(str(path))
        for d in file_docs:
            d.metadata.update({
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                METADATA_SOURCE_KEY: normalized,
                "page": d.metadata.get("page", 0),
            })
        return file_docs
    for root, _, files in os.walk(path):
        for fname in files:
            docs.extend(load_docs_for_path(Path(root) / fname))
    return docs

# Database operations
def save_message(role, content):
    with get_conn() as c:
        c.execute(
            "INSERT INTO messages (role,content,ts) VALUES (?,?,?)",
            (role, content, time.time()),
        )
        c.commit()


def list_messages():
    with get_conn() as c:
        return c.execute("SELECT * FROM messages ORDER BY ts ASC").fetchall()


def clear_messages():
    with get_conn() as c:
        c.execute("DELETE FROM messages")
        c.commit()


def set_setting(key, value):
    with get_conn() as c:
        c.execute("INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)", (key, str(value)))
        c.commit()


def get_setting(key, default=None):
    with get_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


# File indexing
def index_path(path: str):
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        return 0, 0

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    # Remove existing entries for this source to prevent duplicates
    removed = delete_from_chroma_by_source(CHROMA_PATH, str(path_obj))
    if removed:
        print(f"Removed {removed} old chunks before reindexing {path_obj}")

    docs = load_docs_for_path(path_obj)
    if not docs:
        return 0, 0
    chunks = split_documents(docs)
    added = add_to_chroma(chunks, CHROMA_PATH)
    return len(docs), added


# def clear_chroma():
#     """Delete the entire Chroma DB for the default chat."""
#     if CHROMA_PATH.exists():
#         shutil.rmtree(CHROMA_PATH)
#     CHROMA_PATH.mkdir(parents=True, exist_ok=True)

def clear_chroma():
    """Clear the Chroma DB safely (handles Windows file locks)."""
    try:
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function())
        ids = db.get()["ids"]
        if ids:
            db.delete(ids=ids)
            print(f"🗑️ Deleted {len(ids)} docs from Chroma via API")
        else:
            print("ℹ️ No docs in Chroma to delete.")
        del db  # release handle
    except Exception as e:
        print(f"⚠️ Could not clear via Chroma API: {e}")

    try:
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
            print("✅ Chroma folder removed.")
    except PermissionError:
        print("⚠️ Could not delete Chroma folder (locked). Will reset on restart.")
        RESET_MARKER.write_text("reset required")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    print("✅ New empty Chroma folder initialized.")


# Streamlit UI
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    # st.subheader("Assistant Settings")

    # Toggle web search
    # use_web = st.checkbox("Enable Web Search", value=bool(int(get_setting("use_web", 0))))
    # set_setting("use_web", int(use_web))
    # print(use_web)

    st.markdown("---")

    # File/folder uploader
    st.subheader("Upload data")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=list(SUPPORTED_LOADERS.keys()),
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            temp_path = Path("uploaded") / f.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as out:
                out.write(f.getbuffer())
            n_docs, n_chunks = index_path(str(temp_path))
            st.success(f"Indexed {n_docs} docs, added {n_chunks} chunks from {f.name}")

    folder_path = st.text_input("Or enter a folder path to index")
    if st.button("Index folder"):
        n_docs, n_chunks = index_path(folder_path)
        st.success(f"Indexed {n_docs} docs, added {n_chunks} chunks")

    if st.button("Show indexed sources"):
        sources = list_sources_in_chroma(CHROMA_PATH)
        if sources:
            st.table(sources)  # ✅ prettier than raw JSON
        else:
            st.info("No sources indexed yet.")

    if st.button("Clear chat history"):
        clear_messages()
        st.success("Chat history cleared.")
        st.rerun()

    if st.button("Clear indexed data"):
        clear_chroma()
        st.success("All indexed data cleared.")
        st.rerun()


msgs = list_messages()
for m in msgs:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Main Chat Function
user_input = st.chat_input("Ask your question…")
if user_input:
    save_message("user", user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply, src_info = query_rag(user_input, str(CHROMA_PATH), use_web=True)
            st.markdown(reply)

            # ✅ show references neatly
            if src_info.get("local_sources"):
                st.caption("Local Sources:")
                for src in src_info["local_sources"]:
                    st.caption(f"- {src}")
    save_message("assistant", reply)
    st.rerun()
