from pathlib import Path
from typing import List, Dict
from filelock import FileLock
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

METADATA_SOURCE_KEY = "source"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


def normalize_source_path(p: str) -> str:
    """Return canonical absolute path for consistent metadata storage."""
    if p is None:
        return ""
    return str(Path(p).expanduser().resolve())


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Create deterministic chunk IDs using 'source' and page metadata.
    IDs: "<abs-source>:<page>:<chunk_index>"
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get(METADATA_SOURCE_KEY) or ""
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def split_documents(documents: List[Document],
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Split using the shared TEXT_SPLITTER (keeps chunking consistent)."""
    # If caller passed different sizes, we could build a new splitter, but keep defaults.
    return TEXT_SPLITTER.split_documents(documents)


def add_to_chroma(chunks: List[Document], persist_dir: Path, batch_size: int = 2000) -> int:
    if not chunks:
        return 0

    # normalize and compute ids
    for c in chunks:
        src = c.metadata.get(METADATA_SOURCE_KEY)
        if src:
            c.metadata[METADATA_SOURCE_KEY] = normalize_source_path(src)

    chunks_with_ids = calculate_chunk_ids(chunks)

    persist_dir.mkdir(parents=True, exist_ok=True)
    db = Chroma(persist_directory=str(persist_dir), embedding_function=get_embedding_function())

    lock_path = persist_dir / ".chroma_lock"
    lock = FileLock(str(lock_path) + ".lock")

    with lock:
        try:
            existing = db.get(include=["ids"])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            existing_ids = set()

        new_chunks = [c for c in chunks_with_ids if c.metadata.get("id") not in existing_ids]
        if not new_chunks:
            return 0

        new_ids = [c.metadata["id"] for c in new_chunks]
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]
            db.add_documents(batch_chunks, ids=batch_ids)

        # persist to disk
        try:
            db.persist()
        except Exception:
            # sometimes chroma driver manages persistence; ignore errors here
            pass

    return len(new_chunks)


def delete_from_chroma_by_source(persist_dir: Path, source_path: str) -> int:
    db = Chroma(persist_directory=str(persist_dir), embedding_function=get_embedding_function())
    normalized = normalize_source_path(source_path)

    lock_path = persist_dir / ".chroma_lock"
    lock = FileLock(str(lock_path) + ".lock")

    with lock:
        try:
            all_docs = db.get(include=["metadatas", "ids"])
            if not all_docs or "ids" not in all_docs:
                return 0

            is_dir = Path(normalized).is_dir()
            ids_to_delete = []
            for doc_id, meta in zip(all_docs.get("ids", []), all_docs.get("metadatas", [])):
                src = meta.get(METADATA_SOURCE_KEY, "")
                if not src:
                    continue
                src_norm = normalize_source_path(src)
                if is_dir:
                    if src_norm.startswith(normalized):
                        ids_to_delete.append(doc_id)
                else:
                    if src_norm == normalized:
                        ids_to_delete.append(doc_id)

            if not ids_to_delete:
                return 0

            db.delete(ids=ids_to_delete)
            try:
                db.persist()
            except Exception:
                pass
            return len(ids_to_delete)
        except Exception:
            return 0


def list_sources_in_chroma(persist_dir: Path) -> Dict[str, int]:
    db = Chroma(persist_directory=str(persist_dir), embedding_function=get_embedding_function())
    try:
        data = db.get(include=["metadatas", "ids"])
        sources: Dict[str, int] = {}
        for doc_id, meta in zip(data.get("ids", []), data.get("metadatas", [])):
            src = meta.get(METADATA_SOURCE_KEY)
            if src:
                n = normalize_source_path(src)
                sources[n] = sources.get(n, 0) + 1
        return sources
    except Exception:
        return {}
