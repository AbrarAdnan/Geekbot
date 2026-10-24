# ADR-0002: SQLite is the source of truth, Chroma is derived, one process owns the storage

- **Status:** accepted (PLAN.md §6.2 to §6.4; D-019 to D-022)
- **Date:** 2026-10-07 (built in Phase 1 and S4.1)

## Context
v1 re-embedded every document on every run and could not say what was indexed. A retrieval system has
two stores that must agree: the text (and its full-text index) and the vectors. Two writers, or one
store trusted over the other, produce documents that exist in one and not in the other.

## Decision
- **SQLite with FTS5 holds every document, chunk and state** (`pending`, `embedding`, `ready`, `failed`
  in an outbox). It is the source of truth. **Chroma holds only vectors**, derived from SQLite, and can be
  rebuilt from it (`geekbot ingest --reindex`).
- A document becomes searchable only when it is `ready`, so a crash between the two writes leaves an
  *unfinished* document that `geekbot doctor --repair` finishes, never a half-visible one.
- **One process owns the storage** (`geekbot serve`), holding an OS-level lock; the Streamlit UI is an
  HTTP client and cannot import the storage (an import-linter contract enforces it in CI). A second
  `serve` is refused with one line. Blocking work never runs on the event loop (worker threads), which a
  heartbeat test checks while an answer streams.
- Chunk and document IDs are content hashes, so an unchanged file embeds nothing the second time.

## Consequences
- Re-ingesting is incremental; a hard-killed ingest is healed, not rebuilt (Gate A, part B).
- The server is the only door; every client (UI, CLI, MCP) goes through it or opens the workspace under
  the same lock.
- Changing the embedder invalidates the vectors (a fingerprint blocks queries until a rebuild).

## Evidence
`project/steps/GA-gate-a-review.md` (ingest twice: second run 0 embeddings; the kill-and-repair
exercise; second server refused), `tests/unit/test_server.py` (heartbeat), `pyproject.toml`
(import-linter contracts).
