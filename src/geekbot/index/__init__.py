"""The stores behind retrieval (PLAN.md §6.2, §6.4, §6.6).

S1.4 builds the first one, ``sqlite`` (documents, chunks and the FTS5 index: the source of truth)
and the ``lock`` that makes one process its only owner. The dense index (Chroma) arrives in S1.5
and the code that keeps the two in step in S1.6.
"""
