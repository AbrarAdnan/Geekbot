"""Ingestion: turning files into searchable chunks (PLAN.md §6.4, §6.6).

S1.1 builds the first stage, ``loaders`` (file -> text), S1.2 the second, ``chunking`` (text ->
citable pieces), S1.3 the third, ``ids`` (a document and a chunk get an identity and a
fingerprint). The stores and the pipeline that joins them arrive in S1.4 - S1.6.
"""
