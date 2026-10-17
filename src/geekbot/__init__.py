"""Geekbot v2: a local RAG engine that measures retrieval quality, controls inference cost,
and treats retrieved content as untrusted data (project/PLAN.md §3).
"""

from importlib.metadata import version

# Read from the installed package metadata, so pyproject.toml is the only place the version
# is written. This needs an installed package, which the src layout guarantees anyway:
# see project/learn/01-packaging.md §1.
__version__: str = version("geekbot")
