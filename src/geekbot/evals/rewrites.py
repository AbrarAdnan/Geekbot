"""Cached rewrites of the golden set's follow-up questions (step S2.6, D-076).

A multi-turn golden item ("And the logs that only record who used it?") cannot be searched as it
stands; in the product the ``condense`` step rewrites it into a question that stands alone. To score
retrieval on those items (D-065, D-076) the evaluation needs the same rewrite, and it needs it to be
**the same on every rung and every rerun**. Otherwise a difference between two rungs could be a
difference between two rewrites. So the rewrite is made once, by the product's own ``condense``
code, and kept in a file.

**The cache key** (D-076) is a hash of: the item (its id, its follow-up and the earlier turns), the
rewrite model's name *and digest* (a re-pulled model may rewrite differently), the text of
``prompts/condense.md`` and the JSON schema of the answer. Change any of them and the old rewrite is
not found, so a stale rewrite can never be scored as if it were current. The model is named in the
file beside every rewrite, and the report names it: a retrieval number on multi-turn items is a
number about that model's rewriting too.

The earlier turns of a golden item are the user's earlier questions only (the file has no
assistant answers), so the rewrite sees a conversation of questions. That is a simplification the
report states; a real chat would also contain the assistant's replies.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geekbot.agent.condense import (
    CONDENSE_PROMPT,
    CondenseOutput,
    build_condense_messages,
    clean_standalone,
)
from geekbot.agent.state import HistoryMessage
from geekbot.evals.datasets import GoldenItem
from geekbot.llm.chat import StructuredChat

__all__ = [
    "REWRITE_CACHE_VERSION",
    "RewriteCache",
    "cached_rewrites",
    "history_of",
    "rewrite_followups",
    "rewrite_key",
]

REWRITE_CACHE_VERSION: Final = 1


def history_of(item: GoldenItem) -> list[HistoryMessage]:
    """The item's earlier turns as the history ``condense`` expects (user messages only)."""
    return [
        HistoryMessage(id=f"t{number}-user", role="user", content=turn, labels=["USER"])
        for number, turn in enumerate(item.earlier_turns, start=1)
    ]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rewrite_key(item: GoldenItem, *, model: str, digest: str) -> str:
    """The cache key of one item under one rewrite model (see the module docstring)."""
    payload = {
        "item": [item.id, item.question, list(item.earlier_turns)],
        "model": [model, digest],
        "prompt": _sha(CONDENSE_PROMPT),
        "schema": _sha(json.dumps(CondenseOutput.model_json_schema(), sort_keys=True)),
        "version": REWRITE_CACHE_VERSION,
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))


@dataclass
class RewriteCache:
    """A JSON file of ``{key: {item, model, digest, question, rewrite}}``. Written after every add."""

    path: Path
    entries: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: Path) -> "RewriteCache":
        """The cache at ``path``; empty if the file does not exist. ``ValueError`` if it is damaged."""
        if not path.is_file():
            return cls(path, {})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data["entries"]
            if not isinstance(entries, dict):
                raise TypeError("entries is not an object")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} is not a rewrite cache: {error}") from error
        return cls(path, {str(key): dict(value) for key, value in entries.items()})

    def get(self, key: str) -> str | None:
        entry = self.entries.get(key)
        return None if entry is None else entry["rewrite"]

    def put(self, key: str, item: GoldenItem, *, model: str, digest: str, rewrite: str) -> None:
        self.entries[key] = {
            "item": item.id,
            "model": model,
            "digest": digest,
            "question": item.question,
            "rewrite": rewrite,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            {"version": REWRITE_CACHE_VERSION, "entries": self.entries},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        partial = self.path.with_suffix(self.path.suffix + ".part")
        partial.write_text(text, encoding="utf-8")
        partial.replace(self.path)


def rewrite_followups(
    items: Iterable[GoldenItem],
    condenser: StructuredChat,
    cache: RewriteCache,
    *,
    model: str,
    digest: str,
    budget_tokens: int,
    on_item: Callable[[str, str, bool], None] | None = None,
) -> dict[str, str]:
    """``{item id: standalone question}`` for every multi-turn item, calling the model only on misses.

    Uses the product's own message builder and cleaner (``build_condense_messages``,
    ``clean_standalone``), so the evaluation rewrites exactly as a conversation would. A reply the
    cleaner rejects falls back to the follow-up as typed, as in the product, and that fallback is
    cached too (so the same weak rewrite is scored every time). ``on_item(id, rewrite, cached)`` is
    called for progress.
    """
    found: dict[str, str] = {}
    for item in items:
        if item.type != "multi_turn":
            continue
        key = rewrite_key(item, model=model, digest=digest)
        hit = cache.get(key)
        if hit is not None:
            found[item.id] = hit
            if on_item is not None:
                on_item(item.id, hit, True)
            continue
        messages = build_condense_messages(
            item.question, history_of(item), budget_tokens=budget_tokens
        )
        reply = condenser.invoke(messages)
        if not isinstance(reply, CondenseOutput):
            raise TypeError(f"expected a CondenseOutput, got {type(reply).__name__}")
        rewrite = clean_standalone(reply.standalone_question, fallback=item.question)
        cache.put(key, item, model=model, digest=digest, rewrite=rewrite)
        found[item.id] = rewrite
        if on_item is not None:
            on_item(item.id, rewrite, False)
    return found


def cached_rewrites(
    items: Iterable[GoldenItem], cache: RewriteCache, *, model: str, digest: str
) -> Mapping[str, str]:
    """The rewrites already in the cache for these items (no model call; missing items are absent)."""
    found: dict[str, str] = {}
    for item in items:
        if item.type == "multi_turn":
            hit = cache.get(rewrite_key(item, model=model, digest=digest))
            if hit is not None:
                found[item.id] = hit
    return found
