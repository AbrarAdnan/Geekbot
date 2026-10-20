"""All Geekbot settings, in one typed place (lesson: project/learn/02-config-cli.md).

Nothing else in the code base reads environment variables. Values come from ``GEEKBOT_*``
environment variables or from a local ``.env`` file (git-ignored; ``.env.example`` documents
the options), and pydantic rejects bad values at startup instead of halfway through a run.

Model profiles (PLAN.md §6.5, D-042, D-043): ``proto`` = light models for fast building,
``target`` = the models every reported number must come from. The default is ``proto`` because
speed of iteration matters most while building; the price is that a result produced without
setting ``GEEKBOT_PROFILE=target`` comes from the small models. Every report therefore records
the resolved model names, which ``geekbot config show`` prints.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["proto", "target"]
# A conversation's mode (PLAN.md §6.7): chat answers from the knowledge base, csv answers about
# one uploaded CSV. Each mode has its own model, and the two are never on the GPU together.
Mode = Literal["chat", "csv"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True)
class ModelSet:
    """The chat model for RAG mode and the model for CSV mode (they never share the GPU)."""

    chat: str
    csv: str


# Target = PLAN.md §1/§6.5 (D-007, D-037). Proto = D-042; the 3b is Gigalogy's own "fast" tier.
PROFILES: Final[dict[Profile, ModelSet]] = {
    "target": ModelSet(chat="gemma4:12b", csv="qwen2.5-coder:7b"),
    "proto": ModelSet(chat="gemma3:4b", csv="qwen2.5-coder:3b"),
}


@dataclass(frozen=True)
class Slos:
    """Latency targets from PLAN.md §6.5. Targets until measured, and measured on ``target``."""

    retrieval_rerank_p95_ms: int = 400
    first_turn_ttft_p50_s: float = 2.5
    first_turn_ttft_p95_s: float = 4.0
    followup_ttft_p95_s: float = 5.0
    end_to_end_p95_s: float = 15.0
    model_loads_during_queries: int = 0


# Fixed by the plan, so deliberately not environment-configurable: a bench that could be
# tuned by an env var could be tuned until it passes.
SLOS: Final = Slos()


class Settings(BaseSettings):
    """Geekbot's settings. Create it with ``Settings()``; it reads the environment itself.

    ``extra="ignore"`` lets a shared ``.env`` hold other tools' variables. The cost is that a
    typo such as ``GEEKBOT_PROFLE`` is silently ignored, which is why ``config show`` exists.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEEKBOT_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    profile: Profile = "proto"
    # Per-model overrides win over the profile.
    chat_model: str | None = None
    csv_model: str | None = None

    ollama_host: str = "http://127.0.0.1:11434"
    # Context sizes (PLAN §6.5: 8k for chat, 16k for the CSV package). Warm-up and real queries
    # must send the same value, or Ollama reloads the model. keep_alive: how long it stays loaded.
    chat_num_ctx: int = 8192
    csv_num_ctx: int = 16384
    keep_alive: str = "30m"
    # CSV mode (step C2, lesson 06 §5). Geekbot owns what touches Ollama and the clock; the
    # package's prompt and pipeline settings stay at the defaults Gigalogy measured. Both values
    # are starting points, not measurements: C3 checks how often the question cap would cut an
    # answer. The cap is soft (the package checks it before each attempt, not during a model
    # call), so the worst case is the cap plus one call.
    # The embedder (step S1.5, PLAN §6.4/§6.5): Ollama's ``nomic-embed-text`` on the CPU, so the
    # chat model keeps the whole GPU. nomic-embed-text was trained with task prefixes and
    # embeds worse without them (v1 bug #8); another embedder (the S2.6 bake-off) brings its
    # own. Changing any of these changes the embedder fingerprint, which blocks queries
    # until the dense index is rebuilt.
    embed_model: str = "nomic-embed-text"
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    # Whether the contextual header (title > headings) is embedded in front of each chunk. On by
    # default; the ablation's rung R5 switches it.
    embed_with_header: bool = True
    embed_batch_size: int = Field(default=32, gt=0)
    # The reranker (step S2.3): a small cross-encoder on the CPU (PLAN §6.5, §9). The default is
    # the 80 MB MiniLM model (Apache-2.0), one of the two the S2.6 bake-off compares. Its weights
    # are downloaded on first use into <data_dir>/models. The depth is how many fused candidates it
    # re-scores. 10 is the depth that (just) fits the 400 ms budget (D-077: retrieval about 330 ms at
    # p50 and 380 ms at p95 on an idle PC; 30 took about 1.07 s); its effect on recall is not measured
    # yet, S2.6 sweeps it. ``geekbot ask-timing`` times it on real candidates.
    # Keep it equal to ``retrieval.rerank.DEFAULT_RERANK_DEPTH`` (a test checks).
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = Field(default=10, gt=0)
    rerank_threads: int | None = Field(default=None, gt=0)
    # The answer graph (step S3.1, PLAN §6.5, §7). The cap on the reply, the sampling temperature
    # and the room for sources in the prompt are starting points, not measurements: S3.3 measures
    # speed and S5.1 sweeps the settings. ``chat_num_ctx`` above is what the model is warmed up
    # with and must also be what every query sends (R-2), which is why the chat model is built
    # from these settings and not from literals.
    chat_num_predict: int = Field(default=512, gt=0)
    # With thinking on (``geekbot ask --think``) the reasoning is generated tokens too, so the reply
    # cap has to leave room for it (step S3.3). An unmeasured guess: S3.3's live run shows how many
    # tokens a thinking model really spends, and only a model with a thinking mode can be tried.
    chat_think_num_predict: int = Field(default=2048, gt=0)
    chat_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # The citation relevance check (S3.5): a citation whose cross-encoder score against the sentence
    # that cites it is below this is shown as weak. -3.0 is a guess from the scale seen in S2.6 (the
    # best chunk of an answerable question scored a median of about 1 to 4, unrelated passages below
    # -4), not a measurement; the check only reports, it never changes an answer.
    cite_weak_below: float = -3.0
    context_budget_tokens: int = Field(default=3000, gt=0)
    # How many chunks the server's answer route asks retrieval for (step S4.2); `geekbot ask --top-k`
    # has the same default. A starting point, not a measurement.
    answer_top_k: int = Field(default=5, ge=1, le=20)
    # Conversation memory (step S3.4). The share of the prompt the earlier turns may take (whole
    # turns, newest first), how many messages a thread keeps, and the cap on the one model call that
    # rewrites a follow-up into a standalone question (PLAN §6.5: 96 tokens). Starting points, not
    # measurements.
    history_budget_tokens: int = Field(default=1000, gt=0)
    history_keep_messages: int = Field(default=40, ge=2)
    condense_num_predict: int = Field(default=96, gt=0)
    csv_llm_timeout_s: int = Field(default=60, gt=0)  # one Ollama call
    csv_question_time_limit_s: int = Field(default=60, gt=0)  # one question
    # CSV conversations on the server (step C5): a loaded CSV session holds its whole table in memory
    # (PLAN.md risk 20), so the registry keeps at most this many sessions and at most this many bytes
    # of table memory in all (pandas' own count), least recently used going first. Both are guesses,
    # not measurements: this PC also runs the user's other projects. A session that alone is over the
    # byte bound is refused at upload.
    csv_max_sessions: int = Field(default=4, ge=1)
    csv_session_budget_bytes: int = Field(default=500_000_000, gt=0)
    data_dir: Path = Path("data")
    log_level: LogLevel = "INFO"
    server_port: int = Field(default=8000, ge=1, le=65535)  # `geekbot serve` (S4.1)
    # The UI's own port (step S4.4): `geekbot up` starts Streamlit on it, 127.0.0.1 only. 8501 is
    # Streamlit's default.
    ui_port: int = Field(default=8501, ge=1, le=65535)
    # The perimeter (step S4.3, PLAN §8.1). The server asks every request (but /healthz) for this key.
    # It is a local password between the server and the UI, not a cloud service's key.
    # Unset, `geekbot serve` makes one and keeps it in <data_dir>/api-key (geekbot.apikey), so the
    # setting is only for a key you want to choose; at least 24 characters.
    api_key: SecretStr | None = None
    # Folders the server may ingest from, besides its own uploads folder (<data_dir>/uploads), as
    # full paths separated by ";" on Windows and ":" elsewhere (os.pathsep). Empty by default: the
    # server then reads nothing but what was uploaded to it. Write it in `.env` WITHOUT quotes: a
    # double-quoted value has its backslash sequences interpreted (C:\notes holds a line break).
    allowed_roots: str = ""

    @property
    def allowed_root_paths(self) -> tuple[Path, ...]:
        """``allowed_roots`` as paths, in order; blank entries (a trailing ";") are dropped."""
        parts = (part.strip() for part in self.allowed_roots.split(os.pathsep))
        return tuple(Path(part) for part in parts if part)

    @property
    def resolved_chat_model(self) -> str:
        """The chat model in use: the override if set, else the profile's."""
        return self.chat_model or PROFILES[self.profile].chat

    @property
    def resolved_csv_model(self) -> str:
        """The CSV-mode model in use: the override if set, else the profile's."""
        return self.csv_model or PROFILES[self.profile].csv


def mask_settings(settings: Settings) -> dict[str, object]:
    """What ``geekbot config show`` prints: every setting, safe to paste into a chat.

    ✍️ YOUR PIECE (step S0.2). Return a dict with exactly these keys, in this order:

    - ``profile``: the profile name (str)
    - ``chat_model``: the *resolved* chat model, not the raw override (str)
    - ``csv_model``: the *resolved* CSV model (str)
    - ``ollama_host``: str
    - ``data_dir``: the path as a str (JSON can't hold a ``Path``)
    - ``log_level``: str
    - ``api_key``: ``"********"`` if a secret is set, ``None`` if it is not

    Why mask: ``SecretStr`` hides itself in ``repr``, but ``.get_secret_value()`` returns the
    real text. The value must never reach this dict, a log line, or the screen.
    """
    return {
        "profile": settings.profile,
        "chat_model": settings.resolved_chat_model,
        "csv_model": settings.resolved_csv_model,
        "ollama_host": settings.ollama_host,
        "data_dir": str(settings.data_dir),
        "log_level": settings.log_level,
        "api_key": "********" if settings.api_key is not None else None,
    }
