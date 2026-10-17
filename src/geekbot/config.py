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
    # re-scores: a starting point, not a measurement (``geekbot rerank-timing`` measures it).
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = Field(default=30, gt=0)
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
    context_budget_tokens: int = Field(default=3000, gt=0)
    # Conversation memory (step S3.4). The share of the prompt the earlier turns may take (whole
    # turns, newest first), how many messages a thread keeps, and the cap on the one model call that
    # rewrites a follow-up into a standalone question (PLAN §6.5: 96 tokens). Starting points, not
    # measurements.
    history_budget_tokens: int = Field(default=1000, gt=0)
    history_keep_messages: int = Field(default=40, ge=2)
    condense_num_predict: int = Field(default=96, gt=0)
    csv_llm_timeout_s: int = Field(default=60, gt=0)  # one Ollama call
    csv_question_time_limit_s: int = Field(default=60, gt=0)  # one question
    data_dir: Path = Path("data")
    log_level: LogLevel = "INFO"
    # Unused until the server's auth (S4.3). It exists now so secret masking has a real secret.
    api_key: SecretStr | None = None

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
