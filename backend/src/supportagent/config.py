"""Settings, read from the environment and from one .env file.

The file is `.env` at the repo root, or the path in SUPPORTAGENT_ENV_FILE. A deployment has no
file and sets every value in the environment.

Nested settings use a double underscore: LLM__MODEL, LLM__BASE_URL, FALLBACK__MODEL, JUDGE__MODEL.

Two rules of the loader are easy to trip over:

- A nested value given in both places is merged key by key. Setting LLM__MODEL in the environment
  keeps every other LLM__ key of the .env file, including an LLM__EXTRA_BODY written for the old
  model, and LLM__EXTRA_BODY='{}' adds nothing and removes nothing.
- An empty value counts as not set.

To clear a value from the environment, set it to the word null: LLM__EXTRA_BODY=null,
LLM__BASE_URL=null. `python -m supportagent.cli.preflight` prints the models as they came out.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/supportagent/config.py -> repo root. Deployments set KB_DIR instead.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class ModelSettings(BaseModel):
    """One chat model behind an OpenAI-compatible endpoint."""

    model: str
    base_url: str | None = None  # defaults to OpenRouter
    api_key: SecretStr | None = None  # defaults to the OpenRouter key
    extra_body: dict[str, Any] | None = None  # for example {"reasoning": {"effort": "none"}}
    strict_tools: bool = False
    json_schema: bool = True  # false for an endpoint that rejects structured output
    # Dollars per million tokens. Without both, calls on this model report no cost.
    input_price: float | None = None
    output_price: float | None = None


def same_model(a: ModelSettings, b: ModelSettings, default_base_url: str) -> bool:
    """Whether two settings name one model at one endpoint.

    A setting without a base URL uses the default one, so `None` and that URL written out are
    the same place. A model must not grade its own answers, and this is what decides whether
    it would.
    """

    def where(spec: ModelSettings) -> str:
        return (spec.base_url or default_base_url).rstrip("/").lower()

    return a.model == b.model and where(a) == where(b)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_parse_none_str="null",
        extra="ignore",
    )

    database_url: str

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    deepgram_api_key: SecretStr | None = None
    # Keys the hashes of the PINs a conversation has used. See redact.session_key. Without it the
    # hashes are salted by the session only, which is enough for mock data and not for real PINs.
    secret_key: SecretStr | None = None

    llm: ModelSettings | None = None  # the model that runs the agent
    fallback: ModelSettings | None = None  # takes over when the first one fails
    judge: ModelSettings | None = None  # checks answers against evidence
    # Without a judge every answer goes out unchecked. Only a development setup may ask for that.
    allow_unchecked_answers: bool = False

    # How long a model call may take before it counts as failed. The defaults suit a hosted
    # model. A local model that has to load its weights first needs longer limits.
    llm_first_token_timeout_s: float = Field(default=8.0, gt=0)
    # Longest silence between two chunks of one stream.
    llm_stall_timeout_s: float = Field(default=10.0, gt=0)
    llm_total_timeout_s: float = Field(default=45.0, gt=0)  # one attempt
    llm_deadline_s: float = Field(default=90.0, gt=0)  # one call, with retries and fallback

    # Store the full prompt and response of every model call for the observability drawer.
    capture_prompts: bool = True

    kb_dir: Path = _REPO_ROOT / "kb"
    # A name that starts with "openrouter:" is served by OpenRouter, for example
    # openrouter:cohere/rerank-5-pro. Any other name is a small ONNX model run in this process.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    # Put in front of queries, not documents, for embedding models trained with an instruction.
    embed_query_instruction: str | None = None
    # Vector length to ask a hosted embedding model for. pgvector indexes at most 2,000.
    embed_dimensions: int | None = None

    # Scope gate thresholds on the reranker's score. They belong to `rerank_model` and must be
    # recalibrated with supportagent.evals.retrieval whenever the reranker changes.
    gate_low: float = -4.0
    gate_high: float = 2.0
    gate_gap: float = 3.0  # how much better another plan's passage has to score. See Gate.

    # Browser origins allowed to open the WebSocket. A page served from the backend's own host is
    # always allowed. Clients that send no Origin are not browsers. They are let in unless
    # `require_origin` is set.
    allowed_origins: list[str] = []
    require_origin: bool = False
    max_message_chars: int = Field(default=2_000, gt=0)
    max_sessions: int = Field(default=200, gt=0)  # live sessions in this process
    # What one visitor may use. A new conversation costs a copy of the mock CRM, a message costs
    # model calls. The address is the socket's, or the value of a header the host's proxy sets.
    max_new_sessions_per_hour: int = Field(default=30, gt=0)  # per address
    max_turns_per_session: int = Field(default=60, gt=0)
    # A turn that failed or was stopped can be picked up this many times. Every pick-up costs
    # model calls, so without a limit one message could be run for ever.
    max_retries_per_turn: int = Field(default=3, ge=0)
    max_connections_per_address: int = Field(default=12, gt=0)  # open sockets at one time
    # What every visitor together may spend on model calls in 24 hours, in dollars. At the limit
    # no new turn starts until old spend has aged out. None means no limit of our own.
    daily_budget_usd: float | None = Field(default=None, gt=0)
    client_address_header: str | None = None  # for example x-real-ip on Railway
    # Conversations and their CRM copies older than this are deleted, hourly. None keeps them.
    session_ttl_hours: float | None = Field(default=None, gt=0)
    hello_timeout_s: float = Field(default=10.0, gt=0)  # a new connection has to say hello
    # A connection that sends nothing for this long, with no turn running, is closed. The page
    # connects again when the visitor comes back.
    idle_timeout_s: float = Field(default=1800.0, gt=0)
    send_timeout_s: float = Field(default=10.0, gt=0)  # one event to one client
    # Lets a client queue simulated failures for its own session, to show how the app recovers.
    fault_injection: bool = False

    # Voice needs DEEPGRAM_API_KEY. Without it the app runs as before and offers no microphone.
    deepgram_base_url: str = "wss://api.deepgram.com"
    stt_model: str = "flux-general-en"
    tts_model: str = "flux-haley-en"  # the voice
    # Words the recogniser should expect. With "PIN" in the list it stops hearing "pen".
    stt_keyterms: list[str] = [
        "Fernhill",
        "PIN",
        "account number",
        "invoice",
        "refund",
        "Enterprise",
    ]
    voice_greeting: bool = True  # say hello when voice starts a conversation
    voice_max_seconds: float = Field(default=600.0, gt=0)  # one voice session
    voice_idle_seconds: float = Field(default=120.0, gt=0)  # nobody says anything
    max_voice_sessions: int = Field(default=4, gt=0)  # at one time, in this process
    # Deepgram's list prices, for the cost the drawer shows and the daily budget counts.
    stt_usd_per_minute: float = Field(default=0.0065, ge=0)
    tts_usd_per_1k_chars: float = Field(default=0.045, ge=0)

    @model_validator(mode="after")
    def _thresholds_in_order(self) -> Self:
        if not all(map(math.isfinite, (self.gate_low, self.gate_high, self.gate_gap))):
            raise ValueError("GATE_LOW, GATE_HIGH and GATE_GAP must be numbers")
        if self.gate_low > self.gate_high:
            raise ValueError("GATE_LOW must not be above GATE_HIGH")
        if self.gate_gap <= 0:
            raise ValueError("GATE_GAP must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    # The file is named here and not on the class, so the variable is read when settings are
    # first asked for and not when this module is imported.
    return Settings(_env_file=os.environ.get("SUPPORTAGENT_ENV_FILE", str(_REPO_ROOT / ".env")))
