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

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/supportagent/config.py -> repo root. Deployments set KB_DIR instead.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class ModelSettings(BaseModel):
    """One chat model behind an OpenAI-compatible endpoint."""

    model: str
    base_url: str | None = None  # defaults to the AI Gateway
    api_key: SecretStr | None = None  # defaults to the AI Gateway key
    extra_body: dict[str, Any] | None = None  # for example {"reasoning_effort": "none"}
    strict_tools: bool = False
    json_schema: bool = True  # false for an endpoint that rejects structured output
    # Dollars per million tokens. Without both, calls on this model report no cost.
    input_price: float | None = None
    output_price: float | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_parse_none_str="null",
        extra="ignore",
    )

    database_url: str

    ai_gateway_api_key: SecretStr | None = None
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"
    deepgram_api_key: SecretStr | None = None

    llm: ModelSettings | None = None  # the model that runs the agent
    fallback: ModelSettings | None = None  # takes over when the first one fails
    judge: ModelSettings | None = None  # checks answers against evidence
    # Without a judge every answer goes out unchecked. Only a development setup may ask for that.
    allow_unchecked_answers: bool = False

    # How long a model call may take before it counts as failed. The defaults suit a hosted
    # model. A local model that has to load its weights first needs longer limits.
    llm_first_token_timeout_s: float = 8.0
    llm_total_timeout_s: float = 45.0  # one attempt
    llm_deadline_s: float = 90.0  # one call, with its retries and its fallback

    # Store the full prompt and response of every model call for the observability drawer.
    capture_prompts: bool = True

    kb_dir: Path = _REPO_ROOT / "kb"
    # A name that starts with "gateway:" is served by the AI Gateway, for example
    # gateway:cohere/rerank-v5-fast. Any other name is a small ONNX model run in this process.
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
    max_message_chars: int = 2_000
    max_sessions: int = 200  # live sessions in this process
    hello_timeout_s: float = 10.0  # how long a new connection has to say hello
    # Lets a client queue simulated failures for its own session, to show how the app recovers.
    fault_injection: bool = False

    @model_validator(mode="after")
    def _thresholds_in_order(self) -> Self:
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
