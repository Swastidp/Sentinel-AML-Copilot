"""The only place in the codebase that reads the environment.

Every other module receives a `Settings` instance (or a value from one) by argument or by calling
`get_settings()`. Nothing else calls `os.environ`. That single rule buys three things:

* **Testability** — a test constructs `Settings(_env_file=None, ...)` with whatever it needs and
  never touches the real environment.
* **One place to audit** — reviewing what this system can reach means reading one file.
* **Startup-time failure** — a malformed value fails here, on a Pydantic validator, instead of
  three layers down inside a driver call.

Credentials are typed `SecretStr`, so a stray `log.info("cfg", settings=s)` or an unhandled
traceback prints `SecretStr('**********')` rather than the value. The redaction processor in
`config/logging.py` is the second line of defence; this is the first.

Connection secrets are *optional* fields with required *accessors*. `Settings()` therefore
constructs successfully with no `.env` at all — which is what lets `make check` run in CI without
any cloud credentials — while `settings.postgres_dsn` raises `MissingConfigurationError` naming the
exact variable the moment an adapter actually needs it.
"""

from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentinel.domain.exceptions import ConfigurationError, MissingConfigurationError

# Placeholder values shipped in .env.example. Reaching production with one of these is a
# configuration bug, not a default, so it is rejected rather than silently accepted.
_PLACEHOLDERS = frozenset(
    {
        "replace-me",
        "replace-with-a-long-random-string",
        "sk-or-v1-replace-me",
    }
)


def _require(value: SecretStr | None, env_key: str) -> str:
    """Unwrap a secret that must be present, or fail naming the variable."""
    if value is None:
        raise MissingConfigurationError(env_key)
    raw = value.get_secret_value()
    if not raw.strip():
        raise MissingConfigurationError(env_key)
    if raw in _PLACEHOLDERS:
        raise ConfigurationError(
            f"{env_key} is still set to the .env.example placeholder value. "
            f"Replace it with a real value."
        )
    return raw


class Settings(BaseSettings):
    """Typed, validated configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unrelated vars in the ambient environment
    )

    # --- runtime -----------------------------------------------------------------------------
    app_env: Literal["local", "ci", "cloud"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # --- Neon Postgres (system of record) ----------------------------------------------------
    database_url: SecretStr | None = None

    # --- Neo4j AuraDB (fund-flow graph) ------------------------------------------------------
    # On Aura, `neo4j_user` and `neo4j_database` are the *instance id* (the leading label of the
    # URI host), not the literal "neo4j" that self-hosted Neo4j uses. Verified against a live
    # instance: auth=("neo4j", pw) returns Neo.ClientError.Security.Unauthorized. Import the
    # credentials file with `scripts/import_aura_credentials.py` rather than typing these.
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"

    # --- Qdrant Cloud (vector store) ---------------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: SecretStr | None = None

    # --- Upstash Redis (real-time velocity counters) -----------------------------------------
    redis_url: SecretStr | None = None

    # --- OpenRouter (LLM gateway) ------------------------------------------------------------
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-v4-pro"
    llm_model_cheap: str = "deepseek/deepseek-v4-flash"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(default=60, ge=5, le=600)

    # --- agent guardrails --------------------------------------------------------------------
    # Enforced in code (agent/nodes/route.py and the human gate), never in a prompt. A model can
    # be talked out of an instruction; it cannot be talked out of an `if` statement.
    agent_max_tool_calls: int = Field(default=10, ge=1, le=50)
    agent_min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    agent_high_risk_score: int = Field(default=80, ge=0, le=100)
    agent_amount_threshold_inr: int = Field(default=1_000_000, ge=0)

    # --- graph projection --------------------------------------------------------------------
    # Drop to 2 if AuraDB Free's real element cap is tighter than the design assumed. Capped at 4
    # because traversal cost grows with branching factor to the power of the radius.
    graph_projection_radius: int = Field(default=3, ge=1, le=4)

    # --- lakehouse ---------------------------------------------------------------------------
    # Swap the two Parquet paths to s3://bucket/... to run against a real data lake; nothing else
    # in the codebase changes.
    data_raw_path: Path = Path("data/raw")
    data_curated_path: Path = Path("data/curated")
    data_seed_path: Path = Path("data/seed")
    data_quality_path: Path = Path("data/quality")
    duckdb_path: Path = Path("data/sentinel.duckdb")

    # --- data-quality gate ---------------------------------------------------------------------
    # Operational knobs, deliberately here and not in `config/weights.py`. The risk *weights* are a
    # model and a weight change is a reviewable code change; these are tolerances an operations team
    # widens or tightens per environment, and a strict local threshold that would block a noisy
    # staging feed is exactly the case an env var exists for. Every check reads its bound from here
    # rather than holding a literal, so "why did the gate block this?" is answerable from config.
    dq_row_count_tolerance: float = Field(default=0.40, gt=0.0, le=1.0)
    #: Rs 10 crore. A single retail payment above this is a data defect, not a large customer.
    dq_max_amount_inr: Decimal = Field(default=Decimal("100000000"), gt=0)
    #: Drift thresholds, as fractions. 0.15 means fifteen percentage *points*, matching
    #: `docs/05-DATA-FLOW.md` §4, not a fifteen-percent relative change.
    dq_null_rate_drift: float = Field(default=0.15, gt=0.0, le=1.0)
    dq_channel_mix_drift: float = Field(default=0.20, gt=0.0, le=1.0)
    #: Maximum KS-style deviation between this batch's amount distribution and the baseline's.
    dq_amount_drift_ks: float = Field(default=0.15, gt=0.0, le=1.0)
    #: How many previous batch profiles form the rolling baseline. One batch a day makes this the
    #: "7-day rolling mean" the check table names.
    dq_baseline_batches: int = Field(default=7, ge=1, le=90)

    # --- synthetic data generation -----------------------------------------------------------
    generator_seed: int = 20260803
    generator_customers: int = Field(default=5_000, ge=10)
    generator_days: int = Field(default=90, ge=7, le=730)

    # --- embeddings --------------------------------------------------------------------------
    # `fastembed` defaults its cache to the system temp directory, which Windows eventually wipes,
    # forcing a re-download at the worst possible moment. Pin it inside the project instead.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_cache_dir: Path = Path(".cache/fastembed")

    # --- PII handling ------------------------------------------------------------------------
    pan_hash_salt: SecretStr | None = None

    # --- UI ----------------------------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    # --- validators --------------------------------------------------------------------------

    @field_validator("neo4j_uri")
    @classmethod
    def _neo4j_uri_must_be_encrypted(cls, value: str) -> str:
        """AuraDB is TLS-only. An unencrypted scheme is a copy-paste mistake, so reject it here
        rather than letting the driver fail with a less obvious handshake error."""
        if value and not value.startswith(("neo4j+s://", "neo4j+ssc://", "bolt+s://")):
            raise ValueError(
                "NEO4J_URI must use an encrypted scheme (neo4j+s://). "
                f"Got: {value.split('://')[0]}://…"
            )
        return value

    @field_validator("api_base_url", "openrouter_base_url", "qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalise once here so no call site has to think about double slashes in a URL."""
        return value.rstrip("/")

    @field_validator("redis_url")
    @classmethod
    def _redis_url_must_be_a_redis_url(cls, value: SecretStr | None) -> SecretStr | None:
        """Reject the two things people actually paste into this variable.

        Upstash's console presents a full `redis-cli --tls -u rediss://...` command and, separately,
        an HTTPS REST endpoint. Neither is what a Redis wire-protocol client accepts. Caught here so
        the pipeline, the API, and the UI all fail identically and legibly rather than each
        surfacing a different driver error.
        """
        if value is None:
            return None
        raw = value.get_secret_value().strip()
        if not raw or raw in _PLACEHOLDERS:
            return value
        if raw.startswith(("redis://", "rediss://", "unix://")):
            if raw.startswith("redis://") and "upstash.io" in raw:
                # Upstash accepts TLS only and closes a plaintext connection without an error
                # message, which surfaces as a bare "Connection closed by server".
                raise ValueError(
                    "REDIS_URL points at Upstash over plaintext redis://. Upstash requires TLS - "
                    "change the scheme to rediss:// (double s)."
                )
            return SecretStr(raw)

        if raw.startswith(("http://", "https://")):
            raise ValueError(
                "REDIS_URL is an HTTP REST endpoint. This client speaks the Redis wire protocol - "
                "use the rediss:// URL from the Upstash console (Connect -> redis-cli tab)."
            )
        problem = (
            "a shell command was pasted; keep only the rediss://... URL, not the surrounding "
            "`redis-cli --tls -u` wrapper"
            if re.search(r"rediss?://", raw)
            else "no URL scheme found; it must begin with rediss://"
        )
        raise ValueError(f"REDIS_URL is malformed - {problem}.")

    # --- required-secret accessors -----------------------------------------------------------

    @property
    def postgres_dsn(self) -> str:
        return _require(self.database_url, "DATABASE_URL")

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        if not self.neo4j_uri:
            raise MissingConfigurationError("NEO4J_URI")
        return self.neo4j_user, _require(self.neo4j_password, "NEO4J_PASSWORD")

    @property
    def qdrant_key(self) -> str:
        if not self.qdrant_url:
            raise MissingConfigurationError("QDRANT_URL")
        return _require(self.qdrant_api_key, "QDRANT_API_KEY")

    @property
    def redis_dsn(self) -> str:
        return _require(self.redis_url, "REDIS_URL")

    @property
    def openrouter_key(self) -> str:
        return _require(self.openrouter_api_key, "OPENROUTER_API_KEY")

    @property
    def pan_salt(self) -> str:
        return _require(self.pan_hash_salt, "PAN_HASH_SALT")

    # --- convenience -------------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    def ensure_data_dirs(self) -> None:
        """Create the lakehouse directories. Called by the pipeline entrypoint, which is why the
        raw and curated zones need no committed `.gitkeep` files."""
        for path in (
            self.data_raw_path,
            self.data_curated_path,
            self.data_quality_path,
            self.embedding_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton.

    Cached because `.env` is read from disk and validators run on every construction, and because
    FastAPI resolves this per request. Call `get_settings.cache_clear()` in a test that needs a
    different configuration.
    """
    return Settings()
