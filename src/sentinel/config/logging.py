"""Structured logging.

Two decisions here, both load-bearing for an AML system:

**Structured, not formatted.** `structlog` emits key-value events, so a log line is queryable data
rather than a sentence to grep. `log.info("alert_scored", account_id=..., score=87)` can be
aggregated; `log.info(f"Scored {account_id} at 87")` cannot. In a real bank this is what makes an
audit trail usable.

**Correlation by `request_id`.** One `POST /v1/investigations` fans out into Postgres reads, Cypher
traversals, vector searches, and several LLM calls. Without a shared id, reconstructing what one
investigation did means guessing from timestamps. The id is bound once in API middleware and rides
along in a `contextvars` context, so every downstream log line carries it with no threading of an
argument through the call stack — and it survives `async` hops, which a thread-local would not.

Redaction is deliberate and structural. `SecretStr` in `config/settings.py` stops credentials
leaking; this processor stops *data* leaking — a PAN, a full phone number, or an account holder's
name must never reach a log file. Masking on a key-name allowlist is imperfect (a value passed
under an unexpected key still gets through), so it is a safety net, not a licence to log PII.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from sentinel.config.settings import Settings

# Event keys whose values are masked before rendering. Substring match, so `customer_pan`,
# `pan_hash`, and `pan` all hit the `pan` entry.
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "dsn",
    "pan",
    "aadhaar",
    "phone",
    "mobile",
    "email",
    "customer_name",
    "salt",
)

_MASK = "***REDACTED***"
_REQUEST_ID_KEY = "request_id"


def _redact_sensitive(
    _logger: object, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Mask values whose key looks sensitive, at any nesting depth."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: (_MASK if _is_sensitive(k) else scrub(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, tuple):
            return tuple(scrub(v) for v in value)
        return value

    for key in list(event_dict):
        event_dict[key] = _MASK if _is_sensitive(key) else scrub(event_dict[key])
    return event_dict


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def configure_logging(settings: Settings) -> None:
    """Install the logging configuration. Idempotent — safe to call from every entrypoint.

    Third-party libraries (uvicorn, httpx, neo4j) log through the standard library, so the stdlib
    root logger is levelled to match. Otherwise a `DEBUG` setting would silently leave the noisiest
    loggers on their own defaults.
    """
    level = getattr(logging, settings.log_level)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,  # pulls in request_id
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_sensitive,  # after merge, before render — nothing escapes it
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level, force=True)
    # These two are chatty at INFO and say nothing an operator needs.
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger. Convention is `get_logger(__name__)`."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


# --- request correlation -----------------------------------------------------------------------


def new_request_id() -> str:
    """A short correlation id. 12 hex chars is ~48 bits — collision-safe for a request trace and
    short enough to read off a terminal, unlike a full UUID."""
    return uuid.uuid4().hex[:12]


def bind_request_id(request_id: str | None = None) -> str:
    """Attach a correlation id to the current context; return the id actually used."""
    rid = request_id or new_request_id()
    structlog.contextvars.bind_contextvars(**{_REQUEST_ID_KEY: rid})
    return rid


def clear_request_context() -> None:
    """Drop all context vars. Must run at the end of every request, or an id leaks into the next
    one handled by the same worker."""
    structlog.contextvars.clear_contextvars()
