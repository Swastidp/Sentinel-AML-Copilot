"""The single exception hierarchy for the whole system.

One root (`SentinelError`) so callers can catch everything this codebase raises without also
swallowing `KeyboardInterrupt` or a driver's own errors. The API layer maps these to HTTP status
codes in one place (`api/errors.py`), which is only possible because the hierarchy is closed.

Closed means *this file is the whole list*. That is what keeps the mapping to one handler with no
per-store imports:

| Exception | HTTP |
|---|---|
| `DomainValidationError` | 422 |
| `NotFoundError` | 404 |
| `ConflictError` | 409 |
| `StoreUnavailableError` | 503 |
| `ConfigurationError` | 500 |

An adapter may subclass any of these to carry driver detail — `PostgresUnavailableError` is fine —
but it must not introduce a sibling root. `api/errors.py` catches the five above and nothing else,
so a new top-level type would fall through to a 500 with no body the caller can act on.

This module lives in `domain` and imports nothing outside the standard library.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Root of every error this system raises deliberately."""


# --- domain invariants -----------------------------------------------------------------------


class DomainValidationError(SentinelError, ValueError):
    """An entity or value object was constructed with a value that cannot be correct.

    Raised from `__post_init__` guards in `entities.py` and `value_objects.py` — a naive timestamp,
    a non-positive transaction amount, a confidence outside 0 to 1. These are programming or
    ingestion errors, not user input, so they fail at construction rather than travelling one layer
    further.

    It subclasses `ValueError` as well as `SentinelError` because `ValueError` is what a parsing
    loop around a `Transaction(...)` call would naturally catch, while the API's single
    `SentinelError` handler must still map it to an HTTP status. Both work.
    """


# --- persistence -----------------------------------------------------------------------------
# Three named conditions, because the API has to tell them apart and `SentinelError` alone cannot.
#
# The first two are deliberately *not* raised by a read: `get(...)` returning `T | None` is the
# convention across `domain/ports/repositories.py`, so a caller decides whether absence is a 404, a
# skip, or a bug. They are for the cases where the caller has no such choice.
#
# The third, `StoreUnavailableError`, is the opposite — it can come out of any method here, read or
# write. It is named in the domain rather than left to each adapter because "the store is
# unreachable" is a condition the *caller* has to handle: it is the difference between "this account
# has no features" and "we could not look", and every port docstring turns on that distinction.
# Naming it once also keeps `api/errors.py` free of a per-store import.


class NotFoundError(SentinelError):
    """A write or a state transition named a row that does not exist.

    `AlertRepository.update_status` on an unknown `alert_id` is the case: unlike a read, silently
    succeeding would lose a state change and leave the alert stuck where it was. Mapped to `404`.

    `kind` is the human name of the thing ("alert", "investigation"), not a table name — the message
    reaches an analyst.
    """

    def __init__(self, kind: str, identifier: str) -> None:
        self.kind = kind
        self.identifier = identifier
        super().__init__(f"no {kind} with id {identifier!r}.")


class ConflictError(SentinelError):
    """A write would overwrite something the design says is immutable.

    Raised by `InvestigationRepository.save_decision` when a decision for that investigation already
    exists. `docs/07-API-DESIGN.md` §5 maps this to `409 Conflict`: an analyst's adjudication is the
    audit record, so a second one has to be rejected rather than silently replace the first.

    Enforced by a unique constraint in the adapter rather than a read-then-write, so two concurrent
    analysts cannot both believe they won.
    """

    def __init__(self, kind: str, identifier: str, reason: str) -> None:
        self.kind = kind
        self.identifier = identifier
        super().__init__(f"cannot write {kind} {identifier!r}: {reason}")


class StoreUnavailableError(SentinelError):
    """A store could not be reached, timed out, or returned something unreadable.

    The one thing this must never be is silence. Every port documents a difference between an absent
    answer and a failed lookup — `None` from a `get`, an empty tuple from a search, zero fan degrees
    from a traversal — and each of those is a *finding* the agent may reason from. A failed read
    flattened into one becomes a fabricated finding: "no shared devices" when nobody looked.
    Raising is the only honest option, which is why this lives in the domain rather than
    being left to each adapter to name for itself.

    Mapped to `503`: it is transient, so the caller should retry rather than treat the answer as
    settled.

    `kind` is the store — "postgres", "neo4j", "qdrant", "redis", "openrouter". `identifier` is
    whatever was being looked up, so a log line says which read failed and not merely that one did.
    Adapters may subclass this to carry driver detail; the point is that one catchable type exists.
    """

    def __init__(self, kind: str, identifier: str, reason: str) -> None:
        self.kind = kind
        self.identifier = identifier
        super().__init__(f"{kind} unavailable while reading {identifier!r}: {reason}")


# --- configuration ---------------------------------------------------------------------------


class ConfigurationError(SentinelError):
    """Settings are absent, malformed, or still holding a placeholder value."""


class MissingConfigurationError(ConfigurationError):
    """A required environment variable was never set.

    Raised at the point of *use* rather than at import time, so unit tests and CI can construct
    `Settings` with no `.env` file at all and still fail loudly the moment something genuinely
    needs a credential.
    """

    def __init__(self, env_key: str) -> None:
        self.env_key = env_key
        super().__init__(
            f"{env_key} is not set. Copy .env.example to .env and fill it in "
            f"(in CI, add {env_key} to GitHub Actions Secrets)."
        )
