"""Opening a Neon connection, and turning driver failures into domain errors.

Two things live here and nothing else, because every other Postgres module needs both and neither
belongs to any one of them.

**`connect()`** is the only place a DSN is read. It comes from `Settings.postgres_dsn`, which raises
`MissingConfigurationError` naming `DATABASE_URL` when it is absent -- no adapter holds a connection
string and none of them calls `os.environ`.

**`translate` / `wrap`** turn a `psycopg.Error` into `PostgresUnavailableError`, a subclass of the
domain's `StoreUnavailableError`. That mapping is the whole reason this module exists rather than
each adapter catching the driver itself. `domain/ports/repositories.py` states the rule every port
docstring then repeats: a store failure is never a `None` and never an empty tuple, because "this
account does not exist" and "we could not look" lead a caller to opposite decisions and only one of
them is a finding. An adapter that let a `psycopg.OperationalError` escape would break the API's
single error handler; one that swallowed it into `None` would fabricate evidence.

**Neon auto-suspends.** A cold start on this project measures around 16 seconds against 3 warm, so
`CONNECT_TIMEOUT` is generous on purpose. A slow first query is the database waking up, not a broken
DSN, and a tight timeout would turn a normal wake-up into a spurious outage.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final

import psycopg

from sentinel.config.settings import Settings, get_settings
from sentinel.domain.exceptions import StoreUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Seconds. Sized for a Neon cold start (measured: ~16s suspended, ~3s warm), not for a warm pool.
CONNECT_TIMEOUT: Final = 45

#: What `StoreUnavailableError.kind` reads as in a log line or an API error body.
STORE: Final = "postgres"


class PostgresUnavailableError(StoreUnavailableError):
    """Postgres could not be reached, or refused the statement for an infrastructural reason.

    A subclass rather than a new root, per the closed hierarchy in `domain/exceptions.py`: it maps
    to `503` through the same handler as every other store, and it carries the driver's own message
    so the log says *which* failure happened rather than only that one did.
    """


def translate(exc: psycopg.Error, identifier: str) -> PostgresUnavailableError:
    """Wrap a driver error, naming what was being read or written when it fired."""
    return PostgresUnavailableError(STORE, identifier, f"{type(exc).__name__}: {exc}")


@contextmanager
def wrap(identifier: str) -> Iterator[None]:
    """Run a block, converting any `psycopg.Error` inside it into `PostgresUnavailableError`.

    Deliberately narrow: it catches `psycopg.Error` and nothing else. A `DomainValidationError`
    raised while mapping a row to an entity is a real defect in the data or the mapping, and burying
    it as "the store is unavailable" would send a caller off to retry a query that will never work.
    """
    try:
        yield
    except psycopg.Error as exc:
        raise translate(exc, identifier) from exc


@contextmanager
def connect(settings: Settings | None = None) -> Iterator[psycopg.Connection[Any]]:
    """Open one connection to Neon and close it on the way out.

    Autocommit is left **off**, which is psycopg's default and the right one here: the bulk loader
    stages a whole table into a temporary table and upserts from it, and that pair has to be one
    transaction or a crash between the two leaves the staging rows visible to nothing and the target
    half-written. Callers that want a statement committed say `connection.commit()`.

    No connection pool. There is exactly one process at a time touching this database today -- the
    batch loader and the migration runner -- and a pool for a single serial caller is an object that
    exists to be configured.
    ponytail: add `psycopg_pool` when the FastAPI service lands at Stage 12 and connections start
    being opened per request rather than per process.
    """
    cfg = settings if settings is not None else get_settings()
    dsn = cfg.postgres_dsn
    try:
        connection = psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT)
    except psycopg.Error as exc:
        raise translate(exc, "connect") from exc
    try:
        yield connection
    finally:
        connection.close()
