"""Bulk Arrow-to-Postgres upserts -- the write path the pipeline uses and the ports deliberately
do not model.

`domain/ports/__init__.py` states the rule: bulk pipeline writes get no port. They have one
implementation, one caller, and nothing to substitute, so a `Protocol` here would be an interface
with a single implementor. `pipeline/load.py` calls this module directly and T5.2's idempotency
claim is proven by an integration test against it rather than through a seam.

**COPY into a temporary table, then one `INSERT ... SELECT ... ON CONFLICT DO UPDATE`.** Not
`executemany` over 205,419 rows. The reason is round trips: `COPY` streams the whole table in one
statement, and the upsert that follows is a single server-side scan. The staging table is `TEMP` and
`ON COMMIT DROP`, so it lives in the session, is invisible to anything else, and disappears when the
transaction ends -- **no `DROP` statement is ever issued against a real table.**

The pair has to be one transaction. Staged rows that never got upserted are worse than useless, and
a target half-written by a crash between the two steps is the failure the whole idempotency story
exists to rule out.

**Every column is named, in both directions.** `spec.columns` pairs a Postgres column with the Arrow
column it comes from, so `account_features.feature_window` reading from the Arrow column `window` is
one line rather than a positional coincidence, and `transactions.txn_date` -- which pyarrow returns
as the *last* column of a partitioned read, not where the write schema put it -- lands correctly
without anyone having to remember that.

**Money crosses as `Decimal`.** Arrow `decimal128(18, 2)` deserialises to `Decimal`, psycopg's text
COPY writes its exact digits, and `numeric(18, 2)` stores them. No step in that chain is a binary
float, which is what makes the paisa assertion in `pipeline/load.py` pass exactly rather than
approximately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from psycopg import sql

from sentinel.config.logging import get_logger
from sentinel.domain.exceptions import DomainValidationError
from sentinel.infrastructure.postgres import connection as pg

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import psycopg
    import pyarrow as pa

log = get_logger(__name__)

#: Rows per `COPY` chunk. Big enough that the per-batch Python overhead disappears against the
#: network round trip, small enough that one chunk of `transactions` is a few MB rather than the
#: whole 205k-row table materialised as Python tuples at once.
CHUNK_ROWS: Final = 20_000


@dataclass(frozen=True, slots=True)
class TableSpec:
    """How one Arrow table maps onto one Postgres table.

    `columns` is ordered `(postgres_column, arrow_column)`. Naming both sides is what lets the one
    column whose name differs -- `feature_window` from `window`, because `WINDOW` is a reserved word
    in SQL -- be a visible line in a spec rather than a silent rename buried in a mapper.

    `key` is the conflict target and must be a real unique constraint or primary key on the table,
    otherwise Postgres rejects the `ON CONFLICT` clause outright. That is a useful property: a spec
    naming the wrong key fails on the first statement instead of quietly inserting duplicates.
    """

    table: str
    key: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = [db for db, _ in self.columns]
        missing = [column for column in self.key if column not in names]
        if missing:
            raise DomainValidationError(
                f"{self.table}: conflict key {missing!r} is not among the loaded columns "
                f"{names!r}; the upsert would have nothing to match on."
            )

    @property
    def pg_columns(self) -> tuple[str, ...]:
        return tuple(db for db, _ in self.columns)

    @property
    def arrow_columns(self) -> tuple[str, ...]:
        return tuple(source for _, source in self.columns)

    @property
    def updatable(self) -> tuple[str, ...]:
        """Columns the upsert overwrites -- everything that is not part of the conflict key."""
        return tuple(name for name in self.pg_columns if name not in self.key)


def _rows(table: pa.Table, columns: Sequence[str]) -> Iterator[tuple[Any, ...]]:
    """Stream the table as tuples in `columns` order, one `CHUNK_ROWS` slice at a time.

    Column-wise rather than `to_pylist()`: the latter builds one dict per row with a fresh copy of
    every key, which for 205k transactions is 3.5 million redundant strings. Selecting by name is
    non-negotiable here -- see the module docstring on where pyarrow puts a partition key.
    """
    for batch in table.to_batches(max_chunksize=CHUNK_ROWS):
        values = [batch.column(name).to_pylist() for name in columns]
        yield from zip(*values, strict=True)


def upsert(connection: psycopg.Connection[Any], spec: TableSpec, table: pa.Table) -> int:
    """Load one Arrow table into its Postgres table and return the number of rows affected.

    Does **not** commit. The caller loads several tables in foreign-key order and commits once, so
    that a failure loading `transactions` does not leave `accounts` populated and the batch
    half-present. Re-running with the same input affects the same number of rows and changes no
    values, which is the T5.2 acceptance criterion.
    """
    missing = [name for name in spec.arrow_columns if name not in table.column_names]
    if missing:
        raise DomainValidationError(
            f"{spec.table}: the source table is missing {missing!r}. Present: "
            f"{table.column_names!r}."
        )

    target = sql.Identifier(spec.table)
    staging = sql.Identifier(f"stg_{spec.table}")
    columns = sql.SQL(", ").join(sql.Identifier(name) for name in spec.pg_columns)
    assignments = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(name))
        for name in spec.updatable
    )
    # A table whose every column is part of the key has nothing to overwrite. None of the five
    # specs is like that today, but `DO UPDATE SET` with an empty list is a syntax error rather
    # than a no-op, so the branch is here instead of a comment saying it cannot happen.
    resolution = (
        sql.SQL("DO UPDATE SET {assignments}").format(assignments=assignments)
        if spec.updatable
        else sql.SQL("DO NOTHING")
    )

    with pg.wrap(f"bulk load {spec.table}"), connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE TEMP TABLE {staging} (LIKE {target}) ON COMMIT DROP").format(
                staging=staging, target=target
            )
        )
        copy_statement = sql.SQL("COPY {staging} ({columns}) FROM STDIN").format(
            staging=staging, columns=columns
        )
        streamed = 0
        with cursor.copy(copy_statement) as copy:
            for row in _rows(table, spec.arrow_columns):
                copy.write_row(row)
                streamed += 1
        cursor.execute(
            sql.SQL(
                "INSERT INTO {target} ({columns}) SELECT {columns} FROM {staging} "
                "ON CONFLICT ({key}) {resolution}"
            ).format(
                target=target,
                columns=columns,
                staging=staging,
                key=sql.SQL(", ").join(sql.Identifier(name) for name in spec.key),
                resolution=resolution,
            )
        )
        affected = cursor.rowcount

    log.info(
        "table upserted",
        table=spec.table,
        streamed=streamed,
        affected=affected,
        columns=len(spec.columns),
    )
    return affected


def count(connection: psycopg.Connection[Any], table: str) -> int:
    """`SELECT count(*)`, read back from the server rather than inferred from what was sent.

    The distinction is the entire point of the verification step in `pipeline/load.py`: a loader
    that reports the number of rows it *streamed* has proved nothing about what the database kept.
    """
    statement = sql.SQL("SELECT count(*) FROM {table}").format(table=sql.Identifier(table))
    with pg.wrap(f"count {table}"), connection.cursor() as cursor:
        cursor.execute(statement)
        row = cursor.fetchone()
    if row is None:  # pragma: no cover - an aggregate always returns one row
        raise pg.PostgresUnavailableError(pg.STORE, f"count {table}", "no row from count query")
    return int(row[0])
