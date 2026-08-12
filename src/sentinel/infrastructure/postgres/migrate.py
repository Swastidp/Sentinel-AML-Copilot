"""The migration runner -- T5.1.

    python -m sentinel.infrastructure.postgres.migrate

Applies every `.sql` file in `migrations/`, in filename order, inside one transaction. Running it a
second time is a no-op and prints byte-identical output, which is the acceptance criterion for T5.1
and is also how the idempotency is proven rather than asserted.

**There is no ledger of applied versions, and that is the design.** The usual `schema_migrations`
table exists to stop a *non-idempotent* migration running twice. This project's rule is stricter:
every migration must be additive and idempotent -- `CREATE TABLE IF NOT EXISTS`,
`CREATE INDEX IF NOT EXISTS`, never a `DROP`, a `TRUNCATE` or a `DELETE`. `check_safety` below
enforces that rule by refusing to execute a file that breaks it, so re-running the whole directory
is safe by construction and a version table would be a second mechanism guarding the same property.
The trade is real and worth naming: without a ledger, an *edited* migration silently re-applies
rather than being rejected as a changed checksum.
ponytail: add a checksum ledger the first time a migration has to be edited after being applied to
an environment someone else owns. Today there is one database and one operator.

**Why the whole directory runs every time.** A file that has already been applied costs one round
trip and changes nothing, because every statement inside it is conditional. That keeps the runner
to fifty lines and makes "did the schema drift?" answerable by running it again.

The `DATABASE_URL` this points at is a live Neon database. The safety check is not decoration.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.exceptions import ConfigurationError
from sentinel.infrastructure.postgres import connection as pg

if TYPE_CHECKING:
    from collections.abc import Sequence

    import psycopg

log = get_logger(__name__)

#: Where the `.sql` files live. Shipped inside the package (see `[tool.setuptools.package-data]`),
#: so an installed wheel can migrate a database without the repo checked out.
MIGRATIONS_DIR: Final = Path(__file__).parent / "migrations"

#: Statements a migration may never contain. Matched on word boundaries and case-insensitively,
#: outside of comments. `DELETE` is included alongside the two obvious ones because a migration that
#: "cleans up" rows is exactly as destructive as one that drops the table, and slower to notice.
_FORBIDDEN: Final = re.compile(
    r"\b(drop\s+(table|schema|database|column)|truncate|delete\s+from)\b", re.IGNORECASE
)

#: Every `CREATE TABLE` and `CREATE INDEX` must be guarded. `CREATE TABLE ... PARTITION OF` inside
#: the DO block is guarded too, via the `format()` string it builds.
_UNGUARDED_CREATE: Final = re.compile(
    r"\bcreate\s+(?:unique\s+)?(?:table|index)\s+(?!if\s+not\s+exists)", re.IGNORECASE
)

_COMMENT: Final = re.compile(r"--[^\n]*")


@dataclass(frozen=True, slots=True)
class SchemaState:
    """What the database looks like after a run. Two runs must produce two equal instances.

    `tables` counts base and partitioned tables in `public`, excluding partitions -- so it is the
    twelve of `docs/02-DATA-MODEL.md` and not twelve plus however many monthly slices of
    `transactions` exist. `partitions` counts those slices separately, and `indexes` excludes the
    per-partition copies Postgres attaches automatically for the same reason.
    """

    tables: tuple[str, ...]
    partitions: int
    indexes: int

    def summary(self) -> str:
        return (
            f"{len(self.tables)} tables, {self.partitions} transaction partitions, "
            f"{self.indexes} indexes"
        )


def migration_files() -> tuple[Path, ...]:
    """Every migration, in the order it must be applied -- which is filename order.

    Zero-padded numeric prefixes (`001_`) make lexical order and intended order the same thing. A
    directory with no migrations in it is a packaging failure, not an empty to-do list, so it raises
    rather than quietly doing nothing.
    """
    files = tuple(sorted(MIGRATIONS_DIR.glob("*.sql")))
    if not files:
        raise ConfigurationError(
            f"no .sql migrations found under {MIGRATIONS_DIR}. If this is an installed package, "
            f"the `[tool.setuptools.package-data]` entry for migrations is missing."
        )
    return files


def check_safety(sql: str, name: str) -> None:
    """Refuse a migration that is destructive or not idempotent.

    Comments are stripped first, so the prose above a statement can say the word "drop" without
    tripping the guard while an actual `DROP TABLE` still does. Raises `ConfigurationError`, which
    is the right family: a migration that breaks the rule is a misconfigured deployment artifact,
    not a store failure.
    """
    body = _COMMENT.sub("", sql)
    destructive = _FORBIDDEN.search(body)
    if destructive:
        raise ConfigurationError(
            f"migration {name} contains {destructive.group(0)!r}. Migrations in this project are "
            f"additive only -- DATABASE_URL points at a live database and nothing here may destroy "
            f"data. Write a new additive migration instead."
        )
    unguarded = _UNGUARDED_CREATE.search(body)
    if unguarded:
        line = body[: unguarded.start()].count("\n") + 1
        raise ConfigurationError(
            f"migration {name} line {line}: {unguarded.group(0).strip()!r} is missing "
            f"IF NOT EXISTS, so running the migration twice would fail. Every statement must be "
            f"conditional -- the runner re-applies the whole directory on every invocation."
        )


def inspect(connection: psycopg.Connection[Any]) -> SchemaState:
    """Read back what exists, so a caller can compare two runs instead of trusting one.

    `relispartition` is what separates the twelve modelled tables from the monthly slices of
    `transactions`, and the same flag separates a declared index from the copies Postgres attaches
    to each partition. Partitioned indexes are `relkind = 'I'`, ordinary ones `'i'`; both count.
    """
    with pg.wrap("schema inspection"), connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.relname
              FROM pg_class c
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public'
               AND c.relkind IN ('r', 'p')
               AND NOT c.relispartition
             ORDER BY c.relname
            """
        )
        tables = tuple(str(row[0]) for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT count(*) FILTER (WHERE c.relkind = 'r' AND c.relispartition),
                   count(*) FILTER (WHERE c.relkind IN ('i', 'I') AND NOT c.relispartition)
              FROM pg_class c
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public'
            """
        )
        counts = cursor.fetchone()
    if counts is None:  # pragma: no cover - an aggregate query always returns exactly one row
        raise pg.PostgresUnavailableError(pg.STORE, "schema inspection", "no row from count query")
    return SchemaState(tables=tables, partitions=int(counts[0]), indexes=int(counts[1]))


def apply(connection: psycopg.Connection[Any], files: Sequence[Path] | None = None) -> SchemaState:
    """Run every migration in one transaction and return the resulting schema state.

    One transaction for the whole directory, not one per file: Postgres does transactional DDL, so a
    failure halfway through leaves the database exactly as it was rather than half-migrated. That is
    the property that makes re-running safe after an interrupted attempt.
    """
    chosen = tuple(files) if files is not None else migration_files()
    for path in chosen:
        sql = path.read_text(encoding="utf-8")
        check_safety(sql, path.name)
        with pg.wrap(f"migration {path.name}"):
            # The whole file goes to the server as one string. Splitting it on semicolons in Python
            # would break the `DO $$ ... $$` block, whose body contains its own.
            connection.execute(sql)
        log.info("migration applied", migration=path.name, statements_bytes=len(sql))
    connection.commit()
    return inspect(connection)


def run(settings: Settings | None = None) -> SchemaState:
    """Open a connection, migrate, and hand back what the schema looks like afterwards."""
    cfg = settings if settings is not None else get_settings()
    with pg.connect(cfg) as connection:
        return apply(connection)


def main(settings: Settings | None = None) -> int:
    """`python -m sentinel.infrastructure.postgres.migrate`.

    Prints the applied filenames and the resulting object counts and nothing else -- no timestamps,
    no durations, no "already applied" branch. That is deliberate: the T5.1 acceptance test is that
    two consecutive runs produce identical output, and any wall-clock value in this block would make
    that impossible to demonstrate honestly.
    """
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    files = migration_files()
    state = run(cfg)
    print("applied:")
    for path in files:
        print(f"  {path.name}")
    print(f"schema:  {state.summary()}")
    for name in state.tables:
        print(f"  {name}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
