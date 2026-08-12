"""Push the curated zone into Neon Postgres, then prove it landed -- T5.3.

    python -m sentinel.pipeline.load

Five tables, in foreign-key order, from the Parquet that `pipeline/promote.py` validated and
`pipeline/features.py` computed. Then a verification pass that is the actual deliverable: row counts
read back **from Postgres** and compared against the same counts computed **in DuckDB** over the
curated files, a money column round-tripped and compared exactly, and the ground-truth table
asserted empty.

**`typology_labels` is never loaded.** The table exists in the schema because it is one of the
twelve in `docs/02-DATA-MODEL.md`, and this module writes nothing to it. It could not if it wanted
to: `pipeline/promote.py` withholds the labels from the curated zone, so there is no curated source
to read. `verify` asserts the table is empty after every load, which turns the convention into a
test that fails rather than a paragraph that ages. The boundary matters because it is the difference
between "my detector works" and a defensible precision figure.

**Counts are compared against a second engine, not against what was sent.** A loader that reports
the number of rows it streamed has proved nothing about what the database kept. DuckDB reads the
same Parquet through a completely separate code path, so the two numbers agreeing means the files,
the Arrow mapping and the upsert all agree.

**Money is checked to the paisa, not to a tolerance.** The chain is `decimal128(18, 2)` in Parquet,
`Decimal` in Python, text `COPY`, `numeric(18, 2)` in Postgres -- no binary float anywhere -- so the
right assertion is exact equality of both the value and its scale. Anything looser would hide the
one class of defect the whole `Decimal` discipline exists to prevent.

**Idempotent.** Every write is `INSERT ... ON CONFLICT DO UPDATE`, so running this twice leaves
identical row counts. `tests/integration/test_postgres_load.py` runs it twice and asserts exactly
that rather than taking this docstring's word for it.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.exceptions import DomainValidationError
from sentinel.infrastructure.postgres import connection as pg
from sentinel.infrastructure.postgres import loader, migrate
from sentinel.infrastructure.postgres.loader import TableSpec
from sentinel.infrastructure.storage import duckdb_engine, parquet_store
from sentinel.pipeline import features

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import psycopg
    import pyarrow as pa

log = get_logger(__name__)


def _spec(
    table: str,
    key: tuple[str, ...],
    schema: pa.Schema,
    renames: Mapping[str, str] | None = None,
) -> TableSpec:
    """Build a load spec from the Arrow schema the writer already declares.

    Derived rather than retyped, so a column added to `parquet_store.CUSTOMER_SCHEMA` or to
    `features.FEATURE_SCHEMA` cannot be silently left out of the load -- it arrives here
    automatically and then fails against the Postgres column list, which is the loud failure.
    `renames` maps an Arrow column onto a differently-named Postgres one; there is exactly one.
    """
    mapping = renames or {}
    return TableSpec(
        table=table,
        key=key,
        columns=tuple((mapping.get(name, name), name) for name in schema.names),
    )


#: The five loads, in the order foreign keys require: customers before accounts, accounts and
#: devices before transactions, accounts before features. `typology_labels` is absent and stays
#: absent -- see the module docstring.
LOADS: Final[tuple[tuple[TableSpec, Callable[[Path], pa.Table]], ...]] = (
    (
        _spec("customers", ("customer_id",), parquet_store.CUSTOMER_SCHEMA),
        lambda root: parquet_store.read_table(root, parquet_store.CUSTOMERS_DIR),
    ),
    (
        _spec("accounts", ("account_id",), parquet_store.ACCOUNT_SCHEMA),
        lambda root: parquet_store.read_table(root, parquet_store.ACCOUNTS_DIR),
    ),
    (
        _spec("devices", ("device_id",), parquet_store.DEVICE_SCHEMA),
        lambda root: parquet_store.read_table(root, parquet_store.DEVICES_DIR),
    ),
    (
        # The conflict target is the composite primary key, because Postgres requires the partition
        # key in every unique constraint on a partitioned table. `txn_date` is derived from `txn_ts`
        # and never changes for a given `txn_id`, so the pair is unique exactly when `txn_id` is.
        _spec("transactions", ("txn_id", "txn_date"), parquet_store.TRANSACTION_SCHEMA),
        parquet_store.read_transactions,
    ),
    (
        # `window` is a reserved word in SQL, so the column is `feature_window` in Postgres and
        # stays `window` in Parquet and on `AccountFeatures`. This is the only rename in the load.
        _spec(
            "account_features",
            ("account_id", "feature_window"),
            features.FEATURE_SCHEMA,
            {"window": "feature_window"},
        ),
        lambda root: parquet_store.read_table(root, features.FEATURES_DIR),
    ),
)


@dataclass(frozen=True, slots=True)
class TableResult:
    """What one table's load did, from three independent vantage points.

    `duckdb_rows` is what a second engine says is in the curated Parquet, `affected` is what the
    upsert reported, and `postgres_rows` is what the server holds afterwards. All three agreeing is
    the claim; any one of them alone is a guess.
    """

    table: str
    duckdb_rows: int
    affected: int
    postgres_rows: int

    @property
    def matches(self) -> bool:
        return self.duckdb_rows == self.postgres_rows


@dataclass(frozen=True, slots=True)
class MoneyProbe:
    """One rupee amount carrying paisa, followed from Parquet to Postgres and back.

    Both the value and its string form are compared. The string catches a scale change -- `12.30`
    stored into a `numeric(18, 0)` would come back as `12`, and `Decimal("12") == Decimal("12.30")`
    is already false, but `Decimal("12.3") == Decimal("12.30")` is *true* while the two are not the
    same stored value. Comparing `str` as well is what makes the check about storage rather than
    about arithmetic.
    """

    txn_id: str
    duckdb_amount: Decimal
    postgres_amount: Decimal

    @property
    def exact(self) -> bool:
        return self.duckdb_amount == self.postgres_amount and str(self.duckdb_amount) == str(
            self.postgres_amount
        )


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Everything T5.3 has to show: per-table counts, the money probe, and the label fence."""

    tables: tuple[TableResult, ...]
    money: MoneyProbe
    typology_label_rows: int
    sample_transaction: Mapping[str, Any]
    sample_feature: Mapping[str, Any]

    @property
    def total_rows(self) -> int:
        return sum(result.postgres_rows for result in self.tables)


# --- the DuckDB side of the comparison ----------------------------------------------------------


def duckdb_counts(curated_root: Path) -> dict[str, int]:
    """Row counts straight off the curated Parquet, computed by DuckDB rather than by pyarrow.

    A deliberately separate code path from the one the loader reads through. If both used pyarrow, a
    bug in the Arrow read would agree with itself and the comparison would prove nothing.
    """
    txn_glob = duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR)
    files = {
        "customers": duckdb_engine.table_file(curated_root, parquet_store.CUSTOMERS_DIR),
        "accounts": duckdb_engine.table_file(curated_root, parquet_store.ACCOUNTS_DIR),
        "devices": duckdb_engine.table_file(curated_root, parquet_store.DEVICES_DIR),
        "account_features": duckdb_engine.table_file(curated_root, features.FEATURES_DIR),
    }
    counts: dict[str, int] = {}
    with duckdb_engine.connect() as connection:
        for table, path in files.items():
            row = connection.execute(
                "SELECT COUNT(*) FROM read_parquet($path)", {"path": path}
            ).fetchone()
            counts[table] = int(row[0]) if row is not None else 0
        row = connection.execute(
            f"SELECT COUNT(*) FROM {duckdb_engine.read_partitioned('$txn_glob')}",
            {"txn_glob": txn_glob},
        ).fetchone()
        counts["transactions"] = int(row[0]) if row is not None else 0
    return counts


def duckdb_paisa_amount(curated_root: Path) -> tuple[str, Decimal]:
    """The lowest-id transaction whose amount is not a whole number of rupees.

    Deterministic by `ORDER BY txn_id`, so the money assertion checks the same payment on every run
    and a failure is reproducible. A batch with no fractional amount at all would make the check
    vacuous, so that raises instead of passing quietly.
    """
    txn_glob = duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR)
    with duckdb_engine.connect() as connection:
        row = connection.execute(
            f"""
            SELECT txn_id, amount
              FROM {duckdb_engine.read_partitioned("$txn_glob")}
             WHERE amount <> trunc(amount)
             ORDER BY txn_id
             LIMIT 1
            """,
            {"txn_glob": txn_glob},
        ).fetchone()
    if row is None:
        raise DomainValidationError(
            "no transaction in the curated zone carries paisa, so the decimal round-trip check "
            "would pass without testing anything. Regenerate the batch."
        )
    return str(row[0]), Decimal(row[1])


# --- loading and verification ---------------------------------------------------------------------


def load_all(connection: psycopg.Connection[Any], curated_root: Path) -> dict[str, int]:
    """Upsert every table in foreign-key order and commit once.

    One transaction for all five: `transactions` carries foreign keys to `accounts` and `devices`,
    so a partial commit would leave the ledger referencing rows that a retry might load differently.
    Committing once also means the temporary staging tables all drop together.
    """
    affected: dict[str, int] = {}
    for spec, read in LOADS:
        clock = time.perf_counter()
        table = read(curated_root)
        affected[spec.table] = loader.upsert(connection, spec, table)
        log.info(
            "curated table loaded",
            table=spec.table,
            rows=table.num_rows,
            duration_ms=int((time.perf_counter() - clock) * 1000),
        )
    connection.commit()
    return affected


def verify(
    connection: psycopg.Connection[Any], curated_root: Path, affected: Mapping[str, int]
) -> LoadReport:
    """Read everything back and raise if any of it disagrees.

    Raises `DomainValidationError` rather than returning a report with a failure flag, because a
    count mismatch is not a result a caller should be able to ignore. The report it returns on
    success is for logging and for the integration test to assert against.
    """
    expected = duckdb_counts(curated_root)
    results = tuple(
        TableResult(
            table=spec.table,
            duckdb_rows=expected[spec.table],
            affected=affected[spec.table],
            postgres_rows=loader.count(connection, spec.table),
        )
        for spec, _ in LOADS
    )
    mismatched = [result for result in results if not result.matches]
    if mismatched:
        detail = ", ".join(
            f"{r.table}: duckdb={r.duckdb_rows} postgres={r.postgres_rows}" for r in mismatched
        )
        raise DomainValidationError(f"row counts disagree after load -- {detail}")

    txn_id, duckdb_amount = duckdb_paisa_amount(curated_root)
    with pg.wrap("money round-trip probe"), connection.cursor() as cursor:
        cursor.execute("SELECT amount FROM transactions WHERE txn_id = %s", (txn_id,))
        amount_row = cursor.fetchone()
    if amount_row is None:
        raise DomainValidationError(
            f"transaction {txn_id} is in the curated zone but not in Postgres, although the row "
            f"counts matched. That means the counts agree by coincidence, not by content."
        )
    money = MoneyProbe(txn_id=txn_id, duckdb_amount=duckdb_amount, postgres_amount=amount_row[0])
    if not money.exact:
        raise DomainValidationError(
            f"money did not round-trip exactly: {txn_id} is {duckdb_amount!r} in Parquet and "
            f"{money.postgres_amount!r} in Postgres. numeric(18, 2) and decimal128(18, 2) must "
            f"agree digit for digit -- do not round this away."
        )

    labels = loader.count(connection, "typology_labels")
    if labels:
        raise DomainValidationError(
            f"typology_labels holds {labels} rows. Nothing in Stage 5 may load the ground truth: "
            f"the detection and scoring paths can read this database, and a label they can reach "
            f"makes every precision figure unfalsifiable."
        )

    return LoadReport(
        tables=results,
        money=money,
        typology_label_rows=labels,
        sample_transaction=_sample_transaction(connection, txn_id),
        sample_feature=_sample_feature(connection),
    )


def _sample_transaction(connection: psycopg.Connection[Any], txn_id: str) -> dict[str, Any]:
    """The row behind the money probe, read back whole so a human can look at it."""
    with pg.wrap(f"sample transaction {txn_id}"), connection.cursor() as cursor:
        cursor.execute("SELECT * FROM transactions WHERE txn_id = %s", (txn_id,))
        names = [column.name for column in cursor.description or ()]
        row = cursor.fetchone()
    return dict(zip(names, row, strict=True)) if row is not None else {}


def _sample_feature(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    """The busiest 7D feature row -- deterministic, and more informative than an arbitrary one.

    Ordered by `txn_count DESC` then `account_id` so the tie-break is total: the same account comes
    back on every run, which is what makes "here is a sample row" a repeatable claim.
    """
    with pg.wrap("sample feature row"), connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM account_features
             WHERE feature_window = '7D'
             ORDER BY txn_count DESC, account_id ASC
             LIMIT 1
            """
        )
        names = [column.name for column in cursor.description or ()]
        row = cursor.fetchone()
    return dict(zip(names, row, strict=True)) if row is not None else {}


def run(settings: Settings | None = None) -> LoadReport:
    """Migrate, load, verify. The whole of T5.3 as one idempotent step.

    The migration runs first and is a no-op when the schema is already there, which is cheaper than
    a "did you remember to bootstrap?" error message and removes the only ordering footgun in the
    stage.
    """
    cfg = settings if settings is not None else get_settings()
    with pg.connect(cfg) as connection:
        state = migrate.apply(connection)
        log.info("schema ready", summary=state.summary())
        affected = load_all(connection, cfg.data_curated_path)
        return verify(connection, cfg.data_curated_path, affected)


def main(settings: Settings | None = None) -> int:
    """`python -m sentinel.pipeline.load`."""
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    report = run(cfg)

    width = max(len(result.table) for result in report.tables)
    print(f"{'TABLE'.ljust(width)}  {'DUCKDB':>10}  {'UPSERTED':>10}  {'POSTGRES':>10}  MATCH")
    for result in report.tables:
        print(
            f"{result.table.ljust(width)}  {result.duckdb_rows:>10,}  {result.affected:>10,}  "
            f"{result.postgres_rows:>10,}  {'yes' if result.matches else 'NO'}"
        )
    print(f"{'total'.ljust(width)}  {'':>10}  {'':>10}  {report.total_rows:>10,}")
    print(
        f"\nmoney round-trip: {report.money.txn_id} parquet={report.money.duckdb_amount} "
        f"postgres={report.money.postgres_amount} exact={report.money.exact}"
    )
    print(f"typology_labels rows: {report.typology_label_rows} (fenced - never loaded)")
    print("\nsample transaction:")
    for key, value in report.sample_transaction.items():
        print(f"  {key:<20} {value!r}")
    print("\nsample account_features row (busiest 7D account):")
    for key, value in report.sample_feature.items():
        print(f"  {key:<32} {value!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
