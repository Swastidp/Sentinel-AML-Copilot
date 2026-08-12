"""Raw (BRONZE) to curated (SILVER) promotion — the step the data-quality gate guards.

`docs/05-DATA-FLOW.md` §4: nothing reaches the curated zone until it has been validated. The gate in
`pipeline/quality.py` decides; this module is the only thing that acts on that decision. It takes a
`GateResult` rather than re-running the checks, and it refuses outright when that result is blocked.

Why a copy at all, when the bytes are already on disk? Because the two zones mean different things.
`data/raw/` is an immutable landing zone — whatever the source sent, defects included.
`data/curated/` is the set of rows a downstream consumer is allowed to trust, and the boundary
between them is a recorded verdict. Every reader from here on (the feature SQL, the Postgres loader,
the graph loader) points at curated, so "was this validated?" is answered by which directory the
query names.
**Ground truth is not promoted.** `typology_labels` stays in the raw zone and this module never
copies it, which is a physical version of the eval boundary in `docs/02-DATA-MODEL.md` §2: the
detection path can only read curated, so it cannot reach the labels even by accident. The lint rule
at T7.1 is the second line of defence, not the first.

**Idempotent, and that is enforced by the write mode rather than by hoping.** Transactions are
written with `PARTITION_BY (txn_date)` and `OVERWRITE`, so a second run replaces the partitions it
wrote the first time instead of adding a duplicate copy of every row; the reference tables are
single files overwritten in place. Filenames are derived from the `batch_id`, never a GUID, for the
same reason `parquet_store` derives them: a re-run must produce the same bytes or "reproducible
batch" is a claim rather than a testable property. Run it with `python -m
sentinel.pipeline.features`, which promotes and then computes features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pyarrow as pa

from sentinel.config.logging import get_logger
from sentinel.domain.exceptions import DataQualityGateFailed, DomainValidationError
from sentinel.infrastructure.storage import duckdb_engine, parquet_store

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from sentinel.pipeline.quality import GateResult

log = get_logger(__name__)

#: The four unpartitioned reference tables, paired with the pyarrow schema that names their columns.
#: The schemas are the source of the SELECT lists below, so a column renamed in `parquet_store`
#: moves here automatically instead of leaving a hand-typed list quietly one column short.
REFERENCE_TABLES: Final = (
    (parquet_store.ACCOUNTS_DIR, parquet_store.ACCOUNT_SCHEMA),
    (parquet_store.CUSTOMERS_DIR, parquet_store.CUSTOMER_SCHEMA),
    (parquet_store.DEVICES_DIR, parquet_store.DEVICE_SCHEMA),
)

#: Deliberately absent from `REFERENCE_TABLES`. Named here so the omission reads as a decision
#: rather than as something that was forgotten. See the module docstring.
WITHHELD_FROM_CURATED: Final = (parquet_store.LABELS_DIR,)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """What one promotion moved, so the caller can log it and a test can assert on it."""

    batch_id: str
    curated_root: Path
    transaction_rows: int
    tables: tuple[str, ...]
    duration_ms: int


def _column_list(schema: pa.Schema) -> str:
    """Render a pyarrow schema's field names as a SQL select list.

    By name, always. `txn_date` is the hive partition key and pyarrow appends it *last* on read
    rather than leaving it where the write schema put it, so `SELECT *` into a positional consumer
    is already wrong — and this is the choke point where that would happen.
    """
    return ", ".join(f'"{name}"' for name in schema.names)


def promote(
    gate: GateResult,
    *,
    raw_root: Path,
    curated_root: Path,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> PromotionResult:
    """Copy the validated batch into the curated zone. Refuses a blocked batch.

    `connection` is optional so a caller already holding a session (the feature run does) reuses it
    rather than opening a second one over the same files.
    """
    if gate.blocked:
        # Defence in depth. `run_gate` already raises on a blocked batch, but `evaluate` does not,
        # and a caller that branched on the result must not be able to promote by forgetting to.
        raise DataQualityGateFailed(
            run_id=gate.run.run_id,
            batch_id=gate.run.batch_id,
            failures=tuple(outcome.check_id for outcome in gate.blocking_failures),
        )
    if connection is None:
        with duckdb_engine.connect() as owned:
            return promote(gate, raw_root=raw_root, curated_root=curated_root, connection=owned)

    clock = time.perf_counter()
    batch_id = gate.run.batch_id
    curated_root.mkdir(parents=True, exist_ok=True)

    rows = _promote_transactions(connection, raw_root, curated_root, batch_id)
    for name, schema in REFERENCE_TABLES:
        _promote_reference_table(connection, raw_root, curated_root, name, schema)

    result = PromotionResult(
        batch_id=batch_id,
        curated_root=curated_root,
        transaction_rows=rows,
        tables=(parquet_store.TRANSACTIONS_DIR, *(name for name, _ in REFERENCE_TABLES)),
        duration_ms=int((time.perf_counter() - clock) * 1000),
    )
    log.info(
        "batch promoted to curated",
        batch_id=result.batch_id,
        status=str(gate.run.status),
        rows=result.transaction_rows,
        tables=list(result.tables),
        withheld=list(WITHHELD_FROM_CURATED),
        curated=str(curated_root),
        duration_ms=result.duration_ms,
    )
    return result


def _promote_transactions(
    connection: duckdb.DuckDBPyConnection, raw_root: Path, curated_root: Path, batch_id: str
) -> int:
    """Copy the partitioned ledger, keeping the `txn_date=...` layout the raw zone uses."""
    source = duckdb_engine.dataset_glob(raw_root, parquet_store.TRANSACTIONS_DIR)
    target = (curated_root / parquet_store.TRANSACTIONS_DIR).as_posix()
    columns = _column_list(parquet_store.TRANSACTION_SCHEMA)
    reader = duckdb_engine.read_partitioned("$source")
    # `ORDER BY txn_id` costs one sort of 205k rows and buys byte-identical output across runs.
    # Without it DuckDB's parallel scan emits rows in whatever order the threads finished in, so two
    # promotions of the same batch produce files with identical *contents* and different *bytes* —
    # and "reproducible batch" stops being something a checksum can prove.
    connection.execute(
        f"COPY (SELECT {columns} FROM {reader} ORDER BY txn_id) TO '{target}' "
        f"(FORMAT PARQUET, PARTITION_BY ({parquet_store.PARTITION_COLUMN}), OVERWRITE, "
        f"FILENAME_PATTERN '{batch_id}-{{i}}')",
        {"source": source},
    )
    counted = connection.execute(
        f"SELECT COUNT(*) FROM {duckdb_engine.read_partitioned('$target')}",
        {"target": duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR)},
    ).fetchone()
    if counted is None:
        raise DomainValidationError(
            f"promotion of batch {batch_id} wrote {target} but reading it back returned no count; "
            f"the curated zone cannot be trusted."
        )
    return int(counted[0])


def _promote_reference_table(
    connection: duckdb.DuckDBPyConnection,
    raw_root: Path,
    curated_root: Path,
    name: str,
    schema: pa.Schema,
) -> None:
    """Copy one unpartitioned reference table as a single file, replacing any previous copy.

    Sorted for the same reason `_promote_transactions` is. These tables are small enough that DuckDB
    is unlikely to reorder them, and "unlikely" is exactly the word that made the ledger's sort
    necessary — a reproducibility guarantee that holds only while the input stays under some
    unstated size is not a guarantee. One sort of a few thousand rows is the whole cost.
    """
    source = duckdb_engine.table_file(raw_root, name)
    directory = curated_root / name
    directory.mkdir(parents=True, exist_ok=True)
    target = (directory / f"{name}.parquet").as_posix()
    # The first column of every reference schema is its primary key — `account_id`, `customer_id`,
    # `device_id` — so this is a total order rather than a tie-broken one.
    connection.execute(
        f"COPY (SELECT {_column_list(schema)} FROM read_parquet($source) "
        f'ORDER BY "{schema.names[0]}") '
        f"TO '{target}' (FORMAT PARQUET)",
        {"source": source},
    )
