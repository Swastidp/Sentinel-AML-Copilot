"""The DuckDB session, and the path helpers every query against the lakehouse goes through.

`docs/02-DATA-MODEL.md` §3 is explicit that DuckDB is **not** a second copy of the warehouse. It
holds no tables of its own: it opens in memory, reads the curated Parquet through
`read_parquet(...)` and exits. That is why `connect()` takes no database file, and why `Settings`
carries no path to one — a persisted `.duckdb` would be a third copy of data that already lives in
Parquet and in Postgres, and the first time it went stale nobody would know which one was right.

Three things this module sets that are not defaults, and each is a correctness fix rather than a
preference:

**`SET TimeZone = 'UTC'`.** `EXTRACT(hour FROM txn_ts)` on a `TIMESTAMPTZ` is evaluated in the
*session's* time zone, not UTC. Left unset, DuckDB picks up the host's zone, so the night-hour
feature (00:00-05:00) would come out five and a half hours off on an IST laptop and correct in CI.
That is the worst class of bug: a feature that is wrong only on some machines. **Hive types are
declared, never inferred.** `txn_date=2026-05-06` is a directory name, and inference reads it as
`VARCHAR`. A string partition key does not break loudly — it makes date range predicates
match nothing, so the query returns an empty result instead of an error. `hive_types` pins it to
`DATE`, the same fix `parquet_store.read_transactions` applies on the pyarrow side.

**Globs are built here, not written out at call sites.** One function means one place where the
`data/curated` layout is encoded, and it is the same shape whether the root is a local directory or
an `s3://` prefix — DuckDB reads both through the identical glob.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import duckdb

from sentinel.config.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

log = get_logger(__name__)

#: Every Parquet dataset in the lakehouse whose files sit under `txn_date=...` directories. Only
#: `transactions` is partitioned; the reference tables are one file each.
HIVE_TYPES: Final = "{'txn_date': DATE}"


def dataset_glob(root: Path, name: str) -> str:
    """The recursive glob for a **partitioned** dataset — `root/name/**/*.parquet`.

    Forward slashes unconditionally. DuckDB accepts them on Windows and a backslash inside a
    single-quoted SQL string is an escape character on other platforms, so normalising here means
    the same query text runs unchanged on a laptop and in CI.
    """
    return (root / name / "**" / "*.parquet").as_posix()


def table_file(root: Path, name: str) -> str:
    """The single-file path for an **unpartitioned** reference table.

    Mirrors `parquet_store._write_single_file`: `root/accounts/accounts.parquet`. Reading it through
    the same convention the writer uses is what keeps the two from drifting.
    """
    return (root / name / f"{name}.parquet").as_posix()


def read_partitioned(path_param: str) -> str:
    """SQL fragment reading a partitioned dataset, with `txn_date` typed as a real `DATE`.

    `path_param` is the name of a bound parameter (`$txn_glob`), never an interpolated path — the
    glob still arrives through the driver rather than through string concatenation.
    """
    return f"read_parquet({path_param}, hive_partitioning = true, hive_types = {HIVE_TYPES})"


@contextmanager
def connect() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open an in-memory DuckDB session configured for this lakehouse, and close it afterwards.

    A context manager rather than a bare constructor because the connection owns a thread pool and a
    memory budget; a leaked one in a long-running process (the API, a Streamlit rerun) holds both.
    """
    connection = duckdb.connect(":memory:")
    try:
        # See the module docstring: this is what makes the night-hour feature machine-independent.
        connection.execute("SET TimeZone = 'UTC'")
        yield connection
    finally:
        connection.close()
