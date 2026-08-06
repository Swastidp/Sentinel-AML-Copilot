"""The raw-zone (BRONZE) writer: domain entities out to date-partitioned Parquet.

This is the only module that knows the physical layout of the lakehouse. Everything upstream works
with `Customer`, `Account`, `Device`, `Transaction` and `TypologyLabel`; everything downstream
(`pipeline/ingest.py`, DuckDB, the data-quality gate) reads the files this writes. Swapping local
disk for `s3://bucket/raw` is a change to `Settings.data_raw_path` and nothing else, which is the
reason the paths are parameters rather than constants.

Four decisions here are load-bearing, and each would be a real defect if reversed:

**`decimal128(18, 2)` for money, never `float64`.** Scale 2 because the paisa is the smallest unit
INR has, so two places is exact rather than approximate. Precision 18 because that is the widest
decimal every engine in this stack — Arrow, Parquet, DuckDB, Postgres `numeric`, Spark — round-trips
without silently widening or truncating, and it leaves room for ₹10^16 against a data-quality cap of
₹10 crore. Writing amounts as `float64` would put binary floating point back into the ledger at the
storage boundary, which defeats every `Decimal` above this line.

**Deterministic file names.** `basename_template` is derived from the `batch_id`, not from a random
GUID, which is what pyarrow uses by default. A GUID would make two runs of the same seed produce
different bytes, and "reproducible batch" would be a claim rather than a testable property.

**`existing_data_behavior="delete_matching"`.** Re-running a batch must replace its partitions, not
append a second copy of every row. Without this the second run doubles the row count and the
`txn_id` uniqueness check in the quality gate fails on data that is not actually wrong — the retry
is. GitHub Actions retries batches, so idempotency is a requirement and not a nicety.

**One row per movement of money, with `direction` read from `from_account`.** A transfer between two
internal accounts is a single row: `DEBIT` for `from_account`, `CREDIT` for `to_account`. Two legs
would double the apparent value moved. The consequence for whoever writes the feature SQL is that
the per-account view must be derived by unpivoting on `from_account`/`to_account` rather than by
filtering the `direction` column, because a row's `direction` is the *initiating* side's.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentinel.domain.entities import Account, Customer, Device, Transaction, TypologyLabel

#: Money type for every amount, balance and income column. See the module docstring for why
#: (18, 2) and not `float64`, and not a wider precision.
AMOUNT_TYPE: Final = pa.decimal128(18, 2)

#: Timestamps keep microsecond resolution and their UTC offset. A naive Parquet timestamp column
#: would lose the offset that the domain's `ensure_aware` guard exists to preserve, and the
#: night-hour feature (00:00-05:00) would then be computed against an unknown local time.
_TS: Final = pa.timestamp("us", tz="UTC")

TRANSACTIONS_DIR: Final = "transactions"
CUSTOMERS_DIR: Final = "customers"
ACCOUNTS_DIR: Final = "accounts"
DEVICES_DIR: Final = "devices"
LABELS_DIR: Final = "typology_labels"

#: The partition key, matching `transactions.txn_date` in `docs/02-DATA-MODEL.md`. Hive-style, so a
#: partition is the directory `txn_date=2026-05-06` and DuckDB prunes on it without reading files.
PARTITION_COLUMN: Final = "txn_date"

TRANSACTION_SCHEMA: Final = pa.schema(
    [
        ("txn_id", pa.string()),
        ("txn_ts", _TS),
        ("from_account", pa.string()),
        ("to_account", pa.string()),
        ("counterparty_handle", pa.string()),
        ("counterparty_bank", pa.string()),
        ("amount", AMOUNT_TYPE),
        ("currency", pa.string()),
        ("channel", pa.string()),
        ("direction", pa.string()),
        ("status", pa.string()),
        ("device_id", pa.string()),
        ("ip_address", pa.string()),
        ("narration", pa.string()),
        (PARTITION_COLUMN, pa.date32()),
        ("ingested_at", _TS),
        ("batch_id", pa.string()),
    ]
)

CUSTOMER_SCHEMA: Final = pa.schema(
    [
        ("customer_id", pa.string()),
        ("full_name", pa.string()),
        ("date_of_birth", pa.date32()),
        ("pan_hash", pa.string()),
        ("phone_masked", pa.string()),
        ("email_masked", pa.string()),
        ("address_line", pa.string()),
        ("city", pa.string()),
        ("state", pa.string()),
        ("pincode", pa.string()),
        ("occupation", pa.string()),
        ("declared_annual_income", AMOUNT_TYPE),
        ("kyc_level", pa.string()),
        ("risk_rating", pa.string()),
        ("is_pep", pa.bool_()),
        ("onboarded_on", pa.date32()),
        ("segment", pa.string()),
    ]
)

ACCOUNT_SCHEMA: Final = pa.schema(
    [
        ("account_id", pa.string()),
        ("customer_id", pa.string()),
        ("account_type", pa.string()),
        ("opened_on", pa.date32()),
        ("status", pa.string()),
        ("branch_code", pa.string()),
        ("ifsc", pa.string()),
        ("current_balance", AMOUNT_TYPE),
        ("last_activity_on", pa.date32()),
        ("is_flagged", pa.bool_()),
        ("created_at", _TS),
    ]
)

DEVICE_SCHEMA: Final = pa.schema(
    [
        ("device_id", pa.string()),
        ("fingerprint", pa.string()),
        ("os_family", pa.string()),
        ("app_version", pa.string()),
        ("first_seen_at", _TS),
        ("last_seen_at", _TS),
    ]
)

LABEL_SCHEMA: Final = pa.schema(
    [
        ("account_id", pa.string()),
        ("is_mule", pa.bool_()),
        ("typology", pa.string()),
        ("ring_id", pa.string()),
        ("is_hard_negative", pa.bool_()),
        ("note", pa.string()),
    ]
)


def _rows(entities: Sequence[object]) -> list[dict[str, Any]]:
    """Entities to `dict` rows, with `StrEnum` members left as the strings they already are.

    `asdict` is used rather than hand-written accessors so a renamed column fails loudly at the
    schema instead of silently writing a null: `from_pylist` matches by field name, and a name the
    schema does not know is dropped. The small reference tables go through here; the transaction
    table is built column-wise below because 200k `asdict` calls are the one place that cost shows.
    """
    return [asdict(entity) for entity in entities]  # type: ignore[call-overload]


def _write_single_file(table: pa.Table, directory: Path, name: str) -> Path:
    """Write one unpartitioned Parquet file, replacing any previous copy.

    Idempotent because the file name is fixed: a re-run overwrites in place rather than adding a
    second file that a directory-level read would union with the first.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    pq.write_table(table, path)
    return path


def write_transactions(
    transactions: Sequence[Transaction], root: Path, *, batch_id: str | None = None
) -> Path:
    """Write the ledger to `root/transactions/txn_date=YYYY-MM-DD/*.parquet`.

    `batch_id` names the files and defaults to the one stamped on the rows, which is the same value
    for every row of a batch. It is taken from the data rather than passed separately so a caller
    cannot name the files after a batch other than the one it is writing.
    """
    directory = root / TRANSACTIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    if not transactions:
        # An empty batch is a real outcome (a filtered re-run) and must not blow up here. Nothing is
        # written, so the previous partitions survive — deleting them would destroy the raw zone on
        # an empty input, which is the opposite of what an immutable landing zone is for.
        return directory

    stamp = batch_id if batch_id is not None else transactions[0].batch_id
    table = pa.table(
        {
            "txn_id": pa.array([t.txn_id for t in transactions], pa.string()),
            "txn_ts": pa.array([t.txn_ts for t in transactions], _TS),
            "from_account": pa.array([t.from_account for t in transactions], pa.string()),
            "to_account": pa.array([t.to_account for t in transactions], pa.string()),
            "counterparty_handle": pa.array(
                [t.counterparty_handle for t in transactions], pa.string()
            ),
            "counterparty_bank": pa.array([t.counterparty_bank for t in transactions], pa.string()),
            "amount": pa.array([t.amount for t in transactions], AMOUNT_TYPE),
            "currency": pa.array([t.currency for t in transactions], pa.string()),
            "channel": pa.array([str(t.channel) for t in transactions], pa.string()),
            "direction": pa.array([str(t.direction) for t in transactions], pa.string()),
            "status": pa.array([str(t.status) for t in transactions], pa.string()),
            "device_id": pa.array([t.device_id for t in transactions], pa.string()),
            "ip_address": pa.array([t.ip_address for t in transactions], pa.string()),
            "narration": pa.array([t.narration for t in transactions], pa.string()),
            PARTITION_COLUMN: pa.array([t.txn_date for t in transactions], pa.date32()),
            "ingested_at": pa.array([t.ingested_at for t in transactions], _TS),
            "batch_id": pa.array([t.batch_id for t in transactions], pa.string()),
        },
        schema=TRANSACTION_SCHEMA,
    )
    pq.write_to_dataset(
        table,
        root_path=str(directory),
        partition_cols=[PARTITION_COLUMN],
        existing_data_behavior="delete_matching",
        basename_template=f"{stamp}-{{i}}.parquet",
    )
    return directory


def write_customers(customers: Sequence[Customer], root: Path) -> Path:
    """Write the customer reference table, unpartitioned — 5,000 rows need no partitions."""
    table = pa.Table.from_pylist(_rows(customers), schema=CUSTOMER_SCHEMA)
    return _write_single_file(table, root / CUSTOMERS_DIR, CUSTOMERS_DIR)


def write_accounts(accounts: Sequence[Account], root: Path) -> Path:
    """Write the account reference table, unpartitioned."""
    table = pa.Table.from_pylist(_rows(accounts), schema=ACCOUNT_SCHEMA)
    return _write_single_file(table, root / ACCOUNTS_DIR, ACCOUNTS_DIR)


def write_devices(devices: Sequence[Device], root: Path) -> Path:
    """Write the device reference table, unpartitioned."""
    table = pa.Table.from_pylist(_rows(devices), schema=DEVICE_SCHEMA)
    return _write_single_file(table, root / DEVICES_DIR, DEVICES_DIR)


def write_typology_labels(labels: Sequence[TypologyLabel], root: Path) -> Path:
    """Write the generator's ground truth to its **own** directory, separate from everything else.

    Physical separation is the point, not tidiness. Nothing in the detection path may read this, and
    a separate directory means a reader of `data/raw/` can see at a glance that the labels are not
    mixed into `transactions/` or `accounts/`. `reporting/metrics.py` and `tests/` are its only
    consumers.
    """
    table = pa.Table.from_pylist(_rows(labels), schema=LABEL_SCHEMA)
    return _write_single_file(table, root / LABELS_DIR, LABELS_DIR)


def read_transactions(root: Path) -> pa.Table:
    """Read the whole partitioned ledger back, with `txn_date` restored as a real date.

    The hive partitioning schema is supplied explicitly because inference reads the directory name
    `txn_date=2026-05-06` as a string. A string partition key silently breaks date range predicates
    downstream, and the failure mode is a query that returns nothing rather than an error.
    """
    directory = root / TRANSACTIONS_DIR
    partitioning = ds.partitioning(pa.schema([(PARTITION_COLUMN, pa.date32())]), flavor="hive")
    dataset = ds.dataset(directory, format="parquet", partitioning=partitioning)
    return dataset.to_table()


def read_table(root: Path, name: str) -> pa.Table:
    """Read one of the unpartitioned reference tables back by directory name."""
    return pq.read_table(root / name / f"{name}.parquet")
