"""The DuckDB feature computation — eleven account-level signals, 21 columns, per rolling window.

One row per `(account_id, window)`, for the 7D and 30D windows, written to
`data/curated/account_features/`. This is the module `docs/02-DATA-MODEL.md` §3 sketches; the query
below is that sketch finished, and the differences from it are the three things the sketch says the
loader has to handle plus the four features it does not show.

Run it with `python -m sentinel.pipeline.features`, or `make features`. It runs the data-quality
gate first and promotes raw to curated only if the gate cleared the batch, so features can never be
computed over rows that were never validated.

---

**What is computed, and where each column comes from.** The table below is grouped by *signal*, not
by column and not by the list in `docs/11-ROADMAP.md` T4.2 — the two do not line up one-for-one, and
reading it as if they did is what made the feature count look inconsistent. All eleven of T4.2's
features are delivered here. Two rows below merge a pair of T4.2 entries that share a definition
(in/out counts, in/out amounts), and two rows are extras T4.2 does not ask for at all:
`balance_delta` and counterparty newness. Twenty-one columns in all, in write order in
`FEATURE_SCHEMA`.

| Feature | Columns | How |
|---|---|---|
| in/out counts | `credit_count`, `debit_count` | legs in the window, by derived leg direction |
| in/out amounts | `total_credit`, `total_debit` | `SUM(amount)` per leg direction |
| pass-through ratio | `pass_through_ratio` | debits over credits, `0.0` when nothing came in |
| unique counterparties | `distinct_credit_counterparties`, `..._debit_...` | distinct handles |
| night-hour share | `night_txn_ratio` | share of legs with `EXTRACT(hour)` in 00:00-05:59 |
| structuring-band count | `structuring_band_count` | legs inside the just-under-Rs-50,000 band |
| dormancy gap | `days_since_last_activity` | see trap 3 below |
| device sharing | `distinct_devices`, `distinct_ips` | distinct non-null fingerprints and IPs |
| velocity z-score | `velocity_zscore` | see "velocity" below |
| balance delta | `balance_delta` | credits minus debits — see `FeatureVector` |
| counterparty newness | `new_counterparty_ratio` | window counterparties unseen before the window |

---

**Trap 1 — unpivot on `from_account`/`to_account`, never filter `direction`.** This ledger stores
one row per movement of money and `direction` records the *initiating* side: it is `DEBIT` on every
row that has a `from_account`, whatever the receiving account experienced. Filtering that column to
build a per-account view produces a table that is right for payers and wrong for every single payee
— and it fails silently, because the row counts still look plausible. The `legs` CTE therefore
derives `leg_direction` from which side of the transfer the account sits on, and a transfer between
two of our own accounts correctly yields two legs from one row.

**Trap 2 — accounts absent from a window need a `LEFT JOIN` with zero defaults.** An aggregate over
the ledger cannot emit a row for an account that did not transact, and 1,284 of 6,200 accounts have
no activity in the 7D window of the real batch. Dropping them would not merely lose rows: every
downstream denominator (population mean, anomaly-model fit, "what fraction of accounts are dormant")
would be computed over the active subset while claiming to describe the book, and `/v1/score` would
return 404 for a live account. So the query is driven **from** the account table, and every
aggregate is coalesced to zero.

**Trap 3 — `days_since_last_activity` needs the `accounts.last_activity_on` fallback.** The batch is
90 days of ledger; the dormancy gaps that matter to rule R005 are 180 to 400 days, and no 90-day
window can express one. `accounts` is a snapshot as of the batch's `window_start`, so its
`last_activity_on` is the last activity *before* the ledger begins — which is precisely the missing
fact. The column is measured to the account's most recent activity **strictly before the feature
window opens**, falling back to `last_activity_on` and then to `opened_on`. Measuring to the last
activity *including* the window would report 0 for exactly the accounts the feature exists to catch
— a dormant-reactivation mule is, by definition, active right now.

**`$as_of` is bound by the caller and read from the data, never `now()`.** `resolve_as_of` takes
`MAX(ingested_at)`, a value the generator fixes deterministically. A feature set that depends on the
wall clock cannot be recomputed identically tomorrow, which breaks both reproducibility and any
attempt to explain a stored score after the fact — the same rule the data-quality gate follows.

**Every column is selected by name.** `txn_date` is the hive partition key and pyarrow appends it
last on read rather than leaving it where the write schema put it, so positional access is already
wrong before anything else goes wrong.

---

**Velocity, and why it is a Poisson z-score rather than a sample z-score.** The signal wanted here
is "sudden throughput against the account's own history", which `application/risk_scorer.py` then
ramps between two thresholds. The obvious implementation — mean and standard deviation of the
account's prior daily counts — has a fatal hole at exactly the case that matters most: an account
that was silent for the whole baseline has a standard deviation of zero, so the burst that
reactivates it divides by zero and scores nothing. Transaction counts are arrival counts, so their
variance equals their mean; the expected count over the window is the account's prior daily rate
times the window length, and the standard error is its square root. `feature_velocity_min_expected`
floors the expectation so a silent account gets a large positive score instead of an undefined one,
and an account with less than `feature_velocity_min_baseline_days` of history scores 0.0 because
there is genuinely nothing to compare against.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.entities import AccountFeatures
from sentinel.domain.enums import FeatureWindow
from sentinel.domain.exceptions import ConfigurationError, DomainValidationError
from sentinel.domain.value_objects import FeatureVector
from sentinel.infrastructure.storage import duckdb_engine, parquet_store
from sentinel.pipeline import promote
from sentinel.pipeline.quality import run_gate

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

log = get_logger(__name__)

#: Bumped whenever the SQL below changes what a column means, so a stored feature row can be traced
#: to the definition that produced it. Same reason `GATE_VERSION` exists for the quality gate.
FEATURE_SET_VERSION: Final = "feat-v1"

#: The curated directory features land in, and the table name a reader asks for. Single-file, like
#: the other unpartitioned reference tables, so `parquet_store.read_table(curated, name)` reads it.
FEATURES_DIR: Final = "account_features"

#: The DuckDB table the two window passes accumulate into before one write.
_STAGING_TABLE: Final = "account_features_staging"

#: The rolling windows T4.2 asks for, in days. 1H and 24H exist in `FeatureWindow` and are served
#: from Redis counters at Stage 10 — they are real-time, not batch, so they are absent here rather
#: than forgotten.
WINDOW_DAYS: Final[dict[FeatureWindow, int]] = {FeatureWindow.D7: 7, FeatureWindow.D30: 30}

#: The night band, 00:00-05:59 inclusive, matching `docs/02-DATA-MODEL.md` §3. A definition rather
#: than a tolerance, so it is a constant here and not a `Settings` knob — a band that wrapped past
#: midnight would need a different comparison, not a different number.
NIGHT_HOUR_START: Final = 0
NIGHT_HOUR_END: Final = 5

_MONEY: Final = pa.decimal128(18, 2)
_TS: Final = pa.timestamp("us", tz="UTC")

#: The output contract. The query is cast to this before anything is written, so a column that
#: changed type in DuckDB fails here rather than three stages downstream in a Postgres `COPY`.
#: Money is `decimal128(18, 2)` for the reason `parquet_store` gives: `float64` would put binary
#: floating point back into the ledger at the storage boundary.
FEATURE_SCHEMA: Final = pa.schema(
    [
        ("account_id", pa.string()),
        ("window", pa.string()),
        ("as_of", _TS),
        ("txn_count", pa.int32()),
        ("credit_count", pa.int32()),
        ("debit_count", pa.int32()),
        ("total_credit", _MONEY),
        ("total_debit", _MONEY),
        ("balance_delta", _MONEY),
        ("distinct_credit_counterparties", pa.int32()),
        ("distinct_debit_counterparties", pa.int32()),
        ("max_single_txn", _MONEY),
        ("avg_txn_amount", _MONEY),
        ("pass_through_ratio", pa.float64()),
        ("night_txn_ratio", pa.float64()),
        ("new_counterparty_ratio", pa.float64()),
        ("structuring_band_count", pa.int32()),
        ("distinct_devices", pa.int32()),
        ("distinct_ips", pa.int32()),
        ("days_since_last_activity", pa.int32()),
        ("velocity_zscore", pa.float64()),
    ]
)


def _feature_sql() -> str:
    """Build the feature query once. Every value the caller controls is a bound parameter.

    A function rather than a module constant only because `read_partitioned` composes the
    `hive_types` fragment, and putting that behind an f-string at import time would mean escaping
    the braces in `{'txn_date': DATE}` for no gain in readability.
    """
    ledger = duckdb_engine.read_partitioned("$txn_glob")
    return f"""
WITH params AS (
    SELECT CAST($as_of AS TIMESTAMPTZ)                     AS as_of,
           CAST($window_days AS INTEGER)                   AS window_days,
           CAST($as_of AS TIMESTAMPTZ)
               - to_days(CAST($window_days AS INTEGER))    AS window_start,
           CAST($band_low  AS DECIMAL(18, 2))              AS band_low,
           CAST($band_high AS DECIMAL(18, 2))              AS band_high,
           CAST($night_start AS INTEGER)                   AS night_start,
           CAST($night_end   AS INTEGER)                   AS night_end,
           CAST($min_baseline_days AS INTEGER)             AS min_baseline_days,
           CAST($min_expected AS DOUBLE)                   AS min_expected
),
-- Failed and reversed payments never moved money. Counting them would inflate every velocity and
-- amount feature with value that does not exist.
settled AS (
    SELECT txn_ts, from_account, to_account, counterparty_handle, amount, device_id, ip_address
    FROM {ledger}
    WHERE status = 'SUCCESS'
),
-- TRAP 1. One row per movement of money becomes one row per account leg. `leg_direction` is derived
-- from which side the account sits on and is NOT the stored `direction` column, which records the
-- initiating side only. An internal transfer yields two legs; a payment out to an external party
-- yields one.
legs AS (
    SELECT from_account                              AS account_id,
           'DEBIT'                                   AS leg_direction,
           COALESCE(to_account, counterparty_handle) AS counterparty,
           amount, txn_ts, device_id, ip_address
    FROM settled
    WHERE from_account IS NOT NULL

    UNION ALL

    SELECT to_account                                  AS account_id,
           'CREDIT'                                    AS leg_direction,
           COALESCE(from_account, counterparty_handle) AS counterparty,
           amount, txn_ts, device_id, ip_address
    FROM settled
    WHERE to_account IS NOT NULL
),
window_legs AS (
    SELECT l.* FROM legs l, params p
    WHERE l.txn_ts >= p.window_start AND l.txn_ts < p.as_of
),
prior_legs AS (
    SELECT l.* FROM legs l, params p
    WHERE l.txn_ts < p.window_start
),
windowed AS (
    SELECT
        w.account_id,
        COUNT(*)                                                       AS txn_count,
        COUNT(*) FILTER (w.leg_direction = 'CREDIT')                   AS credit_count,
        COUNT(*) FILTER (w.leg_direction = 'DEBIT')                    AS debit_count,
        COALESCE(SUM(w.amount) FILTER (w.leg_direction = 'CREDIT'), 0) AS total_credit,
        COALESCE(SUM(w.amount) FILTER (w.leg_direction = 'DEBIT'), 0)  AS total_debit,
        COUNT(DISTINCT w.counterparty)
            FILTER (w.leg_direction = 'CREDIT')                        AS distinct_credit_cps,
        COUNT(DISTINCT w.counterparty)
            FILTER (w.leg_direction = 'DEBIT')                         AS distinct_debit_cps,
        MAX(w.amount)                                                  AS max_single_txn,
        AVG(w.amount)                                                  AS avg_txn_amount,
        COUNT(*) FILTER (
            EXTRACT(hour FROM w.txn_ts) BETWEEN p.night_start AND p.night_end
        )                                                              AS night_count,
        COUNT(*) FILTER (w.amount BETWEEN p.band_low AND p.band_high)  AS structuring_band_count,
        COUNT(DISTINCT w.device_id)                                    AS distinct_devices,
        COUNT(DISTINCT w.ip_address)                                   AS distinct_ips
    FROM window_legs w, params p
    GROUP BY w.account_id
),
-- A counterparty is "new" when this account never dealt with it before the window opened. NULLs are
-- excluded from both sides: an equality join on NULL never matches, so a cash deposit with no
-- counterparty at all would otherwise be counted as a new relationship every single window.
window_counterparties AS (
    SELECT DISTINCT account_id, counterparty FROM window_legs WHERE counterparty IS NOT NULL
),
prior_counterparties AS (
    SELECT DISTINCT account_id, counterparty FROM prior_legs WHERE counterparty IS NOT NULL
),
newness AS (
    SELECT wc.account_id,
           COUNT(*)                                  AS window_cp_count,
           COUNT(*) FILTER (pc.counterparty IS NULL) AS new_cp_count
    FROM window_counterparties wc
    LEFT JOIN prior_counterparties pc
           ON pc.account_id = wc.account_id AND pc.counterparty = wc.counterparty
    GROUP BY wc.account_id
),
-- TRAP 3, first half: the account's own history *before* the window. `prior_txn_count` also feeds
-- the velocity baseline, so both read the same rows and cannot disagree about what "before" means.
history AS (
    SELECT account_id,
           MAX(CAST(txn_ts AS DATE)) AS last_prior_date,
           COUNT(*)                  AS prior_txn_count
    FROM prior_legs
    GROUP BY account_id
),
-- How many days of ledger sit behind the window. Zero-filled by construction: it counts calendar
-- days, not days on which the account happened to transact, so a quiet account is measured as quiet
-- rather than as absent. The ledger's own start is aggregated in its own CTE so the join against
-- `params` happens after the aggregation rather than inside it.
ledger AS (
    SELECT MIN(CAST(txn_ts AS DATE)) AS first_date FROM legs
),
baseline AS (
    SELECT GREATEST(date_diff('day', g.first_date, CAST(p.window_start AS DATE)), 0)
               AS baseline_days
    FROM ledger g, params p
),
-- TRAP 2. The account table drives the result, so an account with no activity in the window gets a
-- row of zeroes rather than vanishing.
scored AS (
    SELECT
        a.account_id,
        CAST($window AS VARCHAR)                                       AS "window",
        p.as_of,
        COALESCE(w.txn_count, 0)                                       AS txn_count,
        COALESCE(w.credit_count, 0)                                    AS credit_count,
        COALESCE(w.debit_count, 0)                                     AS debit_count,
        COALESCE(w.total_credit, 0)                                    AS total_credit,
        COALESCE(w.total_debit, 0)                                     AS total_debit,
        COALESCE(w.total_credit, 0) - COALESCE(w.total_debit, 0)       AS balance_delta,
        COALESCE(w.distinct_credit_cps, 0)                             AS distinct_credit_cps,
        COALESCE(w.distinct_debit_cps, 0)                              AS distinct_debit_cps,
        COALESCE(w.max_single_txn, 0)                                  AS max_single_txn,
        COALESCE(w.avg_txn_amount, 0)                                  AS avg_txn_amount,
        -- Matches `entities.compute_pass_through_ratio` exactly, zero-credit case included. The two
        -- must agree or the batch path and the serving path mean different things by one word.
        CASE WHEN COALESCE(w.total_credit, 0) > 0
             THEN CAST(w.total_debit AS DOUBLE) / CAST(w.total_credit AS DOUBLE)
             ELSE 0.0 END                                              AS pass_through_ratio,
        CASE WHEN COALESCE(w.txn_count, 0) > 0
             THEN CAST(w.night_count AS DOUBLE) / CAST(w.txn_count AS DOUBLE)
             ELSE 0.0 END                                              AS night_txn_ratio,
        CASE WHEN COALESCE(n.window_cp_count, 0) > 0
             THEN CAST(n.new_cp_count AS DOUBLE) / CAST(n.window_cp_count AS DOUBLE)
             ELSE 0.0 END                                              AS new_counterparty_ratio,
        COALESCE(w.structuring_band_count, 0)                          AS structuring_band_count,
        COALESCE(w.distinct_devices, 0)                                AS distinct_devices,
        COALESCE(w.distinct_ips, 0)                                    AS distinct_ips,
        -- TRAP 3, second half: ledger history first, then the account snapshot, then the open date.
        -- `GREATEST(0, ...)` guards the one case the snapshot can produce a future date in — a
        -- re-run against a shorter window.
        GREATEST(0, date_diff('day',
            COALESCE(h.last_prior_date, a.last_activity_on, a.opened_on),
            CAST(p.as_of AS DATE)))                                    AS days_since_last_activity,
        CASE
            WHEN b.baseline_days < p.min_baseline_days THEN 0.0
            ELSE (COALESCE(w.txn_count, 0) - GREATEST(
                      CAST(COALESCE(h.prior_txn_count, 0) AS DOUBLE)
                          / b.baseline_days * p.window_days,
                      p.min_expected))
                 / sqrt(GREATEST(
                      CAST(COALESCE(h.prior_txn_count, 0) AS DOUBLE)
                          / b.baseline_days * p.window_days,
                      p.min_expected))
        END                                                            AS velocity_zscore
    FROM read_parquet($accounts_file) a
    CROSS JOIN params p
    CROSS JOIN baseline b
    LEFT JOIN windowed w ON w.account_id = a.account_id
    LEFT JOIN newness  n ON n.account_id = a.account_id
    LEFT JOIN history  h ON h.account_id = a.account_id
)
SELECT
    account_id,
    "window",
    as_of,
    CAST(txn_count    AS INTEGER)                    AS txn_count,
    CAST(credit_count AS INTEGER)                    AS credit_count,
    CAST(debit_count  AS INTEGER)                    AS debit_count,
    CAST(total_credit  AS DECIMAL(18, 2))            AS total_credit,
    CAST(total_debit   AS DECIMAL(18, 2))            AS total_debit,
    CAST(balance_delta AS DECIMAL(18, 2))            AS balance_delta,
    CAST(distinct_credit_cps AS INTEGER)             AS distinct_credit_counterparties,
    CAST(distinct_debit_cps  AS INTEGER)             AS distinct_debit_counterparties,
    CAST(max_single_txn AS DECIMAL(18, 2))           AS max_single_txn,
    CAST(avg_txn_amount AS DECIMAL(18, 2))           AS avg_txn_amount,
    CAST(pass_through_ratio     AS DOUBLE)           AS pass_through_ratio,
    CAST(night_txn_ratio        AS DOUBLE)           AS night_txn_ratio,
    CAST(new_counterparty_ratio AS DOUBLE)           AS new_counterparty_ratio,
    CAST(structuring_band_count AS INTEGER)          AS structuring_band_count,
    CAST(distinct_devices AS INTEGER)                AS distinct_devices,
    CAST(distinct_ips     AS INTEGER)                AS distinct_ips,
    CAST(days_since_last_activity AS INTEGER)        AS days_since_last_activity,
    CAST(velocity_zscore AS DOUBLE)                  AS velocity_zscore
FROM scored
ORDER BY account_id
"""


def resolve_as_of(connection: duckdb.DuckDBPyConnection, curated_root: Path) -> datetime:
    """The instant every window is measured back from, read off the batch itself.

    `MAX(ingested_at)`, not `now()` and not `MAX(txn_ts)`. The writer stamps `ingested_at`
    identically on every row of a batch from a deterministic value, so the same bytes give the same
    features on any day of the year. `MAX(txn_ts)` would move with a single mis-dated row, which is
    exactly the defect the data-quality gate's future-timestamp check exists to catch.
    """
    row = connection.execute(
        f"SELECT MAX(ingested_at) FROM {duckdb_engine.read_partitioned('$txn_glob')}",
        {"txn_glob": duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR)},
    ).fetchone()
    if row is None or row[0] is None:
        raise DomainValidationError(
            f"the curated zone at {curated_root} holds no transactions, so there is no `as_of` to "
            f"compute features against. Promote a batch first."
        )
    stamp: datetime = row[0]
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def compute_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    curated_root: Path,
    as_of: datetime,
    settings: Settings,
) -> pa.Table:
    """Run the feature SQL once per window and return the union, cast to `FEATURE_SCHEMA`.

    The windows accumulate into one DuckDB staging table rather than being concatenated in Python,
    so the whole result is materialised once and the `COPY` that writes it reads the same rows the
    caller inspects. Returned as an Arrow table, never a pandas frame: pandas has no decimal dtype
    and would silently turn every rupee column into a `float64`.
    """
    if settings.feature_structuring_high_inr <= settings.feature_structuring_low_inr:
        raise ConfigurationError(
            f"FEATURE_STRUCTURING_HIGH_INR ({settings.feature_structuring_high_inr}) must exceed "
            f"FEATURE_STRUCTURING_LOW_INR ({settings.feature_structuring_low_inr}); an inverted "
            f"band matches nothing and would report every account as clean."
        )
    sql = _feature_sql()
    base = {
        "as_of": as_of,
        "band_low": str(settings.feature_structuring_low_inr),
        "band_high": str(settings.feature_structuring_high_inr),
        "night_start": NIGHT_HOUR_START,
        "night_end": NIGHT_HOUR_END,
        "min_baseline_days": settings.feature_velocity_min_baseline_days,
        "min_expected": settings.feature_velocity_min_expected,
        "txn_glob": duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR),
        "accounts_file": duckdb_engine.table_file(curated_root, parquet_store.ACCOUNTS_DIR),
    }
    for index, (window, days) in enumerate(WINDOW_DAYS.items()):
        verb = (
            f"CREATE OR REPLACE TEMP TABLE {_STAGING_TABLE} AS"
            if index == 0
            else f"INSERT INTO {_STAGING_TABLE}"
        )
        connection.execute(f"{verb} {sql}", {**base, "window": str(window), "window_days": days})
        log.debug("feature window computed", window=str(window), days=days)

    # `to_arrow_table`, not `arrow()`: the latter hands back a streaming `RecordBatchReader` in
    # DuckDB 1.5, which is single-pass and cannot be written and then inspected.
    table: pa.Table = connection.table(_STAGING_TABLE).to_arrow_table()
    return table.cast(FEATURE_SCHEMA)


def write_features(table: pa.Table, curated_root: Path) -> Path:
    """Write the feature table to `curated/account_features/account_features.parquet`.

    One fixed filename, overwritten in place, so a second run replaces the previous features instead
    of leaving a directory-level read to union two generations of them. Same idempotency rule as the
    reference tables in `parquet_store`.
    """
    directory = curated_root / FEATURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{FEATURES_DIR}.parquet"
    # `pyarrow.parquet` ships `py.typed` but leaves its public functions unannotated, so this is a
    # `no-untyped-call` under --strict. Ignored on the one line rather than by widening the
    # `[[tool.mypy.overrides]]` block that `parquet_store` and `dq_store` use — that would relax the
    # check across all 500 lines of this module to buy the same one call.
    pq.write_table(table, path)  # type: ignore[no-untyped-call]
    return path


def to_feature_vectors(table: pa.Table) -> tuple[FeatureVector, ...]:
    """Map computed rows onto the domain type the `FeatureStore` port returns.

    Not used by the batch write — Parquet is written straight from Arrow — but it is what the
    Stage 5 Postgres loader and the Redis loader will both call, and having it here means the SQL's
    column names are checked against `FeatureVector`'s fields by a unit test rather than by a stack
    trace at load time. Amounts arrive as `Decimal` because the Arrow columns are
    `decimal128(18, 2)`.
    """
    return tuple(
        FeatureVector(
            account_id=str(row["account_id"]),
            window=FeatureWindow(row["window"]),
            as_of=row["as_of"],
            txn_count=int(row["txn_count"]),
            total_credit=Decimal(row["total_credit"]),
            total_debit=Decimal(row["total_debit"]),
            balance_delta=Decimal(row["balance_delta"]),
            distinct_credit_counterparties=int(row["distinct_credit_counterparties"]),
            distinct_debit_counterparties=int(row["distinct_debit_counterparties"]),
            max_single_txn=Decimal(row["max_single_txn"]),
            avg_txn_amount=Decimal(row["avg_txn_amount"]),
            night_txn_ratio=float(row["night_txn_ratio"]),
            new_counterparty_ratio=float(row["new_counterparty_ratio"]),
            distinct_devices=int(row["distinct_devices"]),
            distinct_ips=int(row["distinct_ips"]),
            days_since_last_activity=int(row["days_since_last_activity"]),
            velocity_zscore=float(row["velocity_zscore"]),
        )
        for row in table.to_pylist()
    )


def to_account_features(table: pa.Table) -> tuple[AccountFeatures, ...]:
    """Map computed rows onto the persisted entity, all 21 columns.

    The twin of `to_feature_vectors`, and separate from it because the two types differ in more than
    verbosity: `AccountFeatures` carries `credit_count`, `debit_count`, `structuring_band_count` and
    the stored `pass_through_ratio`, and the rule engine needs every one of them —
    `structuring_band_count` *is* rule R004. Reading rules off a `FeatureVector` would have meant
    R004 could not be written at all.

    Construction is the check: `AccountFeatures.__post_init__` re-derives `pass_through_ratio` from
    the two totals and rejects the row if the stored column disagrees, so a feature file whose batch
    and serving definitions had drifted fails here rather than producing quiet, wrong alerts.
    """
    return tuple(
        AccountFeatures(
            account_id=str(row["account_id"]),
            window=FeatureWindow(row["window"]),
            as_of=row["as_of"],
            txn_count=int(row["txn_count"]),
            credit_count=int(row["credit_count"]),
            debit_count=int(row["debit_count"]),
            total_credit=Decimal(row["total_credit"]),
            total_debit=Decimal(row["total_debit"]),
            balance_delta=Decimal(row["balance_delta"]),
            distinct_credit_counterparties=int(row["distinct_credit_counterparties"]),
            distinct_debit_counterparties=int(row["distinct_debit_counterparties"]),
            max_single_txn=Decimal(row["max_single_txn"]),
            avg_txn_amount=Decimal(row["avg_txn_amount"]),
            pass_through_ratio=float(row["pass_through_ratio"]),
            night_txn_ratio=float(row["night_txn_ratio"]),
            new_counterparty_ratio=float(row["new_counterparty_ratio"]),
            structuring_band_count=int(row["structuring_band_count"]),
            distinct_devices=int(row["distinct_devices"]),
            distinct_ips=int(row["distinct_ips"]),
            days_since_last_activity=int(row["days_since_last_activity"]),
            velocity_zscore=float(row["velocity_zscore"]),
        )
        for row in table.to_pylist()
    )


def run(settings: Settings | None = None) -> Path:
    """Gate, promote, compute, write. The whole of Stage 4 as one idempotent step.

    The gate runs first and raises `DataQualityGateFailed` on a blocking failure, so a bad batch
    never reaches `promote` — and `promote` refuses a blocked `GateResult` anyway, because a caller
    that used `evaluate` instead of `run_gate` would otherwise have to remember to check.
    """
    cfg = settings if settings is not None else get_settings()
    gate = run_gate(cfg)
    with duckdb_engine.connect() as connection:
        promotion = promote.promote(
            gate,
            raw_root=cfg.data_raw_path,
            curated_root=cfg.data_curated_path,
            connection=connection,
        )
        clock = time.perf_counter()
        as_of = resolve_as_of(connection, cfg.data_curated_path)
        table = compute_features(
            connection, curated_root=cfg.data_curated_path, as_of=as_of, settings=cfg
        )
        path = write_features(table, cfg.data_curated_path)

    log.info(
        "account features written",
        batch_id=promotion.batch_id,
        feature_set=FEATURE_SET_VERSION,
        as_of=as_of.isoformat(),
        windows=[str(window) for window in WINDOW_DAYS],
        rows=table.num_rows,
        accounts=table.num_rows // len(WINDOW_DAYS),
        path=str(path),
        duration_ms=int((time.perf_counter() - clock) * 1000),
    )
    return path


def main(settings: Settings | None = None) -> None:
    """`python -m sentinel.pipeline.features`, or `make features`.

    No `try`/`except` around `run`: a blocked batch must surface as a non-zero exit, and
    `DataQualityGateFailed` already carries the run id, the batch id and the failed check names. The
    gate has persisted its report by the time it raises, so the audit trail survives the failure.
    """
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    cfg.ensure_data_dirs()
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    main()
