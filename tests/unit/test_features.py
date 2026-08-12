"""T4.3 — every feature's expected value, asserted against a hand-built ledger.

**The fixtures are built at the DataFrame and Parquet level, deliberately**, for the same reason
`test_dq_checks.py` gives: `Transaction.__post_init__` rejects `amount <= 0` and a naive timestamp,
so several of the shapes this file has to reproduce — a `FAILED` row for ₹999,999 that must be
ignored, a batch whose `direction` column says the opposite of what the receiving account
experienced — cannot be assembled from domain objects at all. The feature SQL reads Parquet, so the
fixture is Parquet.

The ledger is thirteen rows over four accounts, and each account exists to pin one thing down:

| Account | What it proves |
|---|---|
| `ACC00000001` | every one of the eleven features on a busy account, in both windows |
| `ACC00000002` | **trap 1** — it only ever *receives*, yet every row it appears in reads `DEBIT` |
| `ACC00000003` | **trap 2** — silent in the window, and must still get a row of zeroes |
| `ACC00000004` | **trap 3** — no ledger rows at all, so its 322-day gap is snapshot-only |

Every expected number below was derived by hand from the fixture and is written out as a literal,
not recomputed from the same expression the query uses. A test that recomputes the implementation is
a test that passes when the implementation is wrong.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentinel.config.settings import Settings
from sentinel.domain.entities import AccountFeatures, DqRun, compute_pass_through_ratio
from sentinel.domain.enums import DqRunStatus, FeatureWindow
from sentinel.domain.exceptions import ConfigurationError, DataQualityGateFailed
from sentinel.infrastructure.storage import duckdb_engine, parquet_store
from sentinel.pipeline import features, promote, quality

# --- the fixture ledger --------------------------------------------------------------------------

BATCH_ID = "BATCH-20260610-20260803-featfix"
#: Every window is measured back from here. It is the `ingested_at` stamp on every fixture row, so
#: `resolve_as_of` finds it in the data rather than reading a clock.
AS_OF = datetime(2026, 8, 3, 23, 59, 59, tzinfo=UTC)

MULE = "ACC00000001"
SINK = "ACC00000002"
QUIET = "ACC00000003"
DORMANT = "ACC00000004"

#: `last_activity_on` on the account snapshot. The first three sit just before the ledger opens,
#: which is what the real generator writes; `DORMANT`'s is 322 days before `AS_OF` and is the only
#: place that fact exists.
SNAPSHOT_LAST_ACTIVITY = {
    MULE: date(2026, 6, 5),
    SINK: date(2026, 6, 5),
    QUIET: date(2026, 6, 5),
    DORMANT: date(2025, 9, 15),
}


def txn(
    txn_id: str,
    stamp: str,
    *,
    amount: str,
    sender: str | None = None,
    receiver: str | None = None,
    handle: str | None = None,
    channel: str = "UPI",
    direction: str = "DEBIT",
    status: str = "SUCCESS",
    device: str | None = None,
    ip: str | None = None,
) -> dict[str, object]:
    """One ledger row, in the shape a Parquet reader hands back.

    Keyword-only and defaulted so each row below reads as the few fields that actually differ
    between them. `direction` defaults to `DEBIT` because that is what the writer stamps on any row
    carrying a `from_account` — the very thing the feature SQL must not trust.
    """
    moment = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
    return {
        "txn_id": txn_id,
        "txn_ts": moment,
        "from_account": sender,
        "to_account": receiver,
        "counterparty_handle": handle,
        "counterparty_bank": "IDFB" if sender and receiver else "HDFC",
        "amount": Decimal(amount),
        "currency": "INR",
        "channel": channel,
        "direction": direction,
        "status": status,
        "device_id": device,
        "ip_address": ip,
        "narration": f"TEST/{txn_id}",
        "txn_date": moment.date(),
        "ingested_at": AS_OF,
        "batch_id": BATCH_ID,
    }


_LEDGER: tuple[dict[str, object], ...] = (
    # --- before both windows: the velocity and counterparty-newness baseline ---------------------
    txn(
        "T01",
        "2026-06-10 10:00",
        receiver=MULE,
        handle="payer.old@upi",
        amount="1000.00",
        direction="CREDIT",
        device="DEV1",
        ip="10.0.0.1",
    ),
    txn(
        "T02",
        "2026-06-11 11:00",
        sender=MULE,
        handle="shop.old@upi",
        amount="500.00",
        device="DEV1",
        ip="10.0.0.1",
    ),
    txn(
        "T03",
        "2026-07-01 12:00",
        sender=QUIET,
        handle="rent.old@upi",
        amount="2000.00",
        channel="NEFT",
    ),
    # --- inside the 30D window only (2026-07-04 23:59:59 onwards) --------------------------------
    txn(
        "T11",
        "2026-07-10 08:00",
        receiver=MULE,
        handle="payerC@upi",
        amount="5000.00",
        direction="CREDIT",
        device="DEV1",
        ip="10.0.0.1",
    ),
    # --- inside both windows (2026-07-27 23:59:59 onwards) ---------------------------------------
    txn(
        "T04",
        "2026-07-29 02:30",
        receiver=MULE,
        handle="payerA@upi",
        amount="46000.00",
        direction="CREDIT",
        device="DEV1",
        ip="10.0.0.1",
    ),
    txn(
        "T05",
        "2026-07-29 03:15",
        receiver=MULE,
        handle="payerB@upi",
        amount="47500.00",
        direction="CREDIT",
        device="DEV1",
        ip="10.0.0.1",
    ),
    # The internal sweep. Stored `direction` is DEBIT because MULE initiated it — for SINK this row
    # is a CREDIT, and nothing in the stored column says so. This is trap 1 in one row.
    txn(
        "T06",
        "2026-07-30 14:00",
        sender=MULE,
        receiver=SINK,
        amount="90000.00",
        channel="IMPS",
        device="DEV2",
        ip="10.0.0.2",
    ),
    txn("T07", "2026-08-01 15:00", sender=MULE, handle="atmcash", amount="2000.00", channel="ATM"),
    txn(
        "T08",
        "2026-08-02 09:00",
        sender=MULE,
        handle="shop.old@upi",
        amount="1000.00",
        device="DEV1",
        ip="10.0.0.1",
    ),
    # --- money that never moved: excluded from every feature -------------------------------------
    txn(
        "T09",
        "2026-08-02 10:00",
        sender=MULE,
        handle="ghost@upi",
        amount="999999.00",
        status="FAILED",
        device="DEV1",
        ip="10.0.0.1",
    ),
    txn(
        "T12",
        "2026-07-31 12:00",
        sender=SINK,
        handle="rev@upi",
        amount="12345.00",
        status="REVERSED",
        device="DEV2",
        ip="10.0.0.2",
    ),
)


def ledger_frame() -> pd.DataFrame:
    """The fixture ledger as one frame, with the nullable text columns left as `object`."""
    frame = pd.DataFrame(list(_LEDGER))
    for column in ("from_account", "to_account", "counterparty_handle", "device_id", "ip_address"):
        frame[column] = frame[column].astype(object)
    return frame


def account_rows() -> list[dict[str, object]]:
    """The account snapshot — the only carrier of pre-ledger activity dates."""
    return [
        {
            "account_id": account_id,
            "customer_id": f"CUST{index:06d}",
            "account_type": "SAVINGS",
            "opened_on": date(2019, 1, 1),
            "status": "DORMANT" if account_id == DORMANT else "ACTIVE",
            "branch_code": "BR001",
            "ifsc": "IDFB0000001",
            "current_balance": Decimal("1000.00"),
            "last_activity_on": last_activity,
            "is_flagged": False,
            "created_at": datetime(2019, 1, 1, tzinfo=UTC),
        }
        for index, (account_id, last_activity) in enumerate(SNAPSHOT_LAST_ACTIVITY.items())
    ]


def label_rows() -> list[dict[str, object]]:
    """Generator ground truth for the four fixture accounts.

    Reading these is permitted in `tests/` and nowhere else, which is the same boundary the real
    batch enforces physically by never promoting the table. `DORMANT` is the reactivation case; the
    other three are ordinary, and they are what gives the cohort comparison a book to stand against.
    Committed as part of the fixture, so the dormancy coverage survives a clean checkout.
    """
    return [
        {
            "account_id": DORMANT,
            "is_mule": True,
            "typology": "DORMANT_REACTIVATION",
            "ring_id": None,
            "is_hard_negative": False,
            "note": "322-day gap carried only by the account snapshot.",
        },
        *(
            {
                "account_id": account_id,
                "is_mule": False,
                "typology": "NONE",
                "ring_id": None,
                "is_hard_negative": False,
                "note": "",
            }
            for account_id in (MULE, SINK, QUIET)
        ),
    ]


def write_zone(root: Path, frame: pd.DataFrame | None = None) -> Path:
    """Land the fixture as a lakehouse zone: partitioned transactions plus the reference tables.

    Uses `parquet_store`'s own schemas, so the bytes carry `decimal128(18, 2)` amounts and
    `txn_date` as the hive partition key — which comes back *last* on read and is why nothing
    selects a column by position.
    """
    table = pa.Table.from_pandas(
        frame if frame is not None else ledger_frame(),
        schema=parquet_store.TRANSACTION_SCHEMA,
        preserve_index=False,
    )
    pq.write_to_dataset(
        table,
        root_path=str(root / parquet_store.TRANSACTIONS_DIR),
        partition_cols=[parquet_store.PARTITION_COLUMN],
        existing_data_behavior="delete_matching",
        basename_template=f"{BATCH_ID}-{{i}}.parquet",
    )
    for name, schema, payload in (
        (parquet_store.ACCOUNTS_DIR, parquet_store.ACCOUNT_SCHEMA, account_rows()),
        (parquet_store.CUSTOMERS_DIR, parquet_store.CUSTOMER_SCHEMA, []),
        (parquet_store.DEVICES_DIR, parquet_store.DEVICE_SCHEMA, []),
        (parquet_store.LABELS_DIR, parquet_store.LABEL_SCHEMA, label_rows()),
    ):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(payload, schema=schema), directory / f"{name}.parquet")
    return root


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    """A `Settings` pointed at a temporary lakehouse, reading no `.env` at all."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_raw_path=tmp_path / "raw",
        data_curated_path=tmp_path / "curated",
        data_quality_path=tmp_path / "quality",
        **overrides,  # type: ignore[arg-type]
    )


def computed(tmp_path: Path, **overrides: object) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute both windows over the fixture and index the rows by `(account_id, window)`."""
    curated = write_zone(tmp_path / "curated")
    cfg = settings_for(tmp_path, **overrides)
    with duckdb_engine.connect() as connection:
        table = features.compute_features(
            connection, curated_root=curated, as_of=AS_OF, settings=cfg
        )
    return {(row["account_id"], row["window"]): row for row in table.to_pylist()}


@pytest.fixture(name="rows")
def rows_fixture(tmp_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return computed(tmp_path)


# --- the eleven features, one busy account -------------------------------------------------------


def test_every_feature_on_the_mule_account_over_seven_days(rows):
    """All 21 columns for `ACC00000001` in the 7D window, each value derived by hand.

    Five settled legs: two credits of 46,000 and 47,500 at 02:30 and 03:15, and three debits of
    90,000, 2,000 and 1,000. The 999,999 `FAILED` row is money that never moved and appears nowhere.
    """
    row = rows[(MULE, "7D")]
    assert row["as_of"] == AS_OF
    assert row["txn_count"] == 5
    assert row["credit_count"] == 2
    assert row["debit_count"] == 3
    assert row["total_credit"] == Decimal("93500.00")
    assert row["total_debit"] == Decimal("93000.00")
    # 93,500 in and 93,000 out: ₹500 left at the end of the week. That is the mule signature the
    # pass-through ratio alone (0.995) cannot distinguish from a trader.
    assert row["balance_delta"] == Decimal("500.00")
    assert row["pass_through_ratio"] == pytest.approx(0.9946524064171123)
    assert row["distinct_credit_counterparties"] == 2  # payerA@upi, payerB@upi
    assert row["distinct_debit_counterparties"] == 3  # ACC00000002, atmcash, shop.old@upi
    assert row["max_single_txn"] == Decimal("90000.00")
    assert row["avg_txn_amount"] == Decimal("37300.00")  # 186,500 / 5
    assert row["night_txn_ratio"] == pytest.approx(0.4)  # 02:30 and 03:15 of five
    # payerA, payerB, ACC00000002 and atmcash are new; shop.old@upi was dealt with in June.
    assert row["new_counterparty_ratio"] == pytest.approx(0.8)
    assert row["structuring_band_count"] == 2  # 46,000 and 47,500 sit under the 50,000 limit
    assert row["distinct_devices"] == 2  # DEV1, DEV2 — the two null device rows do not count
    assert row["distinct_ips"] == 2
    # Last activity strictly before the window opened was 2026-07-10, 24 days before `as_of`.
    assert row["days_since_last_activity"] == 24
    # 3 prior legs over 47 baseline days is 0.45 expected in a week, floored to 1.0: (5 - 1) / √1.
    assert row["velocity_zscore"] == pytest.approx(4.0)


def test_the_thirty_day_window_sees_what_the_seven_day_window_cannot(rows):
    """The same account, one window wider: the 2026-07-10 credit of ₹5,000 joins the aggregate."""
    row = rows[(MULE, "30D")]
    assert row["txn_count"] == 6
    assert row["credit_count"] == 3
    assert row["debit_count"] == 3
    assert row["total_credit"] == Decimal("98500.00")
    assert row["total_debit"] == Decimal("93000.00")
    assert row["balance_delta"] == Decimal("5500.00")
    assert row["pass_through_ratio"] == pytest.approx(0.9441624365482234)
    assert row["distinct_credit_counterparties"] == 3
    assert row["distinct_debit_counterparties"] == 3
    assert row["max_single_txn"] == Decimal("90000.00")
    assert row["avg_txn_amount"] == Decimal("31916.67")  # 191,500 / 6, rounded to the paisa
    assert row["night_txn_ratio"] == pytest.approx(2 / 6)
    assert row["new_counterparty_ratio"] == pytest.approx(5 / 6)
    assert row["structuring_band_count"] == 2
    assert row["distinct_devices"] == 2
    assert row["distinct_ips"] == 2
    assert row["days_since_last_activity"] == 53  # 2026-06-11, the last leg before 2026-07-04
    # 2 prior legs over 24 baseline days is 2.5 expected in thirty: (6 - 2.5) / √2.5.
    assert row["velocity_zscore"] == pytest.approx(2.2135943621178655)


# --- trap 1: the direction column is the initiating side, not the account's -----------------------


def test_the_receiving_account_is_credited_although_the_row_says_debit(rows):
    """`ACC00000002` appears in exactly one settled row, stored as `direction = 'DEBIT'`.

    Filtering on that column would give this account a debit of ₹90,000 and a credit of nothing — a
    perfect inversion of what happened, on a row that looks entirely normal. The unpivot is what
    makes the sink read as a sink.
    """
    ledger = ledger_frame()
    sweep = ledger[ledger["txn_id"] == "T06"].iloc[0]
    assert sweep["to_account"] == SINK
    assert sweep["direction"] == "DEBIT", (
        "the fixture must store the initiating side, or the trap it exists to catch is absent"
    )

    row = rows[(SINK, "7D")]
    assert row["txn_count"] == 1
    assert row["credit_count"] == 1
    assert row["debit_count"] == 0
    assert row["total_credit"] == Decimal("90000.00")
    assert row["total_debit"] == Decimal("0.00")
    assert row["balance_delta"] == Decimal("90000.00")
    assert row["pass_through_ratio"] == pytest.approx(0.0)  # nothing left again
    assert row["distinct_credit_counterparties"] == 1  # the mule, named by its account id
    assert row["distinct_debit_counterparties"] == 0
    assert row["max_single_txn"] == Decimal("90000.00")
    assert row["avg_txn_amount"] == Decimal("90000.00")
    assert row["night_txn_ratio"] == pytest.approx(0.0)
    assert row["new_counterparty_ratio"] == pytest.approx(1.0)
    assert row["structuring_band_count"] == 0
    assert row["distinct_devices"] == 1
    assert row["distinct_ips"] == 1
    assert row["days_since_last_activity"] == 59  # no ledger history: the snapshot's 2026-06-05
    assert row["velocity_zscore"] == pytest.approx(0.0)  # one leg vs a floored expectation of 1


def test_a_reversed_payment_is_not_a_debit(rows):
    """`ACC00000002` sent ₹12,345 that was reversed. Money that came back never moved."""
    assert rows[(SINK, "7D")]["total_debit"] == Decimal("0.00")
    assert rows[(SINK, "30D")]["debit_count"] == 0


def test_a_failed_payment_never_reaches_a_feature(rows):
    """The ₹999,999 `FAILED` row would be the mule's largest transaction if it counted."""
    assert rows[(MULE, "7D")]["max_single_txn"] == Decimal("90000.00")
    assert rows[(MULE, "30D")]["txn_count"] == 6


# --- trap 2: an account with no activity in the window --------------------------------------------


def test_a_silent_account_gets_a_row_of_zeroes_rather_than_vanishing(rows):
    """`ACC00000003` last moved money on 2026-07-01 and is absent from both windows.

    Dropping it would not merely lose a row — every population denominator downstream would then be
    computed over the active accounts while claiming to describe the whole book.
    """
    row = rows[(QUIET, "7D")]
    assert row["txn_count"] == 0
    assert row["credit_count"] == 0
    assert row["debit_count"] == 0
    assert row["total_credit"] == Decimal("0.00")
    assert row["total_debit"] == Decimal("0.00")
    assert row["balance_delta"] == Decimal("0.00")
    assert row["distinct_credit_counterparties"] == 0
    assert row["distinct_debit_counterparties"] == 0
    assert row["max_single_txn"] == Decimal("0.00")
    assert row["avg_txn_amount"] == Decimal("0.00")
    assert row["pass_through_ratio"] == pytest.approx(0.0)
    assert row["night_txn_ratio"] == pytest.approx(0.0)
    assert row["new_counterparty_ratio"] == pytest.approx(0.0)
    assert row["structuring_band_count"] == 0
    assert row["distinct_devices"] == 0
    assert row["distinct_ips"] == 0
    assert row["days_since_last_activity"] == 33  # 2026-07-01 to 2026-08-03
    assert row["velocity_zscore"] == pytest.approx(-1.0)  # (0 - 1.0) / √1.0
    # The 30D window has a longer expectation to miss: 1 leg over 24 days is 1.25 in thirty.
    assert rows[(QUIET, "30D")]["velocity_zscore"] == pytest.approx(-1.118033988749895)


def test_every_account_gets_a_row_in_every_window(rows):
    assert len(rows) == 4 * 2
    for account_id in (MULE, SINK, QUIET, DORMANT):
        for window in (FeatureWindow.D7, FeatureWindow.D30):
            assert (account_id, str(window)) in rows


# --- trap 3: the dormancy gap the ledger cannot hold ----------------------------------------------


def test_the_dormancy_gap_comes_from_the_account_snapshot(rows):
    """`ACC00000004` has no ledger rows at all, and its gap is 322 days.

    The fixture ledger is 54 days long, so no aggregate over it can produce that number. It exists
    only on `accounts.last_activity_on`, which the generator writes as a snapshot taken before the
    batch begins precisely so this is possible. Without the fallback the answer would collapse to
    the ledger's own age and rule R005 would never fire.
    """
    row = rows[(DORMANT, "7D")]
    assert row["days_since_last_activity"] == 322  # 2025-09-15 to 2026-08-03
    assert rows[(DORMANT, "30D")]["days_since_last_activity"] == 322
    assert row["txn_count"] == 0
    assert row["velocity_zscore"] == pytest.approx(-1.0)


def test_ledger_history_wins_over_the_snapshot_when_it_exists(rows):
    """`ACC00000001` transacted on 2026-07-10, well after its snapshot date of 2026-06-05.

    The snapshot is a fallback, not a floor: observed movement inside the validated batch is better
    evidence than a date carried in from before it.
    """
    assert rows[(MULE, "7D")]["days_since_last_activity"] == 24  # not 59


# --- reproducibility and configuration ------------------------------------------------------------


def test_as_of_is_read_from_the_batch_and_not_from_the_clock(tmp_path: Path):
    """`MAX(ingested_at)`, a value the writer stamps deterministically on every row of a batch."""
    curated = write_zone(tmp_path / "curated")
    with duckdb_engine.connect() as connection:
        assert features.resolve_as_of(connection, curated) == AS_OF


def test_the_structuring_band_is_a_settings_knob_not_a_literal(tmp_path: Path):
    """Widening the band to ₹1,000-₹100,000 sweeps in every one of the mule's five legs."""
    rows = computed(
        tmp_path,
        feature_structuring_low_inr=Decimal("1000"),
        feature_structuring_high_inr=Decimal("100000"),
    )
    assert rows[(MULE, "7D")]["structuring_band_count"] == 5


def test_an_inverted_structuring_band_is_refused(tmp_path: Path):
    """A band whose top is below its bottom matches nothing, so every account reads as clean."""
    curated = write_zone(tmp_path / "curated")
    cfg = settings_for(
        tmp_path,
        feature_structuring_low_inr=Decimal("50000"),
        feature_structuring_high_inr=Decimal("45000"),
    )
    with duckdb_engine.connect() as connection, pytest.raises(ConfigurationError, match="exceed"):
        features.compute_features(connection, curated_root=curated, as_of=AS_OF, settings=cfg)


def test_velocity_is_zero_when_there_is_not_enough_history_to_measure(tmp_path: Path):
    """47 baseline days is less than the 60 this run demands, so no account gets a score."""
    rows = computed(tmp_path, feature_velocity_min_baseline_days=60)
    assert all(row["velocity_zscore"] == 0.0 for row in rows.values())


def test_the_sql_and_the_domain_agree_on_what_pass_through_means(rows):
    """The classic production bug is the batch path and the serving path drifting apart.

    `compute_pass_through_ratio` is the single definition; the SQL is a second implementation of it,
    and `AccountFeatures.__post_init__` will reject a stored row where the two differ by more than
    1e-4. Checked on every row rather than on one, because the disagreement would only show on
    particular amounts.
    """
    for row in rows.values():
        assert row["pass_through_ratio"] == pytest.approx(
            compute_pass_through_ratio(row["total_credit"], row["total_debit"]), abs=1e-9
        )


def test_computing_twice_gives_identical_rows(tmp_path: Path):
    """`as_of` is bound, never read from the clock, so a second run cannot drift."""
    assert computed(tmp_path) == computed(tmp_path)


# --- the domain boundary ------------------------------------------------------------------------


def test_the_stored_row_entity_carries_every_column_the_feature_file_has():
    """`AccountFeatures` is the persisted shape of exactly this table, so the two must not drift.

    They did: the entity sat four columns behind the file (`credit_count`, `debit_count`,
    `structuring_band_count`, `balance_delta`) for as long as nothing loaded the table. The Stage 5
    loader is the first caller that would have noticed, at load time, in Postgres. This notices
    here, in a millisecond, on any run.
    """
    entity = [field.name for field in dataclasses.fields(AccountFeatures)]
    assert entity == features.FEATURE_SCHEMA.names


def test_rows_map_onto_feature_vectors_including_the_new_balance_delta(tmp_path: Path):
    """Every column the SQL produces lines up with a `FeatureVector` field.

    This is what the Stage 5 Postgres loader will call, so a column renamed in the query fails here
    rather than at load time — and it is the only place `balance_delta` currently reaches the domain
    type, since no scorer reads it yet.
    """
    curated = write_zone(tmp_path / "curated")
    cfg = settings_for(tmp_path)
    with duckdb_engine.connect() as connection:
        table = features.compute_features(
            connection, curated_root=curated, as_of=AS_OF, settings=cfg
        )
    vectors = features.to_feature_vectors(table)
    assert len(vectors) == 8
    mule = next(v for v in vectors if v.account_id == MULE and v.window is FeatureWindow.D7)
    assert mule.as_of == AS_OF
    assert mule.total_credit == Decimal("93500.00")
    assert mule.balance_delta == Decimal("500.00")
    # The property and the SQL are the same definition — `compute_pass_through_ratio` — so the two
    # numbers must agree to the paisa or the batch and serving paths mean different things.
    assert mule.pass_through_ratio == pytest.approx(0.9946524064171123)


def test_the_written_parquet_keeps_money_as_a_decimal(tmp_path: Path):
    """`float64` rupees at the storage boundary would undo every `Decimal` above this line."""
    curated = write_zone(tmp_path / "curated")
    cfg = settings_for(tmp_path)
    with duckdb_engine.connect() as connection:
        table = features.compute_features(
            connection, curated_root=curated, as_of=AS_OF, settings=cfg
        )
    path = features.write_features(table, curated)
    reread = pq.read_table(path)
    assert reread.schema.equals(features.FEATURE_SCHEMA)
    assert reread.num_rows == 8
    assert reread.column("total_credit").type == pa.decimal128(18, 2)
    # Same path the rest of the lakehouse is read by, so a future reader needs no special case.
    assert parquet_store.read_table(curated, features.FEATURES_DIR).num_rows == 8


# --- promotion (T4.1) ---------------------------------------------------------------------------


def passing_gate(status: DqRunStatus = DqRunStatus.PASSED) -> quality.GateResult:
    """A `GateResult` with the verdict under test, built directly rather than by running the gate.

    The twelve checks have their own file. What is being tested here is the single decision
    promotion makes off the result, so the result is the input.
    """
    run = DqRun(
        run_id="DQRUN-featfixture",
        batch_id=BATCH_ID,
        started_at=AS_OF,
        finished_at=AS_OF,
        status=status,
        rows_in=len(_LEDGER),
        rows_loaded=0 if status is DqRunStatus.FAILED else len(_LEDGER),
        rows_rejected=len(_LEDGER) if status is DqRunStatus.FAILED else 0,
        checks_total=12,
        checks_failed=1 if status is DqRunStatus.FAILED else 0,
        report={},
    )
    return quality.GateResult(run=run, outcomes=(), report={}, profile=None)


def test_promotion_populates_curated_and_duckdb_can_query_it(tmp_path: Path):
    raw = write_zone(tmp_path / "raw")
    curated = tmp_path / "curated"
    result = promote.promote(passing_gate(), raw_root=raw, curated_root=curated)

    assert result.transaction_rows == len(_LEDGER)
    assert result.batch_id == BATCH_ID
    with duckdb_engine.connect() as connection:
        counted = connection.execute(
            f"SELECT COUNT(*) FROM {duckdb_engine.read_partitioned('$glob')}",
            {
                "glob": duckdb_engine.dataset_glob(curated, parquet_store.TRANSACTIONS_DIR),
            },
        ).fetchone()
    assert counted is not None
    assert counted[0] == len(_LEDGER)
    assert (curated / parquet_store.ACCOUNTS_DIR / "accounts.parquet").is_file()


def test_promotion_is_idempotent_to_the_byte(tmp_path: Path):
    """Running the same batch twice must replace its partitions, never append a second copy."""
    raw = write_zone(tmp_path / "raw")
    curated = tmp_path / "curated"
    promote.promote(passing_gate(), raw_root=raw, curated_root=curated)
    first = {p.relative_to(curated).as_posix(): p.read_bytes() for p in curated.rglob("*.parquet")}
    second_result = promote.promote(passing_gate(), raw_root=raw, curated_root=curated)
    second = {p.relative_to(curated).as_posix(): p.read_bytes() for p in curated.rglob("*.parquet")}

    assert second_result.transaction_rows == len(_LEDGER)
    assert first == second


class RecordingConnection:
    """A DuckDB connection that keeps every statement handed to it, then delegates.

    `promote` reaches the disk through exactly one method, so proxying that one method is enough.
    It exists for the test below, which has to assert on the SQL rather than on the output bytes.
    """

    def __init__(self, inner: duckdb.DuckDBPyConnection) -> None:
        self.inner = inner
        self.statements: list[str] = []

    def execute(self, sql: str, parameters: object = None) -> object:
        self.statements.append(sql)
        return self.inner.execute(sql, parameters)


def test_every_promotion_copy_is_sorted_so_the_bytes_cannot_drift(tmp_path: Path):
    """The `ORDER BY` in each promotion `COPY` is load-bearing, and is asserted on directly.

    The byte-comparison test above cannot see it. This fixture ledger is eleven rows — far below
    the size at which DuckDB splits a scan across threads — so the output is byte-identical whether
    the sort is there or not, and deleting the line leaves that test green. At 205k rows the sort is
    the only thing keeping two promotions of one batch producing the same bytes, so what is checked
    here is the statement that actually ran.
    """
    raw = write_zone(tmp_path / "raw")
    curated = tmp_path / "curated"
    with duckdb_engine.connect() as connection:
        spy = RecordingConnection(connection)
        promote.promote(
            passing_gate(),
            raw_root=raw,
            curated_root=curated,
            connection=spy,  # type: ignore[arg-type]
        )

    copies = [sql for sql in spy.statements if sql.startswith("COPY")]
    assert len(copies) == 1 + len(promote.REFERENCE_TABLES)

    ledger = next(sql for sql in copies if f"/{parquet_store.TRANSACTIONS_DIR}'" in sql)
    assert "ORDER BY txn_id)" in ledger, (
        "the ledger COPY must sort; without it DuckDB's parallel scan emits rows in whatever order "
        "the threads finished in, and a re-run of one batch produces different bytes"
    )
    # The small tables too. Theirs is the sort nothing else would notice missing, which is exactly
    # why it needs a test rather than a re-run that happened to come out stable.
    for name, schema in promote.REFERENCE_TABLES:
        statement = next(sql for sql in copies if f"/{name}/{name}.parquet'" in sql)
        assert f'ORDER BY "{schema.names[0]}")' in statement


def test_a_blocked_batch_cannot_be_promoted(tmp_path: Path):
    """The gate's verdict is the only thing standing between raw and curated, so it is checked here
    as well as in `run_gate` — a caller that used `evaluate` gets no chance to forget."""
    raw = write_zone(tmp_path / "raw")
    curated = tmp_path / "curated"
    with pytest.raises(DataQualityGateFailed):
        promote.promote(passing_gate(DqRunStatus.FAILED), raw_root=raw, curated_root=curated)
    assert not (curated / parquet_store.TRANSACTIONS_DIR).exists()


def test_ground_truth_is_never_promoted(tmp_path: Path):
    """`typology_labels` exists in the raw zone and must not follow the batch into curated.

    Physical separation, not convention: the detection path reads curated, so it cannot reach the
    labels even by accident. The lint rule at T7.1 is the second line of defence, not the first.
    """
    raw = write_zone(tmp_path / "raw")
    curated = tmp_path / "curated"
    assert (raw / parquet_store.LABELS_DIR).is_dir()
    promote.promote(passing_gate(), raw_root=raw, curated_root=curated)
    assert not (curated / parquet_store.LABELS_DIR).exists()


# --- the dormant cohort ---------------------------------------------------------------------


def test_the_dormant_cohort_separates_from_the_book_on_committed_data(tmp_path: Path):
    """The trap-3 fallback checked as a *cohort*, against labels, on data that ships with the repo.

    This is the same question the real-batch test below asks, and it is duplicated here on purpose:
    that one reads `data/curated/`, which is gitignored, so on a clean checkout it skips and the
    suite still reports green. Coverage that disappears with the working tree is not coverage. The
    fixture's four accounts are enough to pose the question — one labelled dormant-reactivation
    whose gap exists only on the account snapshot, and three ordinary accounts to be the book.
    """
    curated = write_zone(tmp_path / "curated")
    cfg = settings_for(tmp_path)
    with duckdb_engine.connect() as connection:
        table = features.compute_features(
            connection, curated_root=curated, as_of=AS_OF, settings=cfg
        )

    typology = {str(row["account_id"]): row["typology"] for row in label_rows()}
    thirty = [row for row in table.to_pylist() if row["window"] == "30D"]
    assert len(thirty) == table.num_rows // 2

    book = [row["days_since_last_activity"] for row in thirty]
    dormant, ordinary = [], []
    for row in thirty:
        target = dormant if typology[row["account_id"]] == "DORMANT_REACTIVATION" else ordinary
        target.append(row["days_since_last_activity"])
    assert dormant and ordinary

    # The fixture ledger spans 54 days, so no aggregate over it can produce a gap above 180. Any
    # value that large can only have come from `accounts.last_activity_on`.
    assert max(dormant) > 180
    assert mean(dormant) > mean(book), (
        "the fallback is not distinguishing the dormant cohort from the book"
    )
    # The other half of "separates", and the half with teeth. Drop `last_activity_on` from the
    # COALESCE and `ACC00000002` — which has no ledger history before the window either — falls
    # through to its 2019 `opened_on` and reads as a seven-year dormancy. Then every account looks
    # dormant, the cohort means still order correctly, and the separation is worthless. Checking
    # that the ordinary accounts stay *under* the threshold is what notices.
    assert max(ordinary) < 180, (
        "an ordinary account is showing a dormancy gap no 54-day ledger can justify; the fallback "
        "chain has skipped `last_activity_on` and landed on `opened_on`"
    )


# --- the real batch -----------------------------------------------------------------------------

REAL_FEATURES = Path("data/curated/account_features/account_features.parquet")


@pytest.mark.skipif(
    not REAL_FEATURES.is_file(),
    reason=(
        f"SKIPPED, NOT PASSED: {REAL_FEATURES} is absent (it is gitignored). The committed-fixture "
        f"twin of this test still ran. Run `make generate` then `make features` to include the "
        f"6,200-account version."
    ),
)
def test_the_real_batch_separates_the_dormant_cohort():
    """The same cohort check at real scale — an extra, never the only copy.

    Reading `typology_labels` is permitted here and nowhere else — this is `tests/`. The assertion
    is deliberately loose: it checks that the fallback produced gaps no 90-day ledger could express,
    not that detection works, which is Stage 7's job to measure honestly.
    """
    labels = pq.read_table(Path("data/raw") / parquet_store.LABELS_DIR / "typology_labels.parquet")
    frame = pq.read_table(REAL_FEATURES).to_pandas()
    thirty = frame[frame["window"] == "30D"]
    assert len(thirty) == len(frame) // 2

    merged = thirty.merge(labels.to_pandas(), on="account_id", how="left")
    dormant = merged[merged["typology"] == "DORMANT_REACTIVATION"]
    assert not dormant.empty
    # A 90-day ledger cannot hold a 180-day gap, so any value above 180 can only have come from
    # `accounts.last_activity_on`.
    assert dormant["days_since_last_activity"].max() > 180
    assert dormant["days_since_last_activity"].mean() > merged["days_since_last_activity"].mean(), (
        "the fallback is not distinguishing the dormant cohort from the book"
    )
