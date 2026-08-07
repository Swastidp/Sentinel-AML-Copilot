"""T3.3 — proof that a bad batch cannot reach the curated zone.

**The fixtures are built at the DataFrame level, deliberately, and this is the whole trick of the
file.** `Transaction.__post_init__` rejects `amount <= 0` and a naive timestamp at construction, so
a corrupt batch literally cannot be assembled out of domain entities — the domain is doing its job.
The defects this gate exists to catch arrive from *upstream*, in a file somebody else wrote, and the
only way to reproduce one is to build the frame the reader hands over rather than the objects the
writer would have refused to create.

Two layers are tested, and both are needed:

* **Per check**, against a twelve-row frame, so a failure names the check that broke rather than
  "the gate went red".
* **End to end**, through `main()` against a temporary raw zone, so the four properties that
  actually matter are asserted together: the run is marked `FAILED`, the process exits non-zero,
  the report is still written, and `rows_loaded` is zero.

The clean-batch assertion runs against the real generated population when it is on disk and skips
with a message when it is not, because CI has no `data/raw/`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentinel.config.settings import Settings
from sentinel.domain.enums import DqCheckStatus, DqRunStatus, DqSeverity
from sentinel.domain.exceptions import DataQualityGateFailed, DomainValidationError
from sentinel.domain.value_objects import BatchProfile
from sentinel.infrastructure.storage import parquet_store
from sentinel.infrastructure.storage.dq_store import FileSystemDqStore
from sentinel.pipeline import checks, quality

# --- fixture construction ----------------------------------------------------------------------

BATCH_ID = "BATCH-20260801-20260802-testfix"
WINDOW_START = date(2026, 8, 1)
WINDOW_END = date(2026, 8, 2)
INGESTED_AT = datetime(2026, 8, 2, 23, 59, 59, tzinfo=UTC)
ACCOUNT_IDS = frozenset({"ACC00000001", "ACC00000002", "ACC00000003", "ACC00000004"})


def clean_frame() -> pd.DataFrame:
    """Twelve rows that pass all eight non-drift checks.

    Shaped like the real ledger rather than minimally: external credits with a null `from_account`,
    external debits with a null `to_account`, and internal `IDFB` transfers naming two known
    accounts and carrying a null `counterparty_handle` — which check 9 must **not** flag.
    """
    rows: list[dict[str, object]] = []
    for index in range(12):
        internal = index % 3 == 0
        outbound = index % 2 == 0
        day = WINDOW_START if index < 6 else WINDOW_END
        rows.append(
            {
                "txn_id": f"TXN{index:012d}",
                "txn_ts": datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                + timedelta(hours=9, minutes=index),
                "from_account": "ACC00000001" if internal or outbound else None,
                "to_account": "ACC00000002" if internal or not outbound else None,
                # An internal transfer has no external VPA. This is by design, not a defect.
                "counterparty_handle": None if internal else f"payee{index}@upi",
                "counterparty_bank": checks.INTERNAL_BANK_CODE if internal else "HDFC",
                "amount": Decimal(f"{(index + 1) * 250}.50"),
                "currency": "INR",
                "channel": "UPI" if index % 4 else "NEFT",
                "direction": "DEBIT" if outbound else "CREDIT",
                "status": "SUCCESS",
                "device_id": f"DEV{index:06d}" if index % 2 else None,
                "ip_address": "10.0.0.1" if index % 2 else None,
                "narration": f"TEST/{index}",
                "txn_date": day,
                "ingested_at": INGESTED_AT,
                "batch_id": BATCH_ID,
            }
        )
    frame = pd.DataFrame(rows)
    # `object` for every text column so a corruption can write `None` into one without pandas
    # coercing it, and so the frame matches what pyarrow hands back for a nullable string column.
    for column in (
        "txn_id",
        "from_account",
        "to_account",
        "counterparty_handle",
        "counterparty_bank",
        "currency",
        "channel",
        "direction",
        "status",
        "device_id",
        "ip_address",
        "narration",
        "batch_id",
    ):
        frame[column] = frame[column].astype(object)
    return frame


def settings_for(tmp_path: Path) -> Settings:
    """A `Settings` pointed at a temporary lakehouse, reading no `.env` at all."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_raw_path=tmp_path / "raw",
        data_curated_path=tmp_path / "curated",
        data_quality_path=tmp_path / "quality",
    )


def context_for(
    frame: pd.DataFrame,
    tmp_path: Path,
    *,
    baseline: tuple[BatchProfile, ...] = (),
    account_ids: frozenset[str] = ACCOUNT_IDS,
) -> checks.CheckContext:
    return checks.build_context(
        batch_id=BATCH_ID,
        transactions=frame,
        account_ids=account_ids,
        baseline=baseline,
        settings=settings_for(tmp_path),
    )


def write_raw_zone(root: Path, frame: pd.DataFrame, account_ids: frozenset[str]) -> None:
    """Land a frame in a raw zone the gate can read, bypassing the domain entities entirely.

    Uses `parquet_store`'s own schemas so the bytes are the shape the real writer produces —
    including `decimal128(18, 2)` amounts and `txn_date` as the hive partition key, which comes back
    as the *last* column and is the reason nothing in the gate selects a column by position.
    """
    transactions_dir = root / parquet_store.TRANSACTIONS_DIR
    table = pa.Table.from_pandas(
        frame, schema=parquet_store.TRANSACTION_SCHEMA, preserve_index=False
    )
    pq.write_to_dataset(
        table,
        root_path=str(transactions_dir),
        partition_cols=[parquet_store.PARTITION_COLUMN],
        existing_data_behavior="delete_matching",
        basename_template="fixture-{i}.parquet",
    )
    accounts_dir = root / parquet_store.ACCOUNTS_DIR
    accounts_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "account_id": account_id,
                    "customer_id": "CUST000001",
                    "account_type": "SAVINGS",
                    "opened_on": date(2020, 1, 1),
                    "status": "ACTIVE",
                    "branch_code": "BR001",
                    "ifsc": "IDFB0000001",
                    "current_balance": Decimal("1000.00"),
                    "last_activity_on": WINDOW_START,
                    "is_flagged": False,
                    "created_at": datetime(2020, 1, 1, tzinfo=UTC),
                }
                for account_id in sorted(account_ids)
            ],
            schema=parquet_store.ACCOUNT_SCHEMA,
        ),
        accounts_dir / f"{parquet_store.ACCOUNTS_DIR}.parquet",
    )


def baseline_profile(
    frame: pd.DataFrame, tmp_path: Path, *, batch_id: str = "BATCH-BASELINE"
) -> BatchProfile:
    """A profile computed from `frame`, as a stand-in for a previous day's batch."""
    profile = checks.compute_profile(
        context_for(frame, tmp_path), datetime(2026, 7, 31, tzinfo=UTC)
    )
    return BatchProfile(
        batch_id=batch_id,
        computed_at=profile.computed_at,
        window_end=date(2026, 7, 31),
        row_count=profile.row_count,
        null_rates=profile.null_rates,
        channel_mix=profile.channel_mix,
        amount_deciles=profile.amount_deciles,
    )


def outcome_for(context: checks.CheckContext, check_id: str) -> checks.CheckOutcome:
    """Run the full suite and pick one result out by id."""
    matches = [item for item in checks.run_all(context) if item.check_id == check_id]
    assert len(matches) == 1, f"{check_id} did not produce exactly one outcome"
    return matches[0]


# --- the framework -----------------------------------------------------------------------------


def test_registry_holds_exactly_twelve_checks_with_unique_ids():
    context = context_for(clean_frame(), Path())
    outcomes = checks.run_all(context)
    assert len(checks.ALL_CHECKS) == 12
    assert len(outcomes) == 12
    assert len({outcome.check_id for outcome in outcomes}) == 12


def test_check_ids_are_numbered_in_the_documented_order():
    # The check table in docs/05-DATA-FLOW.md is numbered 1..12 and the report is read against it,
    # so a reordered registry would silently renumber someone's runbook.
    ids = [outcome.check_id for outcome in checks.run_all(context_for(clean_frame(), Path()))]
    assert [item.split("_", 1)[0] for item in ids] == [f"DQ{n:02d}" for n in range(1, 13)]


def test_severities_match_the_documented_table():
    by_id = {o.check_id: o.severity for o in checks.run_all(context_for(clean_frame(), Path()))}
    blocking = [k for k, v in by_id.items() if v is DqSeverity.BLOCKING]
    warning = [k for k, v in by_id.items() if v is DqSeverity.WARNING]
    assert sorted(item[:4] for item in blocking) == [
        "DQ01",
        "DQ02",
        "DQ03",
        "DQ04",
        "DQ05",
        "DQ06",
        "DQ07",
        "DQ08",
    ]
    assert sorted(item[:4] for item in warning) == ["DQ09", "DQ10", "DQ11", "DQ12"]


def test_an_outcome_must_carry_a_message():
    with pytest.raises(DomainValidationError):
        checks.CheckOutcome(
            check_id="DQ00_x",
            name="x",
            severity=DqSeverity.WARNING,
            status=DqCheckStatus.PASS,
            target_table="transactions",
            target_column=None,
            expectation="",
            observed="",
            message="   ",
        )


def test_a_skipped_check_never_blocks_the_batch():
    # The property the whole SKIPPED status exists for: four BLOCKING drift checks cannot run on a
    # first batch, and that must not be expressible as "the blocking gate was satisfied".
    outcome = outcome_for(context_for(clean_frame(), Path()), "DQ01_row_count_tolerance")
    assert outcome.status is DqCheckStatus.SKIPPED
    assert outcome.severity is DqSeverity.BLOCKING
    assert outcome.blocks_batch is False
    assert outcome.is_unverified is True
    assert quality.decide_status([outcome]) is DqRunStatus.PASSED_WITH_WARNINGS


def test_window_is_derived_from_the_batch_not_the_clock():
    window = checks.derive_window(clean_frame())
    assert window.start == datetime(2026, 8, 1, tzinfo=UTC)
    assert window.end == INGESTED_AT
    assert window.end_date == WINDOW_END


def test_window_derivation_rejects_an_empty_batch():
    with pytest.raises(DomainValidationError):
        checks.derive_window(clean_frame().iloc[0:0])


def test_clean_fixture_passes_every_non_drift_check():
    outcomes = checks.run_all(context_for(clean_frame(), Path()))
    failures = [o.check_id for o in outcomes if o.status is DqCheckStatus.FAIL]
    skipped = [o.check_id for o in outcomes if o.is_unverified]
    assert failures == []
    assert len(skipped) == 4  # the four baseline-dependent checks, and only those


# --- the eight blocking checks -----------------------------------------------------------------


def test_check_02_catches_a_duplicate_txn_id():
    frame = clean_frame()
    frame.loc[5, "txn_id"] = frame.loc[4, "txn_id"]
    outcome = outcome_for(context_for(frame, Path()), "DQ02_txn_id_unique")
    assert outcome.status is DqCheckStatus.FAIL
    assert outcome.blocks_batch
    assert "1 duplicate" in outcome.observed


def test_check_03_catches_a_null_in_a_mandatory_field():
    frame = clean_frame()
    frame.loc[2, "channel"] = None
    outcome = outcome_for(context_for(frame, Path()), "DQ03_mandatory_not_null")
    assert outcome.status is DqCheckStatus.FAIL
    assert "channel=1" in outcome.observed


def test_check_03_and_07_do_not_double_count_the_same_null():
    # A null channel is one defect. Counting it as both a missing mandatory field and an unknown
    # enum value would inflate `checks_failed` and mislead whoever is triaging the feed.
    frame = clean_frame()
    frame.loc[2, "channel"] = None
    outcome = outcome_for(context_for(frame, Path()), "DQ07_enum_domains")
    assert outcome.status is DqCheckStatus.PASS


def test_check_04_catches_a_negative_amount():
    frame = clean_frame()
    frame.loc[0, "amount"] = Decimal("-500.00")
    outcome = outcome_for(context_for(frame, Path()), "DQ04_amount_bounds")
    assert outcome.status is DqCheckStatus.FAIL
    assert "1 non-positive" in outcome.observed


def test_check_04_catches_zero_and_the_ten_crore_cap_in_decimal():
    frame = clean_frame()
    frame.loc[0, "amount"] = Decimal("0.00")
    frame.loc[1, "amount"] = Decimal("100000000.01")  # one paisa over Rs 10 crore
    frame.loc[2, "amount"] = Decimal("100000000.00")  # exactly at the cap - legal
    outcome = outcome_for(context_for(frame, Path()), "DQ04_amount_bounds")
    assert outcome.status is DqCheckStatus.FAIL
    assert "1 non-positive" in outcome.observed
    assert "1 above cap" in outcome.observed


def test_check_04_counts_an_amount_that_is_not_a_decimal_as_a_violation():
    # A bounds check that silently drops what it cannot compare is a check that cannot fail.
    # Unreachable through decimal128 Parquet, reachable the moment a CSV or JSON feed appears.
    frame = clean_frame()
    frame.loc[0, "amount"] = -500.0  # a float, not a Decimal
    outcome = outcome_for(context_for(frame, Path()), "DQ04_amount_bounds")
    assert outcome.status is DqCheckStatus.FAIL
    assert outcome.blocks_batch
    assert "1 not decimal" in outcome.observed


def test_check_05_catches_an_orphan_account_reference():
    frame = clean_frame()
    frame.loc[1, "from_account"] = "ACC99999999"
    outcome = outcome_for(context_for(frame, Path()), "DQ05_referential_integrity")
    assert outcome.status is DqCheckStatus.FAIL
    assert outcome.blocks_batch
    assert "from_account=1" in outcome.observed


def test_check_06_catches_a_future_dated_timestamp():
    frame = clean_frame()
    frame.loc[3, "txn_ts"] = INGESTED_AT + timedelta(days=30)
    outcome = outcome_for(context_for(frame, Path()), "DQ06_txn_ts_in_window")
    assert outcome.status is DqCheckStatus.FAIL
    assert "1 future-dated" in outcome.observed


def test_check_06_catches_a_timestamp_before_the_window():
    frame = clean_frame()
    frame.loc[3, "txn_ts"] = datetime(2025, 1, 1, tzinfo=UTC)
    outcome = outcome_for(context_for(frame, Path()), "DQ06_txn_ts_in_window")
    assert outcome.status is DqCheckStatus.FAIL
    assert "1 early" in outcome.observed


def test_check_07_catches_an_unknown_channel_and_direction():
    frame = clean_frame()
    frame.loc[0, "channel"] = "CRYPTO"
    frame.loc[1, "direction"] = "SIDEWAYS"
    outcome = outcome_for(context_for(frame, Path()), "DQ07_enum_domains")
    assert outcome.status is DqCheckStatus.FAIL
    assert "channel=1" in outcome.observed
    assert "direction=1" in outcome.observed


def test_check_08_catches_a_foreign_currency_and_a_null_one():
    frame = clean_frame()
    frame.loc[0, "currency"] = "USD"
    frame.loc[1, "currency"] = None
    outcome = outcome_for(context_for(frame, Path()), "DQ08_currency_inr")
    assert outcome.status is DqCheckStatus.FAIL
    assert "2 rows not INR" in outcome.observed


# --- check 9, the rewritten one ----------------------------------------------------------------


def test_check_09_does_not_flag_a_null_counterparty_handle_on_an_internal_row():
    # The regression this check was rewritten to avoid: `counterparty_handle` is null on every
    # internal transfer by design, and flagging it would make the check fire on a clean batch.
    frame = clean_frame()
    internal = frame[frame["counterparty_bank"].eq(checks.INTERNAL_BANK_CODE)]
    assert len(internal) > 0
    assert internal["counterparty_handle"].isna().all()
    outcome = outcome_for(context_for(frame, Path()), "DQ09_internal_edge_integrity")
    assert outcome.status is DqCheckStatus.PASS


def test_check_09_catches_an_internal_transfer_missing_an_endpoint():
    frame = clean_frame()
    internal_index = frame.index[frame["counterparty_bank"].eq(checks.INTERNAL_BANK_CODE)][0]
    frame.loc[internal_index, "to_account"] = None
    outcome = outcome_for(context_for(frame, Path()), "DQ09_internal_edge_integrity")
    assert outcome.status is DqCheckStatus.FAIL
    assert outcome.severity is DqSeverity.WARNING
    assert outcome.blocks_batch is False


def test_check_09_catches_an_internal_transfer_naming_an_unknown_account():
    frame = clean_frame()
    internal_index = frame.index[frame["counterparty_bank"].eq(checks.INTERNAL_BANK_CODE)][0]
    frame.loc[internal_index, "to_account"] = "ACC77777777"
    outcome = outcome_for(context_for(frame, Path()), "DQ09_internal_edge_integrity")
    assert outcome.status is DqCheckStatus.FAIL
    assert "naming an unknown account" in outcome.observed


def test_check_09_counts_defective_rows_not_defective_endpoints():
    # One internal transfer with both endpoints wrong is ONE defective row. Summing per-endpoint
    # counts let `observed` report more defects than the batch has internal rows.
    frame = clean_frame()
    internal = frame.index[frame["counterparty_bank"].eq(checks.INTERNAL_BANK_CODE)]
    frame.loc[internal[0], "from_account"] = "ACC77777777"
    frame.loc[internal[0], "to_account"] = "ACC88888888"
    outcome = outcome_for(context_for(frame, Path()), "DQ09_internal_edge_integrity")
    assert outcome.status is DqCheckStatus.FAIL
    assert f"1 defective of {len(internal):,} internal rows" in outcome.observed


# --- the four drift checks ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "check_id",
    [
        "DQ01_row_count_tolerance",
        "DQ10_null_rate_drift",
        "DQ11_channel_mix_drift",
        "DQ12_amount_distribution_drift",
    ],
)
def test_drift_checks_are_skipped_without_a_baseline(check_id):
    outcome = outcome_for(context_for(clean_frame(), Path()), check_id)
    assert outcome.status is DqCheckStatus.SKIPPED
    assert "no baseline available" in outcome.observed


@pytest.mark.parametrize(
    "check_id",
    [
        "DQ01_row_count_tolerance",
        "DQ10_null_rate_drift",
        "DQ11_channel_mix_drift",
        "DQ12_amount_distribution_drift",
    ],
)
def test_drift_checks_pass_against_an_identical_baseline(check_id, tmp_path):
    frame = clean_frame()
    baseline = (baseline_profile(frame, tmp_path),)
    outcome = outcome_for(context_for(frame, tmp_path, baseline=baseline), check_id)
    assert outcome.status is DqCheckStatus.PASS, outcome.message


def test_check_01_fails_when_the_row_count_halves(tmp_path):
    frame = clean_frame()
    baseline = (baseline_profile(frame, tmp_path),)
    shrunk = frame.iloc[:4].copy()  # 4 of 12 rows is -67%, well past the +/-40% tolerance
    outcome = outcome_for(
        context_for(shrunk, tmp_path, baseline=baseline), "DQ01_row_count_tolerance"
    )
    assert outcome.status is DqCheckStatus.FAIL
    assert outcome.blocks_batch


def test_check_10_fails_when_an_optional_column_stops_arriving(tmp_path):
    frame = clean_frame()
    baseline = (baseline_profile(frame, tmp_path),)
    blanked = frame.copy()
    blanked["device_id"] = None  # 50% populated in the baseline, 0% now: a 50pp move
    outcome = outcome_for(context_for(blanked, tmp_path, baseline=baseline), "DQ10_null_rate_drift")
    assert outcome.status is DqCheckStatus.FAIL
    assert "device_id" in outcome.observed


def test_check_11_fails_when_the_channel_mix_flips(tmp_path):
    frame = clean_frame()
    baseline = (baseline_profile(frame, tmp_path),)
    shifted = frame.copy()
    shifted["channel"] = "RTGS"
    outcome = outcome_for(
        context_for(shifted, tmp_path, baseline=baseline), "DQ11_channel_mix_drift"
    )
    assert outcome.status is DqCheckStatus.FAIL
    assert "RTGS" in outcome.observed


def test_check_12_fails_when_the_amount_distribution_shifts(tmp_path):
    frame = clean_frame()
    baseline = (baseline_profile(frame, tmp_path),)
    inflated = frame.copy()
    inflated["amount"] = [value * 1000 for value in inflated["amount"]]
    outcome = outcome_for(
        context_for(inflated, tmp_path, baseline=baseline), "DQ12_amount_distribution_drift"
    )
    assert outcome.status is DqCheckStatus.FAIL
    assert "D=" in outcome.observed


# --- the gate, end to end ----------------------------------------------------------------------

CORRUPTIONS: dict[str, tuple[str, object, str]] = {
    "negative_amount": ("amount", Decimal("-1.00"), "DQ04_amount_bounds"),
    "null_mandatory_field": ("channel", None, "DQ03_mandatory_not_null"),
    "orphan_account": ("from_account", "ACC66666666", "DQ05_referential_integrity"),
    "future_timestamp": ("txn_ts", INGESTED_AT + timedelta(days=365), "DQ06_txn_ts_in_window"),
}


def corrupt(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    corrupted = frame.copy()
    corrupted.loc[0, column] = value
    return corrupted


@pytest.mark.parametrize(("label", "spec"), sorted(CORRUPTIONS.items()))
def test_a_blocking_corruption_fails_the_run_exits_non_zero_and_still_writes_the_report(
    label, spec, tmp_path
):
    column, value, expected_check = spec
    settings = settings_for(tmp_path)
    write_raw_zone(settings.data_raw_path, corrupt(clean_frame(), column, value), ACCOUNT_IDS)

    with pytest.raises(SystemExit) as exit_info:
        quality.main(settings)
    assert exit_info.value.code == 1, f"{label} must exit non-zero"

    store = FileSystemDqStore(settings.data_quality_path)
    report = _read_report(store.report_path)
    assert report["status"] == str(DqRunStatus.FAILED)
    assert report["rows_in"] == 12
    assert report["rows_loaded"] == 0, "a blocked batch must promote nothing"
    assert report["rows_rejected"] == 12
    failed = {item["check_id"] for item in report["checks"] if item["status"] == "FAIL"}
    assert expected_check in failed, f"{label} should have tripped {expected_check}, got {failed}"
    # Nothing is promoted, and the audit trail survived the failure.
    assert _persisted_runs(settings) == 1
    assert _persisted_results(settings) == 12


def test_a_duplicate_txn_id_blocks_the_batch(tmp_path):
    # Kept out of the parametrised set because it needs a second row touched, not one cell.
    frame = clean_frame()
    frame.loc[1, "txn_id"] = frame.loc[0, "txn_id"]
    settings = settings_for(tmp_path)
    write_raw_zone(settings.data_raw_path, frame, ACCOUNT_IDS)

    with pytest.raises(SystemExit) as exit_info:
        quality.main(settings)
    assert exit_info.value.code == 1
    report = _read_report(FileSystemDqStore(settings.data_quality_path).report_path)
    assert report["status"] == str(DqRunStatus.FAILED)
    assert "DQ02_txn_id_unique" in {
        item["check_id"] for item in report["checks"] if item["status"] == "FAIL"
    }


def test_a_blocked_batch_loads_no_rows_and_writes_no_baseline_profile(tmp_path):
    settings = settings_for(tmp_path)
    write_raw_zone(
        settings.data_raw_path, corrupt(clean_frame(), "amount", Decimal("-1")), ACCOUNT_IDS
    )
    store = FileSystemDqStore(settings.data_quality_path)

    with pytest.raises(DataQualityGateFailed) as failure:
        quality.run_gate(settings, store=store)

    assert failure.value.batch_id == BATCH_ID
    assert "DQ04_amount_bounds" in failure.value.failures
    # A batch the gate refused is not a description of normal, so it must not widen the baseline
    # that would judge the next one.
    assert store.recent_profiles(limit=7) == ()


def test_a_warning_only_corruption_is_recorded_and_does_not_block(tmp_path):
    # A broken internal edge degrades the Stage 8 graph load; it does not corrupt the ledger. The
    # batch proceeds and the defect is on the record.
    frame = clean_frame()
    internal_index = frame.index[frame["counterparty_bank"].eq(checks.INTERNAL_BANK_CODE)][0]
    frame.loc[internal_index, "to_account"] = None
    settings = settings_for(tmp_path)
    write_raw_zone(settings.data_raw_path, frame, ACCOUNT_IDS)
    store = FileSystemDqStore(settings.data_quality_path)

    result = quality.run_gate(settings, store=store)  # does not raise

    assert result.blocked is False
    assert result.run.status is DqRunStatus.PASSED_WITH_WARNINGS
    assert result.run.rows_loaded == 12
    assert [outcome.check_id for outcome in result.warnings] == ["DQ09_internal_edge_integrity"]
    report = _read_report(store.report_path)
    assert any(
        item["check_id"] == "DQ09_internal_edge_integrity" and item["status"] == "FAIL"
        for item in report["checks"]
    )
    # It proceeded, so it is a legitimate baseline for the next batch.
    assert len(store.recent_profiles(limit=7)) == 1


def test_a_second_batch_is_measured_against_the_first_batchs_profile(tmp_path):
    """The baseline actually round-trips: run two batches and the drift checks stop being skipped.

    Without this the `SKIPPED` status would be permanent and four checks would be decoration.
    """
    settings = settings_for(tmp_path)
    store = FileSystemDqStore(settings.data_quality_path)

    write_raw_zone(settings.data_raw_path, clean_frame(), ACCOUNT_IDS)
    first = quality.run_gate(settings, store=store)
    assert len(first.unverified) == 4

    second_frame = clean_frame()
    second_frame["batch_id"] = "BATCH-20260802-20260803-second0"
    write_raw_zone(settings.data_raw_path, second_frame, ACCOUNT_IDS)
    second = quality.run_gate(settings, store=store)

    assert second.unverified == ()
    assert second.run.status is DqRunStatus.PASSED
    assert second.report["baseline_batches"] == 1


def test_the_gate_refuses_an_empty_raw_zone(tmp_path):
    # An empty batch must fail loudly rather than sail through twelve vacuously-true checks and
    # report a clean pass on nothing at all.
    settings = settings_for(tmp_path)
    write_raw_zone(settings.data_raw_path, clean_frame(), ACCOUNT_IDS)
    for partition in (settings.data_raw_path / parquet_store.TRANSACTIONS_DIR).rglob("*.parquet"):
        partition.unlink()
    with pytest.raises(DomainValidationError):
        quality.run_gate(settings, store=FileSystemDqStore(settings.data_quality_path))


def test_persisted_check_results_mirror_the_dq_check_results_columns(tmp_path):
    settings = settings_for(tmp_path)
    write_raw_zone(settings.data_raw_path, clean_frame(), ACCOUNT_IDS)
    quality.run_gate(settings, store=FileSystemDqStore(settings.data_quality_path))

    results_dir = settings.data_quality_path / "dq_check_results"
    table = pq.read_table(next(results_dir.glob("*.parquet")))
    assert table.column_names == [
        "result_id",
        "run_id",
        "check_name",
        "target_table",
        "target_column",
        "severity",
        "status",
        "expectation",
        "observed",
    ]
    # `result_id` is a bigserial in Postgres; inventing values here would collide at Stage 5.
    assert table.column("result_id").null_count == table.num_rows


# --- the real generated batch ------------------------------------------------------------------

REAL_RAW = Path("data/raw")


@pytest.mark.skipif(
    not (REAL_RAW / "transactions").is_dir(),
    reason="no generated raw zone on disk; run `python -m sentinel.pipeline.generate` first",
)
def test_the_real_generated_batch_passes_every_blocking_check(tmp_path):
    """Ground truth from `docs/05-DATA-FLOW.md` §2: seed 20260803 is a verified-clean population.

    The drift checks are expected to be `SKIPPED` here — this is a single generated batch with no
    history — and the assertion is deliberately on *blocking failures*, not on a clean sweep.
    """
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_raw_path=REAL_RAW,
        data_quality_path=tmp_path / "quality",
    )
    result = quality.run_gate(settings, store=FileSystemDqStore(settings.data_quality_path))

    assert result.blocked is False
    assert result.blocking_failures == ()
    assert [outcome.check_id for outcome in result.warnings] == []
    assert result.run.rows_in == 205_419
    assert result.run.rows_loaded == 205_419
    assert len(result.unverified) == 4


# --- small helpers -----------------------------------------------------------------------------


def _read_report(path: Path) -> dict:
    import json

    assert path.is_file(), f"the report must be written even when the batch is blocked: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _persisted_runs(settings: Settings) -> int:
    return len(list((settings.data_quality_path / "dq_runs").glob("*.parquet")))


def _persisted_results(settings: Settings) -> int:
    table = pq.read_table(next((settings.data_quality_path / "dq_check_results").glob("*.parquet")))
    return int(table.num_rows)
