"""Live-database tests for Stage 5. Marked `integration`, so `make test` never runs them.

These are the acceptance tests for T5.1, T5.2 and T5.3, and each is written the way the roadmap
words it:

* T5.1 -- "running it twice is a no-op": apply the migrations twice and compare the resulting schema
  state object, not a log line.
* T5.2 -- "loading the same batch twice leaves identical row counts": run the whole load twice and
  compare the two reports.
* T5.3 -- "counts match; a sample row inspected": `verify` raises unless Postgres agrees with DuckDB
  and the money column round-trips digit for digit.

**Nothing here deletes anything.** `DATABASE_URL` points at a live database and the project rule is
additive-only, which shapes one test in particular: the data-quality store test writes a *fixed*
sentinel `run_id` rather than a random one, so re-running the suite upserts the same single row
instead of accumulating a new one every time. That row is left behind on purpose -- it is the
cheapest honest alternative to a cleanup that the rules forbid.

They are slow. The full load moves 205,419 transactions across the network twice, and Neon
auto-suspends, so a cold first connection takes around 16 seconds against 3 warm. That is the
database waking up, not a failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from sentinel.config.settings import get_settings
from sentinel.domain.entities import DqCheckResult, DqRun
from sentinel.domain.enums import DqCheckStatus, DqRunStatus, DqSeverity
from sentinel.domain.exceptions import ConfigurationError, DomainValidationError
from sentinel.infrastructure.postgres import connection as pg
from sentinel.infrastructure.postgres import loader, migrate
from sentinel.infrastructure.postgres.dq_store import PostgresDqStore
from sentinel.infrastructure.postgres.repositories import (
    PostgresAccountRepository,
    PostgresDqRepository,
    PostgresTransactionRepository,
)
from sentinel.infrastructure.storage.dq_store import FileSystemDqStore
from sentinel.pipeline import load

pytestmark = pytest.mark.integration

#: A fixed identity, so the store test is idempotent against a database it may not clean up.
SELFTEST_RUN_ID = "DQ-SELFTEST-0001"


@pytest.fixture(scope="module")
def settings():
    cfg = get_settings()
    try:
        assert cfg.postgres_dsn
    except ConfigurationError as exc:
        pytest.skip(f"DATABASE_URL is not configured: {exc}")
    return cfg


@pytest.fixture(scope="module")
def connection(settings):
    with pg.connect(settings) as conn:
        migrate.apply(conn)
        yield conn


# --- T5.1 ---------------------------------------------------------------------------------------


def test_running_the_migration_twice_is_a_no_op(connection):
    first = migrate.apply(connection)
    second = migrate.apply(connection)
    assert first == second
    assert len(first.tables) == 12
    assert first.partitions >= 13  # twelve months of 2026 plus the DEFAULT partition


def test_the_schema_holds_all_twelve_modelled_tables(connection):
    state = migrate.inspect(connection)
    assert set(state.tables) == {
        "customers",
        "accounts",
        "devices",
        "transactions",
        "account_features",
        "alerts",
        "investigations",
        "tool_calls",
        "human_decisions",
        "dq_runs",
        "dq_check_results",
        "typology_labels",
    }


# --- T5.2 and T5.3 -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded(settings):
    """One full load, shared by the tests below. Slow -- about a minute against a warm Neon."""
    return load.run(settings)


def test_postgres_counts_match_duckdb(loaded):
    for result in loaded.tables:
        assert result.matches, (
            f"{result.table}: duckdb={result.duckdb_rows} pg={result.postgres_rows}"
        )
    assert {result.table: result.postgres_rows for result in loaded.tables} == {
        "customers": 5_000,
        "accounts": 6_200,
        "devices": 2_500,
        "transactions": 205_419,
        "account_features": 12_400,
    }


def test_money_round_trips_to_the_paisa(loaded):
    """Not "close enough". `decimal128(18, 2)` and `numeric(18, 2)` agree digit for digit."""
    assert loaded.money.exact
    assert loaded.money.duckdb_amount == loaded.money.postgres_amount
    assert str(loaded.money.duckdb_amount) == str(loaded.money.postgres_amount)
    assert loaded.money.duckdb_amount % 1 != 0, "the probe must carry paisa or it tests nothing"


def test_reloading_the_same_batch_leaves_identical_counts(loaded, settings):
    """The T5.2 acceptance criterion, run rather than claimed."""
    again = load.run(settings)
    before = {result.table: result.postgres_rows for result in loaded.tables}
    after = {result.table: result.postgres_rows for result in again.tables}
    assert before == after
    assert again.money.exact


def test_the_ground_truth_table_stays_empty(loaded, connection):
    """The eval boundary: nothing the detection or scoring path can read may hold a label."""
    assert loaded.typology_label_rows == 0
    assert loader.count(connection, "typology_labels") == 0


def test_the_sample_rows_are_real(loaded):
    assert loaded.sample_transaction["amount"] == loaded.money.postgres_amount
    assert loaded.sample_feature["feature_window"] == "7D"
    # `feature_window`, not `window` -- decision 1, observed on a real row.
    assert "window" not in loaded.sample_feature


# --- repositories over the loaded data ----------------------------------------------------------


def test_account_repository_reads_a_loaded_account(loaded, connection):
    repository = PostgresAccountRepository(connection)
    account_id = loaded.sample_feature["account_id"]
    account = repository.get(account_id)
    assert account is not None
    assert account.account_id == account_id
    assert isinstance(account.current_balance, Decimal)
    assert repository.get("ACC-does-not-exist") is None


def test_account_profile_joins_the_customer(loaded, connection):
    repository = PostgresAccountRepository(connection)
    profile = repository.get_profile(loaded.sample_feature["account_id"])
    assert profile is not None
    assert profile.tenure_months >= 0
    assert isinstance(profile.declared_annual_income, Decimal)


def test_transaction_repository_reads_both_legs_in_a_window(loaded, connection):
    repository = PostgresTransactionRepository(connection)
    sample = loaded.sample_transaction
    end = sample["txn_ts"] + timedelta(seconds=1)
    start = end - timedelta(days=1)
    rows = repository.list_for_account(sample["from_account"], start=start, end=end, limit=50)
    assert any(row.txn_id == sample["txn_id"] for row in rows)
    # Newest first, total order -- the contract paging depends on.
    assert list(rows) == sorted(rows, key=lambda t: (t.txn_ts, t.txn_id), reverse=True)


def test_transaction_repository_refuses_a_naive_window(connection):
    repository = PostgresTransactionRepository(connection)
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        repository.list_for_account(
            "ACC00000001", start=datetime(2026, 5, 1), end=datetime(2026, 5, 2), limit=1
        )


def test_get_on_a_missing_transaction_is_none_not_an_error(connection):
    assert PostgresTransactionRepository(connection).get("TXN-nope") is None


# --- the Postgres DqReportStore -------------------------------------------------------------


def test_dq_store_persists_a_run_and_upserts_on_a_replay(connection, tmp_path):
    """Writes one fixed sentinel run, twice, and proves the second changes nothing.

    A retried GitHub Actions step replays an identical run. It must leave one row and one result,
    not two of each -- which is exactly why `dq_check_results` carries a unique constraint on
    `(run_id, check_name)` rather than relying on its `bigserial`.
    """
    store = PostgresDqStore(connection, FileSystemDqStore(tmp_path))
    started = datetime(2026, 8, 3, 23, 59, 59, tzinfo=UTC)
    run = DqRun(
        run_id=SELFTEST_RUN_ID,
        batch_id="SELFTEST",
        started_at=started,
        finished_at=started,
        status=DqRunStatus.PASSED,
        rows_in=1,
        rows_loaded=1,
        rows_rejected=0,
        checks_total=1,
        checks_failed=0,
        report={"note": "integration self-test; fixed id so a re-run upserts"},
    )
    result = DqCheckResult(
        result_id=None,
        run_id=SELFTEST_RUN_ID,
        check_name="selftest_probe",
        target_table="transactions",
        target_column=None,
        severity=DqSeverity.WARNING,
        status=DqCheckStatus.PASS,
        expectation="the adapter writes",
        observed="it wrote",
    )
    store.save_run(run, [result])
    store.save_run(run, [result])

    repository = PostgresDqRepository(connection)
    results = repository.list_check_results(SELFTEST_RUN_ID)
    assert len(results) == 1
    assert results[0].check_name == "selftest_probe"
    # `result_id` is assigned by the database, exactly as the Parquet writer's null anticipated.
    assert results[0].result_id is not None

    runs = [row for row in repository.list_runs(limit=100) if row.run_id == SELFTEST_RUN_ID]
    assert len(runs) == 1
    assert runs[0].status is DqRunStatus.PASSED
    assert runs[0].report["note"].startswith("integration self-test")
