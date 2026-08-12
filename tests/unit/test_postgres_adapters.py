"""Offline tests for the Postgres adapters -- port conformance and row mapping.

Two things are checkable without a database, and both are worth checking here rather than only in
the integration suite that CI skips:

**Conformance.** Every repository is bound to its port at runtime. That catches a missing or
renamed method. It does *not* check signatures -- `@runtime_checkable` cannot, as
`tests/unit/test_ports.py::test_isinstance_ignores_signatures` demonstrates -- so the static half
lives in `repositories._conformance` and `dq_store._conformance`, which `mypy --strict` checks over
`src`. Neither half is sufficient alone.

**Row mapping.** `_to_account` and friends are pure functions from a dict to an entity, which makes
them the cheapest possible test and the most likely thing to break: a renamed column shows up here
as a `KeyError` instead of at 205,419 rows into a load.

`_rows` in the loader gets its own test for the trap the pipeline keeps hitting -- pyarrow returns a
partitioned table's partition key as the **last** column, wherever the write schema put it, so the
loader must select by name. The test builds a table with the columns deliberately out of order.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from sentinel.domain import ports
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    DqCheckStatus,
    DqSeverity,
    InvestigationStatus,
    Recommendation,
    Typology,
)
from sentinel.infrastructure.postgres import loader, repositories
from sentinel.infrastructure.postgres.dq_store import PostgresDqStore
from sentinel.pipeline.quality import DqReportStore

# The adapters never touch the connection at construction time, so `None` is a fine stand-in for
# an `isinstance` check. Anything that did touch it would fail loudly here rather than silently.
_ADAPTERS = [
    (repositories.PostgresAccountRepository(None), ports.AccountRepository),  # type: ignore[arg-type]
    (repositories.PostgresTransactionRepository(None), ports.TransactionRepository),  # type: ignore[arg-type]
    (repositories.PostgresAlertRepository(None), ports.AlertRepository),  # type: ignore[arg-type]
    (repositories.PostgresInvestigationRepository(None), ports.InvestigationRepository),  # type: ignore[arg-type]
    (repositories.PostgresDqRepository(None), ports.DqRepository),  # type: ignore[arg-type]
]


@pytest.mark.parametrize(
    ("adapter", "port"), _ADAPTERS, ids=lambda value: getattr(value, "__name__", "")
)
def test_each_adapter_satisfies_its_port(adapter, port):
    assert isinstance(adapter, port)


def test_the_five_postgres_repositories_cover_the_five_repository_ports():
    """Guards the negative control: an adapter missing from the list above is unchecked."""
    covered = {port.__name__ for _, port in _ADAPTERS}
    assert covered == {
        "AccountRepository",
        "TransactionRepository",
        "AlertRepository",
        "InvestigationRepository",
        "DqRepository",
    }


def test_postgres_dq_store_matches_the_dq_report_store_protocol():
    """`DqReportStore` is not `@runtime_checkable`, so compare the four signatures by hand.

    The static check in `dq_store._conformance` is the real one. This exists so a contributor who
    runs only `pytest` still gets told when a method is dropped or its keywords change.
    """
    for name in ("save_run", "write_report", "save_profile", "recent_profiles"):
        expected = inspect.signature(getattr(DqReportStore, name))
        actual = inspect.signature(getattr(PostgresDqStore, name))
        assert actual.parameters.keys() == expected.parameters.keys(), name


# --- row mapping -----------------------------------------------------------------------------


def test_to_account_maps_every_column_and_parses_its_enums():
    row = {
        "account_id": "ACC1",
        "customer_id": "CUS1",
        "account_type": "SAVINGS",
        "opened_on": date(2024, 1, 5),
        "status": "DORMANT",
        "branch_code": "BR1",
        "ifsc": "IFSC0001",
        "current_balance": Decimal("1234.56"),
        "last_activity_on": date(2024, 6, 1),
        "is_flagged": False,
        "created_at": datetime(2024, 1, 5, tzinfo=UTC),
    }
    account = repositories._to_account(row)
    assert account.account_type is AccountType.SAVINGS
    assert account.status is AccountStatus.DORMANT
    # Money must stay a Decimal all the way through -- a float here would defeat numeric(18, 2).
    assert account.current_balance == Decimal("1234.56")
    assert isinstance(account.current_balance, Decimal)


def test_to_account_rejects_a_column_the_domain_does_not_know():
    row = {
        "account_id": "ACC1",
        "customer_id": "CUS1",
        "account_type": "OFFSHORE",
        "opened_on": date(2024, 1, 5),
        "status": "ACTIVE",
        "branch_code": "BR1",
        "ifsc": "IFSC0001",
        "current_balance": Decimal("0.00"),
        "last_activity_on": None,
        "is_flagged": False,
        "created_at": datetime(2024, 1, 5, tzinfo=UTC),
    }
    with pytest.raises(ValueError, match="OFFSHORE"):
        repositories._to_account(row)


def test_to_investigation_reads_a_running_run_as_all_nones():
    """A `RUNNING` investigation has no recommendation yet; that is honest, not a partial read."""
    row = {
        "investigation_id": "INV1",
        "alert_id": "ALT1",
        "agent_recommendation": None,
        "agent_confidence": None,
        "computed_risk_score": None,
        "detected_typology": None,
        "hypotheses": [],
        "narrative": None,
        "requires_human": True,
        "llm_model": "deepseek/deepseek-v4-flash",
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": Decimal("0.000000"),
        "latency_ms": 0,
        "tool_call_count": 0,
        "status": "RUNNING",
        "started_at": datetime(2026, 8, 1, tzinfo=UTC),
        "completed_at": None,
    }
    investigation = repositories._to_investigation(row)
    assert investigation.status is InvestigationStatus.RUNNING
    assert investigation.agent_recommendation is None
    assert investigation.hypotheses == ()


def test_to_investigation_parses_a_completed_run():
    row = {
        "investigation_id": "INV1",
        "alert_id": "ALT1",
        "agent_recommendation": "ESCALATE",
        "agent_confidence": 0.82,
        "computed_risk_score": 74.0,
        "detected_typology": "LAYERING",
        "hypotheses": [{"claim": "funds forwarded within minutes"}],
        "narrative": "Account received and forwarded.",
        "requires_human": True,
        "llm_model": "deepseek/deepseek-v4-pro",
        "tokens_in": 900,
        "tokens_out": 300,
        "cost_usd": Decimal("0.000412"),
        "latency_ms": 4100,
        "tool_call_count": 5,
        "status": "AWAITING_DECISION",
        "started_at": datetime(2026, 8, 1, tzinfo=UTC),
        "completed_at": datetime(2026, 8, 1, 0, 0, 5, tzinfo=UTC),
    }
    investigation = repositories._to_investigation(row)
    assert investigation.agent_recommendation is Recommendation.ESCALATE
    assert investigation.detected_typology is Typology.LAYERING
    # Six decimal places survive: the reason cost_usd is not numeric(18, 2).
    assert investigation.cost_usd == Decimal("0.000412")


def test_to_dq_check_result_carries_the_blocking_verdict():
    row = {
        "result_id": 7,
        "run_id": "RUN1",
        "check_name": "duplicate_txn_id",
        "target_table": "transactions",
        "target_column": "txn_id",
        "severity": "BLOCKING",
        "status": "FAIL",
        "expectation": "0 duplicates",
        "observed": "37 duplicate txn_id",
    }
    result = repositories._to_dq_check_result(row)
    assert result.severity is DqSeverity.BLOCKING
    assert result.status is DqCheckStatus.FAIL
    assert result.blocks_batch is True


# --- the loader's column handling ---------------------------------------------------------------


def test_rows_selects_by_name_even_when_the_partition_key_moved():
    """The trap: pyarrow appends `txn_date` last on a partitioned read, wherever it was written.

    The table below has the columns in an order no writer produced. Positional access would put the
    date in the amount's slot; name-based access does not care.
    """
    table = pa.table(
        {
            "txn_date": pa.array([date(2026, 5, 6)], pa.date32()),
            "amount": pa.array([Decimal("7890.06")], pa.decimal128(18, 2)),
            "txn_id": pa.array(["TXN1"], pa.string()),
        }
    )
    rows = list(loader._rows(table, ("txn_id", "amount", "txn_date")))
    assert rows == [("TXN1", Decimal("7890.06"), date(2026, 5, 6))]


def test_rows_streams_every_row_across_chunk_boundaries():
    size = loader.CHUNK_ROWS + 25
    table = pa.table({"account_id": pa.array([f"ACC{i}" for i in range(size)], pa.string())})
    rows = list(loader._rows(table, ("account_id",)))
    assert len(rows) == size
    assert rows[0] == ("ACC0",) and rows[-1] == (f"ACC{size - 1}",)


def test_spec_updatable_excludes_the_conflict_key():
    """The `DO UPDATE SET` list must not try to overwrite the column it matched on."""
    spec = loader.TableSpec(
        table="account_features",
        key=("account_id", "feature_window"),
        columns=(
            ("account_id", "account_id"),
            ("feature_window", "window"),
            ("txn_count", "txn_count"),
        ),
    )
    assert spec.updatable == ("txn_count",)
    assert spec.arrow_columns == ("account_id", "window", "txn_count")
