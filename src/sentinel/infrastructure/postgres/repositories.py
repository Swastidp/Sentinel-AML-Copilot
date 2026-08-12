"""The five domain repositories, implemented over `psycopg` -- T5.2.

One class per port in `domain/ports/repositories.py`. None of them imports its protocol: the ports
are `Protocol`s, so an adapter satisfies one structurally and the dependency arrow from
`infrastructure` to `domain` exists only at type-check time. `tests/unit/test_postgres_adapters.py`
binds each class to its port so that assignability is actually checked, the same trick
`tests/unit/test_port_conformance.py` uses for the hand-written stubs.

Five conventions hold across all of them, and each is load-bearing:

**Every column is selected by name, never by position.** `dict_row` is set on every cursor. This is
the same rule the feature SQL follows for the reason `pipeline/features.py` gives -- pyarrow moves
the partition key to the end of a `transactions` read -- and it costs nothing to hold everywhere.

**Every driver error becomes `PostgresUnavailableError`.** Wrapped by `connection.wrap`, which
catches `psycopg.Error` and nothing else. A store failure is never returned as `None` and never as
an empty tuple: "this account does not exist" and "we could not look" lead a caller to opposite
decisions, and only one of them is a finding.

**Writes commit; reads do not.** Each write method commits its own statement, because the ports
document the workflow as a *sequence* of independent writes -- `save_decision` deliberately does not
transition the investigation or close the alert, the caller does that next. A repository that left
transactions open would make that sequence one long-lived lock against a serving database.

**Writes upsert, except the one that must not.** `INSERT ... ON CONFLICT DO UPDATE` everywhere, so
re-persisting a checkpointed agent run is a no-op rather than a duplicate. `save_decision` is the
exception: a human adjudication is immutable, and a second one raises `ConflictError` off a unique
constraint rather than a read-then-write, so two concurrent analysts cannot both believe they won.

**Enums are re-parsed, not trusted.** A `text` column coming back as `Severity(row["severity"])`
fails loudly on a value the domain does not know, which is what makes the `CHECK` constraints in the
migration and the `StrEnum`s in `domain/enums.py` two halves of one guarantee.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from sentinel.config.logging import get_logger
from sentinel.domain.entities import (
    Account,
    Alert,
    DqCheckResult,
    DqRun,
    HumanDecision,
    Investigation,
    ToolCall,
    Transaction,
)
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    AlertStatus,
    Channel,
    CustomerSegment,
    Direction,
    DqCheckStatus,
    DqRunStatus,
    DqSeverity,
    InvestigationStatus,
    KycLevel,
    Recommendation,
    RiskRating,
    Severity,
    TransactionStatus,
    Typology,
)
from sentinel.domain.exceptions import ConflictError, DomainValidationError, NotFoundError
from sentinel.domain.value_objects import AccountProfile
from sentinel.infrastructure.postgres import connection as pg

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from sentinel.domain import ports

log = get_logger(__name__)

#: Days of silence after which `AccountProfile.is_dormant` reads true, on top of the bank's own
#: `DORMANT` status flag. A display and context value only -- rule R005 scores dormancy from
#: `account_features.days_since_last_activity`, which is measured, not thresholded here.
#: ponytail: a literal, because nothing scores on it. Move it to `Settings` the first time an
#: outcome depends on the number rather than on the measured gap.
DORMANT_AFTER_DAYS: Final = 90

#: Roughly. Used only to render tenure in months for the agent's account-profile tool; a month is
#: not a fixed number of days and no decision turns on the third significant figure.
_DAYS_PER_MONTH: Final = 30.44

#: Severity ordering for the alert queue. Postgres has no ordering on a `text` column that matches
#: the conceptual rank, and `domain/enums.py` deliberately refuses to make `Severity` comparable, so
#: the rank is spelled out here -- in SQL, where the `ORDER BY` needs it.
_SEVERITY_RANK: Final = """
    CASE severity WHEN 'CRITICAL' THEN 4 WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END
"""


# --- row mapping ---------------------------------------------------------------------------------
# Free functions rather than classmethods: they are pure, they are the part most likely to be read
# next to the DDL, and a test can call one with a plain dict.


def _to_account(row: DictRow) -> Account:
    return Account(
        account_id=row["account_id"],
        customer_id=row["customer_id"],
        account_type=AccountType(row["account_type"]),
        opened_on=row["opened_on"],
        status=AccountStatus(row["status"]),
        branch_code=row["branch_code"],
        ifsc=row["ifsc"],
        current_balance=row["current_balance"],
        last_activity_on=row["last_activity_on"],
        is_flagged=row["is_flagged"],
        created_at=row["created_at"],
    )


def _to_transaction(row: DictRow) -> Transaction:
    return Transaction(
        txn_id=row["txn_id"],
        txn_ts=row["txn_ts"],
        from_account=row["from_account"],
        to_account=row["to_account"],
        counterparty_handle=row["counterparty_handle"],
        counterparty_bank=row["counterparty_bank"],
        amount=row["amount"],
        currency=row["currency"],
        channel=Channel(row["channel"]),
        direction=Direction(row["direction"]),
        status=TransactionStatus(row["status"]),
        device_id=row["device_id"],
        ip_address=row["ip_address"],
        narration=row["narration"],
        txn_date=row["txn_date"],
        ingested_at=row["ingested_at"],
        batch_id=row["batch_id"],
    )


def _to_alert(row: DictRow) -> Alert:
    return Alert(
        alert_id=row["alert_id"],
        account_id=row["account_id"],
        rule_code=row["rule_code"],
        rule_name=row["rule_name"],
        triggered_at=row["triggered_at"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        severity=Severity(row["severity"]),
        rule_score=row["rule_score"],
        anomaly_score=row["anomaly_score"],
        trigger_evidence=row["trigger_evidence"],
        status=AlertStatus(row["status"]),
        created_at=row["created_at"],
    )


def _to_investigation(row: DictRow) -> Investigation:
    """Map an `investigations` row. `risk_components` is deliberately not read -- see `save`."""
    recommendation = row["agent_recommendation"]
    typology = row["detected_typology"]
    return Investigation(
        investigation_id=row["investigation_id"],
        alert_id=row["alert_id"],
        agent_recommendation=Recommendation(recommendation) if recommendation else None,
        agent_confidence=row["agent_confidence"],
        computed_risk_score=row["computed_risk_score"],
        detected_typology=Typology(typology) if typology else None,
        hypotheses=tuple(row["hypotheses"]),
        narrative=row["narrative"],
        requires_human=row["requires_human"],
        llm_model=row["llm_model"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        tool_call_count=row["tool_call_count"],
        status=InvestigationStatus(row["status"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _to_tool_call(row: DictRow) -> ToolCall:
    return ToolCall(
        call_id=row["call_id"],
        investigation_id=row["investigation_id"],
        sequence_no=row["sequence_no"],
        tool_name=row["tool_name"],
        tool_input=row["tool_input"],
        tool_output=row["tool_output"],
        is_error=row["is_error"],
        error_message=row["error_message"],
        duration_ms=row["duration_ms"],
        called_at=row["called_at"],
    )


def _to_human_decision(row: DictRow) -> HumanDecision:
    return HumanDecision(
        decision_id=row["decision_id"],
        investigation_id=row["investigation_id"],
        analyst_id=row["analyst_id"],
        final_decision=Recommendation(row["final_decision"]),
        agreed_with_agent=row["agreed_with_agent"],
        rationale=row["rationale"],
        decided_at=row["decided_at"],
    )


def _to_dq_run(row: DictRow) -> DqRun:
    return DqRun(
        run_id=row["run_id"],
        batch_id=row["batch_id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        status=DqRunStatus(row["status"]),
        rows_in=row["rows_in"],
        rows_loaded=row["rows_loaded"],
        rows_rejected=row["rows_rejected"],
        checks_total=row["checks_total"],
        checks_failed=row["checks_failed"],
        report=row["report"],
    )


def _to_dq_check_result(row: DictRow) -> DqCheckResult:
    return DqCheckResult(
        result_id=row["result_id"],
        run_id=row["run_id"],
        check_name=row["check_name"],
        target_table=row["target_table"],
        target_column=row["target_column"],
        severity=DqSeverity(row["severity"]),
        status=DqCheckStatus(row["status"]),
        expectation=row["expectation"],
        observed=row["observed"],
    )


# --- shared plumbing -----------------------------------------------------------------------------


class _PostgresRepository:
    """Connection ownership and the one cursor helper all five adapters share.

    The connection is **injected, not opened here**. A repository that opened its own connection
    could not participate in a caller's transaction and would reconnect on every call; taking one in
    the constructor is what lets the composition root decide -- one per process for the batch, one
    per request when the API lands.
    """

    def __init__(self, connection: psycopg.Connection[Any]) -> None:
        self._connection = connection

    @contextmanager
    def _cursor(self, identifier: str) -> Iterator[psycopg.Cursor[DictRow]]:
        """A `dict_row` cursor whose driver errors surface as `PostgresUnavailableError`.

        `identifier` is whatever was being looked up, so a failure says which read broke rather than
        only that one did. The `pg.wrap` sits outside the cursor so an error raised while *fetching*
        -- which is where a dropped connection usually shows up -- is translated too.
        """
        with pg.wrap(identifier), self._connection.cursor(row_factory=dict_row) as cursor:
            yield cursor


# --- accounts ------------------------------------------------------------------------------------


class PostgresAccountRepository(_PostgresRepository):
    """`AccountRepository` over `accounts` and `customers`."""

    def __init__(
        self,
        connection: psycopg.Connection[Any],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """`clock` exists because `AccountProfile` refuses to read one.

        `tenure_months` and `is_dormant` are both relative to now, and the port says the adapter
        resolves them against its own clock at read time -- two calls a month apart legitimately
        differ. Injecting the clock is what lets a test assert an exact tenure instead of a range.
        """
        super().__init__(connection)
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def get(self, account_id: str) -> Account | None:
        with self._cursor(f"account {account_id}") as cursor:
            cursor.execute("SELECT * FROM accounts WHERE account_id = %s", (account_id,))
            row = cursor.fetchone()
        return _to_account(row) if row is not None else None

    def get_profile(self, account_id: str) -> AccountProfile | None:
        """The joined account-plus-customer view the agent's first tool returns.

        One query rather than two `get`s: the whole reason this method is separate from `get` is
        that a caller who needs the customer context should pay for one round trip, not two.
        """
        with self._cursor(f"account profile {account_id}") as cursor:
            cursor.execute(
                """
                SELECT a.account_id, a.customer_id, a.account_type, a.status, a.opened_on,
                       a.current_balance, a.last_activity_on,
                       c.kyc_level, c.declared_annual_income, c.risk_rating, c.segment, c.is_pep
                  FROM accounts a
                  JOIN customers c ON c.customer_id = a.customer_id
                 WHERE a.account_id = %s
                """,
                (account_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        today = self._clock().date()
        tenure_days = max((today - row["opened_on"]).days, 0)
        last_activity = row["last_activity_on"]
        status = AccountStatus(row["status"])
        # Never transacted counts as dormant: an account open for a year with no activity at all is
        # the strongest form of the thing this flag describes, not an unknown.
        silent_days = (today - last_activity).days if last_activity is not None else tenure_days
        return AccountProfile(
            account_id=row["account_id"],
            customer_id=row["customer_id"],
            account_type=AccountType(row["account_type"]),
            status=status,
            tenure_months=int(tenure_days / _DAYS_PER_MONTH),
            is_dormant=status is AccountStatus.DORMANT or silent_days >= DORMANT_AFTER_DAYS,
            current_balance=row["current_balance"],
            kyc_level=KycLevel(row["kyc_level"]),
            declared_annual_income=row["declared_annual_income"],
            risk_rating=RiskRating(row["risk_rating"]),
            segment=CustomerSegment(row["segment"]),
            is_pep=row["is_pep"],
        )


# --- transactions --------------------------------------------------------------------------------


class PostgresTransactionRepository(_PostgresRepository):
    """`TransactionRepository` over the partitioned `transactions` table."""

    def get(self, txn_id: str) -> Transaction | None:
        """One payment by id.

        The primary key is `(txn_id, txn_date)` because Postgres requires the partition key in every
        unique constraint, so this query cannot prune partitions and scans all of them. That is
        acceptable for a rare single-row lookup and would not be for a hot path; the hot paths all
        carry a timestamp range and use `idx_txn_from_ts` / `idx_txn_to_ts`.
        """
        with self._cursor(f"transaction {txn_id}") as cursor:
            cursor.execute("SELECT * FROM transactions WHERE txn_id = %s LIMIT 1", (txn_id,))
            row = cursor.fetchone()
        return _to_transaction(row) if row is not None else None

    def list_for_account(
        self,
        account_id: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[Transaction, ...]:
        """Both legs of the account's activity in a half-open window, newest first.

        `txn_date` is constrained alongside `txn_ts` so the planner can prune partitions. The two
        bounds agree by construction -- `txn_date` is the date part of `txn_ts` -- and the date
        comparison is inclusive on both ends because a partition holds whole days.
        """
        for name, value in (("start", start), ("end", end)):
            if value.utcoffset() is None:
                raise DomainValidationError(
                    f"{name} must be timezone-aware, got the naive datetime {value.isoformat()}. "
                    f"The same wall-clock string is a different instant in a different offset, "
                    f"so a night-hour window would silently shift."
                )
        with self._cursor(f"transactions for {account_id}") as cursor:
            cursor.execute(
                """
                SELECT * FROM transactions
                 WHERE (from_account = %(account)s OR to_account = %(account)s)
                   AND txn_ts >= %(start)s AND txn_ts < %(end)s
                   AND txn_date BETWEEN %(start)s::date AND %(end)s::date
                 ORDER BY txn_ts DESC, txn_id DESC
                 LIMIT %(limit)s
                """,
                {"account": account_id, "start": start, "end": end, "limit": limit},
            )
            rows = cursor.fetchall()
        return tuple(_to_transaction(row) for row in rows)


# --- alerts --------------------------------------------------------------------------------------


class PostgresAlertRepository(_PostgresRepository):
    """`AlertRepository` over `alerts`, plus the batch write the detection pipeline needs.

    **`save_many` is deliberately not on the `AlertRepository` port.** The port serves the agent,
    the API and the console, and none of them may create an alert -- an alert is raised by detection
    or by a vendor TMS feed, never by the thing triaging it. Same reasoning as
    `PostgresDqRepository`, which is read-only because the gate owns its own writes: the capability
    exists on the adapter the pipeline holds, and is absent from the interface everything else gets.
    """

    def save_many(self, alerts: Sequence[Alert]) -> int:
        """Upsert a batch of alerts, keyed on `alert_id`. Returns the number of rows written.

        `status` is **not** in the `SET` list, and that is the important line here. `alert_id` is
        derived deterministically from `(account_id, rule_code, window_end)`, so a re-run of the
        same batch collides with every row it wrote last time -- which is exactly what makes the
        load idempotent. If the update reset `status`, a re-run would drag an alert an analyst had
        already moved to `IN_REVIEW`, or the human gate had parked in `AWAITING_DECISION`, back to
        `OPEN`. Re-running detection would silently undo human work. `created_at` is held for the
        same reason: the row's first appearance is a fact about the past.

        An empty batch is a successful no-op. A run that raised nothing is a real outcome, and
        making the caller branch on it would be noise.
        """
        if not alerts:
            return 0
        with self._cursor(f"alert batch of {len(alerts)}") as cursor:
            cursor.executemany(
                """
                INSERT INTO alerts (
                    alert_id, account_id, rule_code, rule_name, triggered_at, window_start,
                    window_end, severity, rule_score, anomaly_score, trigger_evidence, status,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_id) DO UPDATE SET
                    account_id       = EXCLUDED.account_id,
                    rule_code        = EXCLUDED.rule_code,
                    rule_name        = EXCLUDED.rule_name,
                    triggered_at     = EXCLUDED.triggered_at,
                    window_start     = EXCLUDED.window_start,
                    window_end       = EXCLUDED.window_end,
                    severity         = EXCLUDED.severity,
                    rule_score       = EXCLUDED.rule_score,
                    anomaly_score    = EXCLUDED.anomaly_score,
                    trigger_evidence = EXCLUDED.trigger_evidence
                """,
                [
                    (
                        alert.alert_id,
                        alert.account_id,
                        alert.rule_code,
                        alert.rule_name,
                        alert.triggered_at,
                        alert.window_start,
                        alert.window_end,
                        str(alert.severity),
                        alert.rule_score,
                        alert.anomaly_score,
                        Jsonb(dict(alert.trigger_evidence)),
                        str(alert.status),
                        alert.created_at,
                    )
                    for alert in alerts
                ],
            )
        self._connection.commit()
        return len(alerts)

    def count(self) -> int:
        """How many alerts the table holds. Used by the detection run to prove the write landed."""
        with self._cursor("alert count") as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM alerts")
            row = cursor.fetchone()
        return int(row["n"]) if row is not None else 0

    def rule_codes_by_account(self) -> dict[str, tuple[str, ...]]:
        """Every alert ever raised, as `account_id -> rule codes`. Read-only, and off the port.

        **Deliberately ignores `status`**, unlike `list_open`. This serves `reporting/metrics.py`,
        which asks "what did detection raise?" -- a question about the detector, not about the work
        queue. Filtering out the alerts an analyst has since closed would let precision improve
        every time somebody dismissed a false positive, which is the one way an accuracy figure can
        be made to look better by doing nothing.

        Two projected columns rather than whole `Alert` objects because that is all the measurement
        reads, and hydrating 839 JSONB payloads to count them would be waste.
        """
        with self._cursor("alert rule codes by account") as cursor:
            cursor.execute(
                "SELECT account_id, rule_code FROM alerts ORDER BY account_id, rule_code"
            )
            rows = cursor.fetchall()
        by_account: dict[str, list[str]] = {}
        for row in rows:
            by_account.setdefault(str(row["account_id"]), []).append(str(row["rule_code"]))
        return {account: tuple(codes) for account, codes in by_account.items()}

    def get(self, alert_id: str) -> Alert | None:
        with self._cursor(f"alert {alert_id}") as cursor:
            cursor.execute("SELECT * FROM alerts WHERE alert_id = %s", (alert_id,))
            row = cursor.fetchone()
        return _to_alert(row) if row is not None else None

    def list_open(
        self,
        *,
        limit: int,
        offset: int = 0,
        severity: Severity | None = None,
        status: AlertStatus | None = None,
    ) -> tuple[Alert, ...]:
        """The work queue: anything not `CLOSED`, highest severity first then newest.

        "Open" is *not* `AlertStatus.OPEN` -- an alert parked in `AWAITING_DECISION` by the human
        gate is the one an analyst most needs to see. The final tie-break on `alert_id` is what
        keeps `limit`/`offset` paging from skipping or repeating a row between requests.
        """
        clauses = ["status <> 'CLOSED'"] if status is None else ["status = %(status)s"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = str(status)
        if severity is not None:
            clauses.append("severity = %(severity)s")
            params["severity"] = str(severity)
        with self._cursor("open alerts") as cursor:
            cursor.execute(
                f"""
                SELECT * FROM alerts
                 WHERE {" AND ".join(clauses)}
                 ORDER BY {_SEVERITY_RANK} DESC, triggered_at DESC, alert_id DESC
                 LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            )
            rows = cursor.fetchall()
        return tuple(_to_alert(row) for row in rows)

    def update_status(self, alert_id: str, status: AlertStatus) -> None:
        """Move one alert. Idempotent -- setting the status it already has is a successful no-op."""
        with self._cursor(f"alert {alert_id}") as cursor:
            cursor.execute(
                "UPDATE alerts SET status = %s WHERE alert_id = %s", (str(status), alert_id)
            )
            affected = cursor.rowcount
        if affected == 0:
            # Unlike a read, a write against a missing row is a programming error: succeeding
            # silently would lose the state change and leave the alert wherever it was.
            self._connection.rollback()
            raise NotFoundError("alert", alert_id)
        self._connection.commit()


# --- investigations ------------------------------------------------------------------------------


class PostgresInvestigationRepository(_PostgresRepository):
    """`InvestigationRepository` over `investigations`, `tool_calls` and `human_decisions`."""

    def save(self, investigation: Investigation) -> None:
        """Insert or update, keyed on `investigation_id`.

        The graph writes the same investigation twice by design -- once as `AWAITING_DECISION`
        before the interrupt, again as `COMPLETE` after the resume -- so this upserts.

        **`risk_components` is not in the `SET` list, on purpose.** The column exists (decision 3 in
        `docs/02-DATA-MODEL.md`) but `Investigation` carries no such field yet, so this adapter has
        nothing to write there. Omitting it from the update means a breakdown written by the agent's
        persist node at Stage 11 survives a later re-save instead of being blanked by an adapter
        that does not know about it.
        """
        with self._cursor(f"investigation {investigation.investigation_id}") as cursor:
            cursor.execute(
                """
                INSERT INTO investigations (
                    investigation_id, alert_id, agent_recommendation, agent_confidence,
                    computed_risk_score, detected_typology, hypotheses, narrative, requires_human,
                    llm_model, tokens_in, tokens_out, cost_usd, latency_ms, tool_call_count,
                    status, started_at, completed_at
                ) VALUES (
                    %(investigation_id)s, %(alert_id)s, %(agent_recommendation)s,
                    %(agent_confidence)s, %(computed_risk_score)s, %(detected_typology)s,
                    %(hypotheses)s, %(narrative)s, %(requires_human)s, %(llm_model)s,
                    %(tokens_in)s, %(tokens_out)s, %(cost_usd)s, %(latency_ms)s,
                    %(tool_call_count)s, %(status)s, %(started_at)s, %(completed_at)s
                )
                ON CONFLICT (investigation_id) DO UPDATE SET
                    alert_id             = EXCLUDED.alert_id,
                    agent_recommendation = EXCLUDED.agent_recommendation,
                    agent_confidence     = EXCLUDED.agent_confidence,
                    computed_risk_score  = EXCLUDED.computed_risk_score,
                    detected_typology    = EXCLUDED.detected_typology,
                    hypotheses           = EXCLUDED.hypotheses,
                    narrative            = EXCLUDED.narrative,
                    requires_human       = EXCLUDED.requires_human,
                    llm_model            = EXCLUDED.llm_model,
                    tokens_in            = EXCLUDED.tokens_in,
                    tokens_out           = EXCLUDED.tokens_out,
                    cost_usd             = EXCLUDED.cost_usd,
                    latency_ms           = EXCLUDED.latency_ms,
                    tool_call_count      = EXCLUDED.tool_call_count,
                    status               = EXCLUDED.status,
                    started_at           = EXCLUDED.started_at,
                    completed_at         = EXCLUDED.completed_at
                """,
                {
                    "investigation_id": investigation.investigation_id,
                    "alert_id": investigation.alert_id,
                    "agent_recommendation": _optional_str(investigation.agent_recommendation),
                    "agent_confidence": investigation.agent_confidence,
                    "computed_risk_score": investigation.computed_risk_score,
                    "detected_typology": _optional_str(investigation.detected_typology),
                    "hypotheses": Jsonb([dict(item) for item in investigation.hypotheses]),
                    "narrative": investigation.narrative,
                    "requires_human": investigation.requires_human,
                    "llm_model": investigation.llm_model,
                    "tokens_in": investigation.tokens_in,
                    "tokens_out": investigation.tokens_out,
                    "cost_usd": investigation.cost_usd,
                    "latency_ms": investigation.latency_ms,
                    "tool_call_count": investigation.tool_call_count,
                    "status": str(investigation.status),
                    "started_at": investigation.started_at,
                    "completed_at": investigation.completed_at,
                },
            )
        self._connection.commit()

    def get(self, investigation_id: str) -> Investigation | None:
        with self._cursor(f"investigation {investigation_id}") as cursor:
            cursor.execute(
                "SELECT * FROM investigations WHERE investigation_id = %s", (investigation_id,)
            )
            row = cursor.fetchone()
        return _to_investigation(row) if row is not None else None

    def save_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        """Append a batch of tool calls, idempotent on `(investigation_id, sequence_no)`.

        An empty batch is a successful no-op: an investigation that made no tool calls is a real
        outcome, and forcing the caller to branch on it would be noise.
        """
        if not calls:
            return
        investigation_ids = {call.investigation_id for call in calls}
        if len(investigation_ids) > 1:
            # The trace is what makes an investigation replayable, and interleaving two runs
            # destroys that. Caught here rather than producing a corrupt audit record.
            raise DomainValidationError(
                f"a tool-call batch must belong to one investigation, got "
                f"{sorted(investigation_ids)!r}."
            )
        with self._cursor(f"tool calls for {investigation_ids.pop()}") as cursor:
            cursor.executemany(
                """
                INSERT INTO tool_calls (
                    investigation_id, sequence_no, tool_name, tool_input, tool_output,
                    is_error, error_message, duration_ms, called_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (investigation_id, sequence_no) DO UPDATE SET
                    tool_name     = EXCLUDED.tool_name,
                    tool_input    = EXCLUDED.tool_input,
                    tool_output   = EXCLUDED.tool_output,
                    is_error      = EXCLUDED.is_error,
                    error_message = EXCLUDED.error_message,
                    duration_ms   = EXCLUDED.duration_ms,
                    called_at     = EXCLUDED.called_at
                """,
                [
                    (
                        call.investigation_id,
                        call.sequence_no,
                        call.tool_name,
                        Jsonb(dict(call.tool_input)),
                        Jsonb(dict(call.tool_output)) if call.tool_output is not None else None,
                        call.is_error,
                        call.error_message,
                        call.duration_ms,
                        call.called_at,
                    )
                    for call in calls
                ],
            )
        self._connection.commit()

    def list_tool_calls(self, investigation_id: str) -> tuple[ToolCall, ...]:
        """The full trace in `sequence_no` order. Ascending order is the contract, not a nicety."""
        with self._cursor(f"tool calls for {investigation_id}") as cursor:
            cursor.execute(
                "SELECT * FROM tool_calls WHERE investigation_id = %s ORDER BY sequence_no ASC",
                (investigation_id,),
            )
            rows = cursor.fetchall()
        return tuple(_to_tool_call(row) for row in rows)

    def get_decision(self, investigation_id: str) -> HumanDecision | None:
        with self._cursor(f"decision for {investigation_id}") as cursor:
            cursor.execute(
                "SELECT * FROM human_decisions WHERE investigation_id = %s", (investigation_id,)
            )
            row = cursor.fetchone()
        return _to_human_decision(row) if row is not None else None

    def save_decision(self, decision: HumanDecision) -> None:
        """Record an adjudication. **Never overwrites** -- a plain `INSERT`, deliberately.

        The unique constraint on `human_decisions.investigation_id` is what settles a race between
        two analysts; a read-then-write would let both believe they won. A foreign-key violation
        means the investigation itself is absent, which is a `NotFoundError` rather than a conflict:
        a decision with nothing to adjudicate is an orphan no audit trail can explain.
        """
        try:
            with self._cursor(f"decision {decision.decision_id}") as cursor:
                cursor.execute(
                    """
                    INSERT INTO human_decisions (
                        decision_id, investigation_id, analyst_id, final_decision,
                        agreed_with_agent, rationale, decided_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        decision.decision_id,
                        decision.investigation_id,
                        decision.analyst_id,
                        str(decision.final_decision),
                        decision.agreed_with_agent,
                        decision.rationale,
                        decision.decided_at,
                    ),
                )
        except pg.PostgresUnavailableError as exc:
            # `_cursor` has already turned the driver error into a domain error, so the original is
            # on `__cause__`. Re-reading it here keeps one translation point rather than two.
            self._connection.rollback()
            raise _classify_decision_failure(exc, decision) from exc
        self._connection.commit()


def _classify_decision_failure(
    exc: pg.PostgresUnavailableError, decision: HumanDecision
) -> Exception:
    """Turn a constraint violation on `human_decisions` into the error the API needs.

    Three outcomes and they map to three different HTTP statuses, which is why this is not one
    `except`: a duplicate is `409`, a missing investigation is `404`, and anything else really is
    the store being unavailable and stays a `503`.
    """
    cause = exc.__cause__
    if isinstance(cause, psycopg.errors.UniqueViolation):
        return ConflictError(
            "human decision",
            decision.investigation_id,
            "this investigation has already been adjudicated; decisions are immutable.",
        )
    if isinstance(cause, psycopg.errors.ForeignKeyViolation):
        return NotFoundError("investigation", decision.investigation_id)
    return exc


# --- data quality --------------------------------------------------------------------------------


class PostgresDqRepository(_PostgresRepository):
    """`DqRepository` over `dq_runs` and `dq_check_results`. **Read-only, and that is the point.**

    The gate in `pipeline/quality.py` writes its own rows through `PostgresDqStore`, including when
    it fails a batch. Nothing in `api/` or `ui/` may write here, so the port that serves them does
    not offer the ability and neither does this class.
    """

    def list_runs(self, *, limit: int, offset: int = 0) -> tuple[DqRun, ...]:
        """Recent runs, newest first, **including failed ones** -- the interesting rows."""
        with self._cursor("dq runs") as cursor:
            cursor.execute(
                """
                SELECT * FROM dq_runs
                 ORDER BY started_at DESC, run_id DESC
                 LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
        return tuple(_to_dq_run(row) for row in rows)

    def list_check_results(self, run_id: str) -> tuple[DqCheckResult, ...]:
        """Every check in one run, blocking failures first, then by name.

        The ordering expression is `DqCheckResult.blocks_batch` written in SQL: a result blocks the
        batch exactly when it is `BLOCKING` and it `FAIL`ed. Sorting on that rather than on severity
        alone means a passing blocking check does not outrank a real warning failure.
        """
        with self._cursor(f"dq check results for {run_id}") as cursor:
            cursor.execute(
                """
                SELECT * FROM dq_check_results
                 WHERE run_id = %s
                 ORDER BY (severity = 'BLOCKING' AND status = 'FAIL') DESC, check_name ASC
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
        return tuple(_to_dq_check_result(row) for row in rows)


def _optional_str(value: object | None) -> str | None:
    """`str(member)` for a `StrEnum`, `None` for `None`. Saves an inline conditional per column."""
    return None if value is None else str(value)


if TYPE_CHECKING:  # pragma: no cover - a static assertion, never imported or executed

    def _conformance(
        connection: psycopg.Connection[Any],
    ) -> tuple[
        ports.AccountRepository,
        ports.TransactionRepository,
        ports.AlertRepository,
        ports.InvestigationRepository,
        ports.DqRepository,
    ]:
        """**This function is the T5.2 conformance check.** It never runs.

        `mypy --strict` verifies that each adapter is assignable to the port in the corresponding
        slot of the return type, which is the only thing that checks *signatures* --
        `@runtime_checkable` compares method names and nothing else, as
        `tests/unit/test_ports.py::test_isinstance_ignores_signatures` demonstrates. The runtime
        half lives in `tests/unit/test_postgres_adapters.py`; both are needed, for the reason
        `tests/unit/test_port_conformance.py` spells out.

        Do not "tidy" the return annotation into `tuple[Any, ...]`. An unannotated or widened
        binding gives mypy nothing to compare against and silently disables the check.
        """
        return (
            PostgresAccountRepository(connection),
            PostgresTransactionRepository(connection),
            PostgresAlertRepository(connection),
            PostgresInvestigationRepository(connection),
            PostgresDqRepository(connection),
        )
