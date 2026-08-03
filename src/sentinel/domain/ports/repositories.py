"""The five seams onto the system of record — accounts, transactions, alerts, investigations, DQ.

`docs/08-DIAGRAMS.md` shows narrow repositories rather than one fat `Repository`, and that is the
point of the file: `ScoringService` never sees an alert, the agent's graph tool never sees a
transaction, and the console's decision handler depends on `InvestigationRepository` alone. Adding a
method to one of these cannot widen what the others expose.

Everything here is a `Protocol`, not an ABC, and that choice does real work. Structural typing means
`PostgresAccountRepository` satisfies `AccountRepository` without importing it, so
`infrastructure/` depends on `domain/` only at type-check time and the dependency arrow in
`docs/01-ARCHITECTURE.md` §2 holds at runtime too. It also means a test fake is a plain class with
the right methods — no base class, no registration, no `super().__init__`.

**A `...` body is the whole implementation of a Protocol method.** It is not a placeholder and there
is nothing to fill in later; the docstring above it is the contract, and an adapter that disagrees
with it is the defect.

Three conventions hold across all five, because callers rely on them:

* **Absence is `None` on a read, `NotFoundError` on a write.** `get` returning `Optional` lets the
  caller decide whether a missing row is a 404 (the API), a skip (a batch loop), or a bug. A write
  has no such choice: silently succeeding against a row that is not there loses a state change, so
  `update_status` raises.
* **Reads that can return many rows return a `tuple`, ordered, and say so.** An unordered result
  makes `limit`/`offset` pagination silently skip and repeat rows, which is the classic paging bug
  and is invisible in testing against small data.
* **A store failure is never a `None` and never an empty tuple.** Every method here may raise
  `StoreUnavailableError`. Collapsing it into an absent result is the defect this convention exists
  to prevent, because "this account does not exist" and "we could not look" lead a caller to
  opposite decisions — and only one of them is a finding.

This module lives in `domain` and imports nothing outside the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

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
from sentinel.domain.enums import AlertStatus, Severity
from sentinel.domain.value_objects import AccountProfile


@runtime_checkable
class AccountRepository(Protocol):
    """Account and customer context, read-only.

    The split between the two methods is a cost decision. `get` is one row from `accounts`;
    `get_profile` joins `customers` and derives tenure, which is more work than a caller that only
    needs a balance should pay for.
    """

    def get(self, account_id: str) -> Account | None:
        """Return the account, or `None` if no such `account_id` exists.

        Raises nothing for a missing account — see the module docstring. Implementations may raise a
        `SentinelError` subclass for a store failure, which is a different condition entirely and
        must not be collapsed into `None`: "this account does not exist" and "Postgres is
        unreachable" lead to different decisions.
        """
        ...

    def get_profile(self, account_id: str) -> AccountProfile | None:
        """Return the joined account-plus-customer profile, or `None` if the account is unknown.

        This is the `get_account_profile` tool's entire answer (`docs/06-AGENT-DESIGN.md` T1).
        `AccountProfile.tenure_months` and `is_dormant` are both relative to now, so the
        implementation resolves them against its own clock at read time; two calls a month apart
        legitimately return different values for the same unchanged row.

        Raises `StoreUnavailableError` on a store failure.
        """
        ...


@runtime_checkable
class TransactionRepository(Protocol):
    """Individual payments — the one high-volume table, and the one the graph deliberately omits.

    `docs/03-GRAPH-SCHEMA.md` §1 draws the boundary this port sits on: a fixed one-hop lookup
    ("who did this account pay yesterday?") is a SQL query and should stay one. Variable-depth
    structure goes to `GraphRepository`; per-payment detail comes from here.
    """

    def get(self, txn_id: str) -> Transaction | None:
        """Return the transaction, or `None` if no such `txn_id` exists.

        Raises nothing for a missing transaction. Raises `StoreUnavailableError` on a store
        failure — an adapter must not wrap a driver error and return `None`, because "this payment
        was never made" and "we could not read the ledger" are not the same answer to an analyst.
        """
        ...

    def list_for_account(
        self,
        account_id: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[Transaction, ...]:
        """Return transactions where `account_id` is either side, newest `txn_ts` first.

        The window is half-open — `start <= txn_ts < end` — so consecutive windows tile without
        double-counting the boundary transaction. Both bounds must be timezone-aware; a naive one is
        a `DomainValidationError`, because the same wall-clock string means different instants in
        different offsets and a night-hour window would silently shift.

        Ordering is newest first and total: `txn_ts DESC, txn_id DESC`. The tie-break on `txn_id`
        matters because bulk-generated transactions share timestamps, and without it `limit` would
        return a different subset on each call.

        Returns an empty tuple for an account with no activity in the window — which is a fact, not
        a missing answer, and is distinct from `get` returning `None` for an account that does not
        exist. Raises `StoreUnavailableError` on a store failure.
        """
        ...


@runtime_checkable
class AlertRepository(Protocol):
    """The alert queue: what the rule engine raised and where each one has got to."""

    def get(self, alert_id: str) -> Alert | None:
        """Return the alert with its `trigger_evidence`, or `None` if no such `alert_id` exists.

        `trigger_evidence` is always populated, never dropped to save bytes: it is the agent's
        starting point, and an alert without it forces a rediscovery of the numbers that raised it.

        Raises nothing for a missing alert. Raises `StoreUnavailableError` on a store failure,
        which must not be collapsed into `None` — an alert the queue cannot read is still open.
        """
        ...

    def list_open(
        self,
        *,
        limit: int,
        offset: int = 0,
        severity: Severity | None = None,
        status: AlertStatus | None = None,
    ) -> tuple[Alert, ...]:
        """Return the alert queue, highest severity first then newest `triggered_at` first.

        "Open" means **any status other than `CLOSED`**, not `AlertStatus.OPEN` specifically. An
        alert sitting in `AWAITING_DECISION` is the one an analyst most needs to see — it is what
        the human gate parked — so restricting this to `OPEN` would hide exactly the work queue the
        console exists to show. Pass `status` to narrow to one stage.

        Ordering is total and deterministic, including a final tie-break on `alert_id`, so
        `limit`/`offset` paging cannot skip or repeat a row between requests.

        `severity` and `status` are AND-ed when both are given. Returns an empty tuple when nothing
        matches. Raises `StoreUnavailableError` on a store failure.
        """
        ...

    def update_status(self, alert_id: str, status: AlertStatus) -> None:
        """Move one alert to `status`.

        Idempotent: setting the status an alert already has is a successful no-op, which is what
        makes a retried API call or a re-run pipeline step safe.

        Raises `NotFoundError` if `alert_id` does not exist — unlike a read, a write against a
        missing row is a programming error, and silently succeeding would lose the state change.
        Raises `StoreUnavailableError` on a store failure.
        """
        ...


@runtime_checkable
class InvestigationRepository(Protocol):
    """Agent runs, their tool traces, and the human decisions that close them.

    The three concerns sit in one port because they are written together by one caller — the
    `persist` node writes the investigation and its full trace in the same unit of work, and the
    decision handler writes the decision and reads the trace back for the response.

    Investigations and decisions are separate tables for the model-risk reason given in
    `entities.py`, and this port keeps them separate: nothing here lets an agent write a decision.
    """

    def save(self, investigation: Investigation) -> None:
        """Insert or update the investigation, keyed on `investigation_id`.

        Idempotent by upsert, because the graph writes the same investigation twice by design: once
        as `AWAITING_DECISION` before the interrupt and again as `COMPLETE` after the resume. A
        second call must overwrite rather than duplicate or fail.

        Raises `StoreUnavailableError` on a store failure.
        """
        ...

    def get(self, investigation_id: str) -> Investigation | None:
        """Return the investigation without its tool trace, or `None` if unknown.

        The trace is a separate call because it is the large part of the payload and the KPI queries
        never want it. Use `list_tool_calls` when it is needed.

        A `RUNNING` or `AWAITING_DECISION` investigation comes back with `agent_recommendation`,
        `narrative` and the rest as `None`. That is the honest representation of "the agent has not
        got there", not a partial read, and a caller must not treat it as one.

        Raises nothing for a missing investigation. Raises `StoreUnavailableError` on a store
        failure — collapsing that into `None` would make a resumable checkpointed run look absent.
        """
        ...

    def save_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        """Append a batch of tool calls to their investigation's trace.

        All calls must share one `investigation_id`; a mixed batch is a `DomainValidationError`,
        because the trace is what makes an investigation replayable and interleaving two runs
        destroys that. `ToolCall.call_id` is `None` on the way in — the column is a `bigserial` and
        the database assigns it — so a caller must re-read to learn the assigned ids.

        Idempotent on `(investigation_id, sequence_no)`: re-persisting a checkpointed run must not
        double the trace. Raises `StoreUnavailableError` on a store failure.
        """
        ...

    def list_tool_calls(self, investigation_id: str) -> tuple[ToolCall, ...]:
        """Return the full tool trace in `sequence_no` order, ascending.

        The order is the contract, not a convenience: the trace is read as a narrative of what the
        agent did, and out-of-order steps would misrepresent which evidence informed which
        conclusion.

        Returns an empty tuple for an investigation that made no tool calls, and **also** for an
        `investigation_id` that does not exist — this method cannot distinguish them, so a caller
        that needs to must call `get` first. That is a deliberate trade: the API already loads the
        investigation before its trace, so a second existence check would be a wasted query.

        Raises `StoreUnavailableError` on a store failure. An empty tuple must never stand in for
        a failed read, or a partial trace would be filed as a complete audit record.
        """
        ...

    def get_decision(self, investigation_id: str) -> HumanDecision | None:
        """Return the human decision for this investigation, or `None` if none has been recorded.

        `None` is the normal state for a `RUNNING` or `AWAITING_DECISION` investigation, and is also
        what an unknown `investigation_id` returns. At most one decision exists per investigation,
        enforced in the store, so this never has to choose between two.

        Reading `None` here is **not** a licence to skip `save_decision`'s conflict check. Between
        the read and the write another analyst can commit one; the unique constraint is what settles
        that, and this method is for rendering, not for guarding.

        Raises `StoreUnavailableError` on a store failure. A failed read returned as `None` would
        let the console draw an unadjudicated case as awaiting a decision that already exists.
        """
        ...

    def save_decision(self, decision: HumanDecision) -> None:
        """Record an analyst's adjudication. **Never overwrites.**

        Decisions are immutable and there is at most one per investigation. A second call for an
        `investigation_id` that already has one raises `ConflictError` rather than updating, and
        `docs/07-API-DESIGN.md` §5 maps that to `409 Conflict`. Implementations enforce it with a
        unique constraint rather than a read-then-write, so two concurrent analysts cannot both win.

        Raises `NotFoundError` if the `investigation_id` does not exist — a decision with nothing to
        adjudicate is an orphan row that no audit trail can explain. Raises
        `StoreUnavailableError` on a store failure.

        This method does **not** transition the investigation or close the alert. Those are separate
        writes through `save` and `AlertRepository.update_status`, sequenced by the caller, so that
        recording what a human decided is never coupled to the workflow's own bookkeeping.
        """
        ...


@runtime_checkable
class DqRepository(Protocol):
    """Data-quality run history — read-only, and backing `GET /v1/dq/runs`.

    Read-only for a reason worth stating: the gate in `pipeline/quality.py` writes its own rows as
    part of the batch it is judging, and it writes them **even when it fails the batch** — that is
    the whole point of the table. Nothing in `api/` or `ui/` may write here, so the port that serves
    them does not offer the ability.

    Two methods, split the same way as the investigation trace. A run is a summary an operator
    scans; its individual check results are the detail behind one row, and the history page never
    wants every check of every run at once.
    """

    def list_runs(self, *, limit: int, offset: int = 0) -> tuple[DqRun, ...]:
        """Return recent data-quality runs, newest `started_at` first.

        Includes failed runs. A `FAILED` run is the most interesting row on the page — it is the one
        that blocked a batch — so filtering to successes would hide exactly what the history exists
        to show.

        Ordering is total, with a tie-break on `run_id`, so `limit`/`offset` paging is stable.
        Returns an empty tuple before the first run has ever executed.

        Raises `StoreUnavailableError` on a store failure. An empty tuple must not stand in for
        one, or a broken dashboard would read as a clean pipeline that has never run.
        """
        ...

    def list_check_results(self, run_id: str) -> tuple[DqCheckResult, ...]:
        """Return every check in one run, blocking failures first, then by `check_name`.

        Failures lead because the reader's question is always "what broke?". `DqCheckResult`
        already answers whether a single result was enough to stop the load, via `blocks_batch`, so
        this method sorts by that rather than inventing a second severity notion.

        Returns an empty tuple both for a run with no checks and for an unknown `run_id` — same
        deliberate trade as `list_tool_calls`, since the page loads the run before its checks.

        Raises `StoreUnavailableError` on a store failure.
        """
        ...
