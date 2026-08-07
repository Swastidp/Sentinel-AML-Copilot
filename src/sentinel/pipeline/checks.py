"""The twelve data-quality checks, as pure functions over one batch.

`docs/05-DATA-FLOW.md` §4 is the specification and this file is the implementation of it, in the
same order and with the same severities. Nothing here opens a file, reads a clock, or writes a row:
a check receives a `CheckContext` that already holds everything it needs and returns a
`CheckOutcome` describing what it found. `pipeline/quality.py` does the I/O and decides the batch's
fate. That split is what makes the checks testable against a five-row DataFrame with no lakehouse
on disk, and it is why a corruption test can prove a bad batch is refused in milliseconds.

Five implementation decisions are load-bearing, and each would be a real defect if reversed:

**Money is compared in `Decimal`, never `float`.** The `amount` column comes back from Parquet as
`decimal128(18, 2)`, which pandas materialises as an object column of `decimal.Decimal`. Converting
to `float` to compare against the Rs 10 crore cap would put binary floating point back into the one
check whose entire job is to police amounts, and a boundary value would land on the wrong side of it
roughly as often as not.

**Columns are selected by name, never by position.** `txn_date` is the hive partition key, so
pyarrow appends it *last* when reading the dataset back rather than leaving it where
`TRANSACTION_SCHEMA` puts it. Any positional access is already wrong today.

**The batch window is derived from the batch.** Check 6 asks whether a timestamp is inside the
window and not future-dated, and the honest reference point is the batch's own `ingested_at` stamp —
a deterministic `window_end 23:59:59Z` written by the generator — not `date.today()`. A clock
reading would make the same bytes pass today and fail tomorrow, which destroys reproducibility and
makes the check untestable.

**A drift check with no baseline is `SKIPPED`, never `PASS`.** Checks 1, 10, 11 and 12 compare
against a 7-day rolling baseline. On a first run there is none. Reporting four passes there would
put green ticks against work that never happened, and a `SKIPPED` result is the only honest answer.
`DqCheckStatus.SKIPPED` never satisfies a blocking gate.

**Check 9 is internal-edge integrity, not debit/credit reconciliation.** The old wording assumed a
two-leg ledger; this system writes one row per movement of money, so there is no second leg and the
check could never fail. What is actually worth checking on a `counterparty_bank='IDFB'` row is that
both endpoints resolve to known accounts, because that is the shape that silently breaks the Stage 8
graph load. `counterparty_handle` is `None` on every one of those rows **by design** — an internal
transfer has no external VPA — so it is not flagged here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Final

import pandas as pd

from sentinel.domain.enums import Channel, Direction, DqCheckStatus, DqSeverity
from sentinel.domain.exceptions import DomainValidationError
from sentinel.domain.value_objects import BatchProfile

if TYPE_CHECKING:
    from sentinel.config.settings import Settings

# --- column groups -----------------------------------------------------------------------------
# Named here rather than inline so the report, the profile, and the checks all agree on which
# columns are mandatory and which are legitimately sparse.

#: Check 3's list, verbatim from the check table.
MANDATORY_COLUMNS: Final = ("txn_id", "txn_ts", "amount", "channel", "direction")

#: Columns that may legitimately be null, and whose null *rate* therefore carries information.
#: `from_account` is null for money arriving from outside the bank, `to_account` for money leaving
#: it, `counterparty_handle`/`counterparty_bank` for rails with no such concept, and
#: `device_id`/`ip_address` for the ~60% of accounts not enrolled in the app. A step change in any
#: of these means an upstream feed changed shape, which is what check 10 is for.
OPTIONAL_COLUMNS: Final = (
    "from_account",
    "to_account",
    "counterparty_handle",
    "counterparty_bank",
    "device_id",
    "ip_address",
)

#: The bank's own IFSC prefix. A transaction naming this as the counterparty bank is an internal
#: transfer: both endpoints are accounts this system holds, and check 9 insists on it.
INTERNAL_BANK_CODE: Final = "IDFB"

#: Quantiles stored in a `BatchProfile` and compared by check 12. Nine inner boundaries rather than
#: eleven: the minimum and maximum of a 200k-row batch are single observations and move on noise,
#: while every boundary here is backed by ~20k rows.
PROFILE_QUANTILES: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

TRANSACTIONS_TABLE: Final = "transactions"

_NO_BASELINE: Final = "no baseline available"


# --- the result type ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """What one check found. The unit the report, the persisted rows, and the gate all read.

    Deliberately *not* a `DqCheckResult`: that entity mirrors the `dq_check_results` table and
    carries a `run_id` and a database-assigned `result_id`, neither of which a pure function can
    know. `to_result` maps one to the other once the runner has a run id, which keeps the check
    functions free of any notion that persistence exists.

    `check_id` is a stable machine key (`DQ04_amount_bounds`) and is what lands in the
    `check_name` column, so a report can be diffed across runs even if the human `name` is reworded.
    `expectation` and `observed` are the two halves an operator reads side by side — "0 duplicates"
    against "37 duplicate txn_id" — and `message` is the one-line summary for a log or a dashboard.
    """

    check_id: str
    name: str
    severity: DqSeverity
    status: DqCheckStatus
    target_table: str
    target_column: str | None
    expectation: str
    observed: str
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise DomainValidationError(
                f"{self.check_id} produced no message; a result must say "
                f"what it found in words a human can read."
            )

    @property
    def blocks_batch(self) -> bool:
        """True when this single result is enough to stop the load.

        `SKIPPED` is excluded on purpose. A check that never ran has not found a defect, so it
        cannot block; what it does instead is downgrade the run to `PASSED_WITH_WARNINGS`, which is
        `pipeline/quality.py`'s job rather than this property's.
        """
        return self.severity is DqSeverity.BLOCKING and self.status is DqCheckStatus.FAIL

    @property
    def is_unverified(self) -> bool:
        """True when the check could not run and therefore verified nothing."""
        return self.status is DqCheckStatus.SKIPPED


# --- the batch window --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchWindow:
    """The half-open interval a batch's timestamps must fall inside, derived from the batch itself.

    `start` is midnight UTC on the earliest partition date. `end` is the batch's own `ingested_at`
    stamp, which the generator fixes at `window_end 23:59:59Z` — a deterministic value, so the same
    bytes give the same window on any day of the year. Nothing here consults `date.today()`.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise DomainValidationError(
                f"batch window runs backwards: {self.end.isoformat()} precedes "
                f"{self.start.isoformat()}."
            )

    @property
    def start_date(self) -> date:
        return self.start.date()

    @property
    def end_date(self) -> date:
        return self.end.date()


def derive_window(transactions: pd.DataFrame) -> BatchWindow:
    """Read the batch's own window off its rows.

    The upper bound is `max(ingested_at)` rather than `max(txn_ts)`, and the difference is the whole
    point: bounding timestamps by the largest timestamp present would make check 6 tautological — a
    row dated the year 3000 would simply move the bound and pass. `ingested_at` is stamped by the
    writer, is identical on every row of a batch, and is the honest "as of" instant to judge a
    payment timestamp against.
    """
    if transactions.empty:
        raise DomainValidationError(
            "cannot derive a batch window from an empty batch; there is nothing to validate."
        )
    earliest_partition = min(transactions["txn_date"])
    ingested = transactions["ingested_at"].max()
    return BatchWindow(
        start=datetime.combine(earliest_partition, time.min, tzinfo=UTC),
        end=ingested.to_pydatetime(),
    )


# --- the input a check receives ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything the twelve checks need, gathered once by the runner.

    One context object rather than four parameters because every check has the same signature —
    that is what lets `ALL_CHECKS` be a plain tuple the runner iterates, with no dispatch table and
    no per-check argument wiring. Checks that need only two of the five fields ignore the rest.

    `account_ids` is a `frozenset` because checks 5 and 9 do 200k membership tests against it; a
    `Series.isin` over a list would be the one place this module's cost showed.

    `baseline` is the most recent batch profiles, newest first, **excluding this batch's own**. An
    empty tuple is the first-run case and is what turns the four drift checks into `SKIPPED`.
    """

    batch_id: str
    transactions: pd.DataFrame
    account_ids: frozenset[str]
    window: BatchWindow
    baseline: tuple[BatchProfile, ...]
    settings: Settings

    @property
    def row_count(self) -> int:
        return len(self.transactions)


CheckFn = Callable[[CheckContext], CheckOutcome]


# --- shared helpers ----------------------------------------------------------------------------


def _outcome(
    *,
    check_id: str,
    name: str,
    severity: DqSeverity,
    column: str | None,
    expectation: str,
    observed: str,
    failed: bool,
    detail: str,
) -> CheckOutcome:
    """Build a pass/fail outcome, so no check has to repeat the status/message boilerplate."""
    status = DqCheckStatus.FAIL if failed else DqCheckStatus.PASS
    verb = "FAILED" if failed else "passed"
    return CheckOutcome(
        check_id=check_id,
        name=name,
        severity=severity,
        status=status,
        target_table=TRANSACTIONS_TABLE,
        target_column=column,
        expectation=expectation,
        observed=observed,
        message=f"{name} {verb}: {detail}",
    )


def _skipped(
    *,
    check_id: str,
    name: str,
    severity: DqSeverity,
    column: str | None,
    expectation: str,
    reason: str = _NO_BASELINE,
) -> CheckOutcome:
    """Build a `SKIPPED` outcome carrying why nothing was verified.

    The reason is repeated into `observed` as well as `message` because `observed` is the column an
    operator scans down a report, and "not evaluated - no baseline available" there is what stops a
    skipped drift check being read as a clean one.
    """
    return CheckOutcome(
        check_id=check_id,
        name=name,
        severity=severity,
        status=DqCheckStatus.SKIPPED,
        target_table=TRANSACTIONS_TABLE,
        target_column=column,
        expectation=expectation,
        observed=f"not evaluated - {reason}",
        message=f"{name} skipped: {reason}, so this check verified nothing.",
    )


def _decimal_amounts(transactions: pd.DataFrame) -> list[Decimal]:
    """The `amount` column as `Decimal`, with nulls dropped.

    Nulls are check 3's business, not check 4's — reporting one missing amount as both a null
    violation and a bounds violation would make `checks_failed` double-count a single defect. Values
    that are not `Decimal` at all cannot occur through the Parquet path (the column is
    `decimal128(18, 2)`), and filtering on the type rather than asserting it keeps a hand-built test
    DataFrame from crashing the check it is trying to exercise.
    """
    return [value for value in transactions["amount"] if isinstance(value, Decimal)]


def _null_rate(transactions: pd.DataFrame, column: str) -> float:
    """Share of rows where `column` is null, in 0..1."""
    if transactions.empty:
        return 0.0
    return float(transactions[column].isna().sum()) / len(transactions)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _baseline_null_rates(baseline: Sequence[BatchProfile]) -> dict[str, float]:
    """Mean null rate per column across the baseline profiles.

    A column absent from a profile is skipped for that profile rather than counted as zero: an older
    batch that did not carry the column at all has no opinion about its null rate, and treating
    silence as 0.0 would manufacture a 100pp drift the moment a new column appeared.
    """
    columns = {column for profile in baseline for column in profile.null_rates}
    return {
        column: _mean([p.null_rates[column] for p in baseline if column in p.null_rates])
        for column in columns
    }


def _baseline_channel_mix(baseline: Sequence[BatchProfile]) -> dict[str, float]:
    """Mean share per channel across the baseline profiles.

    Absence *is* zero here, unlike null rates: a channel missing from a profile means that batch
    carried no transactions on it, which is a share of exactly 0.0 and a real fact to compare
    against. A rail appearing for the first time is precisely the drift this check should catch.
    """
    channels = {channel for profile in baseline for channel in profile.channel_mix}
    return {
        channel: _mean([p.channel_mix.get(channel, 0.0) for p in baseline]) for channel in channels
    }


def _quantile(sorted_amounts: Sequence[Decimal], fraction: float) -> Decimal:
    """Nearest-rank quantile over pre-sorted amounts, in `Decimal`.

    Nearest-rank rather than interpolated because an interpolated boundary is the average of two
    rupee amounts, and averaging money in `float` is the thing this module refuses to do. The
    boundary only has to be a stable, comparable rupee figure; it does not have to be a value the
    batch never contained.
    """
    index = min(len(sorted_amounts) - 1, int(fraction * len(sorted_amounts)))
    return sorted_amounts[index]


def compute_profile(context: CheckContext, computed_at: datetime) -> BatchProfile:
    """Summarise this batch so the *next* one has something to be compared against.

    Persisted only by a run that was not blocked — a batch the gate refused is not a description of
    what normal looks like, and folding it into the baseline would widen the tolerances that just
    caught it.
    """
    amounts = sorted(_decimal_amounts(context.transactions))
    rows = context.row_count
    channel_counts = context.transactions["channel"].value_counts()
    return BatchProfile(
        batch_id=context.batch_id,
        computed_at=computed_at,
        window_end=context.window.end_date,
        row_count=rows,
        null_rates={
            column: _null_rate(context.transactions, column) for column in OPTIONAL_COLUMNS
        },
        channel_mix={str(channel): int(count) / rows for channel, count in channel_counts.items()}
        if rows
        else {},
        amount_deciles=tuple(_quantile(amounts, q) for q in PROFILE_QUANTILES) if amounts else (),
    )


# --- 1. row count vs the rolling baseline ------------------------------------------------------


def check_row_count(context: CheckContext) -> CheckOutcome:
    """Row count within +/-40% of the rolling mean. BLOCKING.

    A batch that arrives at a fraction of its usual size is a truncated feed, and one that arrives
    at several times its usual size is a duplicated load. Both are silent — every individual row
    passes every other check — which is why this one is blocking despite being the crudest.
    """
    check_id, name = "DQ01_row_count_tolerance", "Row count within rolling tolerance"
    tolerance = context.settings.dq_row_count_tolerance
    expectation = (
        f"within +/-{tolerance:.0%} of the {context.settings.dq_baseline_batches}-batch mean"
    )

    if not context.baseline:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.BLOCKING,
            column=None,
            expectation=expectation,
        )

    mean = _mean([float(profile.row_count) for profile in context.baseline])
    rows = context.row_count
    if mean <= 0:
        # Every baseline batch was empty. There is a mean, but comparing against zero would make
        # any non-empty batch infinitely out of tolerance, which says nothing about this batch.
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.BLOCKING,
            column=None,
            expectation=expectation,
            reason="every baseline batch was empty",
        )

    low, high = mean * (1.0 - tolerance), mean * (1.0 + tolerance)
    drift = (rows - mean) / mean
    return _outcome(
        check_id=check_id,
        name=name,
        severity=DqSeverity.BLOCKING,
        column=None,
        expectation=f"{low:,.0f} to {high:,.0f} rows ({expectation})",
        observed=f"{rows:,} rows, {drift:+.1%} against a baseline mean of {mean:,.0f}",
        failed=not low <= rows <= high,
        detail=f"{rows:,} rows against a baseline mean of {mean:,.0f} ({drift:+.1%})",
    )


# --- 2. txn_id uniqueness ----------------------------------------------------------------------


def check_txn_id_unique(context: CheckContext) -> CheckOutcome:
    """`txn_id` is unique across the batch. BLOCKING.

    A duplicate identifier double-counts money in every downstream aggregate and makes the Postgres
    upsert at Stage 5 non-deterministic about which copy wins.
    """
    duplicates = int(context.transactions["txn_id"].duplicated().sum())
    return _outcome(
        check_id="DQ02_txn_id_unique",
        name="txn_id uniqueness",
        severity=DqSeverity.BLOCKING,
        column="txn_id",
        expectation="0 duplicate txn_id",
        observed=f"{duplicates:,} duplicate txn_id",
        failed=duplicates > 0,
        detail=f"{duplicates:,} of {context.row_count:,} rows repeat an existing txn_id",
    )


# --- 3. mandatory fields non-null --------------------------------------------------------------


def check_mandatory_fields(context: CheckContext) -> CheckOutcome:
    """`txn_id`, `txn_ts`, `amount`, `channel`, `direction` are all present. BLOCKING.

    These five are the columns without which a row is not a transaction at all: it cannot be
    identified, placed in time, valued, attributed to a rail, or given a sign.
    """
    per_column = {
        column: int(context.transactions[column].isna().sum()) for column in MANDATORY_COLUMNS
    }
    offenders = {column: count for column, count in per_column.items() if count}
    total = sum(offenders.values())
    listed = ", ".join(f"{column}={count:,}" for column, count in sorted(offenders.items()))
    return _outcome(
        check_id="DQ03_mandatory_not_null",
        name="Mandatory fields non-null",
        severity=DqSeverity.BLOCKING,
        column=None,
        expectation=f"0 nulls across {', '.join(MANDATORY_COLUMNS)}",
        observed=f"{total:,} nulls ({listed})" if offenders else "0 nulls",
        failed=total > 0,
        detail=f"{total:,} null value(s) in mandatory columns ({listed})"
        if offenders
        else f"all {len(MANDATORY_COLUMNS)} mandatory columns fully populated",
    )


# --- 4. amount bounds --------------------------------------------------------------------------


def check_amount_bounds(context: CheckContext) -> CheckOutcome:
    """`0 < amount <= Rs 10 crore`. BLOCKING.

    Compared in `Decimal` throughout. Direction carries the sign in this model, so a non-positive
    amount is a defect rather than a refund, and a retail payment above the cap is a misplaced
    decimal point rather than a large customer.
    """
    cap = context.settings.dq_max_amount_inr
    column = context.transactions["amount"]
    present = column[column.notna()]
    amounts = [value for value in present if isinstance(value, Decimal)]
    # A non-Decimal amount is a violation, not a value to skip. It cannot arrive through the
    # decimal128(18, 2) Parquet path, but a CSV or JSON feed would make it reachable — and a
    # bounds check that silently drops what it cannot compare is a check that cannot fail.
    malformed = len(present) - len(amounts)
    non_positive = sum(1 for value in amounts if value <= 0)
    over_cap = sum(1 for value in amounts if value > cap)
    total = non_positive + over_cap + malformed
    return _outcome(
        check_id="DQ04_amount_bounds",
        name="Amount within bounds",
        severity=DqSeverity.BLOCKING,
        column="amount",
        expectation=f"0 < amount <= INR {cap:,}",
        observed=f"{total:,} violation(s) ({non_positive:,} non-positive, {over_cap:,} above cap, "
        f"{malformed:,} not decimal)",
        failed=total > 0,
        detail=f"{non_positive:,} non-positive, {over_cap:,} above INR {cap:,} and {malformed:,} "
        f"not stored as a decimal, across {len(present):,} non-null amounts",
    )


# --- 5. referential integrity ------------------------------------------------------------------


def _unknown_mask(
    transactions: pd.DataFrame, column: str, known: frozenset[str]
) -> pd.Series[bool]:
    """Per-row: `column` is set and names an account the `accounts` table does not have.

    Null is check 3's business, so `notna()` keeps one missing value from being counted twice.
    """
    values = transactions[column]
    return values.notna() & ~values.isin(known)


def _orphans(transactions: pd.DataFrame, column: str, known: frozenset[str]) -> int:
    """Non-null values of `column` that name an account the `accounts` table does not have."""
    return int(_unknown_mask(transactions, column, known).sum())


def check_referential_integrity(context: CheckContext) -> CheckOutcome:
    """Every non-null `from_account` / `to_account` exists in `accounts`. BLOCKING.

    An orphan endpoint is a transaction attributed to an account nobody can look up: the feature
    join drops it, the graph load creates a phantom node, and the alert it should have raised is
    never raised at all.
    """
    from_orphans = _orphans(context.transactions, "from_account", context.account_ids)
    to_orphans = _orphans(context.transactions, "to_account", context.account_ids)
    total = from_orphans + to_orphans
    return _outcome(
        check_id="DQ05_referential_integrity",
        name="Account referential integrity",
        severity=DqSeverity.BLOCKING,
        column=None,
        expectation="0 endpoints outside the accounts table",
        observed=f"{total:,} orphans (from_account={from_orphans:,}, to_account={to_orphans:,})",
        failed=total > 0,
        detail=f"{from_orphans:,} unknown from_account and {to_orphans:,} unknown to_account "
        f"against {len(context.account_ids):,} known accounts",
    )


# --- 6. timestamps inside the window -----------------------------------------------------------


def check_txn_ts_window(context: CheckContext) -> CheckOutcome:
    """`txn_ts` lies inside the batch window and is not future-dated. BLOCKING.

    The window comes from the batch (see `derive_window`), never from the wall clock. A future-dated
    payment breaks every rolling feature window it lands in and would be counted as "recent
    activity" forever.
    """
    stamps = context.transactions["txn_ts"]
    opened, closed = context.window.start.isoformat(), context.window.end.isoformat()
    before = int((stamps < context.window.start).sum())
    after = int((stamps > context.window.end).sum())
    total = before + after
    return _outcome(
        check_id="DQ06_txn_ts_in_window",
        name="Timestamp inside the batch window",
        severity=DqSeverity.BLOCKING,
        column="txn_ts",
        expectation=f"{opened} <= txn_ts <= {closed}",
        observed=f"{total:,} outside the window ({before:,} early, {after:,} future-dated)",
        failed=total > 0,
        detail=f"{before:,} before {opened} and {after:,} after {closed}",
    )


# --- 7. enum domains ---------------------------------------------------------------------------


def _unknown_values(transactions: pd.DataFrame, column: str, allowed: frozenset[str]) -> int:
    """Non-null values of `column` outside `allowed`.

    Nulls are excluded because check 3 already owns them for `channel` and `direction`; counting
    them twice would inflate `checks_failed` on one defect.
    """
    series = transactions[column]
    return int((series.notna() & ~series.isin(allowed)).sum())


def check_enum_domains(context: CheckContext) -> CheckOutcome:
    """`channel` and `direction` hold only values the domain enums define. BLOCKING.

    Adding a payment rail is a deliberate act with a migration behind it, so an unrecognised value
    is an upstream schema change arriving unannounced — the case where continuing silently is worst.
    """
    channels = frozenset(str(member) for member in Channel)
    directions = frozenset(str(member) for member in Direction)
    bad_channel = _unknown_values(context.transactions, "channel", channels)
    bad_direction = _unknown_values(context.transactions, "direction", directions)
    total = bad_channel + bad_direction
    return _outcome(
        check_id="DQ07_enum_domains",
        name="Channel and direction in the allowed enum",
        severity=DqSeverity.BLOCKING,
        column=None,
        expectation=f"channel in {{{', '.join(sorted(channels))}}}, "
        f"direction in {{{', '.join(sorted(directions))}}}",
        observed=f"{total:,} unknown values (channel={bad_channel:,}, direction={bad_direction:,})",
        failed=total > 0,
        detail=f"{bad_channel:,} unrecognised channel and {bad_direction:,} unrecognised direction",
    )


# --- 8. currency -------------------------------------------------------------------------------


def check_currency(context: CheckContext) -> CheckOutcome:
    """Every row is `INR`. BLOCKING.

    This is a domestic retail ledger and every threshold in the system — the Rs 10 crore cap, the
    structuring band, the materiality ramp — is denominated in rupees. A foreign-currency row would
    be compared against rupee thresholds without anything converting it. Nulls count as violations
    here: `currency` is not in the mandatory list, so no other check would catch a missing one.
    """
    series = context.transactions["currency"]
    violations = int((~series.eq("INR")).sum())
    return _outcome(
        check_id="DQ08_currency_inr",
        name="Currency is INR",
        severity=DqSeverity.BLOCKING,
        column="currency",
        expectation="100% INR",
        observed=f"{violations:,} rows not INR",
        failed=violations > 0,
        detail=f"{context.row_count - violations:,} of {context.row_count:,} rows are INR",
    )


# --- 9. internal-edge integrity ----------------------------------------------------------------


def check_internal_edge_integrity(context: CheckContext) -> CheckOutcome:
    """Every internal transfer names two known accounts. WARNING.

    **Rewritten on 2026-08-03.** This slot previously held "debit/credit reconciliation - internal
    transfers balance within Rs 1", which assumes a two-leg ledger. This system stores one row per
    movement of money, so there is no second leg and the check could never fail. What is worth
    checking on a `counterparty_bank='IDFB'` row is that both endpoints resolve: a null or unknown
    `to_account` there is a genuine defect and is the shape that silently breaks the Stage 8 graph
    load, where an internal edge is expected to connect two nodes that both exist.

    `counterparty_handle` is null on all of these rows by design - an internal transfer has no
    external VPA - and is deliberately not flagged.

    WARNING rather than BLOCKING per the check table: a broken internal edge degrades the graph
    projection, it does not corrupt the ledger, and blocking a 200k-row batch over a handful of
    them would be the wrong trade.
    """
    internal = context.transactions[
        context.transactions["counterparty_bank"].eq(INTERNAL_BANK_CODE)
    ]
    if internal.empty:
        return _outcome(
            check_id="DQ09_internal_edge_integrity",
            name="Internal-edge integrity",
            severity=DqSeverity.WARNING,
            column="counterparty_bank",
            expectation=f"both endpoints known on every {INTERNAL_BANK_CODE} row",
            observed=f"0 {INTERNAL_BANK_CODE} rows in this batch",
            failed=False,
            detail=f"the batch contains no {INTERNAL_BANK_CODE} internal transfers",
        )

    # Counted per row, not per endpoint: one internal transfer with two bad endpoints is one
    # defective row. Summing endpoint counts let `observed` report more defects than rows.
    missing_mask = internal["from_account"].isna() | internal["to_account"].isna()
    unknown_mask = _unknown_mask(internal, "from_account", context.account_ids) | _unknown_mask(
        internal, "to_account", context.account_ids
    )
    missing = int(missing_mask.sum())
    unknown = int(unknown_mask.sum())
    total = int((missing_mask | unknown_mask).sum())
    return _outcome(
        check_id="DQ09_internal_edge_integrity",
        name="Internal-edge integrity",
        severity=DqSeverity.WARNING,
        column="counterparty_bank",
        expectation=f"both endpoints known on every {INTERNAL_BANK_CODE} row",
        observed=f"{total:,} defective of {len(internal):,} internal rows "
        f"({missing:,} missing an endpoint, {unknown:,} naming an unknown account)",
        failed=total > 0,
        detail=f"{missing:,} internal transfer(s) missing an endpoint and {unknown:,} naming an "
        f"account outside the accounts table, of {len(internal):,} internal rows",
    )


# --- 10. null-rate drift -----------------------------------------------------------------------


def check_null_rate_drift(context: CheckContext) -> CheckOutcome:
    """No optional column's null rate moves more than 15pp against the baseline. WARNING.

    A column that suddenly stops arriving still passes every row-level check — every value it does
    carry is valid. The only way to see it is to compare against what the feed usually looks like.
    """
    check_id, name = "DQ10_null_rate_drift", "Null-rate drift"
    limit = context.settings.dq_null_rate_drift
    expectation = f"no optional column drifts more than {limit * 100:.0f} percentage points"

    if not context.baseline:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column=None,
            expectation=expectation,
        )

    baseline = _baseline_null_rates(context.baseline)
    drifts = {
        column: _null_rate(context.transactions, column) - baseline[column]
        for column in OPTIONAL_COLUMNS
        if column in baseline
    }
    if not drifts:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column=None,
            expectation=expectation,
            reason="the baseline records no optional-column null rates",
        )

    offenders = {column: drift for column, drift in drifts.items() if abs(drift) > limit}
    worst_column = max(drifts, key=lambda column: abs(drifts[column]))
    listed = ", ".join(f"{column} {drift:+.1%}" for column, drift in sorted(offenders.items()))
    return _outcome(
        check_id=check_id,
        name=name,
        severity=DqSeverity.WARNING,
        column=None,
        expectation=expectation,
        observed=f"{len(offenders)} column(s) beyond tolerance"
        + (
            f": {listed}"
            if offenders
            else f"; largest move {worst_column} {drifts[worst_column]:+.1%}"
        ),
        failed=bool(offenders),
        detail=f"largest move is {worst_column} at {drifts[worst_column]:+.1%} across "
        f"{len(drifts)} optional column(s)",
    )


# --- 11. channel-mix drift ---------------------------------------------------------------------


def check_channel_mix_drift(context: CheckContext) -> CheckOutcome:
    """No channel's share of the batch moves more than 20pp against the baseline. WARNING.

    UPI dominating by count is the shape of Indian retail. A batch where that flips has either lost
    a feed or gained one, and every velocity feature computed over it is measuring a different
    population from the one the rules were tuned against.
    """
    check_id, name = "DQ11_channel_mix_drift", "Channel-mix drift"
    limit = context.settings.dq_channel_mix_drift
    expectation = f"no channel share moves more than {limit * 100:.0f} percentage points"

    if not context.baseline:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column="channel",
            expectation=expectation,
        )

    baseline = _baseline_channel_mix(context.baseline)
    rows = context.row_count
    counts = context.transactions["channel"].value_counts()
    current = {str(channel): int(count) / rows for channel, count in counts.items()} if rows else {}
    # The union, not the current batch's keys: a channel that vanished entirely has drifted by its
    # whole baseline share, and iterating only over what arrived would miss exactly that case.
    drifts = {
        channel: current.get(channel, 0.0) - baseline.get(channel, 0.0)
        for channel in set(current) | set(baseline)
    }
    if not drifts:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column="channel",
            expectation=expectation,
            reason="neither the batch nor the baseline records any channel",
        )

    offenders = {channel: drift for channel, drift in drifts.items() if abs(drift) > limit}
    worst = max(drifts, key=lambda channel: abs(drifts[channel]))
    listed = ", ".join(f"{channel} {drift:+.1%}" for channel, drift in sorted(offenders.items()))
    return _outcome(
        check_id=check_id,
        name=name,
        severity=DqSeverity.WARNING,
        column="channel",
        expectation=expectation,
        observed=f"{len(offenders)} channel(s) beyond tolerance"
        + (f": {listed}" if offenders else f"; largest move {worst} {drifts[worst]:+.1%}"),
        failed=bool(offenders),
        detail=f"largest move is {worst} at {drifts[worst]:+.1%} across {len(drifts)} channel(s)",
    )


# --- 12. amount-distribution drift -------------------------------------------------------------


def check_amount_distribution_drift(context: CheckContext) -> CheckOutcome:
    """KS-style comparison of the amount distribution against the baseline deciles. WARNING.

    The statistic is the largest gap between where the baseline's deciles *should* cut this batch
    and where they actually do: if the 30th-percentile boundary of a normal batch cuts this one at
    the 55th percentile, D is 0.25 and the distribution has moved. Deciles rather than a mean
    because a mean cannot separate "a few huge transfers arrived" from "the whole population
    shifted", and only the second is drift worth reporting.
    """
    check_id, name = "DQ12_amount_distribution_drift", "Amount-distribution drift"
    limit = context.settings.dq_amount_drift_ks
    expectation = f"KS-style deviation D <= {limit:.2f} against the baseline deciles"

    usable = [
        profile
        for profile in context.baseline
        if len(profile.amount_deciles) == len(PROFILE_QUANTILES)
    ]
    if not usable:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column="amount",
            expectation=expectation,
            reason=_NO_BASELINE if not context.baseline else "no baseline carries amount deciles",
        )

    amounts = sorted(_decimal_amounts(context.transactions))
    if not amounts:
        return _skipped(
            check_id=check_id,
            name=name,
            severity=DqSeverity.WARNING,
            column="amount",
            expectation=expectation,
            reason="the batch carries no readable amounts",
        )

    # Pool the baseline decile ladders position by position. Averaging the i-th boundary across
    # batches is a sound pooled estimate of that boundary and keeps the ladder monotonic, which
    # concatenating and re-quantising every historical amount would also do at far greater cost.
    total = len(amounts)
    deviation = 0.0
    worst_at = 0.0
    for index, fraction in enumerate(PROFILE_QUANTILES):
        boundary = sum((p.amount_deciles[index] for p in usable), Decimal(0)) / len(usable)
        # Rank of the boundary in this batch: how much of the batch sits at or below it.
        at_or_below = sum(1 for value in amounts if value <= boundary)
        gap = abs(at_or_below / total - fraction)
        if gap > deviation:
            deviation, worst_at = gap, fraction
    return _outcome(
        check_id=check_id,
        name=name,
        severity=DqSeverity.WARNING,
        column="amount",
        expectation=expectation,
        observed=f"D={deviation:.3f} at the {worst_at:.0%} decile",
        failed=deviation > limit,
        detail=f"largest deviation D={deviation:.3f} at the {worst_at:.0%} decile against "
        f"{len(usable)} baseline batch(es)",
    )


# --- the registry ------------------------------------------------------------------------------

#: The twelve checks in the order `docs/05-DATA-FLOW.md` §4 lists them. The runner iterates this
#: tuple and nothing else, so adding a thirteenth check is one entry here plus one function above —
#: there is no dispatch table, no registration decorator, and no plugin lookup to keep in step.
ALL_CHECKS: Final[tuple[CheckFn, ...]] = (
    check_row_count,
    check_txn_id_unique,
    check_mandatory_fields,
    check_amount_bounds,
    check_referential_integrity,
    check_txn_ts_window,
    check_enum_domains,
    check_currency,
    check_internal_edge_integrity,
    check_null_rate_drift,
    check_channel_mix_drift,
    check_amount_distribution_drift,
)


def run_all(context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Run every check against one batch, in table order. Pure — no I/O, no clock, no randomness.

    Every check runs; none short-circuits on an earlier failure. An operator fixing a broken feed
    needs the whole list of what is wrong with it, not the first thing that went wrong.
    """
    return tuple(check(context) for check in ALL_CHECKS)


def build_context(
    *,
    batch_id: str,
    transactions: pd.DataFrame,
    account_ids: frozenset[str],
    baseline: Sequence[BatchProfile] = (),
    settings: Settings,
) -> CheckContext:
    """Assemble a `CheckContext`, deriving the batch window from the rows themselves."""
    return CheckContext(
        batch_id=batch_id,
        transactions=transactions,
        account_ids=account_ids,
        window=derive_window(transactions),
        baseline=tuple(baseline),
        settings=settings,
    )


def summarise(outcomes: Sequence[CheckOutcome]) -> Mapping[str, int]:
    """Counts by status, for a log line and the report header."""
    return {
        "total": len(outcomes),
        "passed": sum(1 for o in outcomes if o.status is DqCheckStatus.PASS),
        "failed": sum(1 for o in outcomes if o.status is DqCheckStatus.FAIL),
        "skipped": sum(1 for o in outcomes if o.is_unverified),
        "blocking_failures": sum(1 for o in outcomes if o.blocks_batch),
    }
