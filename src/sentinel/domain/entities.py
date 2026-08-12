"""The twelve persisted row shapes, as plain Python objects.

One dataclass per table in `docs/02-DATA-MODEL.md`, in the same column order, so a reviewer can
read the schema and this file side by side. Nothing here knows that Postgres exists: adapters in
`infrastructure/postgres/` map rows to these, and every other layer works with these instead of a
`dict` or a driver row. That is the whole point of the layer — the rule engine, the agent, and the
API all fail at import time if a column is renamed, rather than returning `None` at runtime.

Four decisions worth defending, because they are the ones that would be wrong if reversed:

**`frozen=True`** — these objects cross layer boundaries. A loaded row is a fact about the past; a
service that mutates one has produced a value that agrees with nothing in the database. Freezing
turns that class of bug into an `AttributeError` at the moment it is written.

**`slots=True`** — around 200k `Transaction` instances exist at once during a batch. Slots drop the
per-instance `__dict__`, which is the difference between a pipeline that fits in memory on a laptop
and one that does not.

**`Decimal` for money, `float` for statistics** — every amount, balance, and cost is `Decimal`,
never `float`. Binary floating point cannot represent 0.1 exactly, so summing 200k transaction
amounts drifts; in a bank a drifting total is a defect, not a rounding detail. Dimensionless
quantities (ratios, z-scores, confidences, similarity) stay `float`: they are never summed into a
balance, they feed scikit-learn, and `Decimal` would buy nothing.

**Timezone-aware timestamps, always** — a naive `datetime` in this system is a bug. It has lost the
offset, so a night-hour feature (00:00-05:00 is the mule dispersal window) silently computes against
the wrong local time. The `ensure_aware` guard below rejects them at construction. `txn_date` is a
`date` because it is the partition key, not an instant.

`Mapping` and `tuple` annotations on the JSONB columns are the read-only halves of `dict` and
`list`. Be precise about what that buys: `frozen=True` stops the *field* being rebound, and the
read-only annotation makes `alert.trigger_evidence["k"] = 1` a **mypy error** — but the object
handed in is still a real `dict` at runtime, and that statement will happily mutate it. Static
checking is the enforcement here, not the interpreter, so callers must not pass a `dict` they
intend to keep mutating.

Two runtime consequences of that, worth knowing before you hit them:

* An entity carrying a JSONB field is **unhashable**. `frozen=True` generates `__hash__` from every
  field, so `hash(alert)` raises `TypeError: unhashable type: 'dict'` — frozen does not imply
  hashable here. Nothing puts these in a `set` or uses them as a `dict` key today; if something
  needs to, key on the identifier (`alert.alert_id`) rather than the object.
* Equality is still sound, because `dict.__eq__` compares by value.

This module lives in `domain` and imports nothing outside the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

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
    FeatureWindow,
    InvestigationStatus,
    KycLevel,
    Recommendation,
    RiskRating,
    Severity,
    TransactionStatus,
    Typology,
)
from sentinel.domain.exceptions import DomainValidationError

# --- field guards ------------------------------------------------------------------------------
# Four functions shared by every `__post_init__` in this package. They live here rather than in a
# fifth module because `value_objects.py` is their only other caller.


def ensure_aware(value: datetime | None, name: str) -> None:
    """Reject a naive timestamp.

    `utcoffset() is None` covers both a missing `tzinfo` and a `tzinfo` that declines to supply an
    offset, which is the actual condition that breaks arithmetic between two timestamps.
    """
    if value is not None and value.utcoffset() is None:
        raise DomainValidationError(
            f"{name} must be timezone-aware, got the naive datetime {value.isoformat()}. "
            f"Attach a tzinfo at the boundary that parsed it."
        )


def ensure_positive(value: Decimal, name: str) -> None:
    """Reject an amount that is zero or negative, for fields where neither can be legitimate."""
    if value <= 0:
        raise DomainValidationError(f"{name} must be greater than zero, got {value}.")


def ensure_non_negative(value: Decimal, name: str) -> None:
    """Reject a negative aggregate. A window total or a cost can be zero but never below it."""
    if value < 0:
        raise DomainValidationError(f"{name} must not be negative, got {value}.")


def ensure_ordered(earlier: date, later: date | None, earlier_name: str, later_name: str) -> None:
    """Reject a pair of instants that runs backwards.

    Used by every start/end and first/last pair in the package. Reversed pairs are silent poison
    rather than loud errors: a device whose `last_seen_at` precedes its `first_seen_at` yields a
    negative dwell time, and a negative investigation latency drags the reported mean down without
    anything failing.

    `later is None` is legitimate everywhere this is called — an in-flight investigation, an account
    that has never transacted — so it is skipped rather than rejected. `datetime` is a subclass of
    `date`, which is why one guard serves both; callers pass a matching pair.
    """
    if later is not None and later < earlier:
        raise DomainValidationError(
            f"{later_name} {later.isoformat()} precedes {earlier_name} {earlier.isoformat()}."
        )


def ensure_in_range(value: float | None, name: str, low: float, high: float) -> None:
    """Reject a statistic outside its defined interval.

    Worth the three lines: a confidence of `88` where `0.88` was meant clears the human gate's
    `>= 0.70` test and lets a case self-close. A range check is what stops a units mistake from
    becoming a governance failure.
    """
    if value is not None and not (low <= value <= high):
        raise DomainValidationError(f"{name} must be between {low} and {high}, got {value}.")


def compute_pass_through_ratio(total_credit: Decimal, total_debit: Decimal) -> float:
    """Debits divided by credits over a window — the mule signature, defined exactly once.

    This function is the single definition. `FeatureVector.pass_through_ratio` returns it and
    `AccountFeatures.__post_init__` checks the stored column against it, so the batch path and the
    serving path cannot drift apart — that drift between how a feature is computed for training and
    for serving is the classic production bug, and one function is what actually prevents it rather
    than a comment saying it must not happen.

    Zero credit means there was nothing to pass through, which is 0.0 rather than undefined —
    matching the `CASE WHEN total_credit > 0 ... ELSE 0` in the DuckDB feature SQL. The division is
    done in `Decimal` and converted once at the end, so no intermediate value is a binary float.
    """
    if total_credit <= 0:
        return 0.0
    return float(total_debit / total_credit)


# --- reference data ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Customer:
    """A retail customer. Every field that could identify a person is masked or hashed.

    `pan_hash` is a salted SHA-256 hex digest of the PAN (Permanent Account Number, India's tax
    identifier) — salt from `Settings.pan_salt`. It is **never** a plaintext PAN, and it is one-way:
    it exists to join and deduplicate, and nothing in this codebase can reverse it. `full_name` is
    retained because an AML narrative is unreadable without it; the graph projection stores only a
    masked form.
    """

    customer_id: str
    full_name: str
    date_of_birth: date
    pan_hash: str
    phone_masked: str
    email_masked: str
    address_line: str
    city: str
    state: str
    pincode: str
    occupation: str
    declared_annual_income: Decimal
    kyc_level: KycLevel
    risk_rating: RiskRating
    is_pep: bool
    onboarded_on: date
    segment: CustomerSegment

    def __post_init__(self) -> None:
        ensure_non_negative(self.declared_annual_income, "declared_annual_income")


@dataclass(frozen=True, slots=True)
class Account:
    """A bank account — the unit everything else in the system is about.

    Alerts are raised on accounts, features are computed per account, and the graph's nodes are
    accounts. `last_activity_on` is `None` for an account that has never transacted, which is
    distinct from one that last transacted long ago; dormancy logic must not conflate them.
    """

    account_id: str
    customer_id: str
    account_type: AccountType
    opened_on: date
    status: AccountStatus
    branch_code: str
    ifsc: str
    current_balance: Decimal
    last_activity_on: date | None
    is_flagged: bool
    created_at: datetime

    def __post_init__(self) -> None:
        # `current_balance` is deliberately unguarded: a CURRENT account can legitimately sit
        # overdrawn, so a negative balance is data, not an error.
        ensure_aware(self.created_at, "created_at")
        ensure_ordered(self.opened_on, self.last_activity_on, "opened_on", "last_activity_on")


@dataclass(frozen=True, slots=True)
class Device:
    """A device fingerprint. Sharing one across accounts of *different* customers is the strongest
    single identity-linkage signal in the system (rule R007, typology `DEVICE_RING`)."""

    device_id: str
    fingerprint: str
    os_family: str
    app_version: str
    first_seen_at: datetime
    last_seen_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.first_seen_at, "first_seen_at")
        ensure_aware(self.last_seen_at, "last_seen_at")
        ensure_ordered(self.first_seen_at, self.last_seen_at, "first_seen_at", "last_seen_at")


# --- transactions ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transaction:
    """One movement of money. The only high-volume entity — roughly 200k per batch.

    `from_account` is `None` for money arriving from outside the bank and `to_account` is `None` for
    money leaving it; at least one side is always an internal account. `counterparty_handle` (a UPI
    VPA or an external account reference) and `device_id` are `None` for rails that have no such
    concept, such as `CASH` at a branch counter.

    `txn_date` duplicates the date part of `txn_ts` on purpose: it is the Postgres partition key and
    the Parquet partition directory, and deriving it per query would defeat partition pruning.
    """

    txn_id: str
    txn_ts: datetime
    from_account: str | None
    to_account: str | None
    counterparty_handle: str | None
    counterparty_bank: str | None
    amount: Decimal
    currency: str
    channel: Channel
    direction: Direction
    status: TransactionStatus
    device_id: str | None
    ip_address: str | None
    narration: str
    txn_date: date
    ingested_at: datetime
    batch_id: str

    def __post_init__(self) -> None:
        ensure_aware(self.txn_ts, "txn_ts")
        ensure_aware(self.ingested_at, "ingested_at")
        # Direction carries the sign; the amount is always a magnitude. A zero-value transaction is
        # not a payment, and a negative one is a reversal that belongs in `status`, not the amount.
        ensure_positive(self.amount, "amount")


# --- features ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccountFeatures:
    """A persisted `account_features` row: one account, one window, one `as_of` instant.

    This is the stored mirror of what DuckDB computed. `FeatureVector` in `value_objects.py` is the
    same numbers as an in-flight computation result — the two are separate because a row can be
    stale and carries no derivation, whereas a vector is what a caller just asked for.

    Ratios and the z-score are `float`: they are dimensionless statistics that go straight into
    scikit-learn, never into a balance.

    The field order is the column order of `docs/02-DATA-MODEL.md`'s `ACCOUNT_FEATURES`, which is
    also the order `pipeline/features.FEATURE_SCHEMA` writes. All three are meant to be readable
    side by side, so a column added to one and missed by another shows up as a misalignment rather
    than as a `KeyError` in the Stage 5 loader.

    **`balance_delta` is required and carries no default**, unlike every other signed statistic that
    could plausibly start at zero. A default would let a loader that forgot the column reconstruct
    `Decimal("0")`, and a zero here is indistinguishable from a genuine break-even week — which is
    precisely the family-pooling false positive in `docs/13-KNOWN-FALSE-POSITIVES.md` that this
    column exists to separate from a real collection mule. Missing must fail loudly at construction.
    """

    account_id: str
    window: FeatureWindow
    as_of: datetime
    txn_count: int
    credit_count: int
    debit_count: int
    total_credit: Decimal
    total_debit: Decimal
    balance_delta: Decimal
    distinct_credit_counterparties: int
    distinct_debit_counterparties: int
    max_single_txn: Decimal
    avg_txn_amount: Decimal
    pass_through_ratio: float
    night_txn_ratio: float
    new_counterparty_ratio: float
    structuring_band_count: int
    distinct_devices: int
    distinct_ips: int
    days_since_last_activity: int
    velocity_zscore: float

    def __post_init__(self) -> None:
        ensure_aware(self.as_of, "as_of")
        ensure_non_negative(self.total_credit, "total_credit")
        ensure_non_negative(self.total_debit, "total_debit")
        # `balance_delta` is deliberately unguarded, unlike the two totals it is derived from: it is
        # signed, and negative is the ordinary case for an account spending down a balance it held
        # before the window opened. Same rule as `FeatureVector.balance_delta`.
        # The three counts join `txn_count` in being unguarded: they are `int` aggregates the SQL
        # coalesces to zero, and there is no bound to check that the type does not already give.
        # `pass_through_ratio` is intentionally not capped at 1.0 — an account can disperse more
        # than it received in the window by spending an opening balance, and that is a real signal.
        ensure_in_range(self.night_txn_ratio, "night_txn_ratio", 0.0, 1.0)
        ensure_in_range(self.new_counterparty_ratio, "new_counterparty_ratio", 0.0, 1.0)
        # The stored column must agree with the one definition of the feature. Without this, a row
        # written by a future adapter that forgot to populate it (0.0 against a real 1.5) would make
        # the batch and serving paths disagree — the exact drift `compute_pass_through_ratio` exists
        # to prevent. `abs_tol` absorbs a round-trip through a scaled Postgres `numeric` column; it
        # is four decimal places, which is far tighter than any genuine contradiction.
        # ponytail: fixed 1e-4 tolerance. If the DDL ever pins a coarser scale than numeric(_, 4),
        # widen it there and here together.
        expected = compute_pass_through_ratio(self.total_credit, self.total_debit)
        if not math.isclose(self.pass_through_ratio, expected, abs_tol=1e-4):
            raise DomainValidationError(
                f"pass_through_ratio {self.pass_through_ratio} contradicts total_debit "
                f"{self.total_debit} / total_credit {self.total_credit}, which is {expected}."
            )


# --- detection ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Alert:
    """One rule firing on one account over one window — the agent's input contract.

    `trigger_evidence` holds the rule's own "why" as JSONB, so the agent starts with the numbers
    that caused the alert instead of having to rediscover them. `anomaly_score` is `None` until the
    Isolation Forest has scored the batch, and is *not* range-checked: the model emits negative
    values for outliers, which is the interesting end.

    Shaped so that a vendor TMS feed (Actimize, SAS, Mantas) can replace the local rule engine
    without the agent changing.
    """

    alert_id: str
    account_id: str
    rule_code: str
    rule_name: str
    triggered_at: datetime
    window_start: datetime
    window_end: datetime
    severity: Severity
    rule_score: float
    anomaly_score: float | None
    trigger_evidence: Mapping[str, Any]
    status: AlertStatus
    created_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.triggered_at, "triggered_at")
        ensure_aware(self.window_start, "window_start")
        ensure_aware(self.window_end, "window_end")
        ensure_aware(self.created_at, "created_at")
        ensure_ordered(self.window_start, self.window_end, "window_start", "window_end")


# --- investigation -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Investigation:
    """One agent run over one alert. A recommendation, never a decision.

    The optional fields are optional in time, not in importance: a `RUNNING` investigation has no
    recommendation, confidence, score, typology, narrative, or completion instant yet. Reading them
    as `None` is the honest representation of "the agent has not got there".

    `computed_risk_score` is calculated in Python from the tool trace and is reproducible with a
    calculator; no number in this row originates inside the language model. `agent_confidence` is
    range-checked because the human gate compares it against a 0.70 threshold — a value on the
    wrong scale there would let a case escape review.
    """

    investigation_id: str
    alert_id: str
    agent_recommendation: Recommendation | None
    agent_confidence: float | None
    computed_risk_score: float | None
    detected_typology: Typology | None
    hypotheses: tuple[Mapping[str, Any], ...]
    narrative: str | None
    requires_human: bool
    llm_model: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    latency_ms: int
    tool_call_count: int
    status: InvestigationStatus
    started_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        ensure_aware(self.started_at, "started_at")
        ensure_aware(self.completed_at, "completed_at")
        # Mean investigation latency is reported from these two; a reversed pair would silently
        # deflate it.
        ensure_ordered(self.started_at, self.completed_at, "started_at", "completed_at")
        ensure_in_range(self.agent_confidence, "agent_confidence", 0.0, 1.0)
        ensure_in_range(self.computed_risk_score, "computed_risk_score", 0.0, 100.0)
        ensure_non_negative(self.cost_usd, "cost_usd")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation inside an investigation, with its input, output, and timing.

    Persisting these is what makes an investigation replayable: the verdict can be reconstructed
    from the trace without re-paying for tokens, which is the auditability requirement rather than
    a nicety.

    `tool_output` is `None` exactly when `is_error` is true, and the constructor enforces both
    halves of that. A success with no output is not a real case: every tool returns a structured
    result, and "no applicable policy found" is an *empty* result (`{"chunks": []}`), not a missing
    one — the distinction matters because an absent output cannot be replayed, while an empty one
    can. An error carrying an output would mean the trace shows a result the investigation never
    used.

    `call_id` is `None` before the row is inserted — the column is a `bigserial`, so the database
    assigns it.
    """

    call_id: int | None
    investigation_id: str
    sequence_no: int
    tool_name: str
    tool_input: Mapping[str, Any]
    tool_output: Mapping[str, Any] | None
    is_error: bool
    error_message: str | None
    duration_ms: int
    called_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.called_at, "called_at")
        where = f"tool call {self.sequence_no} ({self.tool_name})"
        if self.is_error:
            if not self.error_message:
                raise DomainValidationError(
                    f"{where} is flagged as an error but carries no error_message; a silent "
                    f"failure is the thing this table exists to prevent."
                )
            if self.tool_output is not None:
                raise DomainValidationError(
                    f"{where} is flagged as an error but carries a tool_output; the trace would "
                    f"show a result the investigation never used."
                )
        elif self.tool_output is None:
            raise DomainValidationError(
                f"{where} succeeded but recorded no tool_output, so the step cannot be replayed. "
                f"An empty result is a value (an empty list of chunks), not a missing one."
            )


@dataclass(frozen=True, slots=True)
class HumanDecision:
    """An analyst's adjudication — a separate table from `Investigation`, and separate on purpose.

    Model-risk governance forbids letting a model close an AML alert. Storing the agent's
    recommendation and the human's decision as different rows means the agent can never overwrite a
    decision, `agreed_with_agent` becomes a real trust metric, and the audit trail records *who*
    decided rather than only what the system output.

    `agreed_with_agent` is stored rather than derived because the agent's recommendation could in
    principle be revised later; the agreement recorded at decision time is the fact worth keeping.
    """

    decision_id: str
    investigation_id: str
    analyst_id: str
    final_decision: Recommendation
    agreed_with_agent: bool
    rationale: str
    decided_at: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.decided_at, "decided_at")
        if not self.rationale.strip():
            # An unexplained decision is not auditable, which defeats the purpose of the table.
            raise DomainValidationError("rationale must not be empty for a human decision.")


# --- data quality ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DqRun:
    """One execution of the data-quality gate over one batch.

    Written even when the gate fails — especially then. The batch is blocked and nothing reaches the
    curated zone, but the audit trail of *why* survives the failure. `finished_at` is `None` while
    the run is in flight.
    """

    run_id: str
    batch_id: str
    started_at: datetime
    finished_at: datetime | None
    status: DqRunStatus
    rows_in: int
    rows_loaded: int
    rows_rejected: int
    checks_total: int
    checks_failed: int
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        ensure_aware(self.started_at, "started_at")
        ensure_aware(self.finished_at, "finished_at")
        if self.checks_failed > self.checks_total:
            raise DomainValidationError(
                f"checks_failed ({self.checks_failed}) exceeds checks_total ({self.checks_total})."
            )


@dataclass(frozen=True, slots=True)
class DqCheckResult:
    """One check within a run.

    `severity` decides the batch's fate: a failed `BLOCKING` check stops the load, a failed
    `WARNING` is recorded and the batch proceeds. `expectation` and `observed` are free text
    because they land in a human-readable report — "0 duplicates" against "37 duplicate txn_id".

    `target_column` is `None` for table-level checks such as row-count tolerance. `result_id` is
    `None` until the `bigserial` column is assigned by the database.
    """

    result_id: int | None
    run_id: str
    check_name: str
    target_table: str
    target_column: str | None
    severity: DqSeverity
    status: DqCheckStatus
    expectation: str
    observed: str

    @property
    def blocks_batch(self) -> bool:
        """True when this single result is enough to stop the load."""
        return self.severity is DqSeverity.BLOCKING and self.status is DqCheckStatus.FAIL


# --- evaluation ground truth -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypologyLabel:
    """Generator ground truth — the evaluation boundary.

    **Nothing in the detection path may read this.** Not the rule engine, not the feature pipeline,
    not the agent. Only `pipeline/generate.py`, `tests/`, and `reporting/metrics.py` touch the
    table, and CI fails if the name appears elsewhere. That separation is the difference between
    "my detector works" and a defensible precision/recall figure.

    `is_hard_negative` marks a legitimate account deliberately generated to *look* like a typology —
    a small merchant with genuine high fan-in, a salary-disbursal account fanning out. Their
    false-positive rate is reported separately because it is the honest difficulty measure.
    """

    account_id: str
    is_mule: bool
    typology: Typology
    ring_id: str | None
    is_hard_negative: bool
    note: str

    def __post_init__(self) -> None:
        if self.is_mule and self.typology is Typology.NONE:
            raise DomainValidationError(
                f"{self.account_id} is labelled a mule but carries typology NONE; the metrics "
                f"report would count it in no per-typology bucket."
            )
        if self.is_mule and self.is_hard_negative:
            raise DomainValidationError(
                f"{self.account_id} is labelled both a mule and a hard negative; a hard negative "
                f"is by definition legitimate."
            )
