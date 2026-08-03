"""Every closed value set in the system, defined once.

These values do not stay inside Python. The same string is written to a Postgres `text` column,
embedded in a JSONB evidence blob, compared inside a rule, filtered on in a Qdrant payload, and
returned in an API response body. Defining them in one module is what keeps those five places
agreeing — a typo'd `"AWAITING_DECISON"` in an adapter becomes an import error rather than a row
that no query will ever match.

`StrEnum` (rather than `Enum`) is deliberate: a member *is* a `str`, so `json.dumps(member)`,
`psycopg`'s text adapter, and an f-string all accept it directly. No converter, no `.value` at
every call site, and no risk of writing `Channel.UPI` (the repr) into a database column.

Member *values* mirror the docs and the database exactly; member *names* are the Python identifier.
The two differ only where the value is not a legal identifier (`FeatureWindow.H24 == "24H"`).

This module lives in `domain` and imports nothing outside the standard library.
"""

from __future__ import annotations

from enum import StrEnum

# --- customer and account ----------------------------------------------------------------------


class KycLevel(StrEnum):
    """Know-Your-Customer completeness. `MIN_KYC` accounts carry materially higher mule risk —
    the same transaction pattern means different things on each, which is why the agent's first
    tool returns it."""

    MIN_KYC = "MIN_KYC"
    FULL_KYC = "FULL_KYC"


class RiskRating(StrEnum):
    """The bank's own standing risk classification of a customer or account."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CustomerSegment(StrEnum):
    """Commercial segment. Drives the expected amount distribution, so it is a denominator for
    "is this throughput plausible for this customer?" rather than a risk signal by itself."""

    MASS = "MASS"
    AFFLUENT = "AFFLUENT"
    HNI = "HNI"
    SME = "SME"


class AccountType(StrEnum):
    SAVINGS = "SAVINGS"
    CURRENT = "CURRENT"
    SALARY = "SALARY"


class AccountStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    FROZEN = "FROZEN"
    CLOSED = "CLOSED"


# --- transactions ------------------------------------------------------------------------------


class Channel(StrEnum):
    """Payment rail. Data-quality check 7 rejects any value outside this set, so adding a rail
    here is a deliberate act with a migration behind it."""

    UPI = "UPI"
    IMPS = "IMPS"
    NEFT = "NEFT"
    RTGS = "RTGS"
    CARD = "CARD"
    CASH = "CASH"
    ATM = "ATM"


class Direction(StrEnum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"


class TransactionStatus(StrEnum):
    """Only `SUCCESS` rows count towards features — failed and reversed money never moved, and
    counting them would inflate every velocity feature."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REVERSED = "REVERSED"


# --- features ----------------------------------------------------------------------------------


class FeatureWindow(StrEnum):
    """The four rolling windows features are computed over.

    Names are `H1`/`H24`/`D7`/`D30` because `1H` is not a legal Python identifier; the *values*
    are the strings stored in `account_features.window` and accepted by the agent's tool schema.
    """

    H1 = "1H"
    H24 = "24H"
    D7 = "7D"
    D30 = "30D"


# --- alerts and investigations -----------------------------------------------------------------


class Severity(StrEnum):
    """Alert severity. Ordered conceptually, but deliberately *not* comparable — a `<` between
    severities would invite arithmetic on a label. Rank explicitly where ranking is needed."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(StrEnum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    AWAITING_DECISION = "AWAITING_DECISION"
    CLOSED = "CLOSED"


class InvestigationStatus(StrEnum):
    """Lifecycle of one agent run. `AWAITING_DECISION` is the LangGraph interrupt state: the graph
    is checkpointed and will resume, it has not failed."""

    RUNNING = "RUNNING"
    AWAITING_DECISION = "AWAITING_DECISION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Recommendation(StrEnum):
    """What the agent proposes — and, separately, what a human decided.

    The same three values serve `investigations.agent_recommendation` and
    `human_decisions.final_decision`, which is precisely what makes `agreed_with_agent` a simple
    equality and the agreement rate a real metric. Note that a `Recommendation` is never an
    action: every `ESCALATE` goes to a human, unconditionally.
    """

    ESCALATE = "ESCALATE"
    MONITOR = "MONITOR"
    CLOSE = "CLOSE"


class Typology(StrEnum):
    """The six injected money-laundering patterns, plus `NONE` for a legitimate account.

    `NONE` exists because `typology_labels` stores a row for hard negatives too — an account that
    looks like a typology and is not one. Absence of a label and a label of "not a mule" are
    different facts, and the metrics need to tell them apart.
    """

    FAN_IN_OUT = "FAN_IN_OUT"
    LAYERING = "LAYERING"
    CIRCULAR = "CIRCULAR"
    STRUCTURING = "STRUCTURING"
    DORMANT_REACTIVATION = "DORMANT_REACTIVATION"
    DEVICE_RING = "DEVICE_RING"
    NONE = "NONE"


class CaseOutcome(StrEnum):
    """How a past case was closed. Retrieved alongside the narrative so the agent reasons from
    precedent ("the two closest matches were escalated") rather than from similarity alone."""

    ESCALATED = "ESCALATED"
    CLOSED_FALSE_POSITIVE = "CLOSED_FALSE_POSITIVE"
    MONITORING = "MONITORING"


# --- graph findings ----------------------------------------------------------------------------


class MuleSignature(StrEnum):
    """The verdict of the fan-in/fan-out shape test — one hop in each direction.

    Values are exactly the strings the Q3 Cypher in `docs/03-GRAPH-SCHEMA.md` returns, because the
    adapter passes them straight through. `NO_CLEAR_SIGNATURE` is the honest third answer: a shape
    that is neither collection nor dispersal, which is most accounts and must not be forced into
    one of the two suspicious buckets.

    The thresholds that decide between these live in the query and in config, never here — a
    signature is a label, and moving a threshold must not require changing the domain.
    """

    COLLECTION_MULE = "COLLECTION_MULE"
    DISPERSAL_MULE = "DISPERSAL_MULE"
    NO_CLEAR_SIGNATURE = "NO_CLEAR_SIGNATURE"


# --- real-time scoring -------------------------------------------------------------------------


class ScoreDecision(StrEnum):
    """Verdict returned by the real-time `/v1/score` path. The score bands that map onto these
    live in config, not here, so a threshold change is not a code change."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


# --- data quality ------------------------------------------------------------------------------


class DqSeverity(StrEnum):
    """Whether a failed check stops the batch.

    This is the entire data-quality gate in two values: a failed `BLOCKING` check means nothing is
    loaded and the workflow goes red; a failed `WARNING` is recorded and the batch proceeds. A late
    number is better than a wrong one reaching a regulatory report.
    """

    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


class DqCheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class DqRunStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
