"""Computed results — the things this system works out rather than stores.

An entity in `entities.py` is a row: it has an identity, it was loaded from somewhere, and it will
be written back. A value object here has no identity and no table. It is the answer a component just
produced — a feature vector a store handed back, the rules that fired, a path the graph traversal
found, a policy chunk retrieval matched. Two value objects with equal fields are the same answer,
which is exactly what dataclass equality gives.

Why they are worth their own module rather than being `dict`s or Pydantic models:

* A `dict` returned across a port has no contract. Three adapters implement `FeatureStore`; a typed
  return is what makes swapping Redis for Aerospike a one-file change instead of a hunt through
  call sites.
* Pydantic belongs on the *edges* — HTTP bodies and LLM tool arguments, where untrusted input needs
  parsing. These objects are built by our own code from already-validated data, so paying Pydantic's
  validation cost on every one of them buys nothing. The domain stays importable with zero
  dependencies, which is what lets a unit test construct one with no setup at all.

Same rules as `entities.py`: `frozen=True, slots=True`, `Decimal` for money, `float` for
dimensionless statistics, timezone-aware timestamps only. The field guards are imported from
`entities.py` rather than duplicated.

This module lives in `domain` and imports nothing outside the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sentinel.domain.entities import (
    compute_pass_through_ratio,
    ensure_aware,
    ensure_in_range,
    ensure_non_negative,
    ensure_ordered,
)
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    CaseOutcome,
    CustomerSegment,
    FeatureWindow,
    KycLevel,
    MuleSignature,
    RiskRating,
    ScoreDecision,
    Severity,
    Typology,
)
from sentinel.domain.exceptions import DomainValidationError


def _ensure_non_negative_int(value: int, name: str) -> None:
    """Integer twin of `ensure_non_negative`, which is annotated for `Decimal` money fields.

    Counts, degrees, token tallies and month spans are `int`; passing one to the `Decimal` guard
    works at runtime but is a `mypy --strict` error, and silencing that would be the wrong fix.
    """
    if value < 0:
        raise DomainValidationError(f"{name} must not be negative, got {value}.")


# --- account context ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccountProfile:
    """Who owns this account and what "normal" looks like for them — the T1 tool's whole answer.

    A join across `customers` and `accounts`, not a row in either, which is why it is a value object
    rather than an entity. It exists because the same transaction pattern means different things on
    different accounts: 14 inbound payers in nine hours is a mule signature on a two-month
    `MIN_KYC` account and a Tuesday on a nine-year merchant. Without this, the risk scorer's
    profile component has no denominator.

    A `Mapping[str, Any]` would not do here for a specific reason: `declared_annual_income` is
    compared against measured throughput to produce a score contribution, and a `dict` carrying a
    `float` income would reintroduce binary floating point into the one comparison that decides
    whether a customer is flagged.

    `tenure_months` and `is_dormant` are supplied by the adapter rather than derived here, because
    both depend on "now" and a value object that silently reads the clock is not reproducible in a
    replay. `status` is carried alongside `is_dormant` because a `FROZEN` account is inactive for a
    completely different reason than a dormant one.
    """

    account_id: str
    customer_id: str
    account_type: AccountType
    status: AccountStatus
    tenure_months: int
    is_dormant: bool
    current_balance: Decimal
    kyc_level: KycLevel
    declared_annual_income: Decimal
    risk_rating: RiskRating
    segment: CustomerSegment
    is_pep: bool

    def __post_init__(self) -> None:
        # `current_balance` stays unguarded for the same reason as on `Account`: a CURRENT account
        # can legitimately sit overdrawn.
        _ensure_non_negative_int(self.tenure_months, "tenure_months")
        ensure_non_negative(self.declared_annual_income, "declared_annual_income")


# --- features ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """The feature set for one account over one window, as a `FeatureStore` returns it.

    The same numbers as an `AccountFeatures` row, with one difference that matters:
    `pass_through_ratio` is **computed here**, not stored. It is the mule signature — money in,
    money straight back out. Both this property and the stored `AccountFeatures` column route
    through `compute_pass_through_ratio`, which is the one definition of the feature: this returns
    it, and the row cross-checks its column against it at construction. That is what actually
    prevents the classic production bug where the batch pipeline and the serving path end up
    disagreeing about what a feature means.

    `as_of` is carried so a caller can compute snapshot staleness. Pre-computed features are always
    somewhat stale; hiding that is how real systems make silent wrong decisions, which is why the
    scoring API returns the age alongside the score.
    """

    account_id: str
    window: FeatureWindow
    as_of: datetime
    txn_count: int
    total_credit: Decimal
    total_debit: Decimal
    distinct_credit_counterparties: int
    distinct_debit_counterparties: int
    max_single_txn: Decimal
    avg_txn_amount: Decimal
    night_txn_ratio: float
    new_counterparty_ratio: float
    distinct_devices: int
    distinct_ips: int
    days_since_last_activity: int
    velocity_zscore: float

    def __post_init__(self) -> None:
        ensure_aware(self.as_of, "as_of")
        ensure_non_negative(self.total_credit, "total_credit")
        ensure_non_negative(self.total_debit, "total_debit")
        ensure_in_range(self.night_txn_ratio, "night_txn_ratio", 0.0, 1.0)
        ensure_in_range(self.new_counterparty_ratio, "new_counterparty_ratio", 0.0, 1.0)

    @property
    def pass_through_ratio(self) -> float:
        """Debits divided by credits over the window — see `compute_pass_through_ratio`, which is
        the single definition this and the stored column both use."""
        return compute_pass_through_ratio(self.total_credit, self.total_debit)

    def snapshot_age_ms(self, now: datetime) -> int:
        """How stale this vector is, in milliseconds. `now` is passed in rather than read from the
        clock so the value is reproducible in a test and in a replay."""
        ensure_aware(now, "now")
        return max(0, int((now - self.as_of).total_seconds() * 1000))


# --- rules and scoring -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleHit:
    """One rule that fired, with the evidence that fired it.

    Produced by the rule engine and turned into an `Alert` row by the caller — the engine itself
    stays free of identifiers and persistence. `evidence` carries the numbers behind the trigger, so
    the "why" is attached at the moment of detection rather than reconstructed later.
    """

    rule_code: str
    rule_name: str
    severity: Severity
    contribution: float
    detail: str
    window_start: datetime
    window_end: datetime
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        ensure_aware(self.window_start, "window_start")
        ensure_aware(self.window_end, "window_end")
        ensure_in_range(self.contribution, "contribution", 0.0, 100.0)
        if not self.detail.strip():
            raise DomainValidationError(
                f"{self.rule_code} produced no detail text; an unexplained contribution to a "
                f"customer-affecting score is not defensible."
            )


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """One named contribution to a score, with a human-readable reason.

    `code` is a rule code (`R004`) or a model identifier (`ML`) — the score is a sum of things that
    each have a name, and this is one of them.
    """

    code: str
    label: str
    contribution: float
    detail: str


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """The real-time scoring verdict: a decision, a number, and why.

    `components` is mandatory and non-empty, enforced in the constructor. A score without an
    explanation is unusable in a bank — every decision has to be defensible to a customer and to a
    regulator — so explainability is a construction invariant here rather than a convention that
    an adapter might forget.

    The band that maps `score` onto `decision` (ALLOW / REVIEW / BLOCK) is config, so this object
    deliberately does not check that the two agree: moving a threshold must not require changing
    the domain.

    `feature_snapshot_age_ms` and `latency_ms` are both measured, not claimed. The first tells the
    caller how stale the features behind the decision were; the second is the server's own
    measurement of how long it took.
    """

    account_id: str
    decision: ScoreDecision
    score: float
    components: tuple[ScoreComponent, ...]
    feature_snapshot_age_ms: int
    latency_ms: int
    model_version: str

    def __post_init__(self) -> None:
        ensure_in_range(self.score, "score", 0.0, 100.0)
        if not self.components:
            raise DomainValidationError(
                f"score {self.score} for {self.account_id} carries no components; a score with no "
                f"explanation cannot be defended to a customer or a regulator."
            )


# --- graph findings ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FundFlowPath:
    """One path money took through the account graph, as the Neo4j traversal returned it.

    `hop_accounts` is the ordered node list, so `hops` equals `len(hop_accounts) - 1`. Because a
    round-trip is just a path whose first and last account are the same, cycle findings need no
    separate type — `is_cycle` reads them off the same object.

    `bottleneck_amount` is the smallest edge value along the path and is the honest measure of how
    much money could actually have travelled the whole way; `path_value` (the sum of edge values)
    overstates it and must never be quoted as an amount at risk.
    """

    hop_accounts: tuple[str, ...]
    hops: int
    path_value: Decimal
    bottleneck_amount: Decimal
    flow_started: datetime
    flow_ended: datetime

    def __post_init__(self) -> None:
        ensure_aware(self.flow_started, "flow_started")
        ensure_aware(self.flow_ended, "flow_ended")
        ensure_ordered(self.flow_started, self.flow_ended, "flow_started", "flow_ended")
        ensure_non_negative(self.path_value, "path_value")
        ensure_non_negative(self.bottleneck_amount, "bottleneck_amount")
        if self.bottleneck_amount > self.path_value:
            # `path_value` sums the edges and `bottleneck_amount` is the smallest of them, so the
            # second can never exceed the first. If it does, the two came from different queries and
            # quoting either as an amount at risk would be wrong.
            raise DomainValidationError(
                f"bottleneck_amount {self.bottleneck_amount} exceeds the total path_value "
                f"{self.path_value}; the smallest edge cannot be larger than the sum of all edges."
            )
        if len(self.hop_accounts) < 2:
            raise DomainValidationError(
                f"a fund-flow path needs at least two accounts, got {self.hop_accounts!r}."
            )
        if self.hops != len(self.hop_accounts) - 1:
            raise DomainValidationError(
                f"hops={self.hops} disagrees with {len(self.hop_accounts)} accounts in the path; "
                f"one of the two came from a different query."
            )

    @property
    def is_cycle(self) -> bool:
        """True when the money returned to where it started — the `CIRCULAR` typology."""
        return self.hop_accounts[0] == self.hop_accounts[-1]


@dataclass(frozen=True, slots=True)
class RingFinding:
    """A cluster of accounts that behave as one operation.

    Rings are found two ways and this type carries both: accounts of *different* customers sharing
    a device fingerprint (`shared_device_id` set), and connected components computed in `networkx`
    over the transfer subgraph. `distinct_customers` is the load-bearing number — five accounts
    owned by one person is a portfolio, five owned by five people on one device is a ring.

    `ring_id` is `None` when the cluster was detected but never assigned a stable identifier; the
    generator's own `ring_id` values live in `typology_labels` and are not readable from here.
    """

    ring_id: str | None
    members: tuple[str, ...]
    distinct_customers: int
    shared_device_id: str | None
    total_internal_value: Decimal

    def __post_init__(self) -> None:
        ensure_non_negative(self.total_internal_value, "total_internal_value")
        if len(self.members) < 2:
            raise DomainValidationError(
                f"a ring needs at least two member accounts, got {self.members!r}."
            )
        if self.distinct_customers > len(self.members):
            raise DomainValidationError(
                f"distinct_customers ({self.distinct_customers}) exceeds the "
                f"{len(self.members)} member accounts; an account has exactly one owner."
            )

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_cross_customer(self) -> bool:
        """True when the ring spans more than one customer, which is what makes it suspicious
        rather than merely a customer holding several accounts."""
        return self.distinct_customers > 1


@dataclass(frozen=True, slots=True)
class FundFlowResult:
    """Everything the graph knows about one account's money movement, in one answer.

    This is deliberately a composite rather than four separate returns. `docs/06-AGENT-DESIGN.md`
    T3 asks for downstream paths, cycles, fan degrees with the derived signature, and ring
    membership as a single tool result, so the port returns them together and the adapter runs the
    Q1, Q2, Q3 and Q5 Cypher behind one call. Splitting them at the port would push four round trips
    to a store 250 ms away into the agent loop for evidence it always wants at once.

    `cycles` reuses `FundFlowPath` rather than introducing a `Cycle` type, because a round-trip *is*
    a path whose first and last account are the same — `FundFlowPath.is_cycle` already says so, and
    the constructor below enforces that every entry here satisfies it. `ring` reuses `RingFinding`
    for the same reason: it already carries the identifier, the members, and the
    `distinct_customers` count that decides whether the cluster is a ring or one person's portfolio.

    `ring` is `None` when the local component analysis found no cluster, which is different from
    finding a cluster of one — a single account is not a ring, and `RingFinding` rejects it.
    """

    account_id: str
    max_hops: int
    downstream_paths: tuple[FundFlowPath, ...]
    cycles: tuple[FundFlowPath, ...]
    fan_in_degree: int
    fan_out_degree: int
    total_in: Decimal
    total_out: Decimal
    signature: MuleSignature
    ring: RingFinding | None

    def __post_init__(self) -> None:
        _ensure_non_negative_int(self.fan_in_degree, "fan_in_degree")
        _ensure_non_negative_int(self.fan_out_degree, "fan_out_degree")
        ensure_non_negative(self.total_in, "total_in")
        ensure_non_negative(self.total_out, "total_out")
        if self.max_hops < 1:
            raise DomainValidationError(
                f"max_hops must be at least 1 to traverse anything, got {self.max_hops}."
            )
        for path in self.cycles:
            if not path.is_cycle:
                # Q2 matches `(a)-[:TRANSFERRED*2..5]->(a)`, so every path it returns closes on the
                # seed. One that does not means a Q1 result was filed under `cycles`, and the
                # narrative would then claim round-tripping that never happened.
                raise DomainValidationError(
                    f"{path.hop_accounts!r} is filed as a cycle but starts at "
                    f"{path.hop_accounts[0]} and ends at {path.hop_accounts[-1]}."
                )

    @property
    def pass_through_ratio(self) -> float:
        """Value out divided by value in, over the graph's one-hop neighbourhood.

        The same function as `FeatureVector.pass_through_ratio` on purpose. The graph and the
        feature store measure this over different scopes — edges versus a rolling window — so the
        two numbers legitimately differ, but never because the *definition* drifted between them.
        """
        return compute_pass_through_ratio(self.total_in, self.total_out)


@dataclass(frozen=True, slots=True)
class IdentifierLink:
    """One shared identifier and the accounts of *other* customers reachable through it.

    `identifier` is a `device_id` — the fingerprint two accounts were both used from.

    `distinct_other_customers` is the load-bearing number and the reason this is not a bare list of
    account ids: two accounts on one device is unremarkable when one person owns both, and is rule
    R007 when two people do.
    """

    identifier: str
    linked_accounts: tuple[str, ...]
    distinct_other_customers: int

    def __post_init__(self) -> None:
        if not self.linked_accounts:
            raise DomainValidationError(
                f"identifier {self.identifier} is reported as shared but links no other accounts."
            )
        if self.distinct_other_customers < 1:
            # The Q4 Cypher filters on `c1.customer_id <> c2.customer_id`, so a link that reaches
            # zero other customers means the inequality was lost from the query — which would turn
            # "this customer uses one phone for two of their own accounts" into a ring finding.
            raise DomainValidationError(
                f"identifier {self.identifier} links {len(self.linked_accounts)} accounts but no "
                f"other customer; a same-customer link is not an identity-linkage signal."
            )
        if self.distinct_other_customers > len(self.linked_accounts):
            # Same arithmetic as `RingFinding`: an account has exactly one owner, so the customers
            # reached cannot outnumber the accounts reached. This is the direction that matters,
            # because `shared_identifiers` orders by this field and R007 fires on it — an inflated
            # count puts a fabricated link at the top of an analyst's queue.
            raise DomainValidationError(
                f"identifier {self.identifier} reaches {self.distinct_other_customers} customers "
                f"through only {len(self.linked_accounts)} accounts; an account has one owner."
            )


@dataclass(frozen=True, slots=True)
class SharedIdentifierResult:
    """The T6 answer: devices this account shares with other customers' accounts.

    Devices only, and that is a narrowing of the tool contract in `docs/06-AGENT-DESIGN.md` T6,
    which also promises addresses. No address query exists in any of the five Cypher statements and
    no account-to-address relationship to hang one on, so a `shared_addresses` field would have been
    permanently empty — which reads to the agent as "checked, nothing found" rather than "never
    checked", and is worse than not promising it. Device sharing is the load-bearing signal.

    An empty tuple is the common and correct outcome — most accounts share nothing — and the agent
    must be able to report that rather than treating an absent answer as an unanswered question.
    """

    account_id: str
    shared_devices: tuple[IdentifierLink, ...]

    @property
    def has_cross_customer_link(self) -> bool:
        """True when any device reaches another customer — the identity-linkage trigger."""
        return bool(self.shared_devices)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One account in a projected subgraph, with the properties the console renders it by."""

    account_id: str
    is_seed: bool
    is_flagged: bool
    risk_rating: RiskRating


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One aggregated `:TRANSFERRED` relationship — a pair of accounts, not a single payment.

    `amount` is the pair's total across `txn_count` transactions, matching the graph projection
    described in `docs/03-GRAPH-SCHEMA.md` §3.1. Quoting it as the size of one payment would
    overstate it by up to `txn_count` times.
    """

    source: str
    target: str
    amount: Decimal
    txn_count: int
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        ensure_non_negative(self.amount, "amount")
        _ensure_non_negative_int(self.txn_count, "txn_count")


@dataclass(frozen=True, slots=True)
class Subgraph:
    """A k-hop neighbourhood as nodes and edges — what the console's network view draws.

    Separate from `FundFlowResult` because it answers a different question. `FundFlowResult` is
    evidence for a verdict; this is a picture. The Q5 Cypher returns only an edge list, so the
    adapter is responsible for materialising `nodes` with their properties.

    The constructor enforces exactly one seed and that every edge endpoint appears in `nodes`. Both
    are the failure modes of hand-built edge lists: a graph library given an edge to an unknown
    node silently invents that node, so the analyst sees a phantom account with no risk rating and
    no way to tell it apart from a real one.
    """

    seed_account_id: str
    max_hops: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        if self.max_hops < 1:
            raise DomainValidationError(
                f"max_hops must be at least 1 to traverse anything, got {self.max_hops}."
            )
        seeds = [node.account_id for node in self.nodes if node.is_seed]
        if seeds != [self.seed_account_id]:
            raise DomainValidationError(
                f"subgraph seeded on {self.seed_account_id} carries {seeds!r} as its seed nodes; "
                f"exactly one node must be flagged as the seed and it must be that account."
            )
        known = {node.account_id for node in self.nodes}
        dangling = sorted(
            {endpoint for edge in self.edges for endpoint in (edge.source, edge.target)} - known
        )
        if dangling:
            raise DomainValidationError(
                f"subgraph edges reference {dangling!r}, which are absent from its {len(known)} "
                f"nodes; a renderer would invent them as accounts with no properties."
            )


# --- data quality ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchProfile:
    """The statistical shape of one ingested batch — the thing a *later* batch is compared against.

    Four of the twelve data-quality checks are drift checks: they ask whether this batch looks like
    the recent past. That question has no answer unless the recent past was measured and kept, so a
    gate run that passes writes one of these. Without it the drift checks can only ever be skipped,
    and a gate that permanently skips a third of its checks is a gate in name only.

    A value object rather than an entity because it has no table in `docs/02-DATA-MODEL.md`: it is
    a summary this system works out, keyed by the batch it describes and never updated afterwards.

    `null_rates` and `channel_mix` are fractions in 0..1, not percentages — the drift thresholds are
    expressed in percentage *points*, and mixing the two units is exactly how a 15pp threshold turns
    into a 1500pp one. `amount_deciles` is `Decimal` because the boundaries are rupee amounts, and
    it is a quantile ladder rather than a mean: a mean amount cannot tell a batch that gained a few
    huge transfers from one whose whole distribution shifted, and only the second is drift.
    """

    batch_id: str
    computed_at: datetime
    window_end: date
    row_count: int
    null_rates: Mapping[str, float]
    channel_mix: Mapping[str, float]
    amount_deciles: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        ensure_aware(self.computed_at, "computed_at")
        _ensure_non_negative_int(self.row_count, "row_count")
        for column, rate in self.null_rates.items():
            ensure_in_range(rate, f"null_rates[{column}]", 0.0, 1.0)
        for channel, share in self.channel_mix.items():
            ensure_in_range(share, f"channel_mix[{channel}]", 0.0, 1.0)
        # A quantile ladder that runs backwards is not a ladder. It would make the KS-style
        # comparison in check 12 report a drift statistic computed against nonsense, which is worse
        # than reporting nothing at all.
        for lower, upper in zip(self.amount_deciles, self.amount_deciles[1:], strict=False):
            if upper < lower:
                raise DomainValidationError(
                    f"amount_deciles must be non-decreasing; {upper} follows {lower}."
                )


# --- retrieval ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    """A retrieved passage of AML policy, with enough provenance to cite it.

    `section_path` is what the narrative quotes, so a recommendation points at a written clause
    instead of inventing one. `typology` is `None` for a general clause that applies regardless of
    pattern — those are returned alongside typology-specific matches rather than filtered out.

    Retrieved text is **evidence, never instruction**. Nothing inside `text` may be treated as a
    directive to the agent; the prompt delimits it and says so explicitly.
    """

    doc_id: str
    doc_title: str
    section_path: str
    typology: Typology | None
    severity_hint: Severity | None
    chunk_index: int
    text: str
    similarity: float


@dataclass(frozen=True, slots=True)
class CaseMatch:
    """A past case that resembles the current one, returned **with its outcome**.

    The outcome is the entire point. Retrieval that returns three similar narratives is generic RAG;
    retrieval that returns three narratives plus what was decided and why lets the agent reason from
    precedent — "the two closest matches were escalated, the third was closed as a false positive
    because the account was a verified merchant; is that true here?" That is how an analyst works.

    Same injection boundary as `PolicyChunk`: `narrative` and `rationale` are evidence, not
    instructions.
    """

    case_id: str
    typology: Typology
    outcome: CaseOutcome
    decided_on: date
    rationale: str
    narrative: str
    similarity: float


# --- language model ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    """One model reply, with what it cost, as an `LLMClient` returns it.

    The usage figures are why this is not just a string. `Investigation.tokens_in`, `tokens_out` and
    `cost_usd` are persisted per investigation and summed into the `llm_spend_usd_30d` KPI, so the
    accounting has to come back with the text rather than be estimated afterwards from a token
    count nobody recorded. `cost_usd` is `Decimal` because it is money and it is summed.

    **`text` and `data` may both be `None`, and callers must handle it.** See §5 of
    `docs/12-PLATFORM-LIMITS.md` for the measured case: a DeepSeek reasoning model can spend the
    whole `max_tokens` budget on hidden reasoning and return no content at all. Representing that as
    an empty string would let a caller write an empty narrative and call the investigation complete;
    representing it as `None` with `finish_reason` attached makes the failure visible. `is_empty` is
    the check to branch on.

    `data` holds the parsed JSON object when the caller asked for structured output. It is a
    `Mapping[str, Any]` and that is honest rather than lazy: this is unvalidated model output, and
    the typed shape it is supposed to conform to is a Pydantic model that lives at the agent
    boundary — the domain must not import it, and pretending the port already validated it would
    move the trust boundary to the wrong layer.
    """

    model: str
    text: str | None
    data: Mapping[str, Any] | None
    finish_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    latency_ms: int

    def __post_init__(self) -> None:
        _ensure_non_negative_int(self.tokens_in, "tokens_in")
        _ensure_non_negative_int(self.tokens_out, "tokens_out")
        _ensure_non_negative_int(self.latency_ms, "latency_ms")
        ensure_non_negative(self.cost_usd, "cost_usd")

    @property
    def is_empty(self) -> bool:
        """True when the model returned neither prose nor structured data — see the class docstring:
        this is a real, measured outcome and not a defensive check against something impossible."""
        return self.text is None and self.data is None
