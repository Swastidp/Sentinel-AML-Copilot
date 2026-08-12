"""Risk weights, normalisation thresholds, score bands, and the human-gate criteria.

Every number the system uses to judge an account lives here, in one file, as named constants with
a stated reason. Nothing in `application/` or `agent/` may contain a numeric literal that decides
an outcome.

Three reasons this matters more than it looks:

* **Auditability.** A regulator's question is never "what did the model say", it is "why is this
  account 87 and not 62". That has to be answerable by pointing at a weight table and a tool
  trace, not by re-running a prompt.
* **Tuning without re-prompting.** An analyst who thinks pass-through is over-weighted changes a
  number here. If the weights lived in a prompt, tuning would mean re-validating an LLM.
* **Reproducibility.** The same evidence and the same weights always produce the same score.
  `docs/06-AGENT-DESIGN.md` §5 states this as the design's most important property.

These are *defaults*, deliberately not read from the environment. A weight change is a reviewable
code change with a test, not a production environment-variable edit — the gate thresholds in
`Settings` are operational knobs, but the scoring model is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from sentinel.domain.enums import FeatureWindow, Severity
from sentinel.domain.exceptions import ConfigurationError

# Component codes. Used as the `code` on every `ScoreComponent`, as keys in the persisted
# `risk_components` breakdown, and in the UI. Stable strings — changing one invalidates history.
PASS_THROUGH: Final = "PASS_THROUGH"
FAN_ASYMMETRY: Final = "FAN_ASYMMETRY"
NETWORK_STRUCTURE: Final = "NETWORK_STRUCTURE"
ACCOUNT_PROFILE: Final = "ACCOUNT_PROFILE"
VELOCITY_ANOMALY: Final = "VELOCITY_ANOMALY"
IDENTITY_LINKAGE: Final = "IDENTITY_LINKAGE"
RULE_HITS: Final = "RULE_HITS"

#: Display and persistence order for the breakdown. `RULE_HITS` is appended rather than inserted at
#: its weight rank, because this tuple is the column order of every stored `risk_components` row —
#: re-ordering it would shift the meaning of existing history for no benefit.
COMPONENT_ORDER: Final = (
    PASS_THROUGH,
    FAN_ASYMMETRY,
    NETWORK_STRUCTURE,
    ACCOUNT_PROFILE,
    VELOCITY_ANOMALY,
    IDENTITY_LINKAGE,
    RULE_HITS,
)

#: How many equal sub-signals make up `ACCOUNT_PROFILE`. Used to check that their shares cannot
#: add up past the component's own weight, which is the one arithmetic mistake a share can make.
PROFILE_SUB_SIGNALS: Final = 4

#: Calendar length of each feature window, in days. Needed to pro-rate a declared *annual* income
#: down to the window that was actually measured. Not a tunable threshold — `FeatureWindow.D7` *is*
#: seven days — but it lives here because it is arithmetic the scorer needs, and `application/`
#: holds no numbers of its own.
WINDOW_DAYS: Final[dict[FeatureWindow, Decimal]] = {
    FeatureWindow.H1: Decimal(1) / Decimal(24),
    FeatureWindow.H24: Decimal(1),
    FeatureWindow.D7: Decimal(7),
    FeatureWindow.D30: Decimal(30),
}

#: Denominator for the same pro-rating. Deliberately not leap-adjusted: a leap year moves this by
#: 0.3%, which is far below the resolution of any threshold in this file.
DAYS_PER_YEAR: Final = Decimal(365)


@dataclass(frozen=True, slots=True)
class RiskWeights:
    """Maximum points each component can contribute. Must total 100.

    The ordering reflects how much each signal distinguishes a mule from a busy ordinary account,
    which is the only question the score answers. Pass-through leads because it is the defining
    behaviour: a mule receives and forwards, it does not accumulate. Graph structure follows,
    because it is the evidence a single-account rule engine cannot produce at all. Profile,
    velocity, and identity are corroborating rather than decisive — a new MIN_KYC account is not
    suspicious by itself, and treating it as such is how false positives are manufactured.

    `rule_hits` shares the top rank with pass-through and is the reason the other six moved. The six
    flow-and-graph components read *shape*, and shape cannot see a named pattern: "4 to 9 debits in
    the Rs 45,000-49,900 band across 5 to 10 days" is R004's whole definition and no ramp over a
    degree or a ratio expresses it. Measured at Stage 2, STRUCTURING accounts reached 40 in 0% of
    cases and LAYERING in 2% — the rule fires, the score comes out near 8, and the case closes.

    **The other six are the v2 weights scaled by exactly 0.8** (25/20/20/15/10/10 becomes
    20/16/16/12/8/8), which leaves 20 for `rule_hits` and totals 100. A uniform rescale is the only
    re-weighting defensible without new measurement: it preserves every relative ordering the
    paragraph above argues for, and it maps every previously recorded score to exactly 0.8x its old
    value, so an auditor can convert a v2 figure to v3 with one multiplication. Redistributing
    unevenly would claim one component is now worth more relative to another, and there is no
    evidence for that claim.
    """

    pass_through: float = 20.0
    fan_asymmetry: float = 16.0
    network_structure: float = 16.0
    account_profile: float = 12.0
    velocity_anomaly: float = 8.0
    identity_linkage: float = 8.0
    rule_hits: float = 20.0

    def __post_init__(self) -> None:
        total = sum(self.as_mapping().values())
        # Exact equality on floats is wrong here; 1e-9 is far tighter than any deliberate change.
        if abs(total - 100.0) > 1e-9:
            raise ConfigurationError(
                f"Risk weights must total 100 so the score is on a 0-100 scale; got {total}."
            )
        if any(weight < 0 for weight in self.as_mapping().values()):
            raise ConfigurationError("Risk weights cannot be negative.")

    def as_mapping(self) -> dict[str, float]:
        """Component code to weight, in `COMPONENT_ORDER`."""
        return {
            PASS_THROUGH: self.pass_through,
            FAN_ASYMMETRY: self.fan_asymmetry,
            NETWORK_STRUCTURE: self.network_structure,
            ACCOUNT_PROFILE: self.account_profile,
            VELOCITY_ANOMALY: self.velocity_anomaly,
            IDENTITY_LINKAGE: self.identity_linkage,
            RULE_HITS: self.rule_hits,
        }


@dataclass(frozen=True, slots=True)
class ScoringThresholds:
    """Where each raw signal starts to count, and where it saturates.

    Every signal is normalised to 0..1 by a linear ramp between a `_floor` and a `_full` value,
    then multiplied by its weight. A ramp rather than a step because a step at any single value
    makes the score discontinuous — two accounts either side of it differ by a whole component,
    which is impossible to defend to an analyst looking at both.

    The floors are the important half: they encode what *ordinary* looks like. A salaried customer
    routinely forwards 70% of what arrives, so `pass_through_floor` sits above that.
    """

    # --- pass-through -------------------------------------------------------------------------
    # Below 0.60, forwarding most of an inbound salary is simply normal spending.
    pass_through_floor: float = 0.60
    pass_through_full: float = 0.95

    # --- fan degrees --------------------------------------------------------------------------
    # A small merchant legitimately reaches 8 distinct payers in a week; a collection mule reaches
    # dozens. The floor is deliberately at the top of the legitimate range.
    fan_degree_floor: float = 8.0
    fan_degree_full: float = 25.0
    # Weight applied when the graph found no clear collection/dispersal signature. Not zero: high
    # degree is still information, it is just much weaker without asymmetry.
    unsigned_fan_factor: float = 0.40

    # --- network structure --------------------------------------------------------------------
    # Shares within the network component; these three sum to 1.0.
    cycle_share: float = 0.50
    hop_depth_share: float = 0.25
    ring_size_share: float = 0.25
    # The ring ramp runs on `distinct_customers`, not on member count: five accounts owned by one
    # person is a portfolio. One customer is the floor because that is not a ring at all.
    ring_size_floor: float = 1.0
    ring_size_full: float = 5.0
    # Below this, a circular flow or an internal ring balance is household money movement, not
    # layering. Rs 1 lakh: a Rs 40,000 loop between siblings and a Rs 40,00,000 laundering loop are
    # the same graph shape and must not be the same score. The ramp starts at a tenth of this.
    material_amount_inr: Decimal = Decimal("100000")

    # --- account profile ----------------------------------------------------------------------
    # Four independent sub-signals, each contributing an equal share of the component.
    profile_sub_signal_share: float = 0.25
    new_account_months: float = 6.0
    # Throughput is compared against declared income *pro-rated to the feature window*, so these
    # are multiples of the window's share of annual income rather than of the annual figure. The
    # 365/7 factor keeps the defaults numerically identical to the old annual-denominator version
    # for the default 7-day window, while 1H, 24H and 30D now score the same behaviour the same way
    # instead of drifting by a quarter of the component purely on which window was requested.
    # What they mean: the floor is reached when window credits hit three times declared *annual*
    # income inside a week, and saturation at twelve times. A deliberately extreme bar — this
    # sub-signal is meant to fire only on throughput no declared income can explain.
    income_multiple_floor: float = 3.0 * 365.0 / 7.0
    income_multiple_full: float = 12.0 * 365.0 / 7.0

    # --- velocity -----------------------------------------------------------------------------
    # A z-score of 2 is the ordinary tail of a normal month, not an anomaly.
    velocity_zscore_floor: float = 2.0
    velocity_zscore_full: float = 6.0
    night_ratio_floor: float = 0.20
    night_ratio_full: float = 0.60

    # --- identity linkage ---------------------------------------------------------------------
    # One other customer on a shared device is a family phone. Three other adults sharing one
    # handset is still a plausible household — a four-adult family saturated this component at 3.0
    # — so the ring only starts at four.
    shared_customers_floor: float = 1.0
    shared_customers_full: float = 4.0

    # --- rule hits ------------------------------------------------------------------------------
    # The ramp runs over the *summed* `RuleHit.contribution` of every rule that fired on the
    # account. The floor is zero, not a positive number: the rule engine has already decided the
    # pattern is worth raising, and a floor above zero would silently discard a rule an analyst can
    # see in the alert list. `RuleHit.contribution` is domain-validated to 0..100, so a full value
    # of 100 means one maximal rule saturates the component and two medium rules at 50 each also
    # do — lower this to make the component fire harder on a single rule.
    rule_contribution_floor: float = 0.0
    rule_contribution_full: float = 100.0

    def __post_init__(self) -> None:
        for name, floor, full in (
            ("pass_through", self.pass_through_floor, self.pass_through_full),
            ("fan_degree", self.fan_degree_floor, self.fan_degree_full),
            ("ring_size", self.ring_size_floor, self.ring_size_full),
            ("income_multiple", self.income_multiple_floor, self.income_multiple_full),
            ("velocity_zscore", self.velocity_zscore_floor, self.velocity_zscore_full),
            ("night_ratio", self.night_ratio_floor, self.night_ratio_full),
            ("shared_customers", self.shared_customers_floor, self.shared_customers_full),
            ("rule_contribution", self.rule_contribution_floor, self.rule_contribution_full),
        ):
            # An inverted ramp constructs fine and then raises from inside `_ramp` — while scoring
            # a live account, on a request that had nothing wrong with it. Catching it here turns a
            # per-request failure into a startup failure, which is where a config error belongs.
            if full <= floor:
                raise ConfigurationError(
                    f"{name}_full must exceed {name}_floor so the ramp is not inverted; got "
                    f"floor={floor}, full={full}."
                )

        for name, fraction in (
            ("unsigned_fan_factor", self.unsigned_fan_factor),
            ("cycle_share", self.cycle_share),
            ("hop_depth_share", self.hop_depth_share),
            ("ring_size_share", self.ring_size_share),
            ("profile_sub_signal_share", self.profile_sub_signal_share),
        ):
            # Unbounded, these are the only way a component can exceed its own weight: an
            # `unsigned_fan_factor` of 5.0 makes FAN_ASYMMETRY worth 100 against a weight of 20,
            # and the score stops being the sum of six bounded contributions.
            if not 0.0 <= fraction <= 1.0:
                raise ConfigurationError(
                    f"{name} is a fraction of a component and must be in 0..1; got {fraction}."
                )

        network_total = self.cycle_share + self.hop_depth_share + self.ring_size_share
        if abs(network_total - 1.0) > 1e-9:
            raise ConfigurationError(
                f"cycle_share, hop_depth_share and ring_size_share must sum to 1.0 so the network "
                f"component cannot exceed its weight; got {network_total}."
            )
        if PROFILE_SUB_SIGNALS * self.profile_sub_signal_share > 1.0 + 1e-9:
            raise ConfigurationError(
                f"{PROFILE_SUB_SIGNALS} profile sub-signals at {self.profile_sub_signal_share} "
                f"each would exceed the ACCOUNT_PROFILE weight."
            )
        if self.material_amount_inr <= 0:
            # Zero is rejected as well as negative: it would make the materiality ramp run from 0
            # to 0, which `_ramp` refuses — again, at scoring time rather than at startup.
            raise ConfigurationError(
                f"material_amount_inr must be positive; got {self.material_amount_inr}."
            )


@dataclass(frozen=True, slots=True)
class ScoreBands:
    """Score-to-decision boundaries for the real-time `/v1/score` path.

    `block_at` must sit at or above `GateConfig.high_risk_score`: an account the fast path would
    block is an account an investigation must not close without a human. A *lower* gate threshold
    is fine and merely more conservative — it queues cases the fast path would have allowed.
    """

    review_at: float = 40.0
    block_at: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 < self.review_at < self.block_at <= 100.0:
            raise ConfigurationError(
                f"Score bands must satisfy 0 < review_at < block_at <= 100; "
                f"got review_at={self.review_at}, block_at={self.block_at}."
            )


@dataclass(frozen=True, slots=True)
class GateConfig:
    """When an investigation may not close itself.

    Defaults mirror `.env.example`; `Settings` overrides them at runtime, because these are
    operational thresholds a bank would tune per risk appetite, unlike the weights.
    """

    min_confidence: float = 0.70
    high_risk_score: float = 80.0
    amount_threshold_inr: Decimal = Decimal("1000000")  # Rs 10 lakh

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError(
                f"min_confidence is a probability and must be in 0..1; got {self.min_confidence}."
            )
        if not 0.0 <= self.high_risk_score <= 100.0:
            raise ConfigurationError(
                f"high_risk_score must be in 0..100; got {self.high_risk_score}."
            )
        if self.amount_threshold_inr < 0:
            raise ConfigurationError("amount_threshold_inr cannot be negative.")


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Everything the scorer needs, passed as one argument.

    Bundled so `assess_risk` takes config explicitly rather than reaching for a module-level
    default. A scorer that reads global state cannot be tested at two different weightings, and
    the eval harness needs exactly that.
    """

    weights: RiskWeights = field(default_factory=RiskWeights)
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)
    bands: ScoreBands = field(default_factory=ScoreBands)
    gate: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        # Not equality. The property that matters is one-directional: nothing the fast path blocks
        # may auto-close, so the gate must fire at or before the block band. A gate at 70 against a
        # block band of 80 satisfies that and is merely stricter. Demanding equality would also turn
        # `AGENT_HIGH_RISK_SCORE` — a documented, range-validated `.env` knob that
        # `GateConfig(high_risk_score=settings.agent_high_risk_score)` is meant to read — into a
        # startup crash the moment an operator sets it to anything but 80.
        if self.gate.high_risk_score > self.bands.block_at:
            raise ConfigurationError(
                f"gate.high_risk_score ({self.gate.high_risk_score}) must not exceed "
                f"bands.block_at ({self.bands.block_at}); otherwise the fast path can block a "
                f"payment while the investigation that explains it auto-closes."
            )


#: The model identifier stamped onto every `ScoreResult`. Bump it whenever a weight, threshold,
#: or formula changes, so a stored score can always be traced to the logic that produced it.
#:
#: v2: cycle and ring sub-signals scaled by amount; the ring ramp moved from member count to
#: `distinct_customers`; identity linkage takes the busiest device rather than the sum across
#: devices and saturates at four other customers; income compared against a window-pro-rated share
#: of declared income; pass-through scored on the drain when there are no credits at all. A v1
#: score and a v2 score of the same account are not comparable, which is what this string is for.
#:
#: v3: a seventh component, `RULE_HITS`, worth 20, reading the rule engine's own `RuleHit`
#: contributions; the other six rescaled by exactly 0.8. A v2 score converts to v3 by multiplying
#: by 0.8 *only* while no rule fired — with the engine landed at T6.1 the two are not comparable.
SCORING_MODEL_VERSION: Final = "rules-v3"


# --- detection: the seven rules -------------------------------------------------------------------


R001: Final = "R001"
R002: Final = "R002"
R003: Final = "R003"
R004: Final = "R004"
R005: Final = "R005"
R006: Final = "R006"
R007: Final = "R007"

#: The pseudo-rule code carried by an alert the Isolation Forest raised on its own, with no rule
#: behind it. Deliberately not an `R0xx` code: a regulator asking "which rule fired?" must get
#: "none — the model found this account unlike the rest of the book", not a rule number that does
#: not exist. `docs/00-OVERVIEW.md` §4.8 already uses `ML` as a component code for the same reason.
ML001: Final = "ML001"


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """The identity of one rule: what it is called, how serious it is, and what it is worth.

    Separated from `RuleThresholds` because the two answer different questions and change for
    different reasons. A threshold moves when the *data* shifts ("8 payers a day is now ordinary
    for this segment"); a contribution moves when the *judgement* shifts ("fan-out on its own tells
    us less than we thought"). Keeping them apart means a tuning change to one cannot silently be
    read as a change to the other in a diff.

    **`contribution` is the number that reaches the score.** `_score_rule_hits` in
    `application/risk_scorer.py` sums the contributions of every rule that fired, ramps the sum
    from `ScoringThresholds.rule_contribution_floor` to `..._full` (0 to 100 by default) and
    multiplies by `RiskWeights.rule_hits` (20). So a rule contributing `c` alone is worth
    `0.2 * c` points out of 100, and the seven values below were chosen against three constraints:

    * **No rule may dominate.** The largest is 45, worth 9.0 points — under half the component. One
      rule firing must never on its own carry an account across a decision band, because a single
      account-level rule is precisely the evidence `docs/00-OVERVIEW.md` §2 argues is structurally
      insufficient.
    * **Corroboration must pay.** Two strong rules together (R002 + R004 = 90) reach 18.0 of the 20
      available. That is the intended shape: rules are cheap alone and decisive in agreement.
    * **The rule most likely to be wrong is worth least.** R003 is the only single-condition rule
      that a documented legitimate cohort trips by design — a payroll account pays 20 to 28 people
      (`docs/13-KNOWN-FALSE-POSITIVES.md`) — so it contributes 20, worth 4.0 points. That is the
      deliberate ceiling on the payroll false positive.

    The ranking itself follows how much innocent behaviour shares the pattern. R004 and R002 lead
    because both are hard to reach by accident: transactions clustered just under a reporting
    threshold is avoidance by construction, and R002 requires a high payer count *and* a large
    total together. R006 sits mid-table despite being the defining mule behaviour, because it is
    the widest-firing rule on real data — a busy trader genuinely forwards almost everything.
    """

    code: str
    name: str
    severity: Severity
    contribution: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.contribution <= 100.0:
            # `RuleHit` range-checks this too, but it does so per account at detection time. A
            # mis-typed contribution should fail when the config is built, not on the first row.
            raise ConfigurationError(
                f"{self.code} contribution must be in 0..100 to satisfy `RuleHit`; got "
                f"{self.contribution}."
            )


#: The seven rules of `docs/00-OVERVIEW.md` §3.1, in code order. A tuple rather than a mapping so
#: the display order is fixed; `rules.py` builds the lookup it needs.
RULE_SPECS: Final[tuple[RuleSpec, ...]] = (
    RuleSpec(R001, "Velocity spike", Severity.MEDIUM, 25.0),
    RuleSpec(R002, "Fan-in concentration", Severity.HIGH, 45.0),
    RuleSpec(R003, "Fan-out dispersal", Severity.MEDIUM, 20.0),
    RuleSpec(R004, "Structuring", Severity.HIGH, 45.0),
    RuleSpec(R005, "Dormant reactivation", Severity.HIGH, 40.0),
    RuleSpec(R006, "Pass-through", Severity.HIGH, 30.0),
    RuleSpec(R007, "Device concentration", Severity.HIGH, 35.0),
)

#: The model-only alert carries no rule contribution at all. It is `0.0` rather than a small
#: positive number because the Isolation Forest is unsupervised: it says "unlike the rest of the
#: book", never "this specific pattern", and feeding an unnamed signal into a score whose whole
#: promise is that every point can be explained would break that promise. The anomaly score rides
#: on the alert row where an analyst can see it; it does not reach `RULE_HITS`.
ML_SPEC: Final = RuleSpec(ML001, "Isolation Forest outlier", Severity.LOW, 0.0)


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """Where each of the seven rules fires. Every number here is from `docs/00-OVERVIEW.md` §3.1.

    Nothing in `pipeline/rules.py` holds a numeric literal that decides an outcome — the same
    discipline this file already imposes on the scorer, for the same auditability reason.

    **Two adaptations to what the batch can actually measure, and both are narrowings rather than
    inventions.** The spec writes five of the seven rules over a 24-hour window. The batch feature
    set computes 7-day and 30-day windows only (`pipeline/features.WINDOW_DAYS`); 1-hour and
    24-hour are real-time counters served from Redis at Stage 10 and do not exist yet. So those
    rules read the **shortest available** window, 7D, which is a *looser* time bound than the spec:
    an account that fires the 24-hour form necessarily fires the 7-day form, so the adaptation
    cannot cause a miss, only an over-fire. R004's spec window is already 7 days and needs none.

    R005 is the exception and reads 30D, for a structural reason rather than a tuning one.
    `days_since_last_activity` is measured to the last activity *strictly before the window opens*
    (`pipeline/features.py`, trap 3). On a 7-day row, an account reactivated three weeks ago reads
    a gap of about 21 days, so no account that has transacted inside the last month can ever show
    180 — the rule would be unfirable by construction on the window that catches the reactivation.
    The 30-day row measures back past the reactivation to the dormancy itself, which is the fact
    R005 is about.
    """

    # --- R001 velocity spike --------------------------------------------------------------------
    #: "More than 5x this account's own median." Compared against the account's own baseline, never
    #: against a global count — the design choice `docs/00-OVERVIEW.md` §3.1 calls out as the reason
    #: a busy shop does not trip a velocity rule every day.
    velocity_multiple: float = 5.0
    #: "...with at least 10 transactions." Without it, two transactions against an expected 0.3 is
    #: a 6x multiple and means nothing at all.
    velocity_min_txn_count: int = 10

    # --- R002 fan-in concentration --------------------------------------------------------------
    fan_in_min_counterparties: int = 8
    #: Rs 2 lakh. The second half of the AND: fan-in alone is what a shopkeeper looks like.
    fan_in_min_credit_inr: Decimal = Decimal("200000")

    # --- R003 fan-out dispersal -------------------------------------------------------------------
    fan_out_min_counterparties: int = 8

    # --- R004 structuring -------------------------------------------------------------------------
    #: Transactions inside the just-under-reporting band. The band itself is
    #: `Settings.feature_structuring_low_inr`/`..._high_inr`, because the feature query counts them;
    #: only the count threshold is a rule decision.
    structuring_min_band_count: int = 3

    # --- R005 dormant reactivation ----------------------------------------------------------------
    dormant_min_idle_days: int = 180
    #: Rs 1 lakh moved after the silence. Credits plus debits, because a reactivated account that
    #: only pays out is the same event as one that only takes in.
    dormant_min_throughput_inr: Decimal = Decimal("100000")

    # --- R006 pass-through ----------------------------------------------------------------------
    pass_through_min_ratio: float = 0.85
    #: "...and closing balance below 10% of throughput." Independent of the ratio only because the
    #: balance is a *stock* and the ratio is a *flow* — see `AccountEvidence.closing_balance` for
    #: how the stock is obtained and why it is the weakest input in the engine on this data.
    pass_through_max_balance_share: float = 0.10

    # --- R007 device concentration ----------------------------------------------------------------
    #: "Shares a device with 4 or more accounts belonging to other customers." Counted over the same
    #: 7-day window the other rules read, not over the whole ledger: an alert saying "these accounts
    #: were used from one handset this week" is evidence, and one saying "at some point in the last
    #: quarter" is not. Measured on the real batch, the 90-day reading fires on 45% of the book and
    #: the 7-day reading on 1.4%.
    device_min_other_accounts: int = 4

    def __post_init__(self) -> None:
        if self.velocity_multiple <= 1.0:
            # At or below 1.0 the rule fires on any account transacting at its own normal rate,
            # which is every active account in the book.
            raise ConfigurationError(
                f"velocity_multiple must exceed 1.0 to mean 'above this account's own baseline'; "
                f"got {self.velocity_multiple}."
            )
        if self.pass_through_min_ratio <= 0.0:
            raise ConfigurationError(
                f"pass_through_min_ratio must be positive; got {self.pass_through_min_ratio}."
            )
        if not 0.0 <= self.pass_through_max_balance_share <= 1.0:
            raise ConfigurationError(
                f"pass_through_max_balance_share is a fraction of throughput and must be in 0..1; "
                f"got {self.pass_through_max_balance_share}."
            )
        for name, count in (
            ("velocity_min_txn_count", self.velocity_min_txn_count),
            ("fan_in_min_counterparties", self.fan_in_min_counterparties),
            ("fan_out_min_counterparties", self.fan_out_min_counterparties),
            ("structuring_min_band_count", self.structuring_min_band_count),
            ("dormant_min_idle_days", self.dormant_min_idle_days),
            ("device_min_other_accounts", self.device_min_other_accounts),
        ):
            if count < 1:
                # Zero would make the condition vacuously true for every account, which reads in
                # the alert list as a rule that fired rather than as one that was switched off.
                raise ConfigurationError(f"{name} must be at least 1; got {count}.")
        for name, amount in (
            ("fan_in_min_credit_inr", self.fan_in_min_credit_inr),
            ("dormant_min_throughput_inr", self.dormant_min_throughput_inr),
        ):
            if amount <= 0:
                raise ConfigurationError(f"{name} must be positive; got {amount}.")


# --- detection: the anomaly model -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IsolationForestConfig:
    """How the unsupervised outlier model is fitted. Never touches `typology_labels`.

    **`contamination` is a stated prior, not a fitted value, and the distinction is the whole
    point.** It tells scikit-learn what share of the population to place on the outlier side of the
    decision boundary, and it is the one parameter a careless implementation would set by peeking
    at the answer sheet. The 0.027 below is taken from `docs/05-DATA-FLOW.md` §2 — the *generation
    plan*, which specifies 167 mule accounts in 6,200. It is not read from
    `data/raw/typology_labels/`, and nothing in the detection path can read that file. In a bank the
    equivalent prior is the historical confirmed-mule rate from closed investigations, which is
    exactly as much of a guess and exactly as defensible.

    Setting it explicitly rather than leaving `'auto'` also fixes the meaning of the mapped score:
    `anomaly.to_anomaly_score` maps `decision_function` so that 50 is the boundary, so "anomaly
    score at or above 50" and "in the most anomalous 2.7% of the book" are the same statement.

    `random_state` matches `Settings.generator_seed`. Isolation Forest picks random split features
    and random split points, so an unseeded fit gives a different model — and therefore different
    alerts — on every run. A detection system whose output changes when nothing changed cannot be
    audited.
    """

    n_estimators: int = 200
    contamination: float = 0.027
    #: `'auto'` in scikit-learn means `min(256, n_samples)`, which is the value the original paper
    #: recommends: small subsamples are what make an isolation tree isolate outliers early instead
    #: of memorising the population. Pinned to the number rather than the string so the fit does not
    #: change meaning if scikit-learn changes what `'auto'` resolves to.
    max_samples: int = 256
    random_state: int = 20260803
    #: Every tree sees every feature. Feature subsampling helps in very high dimension; there are 18
    #: features here, and dropping some of them from a tree only blunts it.
    max_features: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination <= 0.5:
            # scikit-learn accepts 'auto' or (0, 0.5]; a config that fails its own validator here
            # gives a clearer message than the one the estimator raises during `fit`.
            raise ConfigurationError(
                f"contamination is the expected outlier share and must be in (0, 0.5]; got "
                f"{self.contamination}."
            )
        if self.n_estimators < 1 or self.max_samples < 1:
            raise ConfigurationError(
                f"n_estimators and max_samples must both be at least 1; got "
                f"{self.n_estimators} and {self.max_samples}."
            )
        if not 0.0 < self.max_features <= 1.0:
            raise ConfigurationError(
                f"max_features is a fraction of the feature set and must be in (0, 1]; got "
                f"{self.max_features}."
            )


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Everything Stage 6 needs, in one argument — the `ScoringConfig` of the detection path.

    Same reason as `ScoringConfig`: a detector that reads module-level globals cannot be run at two
    settings in one process, and the Stage 7 metrics harness needs exactly that to report how the
    numbers move when a threshold does.
    """

    specs: tuple[RuleSpec, ...] = RULE_SPECS
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)

    def __post_init__(self) -> None:
        codes = [spec.code for spec in self.specs]
        if len(set(codes)) != len(codes):
            # `alert_id` is derived from `(account_id, rule_code, window_end)`, so a duplicated code
            # would make two different rules upsert onto one alert row and silently lose one.
            raise ConfigurationError(f"rule codes must be unique; got {codes!r}.")

    def spec(self, code: str) -> RuleSpec:
        """The spec for one rule code, or a `ConfigurationError` naming what is configured."""
        for candidate in self.specs:
            if candidate.code == code:
                return candidate
        raise ConfigurationError(
            f"no rule spec configured for {code!r}; configured codes are "
            f"{[s.code for s in self.specs]!r}."
        )


#: Bumped whenever a rule's definition, threshold or contribution changes, so a stored alert can be
#: traced to the logic that raised it. Same contract as `SCORING_MODEL_VERSION` and `GATE_VERSION`.
RULE_ENGINE_VERSION: Final = "rules-r001-r007-v1"
