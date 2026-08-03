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

from sentinel.domain.enums import FeatureWindow
from sentinel.domain.exceptions import ConfigurationError

# Component codes. Used as the `code` on every `ScoreComponent`, as keys in the persisted
# `risk_components` breakdown, and in the UI. Stable strings — changing one invalidates history.
PASS_THROUGH: Final = "PASS_THROUGH"
FAN_ASYMMETRY: Final = "FAN_ASYMMETRY"
NETWORK_STRUCTURE: Final = "NETWORK_STRUCTURE"
ACCOUNT_PROFILE: Final = "ACCOUNT_PROFILE"
VELOCITY_ANOMALY: Final = "VELOCITY_ANOMALY"
IDENTITY_LINKAGE: Final = "IDENTITY_LINKAGE"

COMPONENT_ORDER: Final = (
    PASS_THROUGH,
    FAN_ASYMMETRY,
    NETWORK_STRUCTURE,
    ACCOUNT_PROFILE,
    VELOCITY_ANOMALY,
    IDENTITY_LINKAGE,
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
    """

    pass_through: float = 25.0
    fan_asymmetry: float = 20.0
    network_structure: float = 20.0
    account_profile: float = 15.0
    velocity_anomaly: float = 10.0
    identity_linkage: float = 10.0

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

    def __post_init__(self) -> None:
        for name, floor, full in (
            ("pass_through", self.pass_through_floor, self.pass_through_full),
            ("fan_degree", self.fan_degree_floor, self.fan_degree_full),
            ("ring_size", self.ring_size_floor, self.ring_size_full),
            ("income_multiple", self.income_multiple_floor, self.income_multiple_full),
            ("velocity_zscore", self.velocity_zscore_floor, self.velocity_zscore_full),
            ("night_ratio", self.night_ratio_floor, self.night_ratio_full),
            ("shared_customers", self.shared_customers_floor, self.shared_customers_full),
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
SCORING_MODEL_VERSION: Final = "rules-v2"
