"""Deterministic risk scoring and the human-in-the-loop gate.

**This module is the answer to "where do you not trust the model?"** Every number that decides an
outcome is computed here, in ordinary Python, from evidence the tools gathered. The LLM plans which
tools to call, forms hypotheses, and writes the narrative. It never produces a score, and it never
decides whether a human must look at a case.

Three properties follow, and they are the point:

* **Reproducible.** Same evidence plus same config always gives the same score. An auditor can
  recompute any stored figure from the tool trace with a calculator.
* **Explainable by construction.** The score is not a number with a story attached afterwards; it
  *is* the sum of six named contributions, each carrying the raw signal that produced it. The
  console breakdown and the narrative's citations read the same objects.
* **Tunable without re-validating a prompt.** Weights and thresholds are data in
  `config/weights.py`.

Missing evidence contributes **zero and says so**. A tool that was never called is recorded as
`not gathered`, never silently treated as a clean result — the difference between "we looked and
found nothing" and "we did not look" is the whole point of an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sentinel.config.weights import (
    ACCOUNT_PROFILE,
    DAYS_PER_YEAR,
    FAN_ASYMMETRY,
    IDENTITY_LINKAGE,
    NETWORK_STRUCTURE,
    PASS_THROUGH,
    VELOCITY_ANOMALY,
    WINDOW_DAYS,
    ScoringConfig,
    ScoringThresholds,
)
from sentinel.domain.enums import KycLevel, MuleSignature, Recommendation, ScoreDecision
from sentinel.domain.value_objects import (
    AccountProfile,
    FeatureVector,
    FundFlowResult,
    ScoreComponent,
    SharedIdentifierResult,
)

_NOT_GATHERED = "not gathered"


# --- inputs and outputs -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskEvidence:
    """Whatever the investigation actually managed to gather.

    Every field is optional because the agent decides which tools to call and any call can fail.
    Bundling them into one object rather than five parameters means adding a seventh signal later
    changes one type instead of every call site — and it makes "what did we know when we scored
    this?" a single serialisable thing.
    """

    account_id: str
    features: FeatureVector | None = None
    profile: AccountProfile | None = None
    flow: FundFlowResult | None = None
    identifiers: SharedIdentifierResult | None = None

    @property
    def gathered_count(self) -> int:
        """How many of the four evidence sources are present.

        A score built on one source is not comparable to one built on four, so callers surface this
        rather than presenting every score as equally well-founded.
        """
        return sum(1 for item in (self.features, self.profile, self.flow, self.identifiers) if item)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """The scorer's whole output: a score, a decision, and the arithmetic behind both.

    Deliberately *not* a `ScoreResult` — that type also carries `latency_ms`,
    `feature_snapshot_age_ms`, and `model_version`, which are properties of a serving request, not
    of the risk judgement. Keeping them apart means the agent's `assess` node and the real-time
    endpoint share one scorer without the agent inventing request-shaped values it has no business
    knowing.
    """

    account_id: str
    score: float
    decision: ScoreDecision
    components: tuple[ScoreComponent, ...]
    evidence_sources: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"score must be in 0..100; got {self.score}")
        if len(self.components) != 6:
            raise ValueError(
                f"expected all six components (missing evidence scores zero, it is not omitted); "
                f"got {len(self.components)}"
            )

    @property
    def reasons(self) -> tuple[str, ...]:
        """Human-readable details for components that actually contributed.

        `/v1/score` guarantees a non-empty `reasons` array, so a caller must fall back to a
        no-signal message when this is empty rather than returning an empty list.
        """
        return tuple(c.detail for c in self.components if c.contribution > 0.0)


# --- normalisation ------------------------------------------------------------------------------


def _ramp(value: float, floor: float, full: float) -> float:
    """Linear 0..1 ramp: 0 at or below `floor`, 1 at or above `full`.

    A ramp rather than a threshold because a step function makes the score discontinuous — two
    accounts either side of a single value would differ by a whole component, which cannot be
    defended to an analyst looking at both.
    """
    if full <= floor:
        raise ValueError(f"ramp needs full > floor; got floor={floor}, full={full}")
    if value <= floor:
        return 0.0
    if value >= full:
        return 1.0
    return (value - floor) / (full - floor)


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _material_fraction(amount: Decimal, th: ScoringThresholds) -> float:
    """How much of a money-scaled signal counts, ramped by the size of the money involved.

    A graph shape on its own says nothing about scale, and scale is what separates layering from
    housekeeping: a Rs 40,000 loop between siblings and a Rs 40,00,000 laundering loop are the same
    cycle to a traversal. Zero below a tenth of `material_amount_inr`, full at it.

    `Decimal` is converted to `float` only here, at the ramp boundary. No money arithmetic above
    this line touches binary floating point.
    """
    return _ramp(float(amount), float(th.material_amount_inr / 10), float(th.material_amount_inr))


# --- the six components -------------------------------------------------------------------------


def _score_pass_through(
    features: FeatureVector | None, weight: float, th: ScoringThresholds
) -> ScoreComponent:
    """Money in, money straight back out — the defining mule behaviour.

    The *flow* ratio and the *stock* balance are two different measurements, and only the flow is
    scored here. Retention within the window is `1 - pass_through_ratio` by construction, so adding
    a second sub-signal on it would count one measurement twice and quietly make this component
    worth 50 rather than 25. What that leaves out is the stock: a legitimate trader who holds a
    large balance throughout the window gets no exculpating credit for it, because a *balance delta*
    over the window is the figure that would supply it and `FeatureVector` carries no such field —
    only `AccountProfile.current_balance`, a single point-in-time reading that cannot distinguish
    "held it all week" from "received it this morning". A known limitation, not a redundancy.

    One case the ratio genuinely cannot see is handled below: an account with no credits at all in
    the window that nevertheless empties itself.
    """
    if features is None:
        return ScoreComponent(PASS_THROUGH, "Pass-through behaviour", 0.0, _NOT_GATHERED)

    if features.total_credit == 0 and features.total_debit > 0:
        # `compute_pass_through_ratio` is 0.0 when there were no credits, so an account holding
        # Rs 50 lakh that drains to zero inside the window scores *nothing* on the component whose
        # own docstring calls itself the defining mule behaviour. That is the DORMANT_REACTIVATION
        # shape in `docs/05-DATA-FLOW.md` §3, where the outflow is the only signal there is. Score
        # the drain instead of the ratio — instead of, never in addition to, or it double-counts.
        normalised = _material_fraction(features.total_debit, th)
        detail = (
            f"INR {features.total_debit:,.0f} left the account in the {features.window} window "
            f"with no matching inflow ({features.txn_count} transactions)"
        )
        return ScoreComponent(PASS_THROUGH, "Pass-through behaviour", normalised * weight, detail)

    ratio = features.pass_through_ratio
    normalised = _ramp(ratio, th.pass_through_floor, th.pass_through_full)
    detail = (
        f"Forwarded {_pct(min(ratio, 1.0))} of credits in the {features.window} window "
        f"({features.txn_count} transactions), retaining {_pct(max(0.0, 1.0 - ratio))}"
    )
    return ScoreComponent(PASS_THROUGH, "Pass-through behaviour", normalised * weight, detail)


def _score_fan_asymmetry(
    flow: FundFlowResult | None, weight: float, th: ScoringThresholds
) -> ScoreComponent:
    """Many payers in and few payees out (collection), or the reverse (dispersal).

    Degree alone is weak — a small merchant has high fan-in. It is the *asymmetry* that separates a
    mule from a business, so an unsigned high degree is discounted rather than ignored.
    """
    if flow is None:
        return ScoreComponent(FAN_ASYMMETRY, "Fan-in / fan-out asymmetry", 0.0, _NOT_GATHERED)
    peak = float(max(flow.fan_in_degree, flow.fan_out_degree))
    degree = _ramp(peak, th.fan_degree_floor, th.fan_degree_full)
    signed = flow.signature is not MuleSignature.NO_CLEAR_SIGNATURE
    factor = 1.0 if signed else th.unsigned_fan_factor
    detail = (
        f"Fan-in {flow.fan_in_degree}, fan-out {flow.fan_out_degree}, "
        f"signature {flow.signature}"
        + ("" if signed else " (degree counted at reduced weight without a clear signature)")
    )
    return ScoreComponent(
        FAN_ASYMMETRY, "Fan-in / fan-out asymmetry", degree * factor * weight, detail
    )


def _score_network(
    flow: FundFlowResult | None, weight: float, th: ScoringThresholds
) -> ScoreComponent:
    """Structure only a graph traversal can see: closed loops, chain depth, ring membership.

    This is the component no account-level rule engine can produce, which makes it the one worth
    demonstrating. A cycle carries half the weight because funds returning to origin has almost no
    innocent explanation in retail banking — *at a material amount*. Structure alone is not enough:
    a graph traversal cannot tell a laundering loop from a family splitting a restaurant bill, so
    the cycle and ring sub-signals are both scaled by how much money actually moved. Without that,
    the four hard negatives in `docs/05-DATA-FLOW.md` §3 score the same as the typologies they
    imitate, which is the whole false-positive problem restated.
    """
    if flow is None:
        return ScoreComponent(NETWORK_STRUCTURE, "Network structure", 0.0, _NOT_GATHERED)

    parts: list[str] = []
    normalised = 0.0

    if flow.cycles:
        # The largest loop, not the sum: cycles through one account overlap, so summing them
        # double-counts the same rupees for the same reason `amount_at_risk` refuses to sum paths.
        largest = max(loop.path_value for loop in flow.cycles)
        normalised += th.cycle_share * _material_fraction(largest, th)
        parts.append(f"{len(flow.cycles)} circular flow(s) detected, largest INR {largest:,.0f}")

    if flow.downstream_paths:
        deepest = max(path.hops for path in flow.downstream_paths)
        # Depth is scored against the depth actually searched, so a 2-hop projection is not
        # penalised for failing to find 4-hop chains it never looked for.
        depth_fraction = min(1.0, deepest / max(flow.max_hops, 1))
        normalised += th.hop_depth_share * depth_fraction
        parts.append(f"onward chain {deepest} hop(s) deep of {flow.max_hops} searched")

    if flow.ring is not None and not flow.ring.is_cross_customer:
        # `distinct_customers` is the load-bearing number, and scoring on `size` ignored it
        # entirely: five accounts owned by one person is a portfolio, five owned by five people on
        # one device is a ring. A single customer's own accounts contribute nothing at all.
        parts.append(
            f"cluster of {flow.ring.size} accounts is a single customer's own accounts, not a ring"
        )
    elif flow.ring is not None:
        # Ramped on the number of *people*, because that is the signal — a ring of eight accounts
        # across two customers is two people, not eight — and scaled by the money moving inside it.
        people = _ramp(float(flow.ring.distinct_customers), th.ring_size_floor, th.ring_size_full)
        normalised += (
            th.ring_size_share * people * _material_fraction(flow.ring.total_internal_value, th)
        )
        parts.append(
            f"ring of {flow.ring.size} accounts across {flow.ring.distinct_customers} customers, "
            f"INR {flow.ring.total_internal_value:,.0f} moving internally"
        )

    detail = "; ".join(parts) if parts else "no cycles, onward chains, or ring membership found"
    return ScoreComponent(
        NETWORK_STRUCTURE, "Network structure", min(1.0, normalised) * weight, detail
    )


def _score_profile(
    profile: AccountProfile | None,
    features: FeatureVector | None,
    weight: float,
    th: ScoringThresholds,
) -> ScoreComponent:
    """Corroborating context: tenure, KYC strength, dormancy, income-versus-throughput.

    Four independent sub-signals, each worth a quarter. None is suspicious alone — a new MIN_KYC
    account is the ordinary case for a first-time customer, and scoring it heavily is exactly how
    false positives get manufactured against the least-established customers.
    """
    if profile is None:
        return ScoreComponent(ACCOUNT_PROFILE, "Account profile risk", 0.0, _NOT_GATHERED)

    share = th.profile_sub_signal_share
    normalised = 0.0
    parts: list[str] = []

    if profile.tenure_months < th.new_account_months:
        normalised += share
        parts.append(f"opened {profile.tenure_months} month(s) ago")

    if profile.kyc_level is KycLevel.MIN_KYC:
        normalised += share
        parts.append("minimum KYC only")

    if profile.is_dormant:
        normalised += share
        parts.append("reactivated after dormancy")

    # Throughput far above declared income is the strongest profile signal, because it is the one
    # a mule cannot avoid: the money has to move through the account to be laundered. The income is
    # pro-rated to the window that was actually measured, so the same account behaving the same way
    # scores the same whether the agent asked for 1 hour or 30 days. Comparing a week of credits
    # against a *year* of income made the sub-signal an artefact of the window chosen — worth up to
    # a quarter of the component — and "we scored you higher because we looked at 30 days" is not
    # something an analyst can defend.
    if features is not None and profile.declared_annual_income > 0:
        window_income = (
            profile.declared_annual_income * WINDOW_DAYS[features.window] / DAYS_PER_YEAR
        )
        multiple = float(features.total_credit / window_income)
        income_fraction = _ramp(multiple, th.income_multiple_floor, th.income_multiple_full)
        normalised += share * income_fraction
        if income_fraction > 0:
            parts.append(
                f"{features.window} credits are {multiple:.1f}x the window's share of declared "
                f"annual income"
            )

    detail = "; ".join(parts) if parts else "established account, full KYC, throughput in line"
    return ScoreComponent(
        ACCOUNT_PROFILE, "Account profile risk", min(1.0, normalised) * weight, detail
    )


def _score_velocity(
    features: FeatureVector | None, weight: float, th: ScoringThresholds
) -> ScoreComponent:
    """Sudden throughput against the account's own history, plus night-hour concentration.

    Averaged rather than summed: they are correlated (a burst at 3am scores on both), and summing
    correlated signals inflates the component beyond its weight.
    """
    if features is None:
        return ScoreComponent(VELOCITY_ANOMALY, "Velocity anomaly", 0.0, _NOT_GATHERED)
    z = _ramp(features.velocity_zscore, th.velocity_zscore_floor, th.velocity_zscore_full)
    night = _ramp(features.night_txn_ratio, th.night_ratio_floor, th.night_ratio_full)
    normalised = (z + night) / 2.0
    detail = (
        f"velocity z-score {features.velocity_zscore:.1f}, "
        f"{_pct(features.night_txn_ratio)} of transactions at night"
    )
    return ScoreComponent(VELOCITY_ANOMALY, "Velocity anomaly", normalised * weight, detail)


def _score_identity(
    identifiers: SharedIdentifierResult | None, weight: float, th: ScoringThresholds
) -> ScoreComponent:
    """Devices shared with accounts belonging to *other* customers.

    One other customer on a shared handset is a family phone, and so are three — four adults in one
    household share one. Four or more is a ring being operated from one device, which is why the
    floor sits at one rather than zero and the full value at four.
    """
    if identifiers is None:
        return ScoreComponent(IDENTITY_LINKAGE, "Identity linkage", 0.0, _NOT_GATHERED)

    # `max`, not `sum`. A household with a shared phone *and* a shared tablet, both reaching the
    # same two relatives, summed to "4 other customer(s)" — a number that is simply false, printed
    # into an audit string by a module whose docstring promises every figure is defensible.
    #
    # This is the conservative approximation available today, not the right answer. The right answer
    # is the union of customers across devices, and it is not computable here: `IdentifierLink`
    # carries account ids plus a per-link customer *count*, never customer ids, so two links cannot
    # be de-duplicated against each other. Fixing it properly means adding customer ids to that
    # value object — a domain change, deferred until T8.3 can supply them.
    # ponytail: max() under-counts a real ring split across two devices; upgrade to a union when
    # `IdentifierLink` carries customer ids.
    others = max((link.distinct_other_customers for link in identifiers.shared_devices), default=0)
    normalised = _ramp(float(others), th.shared_customers_floor, th.shared_customers_full)
    # "at least" rather than a flat count, because `others` is the busiest single device and the
    # customers behind two devices may be the same people. Understated on purpose, and said so.
    detail = (
        f"{len(identifiers.shared_devices)} shared device(s) reaching at least {others} other "
        f"customer(s)"
        if identifiers.shared_devices
        else "no devices shared with other customers"
    )
    return ScoreComponent(IDENTITY_LINKAGE, "Identity linkage", normalised * weight, detail)


# --- public API ---------------------------------------------------------------------------------


def assess_risk(evidence: RiskEvidence, config: ScoringConfig | None = None) -> RiskAssessment:
    """Score an account from gathered evidence. Pure: no I/O, no clock, no randomness.

    All six components are always present in the result. A component with no evidence contributes
    zero and reports `not gathered`, so the breakdown distinguishes "we checked and it was clean"
    from "we never checked" — collapsing those two would let an unlucky investigation look like a
    thorough one.
    """
    cfg = config or ScoringConfig()
    weights = cfg.weights
    th = cfg.thresholds

    components = (
        _score_pass_through(evidence.features, weights.pass_through, th),
        _score_fan_asymmetry(evidence.flow, weights.fan_asymmetry, th),
        _score_network(evidence.flow, weights.network_structure, th),
        _score_profile(evidence.profile, evidence.features, weights.account_profile, th),
        _score_velocity(evidence.features, weights.velocity_anomaly, th),
        _score_identity(evidence.identifiers, weights.identity_linkage, th),
    )

    # Clamped because float addition of six weighted terms can land a hair above 100.
    score = min(100.0, sum(c.contribution for c in components))
    return RiskAssessment(
        account_id=evidence.account_id,
        score=score,
        decision=decide(score, cfg),
        components=components,
        evidence_sources=evidence.gathered_count,
    )


def decide(score: float, config: ScoringConfig | None = None) -> ScoreDecision:
    """Map a score onto the real-time allow/review/block decision."""
    bands = (config or ScoringConfig()).bands
    if score >= bands.block_at:
        return ScoreDecision.BLOCK
    if score >= bands.review_at:
        return ScoreDecision.REVIEW
    return ScoreDecision.ALLOW


def amount_at_risk(evidence: RiskEvidence) -> Decimal:
    """Money that actually left the account in the window, for the gate's amount criterion.

    The larger of `flow.total_out` and `features.total_debit`, whichever of the two is present.
    Deliberately **not** the sum of `path_value` across downstream paths: paths share edges, so
    summing them double-counts the same rupees and can report a figure many times larger than the
    account ever moved. Over-reporting here would trip the human gate on amount alone and make the
    criterion meaningless.

    Equally, a zero from the graph must never override a non-zero measured debit. `total_out=0` is
    exactly what a partial or stale graph projection looks like, and treating the presence of a
    `flow` object as authority over its content silently returned `Decimal('0')` for an account with
    Rs 50 lakh of measured debits — five times the gate threshold, and the gate did not fire. The
    two figures measure different scopes (graph edges versus a rolling window) and legitimately
    differ, so taking the maximum is the honest reading rather than a preference between them.
    """
    figures = []
    if evidence.flow is not None:
        figures.append(evidence.flow.total_out)
    if evidence.features is not None:
        figures.append(evidence.features.total_debit)
    return max(figures, default=Decimal(0))


# --- the human gate ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether a human must decide, and every reason why.

    The reasons are not decoration. An analyst opening a queued case needs to know which criterion
    parked it there, and `docs/06-AGENT-DESIGN.md` §6 specifies a bare boolean — which would make
    the console unable to answer "why is this waiting for me?". Returning the reasons costs nothing
    and is what makes the gate auditable rather than merely present.
    """

    required: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.required and not self.reasons:
            raise ValueError("a gate that fires must say why")
        if not self.required and self.reasons:
            raise ValueError("a gate that did not fire cannot carry reasons")


def evaluate_gate(
    *,
    confidence: float | None,
    risk_score: float | None,
    recommendation: Recommendation | None,
    amount_inr: Decimal,
    errors: tuple[str, ...] = (),
    config: ScoringConfig | None = None,
) -> GateDecision:
    """Decide whether the investigation may close itself.

    Keyword-only on purpose: five conditions of similar type, where a positional swap between
    `confidence` and `risk_score` would silently invert the gate. Nothing here consults an LLM, and
    the caller cannot override the outcome — in the graph this is a real `interrupt`, so a fired
    gate physically prevents the run from reaching `close`.

    Every criterion is evaluated, not short-circuited, so the reasons list is complete.
    """
    cfg = (config or ScoringConfig()).gate
    reasons: list[str] = []

    if confidence is not None and not 0.0 <= confidence <= 1.0:
        # A malformed model output of 99.0 would otherwise sail past the confidence criterion and
        # let the run auto-close on a number that is not a probability at all.
        raise ValueError(f"confidence is a probability and must be in 0..1; got {confidence}")

    if confidence is None:
        reasons.append("the agent produced no confidence value")
    elif confidence < cfg.min_confidence:
        # The reported value is rendered exactly, and outside the comparative clause. Formatting
        # both sides at two decimals produced "confidence 0.70 is below the 0.70 threshold" for a
        # confidence of 0.6999999 — a sentence that contradicts itself in an audit trail.
        reasons.append(
            f"confidence is below the {cfg.min_confidence:.2f} threshold (reported {confidence!r})"
        )

    if risk_score is not None and risk_score >= cfg.high_risk_score:
        reasons.append(f"risk score {risk_score:.0f} is at or above {cfg.high_risk_score:.0f}")

    # Unconditional: the agent has no authority to file anything, so every escalation is reviewed
    # even at high confidence and low score.
    if recommendation is Recommendation.ESCALATE:
        reasons.append("the recommendation is ESCALATE, which is always reviewed by a human")

    if amount_inr >= cfg.amount_threshold_inr:
        reasons.append(
            f"INR {amount_inr:,.0f} at risk is at or above the "
            f"INR {cfg.amount_threshold_inr:,.0f} threshold"
        )

    if errors:
        reasons.append(
            f"{len(errors)} tool error(s) occurred, so the evidence is incomplete: {errors[0]}"
        )

    return GateDecision(required=bool(reasons), reasons=tuple(reasons))


def requires_human(
    *,
    confidence: float | None,
    risk_score: float | None,
    recommendation: Recommendation | None,
    amount_inr: Decimal,
    errors: tuple[str, ...] = (),
    config: ScoringConfig | None = None,
) -> bool:
    """Whether the investigation must be reviewed by a human.

    `evaluate_gate` is the richer form and should be preferred anywhere the reasons are needed — an
    analyst opening a queued case has to know *which* criterion parked it there, and a bare boolean
    cannot say. This exists under the name `docs/06-AGENT-DESIGN.md` §6 and the T1.3 roadmap row
    both specify, so a T3 implementer greping for it finds the gate that already exists instead of
    concluding there is none and writing a second one.
    """
    return evaluate_gate(
        confidence=confidence,
        risk_score=risk_score,
        recommendation=recommendation,
        amount_inr=amount_inr,
        errors=errors,
        config=config,
    ).required
