"""Tests for the deterministic risk scorer and the human gate.

This is the most important test file in the project, because the scorer carries the project's
central claim: **the LLM never computes a number that matters**. If the arithmetic here is wrong,
"reproducible and auditable" is a slogan rather than a property.

Two things are tested that a naive suite would skip:

* **Every hard negative in `docs/05-DATA-FLOW.md` §3 scores ALLOW.** High fan-in with no mule
  signature, a family's circular pooling, a genuine dormant return — these are what ordinary
  accounts look like. A scorer that flags them is worse than no scorer, because false positives are
  the problem this system exists to reduce. Their scores are pinned rather than bounded, so a
  regression shows up as a changed number instead of a still-passing inequality.
* **The gate reports *every* reason, not the first.** An analyst opening a queued case needs to
  know all the criteria that parked it there.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from sentinel.application.risk_scorer import (
    GateDecision,
    RiskAssessment,
    RiskEvidence,
    _ramp,
    amount_at_risk,
    assess_risk,
    decide,
    evaluate_gate,
    requires_human,
)
from sentinel.config.weights import (
    ACCOUNT_PROFILE,
    COMPONENT_ORDER,
    FAN_ASYMMETRY,
    IDENTITY_LINKAGE,
    NETWORK_STRUCTURE,
    PASS_THROUGH,
    VELOCITY_ANOMALY,
    GateConfig,
    RiskWeights,
    ScoreBands,
    ScoringConfig,
    ScoringThresholds,
)
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    CustomerSegment,
    FeatureWindow,
    KycLevel,
    MuleSignature,
    Recommendation,
    RiskRating,
    ScoreDecision,
)
from sentinel.domain.exceptions import ConfigurationError
from sentinel.domain.value_objects import (
    AccountProfile,
    FeatureVector,
    FundFlowPath,
    FundFlowResult,
    IdentifierLink,
    RingFinding,
    ScoreComponent,
    SharedIdentifierResult,
)

T0 = datetime(2026, 8, 1, tzinfo=UTC)
T1 = datetime(2026, 8, 3, tzinfo=UTC)
CFG = ScoringConfig()


# --- factories ----------------------------------------------------------------------------------


def features(
    *,
    credit: str = "100000",
    debit: str = "10000",
    night: float = 0.0,
    zscore: float = 0.0,
    txn_count: int = 20,
    window: FeatureWindow = FeatureWindow.D7,
) -> FeatureVector:
    return FeatureVector(
        account_id="ACC1",
        window=window,
        as_of=T1,
        txn_count=txn_count,
        total_credit=Decimal(credit),
        total_debit=Decimal(debit),
        distinct_credit_counterparties=3,
        distinct_debit_counterparties=3,
        max_single_txn=Decimal("5000"),
        avg_txn_amount=Decimal("2000"),
        night_txn_ratio=night,
        new_counterparty_ratio=0.1,
        distinct_devices=1,
        distinct_ips=1,
        days_since_last_activity=0,
        velocity_zscore=zscore,
    )


def profile(
    *,
    tenure: int = 60,
    kyc: KycLevel = KycLevel.FULL_KYC,
    dormant: bool = False,
    income: str = "1200000",
) -> AccountProfile:
    return AccountProfile(
        account_id="ACC1",
        customer_id="CUST1",
        account_type=AccountType.SAVINGS,
        status=AccountStatus.ACTIVE,
        tenure_months=tenure,
        is_dormant=dormant,
        current_balance=Decimal("25000"),
        kyc_level=kyc,
        declared_annual_income=Decimal(income),
        risk_rating=RiskRating.LOW,
        segment=CustomerSegment.MASS,
        is_pep=False,
    )


def path(hops: int = 3, value: str = "500000") -> FundFlowPath:
    return FundFlowPath(
        hop_accounts=tuple(f"ACC{i}" for i in range(hops + 1)),
        hops=hops,
        path_value=Decimal(value),
        bottleneck_amount=Decimal(value) / 2,
        flow_started=T0,
        flow_ended=T1,
    )


def cycle(hops: int = 3, value: str = "500000") -> FundFlowPath:
    """A path that closes on its origin.

    `FundFlowPath` rejects an open path filed under `cycles`, so the loop has to be built properly:
    n hops means n intermediate accounts plus the seed repeated at the end.
    """
    return FundFlowPath(
        hop_accounts=(*(f"ACC{i}" for i in range(hops)), "ACC0"),
        hops=hops,
        path_value=Decimal(value),
        bottleneck_amount=Decimal(value) / 2,
        flow_started=T0,
        flow_ended=T1,
    )


def ring_finding(
    *, members: int = 5, customers: int | None = None, value: str = "900000"
) -> RingFinding:
    """A ring of `members` accounts across `customers` distinct owners.

    `customers` defaults to `members` — the suspicious case, one account per person. Pass 1 for a
    single customer's own portfolio, which is not a ring at all.
    """
    return RingFinding(
        ring_id="R1",
        members=tuple(f"ACC{i}" for i in range(members)),
        distinct_customers=members if customers is None else customers,
        shared_device_id="DEV1",
        total_internal_value=Decimal(value),
    )


def flow(
    *,
    fan_in: int = 0,
    fan_out: int = 0,
    signature: MuleSignature = MuleSignature.NO_CLEAR_SIGNATURE,
    cycles: tuple[FundFlowPath, ...] = (),
    paths: tuple[FundFlowPath, ...] = (),
    ring: RingFinding | None = None,
    total_out: str = "60000",
    max_hops: int = 3,
) -> FundFlowResult:
    """A graph answer.

    `total_out` is non-zero by default. It used to default to `"0"`, which meant most tests in this
    file silently exercised the "graph reports nothing left the account" path of `amount_at_risk` —
    the exact shape of a partial graph load — without any of them meaning to. Tests that do mean it
    pass `total_out="0"` explicitly.
    """
    return FundFlowResult(
        account_id="ACC1",
        max_hops=max_hops,
        downstream_paths=paths,
        cycles=cycles,
        fan_in_degree=fan_in,
        fan_out_degree=fan_out,
        total_in=Decimal("0"),
        total_out=Decimal(total_out),
        signature=signature,
        ring=ring,
    )


def identifiers(*, devices: int = 0, others: int = 0) -> SharedIdentifierResult:
    """Build a shared-identifier result.

    `IdentifierLink` refuses to exist with `distinct_other_customers=0` — the Q4 Cypher filters on
    `c1.customer_id <> c2.customer_id`, so a link reaching nobody else means the inequality was lost
    from the query. "Checked, nothing shared" is therefore an *empty* tuple, never a link with a
    zero count, so this factory rejects the combination rather than building an impossible object.
    """
    if devices and not others:
        raise AssertionError(
            "no-cross-customer-sharing is represented by devices=0, not by a link with others=0"
        )
    links = tuple(
        IdentifierLink(
            identifier=f"DEV{i}",
            linked_accounts=tuple(f"ACC9{j}" for j in range(others)),
            distinct_other_customers=others,
        )
        for i in range(devices)
    )
    return SharedIdentifierResult(account_id="ACC1", shared_devices=links)


def component(assessment, code: str):
    return next(c for c in assessment.components if c.code == code)


# --- config validation --------------------------------------------------------------------------


def test_default_weights_total_one_hundred():
    """The score is presented as 0-100, so the weights must actually sum to 100."""
    assert sum(RiskWeights().as_mapping().values()) == pytest.approx(100.0)


def test_weights_that_do_not_total_one_hundred_are_rejected():
    with pytest.raises(ConfigurationError, match="total 100"):
        RiskWeights(pass_through=30.0)


def test_negative_weights_are_rejected():
    """The weights must total 100 *and* be non-negative, and the two guards are not the same one.

    The previous version of this test summed to 95, so the total-100 guard raised first and a bare
    `pytest.raises` could not tell which branch fired — the negativity check was never reached.
    """
    # Sums to exactly 100 (-25 + 45 + 40 + 20 + 10 + 10), so the total guard cannot fire instead.
    with pytest.raises(ConfigurationError, match="cannot be negative"):
        RiskWeights(
            pass_through=-25.0, fan_asymmetry=45.0, network_structure=40.0, account_profile=20.0
        )


@pytest.mark.parametrize(
    ("review", "block"),
    [(80.0, 40.0), (0.0, 80.0), (40.0, 120.0), (40.0, 40.0)],
)
def test_incoherent_score_bands_are_rejected(review, block):
    with pytest.raises(ConfigurationError):
        ScoreBands(review_at=review, block_at=block)


def test_the_gate_threshold_cannot_exceed_the_block_band():
    """Otherwise the fast path can block a payment while the case explaining it auto-closes."""
    with pytest.raises(ConfigurationError, match="must not exceed"):
        ScoringConfig(bands=ScoreBands(block_at=70.0), gate=GateConfig(high_risk_score=80.0))


def test_a_gate_stricter_than_the_block_band_is_allowed():
    """A gate at 70 against a block band of 80 queues more cases, which violates nothing.

    Equality here used to be mandatory, which would have turned `AGENT_HIGH_RISK_SCORE` — a
    documented, range-validated `.env` knob — into a startup crash for any value but 80.
    """
    cfg = ScoringConfig(bands=ScoreBands(block_at=80.0), gate=GateConfig(high_risk_score=70.0))
    assert cfg.gate.high_risk_score == 70.0


def test_confidence_threshold_must_be_a_probability():
    with pytest.raises(ConfigurationError, match="probability"):
        GateConfig(min_confidence=1.5)


def test_a_high_risk_score_outside_zero_to_one_hundred_is_rejected():
    with pytest.raises(ConfigurationError, match=r"must be in 0\.\.100"):
        GateConfig(high_risk_score=150.0)


def test_a_negative_amount_threshold_is_rejected():
    with pytest.raises(ConfigurationError, match="cannot be negative"):
        GateConfig(amount_threshold_inr=Decimal("-1"))


# --- threshold validation -----------------------------------------------------------------------


def test_default_thresholds_construct():
    """The guards below must not reject the shipped defaults."""
    assert ScoringThresholds().material_amount_inr == Decimal("100000")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pass_through_floor": 0.99, "pass_through_full": 0.95},
        {"fan_degree_floor": 30.0, "fan_degree_full": 25.0},
        {"ring_size_floor": 9.0, "ring_size_full": 5.0},
        {"income_multiple_floor": 900.0, "income_multiple_full": 100.0},
        {"velocity_zscore_floor": 9.0, "velocity_zscore_full": 6.0},
        {"night_ratio_floor": 0.9, "night_ratio_full": 0.6},
        {"shared_customers_floor": 9.0, "shared_customers_full": 4.0},
    ],
)
def test_an_inverted_ramp_is_rejected_at_construction_not_at_scoring_time(kwargs):
    """An inverted ramp used to raise a bare `ValueError` from inside the scorer, mid-request.

    A config error must surface at startup. Failing while scoring a live account turns a typo in a
    threshold table into an outage on requests that had nothing wrong with them.
    """
    with pytest.raises(ConfigurationError, match="must exceed"):
        ScoringThresholds(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unsigned_fan_factor": 5.0},
        {"unsigned_fan_factor": -0.1},
        {"cycle_share": 1.5, "hop_depth_share": 0.0, "ring_size_share": -0.5},
        {"profile_sub_signal_share": 1.5},
    ],
)
def test_a_fraction_outside_zero_to_one_is_rejected(kwargs):
    """`unsigned_fan_factor=5.0` makes FAN_ASYMMETRY worth 100 against a weight of 20.

    This is the only way a component can exceed its own weight, which would break the claim that
    the score is the sum of six bounded contributions.
    """
    with pytest.raises(ConfigurationError, match=r"must be in 0\.\.1"):
        ScoringThresholds(**kwargs)


def test_network_shares_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ConfigurationError, match=r"must sum to 1\.0"):
        ScoringThresholds(cycle_share=0.9, hop_depth_share=0.25, ring_size_share=0.25)


def test_four_profile_sub_signals_cannot_exceed_the_component_weight():
    with pytest.raises(ConfigurationError, match="would exceed the ACCOUNT_PROFILE weight"):
        ScoringThresholds(profile_sub_signal_share=0.5)


@pytest.mark.parametrize("amount", [Decimal("-1"), Decimal("0")])
def test_a_non_positive_material_amount_is_rejected(amount):
    """Zero as well as negative: it would make the materiality ramp run from 0 to 0."""
    with pytest.raises(ConfigurationError, match="must be positive"):
        ScoringThresholds(material_amount_inr=amount)


def test_a_custom_threshold_set_changes_the_score_without_touching_the_scorer():
    """Thresholds are data, same as weights — a tuner edits config, not the scorer."""
    evidence = RiskEvidence(account_id="ACC1", features=features(credit="100000", debit="70000"))
    default = assess_risk(evidence)
    lenient = assess_risk(
        evidence, ScoringConfig(thresholds=ScoringThresholds(pass_through_floor=0.75))
    )
    # 0.70 sits above the shipped floor of 0.60 and below a floor of 0.75.
    assert component(default, PASS_THROUGH).contribution > 0.0
    assert component(lenient, PASS_THROUGH).contribution == 0.0


# --- the ramp -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (2.0, 0.0), (4.0, 0.5), (6.0, 1.0), (99.0, 1.0)],
)
def test_ramp_is_linear_between_floor_and_full_and_clamped_outside(value, expected):
    assert _ramp(value, 2.0, 6.0) == pytest.approx(expected)


def test_ramp_rejects_an_inverted_range():
    """A floor above the full value would silently invert the signal."""
    with pytest.raises(ValueError, match="full > floor"):
        _ramp(1.0, 6.0, 2.0)


# --- missing evidence ---------------------------------------------------------------------------


def test_with_no_evidence_at_all_the_score_is_zero_and_every_component_says_so():
    """Absence of evidence must never read as absence of risk."""
    result = assess_risk(RiskEvidence(account_id="ACC1"))
    assert result.score == 0.0
    assert result.decision is ScoreDecision.ALLOW
    assert result.evidence_sources == 0
    assert len(result.components) == 6
    assert all(c.contribution == 0.0 for c in result.components)
    assert all(c.detail == "not gathered" for c in result.components)


def test_all_six_components_are_always_present_and_in_a_stable_order():
    """The console renders this breakdown and the narrative cites it, so order must not drift."""
    result = assess_risk(RiskEvidence(account_id="ACC1", features=features()))
    assert tuple(c.code for c in result.components) == COMPONENT_ORDER


def test_a_gathered_but_clean_source_is_distinguishable_from_a_missing_one():
    """ "We checked and it was clean" and "we never checked" must not collapse into one value."""
    checked = assess_risk(RiskEvidence(account_id="ACC1", flow=flow()))
    missing = assess_risk(RiskEvidence(account_id="ACC1"))
    assert component(checked, NETWORK_STRUCTURE).contribution == 0.0
    assert component(missing, NETWORK_STRUCTURE).contribution == 0.0
    assert component(checked, NETWORK_STRUCTURE).detail != "not gathered"
    assert component(missing, NETWORK_STRUCTURE).detail == "not gathered"
    assert checked.evidence_sources == 1 and missing.evidence_sources == 0


def test_evidence_sources_counts_what_was_actually_supplied():
    ev = RiskEvidence(account_id="ACC1", features=features(), profile=profile())
    assert ev.gathered_count == 2


# --- individual components ----------------------------------------------------------------------


def test_full_pass_through_earns_the_whole_component_and_no_more():
    """Retention within the window is 1 - pass_through; scoring both would double-count to 50."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(credit="100000", debit="100000"))
    )
    assert component(result, PASS_THROUGH).contribution == pytest.approx(25.0)


def test_ordinary_spending_earns_nothing_for_pass_through():
    """A salaried customer forwarding half of an inbound salary is not a mule."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(credit="100000", debit="50000"))
    )
    assert component(result, PASS_THROUGH).contribution == 0.0


def test_debits_above_credits_saturate_rather_than_overflow_the_component():
    """A ratio above 1.0 is real — the account paid out more than came in, from a standing balance.

    The ramp clamps at 1.0, and the detail must not print "retaining -50%".
    """
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(credit="100000", debit="150000"))
    )
    detail = component(result, PASS_THROUGH).detail
    assert component(result, PASS_THROUGH).contribution == pytest.approx(25.0)
    assert "Forwarded 100%" in detail
    assert "retaining 0%" in detail


def test_an_account_that_drains_with_no_inflow_is_scored_on_the_drain():
    """`pass_through_ratio` is 0.0 with no credits, so the DORMANT_REACTIVATION shape scored zero.

    An account holding Rs 50 lakh that empties itself inside the window is the defining mule
    behaviour, and the outflow is the only measurement that shows it.
    """
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(credit="0", debit="5000000"))
    )
    assert component(result, PASS_THROUGH).contribution == pytest.approx(25.0)
    assert "no matching inflow" in component(result, PASS_THROUGH).detail


def test_a_small_drain_with_no_inflow_is_ramped_not_flagged():
    """Rs 30,000 leaving a dormant account is housekeeping, not layering."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(credit="0", debit="30000"))
    )
    # Rs 30,000 sits 2/9 of the way up a ramp from Rs 10,000 to Rs 1 lakh.
    assert component(result, PASS_THROUGH).contribution == pytest.approx(25.0 * 2 / 9)


def test_an_account_with_no_movement_at_all_scores_nothing_for_pass_through():
    """No credits *and* no debits is a quiet account, not a drain — the branch must not fire."""
    result = assess_risk(RiskEvidence(account_id="ACC1", features=features(credit="0", debit="0")))
    assert component(result, PASS_THROUGH).contribution == 0.0
    assert "no matching inflow" not in component(result, PASS_THROUGH).detail


def test_high_fan_degree_without_a_signature_is_discounted_not_ignored():
    unsigned = assess_risk(RiskEvidence(account_id="ACC1", flow=flow(fan_in=25)))
    signed = assess_risk(
        RiskEvidence(
            account_id="ACC1", flow=flow(fan_in=25, signature=MuleSignature.COLLECTION_MULE)
        )
    )
    unsigned_points = component(unsigned, FAN_ASYMMETRY).contribution
    signed_points = component(signed, FAN_ASYMMETRY).contribution
    assert signed_points == pytest.approx(20.0)
    assert 0.0 < unsigned_points < signed_points
    assert unsigned_points == pytest.approx(20.0 * 0.40)


def test_a_dispersal_signature_counts_at_full_weight_like_a_collection_one():
    """Both signatures are asymmetry; only `NO_CLEAR_SIGNATURE` is discounted."""
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            flow=flow(fan_in=1, fan_out=25, signature=MuleSignature.DISPERSAL_MULE),
        )
    )
    assert component(result, FAN_ASYMMETRY).contribution == pytest.approx(20.0)
    assert "DISPERSAL_MULE" in component(result, FAN_ASYMMETRY).detail


def test_a_material_cycle_alone_carries_half_the_network_component():
    """Funds returning to origin has almost no innocent explanation in retail banking."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(cycles=(cycle(hops=3, value="500000"),)))
    )
    assert component(result, NETWORK_STRUCTURE).contribution == pytest.approx(10.0)


def test_a_small_cycle_is_scaled_down_because_a_traversal_cannot_see_scale():
    """A Rs 40,000 loop between siblings and a Rs 40,00,000 laundering loop are the same shape.

    Amount-blind, they scored identically, which is the single biggest lever on the family-pooling
    false positive.
    """
    result = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(cycles=(cycle(hops=3, value="40000"),)))
    )
    # Rs 40,000 sits a third of the way up a ramp from Rs 10,000 to Rs 1 lakh.
    assert component(result, NETWORK_STRUCTURE).contribution == pytest.approx(10.0 / 3)


def test_hop_depth_is_scored_against_the_depth_actually_searched():
    """A 2-hop projection must not be penalised for missing 4-hop chains it never looked for.

    Exact values, not just `shallow > deep`: any monotone-but-wrong formula passes an inequality.
    """
    shallow = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(paths=(path(hops=2),), max_hops=2))
    )
    deep = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(paths=(path(hops=2),), max_hops=4))
    )
    # 20 points x a 0.25 share x (2 hops found / 2 searched), then x (2 / 4).
    assert component(shallow, NETWORK_STRUCTURE).contribution == pytest.approx(5.0)
    assert component(deep, NETWORK_STRUCTURE).contribution == pytest.approx(2.5)


def test_the_network_component_cannot_exceed_its_weight():
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            flow=flow(
                cycles=(cycle(hops=3),),
                paths=(path(hops=3),),
                ring=ring_finding(members=8),
            ),
        )
    )
    assert component(result, NETWORK_STRUCTURE).contribution == pytest.approx(20.0)


def test_a_ring_of_one_customers_own_accounts_contributes_nothing():
    """`distinct_customers` is the load-bearing number, and the ramp used to ignore it entirely.

    Five accounts owned by one person is a portfolio; five owned by five people on one device is a
    ring. Scoring on member count alone put a customer's own five accounts at 40.00.
    """
    result = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(ring=ring_finding(members=5, customers=1)))
    )
    assert component(result, NETWORK_STRUCTURE).contribution == 0.0
    assert "single customer's own accounts" in component(result, NETWORK_STRUCTURE).detail


@pytest.mark.parametrize(
    ("customers", "expected_fraction"),
    [(2, 0.25), (3, 0.50), (4, 0.75), (5, 1.0), (8, 1.0)],
)
def test_the_ring_ramp_runs_on_people_not_accounts(customers, expected_fraction):
    """A ring of eight accounts across two customers is two people, not eight."""
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1", flow=flow(ring=ring_finding(members=8, customers=customers))
        )
    )
    assert component(result, NETWORK_STRUCTURE).contribution == pytest.approx(
        20.0 * 0.25 * expected_fraction
    )


def test_a_ring_moving_trivial_amounts_is_scaled_down():
    """Four siblings moving Rs 40,000 between them is a household, not a laundering cell."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", flow=flow(ring=ring_finding(members=4, value="40000")))
    )
    # 20 points x a 0.25 share x (3/4 of the customer ramp) x (1/3 of the materiality ramp).
    assert component(result, NETWORK_STRUCTURE).contribution == pytest.approx(
        20.0 * 0.25 * 0.75 / 3
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_quarters"),
    [
        ({}, 0),
        ({"tenure": 2}, 1),
        ({"kyc": KycLevel.MIN_KYC}, 1),
        ({"dormant": True}, 1),
        ({"tenure": 2, "kyc": KycLevel.MIN_KYC, "dormant": True}, 3),
    ],
)
def test_profile_sub_signals_each_contribute_one_quarter(kwargs, expected_quarters):
    """No single profile attribute is damning; a new MIN_KYC account is the ordinary first case."""
    result = assess_risk(RiskEvidence(account_id="ACC1", profile=profile(**kwargs)))
    assert component(result, ACCOUNT_PROFILE).contribution == pytest.approx(
        15.0 * 0.25 * expected_quarters
    )


def test_all_four_profile_sub_signals_saturate_the_component_at_exactly_its_weight():
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            features=features(credit="12000000"),
            profile=profile(tenure=1, kyc=KycLevel.MIN_KYC, dormant=True, income="1000000"),
        )
    )
    assert component(result, ACCOUNT_PROFILE).contribution == pytest.approx(15.0)


def test_throughput_far_above_declared_income_contributes():
    """The signal a mule cannot avoid: the money has to move through the account."""
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            features=features(credit="12000000"),
            profile=profile(income="1000000"),
        )
    )
    points = component(result, ACCOUNT_PROFILE).contribution
    assert points == pytest.approx(15.0 * 0.25)
    assert "declared annual income" in component(result, ACCOUNT_PROFILE).detail


def test_the_income_ramp_reports_a_partial_value_not_only_saturation():
    """Halfway up the ramp must earn half the sub-signal, or the ramp is really a step.

    Rs 75 lakh of weekly credits against Rs 10 lakh of declared annual income is 7.5x annual income
    inside a week, which is the midpoint between the 3x floor and the 12x full value.
    """
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            features=features(credit="7500000"),
            profile=profile(income="1000000"),
        )
    )
    assert component(result, ACCOUNT_PROFILE).contribution == pytest.approx(15.0 * 0.25 * 0.5)


@pytest.mark.parametrize(
    ("window", "credit"),
    [
        (FeatureWindow.H1, "89285.714286"),
        (FeatureWindow.H24, "2142857.142857"),
        (FeatureWindow.D7, "15000000"),
        (FeatureWindow.D30, "64285714.285714"),
    ],
)
def test_the_income_comparison_is_pro_rated_to_the_window_asked_for(window, credit):
    """The same throughput *rate* must score the same whichever window the agent requested.

    Dividing window credits by an *annual* income made the sub-signal an artefact of the request:
    the same account behaving the same way scored a full quarter-component apart between 7D and 30D,
    and "we scored you higher because we looked at 30 days" is not defensible. Every row here is the
    same rate — 7.5x a Rs 20 lakh annual income per week — which is the midpoint of the ramp.
    """
    result = assess_risk(
        RiskEvidence(
            account_id="ACC1",
            features=features(credit=credit, window=window),
            profile=profile(income="2000000"),
        )
    )
    assert component(result, ACCOUNT_PROFILE).contribution == pytest.approx(
        15.0 * 0.25 * 0.5, abs=0.001
    )


def test_income_comparison_is_skipped_when_income_is_zero():
    """Dividing by a zero declared income must not raise or invent a signal."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", features=features(), profile=profile(income="0"))
    )
    assert component(result, ACCOUNT_PROFILE).contribution == 0.0


def test_velocity_averages_its_two_correlated_signals():
    """A 3am burst scores on both; summing correlated signals would inflate past the weight."""
    both = assess_risk(RiskEvidence(account_id="ACC1", features=features(night=0.6, zscore=6.0)))
    one = assess_risk(RiskEvidence(account_id="ACC1", features=features(night=0.6, zscore=0.0)))
    assert component(both, VELOCITY_ANOMALY).contribution == pytest.approx(10.0)
    assert component(one, VELOCITY_ANOMALY).contribution == pytest.approx(5.0)


def test_a_negative_velocity_zscore_scores_nothing_rather_than_inverting():
    """Quieter than usual is not an anomaly. The ramp must clamp, not run backwards."""
    result = assess_risk(RiskEvidence(account_id="ACC1", features=features(zscore=-3.0)))
    assert component(result, VELOCITY_ANOMALY).contribution == 0.0


def test_night_activity_exactly_at_the_floor_scores_nothing():
    """The floor is what ordinary looks like, so it must be excluded, not included."""
    result = assess_risk(RiskEvidence(account_id="ACC1", features=features(night=0.20)))
    assert component(result, VELOCITY_ANOMALY).contribution == 0.0


def test_one_other_customer_on_a_shared_device_scores_nothing():
    """That is a family phone, not a ring."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", identifiers=identifiers(devices=1, others=1))
    )
    assert component(result, IDENTITY_LINKAGE).contribution == 0.0


def test_three_other_adults_on_one_handset_is_a_household_not_a_ring():
    """Four adults sharing one phone used to saturate this component. It is now two thirds."""
    result = assess_risk(
        RiskEvidence(account_id="ACC1", identifiers=identifiers(devices=1, others=3))
    )
    assert component(result, IDENTITY_LINKAGE).contribution == pytest.approx(10.0 * 2 / 3)


def test_four_other_customers_on_shared_devices_saturate_identity_linkage():
    result = assess_risk(
        RiskEvidence(account_id="ACC1", identifiers=identifiers(devices=1, others=4))
    )
    assert component(result, IDENTITY_LINKAGE).contribution == pytest.approx(10.0)


def test_two_devices_reaching_the_same_relatives_are_not_counted_twice():
    """A household with a shared phone *and* a shared tablet is still one household.

    Summing across devices reported "4 other customer(s)" for two relatives and saturated the
    component at double the true signal — a false number in an audit string. `max` is the
    conservative approximation available while `IdentifierLink` carries counts rather than ids, and
    the detail says "at least" for exactly that reason.
    """
    one_device = assess_risk(
        RiskEvidence(account_id="ACC1", identifiers=identifiers(devices=1, others=2))
    )
    two_devices = assess_risk(
        RiskEvidence(account_id="ACC1", identifiers=identifiers(devices=2, others=2))
    )
    linkage = component(two_devices, IDENTITY_LINKAGE)
    assert linkage.contribution == component(one_device, IDENTITY_LINKAGE).contribution
    assert "at least 2 other customer(s)" in linkage.detail


# --- whole-account scenarios ---------------------------------------------------------------------


def full_signal_evidence() -> RiskEvidence:
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="12000000", debit="12000000", night=0.6, zscore=6.0),
        profile=profile(tenure=1, kyc=KycLevel.MIN_KYC, dormant=True, income="1000000"),
        flow=flow(
            fan_in=25,
            fan_out=1,
            signature=MuleSignature.COLLECTION_MULE,
            cycles=(cycle(hops=3),),
            paths=(path(hops=3),),
            ring=ring_finding(members=5),
            total_out="12000000",
        ),
        identifiers=identifiers(devices=2, others=4),
    )


def test_a_textbook_mule_scores_at_the_top_and_blocks():
    result = assess_risk(full_signal_evidence())
    assert result.score == pytest.approx(100.0)
    assert result.decision is ScoreDecision.BLOCK
    assert result.evidence_sources == 4


# --- the four hard negatives ---------------------------------------------------------------------
#
# `docs/05-DATA-FLOW.md` §3 lists four legitimate patterns that *look* like typologies. They are
# the reason the metrics mean anything, and they are the cases that decide whether this scorer is
# worth deploying: an analyst drowning in reviewed merchants is worse off than with no scorer.
#
# Values are realistic rather than convenient. A merchant sweeps 85-95% of a week's takings to
# suppliers — the earlier version of this test used a merchant *retaining* 70% of its takings, which
# is not how a business runs, and scored 5.65 against a `< 40` assertion. A 34-point margin would
# have survived pass-through being mis-weighted six-fold.
#
# Each score is pinned with `pytest.approx`, not bounded, so a regression reads as a changed number.


def small_merchant() -> RiskEvidence:
    """Genuine high fan-in: 22 payers a week, 90% swept to suppliers, nine-year full-KYC account."""
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="500000", debit="450000", night=0.05, zscore=1.0, txn_count=180),
        profile=profile(tenure=108, kyc=KycLevel.FULL_KYC, income="3000000"),
        flow=flow(
            fan_in=22, fan_out=4, signature=MuleSignature.NO_CLEAR_SIGNATURE, total_out="450000"
        ),
        identifiers=identifiers(),
    )


def merchant_on_a_household_phone() -> RiskEvidence:
    """The same shop, harder: a 24-hour restaurant whose handset four adults share.

    Sweeps 91% of takings to suppliers, trades past midnight, genuine fan-in of 22. This is the
    profile that scored 40.06 and landed in REVIEW — every element of it innocent.
    """
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="500000", debit="455000", night=0.30, zscore=1.2, txn_count=200),
        profile=profile(tenure=108, kyc=KycLevel.FULL_KYC, income="3000000"),
        flow=flow(
            fan_in=22, fan_out=4, signature=MuleSignature.NO_CLEAR_SIGNATURE, total_out="455000"
        ),
        identifiers=identifiers(devices=1, others=3),
    )


def salary_day_fan_out() -> RiskEvidence:
    """A corporate salary-disbursal account paying 24 staff, same recipients every month.

    Fan-in of five (parent company, two subsidiaries, a client, a refund) is what keeps the Q3
    signature at `NO_CLEAR_SIGNATURE`: `DISPERSAL_MULE` needs fan-in of three or fewer. The
    asymmetry test earning its place.
    """
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="6500000", debit="4900000", night=0.02, zscore=1.8, txn_count=40),
        profile=profile(tenure=120, kyc=KycLevel.FULL_KYC, income="60000000"),
        flow=flow(
            fan_in=5, fan_out=24, signature=MuleSignature.NO_CLEAR_SIGNATURE, total_out="4900000"
        ),
        identifiers=identifiers(),
    )


def family_pooling() -> RiskEvidence:
    """Four siblings pooling household money: Rs 60,000 of weekly credits, a Rs 40,000 loop.

    A real cycle across a real ring of four real customers, which is why this scored 49.70 — the
    graph shape is genuinely the CIRCULAR typology. The amounts are what separate them.
    """
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="60000", debit="50000", night=0.05, zscore=0.5, txn_count=14),
        profile=profile(tenure=84, kyc=KycLevel.FULL_KYC, income="800000"),
        flow=flow(
            fan_in=5,
            fan_out=4,
            signature=MuleSignature.NO_CLEAR_SIGNATURE,
            cycles=(cycle(hops=3, value="40000"),),
            ring=ring_finding(members=4, value="40000"),
            total_out="50000",
        ),
        identifiers=identifiers(),
    )


def genuine_dormant_return() -> RiskEvidence:
    """Dormant 210 days, then one large legitimate credit and no onward dispersal."""
    return RiskEvidence(
        account_id="ACC1",
        features=features(credit="500000", debit="20000", night=0.0, zscore=3.5, txn_count=3),
        profile=profile(tenure=120, kyc=KycLevel.FULL_KYC, dormant=True, income="1500000"),
        flow=flow(
            fan_in=1, fan_out=1, signature=MuleSignature.NO_CLEAR_SIGNATURE, total_out="20000"
        ),
        identifiers=identifiers(),
    )


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (small_merchant, 28.02),
        (merchant_on_a_household_phone, 36.65),
        (salary_day_fan_out, 18.52),
        (family_pooling, 21.25),
        (genuine_dormant_return, 5.62),
    ],
    ids=["merchant", "merchant_household_phone", "salary_fan_out", "family_pooling", "dormant"],
)
def test_every_hard_negative_is_allowed(build, expected):
    result = assess_risk(build())
    assert result.score == pytest.approx(expected, abs=0.01)
    assert result.decision is ScoreDecision.ALLOW, (
        f"{build.__name__} scored {result.score}, which would review or block a legitimate account"
    )


def test_no_component_can_exceed_its_own_weight():
    """Why the clamp in `assess_risk` is unreachable, asserted directly.

    The previous test here checked the *clamped* score against 100, which
    `RiskAssessment.__post_init__` has already enforced by raising — so it could not fail. The real
    invariant is per component: six bounded contributions is what makes the sum bounded.
    """
    weights = ScoringConfig().weights.as_mapping()
    result = assess_risk(full_signal_evidence())
    for item in result.components:
        assert item.contribution <= weights[item.code] + 1e-9, item
    assert sum(c.contribution for c in result.components) == pytest.approx(100.0)


def _digest(result: RiskAssessment) -> str:
    """A stable fingerprint of everything the scorer decided, contributions and prose included."""
    payload = "|".join(f"{c.code}={c.contribution:.9f}:{c.detail}" for c in result.components)
    return hashlib.sha256(f"{result.score:.9f}|{payload}".encode()).hexdigest()


def test_scoring_is_reproducible():
    """Same evidence, same config, same number — the whole audit claim rests on this.

    Pinned against a digest rather than only re-scored in-process: a dict or set iteration order
    leaking into a contribution or a detail string would reproduce perfectly within one run and
    differ between runs, because `PYTHONHASHSEED` is fresh per process.
    """
    evidence = full_signal_evidence()
    first = assess_risk(evidence)
    assert _digest(first) == _digest(assess_risk(evidence))
    assert _digest(first) == "9d48896a03706a52d13bc6a828627458d96b86b12393ada82f43d990172c0298"


def test_reasons_lists_only_components_that_contributed():
    clean = assess_risk(RiskEvidence(account_id="ACC1", features=features()))
    assert clean.reasons == ()
    assert len(assess_risk(full_signal_evidence()).reasons) == 6


def test_changing_a_weight_changes_the_score_without_touching_the_scorer():
    """Tuning must be a config edit, not a code edit — this is why weights are data."""
    evidence = RiskEvidence(account_id="ACC1", features=features(credit="100000", debit="100000"))
    default = assess_risk(evidence)
    reweighted = assess_risk(
        evidence,
        ScoringConfig(
            weights=RiskWeights(
                pass_through=50.0,
                fan_asymmetry=10.0,
                network_structure=20.0,
                account_profile=10.0,
                velocity_anomaly=5.0,
                identity_linkage=5.0,
            )
        ),
    )
    assert component(default, PASS_THROUGH).contribution == pytest.approx(25.0)
    assert component(reweighted, PASS_THROUGH).contribution == pytest.approx(50.0)


# --- decision bands -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, ScoreDecision.ALLOW),
        (39.9, ScoreDecision.ALLOW),
        (40.0, ScoreDecision.REVIEW),
        (79.9, ScoreDecision.REVIEW),
        (80.0, ScoreDecision.BLOCK),
        (100.0, ScoreDecision.BLOCK),
    ],
)
def test_band_boundaries_are_inclusive_at_the_lower_edge(score, expected):
    assert decide(score) is expected


# --- amount at risk -----------------------------------------------------------------------------


def test_amount_at_risk_takes_the_larger_of_the_graph_and_the_measured_figure():
    """The two measure different scopes — graph edges versus a rolling window — both are real."""
    ev = RiskEvidence(
        account_id="ACC1", features=features(debit="10000"), flow=flow(total_out="750000")
    )
    assert amount_at_risk(ev) == Decimal("750000")


def test_a_zero_graph_figure_never_overrides_a_measured_debit():
    """`total_out=0` is exactly what a partial or stale graph projection looks like.

    Treating the presence of a `flow` object as authority over its content returned `Decimal('0')`
    for an account with Rs 50 lakh of measured debits — five times the gate threshold — and the
    human gate silently did not fire on amount.
    """
    ev = RiskEvidence(
        account_id="ACC1", features=features(debit="5000000"), flow=flow(total_out="0")
    )
    assert amount_at_risk(ev) == Decimal("5000000")


def test_amount_at_risk_falls_back_to_debits_when_the_graph_was_not_queried():
    assert amount_at_risk(
        RiskEvidence(account_id="ACC1", features=features(debit="42000"))
    ) == Decimal("42000")


def test_amount_at_risk_is_zero_with_no_evidence():
    assert amount_at_risk(RiskEvidence(account_id="ACC1")) == Decimal(0)


def test_amount_at_risk_does_not_sum_path_values():
    """Paths share edges, so summing `path_value` double-counts the same rupees.

    Three 500k paths would report 1.5m against an account the graph says moved nothing, and trip the
    gate on amount alone, which is why the honest figure is what actually left.
    """
    ev = RiskEvidence(
        account_id="ACC1",
        flow=flow(
            paths=(path(value="500000"), path(value="500000"), path(value="500000")),
            total_out="0",
        ),
    )
    assert amount_at_risk(ev) == Decimal("0")


# --- the human gate -----------------------------------------------------------------------------


def test_a_confident_low_risk_close_needs_no_human():
    gate = evaluate_gate(
        confidence=0.95,
        risk_score=12.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("5000"),
    )
    assert gate.required is False
    assert gate.reasons == ()


def test_low_confidence_fires_the_gate():
    gate = evaluate_gate(
        confidence=0.4,
        risk_score=10.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("0"),
    )
    assert gate.required is True
    assert any("confidence" in r for r in gate.reasons)


def test_absent_confidence_fires_the_gate():
    """No confidence value is not the same as a high one."""
    gate = evaluate_gate(
        confidence=None,
        risk_score=10.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("0"),
    )
    assert gate.required is True
    assert any("no confidence" in r for r in gate.reasons)


def test_a_high_risk_score_fires_the_gate():
    gate = evaluate_gate(
        confidence=0.99,
        risk_score=80.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("0"),
    )
    assert gate.required is True
    assert any("risk score" in r for r in gate.reasons)


def test_escalate_always_fires_the_gate_even_when_confident_and_low_risk():
    """The agent has no authority to file anything, so every escalation is reviewed."""
    gate = evaluate_gate(
        confidence=1.0,
        risk_score=0.0,
        recommendation=Recommendation.ESCALATE,
        amount_inr=Decimal("0"),
    )
    assert gate.required is True
    assert any("ESCALATE" in r for r in gate.reasons)


def test_a_large_amount_fires_the_gate():
    gate = evaluate_gate(
        confidence=0.99,
        risk_score=5.0,
        recommendation=Recommendation.MONITOR,
        amount_inr=Decimal("1000000"),
    )
    assert gate.required is True
    assert any("at risk" in r for r in gate.reasons)


def test_a_tool_error_fires_the_gate():
    """Incomplete evidence cannot support an auto-close."""
    gate = evaluate_gate(
        confidence=0.99,
        risk_score=5.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("0"),
        errors=("neo4j traversal timed out",),
    )
    assert gate.required is True
    assert any("neo4j traversal timed out" in r for r in gate.reasons)


def test_the_gate_reports_every_reason_not_just_the_first():
    """An analyst needs all the criteria that queued a case, not whichever was checked first."""
    gate = evaluate_gate(
        confidence=0.2,
        risk_score=95.0,
        recommendation=Recommendation.ESCALATE,
        amount_inr=Decimal("5000000"),
        errors=("qdrant unavailable",),
    )
    assert gate.required is True
    assert len(gate.reasons) == 5


def test_gate_thresholds_are_configurable():
    strict = ScoringConfig(bands=ScoreBands(block_at=50.0), gate=GateConfig(high_risk_score=50.0))
    args = {
        "confidence": 0.99,
        "risk_score": 60.0,
        "recommendation": Recommendation.MONITOR,
        "amount_inr": Decimal("0"),
    }
    assert evaluate_gate(**args).required is False
    assert evaluate_gate(**args, config=strict).required is True


def test_an_absent_recommendation_does_not_fire_the_gate_by_itself():
    """`None` is not `ESCALATE`. An agent producing no recommendation is caught by confidence."""
    gate = evaluate_gate(
        confidence=0.95,
        risk_score=10.0,
        recommendation=None,
        amount_inr=Decimal("0"),
    )
    assert gate.required is False
    assert gate.reasons == ()


@pytest.mark.parametrize("confidence", [99.0, 1.5, -0.1])
def test_a_confidence_that_is_not_a_probability_is_rejected(confidence):
    """A malformed model output above 1.0 must not sail past the confidence criterion."""
    with pytest.raises(ValueError, match=r"must be in 0\.\.1"):
        evaluate_gate(
            confidence=confidence,
            risk_score=10.0,
            recommendation=Recommendation.CLOSE,
            amount_inr=Decimal("0"),
        )


def test_the_confidence_reason_does_not_contradict_itself_on_a_rounding_boundary():
    """Rendering both sides at two decimals produced "0.70 is below the 0.70 threshold"."""
    gate = evaluate_gate(
        confidence=0.6999999,
        risk_score=10.0,
        recommendation=Recommendation.CLOSE,
        amount_inr=Decimal("0"),
    )
    assert gate.reasons == ("confidence is below the 0.70 threshold (reported 0.6999999)",)


def test_a_firing_gate_must_state_a_reason():
    with pytest.raises(ValueError, match="must say why"):
        GateDecision(required=True, reasons=())


def test_a_silent_gate_cannot_carry_reasons():
    with pytest.raises(ValueError, match="cannot carry reasons"):
        GateDecision(required=False, reasons=("something",))


# --- requires_human -----------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.95, 0.4])
def test_requires_human_agrees_with_evaluate_gate(confidence):
    """The name the design doc and the roadmap specify, delegating to the richer form."""
    args = {
        "confidence": confidence,
        "risk_score": 12.0,
        "recommendation": Recommendation.CLOSE,
        "amount_inr": Decimal("5000"),
    }
    assert requires_human(**args) is evaluate_gate(**args).required


def test_requires_human_honours_a_custom_config():
    strict = ScoringConfig(bands=ScoreBands(block_at=50.0), gate=GateConfig(high_risk_score=50.0))
    args = {
        "confidence": 0.99,
        "risk_score": 60.0,
        "recommendation": Recommendation.MONITOR,
        "amount_inr": Decimal("0"),
    }
    assert requires_human(**args) is False
    assert requires_human(**args, config=strict) is True


# --- assessment invariants ----------------------------------------------------------------------


def _six_components():
    return tuple(ScoreComponent(code, code, 0.0, "clean") for code in COMPONENT_ORDER)


@pytest.mark.parametrize("score", [-1.0, 101.0])
def test_an_assessment_outside_zero_to_one_hundred_is_rejected(score):
    with pytest.raises(ValueError, match=r"score must be in 0\.\.100"):
        RiskAssessment(
            account_id="ACC1",
            score=score,
            decision=ScoreDecision.ALLOW,
            components=_six_components(),
            evidence_sources=0,
        )


def test_an_assessment_missing_a_component_is_rejected():
    """Missing evidence scores zero and says so; it is never omitted from the breakdown."""
    with pytest.raises(ValueError, match="all six components"):
        RiskAssessment(
            account_id="ACC1",
            score=0.0,
            decision=ScoreDecision.ALLOW,
            components=_six_components()[:5],
            evidence_sources=0,
        )
