"""T6.1/T6.2/T6.3 — the rule engine, the anomaly model, and the alert rows they produce.

Three things are worth asserting here and the rest is arithmetic:

* **Each rule fires on exactly its own condition and not on a near miss.** Every rule gets a
  positive case and a negative case one step below its threshold, because an off-by-one in a
  detection threshold is invisible in aggregate counts and obvious per rule.
* **Nothing decides an outcome from a literal.** The threshold cases are built by reading
  `RuleThresholds`, so moving a threshold in config moves the test with it. A test that hard-coded
  `8` would pass forever against a rule that had quietly stopped reading its config.
* **A re-run produces identical alerts.** That is the whole of T6.3's upsert claim, and it is
  checkable without a database: same inputs, same `alert_id`s, same rows.

The Postgres write itself is covered by `tests/integration/test_detection_alerts.py`, which needs a
live store and carries the `integration` marker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from sentinel.application.risk_scorer import RiskEvidence, assess_risk
from sentinel.config.weights import (
    ML_SPEC,
    R001,
    R002,
    R003,
    R004,
    R005,
    R006,
    R007,
    RULE_SPECS,
    DetectionConfig,
    IsolationForestConfig,
    RuleSpec,
    RuleThresholds,
)
from sentinel.domain.entities import AccountFeatures
from sentinel.domain.enums import AlertStatus, FeatureWindow, Severity
from sentinel.domain.exceptions import ConfigurationError, DomainValidationError
from sentinel.pipeline import anomaly, detect, rules

AS_OF = datetime(2026, 8, 4, 0, 0, tzinfo=UTC)
TH = RuleThresholds()


# --- fixtures -------------------------------------------------------------------------------------


def make_row(window: FeatureWindow = FeatureWindow.D7, **overrides: object) -> AccountFeatures:
    """A quiet, unremarkable feature row. Every rule test moves exactly one field off this."""
    fields: dict[str, object] = {
        "account_id": "ACC00000001",
        "window": window,
        "as_of": AS_OF,
        "txn_count": 4,
        "credit_count": 2,
        "debit_count": 2,
        "total_credit": Decimal("40000.00"),
        "total_debit": Decimal("20000.00"),
        "balance_delta": Decimal("20000.00"),
        "distinct_credit_counterparties": 2,
        "distinct_debit_counterparties": 2,
        "max_single_txn": Decimal("20000.00"),
        "avg_txn_amount": Decimal("15000.00"),
        "pass_through_ratio": 0.5,
        "night_txn_ratio": 0.1,
        "new_counterparty_ratio": 0.1,
        "structuring_band_count": 0,
        "distinct_devices": 1,
        "distinct_ips": 1,
        "days_since_last_activity": 2,
        "velocity_zscore": 0.2,
    }
    fields.update(overrides)
    return AccountFeatures(**fields)  # type: ignore[arg-type]


def make_evidence(
    *,
    d7: AccountFeatures | None = None,
    d30: AccountFeatures | None = None,
    closing_balance: Decimal = Decimal("500000.00"),
    shared_device_accounts: int = 0,
    shared_device_customers: int = 0,
) -> rules.AccountEvidence:
    seven = d7 if d7 is not None else make_row(FeatureWindow.D7)
    thirty = d30 if d30 is not None else make_row(FeatureWindow.D30)
    return rules.AccountEvidence(
        account_id=seven.account_id,
        rows={FeatureWindow.D7: seven, FeatureWindow.D30: thirty},
        closing_balance=closing_balance,
        shared_device_accounts=shared_device_accounts,
        shared_device_customers=shared_device_customers,
    )


def codes(hits: tuple[object, ...]) -> list[str]:
    return [hit.rule_code for hit in hits]  # type: ignore[attr-defined]


def velocity_row(txn_count: int, multiple: float) -> AccountFeatures:
    """A 7D row whose stored z-score implies exactly `multiple` times the account's own baseline.

    Built by running `pipeline/features.py`'s formula forwards — `z = (n - E) / sqrt(E)` with `E = n
    / multiple` — so the test exercises `implied_expected_count` against the definition it inverts
    rather than against a number copied out of the implementation.
    """
    expected = txn_count / multiple
    zscore = (txn_count - expected) / (expected**0.5)
    return make_row(txn_count=txn_count, velocity_zscore=zscore)


# --- config ---------------------------------------------------------------------------------------


def test_all_seven_rules_are_configured_and_implemented():
    assert [spec.code for spec in RULE_SPECS] == [R001, R002, R003, R004, R005, R006, R007]
    assert set(rules.RULES) == {spec.code for spec in RULE_SPECS}


def test_no_single_rule_can_dominate_the_component():
    """The design constraint stated in `RuleSpec`: the largest contribution is under half of 100,
    so one rule alone can never carry more than half the `RULE_HITS` weight."""
    assert max(spec.contribution for spec in RULE_SPECS) <= 50.0


def test_fan_out_is_the_cheapest_rule():
    """R003 is the one single-condition rule a documented legitimate cohort trips by design, so it
    must stay the least expensive of the seven. See `docs/13-KNOWN-FALSE-POSITIVES.md`."""
    fan_out = next(spec for spec in RULE_SPECS if spec.code == R003)
    assert fan_out.contribution == min(spec.contribution for spec in RULE_SPECS)


def test_contribution_out_of_range_is_a_startup_error():
    with pytest.raises(ConfigurationError, match="contribution"):
        RuleSpec("R999", "Impossible", Severity.LOW, 140.0)


def test_duplicate_rule_codes_are_rejected():
    spec = RuleSpec(R001, "Velocity spike", Severity.MEDIUM, 25.0)
    with pytest.raises(ConfigurationError, match="unique"):
        DetectionConfig(specs=(spec, spec))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"velocity_multiple": 1.0},
        {"fan_in_min_counterparties": 0},
        {"fan_in_min_credit_inr": Decimal("0")},
        {"pass_through_max_balance_share": 1.5},
    ],
)
def test_nonsense_thresholds_fail_at_construction(kwargs):
    with pytest.raises(ConfigurationError):
        RuleThresholds(**kwargs)


def test_contamination_must_be_a_share():
    with pytest.raises(ConfigurationError, match="contamination"):
        IsolationForestConfig(contamination=0.9)


# --- evidence -------------------------------------------------------------------------------------


def test_evidence_rejects_a_row_from_another_account():
    with pytest.raises(DomainValidationError, match="joined on the wrong key"):
        rules.AccountEvidence(
            account_id="ACC00000001",
            rows={FeatureWindow.D7: make_row(account_id="ACC00000002")},
            closing_balance=Decimal("1"),
            shared_device_accounts=0,
            shared_device_customers=0,
        )


def test_evidence_rejects_more_customers_than_accounts():
    with pytest.raises(DomainValidationError, match="one owner"):
        make_evidence(shared_device_accounts=2, shared_device_customers=5)


def test_missing_window_raises_rather_than_skipping_the_rule():
    evidence = rules.AccountEvidence(
        account_id="ACC00000001",
        rows={FeatureWindow.D7: make_row(FeatureWindow.D7)},
        closing_balance=Decimal("1"),
        shared_device_accounts=0,
        shared_device_customers=0,
    )
    with pytest.raises(DomainValidationError, match="no 30D feature row"):
        rules.r005_dormant_reactivation(evidence, TH)


# --- R001 velocity spike --------------------------------------------------------------------------


def test_implied_expected_count_inverts_the_feature_formula():
    for txn_count, multiple in ((10, 5.0), (40, 2.0), (13, 7.5)):
        row = velocity_row(txn_count, multiple)
        recovered = rules.implied_expected_count(row.txn_count, row.velocity_zscore)
        assert recovered == pytest.approx(txn_count / multiple, rel=1e-9)


def test_implied_expected_count_is_zero_for_an_empty_window():
    assert rules.implied_expected_count(0, 0.0) == 0.0


def test_r001_fires_at_the_configured_multiple():
    row = velocity_row(TH.velocity_min_txn_count, TH.velocity_multiple)
    finding = rules.r001_velocity_spike(make_evidence(d7=row), TH)
    assert finding is not None
    assert finding.evidence["multiple"] == pytest.approx(TH.velocity_multiple, rel=1e-6)


def test_r001_ignores_a_big_multiple_on_too_few_transactions():
    """The second half of the spec. Two transactions against an expected 0.3 is a 6x multiple and
    is noise, which is why the rule needs a floor on the count as well."""
    row = velocity_row(TH.velocity_min_txn_count - 1, TH.velocity_multiple * 2)
    assert rules.r001_velocity_spike(make_evidence(d7=row), TH) is None


def test_r001_ignores_a_busy_account_at_its_own_normal_rate():
    row = velocity_row(200, 1.0)
    assert rules.r001_velocity_spike(make_evidence(d7=row), TH) is None


# --- R002 fan-in concentration --------------------------------------------------------------------


def test_r002_needs_both_conditions():
    payers = TH.fan_in_min_counterparties
    amount = TH.fan_in_min_credit_inr
    both = make_row(
        distinct_credit_counterparties=payers,
        total_credit=amount,
        total_debit=Decimal("0.00"),
        balance_delta=amount,
        pass_through_ratio=0.0,
    )
    assert rules.r002_fan_in_concentration(make_evidence(d7=both), TH) is not None

    # The merchant shape: plenty of payers, not enough money. This is the hard negative the second
    # condition exists for.
    small = make_row(
        distinct_credit_counterparties=payers + 10,
        total_credit=amount - Decimal("1.00"),
        total_debit=Decimal("0.00"),
        balance_delta=amount - Decimal("1.00"),
        pass_through_ratio=0.0,
    )
    assert rules.r002_fan_in_concentration(make_evidence(d7=small), TH) is None

    few = make_row(
        distinct_credit_counterparties=payers - 1,
        total_credit=amount * 10,
        total_debit=Decimal("0.00"),
        balance_delta=amount * 10,
        pass_through_ratio=0.0,
    )
    assert rules.r002_fan_in_concentration(make_evidence(d7=few), TH) is None


# --- R003 fan-out dispersal -----------------------------------------------------------------------


def test_r003_fires_on_the_recipient_count_alone():
    row = make_row(distinct_debit_counterparties=TH.fan_out_min_counterparties)
    finding = rules.r003_fan_out_dispersal(make_evidence(d7=row), TH)
    assert finding is not None
    below = make_row(distinct_debit_counterparties=TH.fan_out_min_counterparties - 1)
    assert rules.r003_fan_out_dispersal(make_evidence(d7=below), TH) is None


def test_r003_fires_on_a_payroll_account_and_that_is_expected():
    """`docs/13-KNOWN-FALSE-POSITIVES.md`: a payroll account pays 20 to 28 people. The rule is
    correct and this is the documented cost of it — the mitigation is the contribution, asserted in
    `test_fan_out_is_the_cheapest_rule`, not a weaker condition."""
    payroll = make_row(
        txn_count=28,
        debit_count=28,
        distinct_debit_counterparties=28,
        total_debit=Decimal("1400000.00"),
        total_credit=Decimal("1400000.00"),
        balance_delta=Decimal("0.00"),
        pass_through_ratio=1.0,
    )
    hits = rules.evaluate(make_evidence(d7=payroll))
    assert R003 in codes(hits)


# --- R004 structuring -----------------------------------------------------------------------------


def test_r004_fires_on_the_band_count():
    row = make_row(structuring_band_count=TH.structuring_min_band_count)
    assert rules.r004_structuring(make_evidence(d7=row), TH) is not None
    below = make_row(structuring_band_count=TH.structuring_min_band_count - 1)
    assert rules.r004_structuring(make_evidence(d7=below), TH) is None


def test_r004_reads_the_seven_day_window_which_is_its_specified_window():
    row = make_row(FeatureWindow.D7, structuring_band_count=9)
    finding = rules.r004_structuring(make_evidence(d7=row), TH)
    assert finding is not None
    assert finding.row.window is FeatureWindow.D7


# --- R005 dormant reactivation --------------------------------------------------------------------


def test_r005_needs_the_silence_and_the_money():
    idle = TH.dormant_min_idle_days
    amount = TH.dormant_min_throughput_inr
    woken = make_row(
        FeatureWindow.D30,
        days_since_last_activity=idle,
        total_credit=amount,
        total_debit=Decimal("0.00"),
        balance_delta=amount,
        pass_through_ratio=0.0,
        txn_count=1,
        credit_count=1,
        debit_count=0,
    )
    assert rules.r005_dormant_reactivation(make_evidence(d30=woken), TH) is not None

    quiet = make_row(
        FeatureWindow.D30,
        days_since_last_activity=idle,
        total_credit=amount - Decimal("1.00"),
        total_debit=Decimal("0.00"),
        balance_delta=amount - Decimal("1.00"),
        pass_through_ratio=0.0,
    )
    assert rules.r005_dormant_reactivation(make_evidence(d30=quiet), TH) is None

    recent = make_row(
        FeatureWindow.D30,
        days_since_last_activity=idle - 1,
        total_credit=amount * 10,
        total_debit=Decimal("0.00"),
        balance_delta=amount * 10,
        pass_through_ratio=0.0,
    )
    assert rules.r005_dormant_reactivation(make_evidence(d30=recent), TH) is None


def test_r005_reads_the_thirty_day_window():
    """Structural, not a tuning choice: on a 7-day row `days_since_last_activity` is measured to the
    last activity before *that* window, so a reactivation inside the last month reads as a short gap
    and the rule could never fire on the window that contains it."""
    woken = make_row(
        FeatureWindow.D30,
        days_since_last_activity=400,
        total_credit=Decimal("900000.00"),
        total_debit=Decimal("0.00"),
        balance_delta=Decimal("900000.00"),
        pass_through_ratio=0.0,
    )
    short = make_row(FeatureWindow.D7, days_since_last_activity=3)
    finding = rules.r005_dormant_reactivation(make_evidence(d7=short, d30=woken), TH)
    assert finding is not None
    assert finding.row.window is FeatureWindow.D30


# --- R006 pass-through ----------------------------------------------------------------------------


def _swept(ratio: float) -> AccountFeatures:
    credit = Decimal("500000.00")
    debit = (credit * Decimal(str(ratio))).quantize(Decimal("0.01"))
    return make_row(
        total_credit=credit,
        total_debit=debit,
        balance_delta=credit - debit,
        pass_through_ratio=float(debit / credit),
    )


def test_r006_needs_the_ratio_and_an_empty_balance():
    row = _swept(0.96)
    ceiling = row.total_credit * Decimal(str(TH.pass_through_max_balance_share))
    swept = make_evidence(d7=row, closing_balance=ceiling - Decimal("1.00"))
    assert rules.r006_pass_through(swept, TH) is not None

    # The trader: forwards almost everything, but is sitting on working capital.
    holding = make_evidence(d7=row, closing_balance=ceiling)
    assert rules.r006_pass_through(holding, TH) is None


def test_r006_ignores_an_account_that_retained_a_real_share():
    row = _swept(TH.pass_through_min_ratio - 0.05)
    assert rules.r006_pass_through(make_evidence(d7=row, closing_balance=Decimal("0")), TH) is None


def test_r006_ignores_an_account_with_no_credits_at_all():
    """`compute_pass_through_ratio` is 0.0 with no inflow, so a division here would be meaningless.
    The drain-with-no-inflow case is the risk scorer's pass-through component, not R006's."""
    row = make_row(
        total_credit=Decimal("0.00"),
        total_debit=Decimal("900000.00"),
        balance_delta=Decimal("-900000.00"),
        pass_through_ratio=0.0,
    )
    assert rules.r006_pass_through(make_evidence(d7=row, closing_balance=Decimal("0")), TH) is None


# --- R007 device concentration --------------------------------------------------------------------


def test_r007_fires_on_other_customers_accounts():
    ring = make_evidence(
        shared_device_accounts=TH.device_min_other_accounts,
        shared_device_customers=TH.device_min_other_accounts,
    )
    assert rules.r007_device_concentration(ring, TH) is not None

    household = make_evidence(
        shared_device_accounts=TH.device_min_other_accounts - 1,
        shared_device_customers=TH.device_min_other_accounts - 1,
    )
    assert rules.r007_device_concentration(household, TH) is None


def test_r007_does_not_fire_on_an_unshared_device():
    assert rules.r007_device_concentration(make_evidence(), TH) is None


# --- the engine -----------------------------------------------------------------------------------


def test_a_quiet_account_produces_no_hits():
    assert rules.evaluate(make_evidence()) == ()


def test_hits_come_back_in_rule_code_order():
    collection = make_row(
        txn_count=22,
        credit_count=19,
        debit_count=3,
        distinct_credit_counterparties=17,
        distinct_debit_counterparties=2,
        total_credit=Decimal("410000.00"),
        total_debit=Decimal("394000.00"),
        balance_delta=Decimal("16000.00"),
        pass_through_ratio=float(Decimal("394000.00") / Decimal("410000.00")),
        structuring_band_count=4,
    )
    hits = rules.evaluate(make_evidence(d7=collection, closing_balance=Decimal("1000.00")))
    assert codes(hits) == sorted(codes(hits))
    assert {R002, R004, R006} <= set(codes(hits))


def test_a_hit_states_the_window_the_rule_actually_read():
    woken = make_row(
        FeatureWindow.D30,
        days_since_last_activity=300,
        total_credit=Decimal("900000.00"),
        total_debit=Decimal("0.00"),
        balance_delta=Decimal("900000.00"),
        pass_through_ratio=0.0,
    )
    (hit,) = rules.evaluate(make_evidence(d30=woken))
    assert hit.rule_code == R005
    assert hit.window_end == AS_OF
    assert hit.window_start == AS_OF - timedelta(days=30)


def test_a_configured_rule_with_no_implementation_fails_loudly():
    config = DetectionConfig(specs=(RuleSpec("R999", "Ghost", Severity.LOW, 10.0),))
    with pytest.raises(DomainValidationError, match="no implementation"):
        rules.evaluate(make_evidence(), config)


def test_every_hit_carries_a_reason_an_analyst_can_read():
    hits = rules.evaluate(
        make_evidence(
            d7=make_row(
                distinct_credit_counterparties=20,
                distinct_debit_counterparties=12,
                total_credit=Decimal("900000.00"),
                total_debit=Decimal("880000.00"),
                balance_delta=Decimal("20000.00"),
                pass_through_ratio=float(Decimal("880000.00") / Decimal("900000.00")),
                structuring_band_count=5,
                txn_count=30,
            ),
            closing_balance=Decimal("500.00"),
            shared_device_accounts=6,
            shared_device_customers=5,
        )
    )
    assert len(hits) >= 4
    for hit in hits:
        assert hit.detail.strip()
        assert hit.evidence


# --- the rule engine reaching the score -----------------------------------------------------------


def test_rule_hits_move_the_deterministic_score():
    """The wiring T6.1 exists to complete: before the engine, `rule_hits` was always `None` and the
    seventh component reported `not gathered`."""
    row = make_row(structuring_band_count=9)
    hits = rules.evaluate(make_evidence(d7=row))
    assert codes(hits) == [R004]

    blind = assess_risk(RiskEvidence(account_id=row.account_id))
    seeing = assess_risk(RiskEvidence(account_id=row.account_id, rule_hits=hits))
    assert blind.score == 0.0
    # 45 contribution, ramped 0..100, times the 20-point RULE_HITS weight.
    assert seeing.score == pytest.approx(9.0)
    assert "R004" in seeing.components[-1].detail


# --- the anomaly model ----------------------------------------------------------------------------


def _matrix(rows: int = 60) -> anomaly.FeatureMatrix:
    generator = np.random.default_rng(7)
    values = generator.normal(size=(rows, len(anomaly.FEATURE_COLUMNS)))
    values[0] = 40.0  # one unmistakable outlier
    return anomaly.FeatureMatrix(
        account_ids=tuple(f"ACC{index:08d}" for index in range(rows)),
        columns=anomaly.FEATURE_COLUMNS,
        values=values,
    )


def test_the_matrix_carries_no_identifier_or_label_column():
    """Fitting on `account_id` would let the model memorise the book, and a label column would
    invalidate every Stage 7 metric. Neither is in the feature list."""
    assert "account_id" not in anomaly.FEATURE_COLUMNS
    assert not {"typology", "is_mule", "is_hard_negative"} & set(anomaly.FEATURE_COLUMNS)


def test_a_nan_in_the_matrix_fails_before_the_estimator_sees_it():
    values = np.zeros((3, len(anomaly.FEATURE_COLUMNS)))
    values[1][0] = float("nan")
    with pytest.raises(DomainValidationError, match="NaN"):
        anomaly.FeatureMatrix(("A", "B", "C"), anomaly.FEATURE_COLUMNS, values)


def test_the_fit_is_reproducible_from_the_seed():
    matrix = _matrix()
    first = anomaly.score(anomaly.fit(matrix), matrix)
    second = anomaly.score(anomaly.fit(matrix), matrix)
    assert first == second


def test_the_outlier_scores_above_every_ordinary_row():
    matrix = _matrix()
    scores = anomaly.score(anomaly.fit(matrix), matrix)
    outlier = scores["ACC00000000"]
    assert outlier == max(scores.values())
    assert outlier > 50.0


def test_the_score_map_is_bounded_and_puts_the_boundary_at_fifty():
    mapped = anomaly.to_anomaly_score(np.array([-2.0, -0.4, 0.0, 0.4, 2.0]))
    assert mapped == [100.0, 70.0, 50.0, 30.0, 0.0]


def test_scoring_across_a_column_reorder_is_refused():
    matrix = _matrix()
    forest = anomaly.fit(matrix)
    shuffled = anomaly.FeatureMatrix(
        matrix.account_ids, tuple(reversed(anomaly.FEATURE_COLUMNS)), matrix.values
    )
    with pytest.raises(DomainValidationError, match="column reorder"):
        anomaly.score(forest, shuffled)


def test_the_model_round_trips_through_the_artefact(tmp_path):
    matrix = _matrix()
    forest = anomaly.fit(matrix)
    path = anomaly.save(forest, tmp_path / "models")
    assert path.exists()
    reloaded = anomaly.load(tmp_path / "models")
    assert reloaded.columns == forest.columns
    assert anomaly.score(reloaded, matrix) == anomaly.score(forest, matrix)


def test_loading_a_missing_model_names_the_command_that_writes_one(tmp_path):
    with pytest.raises(DomainValidationError, match="make detect"):
        anomaly.load(tmp_path / "never-fitted")


def test_loading_an_artefact_from_another_version_is_refused(tmp_path):
    """A stored `anomaly_score` is only comparable to one produced by the same feature list and the
    same score mapping. Silently reusing a stale artefact would make two batches incomparable with
    no error anywhere."""
    matrix = _matrix()
    stale = anomaly.PersistedForest(
        model=anomaly.fit(matrix).model,
        columns=anomaly.FEATURE_COLUMNS,
        config=IsolationForestConfig(),
        version="iforest-v0",
        fitted_at=AS_OF,
        fitted_rows=matrix.values.shape[0],
    )
    root = tmp_path / "models"
    anomaly.save(stale, root)
    with pytest.raises(DomainValidationError, match="not comparable"):
        anomaly.load(root)


# --- alert construction ---------------------------------------------------------------------------


def test_alert_ids_are_deterministic_so_a_rerun_upserts():
    row = make_row(structuring_band_count=6)
    evidence = (make_evidence(d7=row),)
    hits = {row.account_id: rules.evaluate(evidence[0])}
    scores = {row.account_id: 12.0}
    decisions = {row.account_id: 0.76}
    first = detect.build_alerts(evidence, hits, scores, decisions, as_of=AS_OF)
    second = detect.build_alerts(evidence, hits, scores, decisions, as_of=AS_OF)
    assert [a.alert_id for a in first] == [a.alert_id for a in second]
    assert first[0].alert_id == f"ALRT-{row.account_id}-{R004}-20260804"


def test_every_alert_carries_a_non_empty_reason():
    """T6.3's acceptance criterion, asserted directly."""
    row = make_row(structuring_band_count=6, distinct_debit_counterparties=11)
    evidence = (make_evidence(d7=row),)
    alerts = detect.build_alerts(
        evidence,
        {row.account_id: rules.evaluate(evidence[0])},
        {row.account_id: 61.0},
        {row.account_id: -0.22},
        as_of=AS_OF,
    )
    assert len(alerts) == 2
    for alert in alerts:
        assert alert.trigger_evidence["reason"].strip()
        assert alert.status is AlertStatus.OPEN
        assert alert.triggered_at == AS_OF
        assert alert.anomaly_score == 61.0


def test_an_outlier_with_no_rule_hit_still_gets_one_alert():
    evidence = (make_evidence(),)
    alerts = detect.build_alerts(
        evidence, {}, {"ACC00000001": 92.0}, {"ACC00000001": -0.84}, as_of=AS_OF
    )
    assert [a.rule_code for a in alerts] == [ML_SPEC.code]
    assert alerts[0].rule_score == 0.0
    assert "no rule fired" in alerts[0].trigger_evidence["reason"]


def test_an_ordinary_account_produces_no_alert_at_all():
    evidence = (make_evidence(),)
    assert (
        detect.build_alerts(evidence, {}, {"ACC00000001": 31.0}, {"ACC00000001": 0.38}, as_of=AS_OF)
        == ()
    )


def test_a_rule_hit_suppresses_the_model_only_alert():
    """The model-only row exists to add reach, not to duplicate a rule that already fired. The
    anomaly score still rides on the rule's alert."""
    row = make_row(structuring_band_count=6)
    evidence = (make_evidence(d7=row),)
    alerts = detect.build_alerts(
        evidence,
        {row.account_id: rules.evaluate(evidence[0])},
        {row.account_id: 99.0},
        {row.account_id: -0.98},
        as_of=AS_OF,
    )
    assert [a.rule_code for a in alerts] == [R004]
    assert alerts[0].anomaly_score == 99.0


def test_an_account_with_no_facts_is_a_join_failure_not_a_zero():
    with pytest.raises(DomainValidationError, match="cross-account facts"):
        detect.build_evidence([make_row()], {})
