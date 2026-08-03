"""Tests for the domain layer — entities, enums, and value objects.

These do not check that a dataclass stores what it was handed; the language guarantees that. They
check the four properties the layer actually promises, each of which would be a real defect if it
silently stopped holding:

1. **Immutability.** A loaded row is a fact about the past. If a service can mutate one, it can
   produce a value that agrees with nothing in the database.
2. **Exact money.** Every amount is `Decimal`. The float test below is the demonstration of why:
   the same arithmetic in binary floating point is already wrong after ten additions, and a batch
   contains 200,000 of them.
3. **Enum round-tripping.** These values are written to Postgres text columns and JSON payloads as
   plain strings and read back. A member that does not survive `str()` and reconstruction would
   corrupt every stored row of that type.
4. **The constructor guards.** Naive timestamps, non-positive amounts, a confidence on the wrong
   scale, a score with no explanation, a mule labelled as a hard negative — each is rejected at
   construction rather than one layer further on.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

import pytest

from sentinel.domain import entities, enums, value_objects
from sentinel.domain.entities import (
    Account,
    AccountFeatures,
    Alert,
    Device,
    DqCheckResult,
    Investigation,
    ToolCall,
    Transaction,
    TypologyLabel,
)
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    AlertStatus,
    Channel,
    Direction,
    DqCheckStatus,
    DqSeverity,
    FeatureWindow,
    InvestigationStatus,
    Recommendation,
    ScoreDecision,
    Severity,
    TransactionStatus,
    Typology,
)
from sentinel.domain.exceptions import DomainValidationError, SentinelError
from sentinel.domain.value_objects import (
    FeatureVector,
    FundFlowPath,
    RingFinding,
    RuleHit,
    ScoreComponent,
    ScoreResult,
)

TS = datetime(2026, 8, 3, 9, 30, tzinfo=UTC)
NAIVE_TS = datetime(2026, 8, 3, 9, 30)  # deliberately naive: the bug this suite asserts against


# --- factories ---------------------------------------------------------------------------------
# Every field is positional-by-name and required, so a test that cares about one field still has to
# supply the rest. These keep that noise out of the tests themselves.


def make_transaction(**overrides: object) -> Transaction:
    fields: dict[str, object] = {
        "txn_id": "TXN0000000001",
        "txn_ts": TS,
        "from_account": "ACC00000001",
        "to_account": "ACC00000002",
        "counterparty_handle": "9876543210@ybl",
        "counterparty_bank": "YESB",
        "amount": Decimal("48500.00"),
        "currency": "INR",
        "channel": Channel.UPI,
        "direction": Direction.DEBIT,
        "status": TransactionStatus.SUCCESS,
        "device_id": "DEV0001842",
        "ip_address": "203.0.113.7",
        "narration": "UPI/P2P/482913",
        "txn_date": date(2026, 8, 3),
        "ingested_at": TS,
        "batch_id": "BATCH20260803",
    }
    fields.update(overrides)
    return Transaction(**fields)  # type: ignore[arg-type]


def make_account(**overrides: object) -> Account:
    fields: dict[str, object] = {
        "account_id": "ACC00042917",
        "customer_id": "CUS00019283",
        "account_type": AccountType.SAVINGS,
        "opened_on": date(2026, 5, 12),
        "status": AccountStatus.ACTIVE,
        "branch_code": "BR0417",
        "ifsc": "IDFB0040101",
        "current_balance": Decimal("12400.55"),
        "last_activity_on": date(2026, 8, 2),
        "is_flagged": False,
        "created_at": TS,
    }
    fields.update(overrides)
    return Account(**fields)  # type: ignore[arg-type]


def make_investigation(**overrides: object) -> Investigation:
    fields: dict[str, object] = {
        "investigation_id": "INV20260803000188",
        "alert_id": "ALT20260803000417",
        "agent_recommendation": Recommendation.ESCALATE,
        "agent_confidence": 0.88,
        "computed_risk_score": 84.5,
        "detected_typology": Typology.FAN_IN_OUT,
        "hypotheses": ({"statement": "Collection mule.", "confidence": 0.88},),
        "narrative": "ACC00042917 received 14 credits ...",
        "requires_human": True,
        "llm_model": "deepseek/deepseek-v4-pro",
        "tokens_in": 87420,
        "tokens_out": 11380,
        "cost_usd": Decimal("0.048"),
        "latency_ms": 38210,
        "tool_call_count": 3,
        "status": InvestigationStatus.COMPLETE,
        "started_at": TS,
        "completed_at": TS,
    }
    fields.update(overrides)
    return Investigation(**fields)  # type: ignore[arg-type]


def make_feature_vector(**overrides: object) -> FeatureVector:
    fields: dict[str, object] = {
        "account_id": "ACC00042917",
        "window": FeatureWindow.H24,
        "as_of": TS,
        "txn_count": 17,
        "total_credit": Decimal("284000.00"),
        "total_debit": Decimal("266960.00"),
        "distinct_credit_counterparties": 14,
        "distinct_debit_counterparties": 2,
        "max_single_txn": Decimal("98000.00"),
        "avg_txn_amount": Decimal("32350.59"),
        "night_txn_ratio": 0.41,
        "new_counterparty_ratio": 0.93,
        "distinct_devices": 1,
        "distinct_ips": 3,
        "days_since_last_activity": 0,
        "velocity_zscore": 6.2,
    }
    fields.update(overrides)
    return FeatureVector(**fields)  # type: ignore[arg-type]


def make_features(**overrides: object) -> AccountFeatures:
    """An `account_features` row. `pass_through_ratio` is stored here (unlike on `FeatureVector`),
    so it must agree with total_debit / total_credit."""
    fields: dict[str, object] = {
        "account_id": "ACC00042917",
        "window": FeatureWindow.H24,
        "as_of": TS,
        "txn_count": 17,
        "total_credit": Decimal("284000.00"),
        "total_debit": Decimal("266960.00"),
        "distinct_credit_counterparties": 14,
        "distinct_debit_counterparties": 2,
        "max_single_txn": Decimal("98000.00"),
        "avg_txn_amount": Decimal("32350.59"),
        "pass_through_ratio": 0.94,
        "night_txn_ratio": 0.41,
        "new_counterparty_ratio": 0.93,
        "distinct_devices": 1,
        "distinct_ips": 3,
        "days_since_last_activity": 0,
        "velocity_zscore": 6.2,
    }
    fields.update(overrides)
    return AccountFeatures(**fields)  # type: ignore[arg-type]


def make_device(**overrides: object) -> Device:
    fields: dict[str, object] = {
        "device_id": "DEV0001842",
        "fingerprint": "fp_9f2c41ab",
        "os_family": "Android",
        "app_version": "8.4.1",
        "first_seen_at": TS,
        "last_seen_at": TS,
    }
    fields.update(overrides)
    return Device(**fields)  # type: ignore[arg-type]


def make_path(**overrides: object) -> FundFlowPath:
    fields: dict[str, object] = {
        "hop_accounts": ("ACC1", "ACC2", "ACC3"),
        "hops": 2,
        "path_value": Decimal("180000.00"),
        "bottleneck_amount": Decimal("42000.00"),
        "flow_started": TS,
        "flow_ended": TS,
    }
    fields.update(overrides)
    return FundFlowPath(**fields)  # type: ignore[arg-type]


def domain_dataclasses() -> list[type]:
    """Every dataclass declared in `entities` and `value_objects`.

    Collected by reflection rather than listed, so an entity added later is covered by the
    frozen/slots invariant test without anyone remembering to add it here.
    """
    found: list[type] = []
    for module in (entities, value_objects):
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__:
                found.append(obj)
    return found


# --- immutability ------------------------------------------------------------------------------


def test_mutating_a_loaded_row_raises():
    txn = make_transaction()
    with pytest.raises(FrozenInstanceError):
        txn.amount = Decimal("1.00")  # type: ignore[misc]


def test_mutating_a_value_object_raises():
    vector = make_feature_vector()
    with pytest.raises(FrozenInstanceError):
        vector.total_debit = Decimal("0")  # type: ignore[misc]


def test_deleting_a_field_raises():
    with pytest.raises(FrozenInstanceError):
        del make_account().current_balance  # type: ignore[misc]


def test_every_domain_dataclass_is_frozen_and_slotted():
    """The two hard constraints of the layer, asserted for all of them at once.

    Slots are not cosmetic: ~200k `Transaction` objects exist simultaneously during a batch, and
    the per-instance `__dict__` slots remove is the difference between fitting in memory and not.
    """
    classes = domain_dataclasses()
    # Exact, not a floor: 12 tables + 16 value objects. A deleted entity must fail the suite rather
    # than quietly satisfy an inequality. T1.2 added the eight shapes the ports return —
    # AccountProfile, FundFlowResult, IdentifierLink, SharedIdentifierResult, GraphNode, GraphEdge,
    # Subgraph, LLMCompletion — so this count moves with the port layer, deliberately.
    assert len(classes) == 28, f"expected 12 entities + 16 value objects, found {classes}"
    for cls in classes:
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} is not frozen"
        assert "__slots__" in cls.__dict__, f"{cls.__name__} is not slotted"


def test_slotted_instances_have_no_instance_dict():
    """The memory claim, verified directly: no `__dict__` means no per-instance hash table, and
    also means a typo'd attribute cannot be silently attached to a row."""
    txn = make_transaction()
    assert not hasattr(txn, "__dict__")
    # A misspelled field cannot be silently attached either. The error type is the stdlib's: for a
    # name that is not a declared field, `frozen=True` + `slots=True` surfaces a `TypeError` from
    # the generated `__setattr__` rather than `FrozenInstanceError`. Accepting all three keeps the
    # test about the behaviour (the write is refused) rather than about CPython's wording.
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        txn.amonut = Decimal("1")  # type: ignore[attr-defined]


# --- money is exact ----------------------------------------------------------------------------


def test_summing_decimal_amounts_is_exact_where_float_is_not():
    """The reason every amount in this codebase is `Decimal`.

    Ten additions of 0.10 are already wrong in binary floating point. A batch performs 200,000 of
    them, and a drifting reconciliation total in a bank is a defect, not a rounding detail.
    """
    amounts = [make_transaction(amount=Decimal("0.10")).amount for _ in range(10)]
    assert sum(amounts) == Decimal("1.00")

    # The same arithmetic a float version of this code would have done. Note the accumulator loop
    # rather than `sum()`: CPython 3.12's `sum()` applies Neumaier compensated summation to floats
    # and hides the drift, which real accumulation code (`total += row.amount`) does not.
    float_total = 0.0
    for _ in range(10):
        float_total += 0.1
    assert float_total != 1.0


def test_large_scale_accumulation_stays_exact():
    """Batch-scale accumulation: a thousand small credits must total to the paisa."""
    total = sum(
        (make_transaction(amount=Decimal("0.01")).amount for _ in range(1000)),
        start=Decimal("0"),
    )
    assert total == Decimal("10.00")
    assert str(total) == "10.00"  # scale is preserved, not just the value


def test_no_money_field_is_typed_float():
    """Guards the rule for entities added later: if it is an amount, a balance, or a cost, it is
    `Decimal`. Ratios and scores are legitimately `float` and are excluded by name."""
    money_words = ("amount", "credit", "debit", "balance", "cost", "income", "value", "txn")
    for cls in domain_dataclasses():
        for field in dataclasses.fields(cls):
            looks_like_money = any(word in field.name for word in money_words)
            is_ratio_or_count = any(
                word in field.name for word in ("ratio", "count", "zscore", "score")
            )
            if looks_like_money and not is_ratio_or_count:
                assert "float" not in str(field.type), f"{cls.__name__}.{field.name} is a float"


# --- enums round-trip --------------------------------------------------------------------------


def test_every_enum_member_survives_a_string_round_trip():
    """These values are stored as plain text in Postgres and as plain strings in JSON. A member
    that does not reconstruct from `str(member)` would corrupt every stored row of its type."""
    enum_classes = [
        obj
        for _, obj in inspect.getmembers(enums, inspect.isclass)
        if issubclass(obj, StrEnum) and obj is not StrEnum
    ]
    # Exact, so deleting an enum fails here rather than passing an inequality. T1.2 added
    # `MuleSignature`, the fan-shape verdict the graph port returns.
    names = sorted(cls.__name__ for cls in enum_classes)
    assert len(enum_classes) == 20, f"expected 20 enums, found {names}"
    for cls in enum_classes:
        for member in cls:
            assert cls(str(member)) is member
            assert f"{member}" == member.value


def test_members_are_usable_directly_as_strings():
    """`StrEnum` is what lets an adapter pass a member where a driver wants text, with no
    `.value` and no converter — and never write the repr into a column by mistake."""
    assert Channel.UPI == "UPI"
    assert f"feat:ACC1:{FeatureWindow.H24}" == "feat:ACC1:24H"
    assert sorted({Severity.HIGH, "HIGH"}) == ["HIGH"]


def test_window_names_differ_from_values_where_the_value_is_not_an_identifier():
    assert FeatureWindow.H24.value == "24H"
    assert FeatureWindow("7D") is FeatureWindow.D7


def test_agreement_is_a_plain_equality_between_agent_and_human():
    """The reason both sides use one `Recommendation` enum: `agreed_with_agent` stays trivial and
    the agreement rate is a real metric rather than a string comparison across two vocabularies."""
    assert Recommendation("ESCALATE") is Recommendation.ESCALATE
    assert Recommendation.ESCALATE != Recommendation.MONITOR


# --- timezone guard ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    ["txn_ts", "ingested_at"],
)
def test_naive_transaction_timestamps_are_rejected(field_name):
    """A naive timestamp has lost its offset, so the night-hour feature (00:00-05:00 is the mule
    dispersal window) would be computed against the wrong local time."""
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        make_transaction(**{field_name: NAIVE_TS})


def test_naive_timestamps_are_rejected_on_value_objects_too():
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        make_feature_vector(as_of=NAIVE_TS)


def test_optional_timestamp_may_be_none_but_not_naive():
    assert make_investigation(completed_at=None).completed_at is None
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        make_investigation(completed_at=NAIVE_TS)


def test_a_domain_validation_error_is_catchable_both_ways():
    """It subclasses `ValueError` for a parsing loop and `SentinelError` for the API's single
    error handler. Both have to keep working."""
    with pytest.raises(ValueError):
        make_transaction(txn_ts=NAIVE_TS)
    with pytest.raises(SentinelError):
        make_transaction(txn_ts=NAIVE_TS)


# --- ordering guards ---------------------------------------------------------------------------
# Reversed instant pairs are silent poison rather than loud errors: they produce negative dwell
# times and negative latencies that drag a reported mean down without anything failing.


def test_a_device_cannot_be_last_seen_before_it_was_first_seen():
    """A negative dwell time breaks every device-age feature computed from this pair."""
    with pytest.raises(DomainValidationError, match="precedes"):
        make_device(last_seen_at=datetime(2026, 8, 2, tzinfo=UTC))


def test_an_investigation_cannot_complete_before_it_started():
    """Mean investigation latency is reported from these two fields."""
    with pytest.raises(DomainValidationError, match="precedes"):
        make_investigation(completed_at=datetime(2026, 8, 2, tzinfo=UTC))


def test_an_account_cannot_have_activity_before_it_was_opened():
    with pytest.raises(DomainValidationError, match="precedes"):
        make_account(last_activity_on=date(2026, 5, 11))


def test_a_fund_flow_path_cannot_end_before_it_started():
    with pytest.raises(DomainValidationError, match="precedes"):
        make_path(flow_ended=datetime(2026, 8, 2, tzinfo=UTC))


def test_equal_instants_are_allowed_and_a_missing_later_instant_is_skipped():
    """Equality is legitimate — a single-transaction device is first and last seen at the same
    moment — and `None` means "not yet", which is not an ordering violation."""
    assert make_device(first_seen_at=TS, last_seen_at=TS).last_seen_at == TS
    assert make_investigation(completed_at=None).completed_at is None
    assert make_account(last_activity_on=None).last_activity_on is None


# --- amount guards -----------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1"), Decimal("-0.01")])
def test_non_positive_transaction_amounts_are_rejected(amount):
    """Direction carries the sign, so the amount is always a magnitude. A negative amount is a
    reversal that belongs in `status`."""
    with pytest.raises(DomainValidationError, match="greater than zero"):
        make_transaction(amount=amount)


def test_negative_window_totals_are_rejected():
    with pytest.raises(DomainValidationError, match="must not be negative"):
        make_feature_vector(total_credit=Decimal("-1"))


def test_negative_cost_is_rejected():
    with pytest.raises(DomainValidationError, match="cost_usd"):
        make_investigation(cost_usd=Decimal("-0.01"))


def test_an_overdrawn_balance_is_allowed():
    """Deliberately not guarded — a CURRENT account can sit overdrawn, so a negative balance is
    data rather than an error."""
    assert make_account(current_balance=Decimal("-8400.00")).current_balance < 0


# --- range guards ------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [88.0, 1.5, -0.1])
def test_confidence_on_the_wrong_scale_is_rejected(confidence):
    """The human gate escalates when confidence is below 0.70. A confidence of `88` where `0.88`
    was meant clears that test and lets a case self-close — a governance failure, not a typo."""
    with pytest.raises(DomainValidationError, match="agent_confidence"):
        make_investigation(agent_confidence=confidence)


def test_confidence_bounds_are_inclusive():
    assert make_investigation(agent_confidence=0.0).agent_confidence == 0.0
    assert make_investigation(agent_confidence=1.0).agent_confidence == 1.0


def test_risk_score_outside_zero_to_hundred_is_rejected():
    with pytest.raises(DomainValidationError, match="computed_risk_score"):
        make_investigation(computed_risk_score=140.0)


def test_ratio_features_are_bounded_but_pass_through_is_not():
    """`night_txn_ratio` is a share of transactions and cannot exceed 1. `pass_through_ratio` can,
    because an account may disperse more than it received by spending an opening balance — and that
    is real signal, so it must not be clamped away."""
    with pytest.raises(DomainValidationError, match="night_txn_ratio"):
        make_feature_vector(night_txn_ratio=1.4)

    vector = make_feature_vector(total_credit=Decimal("100"), total_debit=Decimal("150"))
    assert vector.pass_through_ratio == 1.5


# --- computed behaviour ------------------------------------------------------------------------


def test_pass_through_ratio_matches_the_feature_sql():
    vector = make_feature_vector(total_credit=Decimal("284000"), total_debit=Decimal("266960"))
    assert round(vector.pass_through_ratio, 4) == 0.94


def test_a_stored_pass_through_ratio_that_contradicts_its_own_totals_is_rejected():
    """The anti-drift check. `AccountFeatures` stores the ratio (schema fidelity) while
    `FeatureVector` computes it, so without this cross-check the batch path and the serving path
    could report different values for the same account — the classic train/serve drift bug."""
    with pytest.raises(DomainValidationError, match="contradicts"):
        make_features(
            total_credit=Decimal("100"), total_debit=Decimal("150"), pass_through_ratio=0.0
        )


def test_a_consistent_stored_ratio_is_accepted_including_rounding():
    """The tolerance exists so a value round-tripped through a scaled Postgres `numeric` column
    still loads. Four decimal places pass; a genuine contradiction does not."""
    row = make_features(
        total_credit=Decimal("100"), total_debit=Decimal("150"), pass_through_ratio=1.5
    )
    assert row.pass_through_ratio == 1.5

    rounded = make_features(
        total_credit=Decimal("284000"), total_debit=Decimal("266960"), pass_through_ratio=0.9400
    )
    assert rounded.pass_through_ratio == 0.94


def test_the_stored_row_and_the_computed_vector_agree_on_the_zero_credit_branch():
    """Both sides route through one function, so the "no credit means 0.0" branch cannot diverge."""
    row = make_features(
        total_credit=Decimal("0"), total_debit=Decimal("50000"), pass_through_ratio=0.0
    )
    vector = make_feature_vector(total_credit=Decimal("0"), total_debit=Decimal("50000"))
    assert row.pass_through_ratio == vector.pass_through_ratio == 0.0


def test_pass_through_ratio_is_zero_when_nothing_came_in():
    """The zero-credit branch. Division would raise; the DuckDB feature SQL returns 0, and one
    definition of a feature in two places is the drift this mirrors deliberately."""
    vector = make_feature_vector(total_credit=Decimal("0"), total_debit=Decimal("50000"))
    assert vector.pass_through_ratio == 0.0


def test_snapshot_age_is_measured_against_a_supplied_clock():
    vector = make_feature_vector(as_of=TS)
    assert vector.snapshot_age_ms(datetime(2026, 8, 3, 9, 30, 0, 340_000, tzinfo=UTC)) == 340
    # A vector stamped slightly ahead of the reader's clock reports 0 rather than a negative age.
    assert vector.snapshot_age_ms(datetime(2026, 8, 3, 9, 29, tzinfo=UTC)) == 0


def test_a_path_returning_to_its_origin_is_a_cycle():
    """Round-tripping (`CIRCULAR`) needs no separate type — a cycle is a path whose first and last
    account are the same."""
    assert make_path(hop_accounts=("ACC1", "ACC2", "ACC1"), hops=2).is_cycle is True
    assert make_path().is_cycle is False


def test_a_path_whose_hop_count_disagrees_with_its_nodes_is_rejected():
    """The two come from the same Cypher `RETURN`; if they disagree, one of them came from a
    different query and the path cannot be trusted."""
    with pytest.raises(DomainValidationError, match="disagrees"):
        make_path(hop_accounts=("ACC1", "ACC2", "ACC3"), hops=3)


def test_a_bottleneck_larger_than_the_whole_path_is_rejected():
    """`bottleneck_amount` is the smallest edge and `path_value` the sum of all of them, so the
    first cannot exceed the second. It is the bottleneck that may be quoted as an amount at risk,
    which is why the pair has to be coherent."""
    with pytest.raises(DomainValidationError, match="exceeds the total path_value"):
        make_path(path_value=Decimal("1"), bottleneck_amount=Decimal("1000000000"))


def test_a_single_hop_path_may_have_bottleneck_equal_to_its_value():
    path = make_path(
        hop_accounts=("ACC1", "ACC2"),
        hops=1,
        path_value=Decimal("42000"),
        bottleneck_amount=Decimal("42000"),
    )
    assert path.bottleneck_amount == path.path_value


def test_a_single_account_is_not_a_path():
    with pytest.raises(DomainValidationError, match="at least two accounts"):
        make_path(hop_accounts=("ACC1",), hops=0)


def test_ring_size_and_cross_customer_flag():
    """Five accounts owned by one person is a portfolio; five owned by five people sharing one
    device is a ring. `distinct_customers` is what separates them."""
    ring = RingFinding(
        ring_id="RING-0007",
        members=("ACC1", "ACC2", "ACC3"),
        distinct_customers=3,
        shared_device_id="DEV0001842",
        total_internal_value=Decimal("412000.00"),
    )
    assert ring.size == 3
    assert ring.is_cross_customer is True

    portfolio = RingFinding(
        ring_id=None,
        members=("ACC1", "ACC2"),
        distinct_customers=1,
        shared_device_id="DEV0001842",
        total_internal_value=Decimal("9000.00"),
    )
    assert portfolio.is_cross_customer is False


def test_more_customers_than_accounts_is_rejected():
    with pytest.raises(DomainValidationError, match="exceeds"):
        RingFinding(
            ring_id=None,
            members=("ACC1", "ACC2"),
            distinct_customers=3,
            shared_device_id=None,
            total_internal_value=Decimal("0"),
        )


# --- explainability invariants -----------------------------------------------------------------


def test_a_score_with_no_components_is_rejected():
    """Explainability by construction: a score without reasons cannot be defended to a customer or
    a regulator, so an adapter cannot forget to attach them."""
    with pytest.raises(DomainValidationError, match="no components"):
        ScoreResult(
            account_id="ACC00042917",
            decision=ScoreDecision.REVIEW,
            score=71.5,
            components=(),
            feature_snapshot_age_ms=340,
            latency_ms=7,
            model_version="iforest-2026-08-03",
        )


def test_a_score_result_keeps_its_component_breakdown():
    result = ScoreResult(
        account_id="ACC00042917",
        decision=ScoreDecision.REVIEW,
        score=71.5,
        components=(
            ScoreComponent("R006", "Pass-through behaviour", 25.0, "24h ratio 0.94"),
            ScoreComponent("ML", "Isolation Forest anomaly", 28.5, "score -0.41"),
        ),
        feature_snapshot_age_ms=340,
        latency_ms=7,
        model_version="iforest-2026-08-03",
    )
    assert [c.code for c in result.components] == ["R006", "ML"]
    # The band that maps score onto decision is config, so the domain does not police the pairing.
    assert result.decision is ScoreDecision.REVIEW


def test_a_rule_hit_without_detail_text_is_rejected():
    with pytest.raises(DomainValidationError, match="no detail"):
        RuleHit(
            rule_code="R004",
            rule_name="Structuring",
            severity=Severity.HIGH,
            contribution=18.0,
            detail="   ",
            window_start=TS,
            window_end=TS,
            evidence={"band_hits": 4},
        )


def test_a_failed_tool_call_must_say_why_it_failed():
    """Silent failure is the thing the tool trace exists to prevent — and any recorded error trips
    the human gate, so an unexplained one would escalate a case with nothing to show the analyst."""
    with pytest.raises(DomainValidationError, match="silent failure"):
        ToolCall(
            call_id=None,
            investigation_id="INV1",
            sequence_no=3,
            tool_name="trace_fund_flow",
            tool_input={"account_id": "ACC1"},
            tool_output=None,
            is_error=True,
            error_message=None,
            duration_ms=318,
            called_at=TS,
        )


def make_tool_call(**overrides: object) -> ToolCall:
    fields: dict[str, object] = {
        "call_id": None,
        "investigation_id": "INV1",
        "sequence_no": 3,
        "tool_name": "trace_fund_flow",
        "tool_input": {"account_id": "ACC1"},
        "tool_output": {"downstream_paths": []},
        "is_error": False,
        "error_message": None,
        "duration_ms": 318,
        "called_at": TS,
    }
    fields.update(overrides)
    return ToolCall(**fields)  # type: ignore[arg-type]


def test_a_successful_tool_call_must_record_its_output():
    """`tool_output is None` exactly when `is_error` — both halves enforced. A success with no
    output cannot be replayed, and an empty result (no policy matched) is a value, not a missing
    one, so it has no legitimate reason to arrive as `None`."""
    with pytest.raises(DomainValidationError, match="recorded no tool_output"):
        make_tool_call(is_error=False, tool_output=None)


def test_a_failed_tool_call_must_not_carry_an_output():
    with pytest.raises(DomainValidationError, match="carries a tool_output"):
        make_tool_call(is_error=True, error_message="Neo4j timeout", tool_output={"paths": []})


def test_an_empty_result_is_a_valid_output():
    """An empty result — no applicable policy matched — is a real answer the agent must cite."""
    assert make_tool_call(tool_output={"chunks": []}).tool_output == {"chunks": []}


def test_a_human_decision_must_carry_a_rationale():
    from sentinel.domain.entities import HumanDecision

    with pytest.raises(DomainValidationError, match="rationale"):
        HumanDecision(
            decision_id="DEC1",
            investigation_id="INV1",
            analyst_id="AN0031",
            final_decision=Recommendation.ESCALATE,
            agreed_with_agent=True,
            rationale="",
            decided_at=TS,
        )


def test_an_alert_window_cannot_end_before_it_starts():
    with pytest.raises(DomainValidationError, match="precedes"):
        Alert(
            alert_id="ALT1",
            account_id="ACC1",
            rule_code="R002",
            rule_name="Fan-in concentration",
            triggered_at=TS,
            window_start=TS,
            window_end=datetime(2026, 8, 2, tzinfo=UTC),
            severity=Severity.HIGH,
            rule_score=70.0,
            anomaly_score=-0.41,
            trigger_evidence={"distinct_credit_counterparties": 14},
            status=AlertStatus.OPEN,
            created_at=TS,
        )


def test_a_negative_anomaly_score_is_allowed():
    """Isolation Forest emits negative values for outliers — the interesting end. Range-checking
    this field would reject exactly the rows the model exists to find."""
    alert = Alert(
        alert_id="ALT1",
        account_id="ACC1",
        rule_code="R002",
        rule_name="Fan-in concentration",
        triggered_at=TS,
        window_start=datetime(2026, 8, 2, tzinfo=UTC),
        window_end=TS,
        severity=Severity.HIGH,
        rule_score=70.0,
        anomaly_score=-0.41,
        trigger_evidence={"distinct_credit_counterparties": 14},
        status=AlertStatus.OPEN,
        created_at=TS,
    )
    assert alert.anomaly_score == -0.41


# --- data-quality gate -------------------------------------------------------------------------


def make_check(severity: DqSeverity, status: DqCheckStatus) -> DqCheckResult:
    return DqCheckResult(
        result_id=None,
        run_id="DQ20260803",
        check_name="txn_id_uniqueness",
        target_table="transactions",
        target_column="txn_id",
        severity=severity,
        status=status,
        expectation="0 duplicates",
        observed="37 duplicate txn_id",
    )


@pytest.mark.parametrize(
    ("severity", "status", "blocks"),
    [
        (DqSeverity.BLOCKING, DqCheckStatus.FAIL, True),
        (DqSeverity.BLOCKING, DqCheckStatus.PASS, False),
        (DqSeverity.WARNING, DqCheckStatus.FAIL, False),
        (DqSeverity.WARNING, DqCheckStatus.PASS, False),
    ],
)
def test_only_a_failed_blocking_check_stops_the_batch(severity, status, blocks):
    """The entire gate in one truth table: a failed BLOCKING check means nothing is loaded, a
    failed WARNING is recorded and the batch proceeds."""
    assert make_check(severity, status).blocks_batch is blocks


def test_a_run_cannot_fail_more_checks_than_it_ran():
    from sentinel.domain.entities import DqRun

    with pytest.raises(DomainValidationError, match="exceeds"):
        DqRun(
            run_id="DQ20260803",
            batch_id="BATCH20260803",
            started_at=TS,
            finished_at=TS,
            status=enums.DqRunStatus.FAILED,
            rows_in=200_000,
            rows_loaded=0,
            rows_rejected=200_000,
            checks_total=12,
            checks_failed=13,
            report={},
        )


# --- evaluation ground truth -------------------------------------------------------------------


def test_a_mule_label_must_name_a_typology():
    """A mule with typology NONE would land in no per-typology bucket and quietly deflate recall."""
    with pytest.raises(DomainValidationError, match="typology NONE"):
        TypologyLabel(
            account_id="ACC1",
            is_mule=True,
            typology=Typology.NONE,
            ring_id=None,
            is_hard_negative=False,
            note="",
        )


def test_a_label_cannot_be_both_a_mule_and_a_hard_negative():
    """A hard negative is a legitimate account that merely looks like a typology. Both flags at
    once would corrupt the hard-negative false-positive rate, which is the honest difficulty
    measure in the metrics report."""
    with pytest.raises(DomainValidationError, match="both a mule and a hard negative"):
        TypologyLabel(
            account_id="ACC1",
            is_mule=True,
            typology=Typology.FAN_IN_OUT,
            ring_id="RING-0007",
            is_hard_negative=True,
            note="",
        )


def test_a_hard_negative_carries_a_real_typology_it_resembles():
    """Legitimate, but labelled with the pattern it imitates — a small merchant looks like
    FAN_IN_OUT, and that is why it is worth generating."""
    label = TypologyLabel(
        account_id="ACC1",
        is_mule=False,
        typology=Typology.FAN_IN_OUT,
        ring_id=None,
        is_hard_negative=True,
        note="Small merchant, steady fan-in across 90 days.",
    )
    assert label.is_hard_negative and not label.is_mule
