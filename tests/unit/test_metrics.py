"""The metrics module -- T7.1.

Every test here runs on a hand-built population of a dozen accounts, because the point of a metrics
module is that its arithmetic is right and the arithmetic is only checkable when the counts are
small enough to do in your head. The real figures come from `make metrics` against the real batch
and are recorded in `docs/13-KNOWN-FALSE-POSITIVES.md`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.domain.exceptions import DomainValidationError
from sentinel.reporting import metrics
from sentinel.reporting.rule_impact import Label, ScoredAccount, ScoreRun

AS_OF = datetime(2026, 8, 3, tzinfo=UTC)


def scored(account_id: str, score: float, *codes: str) -> ScoredAccount:
    """One account. `score_before` is deliberately different so a mix-up cannot pass."""
    return ScoredAccount(
        account_id=account_id,
        score_before=score / 2,
        score_after=score,
        rule_codes=codes,
    )


def mule(account_id: str, typology: str) -> Label:
    return Label(
        account_id=account_id,
        is_mule=True,
        typology=typology,
        is_hard_negative=False,
        note="injected",
    )


def hard_negative(account_id: str, cohort: str) -> Label:
    return Label(
        account_id=account_id,
        is_mule=False,
        typology="NONE",
        is_hard_negative=True,
        note=f"{cohort}: looks guilty, is not",
    )


# --- ConfusionMatrix ------------------------------------------------------------------------------


def test_the_four_cells_and_every_rate_derived_from_them():
    matrix = metrics.ConfusionMatrix(
        true_positives=6, false_positives=2, false_negatives=4, true_negatives=88
    )
    assert matrix.population == 100
    assert matrix.predicted_positive == 8
    assert matrix.actual_positive == 10
    assert matrix.precision == pytest.approx(0.75)
    assert matrix.recall == pytest.approx(0.60)
    # 2 * 0.75 * 0.6 / 1.35
    assert matrix.f1 == pytest.approx(0.6666666667)
    assert matrix.false_positive_rate == pytest.approx(2 / 90)


def test_a_classifier_that_flags_nothing_has_precision_zero_not_one():
    """The single most flattering lie available here, refused explicitly."""
    matrix = metrics.ConfusionMatrix(
        true_positives=0, false_positives=0, false_negatives=10, true_negatives=90
    )
    assert matrix.precision == 0.0
    assert matrix.recall == 0.0
    assert matrix.f1 == 0.0


def test_a_population_with_no_positives_has_recall_zero_not_a_crash():
    matrix = metrics.ConfusionMatrix(
        true_positives=0, false_positives=3, false_negatives=0, true_negatives=97
    )
    assert matrix.recall == 0.0
    assert matrix.precision == 0.0
    assert matrix.false_positive_rate == pytest.approx(0.03)


# --- confusion() ----------------------------------------------------------------------------------


def test_confusion_counts_the_four_cells():
    matrix = metrics.confusion(
        predicted={"a", "b", "c"},
        actual={"b", "c", "d"},
        population={"a", "b", "c", "d", "e", "f"},
    )
    assert (matrix.true_positives, matrix.false_positives) == (2, 1)
    assert (matrix.false_negatives, matrix.true_negatives) == (1, 2)
    assert matrix.population == 6


def test_confusion_refuses_a_prediction_outside_the_population():
    """A matrix over two different books adds up and means nothing."""
    with pytest.raises(DomainValidationError, match="outside the evaluated population"):
        metrics.confusion(predicted={"ghost"}, actual={"a"}, population={"a", "b"})


def test_confusion_refuses_a_label_outside_the_population():
    with pytest.raises(DomainValidationError, match="outside the evaluated population"):
        metrics.confusion(predicted={"a"}, actual={"ghost"}, population={"a", "b"})


# --- the curve ------------------------------------------------------------------------------------


def test_the_curve_separates_a_perfectly_ordered_population():
    scores = {f"A{i:02d}": float(i * 10) for i in range(10)}
    mules = {"A07", "A08", "A09"}
    curve = metrics.build_curve(scores, mules, set(scores))
    assert curve.roc_auc == pytest.approx(1.0)
    assert curve.average_precision == pytest.approx(1.0)
    assert curve.best.matrix.f1 == pytest.approx(1.0)
    assert curve.best.threshold == pytest.approx(70.0)


def test_the_best_point_is_searched_over_every_distinct_score_not_the_printed_grid():
    """A best F1 at 33 must be found even though the printed table only walks multiples of five."""
    scores = {"a": 33.0, "b": 34.0, "c": 10.0, "d": 9.0}
    curve = metrics.build_curve(scores, {"a", "b"}, set(scores))
    assert curve.best.threshold == pytest.approx(33.0)
    assert curve.best.matrix.f1 == pytest.approx(1.0)
    assert all(point.threshold % metrics.SWEEP_STEP == 0 for point in curve.points)


def test_the_printed_grid_starts_by_flagging_everyone():
    scores = {"a": 5.0, "b": 60.0, "c": 0.0}
    curve = metrics.build_curve(scores, {"b"}, set(scores))
    first = curve.points[0]
    assert first.threshold == 0.0
    assert first.matrix.predicted_positive == 3
    assert first.matrix.recall == pytest.approx(1.0)


def test_a_curve_needs_at_least_one_positive():
    with pytest.raises(DomainValidationError, match="no labelled mule"):
        metrics.build_curve({"a": 10.0}, set(), {"a"})


def test_a_curve_needs_a_population():
    with pytest.raises(DomainValidationError, match="zero accounts"):
        metrics.build_curve({}, set(), set())


# --- build_report ---------------------------------------------------------------------------------


def sample_report() -> metrics.MetricsReport:
    """Twelve accounts: four mules, three hard negatives, five unlabelled.

    Scores are chosen so the headline classifier at REVIEW=40 catches two mules, misses two, and
    picks up one hard negative and one unlabelled account. Every count below is checkable by eye.
    """
    run = ScoreRun(
        as_of=AS_OF,
        accounts=(
            scored("M1", 90.0, "R002"),  # mule, caught
            scored("M2", 55.0, "R006"),  # mule, caught
            scored("M3", 12.0),  # mule, missed
            scored("M4", 39.0, "R004"),  # mule, missed by one point
            scored("H1", 44.0, "R003"),  # hard negative, false positive
            scored("H2", 20.0),
            scored("H3", 8.0),
            scored("U1", 41.0),  # unlabelled, false positive
            scored("U2", 5.0),
            scored("U3", 6.0),
            scored("U4", 7.0),
            scored("U5", 8.0),
        ),
    )
    labels = (
        mule("M1", "FAN_IN_OUT"),
        mule("M2", "FAN_IN_OUT"),
        mule("M3", "STRUCTURING"),
        mule("M4", "STRUCTURING"),
        hard_negative("H1", "payroll"),
        hard_negative("H2", "payroll"),
        hard_negative("H3", "merchant"),
    )
    alerts = {
        "M1": ("R002", "R006"),
        "M2": ("R006",),
        "M4": ("R004",),
        "H1": ("R003",),
        "GHOST": ("R001",),  # an account from an earlier generation
    }
    return metrics.build_report(run, labels, alerts)


def test_the_headline_matrix_is_the_score_at_the_review_band():
    report = sample_report()
    assert report.review_at == 40.0
    assert report.by_score.true_positives == 2  # M1, M2
    assert report.by_score.false_positives == 2  # H1, U1
    assert report.by_score.false_negatives == 2  # M3, M4
    assert report.by_score.true_negatives == 6
    assert report.by_score.precision == pytest.approx(0.5)
    assert report.by_score.recall == pytest.approx(0.5)
    assert report.by_score.f1 == pytest.approx(0.5)


def test_the_alert_classifier_is_reported_separately_and_disagrees():
    """M4 alerts but scores 39, so the two classifiers must not produce the same matrix."""
    report = sample_report()
    assert report.by_alert.true_positives == 3  # M1, M2, M4
    assert report.by_alert.false_positives == 1  # H1
    assert report.by_alert.false_negatives == 1  # M3
    assert report.by_alert.recall == pytest.approx(0.75)
    assert report.by_alert.recall > report.by_score.recall


def test_per_alert_precision_counts_alerts_not_accounts():
    """M1 raises two alerts, so the alert unit and the account unit give different numbers."""
    report = sample_report()
    # M1 x2, M2, M4, H1. GHOST's alert is off-batch and excluded from all five figures.
    assert report.alerts == 5
    assert report.alerts_on_mules == 4
    assert report.alert_precision == pytest.approx(0.8)
    assert report.alerts_off_batch == 1
    assert report.alerts_per_mule_found == pytest.approx(5 / 3)
    # The account unit says 4 of 5 accounts alerted are mules is not the same question:
    # by_alert.precision counts accounts, and M1's second alert does not appear in it.
    assert report.by_alert.precision == pytest.approx(0.75)


def test_the_population_counts_split_three_ways():
    report = sample_report()
    assert (report.accounts, report.mules, report.hard_negatives) == (12, 4, 3)
    assert report.unlabelled == 5


def test_per_typology_recall_is_reported_per_typology():
    report = sample_report()
    assert {group.name: (group.hit, group.n) for group in report.typology_recall} == {
        "FAN_IN_OUT": (2, 2),
        "STRUCTURING": (0, 2),
    }
    # And the alert view catches one structuring account the score does not.
    assert {group.name: group.hit for group in report.typology_alert_recall} == {
        "FAN_IN_OUT": 2,
        "STRUCTURING": 1,
    }


def test_hard_negatives_keep_their_own_cohorts():
    """Folding them into the unlabelled pool would hide the rate they were injected to expose."""
    report = sample_report()
    rates = {group.name: (group.hit, group.n) for group in report.hard_negative_fp}
    assert rates == {"HARD_NEG:payroll": (1, 2), "HARD_NEG:merchant": (0, 1)}
    names = [group.name for group in report.hard_negative_fp]
    assert names == sorted(names)  # stable ordering, so two runs print identically


def test_per_rule_alert_precision_covers_every_configured_rule():
    report = sample_report()
    by_code = {group.name: (group.hit, group.n) for group in report.per_rule}
    assert by_code["R002"] == (1, 1)
    assert by_code["R006"] == (2, 2)
    assert by_code["R003"] == (0, 1)
    assert by_code["R004"] == (1, 1)
    # R001 fired only on the off-batch ghost, so it counts nowhere. It is still listed, because
    # a rule that raised nothing on this batch is a fact worth printing.
    assert by_code["R001"] == (0, 0)
    assert by_code["R005"] == (0, 0)
    assert by_code["ML001"] == (0, 0)


def test_rule_hit_coverage_is_reported_for_both_sides():
    report = sample_report()
    assert report.mules_with_rule_hits == 3  # M1, M2, M4
    assert report.hard_negatives_with_rule_hits == 1  # H1


def test_the_curve_rides_on_score_after_not_score_before():
    """`score_before` is half of `score_after` in this fixture, so a mix-up changes the AUC."""
    report = sample_report()
    # 27 of the 32 mule/innocent pairs are ordered correctly on `score_after`; on `score_before`
    # the halved scores would give a different set of ties and a different figure.
    assert report.curve.roc_auc == pytest.approx(27 / 32)
    assert report.curve.best.threshold == pytest.approx(12.0)
    assert report.curve.best.matrix.f1 == pytest.approx(2 * (4 / 7) / (4 / 7 + 1))
    assert report.as_of == AS_OF


def test_the_report_prints_without_blowing_up(capsys):
    """The printer is what gets pasted into the docs, so it is exercised, not just the numbers."""
    metrics._print(sample_report())
    out = capsys.readouterr().out
    assert "HEADLINE" in out
    assert "NOT applied" in out
    assert "FLOOR, not a result" in out
