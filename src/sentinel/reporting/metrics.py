"""Precision, recall, F1 and the confusion matrix against the ground truth -- T7.1.

    python -m sentinel.reporting.metrics       (or `make metrics`)

This is the module `docs/09-FOLDER-STRUCTURE.md` names as the evaluation boundary. It reads
`typology_labels` and `scripts/check_label_leakage.py` allow-lists it for that; nothing in
`pipeline/`, `application/`, `api/` or `agent/` may import it. Labels are opened **after** every
score already exists -- `rule_impact.score_accounts` is deliberately label-free, and this module
calls it before it calls `rule_impact.read_labels`.

---

## The unit of evaluation, stated before the numbers

"Precision" is ambiguous on this data and the choice moves it a long way, so it is fixed here:

**The headline is per account, and the classifier is `risk score >= REVIEW (40)`.**

Three reasons, in order of weight:

1. **The ground truth is per account.** A `typology_labels` row is an account. Recall -- and
   therefore F1 -- has no denominator at any other unit: there is no such thing as an "alert that
   should have been raised" in the label file.
2. **The score is the decision.** `ScoreBands.review_at` is what `docs/13-KNOWN-FALSE-POSITIVES.md`
   and the `/v1/score` path both treat as the band where a case opens. Measuring anything else
   would measure something the system does not act on.
3. **Only the score has a knob.** It is continuous, so the threshold sweeps and a
   precision-recall curve exists. A single operating point at 40 answers far less than the curve.

**The other two units are reported, not hidden.** Per account with the *alert set* as the
classifier ("did detection put this account in front of an analyst at all?"), and per alert
("was this alert worth the twenty minutes?"). Where they disagree with the headline, the
disagreement is the finding.

---

## Every number here is a floor, and this is not a hedge

Two of the seven scoring components -- `FAN_ASYMMETRY` (16) and `NETWORK_STRUCTURE` (16) -- need a
`FundFlowResult` that Stage 8 has not built yet, so they report *not gathered* and contribute
nothing. **68 of 100 points are reachable**, and `review_at` was calibrated for a scorer with all
seven. A mule that would clear 40 on graph evidence cannot clear it here, so recall is understated
by an unknown amount and precision is measured on a truncated score. Raising the missing components
by faking a `FundFlowResult` would be worse: it would report "no cycles found" for an account nobody
traced.

**Nothing in this module tunes anything.** No threshold is chosen by looking at the labels, no
contribution is re-fitted, no weight moves. The curve reports what each threshold *would* give; it
is not permission to pick the best one. Doing that would make every figure here worthless, which is
the exact failure `scripts/check_label_leakage.py` exists to make expensive.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.config.weights import ML_SPEC, RULE_SPECS, ScoringConfig
from sentinel.domain.exceptions import DomainValidationError
from sentinel.infrastructure.postgres import connection as pg
from sentinel.infrastructure.postgres.repositories import PostgresAlertRepository
from sentinel.reporting import rule_impact

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from datetime import datetime

    from sentinel.reporting.rule_impact import Label, ScoreRun

log = get_logger(__name__)

#: Thresholds the printed operating-point table walks. Every fifth point rather than every score,
#: because the table is meant to be read by a human deciding where to run, and 6,200 rows is not.
#: The curve's `average_precision` is computed over every distinct score, not over this grid.
SWEEP_STEP: Final = 5


# --- the primitives -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Raw counts first. Every rate below is derived from them and none is stored.

    Counts rather than rates are the stored form on purpose: `precision = 0.5` is unreadable when it
    means one true positive and one false positive, and an F1 quoted without `tp` hides exactly that
    case.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def population(self) -> int:
        cells = (
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.true_negatives,
        )
        return sum(cells)

    @property
    def predicted_positive(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def actual_positive(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        """Of what we flagged, how much was real. Zero when nothing was flagged.

        Zero rather than `nan` or a raise: a classifier that flags nothing has precision that is
        undefined in arithmetic and worthless in practice, and reporting it as 1.0 -- which the
        "vacuously correct" reading would give -- is the single most flattering lie available here.
        """
        return self.true_positives / self.predicted_positive if self.predicted_positive else 0.0

    @property
    def recall(self) -> float:
        """Of what was real, how much we caught."""
        return self.true_positives / self.actual_positive if self.actual_positive else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Of the innocent accounts, how many we flagged -- the analyst's cost, per head."""
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives else 0.0


@dataclass(frozen=True, slots=True)
class GroupResult:
    """`hit` out of `n`, named. Serves per-typology recall and per-cohort false positives both.

    One type for both because they are arithmetically the same question asked of different groups --
    "how often did this subset get flagged" -- and two dataclasses differing only in field names
    would invite the two tables to drift apart.
    """

    name: str
    n: int
    hit: int

    @property
    def rate(self) -> float:
        return self.hit / self.n if self.n else 0.0


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """What the headline classifier would give at one threshold."""

    threshold: float
    matrix: ConfusionMatrix


@dataclass(frozen=True, slots=True)
class Curve:
    """The precision-recall curve, summarised, plus the point that maximises F1.

    `best` is reported to answer "what threshold would you actually run at?" and **is not applied**.
    Moving `review_at` to it would be fitting the operating point to the answer sheet.
    """

    average_precision: float
    roc_auc: float
    points: tuple[OperatingPoint, ...]
    best: OperatingPoint


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """Everything T7.1 asks for, computed once so the printer and the tests read the same object."""

    as_of: datetime
    review_at: float
    accounts: int
    mules: int
    hard_negatives: int
    unlabelled: int

    #: Headline: per account, `score_after >= review_at`.
    by_score: ConfusionMatrix
    #: Same unit, different classifier: per account, "detection raised at least one alert".
    by_alert: ConfusionMatrix

    #: Alert rows on accounts in *this* batch. `alerts_off_batch` holds the rest -- rows left in
    #: Neon by an earlier generated population, which are neither a true nor a false positive here
    #: and are reported separately rather than quietly counted or quietly dropped.
    alerts: int
    alerts_on_mules: int
    alerts_off_batch: int
    per_rule: tuple[GroupResult, ...]

    typology_recall: tuple[GroupResult, ...]
    typology_alert_recall: tuple[GroupResult, ...]
    hard_negative_fp: tuple[GroupResult, ...]
    hard_negative_alert_fp: tuple[GroupResult, ...]

    mules_with_rule_hits: int
    hard_negatives_with_rule_hits: int

    curve: Curve

    @property
    def alert_precision(self) -> float:
        """Per alert: the share of an analyst's queue that is a genuine mule."""
        return self.alerts_on_mules / self.alerts if self.alerts else 0.0

    @property
    def alerts_per_mule_found(self) -> float:
        """How many alerts an analyst reads per mule confirmed. The workload number."""
        found = self.by_alert.true_positives
        return self.alerts / found if found else 0.0


# --- pure computation -----------------------------------------------------------------------------


def confusion(
    *,
    predicted: AbstractSet[str],
    actual: AbstractSet[str],
    population: AbstractSet[str],
) -> ConfusionMatrix:
    """Count the four cells. Both input sets must lie inside `population`.

    The containment check is not ceremony. An account in `predicted` that is not in `population`
    silently inflates precision's denominator without ever being counted as a negative, and the
    resulting matrix still adds up -- so the error is invisible in the output. Failing here is the
    only place it can be caught.
    """
    for name, group in (("predicted", predicted), ("actual", actual)):
        stray = sorted(group - population)
        if stray:
            raise DomainValidationError(
                f"{len(stray)} account(s) in `{name}` are outside the evaluated population, "
                f"starting with {stray[:3]!r}. A confusion matrix over two different books adds up "
                f"and means nothing."
            )
    true_positives = len(predicted & actual)
    return ConfusionMatrix(
        true_positives=true_positives,
        false_positives=len(predicted) - true_positives,
        false_negatives=len(actual) - true_positives,
        true_negatives=len(population) - len(predicted | actual),
    )


def _grouped(
    members: Mapping[str, Collection[str]], flagged: AbstractSet[str]
) -> tuple[GroupResult, ...]:
    """One `GroupResult` per named group, sorted by name so two runs print identically."""
    return tuple(
        GroupResult(name=name, n=len(accounts), hit=len(set(accounts) & flagged))
        for name, accounts in sorted(members.items())
    )


def build_curve(
    scores: Mapping[str, float], mules: AbstractSet[str], population: AbstractSet[str]
) -> Curve:
    """Sweep the threshold. `average_precision` and `roc_auc` come from scikit-learn.

    Average precision rather than the trapezoidal area under the interpolated curve: on a book
    that is 2.7% positive the interpolated form is optimistically biased, and AP is the
    step-wise sum scikit-learn documents as the honest estimator for exactly that case.

    The printed grid is coarse (`SWEEP_STEP`) but `best` is searched over **every distinct score**,
    so the best-F1 point is a real operating point and not the nearest multiple of five.
    """
    if not population:
        raise DomainValidationError("cannot build a precision-recall curve over zero accounts.")
    ordered = sorted(population)
    truth = [1 if account in mules else 0 for account in ordered]
    values = [scores.get(account, 0.0) for account in ordered]
    if not any(truth):
        raise DomainValidationError(
            "no labelled mule is present in the scored population, so recall has no denominator "
            "and no curve exists. Check that `typology_labels` and the feature batch agree."
        )

    def at(threshold: float) -> OperatingPoint:
        flagged = {account for account in ordered if scores.get(account, 0.0) >= threshold}
        return OperatingPoint(
            threshold=threshold,
            matrix=confusion(predicted=flagged, actual=set(mules), population=set(ordered)),
        )

    grid = tuple(at(float(step)) for step in range(0, 101, SWEEP_STEP))
    # Every distinct score is a candidate operating point; anything between two of them flags the
    # same accounts, so this is the whole search space and not a sample of it.
    candidates = (at(value) for value in sorted(set(values)))
    return Curve(
        average_precision=float(average_precision_score(truth, values)),
        roc_auc=float(roc_auc_score(truth, values)),
        points=grid,
        # Ties in F1 break towards the *higher* threshold: two operating points that trade recall
        # against precision equally well are not equally cheap, and the one flagging fewer accounts
        # costs fewer analyst-hours for the same measured quality.
        best=max(candidates, key=lambda point: (point.matrix.f1, point.threshold)),
    )


def build_report(
    run: ScoreRun,
    labels: Sequence[Label],
    alerts: Mapping[str, tuple[str, ...]],
    scoring: ScoringConfig | None = None,
) -> MetricsReport:
    """Turn scores, labels and alerts into every figure T7.1 asks for. Pure -- no I/O, no clock."""
    config = scoring if scoring is not None else ScoringConfig()
    review_at = config.bands.review_at

    population = {account.account_id for account in run.accounts}
    scores = {account.account_id: account.score_after for account in run.accounts}
    with_hits = {account.account_id for account in run.accounts if account.rule_codes}

    mules = {label.account_id for label in labels if label.is_mule} & population
    hard_negatives = {label.account_id for label in labels if label.is_hard_negative} & population

    flagged_by_score = {account for account, score in scores.items() if score >= review_at}
    # An alert row whose account is not in this batch is a stale row from an earlier generation, not
    # a false positive. Counted and reported rather than silently dropped or silently counted.
    alerted = set(alerts) & population

    typologies: dict[str, list[str]] = {}
    cohorts: dict[str, list[str]] = {}
    for label in labels:
        if label.account_id not in population:
            continue
        bucket = cohorts if label.is_hard_negative else typologies
        bucket.setdefault(label.cohort, []).append(label.account_id)

    # Every configured rule is listed even when it raised nothing, because "R005 fired zero times"
    # is a finding and an absent row reads as an oversight.
    rule_totals: dict[str, list[int]] = {spec.code: [0, 0] for spec in (*RULE_SPECS, ML_SPEC)}
    for account in alerted:
        for code in alerts[account]:
            tally = rule_totals.setdefault(code, [0, 0])
            tally[0] += 1
            tally[1] += 1 if account in mules else 0

    return MetricsReport(
        as_of=run.as_of,
        review_at=review_at,
        accounts=len(population),
        mules=len(mules),
        hard_negatives=len(hard_negatives),
        unlabelled=len(population) - len({label.account_id for label in labels} & population),
        by_score=confusion(predicted=flagged_by_score, actual=mules, population=population),
        by_alert=confusion(predicted=alerted, actual=mules, population=population),
        alerts=sum(len(alerts[account]) for account in alerted),
        alerts_on_mules=sum(len(alerts[account]) for account in alerted & mules),
        alerts_off_batch=sum(
            len(codes) for account, codes in alerts.items() if account not in population
        ),
        per_rule=tuple(
            GroupResult(name=code, n=raised, hit=on_mules)
            for code, (raised, on_mules) in sorted(rule_totals.items())
        ),
        typology_recall=_grouped(typologies, flagged_by_score),
        typology_alert_recall=_grouped(typologies, alerted),
        hard_negative_fp=_grouped(cohorts, flagged_by_score),
        hard_negative_alert_fp=_grouped(cohorts, alerted),
        mules_with_rule_hits=len(mules & with_hits),
        hard_negatives_with_rule_hits=len(hard_negatives & with_hits),
        curve=build_curve(scores, mules, population),
    )


# --- I/O and the entrypoint -----------------------------------------------------------------------


def compute(settings: Settings | None = None) -> MetricsReport:
    """Score the batch, then read the labels, then read the alerts. That order is the contract.

    Alerts come from Neon rather than being recomputed here, because the question the alert-level
    figures answer is "what is sitting in the analyst queue", and the queue is the table.
    """
    cfg = settings if settings is not None else get_settings()
    run = rule_impact.score_accounts(cfg)
    labels = rule_impact.read_labels(cfg)
    with pg.connect(cfg) as connection:
        alerts = PostgresAlertRepository(connection).rule_codes_by_account()
    log.info(
        "metrics inputs gathered",
        accounts=len(run.accounts),
        labels=len(labels),
        alerted_accounts=len(alerts),
    )
    return build_report(run, labels, alerts)


def _matrix_lines(title: str, matrix: ConfusionMatrix) -> list[str]:
    return [
        f"{title}",
        f"  TP {matrix.true_positives:>6,}   FP {matrix.false_positives:>6,}",
        f"  FN {matrix.false_negatives:>6,}   TN {matrix.true_negatives:>6,}",
        f"  precision {matrix.precision:.4f}   recall {matrix.recall:.4f}   f1 {matrix.f1:.4f}   "
        f"fpr {matrix.false_positive_rate:.4f}",
    ]


def _group_lines(title: str, groups: Sequence[GroupResult], label: str) -> list[str]:
    if not groups:
        return [title, "  (none)"]
    width = max(len(group.name) for group in groups)
    return [
        title,
        *(
            f"  {group.name.ljust(width)}  {group.hit:>4,} / {group.n:>4,}  {group.rate:>7.1%}"
            f"  {label}"
            for group in groups
        ),
    ]


def _print(report: MetricsReport) -> None:
    """The whole report as text. Deliberately plain -- this output is pasted into the docs."""
    lines: list[str] = [
        f"as_of {report.as_of.isoformat()}  |  REVIEW threshold {report.review_at:.0f}",
        f"population {report.accounts:,} accounts  |  {report.mules:,} labelled mules  |  "
        f"{report.hard_negatives:,} hard negatives  |  {report.unlabelled:,} unlabelled",
        "",
        *_matrix_lines(
            "HEADLINE - per account, classifier = risk score >= REVIEW", report.by_score
        ),
        "",
        *_matrix_lines("per account, classifier = detection raised an alert", report.by_alert),
        "",
        f"per alert - {report.alerts:,} alerts, {report.alerts_on_mules:,} on a labelled mule  |  "
        f"precision {report.alert_precision:.4f}",
        f"           {report.alerts_per_mule_found:.1f} alerts read per mule found  |  "
        f"{report.alerts_off_batch:,} alert(s) on accounts outside this batch",
        "",
        *_group_lines("per-alert precision by rule", report.per_rule, "on a mule"),
        "",
        *_group_lines("per-typology recall (score >= REVIEW)", report.typology_recall, "recalled"),
        "",
        *_group_lines(
            "per-typology recall (alerted at all)", report.typology_alert_recall, "alerted"
        ),
        "",
        *_group_lines(
            "per-hard-negative false positives (score >= REVIEW)",
            report.hard_negative_fp,
            "flagged",
        ),
        "",
        *_group_lines(
            "per-hard-negative false positives (alerted at all)",
            report.hard_negative_alert_fp,
            "alerted",
        ),
        "",
        f"rule engine: {report.mules_with_rule_hits:,} of {report.mules:,} mules fire a rule  |  "
        f"{report.hard_negatives_with_rule_hits:,} of {report.hard_negatives:,} hard negatives do",
        "",
        f"average precision {report.curve.average_precision:.4f}  |  "
        f"ROC AUC {report.curve.roc_auc:.4f}",
        f"best F1 at threshold {report.curve.best.threshold:.2f}: "
        f"precision {report.curve.best.matrix.precision:.4f}  "
        f"recall {report.curve.best.matrix.recall:.4f}  f1 {report.curve.best.matrix.f1:.4f}  "
        f"(reported, NOT applied)",
        "",
        f"{'THRESH':>6}  {'FLAGGED':>8}  {'TP':>5}  {'FP':>6}  {'PREC':>7}  {'RECALL':>7}  "
        f"{'F1':>7}",
    ]
    for point in report.curve.points:
        matrix = point.matrix
        lines.append(
            f"{point.threshold:>6.0f}  {matrix.predicted_positive:>8,}  "
            f"{matrix.true_positives:>5,}  {matrix.false_positives:>6,}  "
            f"{matrix.precision:>7.4f}  {matrix.recall:>7.4f}  {matrix.f1:>7.4f}"
        )
    lines += [
        "",
        "FLOOR, not a result: FAN_ASYMMETRY (16) and NETWORK_STRUCTURE (16) need the Stage 8 graph",
        "and report `not gathered`, so 68 of 100 points are reachable while REVIEW at 40 was",
        "calibrated for all seven components. Nothing above was tuned against these labels.",
    ]
    print("\n".join(lines))


def main(settings: Settings | None = None) -> int:
    """`python -m sentinel.reporting.metrics`, or `make metrics`."""
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    _print(compute(cfg))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
