"""What the rule engine did to the score. **Reads the ground truth -- reporting only.**

    python -m sentinel.reporting.rule_impact

This module is *not* part of the detection path and nothing in `pipeline/` or `application/` may
import it. It lives under `reporting/` for exactly the reason T7.1 gives: measuring against
`typology_labels` is legitimate, detecting against them is not, and the two must be separable by
directory so a lint rule can enforce the boundary. The labels are read **after** every score has
already been computed, purely to group the results into cohorts.

---

**The question it answers.** `docs/00-OVERVIEW.md` §3.1 records a measured failure: the six
flow-and-graph scoring components are blind to a named pattern, so STRUCTURING accounts reached the
40-point REVIEW threshold **0% of the time** and LAYERING **2%**. A seventh component was added to
read `RuleHit`, and until T6.1 nothing produced one, so the fix was wired but unmeasurable. This
measures it.

**The comparison is controlled and the control is the point.** The same accounts, the same evidence,
scored twice: once with `rule_hits=None` (the world before T6.1, where the component reports *not
gathered*) and once with the hits the engine actually produced. The difference is attributable to
the rule engine and to nothing else.

---

**What this is not.** It is **not** the Stage 7 accuracy measurement, and the score it computes is
**not** the score the agent will produce. Two of the seven components read graph evidence that does
not exist yet:

| Component | Weight | Available here? |
|---|---|---|
| `PASS_THROUGH` | 20 | yes, from the feature row |
| `RULE_HITS` | 20 | yes, from the engine |
| `ACCOUNT_PROFILE` | 12 | yes, from accounts joined to customers |
| `VELOCITY_ANOMALY` | 8 | yes, from the feature row |
| `IDENTITY_LINKAGE` | 8 | yes, from the same device join R007 uses |
| `FAN_ASYMMETRY` | 16 | **no** -- needs `FundFlowResult.signature`, Stage 8 |
| `NETWORK_STRUCTURE` | 16 | **no** -- needs cycles, chain depth and rings, Stage 8 |

So **68 of the 100 points are reachable** and 32 are structurally absent. A `FundFlowResult` could
be faked from counterparty counts, and that would be worse than leaving it out: it would report "no
cycles found" for an account nobody traced, which is precisely the collapse between *checked and
clean* and *never checked* that `_NOT_GATHERED` exists to prevent. Every absolute figure below is
therefore a **floor**, not an estimate, and the REVIEW threshold it is compared against was
calibrated for a scorer with all seven components available.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

from sentinel.application.risk_scorer import RiskEvidence, assess_risk
from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.config.weights import RULE_SPECS, DetectionConfig, ScoringConfig
from sentinel.domain.enums import (
    AccountStatus,
    AccountType,
    CustomerSegment,
    KycLevel,
    RiskRating,
)
from sentinel.domain.value_objects import (
    AccountProfile,
    FeatureVector,
    IdentifierLink,
    SharedIdentifierResult,
)
from sentinel.infrastructure.storage import duckdb_engine, parquet_store
from sentinel.pipeline import detect, features, rules

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

log = get_logger(__name__)

#: The window the score is built from -- the same one the rules and the anomaly model read.
SCORE_WINDOW: Final = detect.DETECTION_WINDOW

#: Where the ground truth lives. Read here and nowhere else in the codebase.
LABELS_DIR: Final = "typology_labels"

#: Roughly a month. Only used to render tenure for `AccountProfile`, exactly as
#: `PostgresAccountRepository` does; nothing turns on the third significant figure.
_DAYS_PER_MONTH: Final = 30.44


@dataclass(frozen=True, slots=True)
class Label:
    """One row of `typology_labels`, as the ground truth the evaluation reads.

    The whole row rather than a cohort string, because `reporting/metrics.py` needs the parts
    separately: `typology` drives per-typology recall, `is_hard_negative` splits the negatives into
    "deliberately built to look guilty" and "nobody labelled it", and those are different
    denominators.
    """

    account_id: str
    is_mule: bool
    typology: str
    is_hard_negative: bool
    note: str

    @property
    def cohort(self) -> str:
        """The reporting bucket. Hard negatives keep their own, never folded into `NONE`.

        The generator's `note` is its own description of the cohort and its first clause names it,
        which is why the split is on `:` rather than on a lookup table that would drift from the
        generator.
        """
        if self.is_hard_negative:
            return f"HARD_NEG:{self.note.split(':')[0]}"
        return self.typology


@dataclass(frozen=True, slots=True)
class ScoredAccount:
    """One account scored twice -- without the rule engine and with it.

    `score_before` is the world before T6.1, where the seventh component reports *not gathered*;
    `score_after` is the score the same evidence produces once the engine's `RuleHit` objects are
    supplied. Nothing else differs between the two, which is what makes the gap attributable.
    """

    account_id: str
    score_before: float
    score_after: float
    rule_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScoreRun:
    """Every account in the batch, scored, with the batch date that produced them."""

    as_of: datetime
    accounts: tuple[ScoredAccount, ...]


@dataclass(frozen=True, slots=True)
class CohortResult:
    """One labelled cohort, scored with and without the rule engine."""

    cohort: str
    accounts: int
    review_before: int
    review_after: int
    mean_before: float
    mean_after: float
    accounts_with_hits: int

    @property
    def review_rate_before(self) -> float:
        return self.review_before / self.accounts if self.accounts else 0.0

    @property
    def review_rate_after(self) -> float:
        return self.review_after / self.accounts if self.accounts else 0.0


def _account_context(settings: Settings) -> dict[str, dict[str, Any]]:
    """Account plus customer, joined -- the input to `AccountProfile`.

    Read through DuckDB rather than through `PostgresAccountRepository` so the measurement runs off
    the same curated files the detector did, with no dependency on what happens to be loaded.
    """
    accounts = duckdb_engine.table_file(settings.data_curated_path, parquet_store.ACCOUNTS_DIR)
    customers = duckdb_engine.table_file(settings.data_curated_path, parquet_store.CUSTOMERS_DIR)
    with duckdb_engine.connect() as connection:
        cursor = connection.execute(
            """
            SELECT a.account_id, a.customer_id, a.account_type, a.status, a.opened_on,
                   a.current_balance, a.last_activity_on,
                   c.kyc_level, c.declared_annual_income, c.risk_rating, c.segment, c.is_pep
              FROM read_parquet($accounts) a
              JOIN read_parquet($customers) c ON c.customer_id = a.customer_id
            """,
            {"accounts": accounts, "customers": customers},
        )
        names = [column[0] for column in cursor.description or ()]
        rows = cursor.fetchall()
    return {str(row[0]): dict(zip(names, row, strict=True)) for row in rows}


def read_labels(settings: Settings) -> tuple[Label, ...]:
    """The ground truth. **This function and `tests/` are the only readers in the project.**

    `reporting/metrics.py` calls this rather than opening the file itself, so there is exactly one
    read of `typology_labels` in `src/` for `scripts/check_label_leakage.py` to allow-list. Every
    caller is downstream of a score that was already computed without it.
    """
    path = (settings.data_raw_path / LABELS_DIR / "**" / "*.parquet").as_posix()
    with duckdb_engine.connect() as connection:
        rows = connection.execute(
            "SELECT account_id, is_mule, typology, is_hard_negative, note "
            "FROM read_parquet($path) ORDER BY account_id",
            {"path": path},
        ).fetchall()
    return tuple(
        Label(
            account_id=str(account_id),
            is_mule=bool(is_mule),
            typology=str(typology),
            is_hard_negative=bool(is_hard_negative),
            note=str(note),
        )
        for account_id, is_mule, typology, is_hard_negative, note in rows
    )


def _profile(context: Mapping[str, Any], as_of_date: Any) -> AccountProfile:
    """The same derivation `PostgresAccountRepository.get_profile` performs, against the batch date.

    `as_of` rather than the wall clock, for the reason the whole pipeline follows: a tenure that
    moves because a day passed makes yesterday's measurement unreproducible.
    """
    tenure_days = max((as_of_date - context["opened_on"]).days, 0)
    last_activity = context["last_activity_on"]
    status = AccountStatus(context["status"])
    silent_days = (as_of_date - last_activity).days if last_activity is not None else tenure_days
    return AccountProfile(
        account_id=str(context["account_id"]),
        customer_id=str(context["customer_id"]),
        account_type=AccountType(context["account_type"]),
        status=status,
        tenure_months=int(tenure_days / _DAYS_PER_MONTH),
        is_dormant=status is AccountStatus.DORMANT or silent_days >= 90,
        current_balance=Decimal(context["current_balance"]),
        kyc_level=KycLevel(context["kyc_level"]),
        declared_annual_income=Decimal(context["declared_annual_income"]),
        risk_rating=RiskRating(context["risk_rating"]),
        segment=CustomerSegment(context["segment"]),
        is_pep=bool(context["is_pep"]),
    )


def _vector(evidence: rules.AccountEvidence) -> FeatureVector:
    """The 7D feature row as the scorer's own input type."""
    row = evidence.row(SCORE_WINDOW)
    return FeatureVector(
        account_id=row.account_id,
        window=row.window,
        as_of=row.as_of,
        txn_count=row.txn_count,
        total_credit=row.total_credit,
        total_debit=row.total_debit,
        balance_delta=row.balance_delta,
        distinct_credit_counterparties=row.distinct_credit_counterparties,
        distinct_debit_counterparties=row.distinct_debit_counterparties,
        max_single_txn=row.max_single_txn,
        avg_txn_amount=row.avg_txn_amount,
        night_txn_ratio=row.night_txn_ratio,
        new_counterparty_ratio=row.new_counterparty_ratio,
        distinct_devices=row.distinct_devices,
        distinct_ips=row.distinct_ips,
        days_since_last_activity=row.days_since_last_activity,
        velocity_zscore=row.velocity_zscore,
    )


def _identifiers(evidence: rules.AccountEvidence) -> SharedIdentifierResult:
    """The device sharing R007 already computed, as the scorer's identity-linkage input.

    One synthetic `IdentifierLink` standing for the account's whole 7-day device footprint. The
    per-device breakdown is not available from the aggregate the detector computes, and inventing a
    split would put a fabricated device id into a value object whose whole job is provenance.
    """
    if evidence.shared_device_accounts < 1 or evidence.shared_device_customers < 1:
        return SharedIdentifierResult(account_id=evidence.account_id, shared_devices=())
    return SharedIdentifierResult(
        account_id=evidence.account_id,
        shared_devices=(
            IdentifierLink(
                identifier=f"{evidence.account_id}-device-footprint",
                linked_accounts=tuple(
                    f"linked-{index}" for index in range(evidence.shared_device_accounts)
                ),
                distinct_other_customers=evidence.shared_device_customers,
            ),
        ),
    )


def score_accounts(settings: Settings | None = None) -> ScoreRun:
    """Score every account in the batch twice -- without rule hits and with them.

    **No label is read here.** The function is deliberately label-free so `reporting/metrics.py` can
    reuse it: the scores exist before the ground truth is opened, which is the ordering the whole
    evaluation boundary rests on.
    """
    cfg = settings if settings is not None else get_settings()
    detection = DetectionConfig()
    scoring = ScoringConfig()

    table = parquet_store.read_table(cfg.data_curated_path, features.FEATURES_DIR)
    rows = features.to_account_features(table)
    as_of = rows[0].as_of
    with duckdb_engine.connect() as connection:
        facts = detect.cross_account_facts(
            connection, curated_root=cfg.data_curated_path, as_of=as_of
        )
    evidence = detect.build_evidence(rows, facts)
    context = _account_context(cfg)

    scored: list[ScoredAccount] = []
    for item in evidence:
        hits = rules.evaluate(item, detection)
        base = RiskEvidence(
            account_id=item.account_id,
            features=_vector(item),
            profile=_profile(context[item.account_id], as_of.date()),
            identifiers=_identifiers(item),
        )
        after_evidence = RiskEvidence(
            account_id=base.account_id,
            features=base.features,
            profile=base.profile,
            identifiers=base.identifiers,
            rule_hits=hits,
        )
        scored.append(
            ScoredAccount(
                account_id=item.account_id,
                score_before=assess_risk(base, scoring).score,
                score_after=assess_risk(after_evidence, scoring).score,
                rule_codes=tuple(hit.rule_code for hit in hits),
            )
        )
    log.info("scored the batch", accounts=len(scored), as_of=as_of.isoformat())
    return ScoreRun(as_of=as_of, accounts=tuple(scored))


def to_cohorts(
    run: ScoreRun, labels: Sequence[Label], scoring: ScoringConfig | None = None
) -> tuple[CohortResult, ...]:
    """Group an already-computed `ScoreRun` by labelled cohort. Pure -- no I/O."""
    config = scoring if scoring is not None else ScoringConfig()
    cohort_of = {label.account_id: label.cohort for label in labels}
    buckets: dict[str, list[ScoredAccount]] = defaultdict(list)
    for account in run.accounts:
        buckets[cohort_of.get(account.account_id, "UNLABELLED")].append(account)

    review_at = config.bands.review_at
    results = [
        CohortResult(
            cohort=cohort,
            accounts=len(scores),
            review_before=sum(1 for item in scores if item.score_before >= review_at),
            review_after=sum(1 for item in scores if item.score_after >= review_at),
            mean_before=sum(item.score_before for item in scores) / len(scores),
            mean_after=sum(item.score_after for item in scores) / len(scores),
            accounts_with_hits=sum(1 for item in scores if item.rule_codes),
        )
        for cohort, scores in buckets.items()
    ]
    return tuple(sorted(results, key=lambda r: (r.cohort == "UNLABELLED", r.cohort)))


def measure(settings: Settings | None = None) -> tuple[CohortResult, ...]:
    """Score every account twice -- without and with rule hits -- and group by labelled cohort."""
    cfg = settings if settings is not None else get_settings()
    return to_cohorts(score_accounts(cfg), read_labels(cfg))


def _print(results: Sequence[CohortResult]) -> None:
    width = max(len(result.cohort) for result in results)
    print(
        f"{'COHORT'.ljust(width)}  {'N':>6}  {'HITS':>6}  {'MEAN BEFORE':>11}  {'MEAN AFTER':>10}  "
        f"{'REVIEW BEFORE':>13}  {'REVIEW AFTER':>12}"
    )
    for result in results:
        print(
            f"{result.cohort.ljust(width)}  {result.accounts:>6,}  "
            f"{result.accounts_with_hits:>6,}  {result.mean_before:>11.2f}  "
            f"{result.mean_after:>10.2f}  {result.review_rate_before:>12.1%}  "
            f"{result.review_rate_after:>11.1%}"
        )
    print(
        "\nPartial evidence: FAN_ASYMMETRY (16) and NETWORK_STRUCTURE (16) need the Stage 8 graph "
        "and report\n`not gathered`, so 68 of 100 points are reachable. Every figure above is a "
        "floor, not an estimate."
    )
    print(f"Rule contributions in force: {[(s.code, s.contribution) for s in RULE_SPECS]}")


def main(settings: Settings | None = None) -> int:
    """`python -m sentinel.reporting.rule_impact`."""
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    _print(measure(cfg))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
