"""Run the rule engine and the anomaly model over the batch and write `alerts` -- T6.3.

    python -m sentinel.pipeline.detect

Reads `data/curated/account_features/account_features.parquet`, joins the two cross-account facts
the rules need, evaluates R001-R007, fits and persists the Isolation Forest, and upserts one alert
per finding into Neon Postgres through `PostgresAlertRepository`.

**Idempotent, and the mechanism matters.** `alert_id` is derived from
`(account_id, rule_code, window_end)`, so re-running the same batch collides with every row it
wrote before and updates it in place -- identical counts, no duplicates. `save_many` deliberately
leaves `status` and `created_at` out of the update, so a re-run cannot drag an alert an analyst has
already moved back to `OPEN`.

**Every timestamp comes from the data, never from the clock.** `triggered_at` and `created_at` are
both the feature set's `as_of`, which `pipeline/features.resolve_as_of` reads off
`MAX(ingested_at)`. That is what makes two runs over the same bytes produce byte-identical alert
rows -- a detector whose output changes because a day passed cannot be replayed or audited.

**`typology_labels` is not read here, or anywhere it could be.** The labels are withheld from the
curated zone by `pipeline/promote.py`, so this module reads a lakehouse that physically does not
contain them.

---

**The two cross-account facts, and why they live in this module rather than in a rule.**

`closing_balance` -- R006's second condition is a *stock*, and a feature row holds only *flow*. The
batch carries `accounts.current_balance`, which `pipeline/generate.py` documents as a snapshot from
before the 90-day window, so it is an opening figure; it is rolled forward through the settled
ledger here. `AccountEvidence` records what that reconstruction is worth on synthetic data.

`shared_device_accounts` -- R007 asks whether this account shares a handset with accounts belonging
to *other customers*, which is a relationship between accounts and cannot come off one row. It is
computed over the same 7-day window the other rules read. Measured on the real batch the 90-day
reading fires R007 on 45% of the book and the 7-day reading on 1.4%; the generator's device pool
(2,500 fingerprints) is exhausted by the enrolled population and wraps, manufacturing
cross-customer sharing that widens the further back you look. That is a Stage 2 data property,
recorded here rather than papered over by moving R007's threshold.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.config.weights import ML_SPEC, RULE_ENGINE_VERSION, DetectionConfig
from sentinel.domain.entities import Alert
from sentinel.domain.enums import AlertStatus
from sentinel.domain.exceptions import DomainValidationError
from sentinel.infrastructure.postgres import connection as pg
from sentinel.infrastructure.postgres import migrate
from sentinel.infrastructure.postgres.repositories import PostgresAlertRepository
from sentinel.infrastructure.storage import duckdb_engine, parquet_store
from sentinel.pipeline import anomaly, features, rules

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import duckdb

    from sentinel.domain.entities import AccountFeatures
    from sentinel.domain.enums import FeatureWindow
    from sentinel.domain.value_objects import RuleHit

log = get_logger(__name__)

#: The anomaly score at or above which an account with no rule hit still gets an alert. Exactly the
#: outlier boundary of `decision_function`, by construction of `anomaly.to_anomaly_score`, so this
#: is not a second tunable threshold -- it is the same `IsolationForestConfig.contamination`
#: decision expressed on the mapped scale. Changing it here without changing contamination would
#: make the alert set and the model's own verdict disagree.
ML_ALERT_SCORE: Final = 50.0

#: The window R007's device sharing and the anomaly fit are measured over. One constant so the SQL,
#: the model and the alert windows cannot drift apart.
DETECTION_WINDOW: Final = anomaly.FIT_WINDOW


def _cross_account_sql() -> str:
    """Closing balance and cross-customer device sharing, in one pass over the ledger.

    One query rather than two because both unpivot the same 205,000-row ledger into account legs,
    and running that twice to avoid a shared CTE would double the only expensive part of the stage.

    The `legs` CTE is the same unpivot `pipeline/features.py` documents as trap 1: `direction`
    records the *initiating* side, so filtering on it builds a table that is right for payers and
    silently wrong for every payee.
    """
    ledger = duckdb_engine.read_partitioned("$txn_glob")
    return f"""
WITH params AS (
    SELECT CAST($as_of AS TIMESTAMPTZ)                  AS as_of,
           CAST($as_of AS TIMESTAMPTZ)
               - to_days(CAST($window_days AS INTEGER)) AS window_start
),
settled AS (
    SELECT t.txn_ts, t.from_account, t.to_account, t.amount, t.device_id
    FROM {ledger} t, params p
    WHERE t.status = 'SUCCESS' AND t.txn_ts < p.as_of
),
legs AS (
    SELECT from_account AS account_id, -amount AS signed_amount, txn_ts, device_id
    FROM settled WHERE from_account IS NOT NULL
    UNION ALL
    SELECT to_account AS account_id, amount AS signed_amount, txn_ts, device_id
    FROM settled WHERE to_account IS NOT NULL
),
-- The whole settled ledger, not the window: `accounts.current_balance` is a snapshot from before
-- the batch, so the closing figure is the opening one rolled forward through everything that has
-- since settled.
movement AS (
    SELECT account_id, SUM(signed_amount) AS net FROM legs GROUP BY account_id
),
window_devices AS (
    SELECT DISTINCT l.account_id, l.device_id
    FROM legs l, params p
    WHERE l.device_id IS NOT NULL AND l.txn_ts >= p.window_start AND l.txn_ts < p.as_of
),
-- R007 counts accounts of *other customers* only. The customer inequality is the load-bearing
-- line: without it a customer's own two phones read as a ring.
links AS (
    SELECT mine.account_id,
           COUNT(DISTINCT theirs.account_id)  AS other_accounts,
           COUNT(DISTINCT owner2.customer_id) AS other_customers
    FROM window_devices mine
    JOIN read_parquet($accounts_file) owner1 ON owner1.account_id = mine.account_id
    JOIN window_devices theirs
      ON theirs.device_id = mine.device_id AND theirs.account_id <> mine.account_id
    JOIN read_parquet($accounts_file) owner2
      ON owner2.account_id = theirs.account_id AND owner2.customer_id <> owner1.customer_id
    GROUP BY mine.account_id
)
SELECT a.account_id,
       CAST(a.current_balance + COALESCE(m.net, 0) AS DECIMAL(18, 2)) AS closing_balance,
       CAST(COALESCE(k.other_accounts, 0) AS INTEGER)                 AS shared_device_accounts,
       CAST(COALESCE(k.other_customers, 0) AS INTEGER)                AS shared_device_customers
FROM read_parquet($accounts_file) a
LEFT JOIN movement m ON m.account_id = a.account_id
LEFT JOIN links    k ON k.account_id = a.account_id
ORDER BY a.account_id
"""


@dataclass(frozen=True, slots=True)
class CrossAccountFact:
    """The two things about an account that no single feature row can hold."""

    closing_balance: Decimal
    shared_device_accounts: int
    shared_device_customers: int


def cross_account_facts(
    connection: duckdb.DuckDBPyConnection, *, curated_root: Path, as_of: datetime
) -> dict[str, CrossAccountFact]:
    """Run the join pass and return one fact per account in the book."""
    rows = connection.execute(
        _cross_account_sql(),
        {
            "as_of": as_of,
            "window_days": features.WINDOW_DAYS[DETECTION_WINDOW],
            "txn_glob": duckdb_engine.dataset_glob(curated_root, parquet_store.TRANSACTIONS_DIR),
            "accounts_file": duckdb_engine.table_file(curated_root, parquet_store.ACCOUNTS_DIR),
        },
    ).fetchall()
    return {
        str(row[0]): CrossAccountFact(
            closing_balance=Decimal(row[1]),
            shared_device_accounts=int(row[2]),
            shared_device_customers=int(row[3]),
        )
        for row in rows
    }


def build_evidence(
    rows: Sequence[AccountFeatures], facts: Mapping[str, CrossAccountFact]
) -> tuple[rules.AccountEvidence, ...]:
    """Group feature rows by account and attach the cross-account facts.

    Ordered by `account_id` so the alert set is byte-identical between runs. An account with
    feature rows but no fact row is a join failure rather than a quiet zero: the fact query is
    driven from the same `accounts` table the feature query is, so a gap means the two read
    different files.
    """
    grouped: dict[str, dict[FeatureWindow, AccountFeatures]] = {}
    for row in rows:
        grouped.setdefault(row.account_id, {})[row.window] = row
    missing = sorted(set(grouped) - set(facts))
    if missing:
        raise DomainValidationError(
            f"{len(missing)} account(s) have feature rows but no cross-account facts, starting "
            f"with {missing[:3]!r}; the feature file and the accounts table disagree about the "
            f"book."
        )
    return tuple(
        rules.AccountEvidence(
            account_id=account_id,
            rows=windows,
            closing_balance=facts[account_id].closing_balance,
            shared_device_accounts=facts[account_id].shared_device_accounts,
            shared_device_customers=facts[account_id].shared_device_customers,
        )
        for account_id, windows in sorted(grouped.items())
    )


def alert_id(account_id: str, rule_code: str, window_end: datetime) -> str:
    """`ALRT-<account>-<rule>-<yyyymmdd>`. Deterministic, readable, and the upsert key.

    Readable rather than a hash on purpose: an analyst reading a row in `psql`, or an engineer
    reading a log line, can see which account and which rule it came from without a lookup. The
    date is what keeps tomorrow's batch from overwriting today's alert on the same account and
    rule.
    """
    return f"ALRT-{account_id}-{rule_code}-{window_end:%Y%m%d}"


def _evidence_payload(
    reason: str,
    detail: Mapping[str, Any],
    *,
    anomaly_score: float | None,
    decision_function: float | None,
) -> dict[str, Any]:
    """The JSONB the agent starts from: the rule's own numbers plus the model's opinion.

    `reason` is the sentence, and it is mandatory and non-empty -- T6.3's acceptance criterion is
    that every alert carries at least one. The version stamps are here so an alert raised six
    months ago can be traced to the thresholds and the model that raised it, which a `rule_code`
    alone cannot do.
    """
    if not reason.strip():
        raise DomainValidationError(
            "an alert must carry a non-empty reason; an unexplained alert costs an analyst the "
            "20 minutes this system exists to save."
        )
    return {
        "reason": reason,
        "rule_engine_version": RULE_ENGINE_VERSION,
        "anomaly_model_version": anomaly.ANOMALY_MODEL_VERSION,
        "anomaly_score": anomaly_score,
        "anomaly_decision_function": decision_function,
        **dict(detail),
    }


def build_alerts(
    evidence: Sequence[rules.AccountEvidence],
    hits: Mapping[str, tuple[RuleHit, ...]],
    scores: Mapping[str, float],
    decisions: Mapping[str, float],
    *,
    as_of: datetime,
) -> tuple[Alert, ...]:
    """One alert per rule hit, plus one per model-only outlier. Pure: no I/O, no clock.

    **Model-only alerts carry `rule_score = 0.0`, and that is deliberate rather than lazy.** The
    Isolation Forest is unsupervised: it says "unlike the rest of the book", never "this pattern",
    so there is no named contribution it could honestly make to a score whose whole promise is that
    every point is explainable. The anomaly score rides on the row where an analyst can see it and
    reaches the risk scorer through nothing at all -- which is why `ML_SPEC.contribution` is zero.
    """
    alerts: list[Alert] = []
    for item in evidence:
        account_hits = hits.get(item.account_id, ())
        score = scores.get(item.account_id)
        decision = decisions.get(item.account_id)
        for hit in account_hits:
            alerts.append(
                Alert(
                    alert_id=alert_id(item.account_id, hit.rule_code, hit.window_end),
                    account_id=item.account_id,
                    rule_code=hit.rule_code,
                    rule_name=hit.rule_name,
                    triggered_at=as_of,
                    window_start=hit.window_start,
                    window_end=hit.window_end,
                    severity=hit.severity,
                    rule_score=hit.contribution,
                    anomaly_score=score,
                    trigger_evidence=_evidence_payload(
                        hit.detail,
                        hit.evidence,
                        anomaly_score=score,
                        decision_function=decision,
                    ),
                    status=AlertStatus.OPEN,
                    created_at=as_of,
                )
            )
        if account_hits or score is None or score < ML_ALERT_SCORE:
            continue
        row = item.row(DETECTION_WINDOW)
        start, end = rules.window_bounds(row)
        alerts.append(
            Alert(
                alert_id=alert_id(item.account_id, ML_SPEC.code, end),
                account_id=item.account_id,
                rule_code=ML_SPEC.code,
                rule_name=ML_SPEC.name,
                triggered_at=as_of,
                window_start=start,
                window_end=end,
                severity=ML_SPEC.severity,
                rule_score=ML_SPEC.contribution,
                anomaly_score=score,
                trigger_evidence=_evidence_payload(
                    (
                        f"no rule fired, but the account is in the most anomalous band at score "
                        f"{score:.1f} of 100 ({row.txn_count} transactions in the {row.window} "
                        f"window)"
                    ),
                    {
                        "txn_count": row.txn_count,
                        "total_credit_inr": str(row.total_credit),
                        "total_debit_inr": str(row.total_debit),
                        "pass_through_ratio": round(row.pass_through_ratio, 4),
                    },
                    anomaly_score=score,
                    decision_function=decision,
                ),
                status=AlertStatus.OPEN,
                created_at=as_of,
            )
        )
    return tuple(alerts)


@dataclass(frozen=True, slots=True)
class DetectionReport:
    """What one detection run did, in the numbers worth printing and asserting on."""

    as_of: datetime
    accounts: int
    hits_by_rule: Mapping[str, int]
    accounts_with_hits: int
    model_only_alerts: int
    alerts_written: int
    alerts_in_postgres: int
    model_path: Path
    duration_ms: int

    @property
    def total_hits(self) -> int:
        return sum(self.hits_by_rule.values())


def detect(settings: Settings | None = None) -> DetectionReport:
    """Fit, evaluate, and upsert. The whole of Stage 6 as one idempotent step.

    The migration runs first and is a no-op when the schema is already there -- the same ordering
    `pipeline/load.run` uses, and for the same reason: it is cheaper than a "did you bootstrap?"
    error message.
    """
    cfg = settings if settings is not None else get_settings()
    config = DetectionConfig()
    clock = time.perf_counter()

    table = parquet_store.read_table(cfg.data_curated_path, features.FEATURES_DIR)
    rows = features.to_account_features(table)
    if not rows:
        raise DomainValidationError(
            f"the feature file under {cfg.data_curated_path / features.FEATURES_DIR} is empty. "
            f"Run `make features` first."
        )
    as_of = rows[0].as_of

    with duckdb_engine.connect() as connection:
        facts = cross_account_facts(connection, curated_root=cfg.data_curated_path, as_of=as_of)
    evidence = build_evidence(rows, facts)

    hits = {item.account_id: found for item in evidence if (found := rules.evaluate(item, config))}
    tally = Counter(hit.rule_code for found in hits.values() for hit in found)

    matrix = anomaly.build_matrix(table, DETECTION_WINDOW)
    forest = anomaly.fit(matrix, config.forest)
    model_path = anomaly.save(forest, cfg.model_path)
    scores = anomaly.score(forest, matrix)
    decisions = anomaly.raw_decision(forest, matrix)

    alerts = build_alerts(evidence, hits, scores, decisions, as_of=as_of)
    model_only = sum(1 for alert in alerts if alert.rule_code == ML_SPEC.code)

    with pg.connect(cfg) as connection_pg:
        state = migrate.apply(connection_pg)
        log.info("schema ready", summary=state.summary())
        repository = PostgresAlertRepository(connection_pg)
        written = repository.save_many(alerts)
        stored = repository.count()

    report = DetectionReport(
        as_of=as_of,
        accounts=len(evidence),
        hits_by_rule={spec.code: tally[spec.code] for spec in config.specs},
        accounts_with_hits=len(hits),
        model_only_alerts=model_only,
        alerts_written=written,
        alerts_in_postgres=stored,
        model_path=model_path,
        duration_ms=int((time.perf_counter() - clock) * 1000),
    )
    log.info(
        "detection run complete",
        rule_engine=RULE_ENGINE_VERSION,
        anomaly_model=anomaly.ANOMALY_MODEL_VERSION,
        as_of=as_of.isoformat(),
        accounts=report.accounts,
        hits=report.total_hits,
        accounts_with_hits=report.accounts_with_hits,
        model_only_alerts=report.model_only_alerts,
        alerts=report.alerts_written,
        duration_ms=report.duration_ms,
    )
    return report


def main(settings: Settings | None = None) -> int:
    """`python -m sentinel.pipeline.detect`, or `make detect`."""
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    cfg.ensure_data_dirs()
    report = detect(cfg)

    config = DetectionConfig()
    print(f"as_of {report.as_of.isoformat()}  |  {report.accounts:,} accounts")
    print(f"\n{'RULE':<6}  {'NAME':<24}  {'SEVERITY':<9}  {'CONTRIB':>7}  {'ACCOUNTS':>9}")
    for spec in (*config.specs, ML_SPEC):
        fired = (
            report.model_only_alerts
            if spec.code == ML_SPEC.code
            else report.hits_by_rule.get(spec.code, 0)
        )
        print(
            f"{spec.code:<6}  {spec.name:<24}  {spec.severity!s:<9}  "
            f"{spec.contribution:>7.0f}  {fired:>9,}"
        )
    print(
        f"\n{report.total_hits:,} rule hit(s) across {report.accounts_with_hits:,} account(s)"
        f"  |  {report.model_only_alerts:,} model-only alert(s)"
    )
    print(
        f"alerts upserted: {report.alerts_written:,}  |  "
        f"alerts in Postgres: {report.alerts_in_postgres:,}"
    )
    print(f"model: {report.model_path}  |  {report.duration_ms:,} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
