"""Live-database tests for Stage 6 -- the first rows `alerts` has ever held.

`PostgresAlertRepository` was written at T5.2 and, until this file existed, had never executed
against the database: the table was empty, so nothing exercised the reads, and there was no writer
at all. These are its acceptance tests, written the way T6.3 words it:

* "alerts in Postgres, each with at least one reason" -- read the rows back and assert the reason.
* "upsert semantics so a re-run leaves identical counts" -- run detection twice and compare, then
  move an alert's status by hand and prove a third run does not reset it.

**Nothing here deletes anything.** `DATABASE_URL` points at a live database and the project rule is
additive-only. The status test therefore moves one alert forward and puts it back through the same
public method rather than deleting a row, and the alert it picks is the one detection would rewrite
anyway.

They are slow: the detection run reads 205,419 transactions through DuckDB and fits the forest on
every run. Neon auto-suspends, so a cold first connection takes around 16 seconds against 3 warm --
that is the database waking up, not a failure.
"""

from __future__ import annotations

import pytest

from sentinel.config.settings import get_settings
from sentinel.config.weights import ML_SPEC, RULE_SPECS
from sentinel.domain.enums import AlertStatus
from sentinel.domain.exceptions import ConfigurationError
from sentinel.infrastructure.postgres import connection as pg
from sentinel.infrastructure.postgres.repositories import PostgresAlertRepository
from sentinel.pipeline import detect

pytestmark = pytest.mark.integration

#: Every code an alert row may legitimately carry.
KNOWN_CODES = {spec.code for spec in RULE_SPECS} | {ML_SPEC.code}


@pytest.fixture(scope="module")
def settings():
    cfg = get_settings()
    try:
        assert cfg.postgres_dsn
    except ConfigurationError as exc:
        pytest.skip(f"DATABASE_URL is not configured: {exc}")
    if not (cfg.data_curated_path / "account_features" / "account_features.parquet").exists():
        pytest.skip("no curated feature file on disk; run `make features` first")
    return cfg


@pytest.fixture(scope="module")
def first_run(settings):
    return detect.detect(settings)


def test_detection_writes_alerts_and_every_one_has_a_reason(settings, first_run):
    assert first_run.alerts_written > 0
    assert first_run.alerts_in_postgres >= first_run.alerts_written
    with pg.connect(settings) as connection:
        alerts = PostgresAlertRepository(connection).list_open(limit=500)
    assert alerts
    for alert in alerts:
        assert alert.rule_code in KNOWN_CODES
        assert alert.trigger_evidence["reason"].strip()
        assert alert.window_end >= alert.window_start
        # The anomaly score is the mapped 0-100 figure; the raw `decision_function` value rides
        # alongside it in the evidence so an analyst can reproduce the transformation.
        assert alert.anomaly_score is not None
        assert 0.0 <= alert.anomaly_score <= 100.0
        assert "anomaly_decision_function" in alert.trigger_evidence


def test_rerunning_detection_leaves_identical_counts(settings, first_run):
    """T6.3's idempotency claim, asserted rather than documented."""
    second = detect.detect(settings)
    assert second.alerts_written == first_run.alerts_written
    assert second.alerts_in_postgres == first_run.alerts_in_postgres
    assert second.hits_by_rule == first_run.hits_by_rule
    assert second.model_only_alerts == first_run.model_only_alerts


def test_a_rerun_does_not_reset_a_status_an_analyst_moved(settings, first_run):
    """The reason `status` is left out of the upsert's `SET` list.

    Without this, re-running detection would drag every alert an analyst had opened -- or the human
    gate had parked in `AWAITING_DECISION` -- back to `OPEN`, silently undoing human work.
    """
    with pg.connect(settings) as connection:
        repository = PostgresAlertRepository(connection)
        target = repository.list_open(limit=1)[0]
        original = target.status
        repository.update_status(target.alert_id, AlertStatus.IN_REVIEW)

    detect.detect(settings)

    with pg.connect(settings) as connection:
        repository = PostgresAlertRepository(connection)
        after = repository.get(target.alert_id)
        assert after is not None
        assert after.status is AlertStatus.IN_REVIEW
        # Additive-only: put it back through the same public method rather than deleting the row.
        repository.update_status(target.alert_id, original)


def test_the_queue_can_be_filtered_by_severity(settings, first_run):
    """`list_open` has never run against real rows before this stage. Its `severity` filter and its
    severity-ranked ordering are only meaningful once the table holds more than one kind of alert.
    """
    with pg.connect(settings) as connection:
        repository = PostgresAlertRepository(connection)
        everything = repository.list_open(limit=50)
        high = repository.list_open(limit=50, severity=everything[0].severity)
    assert high
    assert {alert.severity for alert in high} == {everything[0].severity}
