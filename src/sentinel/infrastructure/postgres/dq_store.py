"""The Postgres half of data-quality persistence -- the Stage 5 side of the Stage 3 seam.

`pipeline/quality.py` declares a local `DqReportStore` protocol and talks to nothing else. Stage 3
satisfied it with `infrastructure/storage/dq_store.FileSystemDqStore` because Postgres had no schema
yet, and said in writing that Stage 5 would be "a new class satisfying that protocol and one line at
the composition root; no check and no part of the runner changes". This is that class, and nothing
in `quality.py` moved to make it fit.

**It composes the filesystem store rather than replacing it**, because only two of the protocol's
four methods have a database to go to:

| method | goes to |
|---|---|
| `save_run` | Postgres `dq_runs` + `dq_check_results` |
| `write_report` | the filesystem -- `dq_report.json` is a CI artifact a workflow step uploads |
| `save_profile` | the filesystem -- a `BatchProfile` is **not** one of the twelve tables |
| `recent_profiles` | the filesystem, same reason |

Inventing a thirteenth table for batch profiles was the alternative and it was rejected: a profile
is a handful of nested maps read one at a time by one caller, `docs/02-DATA-MODEL.md` does not model
it, and adding a table to the system of record to avoid a two-line delegation is the wrong trade.
The delegation is explicit here rather than hidden, so a reader can see exactly which half of the
gate's output reaches the database.

**Both writes upsert.** A gate re-run against the same batch produces a new `run_id`, so collisions
are rare -- but a retried GitHub Actions step replaying an identical run must leave one row and
twelve results, not two and twenty-four. `dq_check_results` conflicts on `(run_id, check_name)`
rather than on its `bigserial`, which is exactly why that unique constraint exists.

**`result_id` is never sent.** The column is a `bigserial` and Postgres assigns it, matching the
null `result_id` the Parquet writer leaves for precisely this moment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from sentinel.config.logging import get_logger
from sentinel.infrastructure.postgres import connection as pg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import psycopg

    from sentinel.domain.entities import DqCheckResult, DqRun
    from sentinel.domain.value_objects import BatchProfile
    from sentinel.infrastructure.storage.dq_store import FileSystemDqStore
    from sentinel.pipeline.quality import DqReportStore

log = get_logger(__name__)


class PostgresDqStore:
    """`DqReportStore` writing runs and check results to Postgres, profiles to the filesystem.

    Structurally typed against `pipeline.quality.DqReportStore` and does not import it at runtime --
    the same arrangement as every other adapter in this package. `_conformance` at the bottom of
    this module is what makes mypy check the four signatures against the protocol.
    """

    def __init__(self, connection: psycopg.Connection[Any], artifacts: FileSystemDqStore) -> None:
        """`artifacts` handles the report file and the baseline profiles; see the module docstring.

        It is a required argument rather than an optional one with a default path. A gate that
        silently wrote its baselines somewhere nobody configured would make the four drift checks
        skip forever, and a permanently-skipping check is the failure mode `DqCheckStatus.SKIPPED`
        exists to make visible.
        """
        self._connection = connection
        self._artifacts = artifacts

    # --- Postgres ---------------------------------------------------------------------------

    def save_run(self, run: DqRun, results: Sequence[DqCheckResult]) -> None:
        """Persist the run and every check result in one transaction.

        Called for a blocked batch as well as a clean one -- especially then. Nothing reaches the
        curated zone, but the record of why survives, and it has to survive *atomically*: a run row
        with no check results is an audit trail that says a batch was blocked and cannot say by
        what.
        """
        with pg.wrap(f"dq run {run.run_id}"), self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO dq_runs (
                    run_id, batch_id, started_at, finished_at, status,
                    rows_in, rows_loaded, rows_rejected, checks_total, checks_failed, report
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    batch_id      = EXCLUDED.batch_id,
                    started_at    = EXCLUDED.started_at,
                    finished_at   = EXCLUDED.finished_at,
                    status        = EXCLUDED.status,
                    rows_in       = EXCLUDED.rows_in,
                    rows_loaded   = EXCLUDED.rows_loaded,
                    rows_rejected = EXCLUDED.rows_rejected,
                    checks_total  = EXCLUDED.checks_total,
                    checks_failed = EXCLUDED.checks_failed,
                    report        = EXCLUDED.report
                """,
                (
                    run.run_id,
                    run.batch_id,
                    run.started_at,
                    run.finished_at,
                    str(run.status),
                    run.rows_in,
                    run.rows_loaded,
                    run.rows_rejected,
                    run.checks_total,
                    run.checks_failed,
                    Jsonb(dict(run.report)),
                ),
            )
            if results:
                cursor.executemany(
                    """
                    INSERT INTO dq_check_results (
                        run_id, check_name, target_table, target_column,
                        severity, status, expectation, observed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, check_name) DO UPDATE SET
                        target_table  = EXCLUDED.target_table,
                        target_column = EXCLUDED.target_column,
                        severity      = EXCLUDED.severity,
                        status        = EXCLUDED.status,
                        expectation   = EXCLUDED.expectation,
                        observed      = EXCLUDED.observed
                    """,
                    [
                        (
                            result.run_id,
                            result.check_name,
                            result.target_table,
                            result.target_column,
                            str(result.severity),
                            str(result.status),
                            result.expectation,
                            result.observed,
                        )
                        for result in results
                    ],
                )
        self._connection.commit()
        log.info(
            "dq run persisted to postgres",
            run_id=run.run_id,
            batch_id=run.batch_id,
            status=str(run.status),
            checks=len(results),
        )

    # --- delegated to the filesystem ----------------------------------------------------------

    def write_report(self, report: Mapping[str, Any]) -> Path:
        """The CI artifact. A file, because a GitHub Actions step uploads a path, not a row."""
        return self._artifacts.write_report(report)

    def save_profile(self, profile: BatchProfile) -> None:
        """Baselines stay on disk -- `BatchProfile` is not one of the twelve modelled tables."""
        self._artifacts.save_profile(profile)

    def recent_profiles(
        self, *, limit: int, exclude_batch_id: str | None = None
    ) -> tuple[BatchProfile, ...]:
        """Read the baselines back from wherever `save_profile` put them."""
        return self._artifacts.recent_profiles(limit=limit, exclude_batch_id=exclude_batch_id)


if TYPE_CHECKING:  # pragma: no cover - a static assertion, never imported or executed

    def _conformance(
        connection: psycopg.Connection[Any], artifacts: FileSystemDqStore
    ) -> DqReportStore:
        """**This is the drop-in claim, checked.** It never runs.

        Stage 3 asserted that a Postgres adapter would satisfy `pipeline.quality.DqReportStore`
        without the runner changing. `mypy --strict` verifies that here, against the protocol as it
        is actually declared, rather than leaving it as a sentence in a docstring. `DqReportStore`
        is not `@runtime_checkable`, so this static binding is the whole of the check -- there is no
        `isinstance` fallback for it.
        """
        return PostgresDqStore(connection, artifacts)
