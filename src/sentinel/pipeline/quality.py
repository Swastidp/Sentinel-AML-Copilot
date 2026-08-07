"""THE DATA-QUALITY GATE — the one thing standing between the raw zone and everything downstream.

`docs/05-DATA-FLOW.md` §4: nothing reaches the curated zone until it has been validated. This module
is that sentence made executable. It reads the batch back out of the raw zone, runs the twelve
checks in `pipeline/checks.py`, decides, persists the verdict, and — if a `BLOCKING` check failed —
exits non-zero so the workflow goes red and no loader runs.

The three rules the gate enforces, and none of them is negotiable at runtime:

* **Any `BLOCKING` check failing means the run is `FAILED` and nothing is promoted.** A wrong number
  reaching a regulatory report is far worse than a late number.
* **`WARNING` failures are logged, persisted, and the batch proceeds.** A degraded graph edge is not
  a reason to hold up 200,000 valid transactions.
* **Both outcomes persist.** The run row, every check result, and the JSON artifact are written
  before the exception is raised. An audit trail that only survives success is not an audit trail —
  the failed run is the single most interesting row the table will ever hold.

A check that could not run (`SKIPPED`, for want of a rolling baseline) never satisfies the gate. It
cannot fail the batch either — nothing was found — so it downgrades the run to
`PASSED_WITH_WARNINGS`, which is the honest reading of "eight checks passed and four never ran".

**Why `DqReportStore` is declared here and not in `domain/ports/`.** That package's own docstring
rules out a port for bulk pipeline writes, on the grounds that they have one implementation and no
second candidate. DQ persistence is the exception that proves it: there are two implementations, a
filesystem one today and a Postgres one at Stage 5, and the whole reason for the seam is that
Postgres has no schema yet. A protocol consumed by exactly one module belongs next to that module,
so it lives here, and `infrastructure/storage/dq_store.py` satisfies it structurally without
importing it.

Run it with `python -m sentinel.pipeline.quality`, or `make quality`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from sentinel.config.logging import configure_logging, get_logger
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.entities import DqCheckResult, DqRun
from sentinel.domain.enums import DqCheckStatus, DqRunStatus
from sentinel.domain.exceptions import DataQualityGateFailed, DomainValidationError
from sentinel.domain.value_objects import BatchProfile
from sentinel.infrastructure.storage import parquet_store
from sentinel.infrastructure.storage.dq_store import FileSystemDqStore
from sentinel.pipeline import checks

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

log = get_logger(__name__)

#: Bumped whenever a check is added, removed, or has its logic changed. Stored in the report so a
#: historical run can be traced to the ruleset that produced it, the same reason
#: `SCORING_MODEL_VERSION` exists for the risk score.
GATE_VERSION: Final = "dq-v1"


# --- the persistence seam ----------------------------------------------------------------------


class DqReportStore(Protocol):
    """Where a gate run and its baseline go. Satisfied today by `FileSystemDqStore`.

    Four methods, and the split matters: a run is written on every execution, a profile only on one
    that was not blocked. A blocked batch is not a description of what normal looks like, and
    folding it into the rolling baseline would widen the very tolerances that just caught it.
    """

    def save_run(self, run: DqRun, results: Sequence[DqCheckResult]) -> None:
        """Persist the run and every check result, whether the batch passed or was blocked."""
        ...

    def write_report(self, report: Mapping[str, Any]) -> Path:
        """Write the JSON artifact and return where it landed."""
        ...

    def save_profile(self, profile: BatchProfile) -> None:
        """Record this batch's statistical shape as a baseline for later batches."""
        ...

    def recent_profiles(
        self, *, limit: int, exclude_batch_id: str | None = None
    ) -> tuple[BatchProfile, ...]:
        """Return recent baseline profiles, newest first, excluding the batch under validation."""
        ...


# --- the verdict -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    """One complete gate execution: what ran, what it found, and what happens next."""

    run: DqRun
    outcomes: tuple[checks.CheckOutcome, ...]
    report: Mapping[str, Any]
    profile: BatchProfile | None

    @property
    def blocked(self) -> bool:
        """True when a `BLOCKING` check failed, so nothing may be promoted to curated."""
        return self.run.status is DqRunStatus.FAILED

    @property
    def blocking_failures(self) -> tuple[checks.CheckOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.blocks_batch)

    @property
    def warnings(self) -> tuple[checks.CheckOutcome, ...]:
        """Non-blocking failures — recorded, and the batch proceeds anyway."""
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status is DqCheckStatus.FAIL and not outcome.blocks_batch
        )

    @property
    def unverified(self) -> tuple[checks.CheckOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.is_unverified)


def decide_status(outcomes: Sequence[checks.CheckOutcome]) -> DqRunStatus:
    """Map twelve outcomes onto the run's verdict. Pure.

    Order of precedence, and the middle rung is the one that earns its place: a blocking failure is
    `FAILED`; anything merely warning *or unverified* is `PASSED_WITH_WARNINGS`; only a clean sweep
    is `PASSED`. Letting a skipped check through as `PASSED` would report a first run — where four
    drift checks cannot run at all — as fully validated.
    """
    if any(outcome.blocks_batch for outcome in outcomes):
        return DqRunStatus.FAILED
    if any(outcome.status is DqCheckStatus.FAIL or outcome.is_unverified for outcome in outcomes):
        return DqRunStatus.PASSED_WITH_WARNINGS
    return DqRunStatus.PASSED


def build_report(
    *,
    run_id: str,
    batch_id: str,
    status: DqRunStatus,
    window: checks.BatchWindow,
    rows_in: int,
    rows_loaded: int,
    baseline_batches: int,
    outcomes: Sequence[checks.CheckOutcome],
    duration_ms: int,
) -> dict[str, Any]:
    """Assemble `dq_report.json`. Pure.

    The full `CheckOutcome` is rendered here, including `check_id` and `message`, which the
    `dq_check_results` table has no columns for. That is deliberate rather than a gap: the table
    mirrors `docs/02-DATA-MODEL.md` exactly so the Stage 5 loader is a straight copy, and the
    richer detail lives in the artifact an operator actually opens.
    """
    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "gate_version": GATE_VERSION,
        "status": str(status),
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "rows_in": rows_in,
        # In the artifact as well as the run row, because "was anything promoted?" is the first
        # question anyone opening a failed report has, and 0 here answers it without a join.
        "rows_loaded": rows_loaded,
        "rows_rejected": rows_in - rows_loaded,
        "baseline_batches": baseline_batches,
        "duration_ms": duration_ms,
        "summary": dict(checks.summarise(outcomes)),
        "checks": [
            {
                "check_id": outcome.check_id,
                "name": outcome.name,
                "severity": str(outcome.severity),
                "status": str(outcome.status),
                "target_table": outcome.target_table,
                "target_column": outcome.target_column,
                "expectation": outcome.expectation,
                "observed": outcome.observed,
                "message": outcome.message,
            }
            for outcome in outcomes
        ],
    }


def new_run_id() -> str:
    """A unique id per execution.

    Random rather than derived from the batch, because two runs over the same batch are two distinct
    events and the history must show both — a retry after a fixed feed is exactly the sequence an
    auditor wants to see.
    """
    return f"DQRUN-{uuid.uuid4().hex[:12]}"


# --- reading the batch -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawBatch:
    """The two tables the gate needs, read back out of the raw zone."""

    batch_id: str
    transactions: pd.DataFrame
    account_ids: frozenset[str]


def load_batch(raw_root: Path) -> RawBatch:
    """Read the transactions and the account keys back from the raw zone.

    Columns are addressed by name throughout — `txn_date` is the hive partition key and pyarrow
    appends it last rather than leaving it where the write schema put it, so any positional access
    is already wrong.
    """
    table = parquet_store.read_transactions(raw_root)
    transactions = table.to_pandas()
    if transactions.empty:
        raise DomainValidationError(
            f"the raw zone at {raw_root} holds no transactions; there is nothing to validate. "
            f"Run `python -m sentinel.pipeline.generate` first."
        )
    accounts = parquet_store.read_table(raw_root, parquet_store.ACCOUNTS_DIR)
    batch_ids = set(transactions["batch_id"].dropna().unique())
    if len(batch_ids) != 1:
        # The gate validates one batch. Two batch ids in the raw zone means a previous run was
        # partially overwritten, and every count the gate reports would mix them.
        raise DomainValidationError(
            f"expected exactly one batch_id in the raw zone, found {len(batch_ids)}: "
            f"{sorted(batch_ids)}."
        )
    return RawBatch(
        batch_id=str(next(iter(batch_ids))),
        transactions=transactions,
        account_ids=frozenset(str(value) for value in accounts.column("account_id").to_pylist()),
    )


# --- the gate ----------------------------------------------------------------------------------


def evaluate(batch: RawBatch, store: DqReportStore, settings: Settings) -> GateResult:
    """Run the twelve checks over one batch and persist the verdict. Does **not** raise on failure.

    Split from `run_gate` so a caller that wants the result object — a test, a future Airflow
    operator, the Streamlit data-quality page — gets it without having to catch an exception to
    read it. `run_gate` is the batch-entrypoint form that turns a blocked batch into a raised
    `DataQualityGateFailed`.
    """
    run_id = new_run_id()
    started_at = datetime.now(UTC)
    clock = time.perf_counter()

    baseline = store.recent_profiles(
        limit=settings.dq_baseline_batches, exclude_batch_id=batch.batch_id
    )
    context = checks.build_context(
        batch_id=batch.batch_id,
        transactions=batch.transactions,
        account_ids=batch.account_ids,
        baseline=baseline,
        settings=settings,
    )
    outcomes = checks.run_all(context)
    status = decide_status(outcomes)
    finished_at = datetime.now(UTC)
    duration_ms = int((time.perf_counter() - clock) * 1000)

    rows_in = context.row_count
    # The gate promotes whole batches, never rows: it holds no notion of a repairable row, so a
    # blocked batch loads nothing and a passing one loads everything. `rows_rejected` is therefore
    # all-or-nothing by design rather than by omission.
    rows_loaded = 0 if status is DqRunStatus.FAILED else rows_in
    report = build_report(
        run_id=run_id,
        batch_id=batch.batch_id,
        status=status,
        window=context.window,
        rows_in=rows_in,
        rows_loaded=rows_loaded,
        baseline_batches=len(baseline),
        outcomes=outcomes,
        duration_ms=duration_ms,
    )
    run = DqRun(
        run_id=run_id,
        batch_id=batch.batch_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        rows_in=rows_in,
        rows_loaded=rows_loaded,
        rows_rejected=rows_in - rows_loaded,
        checks_total=len(outcomes),
        checks_failed=sum(1 for o in outcomes if o.status is DqCheckStatus.FAIL),
        report=report,
    )
    results = tuple(_to_check_result(run_id, outcome) for outcome in outcomes)

    # Persist first, decide afterwards. The audit trail has to survive the failure, so a blocked
    # batch must never take the report down with it.
    store.save_run(run, results)
    report_path = store.write_report(report)

    profile: BatchProfile | None = None
    if status is not DqRunStatus.FAILED:
        profile = checks.compute_profile(context, finished_at)
        store.save_profile(profile)

    _log_outcomes(run, outcomes, report_path)
    return GateResult(run=run, outcomes=outcomes, report=report, profile=profile)


def run_gate(settings: Settings | None = None, *, store: DqReportStore | None = None) -> GateResult:
    """Validate the raw zone. Raises `DataQualityGateFailed` when a `BLOCKING` check failed.

    The exception is raised **after** everything has been persisted, exactly as
    `docs/05-DATA-FLOW.md` §4 sketches it. Callers that would rather branch than catch should use
    `evaluate` and read `GateResult.blocked`.
    """
    cfg = settings if settings is not None else get_settings()
    target = store if store is not None else FileSystemDqStore(cfg.data_quality_path)
    batch = load_batch(cfg.data_raw_path)
    result = evaluate(batch, target, cfg)
    if result.blocked:
        raise DataQualityGateFailed(
            run_id=result.run.run_id,
            batch_id=result.run.batch_id,
            failures=tuple(outcome.check_id for outcome in result.blocking_failures),
        )
    return result


# --- helpers -----------------------------------------------------------------------------------


def _to_check_result(run_id: str, outcome: checks.CheckOutcome) -> DqCheckResult:
    """Map a pure `CheckOutcome` onto the persisted `dq_check_results` row shape."""
    return DqCheckResult(
        result_id=None,
        run_id=run_id,
        check_name=outcome.check_id,
        target_table=outcome.target_table,
        target_column=outcome.target_column,
        severity=outcome.severity,
        status=outcome.status,
        expectation=outcome.expectation,
        observed=outcome.observed,
    )


def _log_outcomes(run: DqRun, outcomes: Sequence[checks.CheckOutcome], report_path: Path) -> None:
    """One line per non-passing check, then one summary line.

    Warnings are logged individually and not merely counted: a `WARNING` failure lets the batch
    through, so the log entry is the only place anyone will ever see it.
    """
    for outcome in outcomes:
        if outcome.blocks_batch:
            log.error("dq check failed", check=outcome.check_id, detail=outcome.message)
        elif outcome.status is DqCheckStatus.FAIL:
            log.warning("dq check warning", check=outcome.check_id, detail=outcome.message)
        elif outcome.is_unverified:
            log.warning("dq check skipped", check=outcome.check_id, detail=outcome.message)

    log.info(
        "data-quality gate complete",
        run_id=run.run_id,
        batch_id=run.batch_id,
        status=str(run.status),
        rows_in=run.rows_in,
        rows_loaded=run.rows_loaded,
        **dict(checks.summarise(outcomes)),
        report=str(report_path),
    )


# --- entrypoint --------------------------------------------------------------------------------


def main(settings: Settings | None = None) -> None:
    """`python -m sentinel.pipeline.quality`. Exits 1 when a `BLOCKING` check failed.

    `SystemExit(1)` rather than a bare `raise`, because the exit code is the contract with GitHub
    Actions: the workflow step goes red, no loader step runs, and the persisted report is still
    there to be uploaded as an artifact. The optional `settings` argument exists so a test can point
    the gate at a temporary lakehouse without monkeypatching the settings singleton.
    """
    cfg = settings if settings is not None else get_settings()
    configure_logging(cfg)
    cfg.ensure_data_dirs()
    try:
        run_gate(cfg)
    except DataQualityGateFailed as exc:
        log.error(
            "batch blocked by the data-quality gate",
            run_id=exc.run_id,
            batch_id=exc.batch_id,
            failed_checks=list(exc.failures),
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    # No `sys.exit(main())`: `main` signals failure by raising `SystemExit(1)` itself, which the
    # interpreter turns into the exit code. Wrapping it would only add a redundant
    # `SystemExit(None)` on the success path.
    main()
