"""Filesystem persistence for data-quality runs, check results, and batch profiles.

`dq_runs` and `dq_check_results` are Postgres tables in `docs/02-DATA-MODEL.md`, and Postgres has no
schema until Stage 5. The gate cannot wait for it: a run that is not recorded is a gate with no
audit trail, which is most of the point of having one. So Stage 3 writes the same two shapes to
`data/quality/` as Parquet, plus a `dq_report.json` artifact for GitHub Actions to upload, and the
runner talks to this through a protocol it defines (`pipeline/quality.py:DqReportStore`). Swapping
in a Postgres adapter at Stage 5 is a new class satisfying that protocol and one line at the
composition root; no check and no part of the runner changes.

The Parquet column sets mirror the two tables exactly — same names, same order — so the Stage 5
adapter is a `COPY` of these files and not a translation.

Three details worth stating:

**`result_id` is written as null.** The column is a `bigserial` in Postgres and the database assigns
it. Inventing a sequence here would produce ids that collide with the real ones the moment the table
exists.

**Batch profiles are JSON, not Parquet.** A profile is a handful of nested maps (null rate per
column, share per channel, a decile ladder) read one at a time by exactly one caller. Parquet's
column store buys nothing at that shape and size, and the JSON is legible when someone wants to know
why a drift check fired. It is also **not** one of the twelve tables, so nothing about Stage 5
depends on its layout.

**Decimals are written as strings.** JSON has one numeric type and it is a binary float. A rupee
decile boundary round-tripped through `float` would come back a hair off, and the whole storage
layer above this refuses to put binary floating point near money.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from sentinel.config.logging import get_logger
from sentinel.domain.value_objects import BatchProfile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sentinel.domain.entities import DqCheckResult, DqRun

log = get_logger(__name__)

#: The artifact GitHub Actions uploads. A fixed name, overwritten each run, because a workflow step
#: cannot glob for a filename it does not know.
REPORT_FILENAME: Final = "dq_report.json"

RUNS_DIR: Final = "dq_runs"
RESULTS_DIR: Final = "dq_check_results"
PROFILES_DIR: Final = "batch_profiles"

_TS: Final = pa.timestamp("us", tz="UTC")

RUN_SCHEMA: Final = pa.schema(
    [
        ("run_id", pa.string()),
        ("batch_id", pa.string()),
        ("started_at", _TS),
        ("finished_at", _TS),
        ("status", pa.string()),
        ("rows_in", pa.int64()),
        ("rows_loaded", pa.int64()),
        ("rows_rejected", pa.int64()),
        ("checks_total", pa.int32()),
        ("checks_failed", pa.int32()),
        # `jsonb` in Postgres. Serialised here rather than exploded into columns so the Stage 5
        # loader hands the string straight to the driver.
        ("report", pa.string()),
    ]
)

CHECK_RESULT_SCHEMA: Final = pa.schema(
    [
        ("result_id", pa.int64()),
        ("run_id", pa.string()),
        ("check_name", pa.string()),
        ("target_table", pa.string()),
        ("target_column", pa.string()),
        ("severity", pa.string()),
        ("status", pa.string()),
        ("expectation", pa.string()),
        ("observed", pa.string()),
    ]
)


def _json_default(value: object) -> str:
    """Render the two types `json` cannot: `Decimal` and `date`/`datetime`.

    `Decimal` becomes a string, never a float — see the module docstring.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


class FileSystemDqStore:
    """Data-quality persistence under a single root directory, one file per run.

    One file per run rather than one appended dataset, for the reason the raw-zone writer gives:
    re-running a batch must not silently double its history. A run id is unique per execution, so a
    retry writes a new file and a crash mid-write leaves the previous runs untouched.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    @property
    def report_path(self) -> Path:
        """Where `dq_report.json` lands. Exposed so a caller can log the path or upload it."""
        return self._root / REPORT_FILENAME

    # --- runs -------------------------------------------------------------------------------

    def save_run(self, run: DqRun, results: Sequence[DqCheckResult]) -> None:
        """Persist one run and all of its check results.

        Called for a failed run as well as a passing one — especially then. The batch is blocked
        and nothing reaches the curated zone, but the record of why survives.
        """
        runs_dir = self._root / RUNS_DIR
        results_dir = self._root / RESULTS_DIR
        runs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        run_table = pa.Table.from_pylist(
            [
                {
                    "run_id": run.run_id,
                    "batch_id": run.batch_id,
                    "started_at": run.started_at,
                    "finished_at": run.finished_at,
                    "status": str(run.status),
                    "rows_in": run.rows_in,
                    "rows_loaded": run.rows_loaded,
                    "rows_rejected": run.rows_rejected,
                    "checks_total": run.checks_total,
                    "checks_failed": run.checks_failed,
                    "report": json.dumps(dict(run.report), default=_json_default),
                }
            ],
            schema=RUN_SCHEMA,
        )
        pq.write_table(run_table, runs_dir / f"{run.run_id}.parquet")

        result_table = pa.Table.from_pylist(
            [
                {
                    "result_id": None,  # bigserial; Postgres assigns it at Stage 5
                    "run_id": result.run_id,
                    "check_name": result.check_name,
                    "target_table": result.target_table,
                    "target_column": result.target_column,
                    "severity": str(result.severity),
                    "status": str(result.status),
                    "expectation": result.expectation,
                    "observed": result.observed,
                }
                for result in results
            ],
            schema=CHECK_RESULT_SCHEMA,
        )
        pq.write_table(result_table, results_dir / f"{run.run_id}.parquet")
        log.info(
            "dq run persisted",
            run_id=run.run_id,
            batch_id=run.batch_id,
            status=str(run.status),
            checks=len(results),
            path=str(runs_dir),
        )

    def write_report(self, report: Mapping[str, Any]) -> Path:
        """Write the human- and CI-readable report artifact, replacing the previous one."""
        self._root.mkdir(parents=True, exist_ok=True)
        path = self.report_path
        path.write_text(
            json.dumps(report, indent=2, default=_json_default, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return path

    # --- baseline profiles -------------------------------------------------------------------

    def save_profile(self, profile: BatchProfile) -> None:
        """Record this batch's statistical shape so the next batch has something to compare to.

        Keyed on `batch_id`, so re-running the same batch overwrites its profile rather than adding
        a second copy that would count twice in the rolling mean.
        """
        directory = self._root / PROFILES_DIR
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_id": profile.batch_id,
            "computed_at": profile.computed_at.isoformat(),
            "window_end": profile.window_end.isoformat(),
            "row_count": profile.row_count,
            "null_rates": dict(profile.null_rates),
            "channel_mix": dict(profile.channel_mix),
            "amount_deciles": [str(value) for value in profile.amount_deciles],
        }
        path = directory / f"{profile.batch_id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info("batch profile recorded", batch_id=profile.batch_id, path=str(path))

    def recent_profiles(
        self, *, limit: int, exclude_batch_id: str | None = None
    ) -> tuple[BatchProfile, ...]:
        """Return up to `limit` profiles, newest `window_end` first.

        `exclude_batch_id` drops the batch currently being validated. Without it, a second run of
        the same batch would compare the batch against itself, report a flawless zero drift, and
        turn four checks into decoration — the failure mode the whole `SKIPPED` status exists to
        avoid.

        A malformed profile file is skipped with a warning rather than raising: a corrupt baseline
        should narrow what the drift checks can say, not stop the gate that protects the ledger.
        """
        directory = self._root / PROFILES_DIR
        if not directory.is_dir():
            return ()
        profiles: list[BatchProfile] = []
        for path in sorted(directory.glob("*.json")):
            profile = self._load_profile(path)
            if profile is None or profile.batch_id == exclude_batch_id:
                continue
            profiles.append(profile)
        profiles.sort(key=lambda item: (item.window_end, item.computed_at), reverse=True)
        return tuple(profiles[:limit])

    @staticmethod
    def _load_profile(path: Path) -> BatchProfile | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return BatchProfile(
                batch_id=str(raw["batch_id"]),
                computed_at=datetime.fromisoformat(raw["computed_at"]).astimezone(UTC),
                window_end=date.fromisoformat(raw["window_end"]),
                row_count=int(raw["row_count"]),
                null_rates={str(k): float(v) for k, v in raw["null_rates"].items()},
                channel_mix={str(k): float(v) for k, v in raw["channel_mix"].items()},
                amount_deciles=tuple(Decimal(str(value)) for value in raw["amount_deciles"]),
            )
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            log.warning("skipping unreadable batch profile", path=str(path), error=str(exc))
            return None
