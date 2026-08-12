"""The Isolation Forest — an unsupervised second opinion on every account, T6.2.

Fit on the feature matrix, persisted, and mapped onto a 0-100 anomaly score. The rules give
defensibility — a regulator asking why a *specific* account was flagged gets a named pattern and a
threshold. This gives reach: it flags accounts unlike the rest of the book without anyone having
described what "unlike" means in advance.

---

**It is fitted on features only. `typology_labels` is never opened, and nothing in this module
could open it if it wanted to** — the labels are withheld from the curated zone by
`pipeline/promote.py` and asserted absent from Postgres by `pipeline/load.verify`, so the only copy
lives in `data/raw/typology_labels/` which no detection module reads. That is not fastidiousness.
Fraud labels are scarce, they arrive months late after an investigation completes, and they
describe last year's patterns against an adversary who reads your detections and adapts. A
supervised model trained on them answers "does this look like known fraud"; this one answers "is
this unusual for this population", and only the second can catch something nobody has labelled yet.
Touching the labels here would also make every Stage 7 precision figure unfalsifiable.

**How it works, briefly.** Pick a random feature, pick a random split point, split. Repeat until
every account sits alone. An account far from the crowd gets separated in few splits; one buried in
the crowd takes many. The average path length across `n_estimators` trees is the anomaly score. No
labels needed, and no distance metric to choose.

**The score mapping is fixed, not batch-relative.** `decision_function` returns
`score_samples - offset_`, and both terms lie in `[-1, 0]`, so the difference lies in `[-1, 1]`.
The map is therefore `50 * (1 - d)`, clipped: a total order preserved, bounded to 0..100, and — the
property worth having — **exactly 50 at `d = 0`, which is scikit-learn's own outlier boundary at
the configured contamination.** So "anomaly score at or above 50" and "in the most anomalous 2.7%
of the book" say the same thing. A min-max rescale over the batch would have been easier and would
have made two batches incomparable, because the same account would score differently depending on
who else happened to be in the file.

**Reproducibility.** `IsolationForestConfig.random_state` seeds the split choices, so refitting the
same rows gives the same model and therefore the same alerts. The persisted artefact records the
config, the column order and the row count it was fitted on, because a model file that cannot say
what it was trained against is not auditable.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.config.logging import get_logger
from sentinel.config.weights import IsolationForestConfig
from sentinel.domain.enums import FeatureWindow
from sentinel.domain.exceptions import DomainValidationError

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

log = get_logger(__name__)

#: Bumped whenever the feature list, the estimator or the score mapping changes, so a stored
#: `anomaly_score` can be traced to the model that produced it.
ANOMALY_MODEL_VERSION: Final = "iforest-v1"

#: The artefact filename under `Settings.model_path`.
MODEL_FILENAME: Final = "isolation_forest.pkl"

#: The window the model is fitted on. One row per account, matching the grain of an alert. 30D is
#: not concatenated alongside it: the two windows are strongly correlated by construction (30D
#: contains 7D), and feeding a model both halves of a near-duplicate makes it confident rather than
#: informed.
FIT_WINDOW: Final = FeatureWindow.D7

#: The columns fed to the estimator, in this order. **Every one is a measured feature.** No
#: identifier, no timestamp, and no label: `account_id` would let the model memorise the book,
#: `as_of` is constant within a batch and therefore pure noise, and the labels are fenced off
#: entirely. The order is fixed and persisted with the model, because scikit-learn matches columns
#: by position and a silently reordered matrix produces confident nonsense.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "txn_count",
    "credit_count",
    "debit_count",
    "total_credit",
    "total_debit",
    "balance_delta",
    "distinct_credit_counterparties",
    "distinct_debit_counterparties",
    "max_single_txn",
    "avg_txn_amount",
    "pass_through_ratio",
    "night_txn_ratio",
    "new_counterparty_ratio",
    "structuring_band_count",
    "distinct_devices",
    "distinct_ips",
    "days_since_last_activity",
    "velocity_zscore",
)


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """The fit input: account ids and the aligned numeric matrix, kept together on purpose.

    Two parallel lists that must not drift is exactly the shape that drifts. Bundling them means a
    caller cannot sort one and forget the other, which would attach every score to the wrong
    account while every count still looked right.
    """

    account_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: np.ndarray[Any, np.dtype[np.float64]]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise DomainValidationError(
                f"the feature matrix must be two-dimensional, got shape {self.values.shape}."
            )
        rows, cols = self.values.shape
        if rows != len(self.account_ids) or cols != len(self.columns):
            raise DomainValidationError(
                f"matrix shape {self.values.shape} disagrees with {len(self.account_ids)} accounts "
                f"and {len(self.columns)} columns."
            )
        if not np.isfinite(self.values).all():
            # A NaN reaches `fit` as a raw ValueError from deep inside scikit-learn. Caught here it
            # names the batch that produced it, and a NaN in a feature column is a Stage 4 defect
            # rather than a modelling one.
            raise DomainValidationError(
                "the feature matrix holds NaN or infinite values; the Isolation Forest cannot be "
                "fitted on them and imputing them here would invent data the batch does not have."
            )


@dataclass(frozen=True, slots=True)
class PersistedForest:
    """The fitted model plus everything needed to reuse it honestly.

    The column order and the config travel with the estimator because scoring a matrix built to a
    different column order is the classic silent model failure: no error, plausible numbers, wrong
    answers. `load` checks them rather than trusting the caller.
    """

    model: IsolationForest
    columns: tuple[str, ...]
    config: IsolationForestConfig
    version: str
    fitted_at: datetime
    fitted_rows: int


def build_matrix(table: pa.Table, window: FeatureWindow = FIT_WINDOW) -> FeatureMatrix:
    """Turn the curated feature table into the model's input for one window.

    Columns are selected **by name** from `FEATURE_COLUMNS`, never by position — the same rule the
    feature SQL and the Postgres adapters follow, and for the same reason. Money arrives as
    `Decimal` and is converted to `float` here, at the model boundary and nowhere earlier: these
    are inputs to a split point, never terms in a balance.
    """
    all_rows = table.to_pylist()
    rows = [row for row in all_rows if row["window"] == str(window)]
    if not rows:
        raise DomainValidationError(
            f"the feature table holds no {window} rows, so there is nothing to fit. The table has "
            f"{table.num_rows} rows across {sorted({str(r['window']) for r in all_rows})!r}."
        )
    missing = [name for name in FEATURE_COLUMNS if name not in rows[0]]
    if missing:
        raise DomainValidationError(
            f"the feature table is missing {missing!r}; the model was defined over "
            f"{list(FEATURE_COLUMNS)!r}."
        )
    account_ids = tuple(str(row["account_id"]) for row in rows)
    values = np.array(
        [[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float64
    )
    return FeatureMatrix(account_ids=account_ids, columns=FEATURE_COLUMNS, values=values)


def fit(matrix: FeatureMatrix, config: IsolationForestConfig | None = None) -> PersistedForest:
    """Fit the forest on the matrix and wrap it with its provenance.

    `n_jobs` is left at the default single worker. The fit is under a second on 6,200 rows, and a
    parallel fit introduces a scheduling order that is not covered by `random_state` — which would
    cost the reproducibility this whole module is built around to save nothing measurable.
    """
    cfg = config if config is not None else IsolationForestConfig()
    estimator = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=min(cfg.max_samples, matrix.values.shape[0]),
        contamination=cfg.contamination,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        bootstrap=False,
    )
    estimator.fit(matrix.values)
    log.info(
        "isolation forest fitted",
        model_version=ANOMALY_MODEL_VERSION,
        rows=matrix.values.shape[0],
        features=len(matrix.columns),
        contamination=cfg.contamination,
        n_estimators=cfg.n_estimators,
    )
    return PersistedForest(
        model=estimator,
        columns=matrix.columns,
        config=cfg,
        version=ANOMALY_MODEL_VERSION,
        fitted_at=datetime.now(UTC),
        fitted_rows=matrix.values.shape[0],
    )


def to_anomaly_score(decision: np.ndarray[Any, np.dtype[np.float64]]) -> list[float]:
    """Map `decision_function` output onto 0-100, higher meaning more anomalous.

    `50 * (1 - d)`, clipped to the interval. See the module docstring for why this fixed affine map
    is used rather than a rescale over whatever happens to be in the batch, and why 50 is the
    outlier boundary rather than an arbitrary midpoint.
    """
    return [float(value) for value in np.clip(50.0 * (1.0 - decision), 0.0, 100.0)]


def score(forest: PersistedForest, matrix: FeatureMatrix) -> dict[str, float]:
    """Account id to 0-100 anomaly score.

    The column check is not defensive noise: a matrix built from a table whose columns moved would
    score perfectly happily and be wrong everywhere, and nothing downstream could tell.
    """
    if matrix.columns != forest.columns:
        raise DomainValidationError(
            f"the model was fitted on {list(forest.columns)!r} but the matrix supplies "
            f"{list(matrix.columns)!r}; scoring across a column reorder produces plausible, wrong "
            f"numbers with no error."
        )
    decision: np.ndarray[Any, np.dtype[np.float64]] = forest.model.decision_function(matrix.values)
    return dict(zip(matrix.account_ids, to_anomaly_score(decision), strict=True))


def raw_decision(forest: PersistedForest, matrix: FeatureMatrix) -> dict[str, float]:
    """Account id to the untransformed `decision_function` value, negative for outliers.

    Kept alongside the mapped score because `alerts.anomaly_score` is documented as tolerating the
    model's own negative scale, and because an analyst reproducing a figure should be able to see
    the number scikit-learn actually returned rather than only our transformation of it. Stored in
    the alert's `trigger_evidence`.
    """
    decision: np.ndarray[Any, np.dtype[np.float64]] = forest.model.decision_function(matrix.values)
    return {
        account_id: float(value)
        for account_id, value in zip(matrix.account_ids, decision, strict=True)
    }


def model_file(model_root: Path) -> Path:
    """Where the artefact lives under `Settings.model_path`."""
    return model_root / MODEL_FILENAME


def save(forest: PersistedForest, model_root: Path) -> Path:
    """Write the artefact, creating the directory if needed. Overwrites in place.

    Stdlib `pickle` rather than `joblib`: joblib's advantage is memory-mapping large NumPy arrays,
    and this model is a few hundred kilobytes of small trees. One fewer import, one fewer mypy
    override, identical result.

    ponytail: pickle is fine for an artefact this process wrote and this process reads. If the file
    ever crosses a trust boundary, swap it for ONNX or a plain parameter dump — never unpickle
    something you did not write.
    """
    model_root.mkdir(parents=True, exist_ok=True)
    path = model_file(model_root)
    path.write_bytes(pickle.dumps(forest, protocol=pickle.HIGHEST_PROTOCOL))
    log.info(
        "isolation forest persisted",
        path=str(path),
        bytes=path.stat().st_size,
        model_version=forest.version,
        fitted_rows=forest.fitted_rows,
    )
    return path


def load(model_root: Path) -> PersistedForest:
    """Read the artefact back, or fail naming the path and the command that writes it."""
    path = model_file(model_root)
    if not path.exists():
        raise DomainValidationError(
            f"no anomaly model at {path}. Run `make detect` (or "
            f"`python -m sentinel.pipeline.detect`) to fit and persist one."
        )
    loaded = pickle.loads(path.read_bytes())
    if not isinstance(loaded, PersistedForest):
        raise DomainValidationError(
            f"{path} does not hold a PersistedForest but a {type(loaded).__name__}; the artefact "
            f"was written by something other than this module."
        )
    if loaded.version != ANOMALY_MODEL_VERSION:
        raise DomainValidationError(
            f"{path} was written by {loaded.version} and this code is {ANOMALY_MODEL_VERSION}; the "
            f"feature list or the score mapping changed, so the stored scores are not comparable. "
            f"Refit rather than reusing it."
        )
    return loaded
