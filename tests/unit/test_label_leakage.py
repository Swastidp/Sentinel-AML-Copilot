"""The label-leakage lint rule -- T7.1.

Two things are asserted, and the second is the one that matters: the rule is clean against the real
repository, **and it fails when deliberately violated**. A lint rule nobody has watched fail is not
known to work -- it is only known to be silent, and silence is what a broken checker produces too.

The traps are covered explicitly because both already exist in the tree and a naive grep hits both:
a docstring that names `typology_labels` while explaining why it must not be read, and
`parquet_store.LABELS_DIR`, which is used to *write* the file and never to read it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_label_leakage import (
    ALLOWED,
    READER_CALLS,
    is_allowed,
    scan_repo,
    scan_source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_real_repository_is_clean():
    """The invariant itself. This is the assertion the build is protecting."""
    violations = scan_repo(REPO_ROOT)
    assert violations == (), "\n".join(str(v) for v in violations)


def test_it_fails_when_deliberately_violated():
    """A leak through the constant, exactly as a real mistake in `pipeline/detect.py` would look."""
    source = """
from sentinel.infrastructure.storage import parquet_store

def detect(root):
    truth = parquet_store.read_table(root, parquet_store.LABELS_DIR)
    return set(truth.column("account_id").to_pylist())
"""
    violations = scan_source(source, "src/sentinel/pipeline/detect.py")
    assert len(violations) == 1
    assert violations[0].reader == "read_table"
    assert violations[0].line == 5


def test_it_fails_on_a_raw_sql_read():
    """The sneakier form: no `LABELS_DIR`, just the table name in a query string."""
    source = """
def peek(connection):
    return connection.execute("SELECT account_id FROM typology_labels").fetchall()
"""
    violations = scan_source(source, "src/sentinel/pipeline/rules.py")
    assert len(violations) == 1
    assert violations[0].reader == "execute"


def test_it_follows_a_label_path_through_locals():
    """Trap: assigning the path to a variable first must not launder it past the checker."""
    source = """
from sentinel.infrastructure.storage import parquet_store

def peek(root, connection):
    directory = parquet_store.LABELS_DIR
    path = root / directory / "*.parquet"
    return connection.execute("SELECT * FROM read_parquet($p)", {"p": path}).fetchall()
"""
    violations = scan_source(source, "src/sentinel/pipeline/rules.py")
    assert [v.reader for v in violations] == ["execute"]


def test_trap_one_a_docstring_mention_is_not_a_read():
    """`pipeline/detect.py` and `pipeline/anomaly.py` both explain the boundary in prose."""
    source = '''
"""`typology_labels` is not read here, or anywhere it could be."""


def fit(matrix):
    """Fitted on features only. `typology_labels` is never opened."""
    return matrix
'''
    assert scan_source(source, "src/sentinel/pipeline/anomaly.py") == ()


def test_trap_two_the_writer_is_not_a_reader():
    """`parquet_store.write_typology_labels` names the labels in the callee, not in an argument."""
    source = """
LABELS_DIR = "typology_labels"


def write_typology_labels(labels, root):
    table = pa.Table.from_pylist(_rows(labels), schema=LABEL_SCHEMA)
    return _write_single_file(table, root / LABELS_DIR, LABELS_DIR)
"""
    assert scan_source(source, "src/sentinel/infrastructure/storage/parquet_store.py") == ()


def test_naming_the_exclusion_is_not_a_read():
    """`pipeline/promote.py` names the labels precisely to record that they are withheld."""
    source = """
from sentinel.infrastructure.storage import parquet_store

WITHHELD_FROM_CURATED = (parquet_store.LABELS_DIR,)
"""
    assert scan_source(source, "src/sentinel/pipeline/promote.py") == ()


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("src/sentinel/reporting/metrics.py", True),
        ("src/sentinel/reporting/rule_impact.py", True),
        ("tests/unit/test_generator.py", True),
        ("scripts/check_label_leakage.py", True),
        ("src/sentinel/pipeline/detect.py", False),
        ("src/sentinel/application/risk_scorer.py", False),
        ("src/sentinel/api/routers/score.py", False),
    ],
)
def test_the_allow_list_is_exactly_the_evaluation_path(path, allowed):
    assert is_allowed(path) is allowed


def test_the_allow_list_has_not_quietly_grown():
    """A pin, not a tautology. Widening the allow-list is how this check dies quietly."""
    assert ALLOWED == (
        "src/sentinel/reporting/metrics.py",
        "src/sentinel/reporting/rule_impact.py",
        "tests/",
        "scripts/check_label_leakage.py",
    )


def test_count_is_not_a_reader_and_that_is_documented():
    """`pipeline/load.py` counts the table to assert it is empty. A count leaks no label.

    Pinned so the limitation stays deliberate: if `count` is ever added to `READER_CALLS`, this
    fails and whoever added it has to decide what happens to the fence's own assertion.
    """
    assert "count" not in READER_CALLS
    source = """
def verify(connection):
    labels = loader.count(connection, "typology_labels")
    if labels:
        raise DomainValidationError("nothing in Stage 5 may load the ground truth")
"""
    assert scan_source(source, "src/sentinel/pipeline/load.py") == ()
