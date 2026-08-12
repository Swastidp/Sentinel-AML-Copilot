"""Fail the build if the ground truth reaches anything but the evaluation path -- T7.1.

    python scripts/check_label_leakage.py        (wired into `make lint` and `make check`)

`docs/09-FOLDER-STRUCTURE.md` states the invariant: `typology_labels` is read by the metrics module
and by `tests/`, and by nothing else. A label that reaches detection or scoring makes every
precision figure in `docs/13-KNOWN-FALSE-POSITIVES.md` unfalsifiable, and the leak would be
invisible in the output -- the numbers would simply be better. So it is checked mechanically.

---

## Why this is an AST walk and not a grep

A grep for `typology_labels` fails immediately, in both directions.

**It fires on prose.** The string appears in six docstrings that exist precisely *because* the
boundary matters -- `pipeline/detect.py` says "not read here, or anywhere it could be",
`pipeline/anomaly.py` says "never opened". A checker that fails the build for documenting the rule
teaches everyone to delete the documentation.

**It fires on writes.** `infrastructure/storage/parquet_store.py` owns `LABELS_DIR` and
`write_typology_labels`; `pipeline/promote.py` names `WITHHELD_FROM_CURATED` to record that the
labels are held back; `pipeline/generate.py` has a `typology_labels` field on its population object.
Writing the ground truth, excluding it, and holding it in the generator that invented it are all the
boundary working, not breaking it.

So the check is: **does a label reference reach a reader?** Docstrings are dropped, a reference is
tracked through local assignment, and a violation is a reader call -- `read_parquet`, `read_table`,
`execute`, and the rest of `READER_CALLS` -- that receives one.

## What it does not catch, stated rather than implied

* **Aggregate reads through a generic helper.** `pipeline/load.py` calls
  `loader.count(connection, "typology_labels")` to assert the table holds zero rows. `count` is not
  in `READER_CALLS`, so that passes -- deliberately: it returns one integer and can leak no label.
  A row-level read of the same table goes through `cursor.execute` and is caught.
* **Indirection through a variable the checker cannot follow** -- a path assembled in another module
  and imported, or built from a runtime string. Taint is tracked within one module only.
* **Anything outside Python.** SQL files are not parsed; the `typology_labels` DDL in
  `migrations/001_initial_schema.sql` is expected and the table is verified empty by
  `pipeline/load.py`.

A checker whose limits are written down is worth more than one that implies it has none.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: The two ways the ground truth is named in Python. `LABELS_DIR` is the constant in
#: `parquet_store`; the literal is what any hand-written path or SQL string would contain.
LABEL_LITERAL: Final = "typology_labels"
LABEL_CONSTANT: Final = "LABELS_DIR"

#: Function and method names that read data. A label reference arriving at one of these is the
#: violation. Names rather than resolved symbols, because the alternative is import resolution
#: across the whole package to catch a mistake that has never once been made subtly.
READER_CALLS: Final = frozenset(
    {
        "execute",
        "executemany",
        "sql",
        "query",
        "read_parquet",
        "read_partitioned",
        "read_table",
        "read_pandas",
        "dataset_glob",
        "table_file",
        "dataset",
        "ParquetFile",
        "read_csv",
        "from_parquet",
        "scan_parquet",
        "load_table",
    }
)

#: Paths permitted to read the labels, as POSIX prefixes relative to the repo root.
#:
#: `reporting/metrics.py` is the evaluation module `docs/09-FOLDER-STRUCTURE.md` names.
#: `reporting/rule_impact.py` holds `read_labels`, the single physical read both reporting modules
#: share -- one reader rather than two copies of the same DuckDB query.
#: `tests/` reads them to assert the boundary itself holds.
#: This file names them, so it must be allowed to name them.
ALLOWED: Final = (
    "src/sentinel/reporting/metrics.py",
    "src/sentinel/reporting/rule_impact.py",
    "tests/",
    "scripts/check_label_leakage.py",
)

#: Where to look. Everything else in the tree is either not Python or not ours.
SCAN_ROOTS: Final = ("src", "scripts", "tests")


@dataclass(frozen=True, slots=True)
class Violation:
    """One reader call that received the ground truth, with enough detail to fix it."""

    path: str
    line: int
    reader: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: `{self.reader}(...)` receives the typology labels"


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Node ids of every docstring constant, so prose about the boundary is not the boundary."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _references_labels(node: ast.AST, *, docstrings: set[int], tainted: set[str]) -> bool:
    """Does this subtree name the ground truth, directly or through a tainted local?"""
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and LABEL_LITERAL in child.value
            and id(child) not in docstrings
        ):
            return True
        if isinstance(child, ast.Attribute) and child.attr in {LABEL_CONSTANT, LABEL_LITERAL}:
            return True
        if isinstance(child, ast.Name) and (child.id == LABEL_CONSTANT or child.id in tainted):
            return True
    return False


def _tainted_names(tree: ast.AST, docstrings: set[int]) -> set[str]:
    """Locals assigned from a label reference, to a fixpoint.

    Iterated rather than single-pass because `a = LABELS_DIR; b = a; c = b` is three assignments
    deep and a single pass would only reach the first. Modules are small, so the loop is free; it is
    bounded anyway, since each pass either adds a name or stops.
    """
    tainted: set[str] = set()
    while True:
        before = len(tainted)
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign | ast.AugAssign):
                targets = [node.target]
            value = getattr(node, "value", None)
            if not targets or value is None:
                continue
            if not _references_labels(value, docstrings=docstrings, tainted=tainted):
                continue
            for target in targets:
                tainted.update(name.id for name in ast.walk(target) if isinstance(name, ast.Name))
        if len(tainted) == before:
            return tainted


def _reader_name(call: ast.Call) -> str | None:
    """The bare callee name of a call, whether `read_parquet(...)` or `conn.execute(...)`."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def scan_source(source: str, path: str) -> tuple[Violation, ...]:
    """Every reader call in one module that receives the ground truth. Pure -- takes text."""
    tree = ast.parse(source, filename=path)
    docstrings = _docstring_ids(tree)
    tainted = _tainted_names(tree, docstrings)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        reader = _reader_name(node)
        if reader is None or reader not in READER_CALLS:
            continue
        # Only the arguments. The callee itself may legitimately be named after the labels --
        # `parquet_store.write_typology_labels(...)` is the writer.
        arguments: list[ast.expr] = [*node.args, *(kw.value for kw in node.keywords)]
        if any(
            _references_labels(argument, docstrings=docstrings, tainted=tainted)
            for argument in arguments
        ):
            violations.append(Violation(path=path, line=node.lineno, reader=reader))
    return tuple(violations)


def is_allowed(path: str) -> bool:
    """Is this module on the allow-list? Prefix match, so `tests/` covers the whole tree."""
    return any(path.startswith(prefix) for prefix in ALLOWED)


def scan_repo(root: Path) -> tuple[Violation, ...]:
    """Walk `SCAN_ROOTS` and return every violation, sorted, skipping the allow-list."""
    violations: list[Violation] = []
    for name in SCAN_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for file in sorted(directory.rglob("*.py")):
            relative = file.relative_to(root).as_posix()
            if is_allowed(relative):
                continue
            violations.extend(scan_source(file.read_text(encoding="utf-8"), relative))
    return tuple(violations)


def main() -> int:
    """`python scripts/check_label_leakage.py`. Exit 1 on any leak, with the file and line."""
    root = Path(__file__).resolve().parent.parent
    violations = scan_repo(root)
    if violations:
        print(f"label leakage: {len(violations)} violation(s)")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nThe ground truth may be read only by "
            + ", ".join(ALLOWED)
            + ".\nMeasuring against `typology_labels` is legitimate; detecting against it is not."
        )
        return 1
    print(f"label leakage: clean ({', '.join(SCAN_ROOTS)} scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
