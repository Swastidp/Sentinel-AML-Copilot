"""Offline tests for the Stage 5 schema and the loader's mapping onto it.

No database. Everything here reads `001_initial_schema.sql` as text and compares it against the
pyarrow schemas the pipeline already writes, so the three column lists that have to agree --
Parquet, the Postgres DDL, and `pipeline/load.LOADS` -- are checked by something rather than by a
promise in three docstrings.

That matters most for the two data-quality tables. `infrastructure/storage/dq_store.py` claims in
writing that its Parquet output has "the same names, same order" as the Postgres tables, so that the
Stage 5 adapter is a load and not a translation. Either that claim is true or the design note is
wrong, and a test is the only thing that can tell the difference.

The SQL parser below is deliberately dumb: it finds a `CREATE TABLE` by name, balances parentheses,
and splits the body on top-level commas. It is not a SQL parser and must not become one -- it exists
to read a file this repository controls, in a format this repository writes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pytest

from sentinel.domain.exceptions import ConfigurationError, DomainValidationError
from sentinel.infrastructure.postgres import migrate
from sentinel.infrastructure.postgres.loader import TableSpec
from sentinel.infrastructure.storage import parquet_store
from sentinel.infrastructure.storage.dq_store import CHECK_RESULT_SCHEMA, RUN_SCHEMA
from sentinel.pipeline import features, load

#: The twelve tables of `docs/02-DATA-MODEL.md`. Partitions of `transactions` are not tables in this
#: sense and are excluded from the count on purpose.
EXPECTED_TABLES = {
    "customers",
    "accounts",
    "devices",
    "transactions",
    "account_features",
    "alerts",
    "investigations",
    "tool_calls",
    "human_decisions",
    "dq_runs",
    "dq_check_results",
    "typology_labels",
}

#: A definition whose *first token* is one of these is a table constraint, not a column. Compared
#: as a whole token, never as a prefix: `check_name` and `checks_total` are real columns, and a
#: `startswith("check")` test silently drops all three of them from `dq_check_results`.
_CONSTRAINT_KEYWORDS = frozenset({"constraint", "primary", "foreign", "unique", "check", "exclude"})

_COMMENT = re.compile(r"--[^\n]*")


@pytest.fixture(scope="module")
def ddl() -> str:
    (path,) = migrate.migration_files()
    return _COMMENT.sub("", path.read_text(encoding="utf-8"))


def _body(ddl: str, table: str) -> str:
    """The text between the parentheses of one `CREATE TABLE`, parens balanced."""
    marker = f"CREATE TABLE IF NOT EXISTS {table} ("
    start = ddl.index(marker) + len(marker)
    depth = 1
    for index in range(start, len(ddl)):
        if ddl[index] == "(":
            depth += 1
        elif ddl[index] == ")":
            depth -= 1
            if depth == 0:
                return ddl[start:index]
    raise AssertionError(f"unbalanced parentheses in the definition of {table}")


def _definitions(ddl: str, table: str) -> list[str]:
    """Top-level comma-separated items of a table body, whitespace collapsed."""
    body = _body(ddl, table)
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [" ".join(item.split()) for item in items if item.strip()]


def columns(ddl: str, table: str) -> list[str]:
    """Column names of one table, in declaration order."""
    return [
        item.split()[0]
        for item in _definitions(ddl, table)
        if item.split()[0].lower() not in _CONSTRAINT_KEYWORDS
    ]


def declaration(ddl: str, table: str, column: str) -> str:
    """The full text of one column's definition, for asserting on its type."""
    for item in _definitions(ddl, table):
        if item.split()[0] == column:
            return item
    raise AssertionError(f"{table} has no column {column}")


# --- the twelve tables ------------------------------------------------------------------------


def test_all_twelve_tables_are_created(ddl):
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl))
    # The monthly partitions are created inside a DO block by `format()`, and `transactions_default`
    # is a partition too; neither is one of the twelve modelled tables.
    assert created - {"transactions_default"} == EXPECTED_TABLES


def test_transactions_is_range_partitioned_on_txn_date(ddl):
    assert "PARTITION BY RANGE (txn_date)" in ddl


def test_a_default_partition_catches_dates_outside_the_declared_range(ddl):
    assert "PARTITION OF transactions DEFAULT" in ddl


# --- column order against what the pipeline already writes ------------------------------------


@pytest.mark.parametrize(
    ("table", "schema"),
    [
        ("customers", parquet_store.CUSTOMER_SCHEMA),
        ("accounts", parquet_store.ACCOUNT_SCHEMA),
        ("devices", parquet_store.DEVICE_SCHEMA),
        ("transactions", parquet_store.TRANSACTION_SCHEMA),
        ("typology_labels", parquet_store.LABEL_SCHEMA),
    ],
)
def test_table_columns_match_the_parquet_writer(ddl, table: str, schema: pa.Schema):
    assert columns(ddl, table) == list(schema.names)


def test_dq_runs_matches_the_gate_s_parquet_output(ddl):
    """The Stage 3 design claim: same names, same order, so Stage 5 is a load not a translation."""
    assert columns(ddl, "dq_runs") == list(RUN_SCHEMA.names)


def test_dq_check_results_matches_the_gate_s_parquet_output(ddl):
    assert columns(ddl, "dq_check_results") == list(CHECK_RESULT_SCHEMA.names)


def test_result_id_is_assigned_by_the_database(ddl):
    """`FileSystemDqStore` writes `result_id` as null precisely because this is a `bigserial`."""
    assert "bigserial" in declaration(ddl, "dq_check_results", "result_id").lower()


def test_account_features_matches_the_feature_schema_with_one_rename(ddl):
    """Decision 1: `window` is a reserved word in SQL, so the column is `feature_window`."""
    expected = [
        "feature_window" if name == "window" else name for name in features.FEATURE_SCHEMA.names
    ]
    assert columns(ddl, "account_features") == expected


# --- the six recorded decisions ---------------------------------------------------------------


def test_the_feature_window_rename_is_the_only_one(ddl):
    assert "feature_window" in columns(ddl, "account_features")
    assert "window" not in columns(ddl, "account_features")


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("customers", "declared_annual_income"),
        ("accounts", "current_balance"),
        ("transactions", "amount"),
        ("account_features", "total_credit"),
        ("account_features", "total_debit"),
        ("account_features", "balance_delta"),
        ("account_features", "max_single_txn"),
        ("account_features", "avg_txn_amount"),
    ],
)
def test_decision_2_money_is_exact_decimal(ddl, table: str, column: str):
    """Rupee money is `numeric(18, 2)`, matching `decimal128(18, 2)` on the Parquet side exactly."""
    assert "numeric(18, 2)" in declaration(ddl, table, column)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("account_features", "pass_through_ratio"),
        ("account_features", "night_txn_ratio"),
        ("account_features", "velocity_zscore"),
        ("alerts", "rule_score"),
        ("alerts", "anomaly_score"),
        ("investigations", "agent_confidence"),
        ("investigations", "computed_risk_score"),
    ],
)
def test_decision_2_scores_are_double_precision(ddl, table: str, column: str):
    assert "double precision" in declaration(ddl, table, column)


def test_decision_2_no_money_column_is_a_float(ddl):
    """The direction that actually matters: a rupee column typed as a float is the defect."""
    money = re.compile(r"(amount|balance|credit|debit|income|txn)\b")
    offenders = [
        f"{table}.{item.split()[0]}"
        for table in EXPECTED_TABLES
        for item in _definitions(ddl, table)
        if "double precision" in item
        and money.search(item.split()[0])
        # Ratios, counts and z-scores carry these words and are dimensionless by design.
        and not item.split()[0].endswith(("_ratio", "_count", "_zscore"))
    ]
    assert offenders == []


def test_decision_2_cost_usd_is_the_documented_exception(ddl):
    """A DeepSeek call costs a fraction of a cent; two places would store every one as 0.00."""
    assert "numeric(18, 6)" in declaration(ddl, "investigations", "cost_usd")


def test_decision_3_investigations_can_store_the_component_breakdown(ddl):
    assert "jsonb" in declaration(ddl, "investigations", "risk_components")


def test_decision_4_account_features_key_is_composite(ddl):
    assert "PRIMARY KEY (account_id, feature_window)" in _body(ddl, "account_features")


def test_decision_4_makes_the_separate_unique_feature_index_redundant(ddl):
    """The data model listed `idx_feat_acct_window`; the composite primary key already is it."""
    assert "idx_feat_acct_window" not in ddl


# --- indexes ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    ["idx_txn_from_ts", "idx_txn_to_ts", "idx_txn_device", "idx_txn_amount"],
)
def test_the_four_transaction_indexes_exist(ddl, index: str):
    assert re.search(rf"CREATE INDEX IF NOT EXISTS {index}\s+ON transactions", ddl)


@pytest.mark.parametrize("index", ["idx_alerts_status", "idx_inv_alert", "idx_toolcalls_inv"])
def test_the_other_documented_indexes_exist(ddl, index: str):
    assert re.search(rf"CREATE INDEX IF NOT EXISTS {index}\s+ON ", ddl)


# --- the safety rule ----------------------------------------------------------------------------


def test_the_real_migration_passes_the_safety_check():
    for path in migrate.migration_files():
        migrate.check_safety(path.read_text(encoding="utf-8"), path.name)


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE accounts;",
        "drop schema public cascade;",
        "TRUNCATE transactions;",
        "DELETE FROM customers WHERE 1 = 1;",
        "ALTER TABLE accounts DROP COLUMN ifsc;",
    ],
)
def test_check_safety_refuses_anything_destructive(statement: str):
    with pytest.raises(ConfigurationError, match="additive only"):
        migrate.check_safety(statement, "bad.sql")


def test_check_safety_refuses_an_unguarded_create():
    with pytest.raises(ConfigurationError, match="IF NOT EXISTS"):
        migrate.check_safety("CREATE TABLE accounts (id text);", "bad.sql")


def test_check_safety_ignores_the_words_inside_a_comment():
    """The real migration's prose talks about DROP and TRUNCATE; only statements count."""
    migrate.check_safety(
        "-- never DROP TABLE anything, never TRUNCATE\nCREATE TABLE IF NOT EXISTS t (id text);\n",
        "fine.sql",
    )


def test_migration_files_are_discoverable_and_ordered():
    files = migrate.migration_files()
    assert files == tuple(sorted(files))
    assert all(path.suffix == ".sql" for path in files)
    assert Path(files[0]).name == "001_initial_schema.sql"


# --- the loader's mapping onto the schema -----------------------------------------------------


def test_every_load_spec_covers_every_column_of_its_table(ddl):
    """A column added to the DDL but not to a spec would silently load as its default.

    This is the join that keeps three files honest at once: the spec is derived from the pyarrow
    schema, and this asserts the derived list is exactly the Postgres table.
    """
    for spec, _ in load.LOADS:
        assert list(spec.pg_columns) == columns(ddl, spec.table), spec.table


def test_load_specs_conflict_on_a_real_key(ddl):
    """`ON CONFLICT (...)` needs a matching unique constraint or Postgres rejects the statement."""
    declared = {
        "customers": ("customer_id",),
        "accounts": ("account_id",),
        "devices": ("device_id",),
        "transactions": ("txn_id", "txn_date"),
        "account_features": ("account_id", "feature_window"),
    }
    for spec, _ in load.LOADS:
        assert spec.key == declared[spec.table]
        rendered = ", ".join(spec.key)
        assert f"PRIMARY KEY ({rendered})" in _body(ddl, spec.table) or (
            len(spec.key) == 1 and "PRIMARY KEY" in declaration(ddl, spec.table, spec.key[0])
        )


def test_typology_labels_is_never_loaded():
    """The eval boundary. The table exists; nothing in the Stage 5 load path writes it."""
    assert "typology_labels" not in {spec.table for spec, _ in load.LOADS}


def test_a_spec_whose_key_is_not_a_loaded_column_is_rejected():
    with pytest.raises(DomainValidationError, match="nothing to match on"):
        TableSpec(table="accounts", key=("account_id",), columns=(("ifsc", "ifsc"),))
