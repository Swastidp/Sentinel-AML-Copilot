"""Tests for the domain ports — the nine Protocol seams.

A Protocol has no behaviour to test; its method bodies are `...` by design. What can be tested is
whether the *seam* works, which is four things, each of which would be a real defect:

1. **A plain class with the right methods satisfies the port.** That is the entire promise of
   structural typing, and it is what lets `infrastructure/` implement a port without importing it
   and lets a test fake exist with no base class. If `@runtime_checkable` were dropped, or a
   protocol quietly became an ABC, every fake in this repository would stop working at once.
2. **A class missing a method does not satisfy it.** Without this, test 1 proves nothing — a check
   that accepts everything is not a check.
3. **Every port is reachable from `sentinel.domain.ports`.** Call sites import from the package, not
   from the module, so a protocol that is defined but not re-exported is invisible to them.
4. **The `ports` package imports nothing outside the standard library.** This is the architectural
   rule from `docs/01-ARCHITECTURE.md` §2 — it is what makes the Redis-to-Aerospike swap a one-file
   change — so it is encoded as a test rather than asserted in a comment. The check walks the AST of
   every module in the package, so it catches a bad import even in a module nothing imports yet.

**What `@runtime_checkable` does not do:** it checks that the named methods are *present*. It does
not compare signatures, parameter names, or return types. `test_isinstance_ignores_signatures` below
demonstrates that explicitly, so nobody reads an `isinstance` pass as proof that an adapter is
compatible. `mypy --strict` is what checks signatures, and it is the reason the strict type check is
part of `make check` rather than optional.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from sentinel.domain import ports
from sentinel.domain.entities import compute_pass_through_ratio
from sentinel.domain.enums import FeatureWindow, MuleSignature, RiskRating
from sentinel.domain.exceptions import (
    ConflictError,
    DomainValidationError,
    NotFoundError,
    SentinelError,
    StoreUnavailableError,
)
from sentinel.domain.ports import (
    AccountRepository,
    AlertRepository,
    DqRepository,
    FeatureStore,
    GraphRepository,
    InvestigationRepository,
    LLMClient,
    TransactionRepository,
    VectorStore,
)
from sentinel.domain.value_objects import (
    FeatureVector,
    FundFlowPath,
    FundFlowResult,
    GraphEdge,
    GraphNode,
    IdentifierLink,
    LLMCompletion,
    Subgraph,
)

PORT_NAMES = [
    "AccountRepository",
    "TransactionRepository",
    "AlertRepository",
    "InvestigationRepository",
    "DqRepository",
    "FeatureStore",
    "GraphRepository",
    "VectorStore",
    "LLMClient",
]

# One module per external system, plus the shared-Postgres repositories and the re-export shim.
# Listed explicitly so that adding a module without wiring it into the import-rule test fails here.
PORT_MODULE_FILES = [
    "__init__.py",
    "feature_store.py",
    "graph.py",
    "llm.py",
    "repositories.py",
    "vectors.py",
]


# --- fakes -------------------------------------------------------------------------------------
# Deliberately minimal, and deliberately not subclassing anything. Each one is the smallest class
# that could stand in for its port in a unit test. The FeatureStore fake is the only one with real
# logic, because it is the only port whose contract carries a precondition worth honouring.


class FakeAccountRepository:
    def get(self, account_id: str) -> None:
        return None

    def get_profile(self, account_id: str) -> None:
        return None


class FakeTransactionRepository:
    def get(self, txn_id: str) -> None:
        return None

    def list_for_account(self, account_id: str, **kwargs: object) -> tuple[object, ...]:
        return ()


class FakeAlertRepository:
    def __init__(self) -> None:
        self.statuses: dict[str, object] = {}

    def get(self, alert_id: str) -> None:
        return None

    def list_open(self, **kwargs: object) -> tuple[object, ...]:
        return ()

    def update_status(self, alert_id: str, status: object) -> None:
        self.statuses[alert_id] = status


class FakeInvestigationRepository:
    def save(self, investigation: object) -> None:
        return None

    def get(self, investigation_id: str) -> None:
        return None

    def save_tool_calls(self, calls: Sequence[object]) -> None:
        return None

    def list_tool_calls(self, investigation_id: str) -> tuple[object, ...]:
        return ()

    def get_decision(self, investigation_id: str) -> None:
        return None

    def save_decision(self, decision: object) -> None:
        return None


class FakeDqRepository:
    def list_runs(self, **kwargs: object) -> tuple[object, ...]:
        return ()

    def list_check_results(self, run_id: str) -> tuple[object, ...]:
        return ()


class FakeFeatureStore:
    """The one fake with behaviour: it enforces `put_features`' single-account precondition.

    That precondition is not incidental — it is what lets an implementation write all four windows
    of an account in one command, which is the whole reason `put_features` takes a sequence
    (`docs/12-PLATFORM-LIMITS.md` §2). A fake that ignored it would let a caller develop against a
    shape Redis cannot serve cheaply.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, FeatureWindow], FeatureVector] = {}

    def get_features(self, account_id: str, window: FeatureWindow) -> FeatureVector | None:
        return self.rows.get((account_id, window))

    def put_features(self, vectors: Sequence[FeatureVector]) -> None:
        accounts = {vector.account_id for vector in vectors}
        if len(accounts) > 1:
            raise DomainValidationError(
                f"put_features covers one account per call, got {sorted(accounts)!r}."
            )
        windows = [vector.window for vector in vectors]
        if len(set(windows)) != len(windows):
            raise DomainValidationError(f"duplicate windows in one call: {windows!r}.")
        for vector in vectors:
            self.rows[(vector.account_id, vector.window)] = vector


class FakeGraphRepository:
    def trace_fund_flow(self, account_id: str, **kwargs: object) -> None:
        return None

    def shared_identifiers(self, account_id: str) -> None:
        return None

    def subgraph(self, account_id: str, **kwargs: object) -> None:
        return None


class FakeVectorStore:
    def search_policy(self, query: str, **kwargs: object) -> tuple[object, ...]:
        return ()

    def search_cases(self, query: str, **kwargs: object) -> tuple[object, ...]:
        return ()


class FakeLLMClient:
    def complete(self, prompt: str, **kwargs: object) -> LLMCompletion:
        return LLMCompletion(
            model="fake",
            text="ok",
            data=None,
            finish_reason="stop",
            tokens_in=1,
            tokens_out=1,
            cost_usd=Decimal("0"),
            latency_ms=0,
        )


FAKES: list[tuple[object, type]] = [
    (FakeAccountRepository(), AccountRepository),
    (FakeTransactionRepository(), TransactionRepository),
    (FakeAlertRepository(), AlertRepository),
    (FakeInvestigationRepository(), InvestigationRepository),
    (FakeDqRepository(), DqRepository),
    (FakeFeatureStore(), FeatureStore),
    (FakeGraphRepository(), GraphRepository),
    (FakeVectorStore(), VectorStore),
    (FakeLLMClient(), LLMClient),
]


def make_vector(account_id: str, window: FeatureWindow) -> FeatureVector:
    return FeatureVector(
        account_id=account_id,
        window=window,
        as_of=datetime(2026, 8, 3, 9, 0, tzinfo=UTC),
        txn_count=3,
        total_credit=Decimal("100.00"),
        total_debit=Decimal("94.00"),
        balance_delta=Decimal("6.00"),
        distinct_credit_counterparties=2,
        distinct_debit_counterparties=1,
        max_single_txn=Decimal("60.00"),
        avg_txn_amount=Decimal("33.33"),
        night_txn_ratio=0.5,
        new_counterparty_ratio=0.25,
        distinct_devices=1,
        distinct_ips=1,
        days_since_last_activity=0,
        velocity_zscore=1.2,
    )


# --- 1 & 2: the protocols have teeth ------------------------------------------------------------


@pytest.mark.parametrize("fake, port", FAKES, ids=[port.__name__ for _, port in FAKES])
def test_a_plain_class_with_the_right_methods_satisfies_the_port(fake, port):
    assert isinstance(fake, port), f"{type(fake).__name__} should satisfy {port.__name__}"


def test_a_class_missing_a_method_is_not_an_instance():
    class HalfFeatureStore:
        """`get_features` only — no `put_features`."""

        def get_features(self, account_id: str, window: FeatureWindow) -> None:
            return None

    assert not isinstance(HalfFeatureStore(), FeatureStore)


@pytest.mark.parametrize("port", [p for _, p in FAKES], ids=[p.__name__ for _, p in FAKES])
def test_an_empty_class_satisfies_no_port(port):
    # Guards against a protocol that declares nothing, which every object would satisfy.
    class Nothing:
        pass

    assert not isinstance(Nothing(), port)


def test_isinstance_ignores_signatures():
    # This is the limit of `@runtime_checkable`, asserted rather than described: the method is
    # present with a completely wrong signature and the isinstance check still passes. Only
    # `mypy --strict` catches this class of mismatch, which is why the strict check is in `make
    # check` and why an isinstance pass must never be read as "this adapter is compatible".
    class WrongSignatures:
        def get_features(self) -> None:  # missing account_id and window entirely
            return None

        def put_features(self, a: int, b: int, c: int) -> None:
            return None

    assert isinstance(WrongSignatures(), FeatureStore)


# --- 3: every port is reachable from the package -------------------------------------------------


@pytest.mark.parametrize("name", PORT_NAMES)
def test_every_port_is_importable_from_the_package(name):
    assert hasattr(ports, name), f"{name} is not re-exported from sentinel.domain.ports"
    assert name in ports.__all__, f"{name} is missing from sentinel.domain.ports.__all__"


def test_all_lists_exactly_the_nine_seams():
    # `__all__` is the public surface; a stray export would let a call site depend on something the
    # package did not mean to promise.
    assert sorted(ports.__all__) == sorted(PORT_NAMES)


# --- 4: the architectural rule, as a test --------------------------------------------------------


def port_module_paths() -> list[pathlib.Path]:
    package_dir = pathlib.Path(ports.__file__).parent
    return sorted(package_dir.glob("*.py"))


def test_the_package_contains_exactly_the_expected_modules():
    assert [path.name for path in port_module_paths()] == PORT_MODULE_FILES


@pytest.mark.parametrize("path", port_module_paths(), ids=lambda p: p.name)
def test_ports_import_only_stdlib_and_the_domain(path):
    """Walk the AST and assert every import resolves to the standard library or `sentinel.domain`.

    Reading the source rather than the imported module is deliberate: a module that is never
    imported by anything still gets checked, and an import that only happens to work because a
    third-party package is installed in this environment is still caught.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot leave the package, so it is internal by construction.
            if node.level > 0:
                continue
            names = [node.module or ""]
        else:
            continue

        for name in names:
            root = name.split(".", 1)[0]
            if root in sys.stdlib_module_names:
                continue
            if name == "sentinel.domain" or name.startswith("sentinel.domain."):
                continue
            offenders.append(name)

    assert not offenders, (
        f"{path.name} imports {offenders!r}. The domain layer imports nothing outside the standard "
        f"library — a FeatureStore signature must never be able to mention redis."
    )


# --- the one piece of real logic in the fakes ----------------------------------------------------


def test_feature_store_round_trips_a_whole_vector():
    store = FakeFeatureStore()
    vector = make_vector("ACC1", FeatureWindow.H24)
    store.put_features([vector])

    assert store.get_features("ACC1", FeatureWindow.H24) == vector
    # A window that was never written is absent, not zero — the distinction the port documents.
    assert store.get_features("ACC1", FeatureWindow.D7) is None
    assert store.get_features("ACC-unknown", FeatureWindow.H24) is None


def test_put_features_rejects_a_batch_spanning_two_accounts():
    store = FakeFeatureStore()
    with pytest.raises(DomainValidationError, match="one account per call"):
        store.put_features(
            [make_vector("ACC1", FeatureWindow.H24), make_vector("ACC2", FeatureWindow.H24)]
        )
    assert store.rows == {}


def test_put_features_rejects_a_duplicated_window():
    store = FakeFeatureStore()
    with pytest.raises(DomainValidationError, match="duplicate windows"):
        store.put_features(
            [make_vector("ACC1", FeatureWindow.H24), make_vector("ACC1", FeatureWindow.H24)]
        )


def test_put_features_accepts_every_window_in_one_call():
    # The shape the port exists to make cheap: one account, all four windows, one call.
    store = FakeFeatureStore()
    store.put_features([make_vector("ACC1", window) for window in FeatureWindow])
    assert len(store.rows) == len(FeatureWindow)


# --- the shapes the ports return -----------------------------------------------------------------
# A port's contract is only as strong as the type it hands back. These cover the guards added for
# T1.2, each of which prevents a specific way a graph adapter can return a plausible-looking lie.


def make_path(*accounts: str) -> FundFlowPath:
    return FundFlowPath(
        hop_accounts=accounts,
        hops=len(accounts) - 1,
        path_value=Decimal("100"),
        bottleneck_amount=Decimal("40"),
        flow_started=datetime(2026, 8, 1, tzinfo=UTC),
        flow_ended=datetime(2026, 8, 2, tzinfo=UTC),
    )


def make_flow(**overrides: object) -> FundFlowResult:
    fields: dict[str, object] = {
        "account_id": "ACC1",
        "max_hops": 3,
        "downstream_paths": (make_path("ACC1", "ACC2"),),
        "cycles": (),
        "fan_in_degree": 14,
        "fan_out_degree": 2,
        "total_in": Decimal("500000"),
        "total_out": Decimal("470000"),
        "signature": MuleSignature.COLLECTION_MULE,
        "ring": None,
    }
    fields.update(overrides)
    return FundFlowResult(**fields)  # type: ignore[arg-type]


def test_fund_flow_result_rejects_a_non_cycle_filed_as_a_cycle():
    # Q2 only matches paths that close on the seed. A Q1 path filed here would make the narrative
    # claim round-tripping that never happened.
    with pytest.raises(DomainValidationError, match="filed as a cycle"):
        make_flow(cycles=(make_path("ACC1", "ACC2", "ACC3"),))


def test_fund_flow_result_accepts_a_real_cycle():
    flow = make_flow(cycles=(make_path("ACC1", "ACC2", "ACC1"),))
    assert flow.cycles[0].is_cycle


def test_fund_flow_pass_through_uses_the_one_definition():
    # The graph and the feature store measure this over different scopes, but through one function.
    flow = make_flow()
    assert flow.pass_through_ratio == compute_pass_through_ratio(flow.total_in, flow.total_out)


def test_identifier_link_rejects_a_same_customer_link():
    # Q4 filters on `c1.customer_id <> c2.customer_id`. Losing that predicate would turn "one person
    # uses one phone for two of their own accounts" into a ring finding.
    with pytest.raises(DomainValidationError, match="not an identity-linkage signal"):
        IdentifierLink(identifier="DEV1", linked_accounts=("ACC2",), distinct_other_customers=0)


def test_identifier_link_rejects_more_customers_than_accounts():
    # An account has exactly one owner, and `shared_identifiers` orders by this field — an inflated
    # count would put a fabricated link at the top of the analyst's queue.
    with pytest.raises(DomainValidationError, match="an account has one owner"):
        IdentifierLink(identifier="DEV1", linked_accounts=("ACC2",), distinct_other_customers=5)


def make_node(account_id: str, *, is_seed: bool = False) -> GraphNode:
    return GraphNode(
        account_id=account_id, is_seed=is_seed, is_flagged=False, risk_rating=RiskRating.LOW
    )


def test_subgraph_rejects_an_edge_to_an_unknown_node():
    # A renderer given an edge to an absent node invents that node, and the analyst then sees a
    # phantom account with no risk rating and no way to tell it from a real one.
    with pytest.raises(DomainValidationError, match="absent from its"):
        Subgraph(
            seed_account_id="ACC1",
            max_hops=2,
            nodes=(make_node("ACC1", is_seed=True),),
            edges=(
                GraphEdge(
                    source="ACC1",
                    target="ACC9",
                    amount=Decimal("42000"),
                    txn_count=1,
                    channels=("UPI",),
                ),
            ),
        )


def test_subgraph_rejects_a_missing_seed_marker():
    with pytest.raises(DomainValidationError, match="as its seed nodes"):
        Subgraph(seed_account_id="ACC1", max_hops=2, nodes=(make_node("ACC1"),), edges=())


def test_subgraph_accepts_a_lone_seed_with_no_edges():
    # An isolated account is a real answer — the view must render one node, not nothing.
    graph = Subgraph(
        seed_account_id="ACC1", max_hops=2, nodes=(make_node("ACC1", is_seed=True),), edges=()
    )
    assert graph.nodes[0].is_seed and graph.edges == ()


def test_the_three_port_failures_are_catchable_and_carry_their_subject():
    # `api/errors.py` maps NotFoundError to 404, ConflictError to 409 and StoreUnavailableError to
    # 503, and builds the response body from these attributes — so they are contract, not debug
    # detail. All three must also be caught by the API's single `SentinelError` handler, which is
    # what makes the hierarchy closed and keeps the mapping in one place with no per-store import.
    missing = NotFoundError("alert", "ALT417")
    assert isinstance(missing, SentinelError)
    assert (missing.kind, missing.identifier) == ("alert", "ALT417")
    assert "ALT417" in str(missing)

    clash = ConflictError("decision", "INV188", "a decision already exists")
    assert isinstance(clash, SentinelError)
    assert (clash.kind, clash.identifier) == ("decision", "INV188")
    assert "already exists" in str(clash)

    down = StoreUnavailableError("redis", "ACC00042917", "connection reset")
    assert isinstance(down, SentinelError)
    assert (down.kind, down.identifier) == ("redis", "ACC00042917")
    assert "connection reset" in str(down)
    # The whole point of the type: a caller must be able to tell "we could not look" apart from
    # "there is nothing there", and `except SentinelError` alone cannot do that.
    assert not isinstance(down, NotFoundError)


def test_a_wholly_empty_subgraph_is_unconstructible():
    # This is *why* `GraphRepository.subgraph` returns `Subgraph | None`. An account outside the
    # projection has no seed node to build — the adapter has no source for its `risk_rating` — so
    # the port answers `None` rather than letting an adapter invent one to satisfy the guard.
    with pytest.raises(DomainValidationError, match="as its seed nodes"):
        Subgraph(seed_account_id="ACC1", max_hops=2, nodes=(), edges=())
