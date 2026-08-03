"""The only check that the nine ports are actually *implementable*.

`test_ports.py` proves a class with the right method *names* satisfies a port, because that is all
`@runtime_checkable` can see. It never compares a parameter list, a keyword name, or a return type.
So a `Protocol` whose signatures contradict themselves — a return type no adapter can produce, a
parameter that cannot be supplied — passes every test in that file, passes `mypy src scripts`
because `src` holds no implementation, and fails for the first time at T5.2 when someone tries to
write the Postgres adapter. This file closes that window.

It works by static checking, not by running anything. Each stub below is a fully-annotated minimal
adapter, and the block at the bottom **assigns each one to its protocol type**. That assignment is
the entire test: `mypy --strict` verifies assignability there, so a signature an adapter cannot
match becomes an error in CI rather than a discovery three stages later.

**The negative control, which is the part to understand before editing this file:** a stub that is
never assigned to its protocol type is a test that cannot fail. Defining `class Accounts` and
walking away proves nothing whatsoever — mypy has no reason to compare it against anything. Every
stub must appear in the `PORTS` block. That block is also what the runtime assertions iterate, so a
stub dropped from it stops being checked twice over rather than silently once.

Verified live: deleting `list_check_results` from the `Dq` stub produces

    error: Incompatible types in assignment (expression has type "Dq", variable has type
    "DqRepository")  [assignment]
    note: "Dq" is missing following "DqRepository" protocol member: list_check_results

Bodies are `raise NotImplementedError` rather than `...` on purpose. In a non-Protocol class `...`
is a function that silently returns `None`, which under `--strict` would demand the return types be
`Optional` and quietly weaken every signature this file exists to pin down.

`make types` runs `mypy src scripts tests/unit/test_port_conformance.py` — this one test file, not
all of `tests/`, because the project's `per-file-ignores` deliberately exempts tests from `ANN`, and
strict-checking the rest would fail on annotations nobody wants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from sentinel.domain.entities import (
    Account,
    Alert,
    DqCheckResult,
    DqRun,
    HumanDecision,
    Investigation,
    ToolCall,
    Transaction,
)
from sentinel.domain.enums import AlertStatus, FeatureWindow, Severity, Typology
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
    AccountProfile,
    CaseMatch,
    FeatureVector,
    FundFlowResult,
    LLMCompletion,
    PolicyChunk,
    SharedIdentifierResult,
    Subgraph,
)


class Accounts:
    def get(self, account_id: str) -> Account | None:
        raise NotImplementedError

    def get_profile(self, account_id: str) -> AccountProfile | None:
        raise NotImplementedError


class Txns:
    def get(self, txn_id: str) -> Transaction | None:
        raise NotImplementedError

    def list_for_account(
        self, account_id: str, *, start: datetime, end: datetime, limit: int
    ) -> tuple[Transaction, ...]:
        raise NotImplementedError


class Alerts:
    def get(self, alert_id: str) -> Alert | None:
        raise NotImplementedError

    def list_open(
        self,
        *,
        limit: int,
        offset: int = 0,
        severity: Severity | None = None,
        status: AlertStatus | None = None,
    ) -> tuple[Alert, ...]:
        raise NotImplementedError

    def update_status(self, alert_id: str, status: AlertStatus) -> None:
        raise NotImplementedError


class Investigations:
    def save(self, investigation: Investigation) -> None:
        raise NotImplementedError

    def get(self, investigation_id: str) -> Investigation | None:
        raise NotImplementedError

    def save_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        raise NotImplementedError

    def list_tool_calls(self, investigation_id: str) -> tuple[ToolCall, ...]:
        raise NotImplementedError

    def get_decision(self, investigation_id: str) -> HumanDecision | None:
        raise NotImplementedError

    def save_decision(self, decision: HumanDecision) -> None:
        raise NotImplementedError


class Dq:
    def list_runs(self, *, limit: int, offset: int = 0) -> tuple[DqRun, ...]:
        raise NotImplementedError

    def list_check_results(self, run_id: str) -> tuple[DqCheckResult, ...]:
        raise NotImplementedError


class Features:
    def get_features(self, account_id: str, window: FeatureWindow) -> FeatureVector | None:
        raise NotImplementedError

    def put_features(self, vectors: Sequence[FeatureVector]) -> None:
        raise NotImplementedError


class Graph:
    def trace_fund_flow(
        self, account_id: str, *, max_hops: int = 3, min_amount: Decimal = Decimal(0)
    ) -> FundFlowResult:
        raise NotImplementedError

    def shared_identifiers(self, account_id: str) -> SharedIdentifierResult:
        raise NotImplementedError

    def subgraph(
        self, account_id: str, *, max_hops: int = 2, min_amount: Decimal = Decimal(0)
    ) -> Subgraph | None:
        raise NotImplementedError


class Vectors:
    def search_policy(
        self,
        query: str,
        *,
        typology: Typology | None = None,
        limit: int = 4,
        score_threshold: float = 0.35,
    ) -> tuple[PolicyChunk, ...]:
        raise NotImplementedError

    def search_cases(
        self, query: str, *, limit: int = 3, score_threshold: float = 0.35
    ) -> tuple[CaseMatch, ...]:
        raise NotImplementedError


class Llm:
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMCompletion:
        raise NotImplementedError


# --- the assignments ARE the test ----------------------------------------------------------------
# Each stub is bound to its protocol type, which is what makes `mypy --strict` compare the two. Do
# not remove an annotation here to "clean up" — an unannotated binding checks nothing.

accounts: AccountRepository = Accounts()
transactions: TransactionRepository = Txns()
alerts: AlertRepository = Alerts()
investigations: InvestigationRepository = Investigations()
dq: DqRepository = Dq()
features: FeatureStore = Features()
graph: GraphRepository = Graph()
vectors: VectorStore = Vectors()
llm: LLMClient = Llm()

PORTS: tuple[tuple[object, type], ...] = (
    (accounts, AccountRepository),
    (transactions, TransactionRepository),
    (alerts, AlertRepository),
    (investigations, InvestigationRepository),
    (dq, DqRepository),
    (features, FeatureStore),
    (graph, GraphRepository),
    (vectors, VectorStore),
    (llm, LLMClient),
)


def test_every_stub_adapter_satisfies_its_port_at_runtime_too() -> None:
    """The weaker half of the check, so this file also participates in `make test`.

    A green `pytest` here means the method names line up. It does **not** mean the signatures do —
    only the `mypy` run over this file proves that, and it is the reason the file exists. Keeping
    both means a contributor who runs only the tests still gets told when a stub loses a method.
    """
    for stub, port in PORTS:
        assert isinstance(stub, port), f"{type(stub).__name__} no longer satisfies {port.__name__}"


def test_every_port_has_a_stub() -> None:
    """Guards the negative control: a port with no stub in `PORTS` is a port nobody checks."""
    from sentinel.domain import ports

    checked = {port.__name__ for _, port in PORTS}
    assert checked == set(ports.__all__), f"unchecked ports: {set(ports.__all__) - checked}"


def test_every_stub_binding_still_carries_its_port_annotation() -> None:
    """The other half of the negative control, and the one worth having a test for.

    Measured, not assumed: deleting the `: AccountRepository` annotation from the binding above
    leaves `mypy --strict` **passing silently** — it has nothing to compare the stub against any
    more, and `isinstance` still succeeds because it is the same object. So the static check can be
    disarmed by an edit that looks like tidying, and no other assertion in this file would notice.

    `from __future__ import annotations` makes these values plain strings, which is all that is
    needed: an unannotated binding simply does not appear in `__annotations__` at all.
    """
    declared = set(__annotations__.values())
    for _, port in PORTS:
        assert port.__name__ in declared, (
            f"the stub bound for {port.__name__} lost its type annotation, which silently disables "
            f"the static conformance check for that port. Restore `name: {port.__name__} = Stub()`."
        )
