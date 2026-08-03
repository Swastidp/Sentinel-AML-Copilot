"""The graph seam — the three questions no SQL query answers cleanly.

`docs/03-GRAPH-SCHEMA.md` §1 is unusually blunt about the boundary this port sits on: a fixed
one-hop lookup belongs in Postgres, and the graph earns its place only for variable-depth structure.
So there is no `get_neighbours` here and no per-payment method. Three methods, mapping onto the five
load-bearing Cypher queries in §5 of that document:

| Method | Cypher |
|---|---|
| `trace_fund_flow` | Q1 downstream trace, Q2 circular flow, Q3 fan signature, Q5 for the ring |
| `shared_identifiers` | Q4 shared-device ring across different customers |
| `subgraph` | Q5 k-hop edge list |

`trace_fund_flow` deliberately bundles four queries behind one call. `docs/06-AGENT-DESIGN.md` T3
specifies exactly that composite as one tool result, and the Qdrant/Aura round trip from India is
measured in hundreds of milliseconds (`docs/12-PLATFORM-LIMITS.md` §4), so exposing the four
separately would put four sequential network hops inside the agent's gather loop for evidence it
always wants together. `shared_identifiers` stays separate for the opposite reason, stated in T6: it
lets the agent ask a narrow identity question cheaply without paying for a traversal.

Q5 backs both `trace_fund_flow` and `subgraph` because two callers want different shapes of the same
neighbourhood — the agent wants ring membership computed from it, the console wants nodes and edges
to draw. Component analysis runs in `networkx` in the adapter, not in the store: Graph Data Science
is an AuraDS feature and this design does not assume it is available.

**Every method here is read-only, and that is a security control rather than an oversight.** The
agent reaches the graph through these, and `docs/04-VECTOR-DESIGN.md` §6 relies on there being
nothing an injected instruction in retrieved text could act on. A mutation method added to this port
would remove that guarantee for the whole agent.

On the "raises" clauses below: an unreachable store and a timed-out traversal both surface as
`StoreUnavailableError`, and the reason that is a *named domain* type rather than each adapter's own
matters most here. Three of this port's answers degrade to "nothing found" — zero downstream paths,
zero shared devices, `NO_CLEAR_SIGNATURE` — and every one of them is evidence the agent may reason
from. A failed traversal returned as one of those is not a missing answer, it is a fabricated one.

This module lives in `domain` and imports nothing outside the standard library — no `neo4j` types,
no `Record`, no driver session.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from sentinel.domain.value_objects import FundFlowResult, SharedIdentifierResult, Subgraph


@runtime_checkable
class GraphRepository(Protocol):
    """Structural questions about how money and identity connect accounts.

    Reads the projected graph, which is flagged accounts plus their N-hop neighbourhood rather than
    the whole customer base (`docs/03-GRAPH-SCHEMA.md` §3.2). Every caller inherits a consequence of
    that: **"not in the graph" is a different fact from "in the graph with no connections"**, and
    the three methods say so in two different ways rather than one.

    `trace_fund_flow` and `shared_identifiers` return a populated result object with empty tuples
    inside it, because their answers degrade gracefully — zero downstream paths and zero shared
    devices are meaningful findings the agent can report either way.

    `subgraph` returns `None` instead, and that asymmetry is deliberate. A `Subgraph` is required to
    contain its seed node, with a `risk_rating` and a flag state. For an account the projection does
    not hold, an adapter has no graph source for those values, so an "empty" `Subgraph` could only
    be built by inventing them — the same class of fabrication the dangling-edge guard on `Subgraph`
    exists to prevent. `None` says "not in the projection" without making anything up.
    """

    def trace_fund_flow(
        self,
        account_id: str,
        *,
        max_hops: int = 3,
        min_amount: Decimal = Decimal(0),
    ) -> FundFlowResult:
        """Trace where money from this account went, whether any came back, and who moved with it.

        `max_hops` bounds the traversal depth and must be between 1 and 4. The upper bound is not
        arbitrary: `-[:TRANSFERRED*1..4]->` on the free tier is the depth the element budget in
        §7 of the graph schema was sized for, and an unbounded pattern on a dense neighbourhood is
        the one query shape that can hang the store. Values outside the range are a
        `DomainValidationError`.

        `min_amount` drops edges whose aggregated total is below it, which is how a caller excludes
        the long tail of small transfers that connect everything to everything. It is a `Decimal`
        because it is money and it is compared against money.

        `downstream_paths` is ordered by `path_value` descending — the largest flow first, because
        that is the one an analyst reads. It is capped by the implementation (25 paths in Q1); a
        capped result is a truncation, not a complete answer, and callers must not compute a total
        exposure by summing it.

        Always returns a `FundFlowResult`, never `None`. An account with no projected edges yields
        empty path tuples, zero fan degrees, and `MuleSignature.NO_CLEAR_SIGNATURE` — an honest
        "nothing found" that the agent can report, which is different from an unanswered question.

        Raises `StoreUnavailableError` if the store is unreachable or the traversal times out.
        """
        ...

    def shared_identifiers(self, account_id: str) -> SharedIdentifierResult:
        """Find accounts of **other** customers that share a device with this one.

        Devices only. `docs/06-AGENT-DESIGN.md` T6 also promises addresses, but no Cypher in the
        graph schema queries one and there is no account-to-address path to query — see
        `SharedIdentifierResult` for why a permanently empty field is worse than an unpromised one.

        The cross-customer restriction is the entire signal and it is enforced in the query, not by
        the caller: two accounts on one device is a person with two accounts, and two accounts of
        two people on one device is rule R007. `IdentifierLink` rejects a link that reaches no other
        customer, so an implementation that loses the inequality predicate fails loudly.

        Always returns a result, never `None`; an empty tuple is the common and correct answer, and
        also what an account outside the projection yields. Unlike `subgraph`, this needs no
        distinction: with no device evidence either way there is nothing the agent could conclude
        differently, and the object carries no field that would have to be invented.

        Ordered by `distinct_other_customers` descending, so the widest link is first, with a
        tie-break on `identifier` to keep the order total.

        Raises `StoreUnavailableError` on a store failure, which must not be flattened into an
        empty result — "no shared devices" is evidence, and a failed read is not.
        """
        ...

    def subgraph(
        self,
        account_id: str,
        *,
        max_hops: int = 2,
        min_amount: Decimal = Decimal(0),
    ) -> Subgraph | None:
        """Return the k-hop neighbourhood as nodes and edges, or `None` if the account is not in the
        projected graph.

        Defaults to 2 hops rather than 3 because this feeds a picture a human reads: a 3-hop
        neighbourhood of a hub account renders as an unreadable mesh. `max_hops` must be between 1
        and 4, same bound and same reason as `trace_fund_flow`.

        `None` means "this account was never projected" — see the class docstring for why that is a
        `None` here and an empty tuple on the other two methods. It is **not** what an isolated
        account returns: an account that is in the graph but has no qualifying edges comes back as a
        `Subgraph` holding exactly its own seed node and no edges, so the console draws one node
        rather than an empty canvas that cannot be told apart from a failed load. The API maps
        `None` to `404`.

        `nodes` and `edges` are in **unspecified order**. Nothing pages this result and a renderer
        lays it out by graph structure rather than by list position, so pinning an order would be a
        promise with no reader. Do not index into either; find the seed via `is_seed`.

        Edges are aggregated per account pair, not per payment (`docs/03-GRAPH-SCHEMA.md` §3.1), so
        `GraphEdge.amount` is a total across `txn_count` transactions.

        Raises `StoreUnavailableError` on a store failure, which must not be returned as `None` —
        an unreachable Neo4j would otherwise render as an account with no network.
        """
        ...
