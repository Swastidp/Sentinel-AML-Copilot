"""The nine seams every external system sits behind.

Re-exported here so that a caller writes `from sentinel.domain.ports import FeatureStore` and does
not have to know which of the five modules it lives in. That matters beyond convenience: the module
split is an organisational choice, and if call sites imported from `ports.feature_store` directly
then moving a protocol would be a rename across the codebase.

The dependency rule this package exists to enforce (`docs/01-ARCHITECTURE.md` §2):

    api / ui / agent  ->  application  ->  domain  <-  infrastructure

Adapters point *inwards*. `application/` and `agent/` are written against these protocols and never
import `infrastructure`; the concrete classes are injected once, at the FastAPI boundary. Because
these are `Protocol`s rather than ABCs, `infrastructure/` does not need to import them either — an
adapter satisfies one by having the right methods, so the arrow from `infrastructure` to `domain`
exists only at type-check time. `tests/unit/test_ports.py` asserts the import rule by walking the
AST of every module here, because an architectural claim that nothing checks is only a comment.

**Where a port belongs, and where one does not.** A port exists where `application/`, `agent/` or
`api/` needs a seam, and where a second implementation is plausible — `FeatureStore` has three
(Redis, Aerospike, in-memory), and every store here has at least a test double. Bulk loading has
neither property. The pipeline writes seven shapes — customers, accounts, devices, transactions,
account features, DQ runs, DQ check results — and it is batch ETL against Postgres with exactly one
implementation and no second candidate. So `pipeline/` calls the Postgres adapter directly, T5.2's
idempotency claim is proven by an integration test against that adapter rather than through a
Protocol, and none of those writes appear in this package. The seam would have been ceremony: an
interface with one implementation, one caller, and nothing to substitute.
`FeatureStore.put_features` is the one write that does live here, because it has three
implementations to justify it.

Nine protocols, five modules: one per external system, plus the five repositories that share
Postgres.
"""

from __future__ import annotations

from sentinel.domain.ports.feature_store import FeatureStore
from sentinel.domain.ports.graph import GraphRepository
from sentinel.domain.ports.llm import LLMClient
from sentinel.domain.ports.repositories import (
    AccountRepository,
    AlertRepository,
    DqRepository,
    InvestigationRepository,
    TransactionRepository,
)
from sentinel.domain.ports.vectors import VectorStore

__all__ = [
    "AccountRepository",
    "AlertRepository",
    "DqRepository",
    "FeatureStore",
    "GraphRepository",
    "InvestigationRepository",
    "LLMClient",
    "TransactionRepository",
    "VectorStore",
]
