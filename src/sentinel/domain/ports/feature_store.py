"""The `FeatureStore` seam — the one port whose replaceability is a stated design goal.

`RedisFeatureStore` implements this today, `InMemoryFeatureStore` implements it in every unit test,
and `AerospikeFeatureStore` is meant to be a single new file in `infrastructure/aerospike/` rather
than a refactor. That only stays true if nothing in this module leaks a vendor concept, so the rule
here is stricter than "no imports": **no Redis vocabulary in a name, a parameter, or a docstring.**
There is no `hgetall`, no `key_prefix`, no `expire`, and deliberately no `ttl` — a time-to-live is a
Redis eviction mechanism, and the staleness this system actually cares about is already carried
honestly by `FeatureVector.as_of` and reported to callers as `feature_snapshot_age_ms`.

**Why `get_features` returns a whole vector and `put_features` takes a whole account.**

This is a port-shaping decision forced by a measured platform limit, not an implementation detail,
which is why it lives in the signatures rather than in the adapter. `docs/12-PLATFORM-LIMITS.md` §2
records that Upstash Redis bills **per command, 500k per month**, and that the original
one-key-per-counter layout would have burned the month's budget in about twelve pipeline runs.
The fix was one hash per account, and it fixes the shape of this port:

* **A read fetches an entire feature vector in one call.** A per-field accessor
  (`get_feature(account_id, window, name)`) would multiply commands by eleven and — the part that
  matters independently of cost — six separate reads can straddle a counter refresh and return a
  feature vector that never existed at any single instant. One call returns one consistent snapshot.
* **A write covers every window of one account in one call.** `put_features` therefore takes a
  sequence rather than a single vector: all four windows for an account are one record, so writing
  them one window at a time would cost four commands where one suffices — the difference between
  ~6,500 and ~26,000 writes per pipeline run.

An adapter over a store that does not bill per command is free to ignore all of that; it simply
cannot be *forced* into the expensive shape by this port, which is the guarantee being bought.

The cost constraint pushed toward the design that was already more correct. Worth remembering when
this port is next changed: a convenience method that reads one field at a time would quietly undo
both the budget and the atomicity.

On the "raises" clauses below: a store failure surfaces as `StoreUnavailableError`, which is a named
domain type and not each adapter's own. Note that this does not weaken the vendor rule above —
`StoreUnavailableError("redis", ...)` names the store in a *value*, not in a signature or an import,
so the port stays swappable. What it buys is the distinction that matters on the millisecond path: a
failed read must never arrive as `None`, which reads as "this account has no features yet".

This module lives in `domain` and imports nothing outside the standard library — a `FeatureStore`
signature must never mention `redis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sentinel.domain.enums import FeatureWindow
from sentinel.domain.value_objects import FeatureVector


@runtime_checkable
class FeatureStore(Protocol):
    """Pre-computed rolling features, keyed by account and window.

    Read on the millisecond path (`POST /v1/score` and the agent's T2 tool), written once per
    account by the nightly pipeline. Two methods, because that is the entire vocabulary those two
    callers need.
    """

    def get_features(self, account_id: str, window: FeatureWindow) -> FeatureVector | None:
        """Return the account's feature vector for one window, in a single round trip.

        Returns `None` when the account has no stored features — a legitimate state for an account
        opened since the last pipeline run, and one the caller must handle rather than treat as an
        error. It is distinct from a vector of zeros, which means "computed, and this account did
        nothing".

        The returned vector is a consistent snapshot: every field in it was written by the same
        pipeline run, so `pass_through_ratio` cannot be computed from a credit total and a debit
        total that belong to different instants. `FeatureVector.as_of` is that run's timestamp, and
        `snapshot_age_ms` turns it into the staleness figure the scoring API returns.

        Raises `StoreUnavailableError` if the store is unreachable or returns a record that
        cannot be read back as a `FeatureVector` — a partially-written record is a data error, not
        an absent one, and must not be flattened into `None`.
        """
        ...

    def put_features(self, vectors: Sequence[FeatureVector]) -> None:
        """Write every window of **one** account as a single record.

        All vectors must carry the same `account_id`; a batch spanning more than one account is a
        `DomainValidationError`. That is the precondition that lets an implementation satisfy the
        command budget above, so it is part of the contract rather than an adapter's private
        assumption. Duplicate windows within one call are also rejected — two vectors for `24H`
        would make the stored result depend on iteration order.

        Idempotent and last-write-wins per window: re-running the pipeline over an unchanged batch
        leaves the store byte-identical, and a re-run over new data replaces the windows it computed
        without disturbing windows it did not. An empty sequence is a successful no-op.

        Not atomic across accounts. A caller writing 6,500 accounts that fails halfway leaves the
        first half updated, which is correct for a feature snapshot — a partially refreshed store is
        stale, and `as_of` makes each account's staleness individually visible.

        Raises `StoreUnavailableError` on a store failure.
        """
        ...
