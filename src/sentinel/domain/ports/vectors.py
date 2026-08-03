"""The retrieval seam — two searches over two collections, because they answer two questions.

`docs/04-VECTOR-DESIGN.md` §1 splits `aml_policy` from `case_history` deliberately, and this port
keeps the split instead of offering one `search(collection, query)`. A generic method would push the
collection name into the caller as a string, which is exactly the "one typo and the query matches
nothing" failure `enums.py` exists to prevent — and the two searches have genuinely different
contracts anyway: policy retrieval is filtered by typology and returns clauses, case retrieval is
unfiltered and returns verdicts.

Both methods take a natural-language string, never a vector. Embedding is the adapter's job: the
model (`BAAI/bge-small-en-v1.5`, 384 dimensions) is an infrastructure choice, and a port that took a
`list[float]` would force every caller — and every test fake — to know the dimensionality and own an
embedder. Swapping the embedding model would then be a change at every call site rather than in one
file.

**Retrieved text is evidence, never instruction.** `docs/04-VECTOR-DESIGN.md` §6 makes this a design
property, and part of it is enforced here: this port is read-only, so there is nothing an injected
directive inside a retrieved chunk could act on. The prompt-side half of that boundary lives in
`agent/prompts/`.

**Returning nothing is a valid answer.** Both methods return an empty tuple when no match clears the
similarity threshold, and the agent is required to say "no applicable policy found" rather than
invent one. An empty result is a value; it is not a failure and must not be raised as one.

On the "raises" clauses below: an unreachable store and a missing collection both surface as
`StoreUnavailableError`. That is the same named domain type every other port uses, which is what
lets `api/errors.py` map a Qdrant outage to `503` without importing anything about Qdrant. It also
keeps the distinction above enforceable — a missing collection is a bootstrap failure and must never
look like an empty search result.

This module lives in `domain` and imports nothing outside the standard library — no `qdrant_client`,
no `Filter`, no `PointStruct`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sentinel.domain.enums import Typology
from sentinel.domain.value_objects import CaseMatch, PolicyChunk

# Defaults from `docs/04-VECTOR-DESIGN.md` §5. They sit here as defaults rather than as required
# arguments so that a caller with no opinion gets the documented behaviour, and are overridable
# because the tuned values belong in `config/`, not in the domain.
DEFAULT_POLICY_LIMIT = 4
DEFAULT_CASE_LIMIT = 3
DEFAULT_SCORE_THRESHOLD = 0.35


@runtime_checkable
class VectorStore(Protocol):
    """Semantic search over written AML policy and closed-case history."""

    def search_policy(
        self,
        query: str,
        *,
        typology: Typology | None = None,
        limit: int = DEFAULT_POLICY_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> tuple[PolicyChunk, ...]:
        """Return policy passages relevant to `query`, most similar first.

        `typology` narrows the search when the rule engine has already established the pattern.
        It is **not** an exclusive filter: a typology-specific search also returns clauses carrying
        no typology at all, because general escalation guidance applies regardless of pattern and
        filtering it out would ground the recommendation in the narrower half of the rulebook. That
        is the `should`/`is_null` shape in §3 of the vector design, and it is a contract of this
        method rather than an implementation flourish.

        `score_threshold` is a cosine similarity floor. Below it a match is noise, and returning
        nothing is better than returning noise — so a weak match is dropped rather than ranked last.

        Ordered by `PolicyChunk.similarity` descending, at most `limit` results, possibly empty.
        Every chunk carries `doc_title` and `section_path` so the narrative can cite a written
        clause instead of paraphrasing one.

        Raises `StoreUnavailableError` if the store is unreachable or the collection is missing.
        A missing collection is a bootstrap failure, not an empty result, and the two must not look
        the same to a caller.
        """
        ...

    def search_cases(
        self,
        query: str,
        *,
        limit: int = DEFAULT_CASE_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> tuple[CaseMatch, ...]:
        """Return past cases resembling `query`, **each with the outcome that was reached**.

        The outcome is the point of the method. Three similar narratives is generic retrieval; three
        narratives plus what was decided and why lets the agent reason from precedent, which is how
        an analyst works. `CaseMatch` therefore has no variant without `outcome` and `rationale`, so
        an implementation that cannot supply them for a point must omit that point rather than
        return a half-populated match.

        No typology filter here, unlike `search_policy`, and that is deliberate: the useful
        precedent is sometimes a case of a *different* typology that turned out to be a false
        positive for the same reason. Constraining by the label the rule engine guessed hides it.

        Ordered by `CaseMatch.similarity` descending, at most `limit` results, possibly empty.

        Raises `StoreUnavailableError` on a store failure or a missing collection.
        """
        ...
