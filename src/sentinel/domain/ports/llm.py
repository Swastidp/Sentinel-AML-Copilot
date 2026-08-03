"""The language-model seam — one method, so that swapping providers is one variable.

`docs/06-AGENT-DESIGN.md` §7 lists "model drift / swap" as a guardrail and answers it with "model id
is env config behind one factory". This port is the other half of that answer: the three nodes that
call a model (`plan`, `reason`, `narrate`) depend on `LLMClient`, and `OpenRouterLLMClient` is the
only file that knows OpenRouter exists. A test double is a class with one method and no network.

**Why `complete` does not return a Pydantic model**, which is what the class diagram sketches. The
domain imports nothing outside the standard library, so it cannot name `BaseModel`; and it should
not want to. Validation belongs at the boundary that owns the schema — `agent/tools/base.py` and the
node that asked — because that is where the retry-once-then-degrade policy lives. A port that
promised a validated object would be claiming a guarantee it cannot make about model output. So
`complete` takes a JSON Schema as a plain mapping and returns raw content plus its usage; the caller
parses it into whatever typed shape it defined.

`json_schema` is a `Mapping[str, Any]` rather than a typed structure for the same reason JSON Schema
itself is untyped: it is a document handed to a provider, and modelling it in the domain would be
inventing a schema language. The mapping goes out unread by this layer.

**The reply may be empty, and every caller must handle it.**
`docs/12-PLATFORM-LIMITS.md` §5 records this as a measured failure, not a hypothetical: a DeepSeek
reasoning model asked for a short answer spent its entire `max_tokens` budget on hidden reasoning
tokens and returned `content=None`. `LLMCompletion.is_empty` is the branch, `finish_reason` is the
diagnosis, and a caller that assumes a string writes an empty narrative and calls the case closed.

On the "raises" clause below: transport, authentication and rate-limit failures all surface as
`StoreUnavailableError` with `kind="openrouter"` — the gateway is a store like any other as far as
the error hierarchy is concerned, and `api/errors.py` maps it to `503` with no LLM-specific import.
A refusal or a truncated answer is not in that category at all: it comes back as a completion, for
the reason given on `complete`.

This module lives in `domain` and imports nothing outside the standard library — no `openai`, no
`langchain_core`, no `httpx`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from sentinel.domain.value_objects import LLMCompletion


@runtime_checkable
class LLMClient(Protocol):
    """A single completion call against a named model, with its cost reported back."""

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        json_schema: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMCompletion:
        """Send one prompt to one model and return the reply with its token and cost accounting.

        `model` is a provider slug (`deepseek/deepseek-v4-pro`) and is always explicit — there is no
        default. Model allocation is a per-node decision worth ~5x in cost between the reasoning and
        the narration model (`docs/06-AGENT-DESIGN.md` §8), so a caller that forgets to choose must
        fail to compile rather than silently spend on the expensive one.

        `max_tokens` is likewise required. It is the only hard ceiling on the cost of a single call,
        and a default here would be a budget decision hidden in the domain layer.

        `temperature` defaults to `0.0`. In a compliance workflow the same evidence should produce
        the same recommendation; a caller that wants variation has to ask for it in writing.

        Pass `json_schema` to request structured output. When given, `LLMCompletion.data` holds
        the parsed JSON object on success and `text` may be `None`; when omitted, `text` holds the
        prose and `data` is `None`. **Neither is guaranteed** — see the module docstring on empty
        replies, and check `LLMCompletion.is_empty` before reading either. The returned `data` is
        unvalidated model output that merely parsed as JSON; conforming it to the schema is the
        caller's job.

        Not idempotent and not cached at this layer: two identical calls cost twice and may differ
        even at temperature zero, because provider-side non-determinism is real. Nothing in this
        system depends on a repeated call returning the same words — reproducibility comes from the
        persisted tool trace and from every number being computed in Python.

        Returns a populated `LLMCompletion` whose `tokens_in`, `tokens_out` and `cost_usd` reflect
        the provider's own reported usage, not an estimate, because those figures are persisted per
        investigation and summed into the reported 30-day spend. When a provider reports no usage,
        an implementation must raise rather than substitute a guess — an invented cost in a
        governance report is worse than a missing one.

        Raises `StoreUnavailableError` on a transport failure, an authentication failure, a rate
        limit, or a reply that is not parseable JSON when `json_schema` was requested. A refusal or
        a truncated answer is **not** an error: it returns a completion with `finish_reason` set,
        because the agent's degraded path records it in `state.errors` and trips the human gate,
        which an exception thrown out of the node would bypass.
        """
        ...
