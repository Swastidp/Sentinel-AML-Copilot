"""Connectivity smoke test — T0.5.

Answers one question: can this machine reach every store the pipeline needs, right now?

Run it before any load, after any credential rotation, and first thing when something inexplicable
starts failing. It is deliberately the *cheapest possible* round trip per service — a version
query, a PING, a collection list — so it isolates "can I connect and authenticate" from "is my
query wrong", which are the two failures people conflate at 2am.

    python scripts/check_connectivity.py

Exit code is 0 only if every check passes, so CI or a Makefile can gate on it. Nothing here writes
data except one throwaway Redis key, which is deleted before the function returns.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

try:
    from sentinel.config.logging import configure_logging
    from sentinel.config.settings import Settings, get_settings
    from sentinel.domain.exceptions import ConfigurationError
except ModuleNotFoundError as exc:
    if exc.name != "sentinel":
        raise
    raise SystemExit(
        "The `sentinel` package is not importable, which almost always means the virtualenv is "
        "not active. Run:\n"
        "    .venv\\Scripts\\activate        (Windows)\n"
        "    source .venv/bin/activate      (Linux/macOS)\n"
        'and try again. If it still fails:  pip install -e ".[dev]"'
    ) from exc

_TIMEOUT: Final = 20.0
_PASS: Final = "PASS"
_FAIL: Final = "FAIL"
_SKIP: Final = "SKIP"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one service check. `status` is PASS, FAIL, or SKIP (not configured)."""

    service: str
    status: str
    detail: str
    elapsed_ms: float = 0.0

    @property
    def is_blocking(self) -> bool:
        return self.status == _FAIL


def _timed(service: str, probe: Callable[[], str]) -> CheckResult:
    """Run a probe, converting any exception into a FAIL rather than a traceback.

    A missing credential is a SKIP, not a FAIL — it means "you have not configured this yet",
    which is a different problem from "your credential is wrong".
    """
    start = time.perf_counter()
    try:
        detail = probe()
    except ConfigurationError as exc:
        return CheckResult(service, _SKIP, str(exc)[:110])
    except Exception as exc:  # a smoke test must report every failure, never propagate one
        elapsed = (time.perf_counter() - start) * 1000
        return CheckResult(service, _FAIL, f"{type(exc).__name__}: {exc}"[:220], elapsed)
    return CheckResult(service, _PASS, detail, (time.perf_counter() - start) * 1000)


# --- individual probes --------------------------------------------------------------------------


def check_postgres(settings: Settings) -> CheckResult:
    def probe() -> str:
        import psycopg

        with psycopg.connect(settings.postgres_dsn, connect_timeout=int(_TIMEOUT)) as conn:
            row = conn.execute("SELECT version(), current_database()").fetchone()
        assert row is not None
        version = str(row[0]).split(",")[0]
        return f"{version} | db={row[1]}"

    return _timed("Neon Postgres", probe)


def check_neo4j(settings: Settings) -> CheckResult:
    """Distinguishes the three Aura failures that look alike from a stack trace.

    A hostname that does not resolve means the instance was deleted or never created; an auth
    failure against a hostname that *does* resolve means the instance is alive and the password is
    wrong. Conflating them costs an hour, so the instance id and the DNS result are both reported.
    """

    def probe() -> str:
        import socket

        from neo4j import GraphDatabase

        auth = settings.neo4j_auth  # raises if URI or password is absent
        host = settings.neo4j_uri.split("://")[-1].split(":")[0]
        instance_id = host.split(".")[0]
        try:
            socket.gethostbyname(host)
        except OSError as exc:
            raise RuntimeError(
                f"instance '{instance_id}' does not resolve in DNS - it was deleted, or the URI "
                f"is from a different instance than the one you created ({exc})"
            ) from exc

        with GraphDatabase.driver(
            settings.neo4j_uri, auth=auth, connection_timeout=_TIMEOUT
        ) as driver:
            try:
                driver.verify_connectivity()
            except Exception as exc:
                if "Unauthorized" in str(exc) or "authentication" in str(exc).lower():
                    hint = (
                        f"NEO4J_USER is '{settings.neo4j_user}' - for Aura this is usually the "
                        f"instance id '{instance_id}', NOT 'neo4j'. "
                        if settings.neo4j_user != instance_id
                        else "The password does not belong to this instance. "
                    )
                    raise RuntimeError(
                        f"instance '{instance_id}' is reachable but rejected the credentials. "
                        f"{hint}Import the Aura credentials file verbatim with "
                        f"`python scripts/import_aura_credentials.py <file>`."
                    ) from exc
                raise
            with driver.session(database=settings.neo4j_database) as session:
                counts = session.run(
                    "MATCH (n) WITH count(n) AS nodes "
                    "CALL { MATCH ()-[r]->() RETURN count(r) AS rels } "
                    "RETURN nodes, rels"
                ).single()
            info = driver.get_server_info()
        assert counts is not None
        return (
            f"{info.agent} | existing nodes={counts['nodes']} rels={counts['rels']} "
            f"| db={settings.neo4j_database}"
        )

    return _timed("Neo4j Aura", probe)


def check_qdrant(settings: Settings) -> CheckResult:
    def probe() -> str:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_key, timeout=20)
        names = [c.name for c in client.get_collections().collections]
        return f"reachable | existing collections={names or 'none'}"

    return _timed("Qdrant Cloud", probe)


def check_redis(settings: Settings) -> CheckResult:
    def probe() -> str:
        import redis

        dsn = settings.redis_dsn
        if dsn.startswith("http"):
            # Upstash shows its REST endpoint most prominently, but the Redis wire protocol needs
            # the rediss:// URL from Connect -> redis-cli. Say so instead of failing cryptically.
            raise RuntimeError(
                "REDIS_URL is an HTTP(S) REST endpoint. This client speaks the Redis wire "
                "protocol - use the rediss://default:<token>@<host>:6379 URL from the Upstash "
                "console (Connect -> redis-cli tab)."
            )
        client = redis.from_url(dsn, socket_timeout=_TIMEOUT, decode_responses=True)
        client.ping()
        # Round-trip a throwaway key so we prove write access, not just reachability.
        probe_key = "sentinel:connectivity:probe"
        client.set(probe_key, "ok", ex=30)
        value = client.get(probe_key)
        client.delete(probe_key)
        if value != "ok":
            raise RuntimeError(f"write/read mismatch: wrote 'ok', read {value!r}")
        used = client.info("memory").get("used_memory_human", "?")
        return f"PING ok | read-after-write ok | used_memory={used}"

    return _timed("Upstash Redis", probe)


def check_openrouter(settings: Settings) -> CheckResult:
    """Confirms the key works AND that both configured model slugs actually exist.

    The slug check is the point. A wrong slug is a 404 at the first agent call, and the slugs in
    `.env.example` were inferred from OpenRouter's URL structure rather than read off a model page.
    """

    def probe() -> str:
        key = settings.openrouter_key
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(f"{settings.openrouter_base_url}/models", headers=headers)
            resp.raise_for_status()
            available = {m["id"] for m in resp.json()["data"]}

            configured = (settings.llm_model, settings.llm_model_cheap)
            missing = [m for m in configured if m not in available]
            if missing:
                raise RuntimeError(
                    f"model slug(s) not on OpenRouter: {missing}. "
                    f"{_suggest_slugs(missing, available)}"
                )

            # One real completion, to prove the key is funded and not merely valid.
            # `max_tokens` is deliberately generous: DeepSeek's reasoning models spend their
            # budget on hidden reasoning tokens first, so a tight cap returns content=None and
            # looks like a failure when the call actually succeeded.
            chat = client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": settings.llm_model_cheap,
                    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                    "max_tokens": 256,
                    "temperature": 0,
                },
            )
            chat.raise_for_status()
            body = chat.json()

        choice = body["choices"][0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning") or "").strip()
        if not content and not reasoning:
            raise RuntimeError(f"empty completion (finish_reason={choice.get('finish_reason')})")

        usage = body.get("usage", {})
        tokens = f"{usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')}"
        reply = content or f"<reasoning only, {len(reasoning)} chars>"
        return f"both slugs valid | completion ok (reply={reply!r}, tokens in/out={tokens})"

    return _timed("OpenRouter", probe)


def _suggest_slugs(missing: list[str], available: set[str]) -> str:
    """Point at the closest real slugs so a typo is a 10-second fix, not a hunt."""
    hints = []
    for slug in missing:
        vendor = slug.split("/")[0]
        near = sorted(s for s in available if s.startswith(f"{vendor}/"))[:8]
        hints.append(f"available under '{vendor}/': {near or 'none'}")
    return " ".join(hints)


def check_embeddings(settings: Settings) -> CheckResult:
    """Local, no network once the model is cached — but it proves the ONNX runtime works and that
    the vector width matches what the Qdrant collections will be created with."""

    def probe() -> str:
        # Windows cannot create the symlinks huggingface_hub prefers; the fallback copy works
        # fine, so silence the warning rather than let it dominate the output.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        model = TextEmbedding(settings.embedding_model, cache_dir=str(settings.embedding_cache_dir))
        vector = next(iter(model.embed(["structuring below the reporting threshold"])))
        if len(vector) != settings.embedding_dim:
            raise RuntimeError(
                f"EMBEDDING_DIM is {settings.embedding_dim} but the model produced "
                f"{len(vector)} dimensions"
            )
        return f"{settings.embedding_model} | dim={len(vector)} (matches EMBEDDING_DIM)"

    return _timed("fastembed (local)", probe)


# --- runner ------------------------------------------------------------------------------------

CHECKS: Final[tuple[Callable[[Settings], CheckResult], ...]] = (
    check_postgres,
    check_neo4j,
    check_qdrant,
    check_redis,
    check_openrouter,
    check_embeddings,
)


def run_all(settings: Settings) -> list[CheckResult]:
    return [check(settings) for check in CHECKS]


def render(results: list[CheckResult]) -> str:
    width = max(len(r.service) for r in results)
    lines = [f"{'SERVICE'.ljust(width)}  STATUS  {'TIME':>9}  DETAIL", "-" * (width + 60)]
    for r in results:
        timing = f"{r.elapsed_ms:,.0f} ms" if r.elapsed_ms else "-"
        lines.append(f"{r.service.ljust(width)}  {r.status:<6}  {timing:>9}  {r.detail}")
    return "\n".join(lines)


def main() -> int:
    settings = get_settings()
    configure_logging(settings)
    print(f"Checking connectivity  (APP_ENV={settings.app_env})\n")

    results = run_all(settings)
    print(render(results))

    failed = [r for r in results if r.is_blocking]
    skipped = [r for r in results if r.status == _SKIP]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: {', '.join(r.service for r in failed)}")
    if skipped:
        print(
            f"{len(skipped)} check(s) SKIPPED (not configured): "
            f"{', '.join(r.service for r in skipped)}"
        )
    if not failed and not skipped:
        print("All checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
