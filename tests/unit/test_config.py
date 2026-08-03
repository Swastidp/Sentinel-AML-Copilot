"""Tests for configuration and logging.

These matter more than their size suggests. `Settings` is the boundary between this codebase and
every credential it can reach, and the redaction processor is the last thing standing between a
customer's PAN and a log file.
"""

from __future__ import annotations

import pytest
import structlog
from pydantic import SecretStr, ValidationError

from sentinel.config.logging import (
    _redact_sensitive,
    bind_request_id,
    clear_request_context,
    new_request_id,
)
from sentinel.config.settings import Settings
from sentinel.domain.exceptions import ConfigurationError, MissingConfigurationError


def make_settings(**overrides) -> Settings:
    """Construct Settings in isolation — `_env_file=None` ignores any real .env on disk, so these
    tests behave identically on a developer machine and in CI."""
    return Settings(_env_file=None, **overrides)


# --- construction without credentials ----------------------------------------------------------


def test_constructs_with_no_env_file_at_all():
    """This is what lets `make check` pass in CI with zero secrets configured."""
    settings = make_settings()
    assert settings.app_env == "local"
    assert settings.graph_projection_radius == 3
    assert settings.embedding_dim == 384


def test_missing_secret_raises_naming_the_variable():
    settings = make_settings()
    with pytest.raises(MissingConfigurationError) as exc:
        _ = settings.postgres_dsn
    assert exc.value.env_key == "DATABASE_URL"
    assert "DATABASE_URL" in str(exc.value)


def test_blank_secret_is_treated_as_missing():
    settings = make_settings(database_url=SecretStr("   "))
    with pytest.raises(MissingConfigurationError):
        _ = settings.postgres_dsn


def test_placeholder_value_is_rejected():
    """A .env copied but never filled in must fail loudly, not connect to nothing."""
    settings = make_settings(neo4j_uri="neo4j+s://x.databases.neo4j.io")
    settings = settings.model_copy(update={"neo4j_password": SecretStr("replace-me")})
    with pytest.raises(ConfigurationError):
        _ = settings.neo4j_auth


def test_secret_is_present_when_configured():
    settings = make_settings(database_url=SecretStr("postgresql://u:p@host/db"))
    assert settings.postgres_dsn == "postgresql://u:p@host/db"


def test_neo4j_auth_requires_a_uri_before_a_password():
    settings = make_settings(neo4j_password=SecretStr("real-password"))
    with pytest.raises(MissingConfigurationError) as exc:
        _ = settings.neo4j_auth
    assert exc.value.env_key == "NEO4J_URI"


# --- validation --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    ["bolt://localhost:7687", "neo4j://abcd.databases.neo4j.io", "http://example.com"],
)
def test_unencrypted_neo4j_uri_is_rejected(uri):
    with pytest.raises(ValidationError, match="encrypted scheme"):
        make_settings(neo4j_uri=uri)


@pytest.mark.parametrize("uri", ["neo4j+s://abcd.databases.neo4j.io", "bolt+s://host:7687"])
def test_encrypted_neo4j_uri_is_accepted(uri):
    assert make_settings(neo4j_uri=uri).neo4j_uri == uri


@pytest.mark.parametrize("radius", [0, 5, -1])
def test_graph_radius_outside_one_to_four_is_rejected(radius):
    with pytest.raises(ValidationError):
        make_settings(graph_projection_radius=radius)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_threshold_must_be_a_probability(confidence):
    with pytest.raises(ValidationError):
        make_settings(agent_min_confidence=confidence)


@pytest.mark.parametrize(
    ("bad_url", "expected"),
    [
        ("https://apn1-x-12345.upstash.io", "REST endpoint"),
        ("redis-cli --tls -u rediss://default:tok@apn1-x.upstash.io:6379", "shell command"),
        ("apn1-x-12345.upstash.io:6379", "no URL scheme"),
        ("redis://default:tok@apn1-x-12345.upstash.io:6379", "requires TLS"),
    ],
)
def test_malformed_redis_urls_are_rejected_with_a_specific_reason(bad_url, expected):
    """The three things actually pasted into REDIS_URL, each with its own diagnosis."""
    with pytest.raises(ValidationError, match=expected):
        make_settings(redis_url=SecretStr(bad_url))


@pytest.mark.parametrize(
    "good_url",
    [
        "rediss://default:token@apn1-example-12345.upstash.io:6379",
        "redis://localhost:6379",
        "unix:///var/run/redis.sock",
    ],
)
def test_valid_redis_urls_are_accepted(good_url):
    assert make_settings(redis_url=SecretStr(good_url)).redis_dsn == good_url


def test_redis_url_error_does_not_leak_the_token():
    """A validation error is printed and often logged, so it must not echo the credential."""
    secret = "supersecrettoken123"
    with pytest.raises(ValidationError) as exc:
        make_settings(redis_url=SecretStr(f"redis-cli -u rediss://default:{secret}@h:6379"))
    assert secret not in str(exc.value)


def test_trailing_slashes_are_normalised():
    settings = make_settings(
        api_base_url="http://localhost:8000/",
        openrouter_base_url="https://openrouter.ai/api/v1/",
    )
    assert settings.api_base_url == "http://localhost:8000"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_secrets_do_not_appear_in_repr():
    """First line of defence: an unhandled traceback that dumps Settings must not print keys."""
    settings = make_settings(database_url=SecretStr("postgresql://user:hunter2@host/db"))
    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings.database_url)


def test_data_dirs_are_created(tmp_path):
    settings = make_settings(
        data_raw_path=tmp_path / "raw",
        data_curated_path=tmp_path / "curated",
        embedding_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "warehouse" / "s.duckdb",
    )
    settings.ensure_data_dirs()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "curated").is_dir()
    assert (tmp_path / "cache").is_dir()
    assert (tmp_path / "warehouse").is_dir()


# --- log redaction -----------------------------------------------------------------------------


def redact(event: dict) -> dict:
    return dict(_redact_sensitive(None, "info", event))


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "neo4j_password",
        "api_key",
        "OPENROUTER_API_KEY",
        "customer_pan",
        "pan_hash",
        "phone",
        "mobile_number",
        "email",
        "customer_name",
        "postgres_dsn",
        "pan_hash_salt",
    ],
)
def test_sensitive_keys_are_masked(key):
    assert redact({key: "sensitive-value"})[key] == "***REDACTED***"


def test_ordinary_keys_pass_through_untouched():
    event = {"event": "alert_scored", "account_id": "ACC001", "risk_score": 87}
    assert redact(event) == event


def test_nested_sensitive_keys_are_masked():
    out = redact({"event": "connect", "config": {"host": "db.neon.tech", "password": "hunter2"}})
    assert out["config"]["password"] == "***REDACTED***"
    assert out["config"]["host"] == "db.neon.tech"


def test_sensitive_keys_inside_lists_are_masked():
    out = redact({"accounts": [{"account_id": "ACC001", "customer_pan": "ABCDE1234F"}]})
    assert out["accounts"][0]["customer_pan"] == "***REDACTED***"
    assert out["accounts"][0]["account_id"] == "ACC001"


# --- request correlation -----------------------------------------------------------------------


def test_request_id_is_bound_to_the_context():
    clear_request_context()
    try:
        rid = bind_request_id()
        assert structlog.contextvars.get_contextvars()["request_id"] == rid
        assert len(rid) == 12
    finally:
        clear_request_context()


def test_an_explicit_request_id_is_honoured():
    clear_request_context()
    try:
        assert bind_request_id("abc123def456") == "abc123def456"
    finally:
        clear_request_context()


def test_clearing_the_context_removes_the_id():
    bind_request_id()
    clear_request_context()
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_request_ids_are_distinct():
    assert len({new_request_id() for _ in range(200)}) == 200
