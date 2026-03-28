"""Tests for _redact_sensitive_values helper in helm.py."""

from __future__ import annotations

import re

import pytest

from vaig.tools.gke.helm import (
    _DEFAULT_SENSITIVE_KEYS,
    _REDACTED,
    _build_sensitive_pattern,
    _is_sensitive_key,
    _is_sensitive_value,
    _redact_sensitive_values,
)

# ── Test: flat dict with sensitive keys ──────────────────────


class TestFlatDictRedaction:
    """Flat (non-nested) dict redaction."""

    def test_password_is_redacted(self) -> None:
        data = {"username": "admin", "password": "s3cret"}
        result = _redact_sensitive_values(data)
        assert result["username"] == "admin"
        assert result["password"] == _REDACTED

    def test_token_is_redacted(self) -> None:
        data = {"access_token": "abc123"}
        result = _redact_sensitive_values(data)
        assert result["access_token"] == _REDACTED

    def test_secret_is_redacted(self) -> None:
        data = {"client_secret": "xyz"}
        result = _redact_sensitive_values(data)
        assert result["client_secret"] == _REDACTED

    def test_key_is_redacted(self) -> None:
        data = {"api_key": "k-123", "tls_key": "-----BEGIN-----"}
        result = _redact_sensitive_values(data)
        assert result["api_key"] == _REDACTED
        assert result["tls_key"] == _REDACTED

    def test_credential_is_redacted(self) -> None:
        data = {"db_credential": "user:pass"}
        result = _redact_sensitive_values(data)
        assert result["db_credential"] == _REDACTED

    def test_private_is_redacted(self) -> None:
        data = {"private_key_pem": "-----BEGIN RSA-----"}
        result = _redact_sensitive_values(data)
        assert result["private_key_pem"] == _REDACTED

    def test_auth_is_redacted(self) -> None:
        data = {"basic_auth": "admin:pass"}
        result = _redact_sensitive_values(data)
        assert result["basic_auth"] == _REDACTED

    def test_connection_string_is_redacted(self) -> None:
        data = {"connection_string": "postgres://user:pass@host/db"}
        result = _redact_sensitive_values(data)
        assert result["connection_string"] == _REDACTED

    def test_connectionstring_no_underscore_is_redacted(self) -> None:
        data = {"connectionstring": "Server=myServer;Database=myDB"}
        result = _redact_sensitive_values(data)
        assert result["connectionstring"] == _REDACTED

    def test_apikey_single_word_is_redacted(self) -> None:
        data = {"apikey": "abc"}
        result = _redact_sensitive_values(data)
        assert result["apikey"] == _REDACTED

    def test_multiple_sensitive_keys(self) -> None:
        data = {
            "host": "db.example.com",
            "port": 5432,
            "db_password": "hunter2",
            "secret_token": "tok_abc",
        }
        result = _redact_sensitive_values(data)
        assert result["host"] == "db.example.com"
        assert result["port"] == 5432
        assert result["db_password"] == _REDACTED
        assert result["secret_token"] == _REDACTED


# ── Test: nested dicts ───────────────────────────────────────


class TestNestedDictRedaction:
    """Multi-level nested dict redaction."""

    def test_two_levels_deep(self) -> None:
        data = {
            "database": {
                "host": "localhost",
                "password": "s3cret",
            },
        }
        result = _redact_sensitive_values(data)
        assert result["database"]["host"] == "localhost"
        assert result["database"]["password"] == _REDACTED

    def test_three_levels_deep(self) -> None:
        data = {
            "app": {
                "services": {
                    "redis": {
                        "host": "redis.local",
                        "auth_token": "tok-xyz",
                    },
                },
            },
        }
        result = _redact_sensitive_values(data)
        assert result["app"]["services"]["redis"]["host"] == "redis.local"
        assert result["app"]["services"]["redis"]["auth_token"] == _REDACTED

    def test_preserves_structure(self) -> None:
        data = {
            "level1": {
                "level2": {
                    "safe": "visible",
                    "secret": "hidden",
                },
                "also_safe": 42,
            },
        }
        result = _redact_sensitive_values(data)
        assert set(result.keys()) == {"level1"}
        assert set(result["level1"].keys()) == {"level2", "also_safe"}
        assert set(result["level1"]["level2"].keys()) == {"safe", "secret"}
        assert result["level1"]["level2"]["safe"] == "visible"
        assert result["level1"]["level2"]["secret"] == _REDACTED


# ── Test: lists of dicts ─────────────────────────────────────


class TestListsOfDicts:
    """Lists containing dicts with sensitive keys."""

    def test_list_of_dicts_with_secrets(self) -> None:
        data = {
            "users": [
                {"name": "alice", "password": "pass1"},
                {"name": "bob", "password": "pass2"},
            ],
        }
        result = _redact_sensitive_values(data)
        assert result["users"][0]["name"] == "alice"
        assert result["users"][0]["password"] == _REDACTED
        assert result["users"][1]["name"] == "bob"
        assert result["users"][1]["password"] == _REDACTED

    def test_nested_list_of_dicts(self) -> None:
        data = {
            "config": {
                "backends": [
                    {"url": "https://api.example.com", "api_key": "k1"},
                    {"url": "https://backup.example.com", "api_key": "k2"},
                ],
            },
        }
        result = _redact_sensitive_values(data)
        assert result["config"]["backends"][0]["url"] == "https://api.example.com"
        assert result["config"]["backends"][0]["api_key"] == _REDACTED
        assert result["config"]["backends"][1]["api_key"] == _REDACTED

    def test_list_of_non_dict_items_unchanged(self) -> None:
        data = {"tags": ["web", "production", "v2"]}
        result = _redact_sensitive_values(data)
        assert result["tags"] == ["web", "production", "v2"]

    def test_mixed_list(self) -> None:
        """List with mixed types (dicts, strings, numbers)."""
        data = {
            "items": [
                {"secret": "hidden"},
                "plain-string",
                42,
                None,
            ],
        }
        result = _redact_sensitive_values(data)
        assert result["items"][0]["secret"] == _REDACTED
        assert result["items"][1] == "plain-string"
        assert result["items"][2] == 42
        assert result["items"][3] is None


# ── Test: case insensitivity ─────────────────────────────────


class TestCaseInsensitivity:
    """Sensitive key matching must be case-insensitive."""

    def test_lowercase(self) -> None:
        assert _redact_sensitive_values({"password": "x"})["password"] == _REDACTED

    def test_uppercase(self) -> None:
        assert _redact_sensitive_values({"PASSWORD": "x"})["PASSWORD"] == _REDACTED

    def test_mixed_case(self) -> None:
        assert _redact_sensitive_values({"Password": "x"})["Password"] == _REDACTED

    def test_camel_like_db_password(self) -> None:
        assert _redact_sensitive_values({"db_Password": "x"})["db_Password"] == _REDACTED

    def test_all_caps_compound(self) -> None:
        assert _redact_sensitive_values({"DB_PASSWORD": "x"})["DB_PASSWORD"] == _REDACTED

    def test_title_case_token(self) -> None:
        assert _redact_sensitive_values({"Access_Token": "x"})["Access_Token"] == _REDACTED


# ── Test: non-sensitive keys NOT redacted ────────────────────


class TestNonSensitiveKeysPreserved:
    """Ensure innocent keys are left alone."""

    def test_host_not_redacted(self) -> None:
        data = {"host": "db.example.com"}
        assert _redact_sensitive_values(data)["host"] == "db.example.com"

    def test_port_not_redacted(self) -> None:
        data = {"port": 5432}
        assert _redact_sensitive_values(data)["port"] == 5432

    def test_replica_count_not_redacted(self) -> None:
        data = {"replicaCount": 3}
        assert _redact_sensitive_values(data)["replicaCount"] == 3

    def test_image_tag_not_redacted(self) -> None:
        data = {"image": {"repository": "nginx", "tag": "1.25"}}
        result = _redact_sensitive_values(data)
        assert result["image"]["repository"] == "nginx"
        assert result["image"]["tag"] == "1.25"

    def test_service_type_not_redacted(self) -> None:
        data = {"service": {"type": "ClusterIP"}}
        assert _redact_sensitive_values(data)["service"]["type"] == "ClusterIP"

    def test_boolean_values_not_redacted(self) -> None:
        data = {"enabled": True, "debug": False}
        result = _redact_sensitive_values(data)
        assert result["enabled"] is True
        assert result["debug"] is False


# ── Test: empty dict, None values ────────────────────────────


class TestEdgeCases:
    """Empty dicts, None values, and other edge cases."""

    def test_empty_dict(self) -> None:
        assert _redact_sensitive_values({}) == {}

    def test_none_value_for_sensitive_key(self) -> None:
        """Even None values under sensitive keys should be redacted."""
        data = {"password": None}
        result = _redact_sensitive_values(data)
        assert result["password"] == _REDACTED

    def test_none_value_for_non_sensitive_key(self) -> None:
        data = {"description": None}
        result = _redact_sensitive_values(data)
        assert result["description"] is None

    def test_numeric_sensitive_value(self) -> None:
        """Numeric values under sensitive keys should also be redacted."""
        data = {"secret_port": 9999}
        result = _redact_sensitive_values(data)
        assert result["secret_port"] == _REDACTED

    def test_does_not_mutate_original(self) -> None:
        """The original dict must NOT be modified."""
        data = {"password": "original", "host": "localhost"}
        result = _redact_sensitive_values(data)
        assert result["password"] == _REDACTED
        assert data["password"] == "original"

    def test_deeply_nested_empty_dict(self) -> None:
        data = {"a": {"b": {"c": {}}}}
        result = _redact_sensitive_values(data)
        assert result == {"a": {"b": {"c": {}}}}

    def test_empty_string_value_sensitive_key(self) -> None:
        data = {"password": ""}
        result = _redact_sensitive_values(data)
        assert result["password"] == _REDACTED


# ── Test: word-boundary matching (the tricky part) ───────────


class TestWordBoundaryMatching:
    """Sensitive words must match as distinct segments, not substrings.

    Boundaries are: start/end of key, ``_``, ``-``, ``.``.
    """

    # Should NOT match (word embedded inside another word)
    def test_keyboard_is_not_sensitive(self) -> None:
        """'keyboard' contains 'key' but it's not a segment boundary."""
        data = {"keyboard": "mechanical"}
        result = _redact_sensitive_values(data)
        assert result["keyboard"] == "mechanical"

    def test_tokenizer_is_not_sensitive(self) -> None:
        """'tokenizer' contains 'token' but not at a boundary."""
        data = {"tokenizer": "bpe"}
        result = _redact_sensitive_values(data)
        assert result["tokenizer"] == "bpe"

    def test_authoritative_is_not_sensitive(self) -> None:
        """'authoritative' contains 'auth' but not at a boundary."""
        data = {"authoritative": True}
        result = _redact_sensitive_values(data)
        assert result["authoritative"] is True

    def test_passworderr_is_not_sensitive(self) -> None:
        """'passworderr' — 'password' not at a trailing boundary."""
        data = {"passworderr": "something"}
        result = _redact_sensitive_values(data)
        assert result["passworderr"] == "something"

    def test_secretariat_is_not_sensitive(self) -> None:
        """'secretariat' — 'secret' not at a trailing boundary."""
        data = {"secretariat": "horse"}
        result = _redact_sensitive_values(data)
        assert result["secretariat"] == "horse"

    def test_privateer_is_not_sensitive(self) -> None:
        """'privateer' — 'private' not at a trailing boundary."""
        data = {"privateer": "ship"}
        result = _redact_sensitive_values(data)
        assert result["privateer"] == "ship"

    # SHOULD match (word at segment boundary)
    def test_db_password_matches(self) -> None:
        data = {"db_password": "abc"}
        assert _redact_sensitive_values(data)["db_password"] == _REDACTED

    def test_password_db_matches(self) -> None:
        data = {"password_db": "abc"}
        assert _redact_sensitive_values(data)["password_db"] == _REDACTED

    def test_hyphen_delimited_matches(self) -> None:
        data = {"redis-password": "abc"}
        assert _redact_sensitive_values(data)["redis-password"] == _REDACTED

    def test_dot_delimited_matches(self) -> None:
        data = {"redis.password": "abc"}
        assert _redact_sensitive_values(data)["redis.password"] == _REDACTED

    def test_standalone_key_matches(self) -> None:
        data = {"key": "value123"}
        assert _redact_sensitive_values(data)["key"] == _REDACTED

    def test_auth_standalone_matches(self) -> None:
        data = {"auth": "bearer xyz"}
        assert _redact_sensitive_values(data)["auth"] == _REDACTED

    def test_mixed_delimiters(self) -> None:
        """E.g. 'db.admin_password' — password at the end after underscore."""
        data = {"db.admin_password": "abc"}
        assert _redact_sensitive_values(data)["db.admin_password"] == _REDACTED


# ── Test: custom sensitive_keys ──────────────────────────────


class TestCustomSensitiveKeys:
    """Override the default sensitive words."""

    def test_custom_keys_only(self) -> None:
        data = {"password": "abc", "ssn": "123-45-6789", "name": "Alice"}
        result = _redact_sensitive_values(data, sensitive_keys=("ssn",))
        # password should NOT be redacted because custom keys override defaults
        assert result["password"] == "abc"
        assert result["ssn"] == _REDACTED
        assert result["name"] == "Alice"

    def test_custom_keys_segment_matching(self) -> None:
        data = {"user_ssn": "123", "ssnote": "not secret"}
        result = _redact_sensitive_values(data, sensitive_keys=("ssn",))
        assert result["user_ssn"] == _REDACTED
        assert result["ssnote"] == "not secret"

    def test_empty_custom_keys_redacts_nothing(self) -> None:
        data = {"password": "abc", "secret": "xyz"}
        result = _redact_sensitive_values(data, sensitive_keys=())
        assert result["password"] == "abc"
        assert result["secret"] == "xyz"


# ── Test: _is_sensitive_key helper ───────────────────────────


class TestIsSensitiveKey:
    """Unit tests for the segment-boundary matcher."""

    @pytest.fixture()
    def default_pattern(self) -> re.Pattern[str]:
        return _build_sensitive_pattern(_DEFAULT_SENSITIVE_KEYS)

    def test_exact_match(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("password", default_pattern) is True

    def test_prefix_segment(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("password_hash", default_pattern) is True

    def test_suffix_segment(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("db_password", default_pattern) is True

    def test_middle_segment(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("db_password_hash", default_pattern) is True

    def test_embedded_no_boundary(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("keyboard", default_pattern) is False

    def test_embedded_no_boundary_auth(self, default_pattern: re.Pattern[str]) -> None:
        assert _is_sensitive_key("authoritative", default_pattern) is False


# ── Test: _build_sensitive_pattern caching ───────────────────


class TestPatternCaching:
    """Ensure the regex cache works."""

    def test_same_tuple_returns_same_pattern(self) -> None:
        keys = ("password", "secret")
        p1 = _build_sensitive_pattern(keys)
        p2 = _build_sensitive_pattern(keys)
        assert p1 is p2

    def test_different_tuple_returns_different_pattern(self) -> None:
        p1 = _build_sensitive_pattern(("password",))
        p2 = _build_sensitive_pattern(("secret",))
        assert p1 is not p2


# ── Test: default sensitive keys coverage ────────────────────


class TestDefaultSensitiveKeys:
    """Every word in _DEFAULT_SENSITIVE_KEYS should trigger redaction."""

    @pytest.mark.parametrize("word", list(_DEFAULT_SENSITIVE_KEYS))
    def test_each_default_key_standalone(self, word: str) -> None:
        data = {word: "sensitive-value"}
        result = _redact_sensitive_values(data)
        assert result[word] == _REDACTED, f"'{word}' should be redacted"

    @pytest.mark.parametrize("word", list(_DEFAULT_SENSITIVE_KEYS))
    def test_each_default_key_as_suffix(self, word: str) -> None:
        key = f"db_{word}"
        data = {key: "sensitive-value"}
        result = _redact_sensitive_values(data)
        assert result[key] == _REDACTED, f"'{key}' should be redacted"


# ── Test: value-based sensitive pattern redaction ────────────


class TestSensitiveValueRedaction:
    """Redaction based on the VALUE content, not the key name."""

    def test_postgres_uri_redacted(self) -> None:
        data = {"dsn": "postgresql://admin:s3cret@db.host:5432/mydb"}
        result = _redact_sensitive_values(data)
        assert result["dsn"] == _REDACTED

    def test_mysql_uri_redacted(self) -> None:
        data = {"url": "mysql://root:password123@localhost:3306/app"}
        result = _redact_sensitive_values(data)
        assert result["url"] == _REDACTED

    def test_mongodb_uri_redacted(self) -> None:
        data = {"connection": "mongodb://user:pass@mongo.svc:27017/db"}
        result = _redact_sensitive_values(data)
        assert result["connection"] == _REDACTED

    def test_generic_userinfo_uri_redacted(self) -> None:
        data = {"value": "amqp://guest:guest@rabbitmq:5672/"}
        result = _redact_sensitive_values(data)
        assert result["value"] == _REDACTED

    def test_pem_private_key_redacted(self) -> None:
        data = {"cert": "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."}
        result = _redact_sensitive_values(data)
        assert result["cert"] == _REDACTED

    def test_ec_private_key_redacted(self) -> None:
        data = {"key": "-----BEGIN EC PRIVATE KEY-----\nabc123..."}
        result = _redact_sensitive_values(data)
        assert result["key"] == _REDACTED

    def test_jwt_token_redacted(self) -> None:
        data = {
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        }
        result = _redact_sensitive_values(data)
        assert result["token"] == _REDACTED

    def test_bearer_token_redacted(self) -> None:
        data = {"auth_header": "Bearer ya29.a0ARrdaM8_long_opaque_token_value_here"}
        result = _redact_sensitive_values(data)
        assert result["auth_header"] == _REDACTED

    def test_benign_url_not_redacted(self) -> None:
        """URLs without userinfo credentials should NOT be redacted."""
        data = {"endpoint": "https://api.example.com/v1/health"}
        result = _redact_sensitive_values(data)
        assert result["endpoint"] == "https://api.example.com/v1/health"

    def test_plain_string_not_redacted(self) -> None:
        data = {"name": "my-deployment", "replicas": "3"}
        result = _redact_sensitive_values(data)
        assert result["name"] == "my-deployment"
        assert result["replicas"] == "3"

    def test_nested_sensitive_value_redacted(self) -> None:
        """Value-based redaction should work in nested dicts too."""
        data = {
            "config": {
                "database_url": "postgresql://user:pass@host/db",
                "host": "localhost",
            }
        }
        result = _redact_sensitive_values(data)
        assert result["config"]["database_url"] == _REDACTED
        assert result["config"]["host"] == "localhost"

    def test_is_sensitive_value_helper(self) -> None:
        """Unit-test the _is_sensitive_value helper directly."""
        assert _is_sensitive_value("postgresql://u:p@host/db") is True
        assert _is_sensitive_value("-----BEGIN PRIVATE KEY-----") is True
        assert _is_sensitive_value("Bearer abcdefghijklmnopqrstuvwxyz") is True
        assert _is_sensitive_value("just a normal string") is False
        assert _is_sensitive_value("https://example.com") is False
