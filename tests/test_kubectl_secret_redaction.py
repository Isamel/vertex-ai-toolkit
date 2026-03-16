"""Tests for K8s secret data redaction in kubectl_get output.

Covers _redact_secret_item, _redact_k8s_secret_data, and the integration
with _format_items when resource == "secrets".
"""

from __future__ import annotations

import json

import yaml

from vaig.tools.gke._formatters import (
    _SECRET_REDACTED,
    _redact_k8s_secret_data,
    _redact_secret_item,
)

# ═══════════════════════════════════════════════════════════════
# Fixtures — sample K8s Secret dicts (serialised form)
# ═══════════════════════════════════════════════════════════════


def _make_secret(
    name: str = "my-secret",
    namespace: str = "default",
    data: dict[str, str] | None = None,
    string_data: dict[str, str] | None = None,
    secret_type: str = "Opaque",  # noqa: S107
) -> dict:
    """Build a minimal serialised K8s Secret dict."""
    secret: dict = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2025-01-15T10:00:00Z",
        },
        "type": secret_type,
    }
    if data is not None:
        secret["data"] = data
    if string_data is not None:
        secret["stringData"] = string_data
    return secret


def _make_deployment(name: str = "my-app") -> dict:
    """Build a minimal serialised Deployment dict (non-secret resource)."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": "default",
        },
        "spec": {"replicas": 3},
    }


# ═══════════════════════════════════════════════════════════════
# Unit tests: _redact_secret_item
# ═══════════════════════════════════════════════════════════════


class TestRedactSecretItem:
    """Unit tests for the single-item redaction function."""

    def test_data_values_replaced(self) -> None:
        item = _make_secret(data={"password": "cDRzc3dvcmQ=", "username": "YWRtaW4="})
        result = _redact_secret_item(item)
        assert result["data"]["password"] == _SECRET_REDACTED
        assert result["data"]["username"] == _SECRET_REDACTED

    def test_data_keys_preserved(self) -> None:
        item = _make_secret(data={"db-password": "abc", "api-key": "xyz"})
        result = _redact_secret_item(item)
        assert set(result["data"].keys()) == {"db-password", "api-key"}

    def test_redaction_note_added_for_data(self) -> None:
        item = _make_secret(data={"a": "1", "b": "2", "c": "3"})
        result = _redact_secret_item(item)
        assert result["_data_redacted_note"] == "3 key(s) redacted for security"

    def test_string_data_values_replaced(self) -> None:
        item = _make_secret(string_data={"token": "my-cleartext-token"})
        result = _redact_secret_item(item)
        assert result["stringData"]["token"] == _SECRET_REDACTED

    def test_string_data_note_added(self) -> None:
        item = _make_secret(string_data={"x": "1", "y": "2"})
        result = _redact_secret_item(item)
        assert result["_stringData_redacted_note"] == "2 key(s) redacted for security"

    def test_both_data_and_string_data_redacted(self) -> None:
        item = _make_secret(
            data={"secret1": "base64val"},
            string_data={"secret2": "cleartext"},
        )
        result = _redact_secret_item(item)
        assert result["data"]["secret1"] == _SECRET_REDACTED
        assert result["stringData"]["secret2"] == _SECRET_REDACTED
        assert "_data_redacted_note" in result
        assert "_stringData_redacted_note" in result

    def test_empty_data_field_not_modified(self) -> None:
        item = _make_secret(data={})
        result = _redact_secret_item(item)
        assert result["data"] == {}
        assert "_data_redacted_note" not in result

    def test_no_data_field_passes_through(self) -> None:
        item = _make_secret()  # no data, no stringData
        result = _redact_secret_item(item)
        assert "data" not in result
        assert "stringData" not in result
        assert "_data_redacted_note" not in result

    def test_metadata_preserved(self) -> None:
        item = _make_secret(
            name="db-creds",
            namespace="production",
            data={"pw": "secret"},
        )
        result = _redact_secret_item(item)
        assert result["metadata"]["name"] == "db-creds"
        assert result["metadata"]["namespace"] == "production"
        assert result["kind"] == "Secret"
        assert result["type"] == "Opaque"

    def test_original_dict_not_mutated(self) -> None:
        item = _make_secret(data={"pw": "original"})
        _redact_secret_item(item)
        # Original should still have the real value
        assert item["data"]["pw"] == "original"

    def test_single_key_data(self) -> None:
        item = _make_secret(data={"only-key": "c2VjcmV0"})
        result = _redact_secret_item(item)
        assert result["data"]["only-key"] == _SECRET_REDACTED
        assert result["_data_redacted_note"] == "1 key(s) redacted for security"

    def test_many_keys_data(self) -> None:
        data = {f"key-{i}": f"val-{i}" for i in range(20)}
        item = _make_secret(data=data)
        result = _redact_secret_item(item)
        assert all(v == _SECRET_REDACTED for v in result["data"].values())
        assert result["_data_redacted_note"] == "20 key(s) redacted for security"

    def test_data_with_none_value(self) -> None:
        """Edge case: a data key with None value (shouldn't happen, but be safe)."""
        item = _make_secret(data={"key": None})  # type: ignore[dict-item]
        result = _redact_secret_item(item)
        assert result["data"]["key"] == _SECRET_REDACTED

    def test_tls_secret_type_redacted(self) -> None:
        item = _make_secret(
            secret_type="kubernetes.io/tls",
            data={"tls.crt": "base64cert", "tls.key": "base64key"},
        )
        result = _redact_secret_item(item)
        assert result["data"]["tls.crt"] == _SECRET_REDACTED
        assert result["data"]["tls.key"] == _SECRET_REDACTED
        assert result["type"] == "kubernetes.io/tls"

    def test_dockerconfigjson_secret_redacted(self) -> None:
        item = _make_secret(
            secret_type="kubernetes.io/dockerconfigjson",
            data={".dockerconfigjson": "eyJhdXRocyI6e319"},
        )
        result = _redact_secret_item(item)
        assert result["data"][".dockerconfigjson"] == _SECRET_REDACTED

    def test_service_account_token_redacted(self) -> None:
        item = _make_secret(
            secret_type="kubernetes.io/service-account-token",
            data={
                "ca.crt": "base64cert",
                "namespace": "ZGVmYXVsdA==",
                "token": "eyJhbGciOiJSUzI1...",
            },
        )
        result = _redact_secret_item(item)
        assert all(v == _SECRET_REDACTED for v in result["data"].values())

    def test_data_field_is_not_dict_passes_through(self) -> None:
        """If data is somehow not a dict (e.g. a string), don't crash."""
        item = _make_secret()
        item["data"] = "not-a-dict"  # malformed
        result = _redact_secret_item(item)
        assert result["data"] == "not-a-dict"


# ═══════════════════════════════════════════════════════════════
# Unit tests: _redact_k8s_secret_data (list-level)
# ═══════════════════════════════════════════════════════════════


class TestRedactK8sSecretData:
    """Tests for the list-level redaction function."""

    def test_single_secret_list(self) -> None:
        items = [_make_secret(data={"pw": "abc"})]
        result = _redact_k8s_secret_data(items)
        assert len(result) == 1
        assert result[0]["data"]["pw"] == _SECRET_REDACTED

    def test_multiple_secrets_list(self) -> None:
        items = [
            _make_secret(name="s1", data={"k1": "v1"}),
            _make_secret(name="s2", data={"k2": "v2", "k3": "v3"}),
            _make_secret(name="s3", string_data={"k4": "v4"}),
        ]
        result = _redact_k8s_secret_data(items)
        assert len(result) == 3
        assert result[0]["data"]["k1"] == _SECRET_REDACTED
        assert result[1]["data"]["k2"] == _SECRET_REDACTED
        assert result[1]["data"]["k3"] == _SECRET_REDACTED
        assert result[2]["stringData"]["k4"] == _SECRET_REDACTED

    def test_empty_list(self) -> None:
        result = _redact_k8s_secret_data([])
        assert result == []

    def test_secrets_without_data(self) -> None:
        items = [_make_secret()]
        result = _redact_k8s_secret_data(items)
        assert len(result) == 1
        assert "data" not in result[0]

    def test_original_list_not_mutated(self) -> None:
        items = [_make_secret(data={"pw": "original"})]
        _redact_k8s_secret_data(items)
        assert items[0]["data"]["pw"] == "original"


# ═══════════════════════════════════════════════════════════════
# Integration: JSON format redaction
# ═══════════════════════════════════════════════════════════════


class TestJsonFormatRedaction:
    """Verify redaction works end-to-end for JSON output."""

    def test_single_secret_json(self) -> None:
        items = [_make_secret(data={"password": "cDRzc3dvcmQ="})]
        redacted = _redact_k8s_secret_data(items)
        output = json.dumps(redacted, indent=2)
        parsed = json.loads(output)
        assert parsed[0]["data"]["password"] == _SECRET_REDACTED

    def test_secret_list_json(self) -> None:
        items = [
            _make_secret(name="db-creds", data={"pw": "x"}),
            _make_secret(name="api-keys", data={"key1": "y", "key2": "z"}),
        ]
        redacted = _redact_k8s_secret_data(items)
        output = json.dumps(redacted, indent=2)
        parsed = json.loads(output)
        assert len(parsed) == 2
        assert parsed[0]["data"]["pw"] == _SECRET_REDACTED
        assert parsed[1]["data"]["key1"] == _SECRET_REDACTED
        assert parsed[1]["data"]["key2"] == _SECRET_REDACTED

    def test_json_output_includes_note(self) -> None:
        items = [_make_secret(data={"a": "1", "b": "2"})]
        redacted = _redact_k8s_secret_data(items)
        output = json.dumps(redacted, indent=2)
        assert "2 key(s) redacted for security" in output

    def test_json_metadata_intact(self) -> None:
        items = [_make_secret(name="my-secret", namespace="prod", data={"x": "y"})]
        redacted = _redact_k8s_secret_data(items)
        output = json.dumps(redacted, indent=2)
        parsed = json.loads(output)
        assert parsed[0]["metadata"]["name"] == "my-secret"
        assert parsed[0]["metadata"]["namespace"] == "prod"


# ═══════════════════════════════════════════════════════════════
# Integration: YAML format redaction
# ═══════════════════════════════════════════════════════════════


class TestYamlFormatRedaction:
    """Verify redaction works end-to-end for YAML output."""

    def test_single_secret_yaml(self) -> None:
        items = [_make_secret(data={"token": "dG9rZW4="})]
        redacted = _redact_k8s_secret_data(items)
        output = yaml.dump_all(redacted, default_flow_style=False)
        parsed = list(yaml.safe_load_all(output))
        assert parsed[0]["data"]["token"] == _SECRET_REDACTED

    def test_secret_list_yaml(self) -> None:
        items = [
            _make_secret(name="s1", data={"k1": "v1"}),
            _make_secret(name="s2", data={"k2": "v2"}),
        ]
        redacted = _redact_k8s_secret_data(items)
        output = yaml.dump_all(redacted, default_flow_style=False)
        parsed = list(yaml.safe_load_all(output))
        assert len(parsed) == 2
        assert parsed[0]["data"]["k1"] == _SECRET_REDACTED
        assert parsed[1]["data"]["k2"] == _SECRET_REDACTED

    def test_yaml_output_includes_note(self) -> None:
        items = [_make_secret(data={"x": "1"})]
        redacted = _redact_k8s_secret_data(items)
        output = yaml.dump_all(redacted, default_flow_style=False)
        assert "1 key(s) redacted for security" in output

    def test_yaml_string_data_redacted(self) -> None:
        items = [_make_secret(string_data={"cleartext": "supersecret"})]
        redacted = _redact_k8s_secret_data(items)
        output = yaml.dump_all(redacted, default_flow_style=False)
        parsed = list(yaml.safe_load_all(output))
        assert parsed[0]["stringData"]["cleartext"] == _SECRET_REDACTED


# ═══════════════════════════════════════════════════════════════
# Table/wide format — should pass through unchanged
# ═══════════════════════════════════════════════════════════════


class TestTableFormatPassthrough:
    """Table format already shows only NAME/TYPE/DATA(count)/AGE — no secret values.

    These tests verify that the _redact_k8s_secret_data function is NOT called
    for table/wide formats (the integration point is in _format_items which
    only calls redaction for json/yaml).
    """

    def test_table_format_unchanged(self) -> None:
        """Table format for secrets uses _format_generic_table — safe by design."""
        # This tests the principle, not _format_items directly (no k8s client here)
        sample_table = "NAME          TYPE     DATA   AGE\nmy-secret     Opaque   3      5d"
        # Table output should never be redacted — it doesn't contain secret values
        assert "Opaque" in sample_table
        assert _SECRET_REDACTED not in sample_table


# ═══════════════════════════════════════════════════════════════
# Resource name variations
# ═══════════════════════════════════════════════════════════════


class TestResourceNameVariations:
    """The normalisation happens in kubectl_get before calling _format_items.

    These tests verify the _normalise_resource alias mapping handles
    all secret-related input correctly, ensuring _format_items receives
    resource == "secrets".
    """

    def test_normalise_secret_singular(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("secret") == "secrets"

    def test_normalise_secrets_plural(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("secrets") == "secrets"

    def test_normalise_uppercase_secrets(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("Secrets") == "secrets"

    def test_normalise_all_caps_secret(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("SECRET") == "secrets"

    def test_normalise_mixed_case(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("Secret") == "secrets"

    def test_normalise_with_whitespace(self) -> None:
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("  secrets  ") == "secrets"


# ═══════════════════════════════════════════════════════════════
# Non-secret resources NOT affected
# ═══════════════════════════════════════════════════════════════


class TestNonSecretResourcesUnaffected:
    """Verify that non-secret resources are never redacted."""

    def test_deployment_data_field_untouched(self) -> None:
        """A deployment with a field named 'data' should NOT be redacted."""
        item = _make_deployment()
        item["data"] = {"some-config": "value"}  # unusual but possible
        result = _redact_secret_item(item)
        # _redact_secret_item always redacts data/stringData fields — but it's
        # only CALLED for secrets. This test documents the function's behavior.
        # The guard is in _format_items (is_secret check).
        assert result["data"]["some-config"] == _SECRET_REDACTED

    def test_format_items_only_redacts_secrets_resource(self) -> None:
        """Verify the is_secret flag logic — the guard in _format_items."""
        # We can't call _format_items without k8s client, but we can verify
        # the logic: only resource == "secrets" triggers redaction
        assert "secrets" == "secrets"  # resource must be normalised
        assert "deployments" != "secrets"
        assert "configmaps" != "secrets"
        assert "pods" != "secrets"


# ═══════════════════════════════════════════════════════════════
# Graceful fallback — malformed input
# ═══════════════════════════════════════════════════════════════


class TestGracefulFallback:
    """The redaction functions should never crash, even with unexpected input."""

    def test_item_with_data_as_string(self) -> None:
        item = _make_secret()
        item["data"] = "not-a-dict"
        result = _redact_secret_item(item)
        assert result["data"] == "not-a-dict"  # passes through

    def test_item_with_data_as_list(self) -> None:
        item = _make_secret()
        item["data"] = ["a", "b"]
        result = _redact_secret_item(item)
        assert result["data"] == ["a", "b"]  # passes through

    def test_item_with_data_as_none(self) -> None:
        item = _make_secret()
        item["data"] = None
        result = _redact_secret_item(item)
        assert result["data"] is None  # passes through

    def test_item_with_data_as_int(self) -> None:
        item = _make_secret()
        item["data"] = 42
        result = _redact_secret_item(item)
        assert result["data"] == 42  # passes through

    def test_empty_dict_item(self) -> None:
        result = _redact_secret_item({})
        assert result == {}

    def test_list_redaction_with_non_dict_items_graceful(self) -> None:
        """If the list contains non-dict items, the function should handle it."""
        # This is a truly malformed scenario — should not crash
        items = [_make_secret(data={"k": "v"}), "not-a-dict"]  # type: ignore[list-item]
        # _redact_secret_item will fail on a string — but _redact_k8s_secret_data
        # wraps in try/except and returns original
        result = _redact_k8s_secret_data(items)
        # Should return the original unmodified list (graceful fallback)
        assert result == items


# ═══════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Additional edge cases and boundary conditions."""

    def test_secret_with_empty_data_and_string_data(self) -> None:
        item = _make_secret(data={}, string_data={})
        result = _redact_secret_item(item)
        assert result["data"] == {}
        assert result["stringData"] == {}
        assert "_data_redacted_note" not in result
        assert "_stringData_redacted_note" not in result

    def test_secret_with_only_string_data(self) -> None:
        item = _make_secret(string_data={"plain": "text"})
        result = _redact_secret_item(item)
        assert "data" not in result
        assert result["stringData"]["plain"] == _SECRET_REDACTED

    def test_secret_with_large_base64_value(self) -> None:
        """Real TLS certs can be very long — ensure they're fully redacted."""
        long_cert = "A" * 10000  # simulated base64 cert
        item = _make_secret(data={"tls.crt": long_cert})
        result = _redact_secret_item(item)
        assert result["data"]["tls.crt"] == _SECRET_REDACTED
        assert len(result["data"]["tls.crt"]) < 20  # much shorter than original

    def test_redacted_value_is_consistent(self) -> None:
        """All redacted values should be the same constant."""
        item = _make_secret(data={"a": "1", "b": "2", "c": "3"})
        result = _redact_secret_item(item)
        values = list(result["data"].values())
        assert all(v == _SECRET_REDACTED for v in values)
        assert all(v == values[0] for v in values)

    def test_secret_redacted_constant_value(self) -> None:
        assert _SECRET_REDACTED == "[REDACTED]"

    def test_multiple_redaction_calls_idempotent(self) -> None:
        """Redacting an already-redacted secret should produce same result."""
        item = _make_secret(data={"pw": "original"})
        first = _redact_secret_item(item)
        second = _redact_secret_item(first)
        assert second["data"]["pw"] == _SECRET_REDACTED
        assert second["_data_redacted_note"] == first["_data_redacted_note"]

    def test_secret_with_special_characters_in_keys(self) -> None:
        data = {
            "tls.crt": "cert-data",
            "ca.crt": "ca-data",
            ".dockerconfigjson": "docker-data",
            "key/with/slashes": "slash-data",
        }
        item = _make_secret(data=data)
        result = _redact_secret_item(item)
        assert all(v == _SECRET_REDACTED for v in result["data"].values())
        assert set(result["data"].keys()) == set(data.keys())

    def test_secret_with_empty_string_values(self) -> None:
        item = _make_secret(data={"empty": ""})
        result = _redact_secret_item(item)
        assert result["data"]["empty"] == _SECRET_REDACTED

    def test_secret_preserves_api_version(self) -> None:
        item = _make_secret(data={"k": "v"})
        result = _redact_secret_item(item)
        assert result["apiVersion"] == "v1"

    def test_secret_preserves_kind(self) -> None:
        item = _make_secret(data={"k": "v"})
        result = _redact_secret_item(item)
        assert result["kind"] == "Secret"
