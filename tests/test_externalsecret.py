"""Tests for ExternalSecret support in kubectl_get and kubectl_describe (Bug #1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear caches before each test."""
    from vaig.tools.gke._cache import clear_discovery_cache
    clear_discovery_cache()


# ── Test data helpers ────────────────────────────────────────


def _make_externalsecret(
    name: str = "my-es",
    namespace: str = "default",
    store: str = "my-store",
    refresh_interval: str = "1h",
    ready: bool = True,
) -> dict:
    """Create a realistic ExternalSecret CRD dict as returned by CustomObjectsApi."""
    return {
        "apiVersion": "external-secrets.io/v1beta1",
        "kind": "ExternalSecret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": None,
        },
        "spec": {
            "secretStoreRef": {"name": store, "kind": "SecretStore"},
            "refreshInterval": refresh_interval,
            "target": {"name": name, "creationPolicy": "Owner"},
            "data": [{"secretKey": "api-key", "remoteRef": {"key": "secret/api-key"}}],
        },
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True" if ready else "False",
                    "reason": "SecretSynced" if ready else "SecretSyncedError",
                    "message": "Secret synced" if ready else "Sync failed",
                    "lastTransitionTime": "2025-01-01T00:00:00Z",
                }
            ]
        },
    }


def _make_gke_config() -> MagicMock:
    """Minimal GKEConfig mock."""
    cfg = MagicMock()
    cfg.default_namespace = "default"
    return cfg


def _make_k8s_clients(custom_api: MagicMock) -> tuple:
    """Return a 4-tuple that mirrors _create_k8s_clients output."""
    core_v1 = MagicMock()
    apps_v1 = MagicMock()
    api_client = MagicMock()
    return (core_v1, apps_v1, custom_api, api_client)


# ── kubectl_get — externalsecrets ────────────────────────────


class TestKubectlGetExternalSecrets:
    """kubectl_get with resource='externalsecret' uses CustomObjectsApi."""

    def test_list_externalsecrets_returns_table(self) -> None:
        """kubectl_get('externalsecret') returns formatted table with ES data."""
        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_externalsecret(name="es-db", store="vault-store", ready=True),
                _make_externalsecret(name="es-api", store="aws-store", ready=False),
            ],
            "metadata": {},
        }

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "externalsecret",
                gke_config=_make_gke_config(),
                namespace="default",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"
        assert "es-db" in result.output
        assert "es-api" in result.output
        assert "vault-store" in result.output
        assert "aws-store" in result.output

        # Verify the right CRD API was called (no label/field selector kwargs when None)
        custom_api.list_namespaced_custom_object.assert_called_once_with(
            group="external-secrets.io",
            version="v1beta1",
            plural="externalsecrets",
            namespace="default",
        )

    def test_list_externalsecrets_alias_es(self) -> None:
        """kubectl_get('es') is an alias for 'externalsecrets'."""
        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [_make_externalsecret(name="es-cache", store="gcp-store")],
            "metadata": {},
        }

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "es",  # alias
                gke_config=_make_gke_config(),
                namespace="staging",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"
        assert "es-cache" in result.output
        custom_api.list_namespaced_custom_object.assert_called_once()

    def test_list_externalsecrets_empty_namespace(self) -> None:
        """kubectl_get with no ES objects returns 'No resources found.'"""
        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [],
            "metadata": {},
        }

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "externalsecrets",
                gke_config=_make_gke_config(),
                namespace="prod",
            )

        assert result.error is False
        assert "No resources found" in result.output

    def test_list_externalsecrets_ready_and_notready_status(self) -> None:
        """Ready/NotReady status is reflected in output."""
        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_externalsecret(name="es-ok", ready=True),
                _make_externalsecret(name="es-broken", ready=False),
            ],
            "metadata": {},
        }

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get("externalsecret", gke_config=_make_gke_config())

        assert result.error is False
        assert "Ready" in result.output
        assert "NotReady" in result.output


# ── kubectl_describe — externalsecret ────────────────────────


class TestKubectlDescribeExternalSecret:
    """kubectl_describe with resource='externalsecret' uses CustomObjectsApi.get."""

    def test_describe_externalsecret_returns_spec_and_status(self) -> None:
        """kubectl_describe('externalsecret', name) returns YAML-dumped spec/status."""
        from vaig.tools.gke.kubectl import kubectl_describe

        raw = _make_externalsecret(name="my-es", namespace="default", store="vault-store")
        custom_api = MagicMock()
        custom_api.get_namespaced_custom_object.return_value = raw

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_describe(
                "externalsecret",
                "my-es",
                gke_config=_make_gke_config(),
                namespace="default",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"
        assert "my-es" in result.output
        assert "Spec:" in result.output
        assert "vault-store" in result.output
        assert "Status:" in result.output
        assert "Events:       <not available for custom resources>" in result.output

        # Verify the right CRD API was called with correct params
        custom_api.get_namespaced_custom_object.assert_called_once_with(
            group="external-secrets.io",
            version="v1beta1",
            plural="externalsecrets",
            namespace="default",
            name="my-es",
        )

    def test_describe_externalsecret_alias(self) -> None:
        """kubectl_describe('es', name) resolves alias and works correctly."""
        from vaig.tools.gke.kubectl import kubectl_describe

        raw = _make_externalsecret(name="es-db", namespace="prod", store="aws-sm")
        custom_api = MagicMock()
        custom_api.get_namespaced_custom_object.return_value = raw

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_describe(
                "es",  # alias
                "es-db",
                gke_config=_make_gke_config(),
                namespace="prod",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"
        assert "es-db" in result.output
        custom_api.get_namespaced_custom_object.assert_called_once()

    def test_describe_externalsecret_not_found_raises_404(self) -> None:
        """kubectl_describe returns error when ES is not found (404)."""
        from kubernetes.client.exceptions import ApiException  # noqa: PLC0415

        from vaig.tools.gke.kubectl import kubectl_describe  # noqa: PLC0415

        custom_api = MagicMock()
        exc = ApiException(status=404, reason="Not Found")
        custom_api.get_namespaced_custom_object.side_effect = exc

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_describe(
                "externalsecret",
                "nonexistent-es",
                gke_config=_make_gke_config(),
                namespace="default",
            )

        assert result.error is True
        assert "not found" in result.output.lower()


# ── kubectl_get ExternalSecrets — json / yaml output formats ─


class TestKubectlGetExternalSecretsOutputFormats:
    """kubectl_get with output_format='json'/'yaml' serialises dict-backed _DictItems correctly."""

    def _make_custom_api_with_two_secrets(self) -> MagicMock:
        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_externalsecret(name="es-db", store="vault-store", ready=True),
                _make_externalsecret(name="es-api", store="aws-store", ready=False),
            ],
            "metadata": {},
        }
        return custom_api

    def test_json_output_contains_both_resources(self) -> None:
        """output_format='json' serialises dict-backed ExternalSecrets to valid JSON."""
        import json

        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = self._make_custom_api_with_two_secrets()

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "externalsecret",
                gke_config=_make_gke_config(),
                namespace="default",
                output="json",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"

        # Must be valid JSON
        parsed = json.loads(result.output)
        assert isinstance(parsed, list), f"Expected JSON list, got {type(parsed)}"
        assert len(parsed) == 2

        names = {item["metadata"]["name"] for item in parsed}
        assert names == {"es-db", "es-api"}, f"Unexpected names: {names}"

        stores = {item["spec"]["secretStoreRef"]["name"] for item in parsed}
        assert stores == {"vault-store", "aws-store"}, f"Unexpected stores: {stores}"

        # Verify the full structure is preserved (not just top-level keys)
        for item in parsed:
            assert "apiVersion" in item
            assert "kind" in item
            assert item["kind"] == "ExternalSecret"
            assert "spec" in item
            assert "status" in item

    def test_yaml_output_contains_both_resources(self) -> None:
        """output_format='yaml' serialises dict-backed ExternalSecrets to valid YAML."""
        import yaml

        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = self._make_custom_api_with_two_secrets()

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "externalsecret",
                gke_config=_make_gke_config(),
                namespace="default",
                output="yaml",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"

        # Must be valid YAML (dump_all produces multi-doc stream)
        docs = list(yaml.safe_load_all(result.output))
        # dump_all may produce a list-of-dicts or individual docs depending on path
        # Normalise: if we get a single list, unpack it
        if len(docs) == 1 and isinstance(docs[0], list):
            docs = docs[0]

        assert len(docs) == 2, f"Expected 2 YAML documents, got {len(docs)}: {docs}"

        names = {doc["metadata"]["name"] for doc in docs}
        assert names == {"es-db", "es-api"}, f"Unexpected names: {names}"

    def test_json_output_preserves_status_conditions(self) -> None:
        """JSON output must include the status.conditions array from the dict-backed item."""
        import json

        from vaig.tools.gke.kubectl import kubectl_get

        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [_make_externalsecret(name="es-one", store="gcp-sm", ready=True)],
            "metadata": {},
        }

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            mock_create.return_value = _make_k8s_clients(custom_api)
            result = kubectl_get(
                "externalsecret",
                gke_config=_make_gke_config(),
                namespace="default",
                output="json",
            )

        assert result.error is False, f"Expected no error, got: {result.output}"
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        item = parsed[0]
        conditions = item["status"]["conditions"]
        assert isinstance(conditions, list)
        assert len(conditions) == 1
        assert conditions[0]["type"] == "Ready"
        assert conditions[0]["status"] == "True"
