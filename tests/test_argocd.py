"""Tests for ArgoCD introspection tools — list, status, history, diff, managed resources."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear caches before each test."""
    from vaig.tools.gke._cache import clear_discovery_cache
    from vaig.tools.gke.argocd import _argocd_namespace_cache, _argocd_ns_cache, _crd_exists_cache
    clear_discovery_cache()
    _argocd_namespace_cache.clear()
    _crd_exists_cache.clear()
    _argocd_ns_cache.clear()


# ── Test data helpers ────────────────────────────────────────


def _make_argocd_app(
    name: str = "my-app",
    project: str = "default",
    sync_status: str = "Synced",
    health_status: str = "Healthy",
    repo: str = "https://github.com/org/repo.git",
    path: str = "manifests/",
    target_revision: str = "HEAD",
    dest_server: str = "https://kubernetes.default.svc",
    dest_namespace: str = "production",
    history: list | None = None,
    resources: list | None = None,
    conditions: list | None = None,
    sync_policy: dict | None = None,
    operation_state: dict | None = None,
) -> dict:
    """Create a realistic ArgoCD Application CRD dict."""
    app: dict = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Application",
        "metadata": {
            "name": name,
            "namespace": "argocd",
        },
        "spec": {
            "project": project,
            "source": {
                "repoURL": repo,
                "path": path,
                "targetRevision": target_revision,
            },
            "destination": {
                "server": dest_server,
                "namespace": dest_namespace,
            },
        },
        "status": {
            "sync": {
                "status": sync_status,
                "revision": "abc123def456",
            },
            "health": {
                "status": health_status,
            },
            "resources": resources or [],
            "history": history or [],
        },
    }

    if sync_policy is not None:
        app["spec"]["syncPolicy"] = sync_policy
    if conditions is not None:
        app["status"]["conditions"] = conditions
    if operation_state is not None:
        app["status"]["operationState"] = operation_state

    return app


def _make_history_entry(
    entry_id: int = 1,
    revision: str = "abc123",
    deployed_at: str = "2025-06-01T10:00:00Z",
    repo: str = "https://github.com/org/repo.git",
) -> dict:
    return {
        "id": entry_id,
        "revision": revision,
        "deployedAt": deployed_at,
        "source": {
            "repoURL": repo,
            "path": "manifests/",
            "targetRevision": "HEAD",
        },
    }


def _make_resource(
    kind: str = "Deployment",
    name: str = "my-deploy",
    namespace: str = "production",
    group: str = "apps",
    sync_status: str = "Synced",
    health_status: str = "Healthy",
    requires_pruning: bool = False,
    hook: bool = False,
) -> dict:
    return {
        "group": group,
        "kind": kind,
        "name": name,
        "namespace": namespace,
        "status": sync_status,
        "health": {"status": health_status},
        "requiresPruning": requires_pruning,
        "hook": hook,
    }


# ── argocd_list_applications ────────────────────────────────


class TestArgocdListApplications:
    """Tests for argocd_list_applications."""

    def test_argocd_list_applications(self) -> None:
        """Multiple apps found — returns table with all apps."""
        from vaig.tools.gke.argocd import argocd_list_applications

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_argocd_app(name="frontend", health_status="Healthy", sync_status="Synced"),
                _make_argocd_app(name="backend", health_status="Degraded", sync_status="OutOfSync"),
                _make_argocd_app(name="worker", health_status="Healthy", sync_status="Synced"),
            ]
        }

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_list_applications(namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "worker" in result.output
        assert "Total applications: 3" in result.output
        assert "OutOfSync" in result.output
        assert "Degraded" in result.output

    def test_argocd_list_applications_empty(self) -> None:
        """No apps found — returns appropriate message."""
        from vaig.tools.gke.argocd import argocd_list_applications

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_list_applications(namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "No ArgoCD applications found" in result.output
        assert "Total applications: 0" in result.output


# ── argocd_app_status ────────────────────────────────────────


class TestArgocdAppStatus:
    """Tests for argocd_app_status."""

    def test_argocd_app_status(self) -> None:
        """Detailed status of a specific app."""
        from vaig.tools.gke.argocd import argocd_app_status

        app = _make_argocd_app(
            name="frontend",
            project="web",
            sync_status="Synced",
            health_status="Healthy",
            repo="https://github.com/org/frontend.git",
            path="k8s/",
            target_revision="v1.2.3",
            dest_namespace="web-prod",
            sync_policy={"automated": {"prune": True, "selfHeal": True}},
            operation_state={
                "phase": "Succeeded",
                "message": "successfully synced",
                "startedAt": "2025-06-01T10:00:00Z",
                "finishedAt": "2025-06-01T10:01:00Z",
            },
            conditions=[
                {
                    "type": "SyncError",
                    "message": "something happened",
                    "lastTransitionTime": "2025-06-01T10:00:00Z",
                }
            ],
        )

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_status(app_name="frontend", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "frontend" in result.output
        assert "web" in result.output
        assert "Synced" in result.output
        assert "Healthy" in result.output
        assert "https://github.com/org/frontend.git" in result.output
        assert "v1.2.3" in result.output
        assert "web-prod" in result.output
        assert "Automated" in result.output
        assert "prune=True" in result.output
        assert "selfHeal=True" in result.output
        assert "Succeeded" in result.output
        assert "SyncError" in result.output

    def test_argocd_app_status_not_found(self) -> None:
        """App doesn't exist — returns error."""
        from vaig.tools.gke.argocd import argocd_app_status

        mock_custom_api = MagicMock()

        # Simulate 404
        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        mock_custom_api.get_namespaced_custom_object.side_effect = exc_class()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            result = argocd_app_status(app_name="ghost-app", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is True
        assert "not found" in result.output


# ── argocd_app_history ───────────────────────────────────────


class TestArgocdAppHistory:
    """Tests for argocd_app_history."""

    def test_argocd_app_history(self) -> None:
        """Multiple history entries — sorted most recent first."""
        from vaig.tools.gke.argocd import argocd_app_history

        history = [
            _make_history_entry(entry_id=1, revision="aaa111", deployed_at="2025-05-01T10:00:00Z"),
            _make_history_entry(entry_id=2, revision="bbb222", deployed_at="2025-05-15T10:00:00Z"),
            _make_history_entry(entry_id=3, revision="ccc333", deployed_at="2025-06-01T10:00:00Z"),
        ]
        app = _make_argocd_app(name="frontend", history=history)

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_history(app_name="frontend", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "aaa111" in result.output
        assert "bbb222" in result.output
        assert "ccc333" in result.output
        assert "Total deployments: 3" in result.output

        # Verify most recent first (id=3 should appear before id=1)
        pos_3 = result.output.index("ccc333")
        pos_1 = result.output.index("aaa111")
        assert pos_3 < pos_1, "Most recent entry should appear first"

    def test_argocd_app_history_no_history(self) -> None:
        """App with empty history."""
        from vaig.tools.gke.argocd import argocd_app_history

        app = _make_argocd_app(name="new-app", history=[])

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_history(app_name="new-app", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "No deployment history found" in result.output
        assert "Total deployments: 0" in result.output


# ── argocd_app_diff ──────────────────────────────────────────


class TestArgocdAppDiff:
    """Tests for argocd_app_diff."""

    def test_argocd_app_diff(self) -> None:
        """Out-of-sync resources are reported."""
        from vaig.tools.gke.argocd import argocd_app_diff

        resources = [
            _make_resource(kind="Deployment", name="web", sync_status="Synced", health_status="Healthy"),
            _make_resource(kind="Service", name="web-svc", sync_status="OutOfSync", health_status="Healthy"),
            _make_resource(kind="ConfigMap", name="config", sync_status="Synced", health_status="Degraded"),
            _make_resource(kind="Ingress", name="ingress", sync_status="Synced", health_status="Healthy"),
        ]
        app = _make_argocd_app(name="frontend", resources=resources)

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_diff(app_name="frontend", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        # OutOfSync service should appear
        assert "web-svc" in result.output
        assert "OutOfSync" in result.output
        # Degraded configmap should appear
        assert "config" in result.output
        assert "Degraded" in result.output
        # Healthy+Synced deployment should NOT appear in the diff
        # (it's synced and healthy) — the diff only shows out-of-sync
        assert "Out-of-sync resources: 2" in result.output

    def test_argocd_app_diff_all_synced(self) -> None:
        """Everything in sync — returns success message."""
        from vaig.tools.gke.argocd import argocd_app_diff

        resources = [
            _make_resource(kind="Deployment", name="web", sync_status="Synced", health_status="Healthy"),
            _make_resource(kind="Service", name="web-svc", sync_status="Synced", health_status="Healthy"),
        ]
        app = _make_argocd_app(name="frontend", resources=resources)

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_diff(app_name="frontend", namespace="argocd", _custom_api=mock_custom_api)

        assert result.error is False
        assert "All resources are in sync and healthy" in result.output


# ── argocd_app_managed_resources ─────────────────────────────


class TestArgocdAppManagedResources:
    """Tests for argocd_app_managed_resources."""

    def test_argocd_app_managed_resources(self) -> None:
        """Multiple resource types — grouped by kind."""
        from vaig.tools.gke.argocd import argocd_app_managed_resources

        resources = [
            _make_resource(kind="Deployment", name="web", group="apps"),
            _make_resource(kind="Deployment", name="worker", group="apps"),
            _make_resource(kind="Service", name="web-svc", group=""),
            _make_resource(kind="ConfigMap", name="config", group="", requires_pruning=True),
            _make_resource(kind="Ingress", name="web-ingress", group="networking.k8s.io"),
        ]
        app = _make_argocd_app(name="frontend", resources=resources)

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object.return_value = app

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            result = argocd_app_managed_resources(
                app_name="frontend", namespace="argocd", _custom_api=mock_custom_api,
            )

        assert result.error is False
        assert "web" in result.output
        assert "worker" in result.output
        assert "web-svc" in result.output
        assert "config" in result.output
        assert "web-ingress" in result.output
        assert "Total managed resources: 5" in result.output
        # Grouped by kind — check kind headers
        assert "Deployment (2)" in result.output
        assert "Service (1)" in result.output
        assert "ConfigMap (1)" in result.output
        # Pruning flag
        assert "Yes" in result.output  # config requires pruning


# ── kubernetes not available ─────────────────────────────────


class TestK8sNotAvailable:
    """Test graceful handling when kubernetes SDK is not installed."""

    def test_k8s_not_available(self) -> None:
        """All functions return error ToolResult when k8s is unavailable."""
        from vaig.tools.gke.argocd import (
            argocd_app_diff,
            argocd_app_history,
            argocd_app_managed_resources,
            argocd_app_status,
            argocd_list_applications,
        )

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result_list = argocd_list_applications(namespace="argocd")
            assert result_list.error is True
            assert "kubernetes" in result_list.output.lower()

            result_status = argocd_app_status(app_name="app", namespace="argocd")
            assert result_status.error is True

            result_history = argocd_app_history(app_name="app", namespace="argocd")
            assert result_history.error is True

            result_diff = argocd_app_diff(app_name="app", namespace="argocd")
            assert result_diff.error is True

            result_managed = argocd_app_managed_resources(app_name="app", namespace="argocd")
            assert result_managed.error is True


# ── _create_argocd_client ────────────────────────────────────


class TestCreateArgocdClient:
    """Tests for _create_argocd_client in _clients.py."""

    @patch("vaig.tools.gke._clients.k8s_client")
    def test_create_argocd_client_cluster_mode(self, mock_k8s_client: MagicMock) -> None:
        """Same-cluster mode returns ('cluster', CustomObjectsApi)."""
        from vaig.tools.gke._clients import _create_argocd_client

        mock_api_client = MagicMock()
        mock_custom_api = MagicMock()
        mock_k8s_client.ApiClient.return_value = mock_api_client
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            mode, client = _create_argocd_client()

        assert mode == "cluster"
        assert client is mock_custom_api
        mock_k8s_client.ApiClient.assert_called_once()
        mock_k8s_client.CustomObjectsApi.assert_called_once_with(mock_api_client)

    def test_create_argocd_client_api_mode_stub(self) -> None:
        """API mode raises NotImplementedError."""
        from vaig.tools.gke._clients import _create_argocd_client

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             pytest.raises(NotImplementedError, match="REST API mode"):
            _create_argocd_client(server="https://argocd.example.com", token="my-token")

    def test_create_argocd_client_context_mode_stub(self) -> None:
        """Context mode raises NotImplementedError."""
        from vaig.tools.gke._clients import _create_argocd_client

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             pytest.raises(NotImplementedError, match="separate-context mode"):
            _create_argocd_client(context="argocd-context")

    def test_create_argocd_client_k8s_unavailable(self) -> None:
        """Raises RuntimeError when kubernetes SDK is not available."""
        from vaig.tools.gke._clients import _create_argocd_client

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"), \
             pytest.raises(RuntimeError, match="kubernetes"):
            _create_argocd_client()


# ── Cache behavior ───────────────────────────────────────────


class TestCacheBehavior:
    """Tests for caching in ArgoCD tools."""

    def test_cache_behavior(self) -> None:
        """Second call returns cached result without hitting the API."""
        from vaig.tools.gke.argocd import argocd_list_applications

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {
            "items": [_make_argocd_app(name="cached-app")]
        }

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            # First call — hits the API
            result1 = argocd_list_applications(namespace="argocd", _custom_api=mock_custom_api)
            assert result1.error is False
            assert "cached-app" in result1.output

            # Second call — should use cache
            result2 = argocd_list_applications(namespace="argocd", _custom_api=mock_custom_api)
            assert result2.error is False
            assert result2.output == result1.output

            # API should have been called only once
            assert mock_custom_api.list_namespaced_custom_object.call_count == 1

    def test_cache_different_namespaces(self) -> None:
        """Different namespaces use different cache keys."""
        from vaig.tools.gke.argocd import argocd_list_applications

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {
            "items": [_make_argocd_app(name="ns-app")]
        }

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True):
            argocd_list_applications(namespace="argocd", _custom_api=mock_custom_api)
            argocd_list_applications(namespace="argocd-staging", _custom_api=mock_custom_api)

            # Should have been called twice (different cache keys)
            assert mock_custom_api.list_namespaced_custom_object.call_count == 2


# ── _check_crd_exists ────────────────────────────────────────


class TestCheckCrdExists:
    """Tests for _check_crd_exists helper."""

    def test_crd_found_returns_true(self) -> None:
        """CRD exists in cluster — returns True."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config"):
            result = _check_crd_exists("applications.argoproj.io")

        assert result is True
        mock_ext_api.read_custom_resource_definition.assert_called_once_with("applications.argoproj.io")

    def test_crd_not_found_returns_false(self) -> None:
        """CRD doesn't exist (404) — returns False."""
        from vaig.tools.gke.argocd import _check_crd_exists

        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.side_effect = exc_class()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd.k8s_exceptions") as mock_k8s_exc, \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config"):
            mock_k8s_exc.ApiException = exc_class
            result = _check_crd_exists("applications.argoproj.io")

        assert result is False

    def test_crd_rbac_forbidden_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        """403 Forbidden (RBAC missing) — returns False and emits a warning log."""
        import logging

        from vaig.tools.gke.argocd import _check_crd_exists

        exc_class = type("ApiException", (Exception,), {"status": 403, "reason": "Forbidden"})
        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.side_effect = exc_class()

        with caplog.at_level(logging.WARNING, logger="vaig.tools.gke.argocd"), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd.k8s_exceptions") as mock_k8s_exc, \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config"):
            mock_k8s_exc.ApiException = exc_class
            result = _check_crd_exists("applications.argoproj.io")

        assert result is False
        assert any(
            "403" in record.message or "Forbidden" in record.message or "RBAC" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ), "Expected a WARNING log mentioning 403/Forbidden/RBAC when CRD check is denied"

    def test_crd_unexpected_exception_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        """Any unexpected exception — returns False and emits a warning log (never propagates)."""
        import logging

        from vaig.tools.gke.argocd import _check_crd_exists

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.side_effect = RuntimeError("network timeout")

        with caplog.at_level(logging.WARNING, logger="vaig.tools.gke.argocd"), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config"):
            result = _check_crd_exists("applications.argoproj.io")

        assert result is False
        assert any(
            "network timeout" in record.message or "Unexpected" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ), "Expected a WARNING log for unexpected exception in CRD check"

    def test_crd_k8s_unavailable_returns_false(self) -> None:
        """kubernetes SDK not installed — returns False without raising."""
        from vaig.tools.gke.argocd import _check_crd_exists

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", False):
            result = _check_crd_exists("applications.argoproj.io")

        assert result is False

    def test_crd_caching_avoids_repeated_api_calls(self) -> None:
        """Second call uses cache — API is called exactly once."""
        from vaig.tools.gke.argocd import _check_crd_exists

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config"):
            result1 = _check_crd_exists("applications.argoproj.io")
            result2 = _check_crd_exists("applications.argoproj.io")

        assert result1 is True
        assert result2 is True
        assert mock_ext_api.read_custom_resource_definition.call_count == 1


# ── _discover_argocd_namespace with CRD pre-check ─────────────


class TestDiscoverArgoCdNamespaceWithCrdPrecheck:
    """Tests for _discover_argocd_namespace() CRD pre-check behaviour."""

    def test_crd_absent_skips_namespace_scanning(self) -> None:
        """When CRD check returns False, namespace probing is never attempted."""
        from vaig.tools.gke.argocd import _discover_argocd_namespace

        mock_custom_api = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False):
            result = _discover_argocd_namespace(mock_custom_api)

        assert result is None
        mock_custom_api.list_namespaced_custom_object.assert_not_called()
        mock_custom_api.list_cluster_custom_object.assert_not_called()

    def test_crd_present_proceeds_to_namespace_scan(self) -> None:
        """When CRD check returns True, namespace probing runs and finds the namespace."""
        from vaig.tools.gke.argocd import _discover_argocd_namespace

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {
            "items": [_make_argocd_app(name="my-app")]
        }

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True):
            result = _discover_argocd_namespace(mock_custom_api)

        assert result == "argocd"  # first common namespace checked

    def test_hub_spoke_crd_present_namespace_not_found(self) -> None:
        """Hub-spoke: CRD exists but no ArgoCD namespace found → returns None."""
        from vaig.tools.gke.argocd import _discover_argocd_namespace

        exc_class = type("ApiException", (Exception,), {"status": 404, "reason": "Not Found"})
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.side_effect = exc_class()
        mock_custom_api.list_cluster_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True), \
             patch("vaig.tools.gke.argocd.k8s_exceptions") as mock_k8s_exc:
            mock_k8s_exc.ApiException = exc_class
            result = _discover_argocd_namespace(mock_custom_api)

        assert result is None

    def test_crd_absent_result_is_cached(self) -> None:
        """When CRD absent, None is cached so second call skips the CRD check too."""
        from vaig.tools.gke.argocd import _discover_argocd_namespace

        mock_custom_api = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False) as mock_crd:
            _discover_argocd_namespace(mock_custom_api)
            _discover_argocd_namespace(mock_custom_api)  # second call

        # _check_crd_exists called only once — second call hits namespace cache
        assert mock_crd.call_count == 1


# ── detect_argocd ────────────────────────────────────────────


def _make_deployment_meta(annotations: dict | None = None) -> MagicMock:
    """Create a mock deployment object with the given annotations."""
    dep = MagicMock()
    dep.metadata.annotations = annotations or {}
    return dep


class TestDetectArgocd:
    """Tests for detect_argocd — three-phase namespace-scoped detection."""

    def test_crd_found_and_annotations_present_returns_true(self) -> None:
        """Returns True when CRD exists AND annotations found in namespace."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [
            _make_deployment_meta({"argocd.argoproj.io/tracking-id": "my-app:argocd/my-app"}),
        ]

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="production")

        assert result is True

    def test_crd_found_but_no_annotations_returns_false(self) -> None:
        """Returns False when CRD exists cluster-wide but namespace has no ArgoCD-managed resources.

        THIS IS THE KEY FIX: previously this would return True (false positive).
        CRD existing cluster-wide is NOT sufficient — the namespace must have ArgoCD annotations.
        """
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [
            _make_deployment_meta({"app": "foo", "version": "1.0"}),
            _make_deployment_meta({"team": "backend"}),
        ]

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="production")

        assert result is False

    def test_crd_not_found_no_annotations_returns_false(self) -> None:
        """Returns False when CRD absent and no matching annotations found."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [
            _make_deployment_meta({"app": "foo", "version": "1.0"}),
        ]

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="production")

        assert result is False

    def test_crd_403_rbac_but_annotations_present_returns_true(self) -> None:
        """Returns True when CRD probe fails (RBAC 403) but annotations are found.

        This is the key scenario: RBAC-restricted clusters deny CRD access but
        deployments carry argocd.argoproj.io/tracking-id annotations that confirm
        ArgoCD is managing resources.
        """
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [
            _make_deployment_meta({"app": "foo"}),
            _make_deployment_meta({"argocd.argoproj.io/tracking-id": "my-app:argocd/my-app"}),
        ]

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="production")

        assert result is True

    def test_crd_403_rbac_managed_by_annotation_returns_true(self) -> None:
        """Returns True when argocd.argoproj.io/managed-by annotation is found."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [
            _make_deployment_meta({"argocd.argoproj.io/managed-by": "my-argocd"}),
        ]

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="staging")

        assert result is True

    def test_crd_not_found_no_annotations_returns_false_rbac_path(self) -> None:
        """Returns False when CRD absent and no annotations after RBAC probe."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = []

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="default")

        assert result is False

    def test_k8s_unavailable_returns_false(self) -> None:
        """Returns False when the kubernetes SDK is not available."""
        from vaig.tools.gke.argocd import detect_argocd

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", False):
            result = detect_argocd(namespace="production")

        assert result is False

    def test_annotation_scan_exception_returns_false_gracefully(self) -> None:
        """Returns False gracefully when annotation scan throws an unexpected error."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.side_effect = RuntimeError("Connection refused")

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argocd(namespace="production")

        assert result is False

    def test_uses_default_namespace_when_empty(self) -> None:
        """Uses 'default' namespace when namespace param is empty string."""
        from vaig.tools.gke.argocd import detect_argocd

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = []

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=False), \
             patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            detect_argocd(namespace="")

        mock_apps_api.list_namespaced_deployment.assert_called_once_with(
            namespace="default",
            limit=50,
        )

    def test_namespace_cache_hit_avoids_api_calls(self) -> None:
        """Cached namespace result is returned without making any API calls."""
        from vaig.tools.gke.argocd import _argocd_ns_cache, detect_argocd

        _argocd_ns_cache["production"] = True

        with patch("vaig.tools.gke.argocd._check_crd_exists") as mock_crd, \
             patch("vaig.tools.gke.argocd._scan_namespace_for_argocd_annotations") as mock_scan:
            result = detect_argocd(namespace="production")

        assert result is True
        mock_crd.assert_not_called()
        mock_scan.assert_not_called()

    def test_different_namespaces_get_independent_results(self) -> None:
        """Namespace A can be managed by ArgoCD while namespace B is not."""
        from vaig.tools.gke.argocd import detect_argocd

        def annotation_scan_side_effect(namespace: str, _api_client: object) -> bool:
            # Only 'argocd-managed' namespace has ArgoCD annotations
            return namespace == "argocd-managed"

        with patch("vaig.tools.gke.argocd._check_crd_exists", return_value=True), \
             patch(
                 "vaig.tools.gke.argocd._scan_namespace_for_argocd_annotations",
                 side_effect=annotation_scan_side_effect,
             ):
            result_managed = detect_argocd(namespace="argocd-managed")
            result_unmanaged = detect_argocd(namespace="no-argocd-here")

        assert result_managed is True
        assert result_unmanaged is False
