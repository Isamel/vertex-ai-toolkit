"""Tests for Argo Rollouts introspection tools — rollouts, analysis runs, analysis templates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear all detection caches before each test."""
    from vaig.tools.gke.argo_rollouts import _rollouts_ns_cache
    from vaig.tools.gke.argocd import _crd_exists_cache
    _crd_exists_cache.clear()
    _rollouts_ns_cache.clear()


# ── Test data helpers ────────────────────────────────────────


def _make_rollout(
    name: str = "my-rollout",
    namespace: str = "production",
    phase: str = "Healthy",
    replicas: int = 3,
    ready_replicas: int = 3,
    available_replicas: int = 3,
    updated_replicas: int = 3,
    strategy: str = "canary",
    step_index: int = 2,
    canary_weight: int = 20,
    conditions: list | None = None,
) -> dict:
    """Create a realistic Argo Rollout CRD dict."""
    if strategy == "canary":
        strategy_spec: dict = {
            "canary": {
                "steps": [
                    {"setWeight": 20},
                    {"pause": {"duration": "1m"}},
                    {"setWeight": 100},
                ]
            }
        }
        canary_status: dict = {
            "currentStepIndex": step_index,
            "weights": {
                "canary": {"weight": canary_weight},
                "stable": {"weight": 100 - canary_weight},
            },
        }
        status_extra: dict = {"canary": canary_status}
    else:
        strategy_spec = {
            "blueGreen": {
                "activeService": "my-rollout-active",
                "previewService": "my-rollout-preview",
            }
        }
        status_extra = {
            "blueGreen": {
                "activeRS": "my-rollout-abc123",
                "previewRS": "my-rollout-def456",
            }
        }

    rollout: dict = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Rollout",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "replicas": replicas,
            "strategy": strategy_spec,
        },
        "status": {
            "phase": phase,
            "readyReplicas": ready_replicas,
            "availableReplicas": available_replicas,
            "updatedReplicas": updated_replicas,
            **status_extra,
        },
    }
    if conditions is not None:
        rollout["status"]["conditions"] = conditions
    return rollout


def _make_analysisrun(
    name: str = "my-rollout-analysis-1",
    namespace: str = "production",
    phase: str = "Successful",
    message: str = "",
    metrics: list | None = None,
) -> dict:
    """Create a realistic AnalysisRun CRD dict."""
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "AnalysisRun",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {},
        "status": {
            "phase": phase,
            "message": message,
            "metricResults": metrics or [
                {"name": "success-rate", "phase": phase},
            ],
        },
    }


def _make_analysistemplate(
    name: str = "success-rate",
    namespace: str = "production",
    metrics: list | None = None,
) -> dict:
    """Create a realistic AnalysisTemplate CRD dict."""
    if metrics is None:
        metrics = [
            {
                "name": "success-rate",
                "provider": {
                    "prometheus": {
                        "address": "http://prometheus:9090",
                        "query": "sum(rate(http_requests_total{status!~'5..'}[5m]))",
                    }
                },
            }
        ]
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "AnalysisTemplate",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "metrics": metrics,
        },
    }


# ── detect_argo_rollouts ─────────────────────────────────────


class TestDetectArgoRollouts:
    """Tests for detect_argo_rollouts."""

    def test_crd_found_and_annotations_present_returns_true(self) -> None:
        """Returns True when CRD exists AND namespace has Rollouts-managed deployments."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        dep = MagicMock()
        dep.metadata.annotations = {"rollout.argoproj.io/revision": "3"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="production")

        assert result is True

    def test_crd_found_but_no_annotations_returns_false(self) -> None:
        """Returns False when CRD exists but namespace has NO Rollouts-managed deployments.

        This is the KEY fix: CRD existence cluster-wide is necessary but not sufficient.
        A namespace without Rollouts annotations must return False even if the CRD is present.
        """
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        dep_no_annotations = MagicMock()
        dep_no_annotations.metadata.annotations = {"app": "my-app", "version": "1.0"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep_no_annotations]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="unmanaged-ns")

        assert result is False

    def test_detect_returns_false_when_k8s_unavailable(self) -> None:
        """Returns False when the kubernetes SDK is not available."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", False):
            result = detect_argo_rollouts()

        assert result is False

    def test_detect_returns_false_when_crd_not_found(self) -> None:
        """Returns False when the CRD returns 404 and no annotations present."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        api_exc = k8s_exc.ApiException(status=404)
        mock_ext_api.read_custom_resource_definition.side_effect = api_exc

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = []

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="production")

        assert result is False

    def test_crd_404_but_rollout_annotations_found_returns_true(self) -> None:
        """Returns True when CRD absent but rollout annotations are present.

        Key RBAC scenario: cluster denies CRD access but Deployment annotations
        like rollout.argoproj.io/revision confirm Argo Rollouts is managing resources.
        """
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.side_effect = k8s_exc.ApiException(status=403)

        dep_without = MagicMock()
        dep_without.metadata.annotations = {"app": "frontend"}

        dep_with = MagicMock()
        dep_with.metadata.annotations = {"rollout.argoproj.io/revision": "5"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep_without, dep_with]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="production")

        assert result is True

    def test_crd_not_found_desired_replicas_annotation_returns_true(self) -> None:
        """Returns True when rollout.argoproj.io/desired-replicas annotation found."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        dep = MagicMock()
        dep.metadata.annotations = {"rollout.argoproj.io/desired-replicas": "3"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._crd_exists_cache", {"rollouts.argoproj.io": False}), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="default")

        assert result is True

    def test_annotation_scan_exception_returns_false_gracefully(self) -> None:
        """Returns False gracefully when annotation scan raises an exception."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.side_effect = RuntimeError("Network error")

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._crd_exists_cache", {"rollouts.argoproj.io": False}), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="production")

        assert result is False

    def test_annotation_fallback_uses_default_namespace_when_empty(self) -> None:
        """Uses 'default' namespace for annotation scan when namespace param is empty."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = []

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argocd._crd_exists_cache", {"rollouts.argoproj.io": False}), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            detect_argo_rollouts(namespace="")

        mock_apps_api.list_namespaced_deployment.assert_called_once_with(
            namespace="default",
            limit=50,
            _request_timeout=30,
        )

    def test_namespace_cache_hit_avoids_api_calls(self) -> None:
        """Second call for same namespace uses cache and skips all API calls."""
        from vaig.tools.gke.argo_rollouts import _rollouts_ns_cache, detect_argo_rollouts

        # Pre-populate cache
        _rollouts_ns_cache[("staging", None)] = True

        mock_ext_api = MagicMock()
        mock_apps_api = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api):
            result = detect_argo_rollouts(namespace="staging")

        assert result is True
        mock_ext_api.read_custom_resource_definition.assert_not_called()
        mock_apps_api.list_namespaced_deployment.assert_not_called()

    def test_different_namespaces_get_independent_results(self) -> None:
        """Two namespaces can independently have different detection results."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        dep_managed = MagicMock()
        dep_managed.metadata.annotations = {"rollout.argoproj.io/revision": "1"}

        dep_plain = MagicMock()
        dep_plain.metadata.annotations = {"app": "frontend"}

        mock_apps_api = MagicMock()

        # CRD query returns empty (no Rollout objects), falling back to annotation scan.
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}

        def _list_by_namespace(namespace: str, limit: int, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.items = [dep_managed] if namespace == "rollouts-ns" else [dep_plain]
            return result

        mock_apps_api.list_namespaced_deployment.side_effect = _list_by_namespace

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result_managed = detect_argo_rollouts(namespace="rollouts-ns")
            result_plain = detect_argo_rollouts(namespace="plain-ns")

        assert result_managed is True
        assert result_plain is False


# ── _format_rollout ──────────────────────────────────────────


class TestFormatRollout:
    """Tests for _format_rollout formatter."""

    def test_format_canary_rollout(self) -> None:
        """Canary rollout shows step index and weight."""
        from vaig.tools.gke.argo_rollouts import _format_rollout

        rollout = _make_rollout(
            name="payment-svc",
            namespace="production",
            phase="Progressing",
            strategy="canary",
            step_index=1,
            canary_weight=20,
        )
        output = _format_rollout(rollout)

        assert "payment-svc" in output
        assert "production" in output
        assert "Progressing" in output
        assert "canary" in output
        assert "step=1" in output
        assert "20%" in output

    def test_format_bluegreen_rollout(self) -> None:
        """BlueGreen rollout shows active and preview RS."""
        from vaig.tools.gke.argo_rollouts import _format_rollout

        rollout = _make_rollout(
            name="frontend",
            namespace="staging",
            phase="Healthy",
            strategy="blueGreen",
        )
        output = _format_rollout(rollout)

        assert "frontend" in output
        assert "staging" in output
        assert "blueGreen" in output
        assert "active=" in output
        assert "preview=" in output

    def test_format_rollout_with_conditions(self) -> None:
        """Conditions are included when present."""
        from vaig.tools.gke.argo_rollouts import _format_rollout

        rollout = _make_rollout(
            conditions=[
                {"type": "Available", "status": "False", "message": "Rollout is degraded"},
            ]
        )
        output = _format_rollout(rollout)

        assert "Available=False" in output
        assert "Rollout is degraded" in output

    def test_format_rollout_missing_fields(self) -> None:
        """Handles partially empty dicts without raising."""
        from vaig.tools.gke.argo_rollouts import _format_rollout

        output = _format_rollout({})

        assert "<unknown>" in output


# ── _format_analysisrun ──────────────────────────────────────


class TestFormatAnalysisRun:
    """Tests for _format_analysisrun formatter."""

    def test_format_successful_run(self) -> None:
        """Successful run shows phase and metric names."""
        from vaig.tools.gke.argo_rollouts import _format_analysisrun

        run = _make_analysisrun(name="my-analysis-1", phase="Successful")
        output = _format_analysisrun(run)

        assert "my-analysis-1" in output
        assert "Successful" in output
        assert "success-rate" in output

    def test_format_failed_run_with_message(self) -> None:
        """Failed run includes error message."""
        from vaig.tools.gke.argo_rollouts import _format_analysisrun

        run = _make_analysisrun(phase="Failed", message="Metric 'success-rate' assessed Failed")
        output = _format_analysisrun(run)

        assert "Failed" in output
        assert "Metric 'success-rate' assessed Failed" in output

    def test_format_analysisrun_missing_fields(self) -> None:
        """Handles empty dicts gracefully."""
        from vaig.tools.gke.argo_rollouts import _format_analysisrun

        output = _format_analysisrun({})

        assert "<unknown>" in output


# ── _format_analysistemplate ─────────────────────────────────


class TestFormatAnalysisTemplate:
    """Tests for _format_analysistemplate formatter."""

    def test_format_template_with_metrics(self) -> None:
        """Template shows metric count and provider type."""
        from vaig.tools.gke.argo_rollouts import _format_analysistemplate

        template = _make_analysistemplate(name="success-rate", namespace="production")
        output = _format_analysistemplate(template)

        assert "success-rate" in output
        assert "production" in output
        assert "1 defined" in output
        assert "prometheus" in output

    def test_format_template_no_metrics(self) -> None:
        """Template with no metrics shows 0 defined."""
        from vaig.tools.gke.argo_rollouts import _format_analysistemplate

        template = _make_analysistemplate(metrics=[])
        output = _format_analysistemplate(template)

        assert "0 defined" in output


# ── kubectl_get_rollout ──────────────────────────────────────


class TestKubectlGetRollout:
    """Tests for kubectl_get_rollout."""

    def test_list_rollouts_in_namespace(self) -> None:
        """Lists multiple rollouts in a namespace."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_rollout(name="frontend", phase="Healthy"),
                _make_rollout(name="backend", phase="Degraded"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="production")

        assert result.error is False
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "Healthy" in result.output
        assert "Degraded" in result.output

    def test_get_single_rollout_by_name(self) -> None:
        """Fetches a single rollout by name."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = _make_rollout(
            name="payment-svc", phase="Progressing"
        )

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="production", name="payment-svc")

        assert result.error is False
        assert "payment-svc" in result.output
        assert "Progressing" in result.output
        mock_api.get_namespaced_custom_object.assert_called_once_with(
            group="argoproj.io",
            version="v1alpha1",
            namespace="production",
            plural="rollouts",
            name="payment-svc",
            _request_timeout=10,
        )

    def test_empty_namespace_returns_no_rollouts(self) -> None:
        """Returns empty message when no rollouts exist."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="empty-ns")

        assert result.error is False
        assert "No Rollouts found" in result.output

    def test_cluster_wide_list_when_no_namespace(self) -> None:
        """Uses cluster-wide list when namespace is empty."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {
            "items": [_make_rollout(name="global-rollout")]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="")

        assert result.error is False
        mock_api.list_cluster_custom_object.assert_called_once()

    def test_404_returns_not_found_message(self) -> None:
        """404 ApiException returns non-error not-found message."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=404)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="production", name="missing-rollout")

        assert result.error is False
        assert "not found" in result.output

    def test_403_returns_rbac_error(self) -> None:
        """403 ApiException returns RBAC error."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=403)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_rollout(namespace="production")

        assert result.error is True
        assert "RBAC" in result.output

    def test_k8s_unavailable_returns_error(self) -> None:
        """Returns error when kubernetes SDK is unavailable."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False):
            result = kubectl_get_rollout(namespace="production")

        assert result.error is True
        assert "kubernetes" in result.output.lower()


# ── kubectl_get_analysisrun ──────────────────────────────────


class TestKubectlGetAnalysisRun:
    """Tests for kubectl_get_analysisrun."""

    def test_list_analysisruns_in_namespace(self) -> None:
        """Lists multiple AnalysisRuns."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysisrun

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_analysisrun(name="run-1", phase="Successful"),
                _make_analysisrun(name="run-2", phase="Failed"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysisrun(namespace="production")

        assert result.error is False
        assert "run-1" in result.output
        assert "run-2" in result.output
        assert "Successful" in result.output
        assert "Failed" in result.output

    def test_get_single_analysisrun_by_name(self) -> None:
        """Fetches a single AnalysisRun by name."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysisrun

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = _make_analysisrun(
            name="my-run", phase="Running"
        )

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysisrun(namespace="production", name="my-run")

        assert result.error is False
        assert "my-run" in result.output
        assert "Running" in result.output

    def test_empty_result_returns_message(self) -> None:
        """Returns a message when no AnalysisRuns found."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysisrun

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysisrun(namespace="production")

        assert result.error is False
        assert "No AnalysisRuns found" in result.output

    def test_403_returns_rbac_error(self) -> None:
        """403 returns RBAC error."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_analysisrun

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=403)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysisrun(namespace="production")

        assert result.error is True
        assert "RBAC" in result.output


# ── kubectl_get_analysistemplate ─────────────────────────────


class TestKubectlGetAnalysisTemplate:
    """Tests for kubectl_get_analysistemplate."""

    def test_list_analysistemplates(self) -> None:
        """Lists AnalysisTemplates in a namespace."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysistemplate

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_analysistemplate(name="success-rate"),
                _make_analysistemplate(name="error-rate"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysistemplate(namespace="production")

        assert result.error is False
        assert "success-rate" in result.output
        assert "error-rate" in result.output

    def test_get_single_template_by_name(self) -> None:
        """Fetches a single AnalysisTemplate by name."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysistemplate

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = _make_analysistemplate(
            name="success-rate",
            metrics=[
                {
                    "name": "success-rate",
                    "provider": {"prometheus": {"query": "rate(http_requests_total[5m])"}},
                },
                {
                    "name": "latency",
                    "provider": {"prometheus": {"query": "histogram_quantile(0.99, ...)"}},
                },
            ],
        )

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysistemplate(namespace="production", name="success-rate")

        assert result.error is False
        assert "success-rate" in result.output
        assert "2 defined" in result.output

    def test_empty_result_returns_message(self) -> None:
        """Returns a message when no templates found."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_analysistemplate

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_analysistemplate(namespace="production")

        assert result.error is False
        assert "No AnalysisTemplates found" in result.output


# ── build_workload_gatherer_prompt ───────────────────────────


class TestBuildWorkloadGathererPrompt:
    """Tests for argo_rollouts_enabled param in build_workload_gatherer_prompt."""

    def test_prompt_without_argo_rollouts(self) -> None:
        """Default prompt does not include Argo Rollouts section."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="production")

        assert "Step 4d" not in prompt
        assert "kubectl_get_rollout" not in prompt

    def test_prompt_with_argo_rollouts_enabled(self) -> None:
        """Prompt includes Argo Rollouts section when enabled."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="production", argo_rollouts_enabled=True
        )

        assert "Step 4d" in prompt
        assert "kubectl_get_rollout" in prompt
        assert "kubectl_get_analysisrun" in prompt
        assert "Rollout → ReplicaSet → Pod" in prompt

    def test_namespace_substitution_in_argo_section(self) -> None:
        """<target> placeholders are replaced with the namespace in the Argo section."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="my-namespace", argo_rollouts_enabled=True
        )

        assert "<target>" not in prompt
        assert "my-namespace" in prompt


# ── Fix B+D: Transient failures not cached (argo_rollouts) ──


class TestTransientErrorsNotCachedRollouts:
    """Transient K8s API errors must NOT be written to _rollouts_ns_cache."""

    def test_transient_scan_503_not_cached(self) -> None:
        """503 on both primary and fallback paths → result not cached, returns False."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import _rollouts_ns_cache, detect_argo_rollouts

        api_exc_503 = k8s_exc.ApiException(status=503)

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.side_effect = api_exc_503

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.side_effect = api_exc_503

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception), \
             patch("time.sleep", return_value=None):
            result = detect_argo_rollouts(namespace="transient-ns")

        assert result is False
        assert ("transient-ns", None) not in _rollouts_ns_cache

    def test_transient_then_success_returns_true_and_caches(self) -> None:
        """First call fails transiently (not cached), second call succeeds and caches True."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import _rollouts_ns_cache, detect_argo_rollouts

        api_exc_503 = k8s_exc.ApiException(status=503)

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_apps_api = MagicMock()

        # First scan: both retry attempts fail with 503, annotation fallback also 503 → not cached
        # Second scan: CRD query succeeds with one Rollout item → cached True
        # _RETRY_ATTEMPTS=2 means the first detect_argo_rollouts call exhausts 2 CRD attempts.
        call_count = {"n": 0}

        def _crd_side_effect(**kwargs: object) -> dict:
            call_count["n"] += 1
            if call_count["n"] <= 2:  # First 2 calls = both retry attempts of first invocation
                raise api_exc_503
            return {"items": [{"kind": "Rollout"}]}

        mock_custom_api.list_namespaced_custom_object.side_effect = _crd_side_effect
        mock_apps_api.list_namespaced_deployment.side_effect = api_exc_503

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception), \
             patch("time.sleep", return_value=None):
            result1 = detect_argo_rollouts(namespace="retry-ns")
            # First call transient — not cached
            assert result1 is False
            assert ("retry-ns", None) not in _rollouts_ns_cache

            result2 = detect_argo_rollouts(namespace="retry-ns")

        assert result2 is True
        assert _rollouts_ns_cache.get(("retry-ns", None)) is True

    def test_definitive_404_on_annotation_scan_returns_false(self) -> None:
        """404 on deployment list is definitive — returns False (no annotation scan needed)."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import _rollouts_ns_cache, detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=404)

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.side_effect = k8s_exc.ApiException(status=404)

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="gone-ns")

        assert result is False
        # 404 is definitive — result IS cached
        assert ("gone-ns", None) in _rollouts_ns_cache
        assert _rollouts_ns_cache[("gone-ns", None)] is False


# ── Fix C: Rollout CRD query ─────────────────────────────────


class TestRolloutCRDQuery:
    """Primary detection path should query Rollout CRDs, not just deployment annotations."""

    def test_rollout_crd_found_returns_true(self) -> None:
        """list_namespaced_custom_object returning Rollout items → True."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {
            "items": [{"kind": "Rollout", "metadata": {"name": "my-rollout"}}]
        }

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="rollouts-ns")

        assert result is True
        mock_custom_api.list_namespaced_custom_object.assert_called_once_with(
            group="argoproj.io",
            version="v1alpha1",
            namespace="rollouts-ns",
            plural="rollouts",
            limit=1,
            _request_timeout=30,
        )

    def test_no_rollout_objects_falls_back_to_annotations(self) -> None:
        """Empty Rollout CRD list falls back to annotation scan."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}

        dep = MagicMock()
        dep.metadata.annotations = {"rollout.argoproj.io/revision": "2"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="annotation-fallback-ns")

        assert result is True
        mock_apps_api.list_namespaced_deployment.assert_called_once()

    def test_crd_query_404_falls_back_to_annotations(self) -> None:
        """404 on Rollout CRD query falls back to annotation scan."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=404)

        dep = MagicMock()
        dep.metadata.annotations = {"rollout.argoproj.io/desired-replicas": "3"}

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.return_value.items = [dep]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts(namespace="fallback-ns")

        assert result is True
        mock_apps_api.list_namespaced_deployment.assert_called_once()


# ── Fix D: Retry logic (argo_rollouts) ──────────────────────


class TestRetryLogicRollouts:
    """Retry helper must attempt up to _RETRY_ATTEMPTS before giving up."""

    def test_crd_query_fails_once_then_succeeds(self) -> None:
        """503 on first CRD query attempt, success on second → returns True."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.side_effect = [
            k8s_exc.ApiException(status=503),
            {"items": [{"kind": "Rollout"}]},
        ]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception), \
             patch("time.sleep", return_value=None) as mock_sleep:
            result = detect_argo_rollouts(namespace="retry-crd-ns")

        assert result is True
        assert mock_custom_api.list_namespaced_custom_object.call_count == 2
        mock_sleep.assert_called_once()

    def test_annotation_fallback_retries_once_on_transient(self) -> None:
        """503 on first annotation scan attempt, success on second → returns True."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        # CRD query returns empty (no Rollout objects), triggering annotation fallback.
        mock_custom_api = MagicMock()
        mock_custom_api.list_namespaced_custom_object.return_value = {"items": []}

        dep = MagicMock()
        dep.metadata.annotations = {"rollout.argoproj.io/revision": "1"}

        success_result = MagicMock()
        success_result.items = [dep]

        mock_apps_api = MagicMock()
        mock_apps_api.list_namespaced_deployment.side_effect = [
            k8s_exc.ApiException(status=503),
            success_result,
        ]

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.client.CustomObjectsApi", return_value=mock_custom_api), \
             patch("kubernetes.client.AppsV1Api", return_value=mock_apps_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception), \
             patch("time.sleep", return_value=None) as mock_sleep:
            result = detect_argo_rollouts(namespace="retry-annot-ns")

        assert result is True
        assert mock_apps_api.list_namespaced_deployment.call_count == 2
        mock_sleep.assert_called_once()


# ── Test data helpers (new CRDs) ─────────────────────────────


def _make_cluster_analysis_template(
    name: str = "global-success-rate",
    metrics: list | None = None,
) -> dict:
    """Create a realistic ClusterAnalysisTemplate CRD dict (cluster-scoped)."""
    if metrics is None:
        metrics = [
            {
                "name": "success-rate",
                "provider": {
                    "prometheus": {
                        "address": "http://prometheus:9090",
                        "query": "sum(rate(http_requests_total{status!~'5..'}[5m]))",
                    }
                },
            }
        ]
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "ClusterAnalysisTemplate",
        "metadata": {
            "name": name,
        },
        "spec": {
            "metrics": metrics,
        },
    }


def _make_experiment(
    name: str = "my-experiment",
    namespace: str = "production",
    phase: str = "Running",
    message: str = "",
    templates: list | None = None,
) -> dict:
    """Create a realistic Experiment CRD dict."""
    if templates is None:
        templates = [
            {"name": "baseline", "replicas": 1},
            {"name": "canary", "replicas": 1},
        ]
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Experiment",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "templates": templates,
        },
        "status": {
            "phase": phase,
            "message": message,
        },
    }


# ── kubectl_get_cluster_analysis_template ────────────────────


class TestKubectlGetClusterAnalysisTemplate:
    """Tests for kubectl_get_cluster_analysis_template."""

    def test_list_all_templates(self) -> None:
        """Lists all ClusterAnalysisTemplates cluster-wide."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {
            "items": [
                _make_cluster_analysis_template(name="global-success-rate"),
                _make_cluster_analysis_template(name="global-latency"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_cluster_analysis_template()

        assert result.error is False
        assert "global-success-rate" in result.output
        assert "global-latency" in result.output

    def test_get_single_template_by_name(self) -> None:
        """Fetches a single ClusterAnalysisTemplate by name."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        mock_api = MagicMock()
        mock_api.get_cluster_custom_object.return_value = _make_cluster_analysis_template(
            name="global-success-rate",
            metrics=[
                {
                    "name": "success-rate",
                    "provider": {"prometheus": {"query": "rate(http_requests_total[5m])"}},
                },
                {
                    "name": "latency",
                    "provider": {"datadog": {"query": "avg:trace.web.request.duration{*}"}},
                },
            ],
        )

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_cluster_analysis_template(name="global-success-rate")

        assert result.error is False
        assert "global-success-rate" in result.output
        assert "2 defined" in result.output
        assert "cluster" in result.output

    def test_empty_list_returns_message(self) -> None:
        """Returns informational message when no ClusterAnalysisTemplates exist."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_cluster_analysis_template()

        assert result.error is False
        assert "No ClusterAnalysisTemplates found" in result.output

    def test_404_by_name_returns_not_found(self) -> None:
        """404 on named get returns not-found message without error flag."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        mock_api = MagicMock()
        mock_api.get_cluster_custom_object.side_effect = k8s_exc.ApiException(status=404)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_cluster_analysis_template(name="does-not-exist")

        assert result.error is False
        assert "not found" in result.output.lower()

    def test_403_returns_rbac_error(self) -> None:
        """403 on list returns RBAC guidance with error=True."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.side_effect = k8s_exc.ApiException(status=403)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_cluster_analysis_template()

        assert result.error is True
        assert "RBAC" in result.output
        assert "clusteranalysistemplates" in result.output

    def test_k8s_unavailable_returns_error(self) -> None:
        """Returns error when kubernetes SDK is not available."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_cluster_analysis_template

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False):
            result = kubectl_get_cluster_analysis_template()

        assert result.error is True


# ── kubectl_get_experiment ───────────────────────────────────


class TestKubectlGetExperiment:
    """Tests for kubectl_get_experiment."""

    def test_list_experiments_in_namespace(self) -> None:
        """Lists Experiments in a specific namespace."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {
            "items": [
                _make_experiment(name="rollout-exp-1"),
                _make_experiment(name="rollout-exp-2", phase="Successful"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment(namespace="production")

        assert result.error is False
        assert "rollout-exp-1" in result.output
        assert "rollout-exp-2" in result.output

    def test_list_experiments_all_namespaces(self) -> None:
        """Lists Experiments across all namespaces when namespace is empty."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.list_cluster_custom_object.return_value = {
            "items": [
                _make_experiment(name="exp-ns-a", namespace="ns-a"),
                _make_experiment(name="exp-ns-b", namespace="ns-b"),
            ]
        }

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment()

        assert result.error is False
        assert "exp-ns-a" in result.output
        assert "exp-ns-b" in result.output
        mock_api.list_cluster_custom_object.assert_called_once()

    def test_get_single_experiment_by_name(self) -> None:
        """Fetches a single Experiment by name."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.return_value = _make_experiment(
            name="canary-exp",
            namespace="production",
            phase="Running",
            templates=[
                {"name": "baseline", "replicas": 2},
                {"name": "canary", "replicas": 2},
            ],
        )

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment(namespace="production", name="canary-exp")

        assert result.error is False
        assert "canary-exp" in result.output
        assert "Running" in result.output
        assert "baseline" in result.output

    def test_empty_namespace_returns_message(self) -> None:
        """Returns informational message when no Experiments found."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment(namespace="production")

        assert result.error is False
        assert "No Experiments found" in result.output

    def test_404_by_name_returns_not_found(self) -> None:
        """404 on named get returns not-found message without error flag."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=404)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment(namespace="production", name="ghost-exp")

        assert result.error is False
        assert "not found" in result.output.lower()
        assert "ghost-exp" in result.output

    def test_403_returns_rbac_error(self) -> None:
        """403 on list returns RBAC guidance with error=True."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.side_effect = k8s_exc.ApiException(status=403)

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api):
            result = kubectl_get_experiment(namespace="production")

        assert result.error is True
        assert "RBAC" in result.output
        assert "experiments" in result.output

    def test_k8s_unavailable_returns_error(self) -> None:
        """Returns error when kubernetes SDK is not available."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_experiment

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False):
            result = kubectl_get_experiment()

        assert result.error is True


# ── _RESOURCE_API_MAP Argo entries ───────────────────────────


class TestResourceApiMapArgoEntries:
    """Verify all Argo Rollouts entries are present in _RESOURCE_API_MAP."""

    def test_rollout_entries_present(self) -> None:
        """rollout and rollouts map to custom_argo_rollouts."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert _RESOURCE_API_MAP.get("rollout") == "custom_argo_rollouts"
        assert _RESOURCE_API_MAP.get("rollouts") == "custom_argo_rollouts"

    def test_analysisrun_entries_present(self) -> None:
        """analysisrun(s) map to custom_argo_rollouts."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert _RESOURCE_API_MAP.get("analysisrun") == "custom_argo_rollouts"
        assert _RESOURCE_API_MAP.get("analysisruns") == "custom_argo_rollouts"

    def test_analysistemplate_entries_present(self) -> None:
        """analysistemplate(s) map to custom_argo_rollouts."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert _RESOURCE_API_MAP.get("analysistemplate") == "custom_argo_rollouts"
        assert _RESOURCE_API_MAP.get("analysistemplates") == "custom_argo_rollouts"

    def test_clusteranalysistemplate_entries_present(self) -> None:
        """clusteranalysistemplate(s) map to custom_argo_rollouts_cluster (cluster-scoped)."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert _RESOURCE_API_MAP.get("clusteranalysistemplate") == "custom_argo_rollouts_cluster"
        assert _RESOURCE_API_MAP.get("clusteranalysistemplates") == "custom_argo_rollouts_cluster"

    def test_experiment_entries_present(self) -> None:
        """experiment(s) map to custom_argo_rollouts."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert _RESOURCE_API_MAP.get("experiment") == "custom_argo_rollouts"
        assert _RESOURCE_API_MAP.get("experiments") == "custom_argo_rollouts"

    def test_clusteranalysistemplate_is_cluster_scoped(self) -> None:
        """ClusterAnalysisTemplate entries are in _CLUSTER_SCOPED_RESOURCES."""
        from vaig.tools.gke._resources import _CLUSTER_SCOPED_RESOURCES

        assert "clusteranalysistemplate" in _CLUSTER_SCOPED_RESOURCES
        assert "clusteranalysistemplates" in _CLUSTER_SCOPED_RESOURCES


# ── HPA/VPA Rollout scaleTargetRef matching ──────────────────


class TestScalingRolloutMatch:
    """Tests that HPA and VPA scaleTargetRef matching works for Rollout kind."""

    def test_hpa_matches_rollout_kind(self) -> None:
        """HPA with scaleTargetRef.kind=Rollout is matched to the Rollout workload."""
        from unittest.mock import MagicMock, patch

        from vaig.tools.gke.scaling import get_scaling_status

        def _make_scaling_gke_config():
            from vaig.core.config import GKEConfig
            return GKEConfig(project="test-project", location="us-central1", cluster="test-cluster")

        mock_hpa = MagicMock()
        mock_hpa.metadata.name = "rollout-hpa"
        mock_hpa.metadata.namespace = "production"
        mock_hpa.spec.scale_target_ref.kind = "Rollout"
        mock_hpa.spec.scale_target_ref.name = "my-rollout"
        mock_hpa.spec.min_replicas = 2
        mock_hpa.spec.max_replicas = 10
        mock_hpa.spec.metrics = []
        mock_hpa.status.current_replicas = 3
        mock_hpa.status.desired_replicas = 3
        mock_hpa.status.conditions = []
        mock_hpa.status.current_metrics = []

        mock_auto_v2 = MagicMock()
        mock_auto_v2.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[mock_hpa]
        )

        with patch("vaig.tools.gke.scaling._clients._create_k8s_clients") as mock_clients, \
             patch("kubernetes.client.AutoscalingV2Api", return_value=mock_auto_v2), \
             patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            mock_clients.return_value = (
                MagicMock(), MagicMock(),
                MagicMock(**{"list_namespaced_custom_object.return_value": {"items": []}}),
                MagicMock()
            )
            result = get_scaling_status("my-rollout", namespace="production", gke_config=_make_scaling_gke_config())

        assert result.error is not True
        assert "rollout-hpa" in result.output

    def test_hpa_deployment_kind_still_matches(self) -> None:
        """HPA with scaleTargetRef.kind=Deployment still matches as before."""
        from unittest.mock import MagicMock, patch

        from vaig.tools.gke.scaling import get_scaling_status

        def _make_scaling_gke_config():
            from vaig.core.config import GKEConfig
            return GKEConfig(project="test-project", location="us-central1", cluster="test-cluster")

        mock_hpa = MagicMock()
        mock_hpa.metadata.name = "deploy-hpa"
        mock_hpa.metadata.namespace = "production"
        mock_hpa.spec.scale_target_ref.kind = "Deployment"
        mock_hpa.spec.scale_target_ref.name = "my-deploy"
        mock_hpa.spec.min_replicas = 2
        mock_hpa.spec.max_replicas = 10
        mock_hpa.spec.metrics = []
        mock_hpa.status.current_replicas = 3
        mock_hpa.status.desired_replicas = 3
        mock_hpa.status.conditions = []
        mock_hpa.status.current_metrics = []

        mock_auto_v2 = MagicMock()
        mock_auto_v2.list_namespaced_horizontal_pod_autoscaler.return_value = MagicMock(
            items=[mock_hpa]
        )

        with patch("vaig.tools.gke.scaling._clients._create_k8s_clients") as mock_clients, \
             patch("kubernetes.client.AutoscalingV2Api", return_value=mock_auto_v2), \
             patch("vaig.tools.gke.scaling._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.scaling._K8S_AVAILABLE", True):
            mock_clients.return_value = (
                MagicMock(), MagicMock(),
                MagicMock(**{"list_namespaced_custom_object.return_value": {"items": []}}),
                MagicMock()
            )
            result = get_scaling_status("my-deploy", namespace="production", gke_config=_make_scaling_gke_config())

        assert result.error is not True
        assert "deploy-hpa" in result.output


# ── discover_workloads with include_rollouts ─────────────────


class TestDiscoverWorkloadsRollouts:
    """Tests for discover_workloads with include_rollouts parameter."""

    def test_include_rollouts_false_does_not_query_argo(self) -> None:
        """When include_rollouts=False, no Argo Rollout queries are made."""
        from vaig.core.config import GKEConfig
        from vaig.tools.gke.discovery import discover_workloads

        cfg = GKEConfig(project="test-project", location="us-central1", cluster="test-cluster")

        mock_apps = MagicMock()
        mock_apps.list_namespaced_deployment.return_value.items = []
        mock_apps.list_namespaced_stateful_set.return_value.items = []
        mock_apps.list_namespaced_daemon_set.return_value.items = []
        mock_apps.list_deployment_for_all_namespaces.return_value.items = []
        mock_apps.list_stateful_set_for_all_namespaces.return_value.items = []
        mock_apps.list_daemon_set_for_all_namespaces.return_value.items = []

        mock_custom = MagicMock()
        mock_core = MagicMock()

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.discovery._clients._create_k8s_clients") as mock_clients:
            mock_clients.return_value = (mock_core, mock_apps, mock_custom, MagicMock())
            result = discover_workloads(gke_config=cfg, include_rollouts=False, force_refresh=True)

        assert result.error is not True
        # Argo custom objects API should NOT be called for rollouts
        rollout_calls = [
            call for call in mock_custom.method_calls
            if "rollout" in str(call).lower()
        ]
        assert len(rollout_calls) == 0

    def test_include_rollouts_true_queries_argo(self) -> None:
        """When include_rollouts=True, Argo Rollout resources are queried."""
        from vaig.core.config import GKEConfig
        from vaig.tools.gke.discovery import discover_workloads

        cfg = GKEConfig(project="test-project", location="us-central1", cluster="test-cluster")

        mock_apps = MagicMock()
        mock_apps.list_deployment_for_all_namespaces.return_value.items = []
        mock_apps.list_stateful_set_for_all_namespaces.return_value.items = []
        mock_apps.list_daemon_set_for_all_namespaces.return_value.items = []

        mock_custom = MagicMock()
        # Return empty rollouts list
        mock_custom.list_cluster_custom_object.return_value = {"items": []}
        mock_core = MagicMock()

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.discovery._clients._create_k8s_clients") as mock_clients:
            mock_clients.return_value = (mock_core, mock_apps, mock_custom, MagicMock())
            result = discover_workloads(gke_config=cfg, include_rollouts=True, force_refresh=True)

        assert result.error is not True

    def test_rollouts_graceful_degradation_on_crd_error(self) -> None:
        """When Argo CRDs are not installed, discovery degrades gracefully (no exception)."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.core.config import GKEConfig
        from vaig.tools.gke.discovery import discover_workloads

        cfg = GKEConfig(project="test-project", location="us-central1", cluster="test-cluster")

        mock_apps = MagicMock()
        mock_apps.list_deployment_for_all_namespaces.return_value.items = []
        mock_apps.list_stateful_set_for_all_namespaces.return_value.items = []
        mock_apps.list_daemon_set_for_all_namespaces.return_value.items = []

        mock_custom = MagicMock()
        # Simulate CRD not installed — 404 on list
        mock_custom.list_cluster_custom_object.side_effect = k8s_exc.ApiException(status=404)
        mock_core = MagicMock()

        with patch("vaig.tools.gke.discovery._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.discovery._clients._create_k8s_clients") as mock_clients:
            mock_clients.return_value = (mock_core, mock_apps, mock_custom, MagicMock())
            # Should NOT raise — graceful degradation
            result = discover_workloads(gke_config=cfg, include_rollouts=True, force_refresh=True)

        # Result should be a valid ToolResult (not an exception)
        assert result is not None
        assert result.error is not True
