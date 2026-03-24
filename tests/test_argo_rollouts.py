"""Tests for Argo Rollouts introspection tools — rollouts, analysis runs, analysis templates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear the shared CRD existence cache before each test."""
    from vaig.tools.gke.argocd import _crd_exists_cache
    _crd_exists_cache.clear()


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

    def test_detect_returns_true_when_crd_exists(self) -> None:
        """Returns True when the rollouts.argoproj.io CRD is present."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        mock_ext_api.read_custom_resource_definition.return_value = MagicMock()

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts()

        assert result is True

    def test_detect_returns_false_when_k8s_unavailable(self) -> None:
        """Returns False when the kubernetes SDK is not available."""
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", False):
            result = detect_argo_rollouts()

        assert result is False

    def test_detect_returns_false_when_crd_not_found(self) -> None:
        """Returns False when the CRD returns 404."""
        from kubernetes.client import exceptions as k8s_exc

        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts

        mock_ext_api = MagicMock()
        api_exc = k8s_exc.ApiException(status=404)
        mock_ext_api.read_custom_resource_definition.side_effect = api_exc

        with patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True), \
             patch("kubernetes.client.ApiextensionsV1Api", return_value=mock_ext_api), \
             patch("kubernetes.config.load_incluster_config", side_effect=Exception("not in cluster")), \
             patch("kubernetes.config.load_kube_config", return_value=None), \
             patch("kubernetes.config.ConfigException", Exception):
            result = detect_argo_rollouts()

        assert result is False


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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
        )

    def test_empty_namespace_returns_no_rollouts(self) -> None:
        """Returns empty message when no rollouts exist."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.return_value = {"items": []}

        with patch("vaig.tools.gke.argo_rollouts._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
            result = kubectl_get_rollout(namespace="production")

        assert result.error is True
        assert "RBAC" in result.output

    def test_k8s_unavailable_returns_error(self) -> None:
        """Returns error when kubernetes SDK is unavailable."""
        from vaig.tools.gke.argo_rollouts import kubectl_get_rollout

        with patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=True):
            result = kubectl_get_rollout(namespace="production")

        assert result.error is True
        assert "kubernetes SDK not available" in result.output


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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
             patch("vaig.tools.gke.argo_rollouts._get_custom_objects_api", return_value=mock_api), \
             patch("vaig.tools.gke.argo_rollouts._clients._k8s_unavailable", return_value=False):
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
