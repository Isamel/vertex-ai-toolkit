"""Tests for GKE tool bug fixes.

Bug 1: kubectl_get 'output_format' param renamed to 'output'.
Bug 2: Missing GitOps CRDs (ArgoCD, Flux) in resource whitelist.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs: object) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
        "proxy_url": "",
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


# ══════════════════════════════════════════════════════════════
# Bug 1: kubectl_get parameter name must be 'output' not 'output_format'
# ══════════════════════════════════════════════════════════════


class TestKubectlGetOutputParam:
    """Bug 1 — The ToolDef must expose 'output' not 'output_format' as the param name."""

    def test_tool_param_name_is_output(self) -> None:
        """The ToolDef parameter for output format must be named 'output'."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        param_names = [p.name for p in kubectl_get_tool.parameters]
        assert "output" in param_names, (
            "ToolDef for kubectl_get must have a parameter named 'output' "
            f"(found: {param_names})"
        )
        assert "output_format" not in param_names, (
            "ToolDef for kubectl_get must NOT have a parameter named 'output_format' — "
            "LLMs use the kubectl convention 'output'"
        )

    def test_lambda_accepts_output_kwarg(self) -> None:
        """Calling the tool execute lambda with output='yaml' must not raise TypeError."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

            api_response = MagicMock()
            api_response.items = []
            core_v1.list_namespaced_pod.return_value = api_response

            # This must NOT raise TypeError — the lambda must accept 'output'
            result = kubectl_get_tool.execute(resource="pods", output="yaml")

        assert isinstance(result, ToolResult)

    def test_lambda_does_not_accept_output_format_kwarg(self) -> None:
        """Calling the tool execute lambda with output_format= must raise TypeError (old broken signature)."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        with pytest.raises(TypeError):
            # old callers passing output_format= should fail — they need to be updated
            kubectl_get_tool.execute(resource="pods", output_format="yaml")  # type: ignore[call-arg]


class TestKubectlGetOutputFormatValidation:
    """Bug 1 — 'name' must be a valid output format in kubectl.kubectl_get."""

    def test_name_format_is_valid(self) -> None:
        """kubectl_get must accept output_format='name' without returning an error."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

            api_response = MagicMock()
            api_response.items = []
            core_v1.list_namespaced_pod.return_value = api_response

            result = kubectl_get("pods", gke_config=cfg, output="name")

        assert result.error is False, (
            f"'name' should be a valid output format, got error: {result.output}"
        )

    def test_invalid_format_still_rejected(self) -> None:
        """kubectl_get must still reject unknown formats like 'xml'."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, output="xml")

        assert result.error is True
        assert "Invalid output_format" in result.output

    def test_all_valid_formats_accepted(self) -> None:
        """All documented formats — table, yaml, json, wide, name — must pass validation."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        for fmt in ("table", "yaml", "json", "wide", "name"):
            with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
                 patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
                core_v1 = MagicMock()
                apps_v1 = MagicMock()
                custom_api = MagicMock()
                mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

                api_response = MagicMock()
                api_response.items = []
                core_v1.list_namespaced_pod.return_value = api_response

                result = kubectl_get("pods", gke_config=cfg, output=fmt)

            assert result.error is False, (
                f"Format '{fmt}' should be valid but got error: {result.output}"
            )


# ══════════════════════════════════════════════════════════════
# Bug 2: GitOps CRDs in resource whitelist
# ══════════════════════════════════════════════════════════════


class TestGitOpsCRDsInWhitelist:
    """Bug 2 — ArgoCD and Flux CRDs must be queryable via kubectl_get."""

    def test_argocd_application_in_resource_map(self) -> None:
        """applications.argoproj.io must be in the resource API map."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert "applications.argoproj.io" in _RESOURCE_API_MAP, (
            "ArgoCD Application CRD must be registered in _RESOURCE_API_MAP"
        )
        assert _RESOURCE_API_MAP["applications.argoproj.io"] == "custom_argocd"

    def test_flux_helmrelease_in_resource_map(self) -> None:
        """helmreleases.helm.toolkit.fluxcd.io must be in the resource API map."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert "helmreleases.helm.toolkit.fluxcd.io" in _RESOURCE_API_MAP, (
            "Flux HelmRelease CRD must be registered in _RESOURCE_API_MAP"
        )
        assert _RESOURCE_API_MAP["helmreleases.helm.toolkit.fluxcd.io"] == "custom_flux_helm"

    def test_flux_kustomization_in_resource_map(self) -> None:
        """kustomizations.kustomize.toolkit.fluxcd.io must be in the resource API map."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert "kustomizations.kustomize.toolkit.fluxcd.io" in _RESOURCE_API_MAP, (
            "Flux Kustomization CRD must be registered in _RESOURCE_API_MAP"
        )
        assert _RESOURCE_API_MAP["kustomizations.kustomize.toolkit.fluxcd.io"] == "custom_flux_kustomize"

    def test_argocd_aliases_resolve(self) -> None:
        """Short names like 'app', 'application', 'argoapp' must resolve to the full CRD name."""
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("app") == "applications.argoproj.io"
        assert _normalise_resource("application") == "applications.argoproj.io"
        assert _normalise_resource("argoapp") == "applications.argoproj.io"

    def test_flux_helm_aliases_resolve(self) -> None:
        """Short names like 'hr', 'helmrelease' must resolve to the full CRD name."""
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("hr") == "helmreleases.helm.toolkit.fluxcd.io"
        assert _normalise_resource("helmrelease") == "helmreleases.helm.toolkit.fluxcd.io"

    def test_flux_kustomization_aliases_resolve(self) -> None:
        """Short names like 'ks', 'kustomization' must resolve to the full CRD name."""
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource("ks") == "kustomizations.kustomize.toolkit.fluxcd.io"
        assert _normalise_resource("kustomization") == "kustomizations.kustomize.toolkit.fluxcd.io"

    @pytest.mark.parametrize("resource,group,version,plural", [
        (
            "applications.argoproj.io",
            "argoproj.io",
            "v1alpha1",
            "applications",
        ),
        (
            "helmreleases.helm.toolkit.fluxcd.io",
            "helm.toolkit.fluxcd.io",
            "v2beta1",
            "helmreleases",
        ),
        (
            "kustomizations.kustomize.toolkit.fluxcd.io",
            "kustomize.toolkit.fluxcd.io",
            "v1",
            "kustomizations",
        ),
    ])
    def test_list_resource_calls_custom_api_namespaced(
        self,
        resource: str,
        group: str,
        version: str,
        plural: str,
    ) -> None:
        """_list_resource() must call list_namespaced_custom_object with correct args."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {"items": []}

        result = _list_resource(
            core_v1, apps_v1, custom_api,
            resource=resource,
            namespace="argocd",
        )

        custom_api.list_namespaced_custom_object.assert_called_once_with(
            group=group,
            version=version,
            namespace="argocd",
            plural=plural,
        )
        # Result should be a _DictItemList (not a ToolResult error)
        assert not isinstance(result, ToolResult) or not result.error

    @pytest.mark.parametrize("resource,group,version,plural", [
        (
            "applications.argoproj.io",
            "argoproj.io",
            "v1alpha1",
            "applications",
        ),
        (
            "helmreleases.helm.toolkit.fluxcd.io",
            "helm.toolkit.fluxcd.io",
            "v2beta1",
            "helmreleases",
        ),
        (
            "kustomizations.kustomize.toolkit.fluxcd.io",
            "kustomize.toolkit.fluxcd.io",
            "v1",
            "kustomizations",
        ),
    ])
    def test_list_resource_calls_custom_api_cluster_scoped(
        self,
        resource: str,
        group: str,
        version: str,
        plural: str,
    ) -> None:
        """_list_resource() must call list_cluster_custom_object when namespace='all'."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = _list_resource(
            core_v1, apps_v1, custom_api,
            resource=resource,
            namespace="all",
        )

        custom_api.list_cluster_custom_object.assert_called_once_with(
            group=group,
            version=version,
            plural=plural,
        )
        assert not isinstance(result, ToolResult) or not result.error

    def test_kubectl_get_argocd_application_end_to_end(self) -> None:
        """kubectl_get('applications.argoproj.io') must succeed via the custom_api path."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())
            custom_api.list_namespaced_custom_object.return_value = {"items": []}

            result = kubectl_get(
                "applications.argoproj.io",
                gke_config=cfg,
                namespace="argocd",
            )

        assert result.error is False, f"Unexpected error: {result.output}"
