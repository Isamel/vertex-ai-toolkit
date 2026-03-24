"""Tests for the newly added Kubernetes resource types in GKE tools.

Verifies that RBAC, Storage, Config/Policy, Networking (EndpointSlices),
and Scheduling/Runtime resource types are registered and correctly dispatched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.tools.base import ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_item_list(items: list | None = None) -> MagicMock:
    """Return a mock K8s list object with .items attribute."""
    mock_list = MagicMock()
    mock_list.items = items or []
    return mock_list


# ══════════════════════════════════════════════════════════════
# Registration tests — all new resources must be in _RESOURCE_API_MAP
# ══════════════════════════════════════════════════════════════


class TestNewResourcesRegistered:
    """All new resource types must be registered in _RESOURCE_API_MAP."""

    @pytest.mark.parametrize("resource,expected_group", [
        # RBAC
        ("roles", "rbac"),
        ("clusterroles", "rbac"),
        ("rolebindings", "rbac"),
        ("clusterrolebindings", "rbac"),
        # Storage
        ("storageclasses", "storage"),
        ("volumeattachments", "storage"),
        ("csidrivers", "storage"),
        ("csinodes", "storage"),
        # Config/Policy
        ("limitranges", "core"),
        # Networking
        ("endpointslices", "discovery"),
        # Scheduling
        ("priorityclasses", "scheduling"),
        # Runtime
        ("runtimeclasses", "node"),
    ])
    def test_resource_in_api_map(self, resource: str, expected_group: str) -> None:
        """Each new resource must be mapped to the correct API group."""
        from vaig.tools.gke._resources import _RESOURCE_API_MAP

        assert resource in _RESOURCE_API_MAP, (
            f"'{resource}' must be in _RESOURCE_API_MAP"
        )
        assert _RESOURCE_API_MAP[resource] == expected_group, (
            f"'{resource}' must map to '{expected_group}', "
            f"got '{_RESOURCE_API_MAP[resource]}'"
        )

    @pytest.mark.parametrize("resource", [
        "roles", "clusterroles", "rolebindings", "clusterrolebindings",
        "storageclasses", "volumeattachments", "csidrivers", "csinodes",
        "limitranges", "endpointslices", "priorityclasses", "runtimeclasses",
    ])
    def test_resource_not_in_known_k8s_resources(self, resource: str) -> None:
        """Resources that are now implemented must NOT be in _KNOWN_K8S_RESOURCES."""
        from vaig.tools.gke._resources import _KNOWN_K8S_RESOURCES

        assert resource not in _KNOWN_K8S_RESOURCES, (
            f"'{resource}' was implemented but still listed in _KNOWN_K8S_RESOURCES — "
            "remove it from that set"
        )


# ══════════════════════════════════════════════════════════════
# Cluster-scoped resources
# ══════════════════════════════════════════════════════════════


class TestClusterScopedResources:
    """Cluster-scoped resources must be in _CLUSTER_SCOPED_RESOURCES."""

    @pytest.mark.parametrize("resource", [
        "clusterroles",
        "clusterrolebindings",
        "storageclasses",
        "csidrivers",
        "csinodes",
        "volumeattachments",
        "priorityclasses",
        "runtimeclasses",
    ])
    def test_cluster_scoped_resource_registered(self, resource: str) -> None:
        """Cluster-scoped resources must be in _CLUSTER_SCOPED_RESOURCES."""
        from vaig.tools.gke._resources import _CLUSTER_SCOPED_RESOURCES

        assert resource in _CLUSTER_SCOPED_RESOURCES, (
            f"'{resource}' is cluster-scoped and must be in _CLUSTER_SCOPED_RESOURCES"
        )

    @pytest.mark.parametrize("resource", [
        "roles",
        "rolebindings",
        "limitranges",
        "endpointslices",
    ])
    def test_namespaced_resource_not_cluster_scoped(self, resource: str) -> None:
        """Namespace-scoped resources must NOT be in _CLUSTER_SCOPED_RESOURCES."""
        from vaig.tools.gke._resources import _CLUSTER_SCOPED_RESOURCES

        assert resource not in _CLUSTER_SCOPED_RESOURCES, (
            f"'{resource}' is namespace-scoped — must NOT be in _CLUSTER_SCOPED_RESOURCES"
        )


# ══════════════════════════════════════════════════════════════
# Alias tests
# ══════════════════════════════════════════════════════════════


class TestNewResourceAliases:
    """Short aliases for new resource types must normalise correctly."""

    @pytest.mark.parametrize("alias,expected", [
        # RBAC
        ("role", "roles"),
        ("clusterrole", "clusterroles"),
        ("cr", "clusterroles"),
        ("rolebinding", "rolebindings"),
        ("rb", "rolebindings"),
        ("clusterrolebinding", "clusterrolebindings"),
        ("crb", "clusterrolebindings"),
        # Storage
        ("storageclass", "storageclasses"),
        ("sc", "storageclasses"),
        ("volumeattachment", "volumeattachments"),
        ("va", "volumeattachments"),
        ("csidriver", "csidrivers"),
        ("csinode", "csinodes"),
        # Config/Policy
        ("limitrange", "limitranges"),
        ("lr", "limitranges"),
        # Networking
        ("endpointslice", "endpointslices"),
        ("eps", "endpointslices"),
        # Scheduling
        ("priorityclass", "priorityclasses"),
        ("pc", "priorityclasses"),
        # Runtime
        ("runtimeclass", "runtimeclasses"),
        ("rc", "runtimeclasses"),
    ])
    def test_alias_normalises_correctly(self, alias: str, expected: str) -> None:
        """_normalise_resource() must resolve each alias to its canonical plural."""
        from vaig.tools.gke._resources import _normalise_resource

        assert _normalise_resource(alias) == expected, (
            f"Alias '{alias}' must normalise to '{expected}', "
            f"got '{_normalise_resource(alias)}'"
        )


# ══════════════════════════════════════════════════════════════
# RBAC dispatch tests
# ══════════════════════════════════════════════════════════════


class TestRBACListDispatch:
    """_list_resource() must call RbacAuthorizationV1Api correctly for RBAC resources."""

    @pytest.mark.parametrize("resource,expected_method", [
        ("clusterroles", "list_cluster_role"),
        ("clusterrolebindings", "list_cluster_role_binding"),
    ])
    def test_cluster_scoped_rbac_resources(
        self, resource: str, expected_method: str,
    ) -> None:
        """Cluster-scoped RBAC resources must call the cluster-level list method."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.RbacAuthorizationV1Api") as mock_rbac_cls:
            rbac_instance = MagicMock()
            mock_rbac_cls.return_value = rbac_instance
            getattr(rbac_instance, expected_method).return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource=resource,
                namespace="default",
            )

        getattr(rbac_instance, expected_method).assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error

    @pytest.mark.parametrize("resource,namespaced_method,all_ns_method", [
        ("roles", "list_namespaced_role", "list_role_for_all_namespaces"),
        ("rolebindings", "list_namespaced_role_binding", "list_role_binding_for_all_namespaces"),
    ])
    def test_namespaced_rbac_with_namespace(
        self, resource: str, namespaced_method: str, all_ns_method: str,
    ) -> None:
        """Namespaced RBAC resources must call namespaced list when namespace is specified."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.RbacAuthorizationV1Api") as mock_rbac_cls:
            rbac_instance = MagicMock()
            mock_rbac_cls.return_value = rbac_instance
            getattr(rbac_instance, namespaced_method).return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource=resource,
                namespace="kube-system",
            )

        getattr(rbac_instance, namespaced_method).assert_called_once_with(namespace="kube-system")
        assert not isinstance(result, ToolResult) or not result.error

    @pytest.mark.parametrize("resource,namespaced_method,all_ns_method", [
        ("roles", "list_namespaced_role", "list_role_for_all_namespaces"),
        ("rolebindings", "list_namespaced_role_binding", "list_role_binding_for_all_namespaces"),
    ])
    def test_namespaced_rbac_all_namespaces(
        self, resource: str, namespaced_method: str, all_ns_method: str,
    ) -> None:
        """Namespaced RBAC resources must call the all-namespaces method when namespace='all'."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.RbacAuthorizationV1Api") as mock_rbac_cls:
            rbac_instance = MagicMock()
            mock_rbac_cls.return_value = rbac_instance
            getattr(rbac_instance, all_ns_method).return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource=resource,
                namespace="all",
            )

        getattr(rbac_instance, all_ns_method).assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error


# ══════════════════════════════════════════════════════════════
# Storage dispatch tests
# ══════════════════════════════════════════════════════════════


class TestStorageListDispatch:
    """_list_resource() must call StorageV1Api correctly for storage resources."""

    @pytest.mark.parametrize("resource,expected_method", [
        ("storageclasses", "list_storage_class"),
        ("volumeattachments", "list_volume_attachment"),
        ("csidrivers", "list_csi_driver"),
        ("csinodes", "list_csi_node"),
    ])
    def test_cluster_scoped_storage_resources(
        self, resource: str, expected_method: str,
    ) -> None:
        """Cluster-scoped storage resources must call the correct StorageV1Api method."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.StorageV1Api") as mock_storage_cls:
            storage_instance = MagicMock()
            mock_storage_cls.return_value = storage_instance
            getattr(storage_instance, expected_method).return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource=resource,
                namespace="default",
            )

        getattr(storage_instance, expected_method).assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error


# ══════════════════════════════════════════════════════════════
# LimitRange (core) dispatch test
# ══════════════════════════════════════════════════════════════


class TestLimitRangeDispatch:
    """_list_resource() must call CoreV1 for limitranges."""

    def test_limitranges_namespaced(self) -> None:
        """limitranges must use list_namespaced_limit_range when namespace is provided."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        core_v1.list_namespaced_limit_range.return_value = _make_item_list()

        result = _list_resource(
            core_v1, apps_v1, custom_api,
            resource="limitranges",
            namespace="default",
        )

        core_v1.list_namespaced_limit_range.assert_called_once_with(namespace="default")
        assert not isinstance(result, ToolResult) or not result.error

    def test_limitranges_all_namespaces(self) -> None:
        """limitranges must use list_limit_range_for_all_namespaces when namespace='all'."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        core_v1.list_limit_range_for_all_namespaces.return_value = _make_item_list()

        result = _list_resource(
            core_v1, apps_v1, custom_api,
            resource="limitranges",
            namespace="all",
        )

        core_v1.list_limit_range_for_all_namespaces.assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error


# ══════════════════════════════════════════════════════════════
# EndpointSlice (discovery) dispatch test
# ══════════════════════════════════════════════════════════════


class TestEndpointSliceDispatch:
    """_list_resource() must call DiscoveryV1Api for endpointslices."""

    def test_endpointslices_namespaced(self) -> None:
        """endpointslices must use list_namespaced_endpoint_slice when namespace is provided."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.DiscoveryV1Api") as mock_disc_cls:
            disc_instance = MagicMock()
            mock_disc_cls.return_value = disc_instance
            disc_instance.list_namespaced_endpoint_slice.return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource="endpointslices",
                namespace="default",
            )

        disc_instance.list_namespaced_endpoint_slice.assert_called_once_with(namespace="default")
        assert not isinstance(result, ToolResult) or not result.error

    def test_endpointslices_all_namespaces(self) -> None:
        """endpointslices must use list_endpoint_slice_for_all_namespaces when namespace='all'."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.DiscoveryV1Api") as mock_disc_cls:
            disc_instance = MagicMock()
            mock_disc_cls.return_value = disc_instance
            disc_instance.list_endpoint_slice_for_all_namespaces.return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource="endpointslices",
                namespace="all",
            )

        disc_instance.list_endpoint_slice_for_all_namespaces.assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error


# ══════════════════════════════════════════════════════════════
# Scheduling and Runtime dispatch tests
# ══════════════════════════════════════════════════════════════


class TestSchedulingAndRuntimeDispatch:
    """_list_resource() must call correct APIs for priorityclasses and runtimeclasses."""

    def test_priorityclasses(self) -> None:
        """priorityclasses must call SchedulingV1Api.list_priority_class."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.SchedulingV1Api") as mock_sched_cls:
            sched_instance = MagicMock()
            mock_sched_cls.return_value = sched_instance
            sched_instance.list_priority_class.return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource="priorityclasses",
                namespace="default",
            )

        sched_instance.list_priority_class.assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error

    def test_runtimeclasses(self) -> None:
        """runtimeclasses must call NodeV1Api.list_runtime_class."""
        from vaig.tools.gke._resources import _list_resource

        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        with patch("vaig.tools.gke._resources.NodeV1Api") as mock_node_cls:
            node_instance = MagicMock()
            mock_node_cls.return_value = node_instance
            node_instance.list_runtime_class.return_value = _make_item_list()

            result = _list_resource(
                core_v1, apps_v1, custom_api,
                resource="runtimeclasses",
                namespace="default",
            )

        node_instance.list_runtime_class.assert_called_once()
        assert not isinstance(result, ToolResult) or not result.error


# ══════════════════════════════════════════════════════════════
# End-to-end kubectl_get integration tests
# ══════════════════════════════════════════════════════════════


class TestNewResourcesEndToEnd:
    """kubectl_get must accept all new resource types without returning an error."""

    def _make_gke_config(self) -> object:
        from vaig.core.config import GKEConfig
        return GKEConfig(
            cluster_name="test-cluster",
            project_id="test-project",
            location="us-central1",
            default_namespace="default",
            kubeconfig_path="",
            context="",
            log_limit=100,
            metrics_interval_minutes=60,
            proxy_url="",
        )

    @pytest.mark.parametrize("resource,patch_target,method_name,namespace", [
        ("clusterroles", "vaig.tools.gke._resources.RbacAuthorizationV1Api", "list_cluster_role", "default"),
        ("clusterrolebindings", "vaig.tools.gke._resources.RbacAuthorizationV1Api", "list_cluster_role_binding", "default"),
        ("roles", "vaig.tools.gke._resources.RbacAuthorizationV1Api", "list_namespaced_role", "default"),
        ("rolebindings", "vaig.tools.gke._resources.RbacAuthorizationV1Api", "list_namespaced_role_binding", "default"),
        ("storageclasses", "vaig.tools.gke._resources.StorageV1Api", "list_storage_class", "default"),
        ("csidrivers", "vaig.tools.gke._resources.StorageV1Api", "list_csi_driver", "default"),
        ("csinodes", "vaig.tools.gke._resources.StorageV1Api", "list_csi_node", "default"),
        ("volumeattachments", "vaig.tools.gke._resources.StorageV1Api", "list_volume_attachment", "default"),
        ("priorityclasses", "vaig.tools.gke._resources.SchedulingV1Api", "list_priority_class", "default"),
        ("runtimeclasses", "vaig.tools.gke._resources.NodeV1Api", "list_runtime_class", "default"),
        ("endpointslices", "vaig.tools.gke._resources.DiscoveryV1Api", "list_namespaced_endpoint_slice", "default"),
    ])
    def test_kubectl_get_new_resource(
        self,
        resource: str,
        patch_target: str,
        method_name: str,
        namespace: str,
    ) -> None:
        """kubectl_get must succeed for each new resource type."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = self._make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients, \
             patch(patch_target) as mock_api_cls:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

            api_instance = MagicMock()
            mock_api_cls.return_value = api_instance
            getattr(api_instance, method_name).return_value = _make_item_list()

            result = kubectl_get(resource, gke_config=cfg, namespace=namespace)

        assert result.error is False, (
            f"kubectl_get('{resource}') returned unexpected error: {result.output}"
        )

    def test_kubectl_get_limitranges(self) -> None:
        """kubectl_get('limitranges') must succeed — uses CoreV1Api."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = self._make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())
            core_v1.list_namespaced_limit_range.return_value = _make_item_list()

            result = kubectl_get("limitranges", gke_config=cfg, namespace="default")

        assert result.error is False, (
            f"kubectl_get('limitranges') returned unexpected error: {result.output}"
        )


# ══════════════════════════════════════════════════════════════
# kubectl_get description test — new resources listed in description
# ══════════════════════════════════════════════════════════════


class TestKubectlGetDescriptionUpdated:
    """kubectl_get tool description must mention all newly added resource types."""

    def _get_tool_description(self) -> str:
        from vaig.core.config import GKEConfig
        from vaig.tools.gke._registry import create_gke_tools

        cfg = GKEConfig(
            cluster_name="test-cluster",
            project_id="test-project",
            location="us-central1",
            default_namespace="default",
            kubeconfig_path="",
            context="",
            log_limit=100,
            metrics_interval_minutes=60,
            proxy_url="",
        )
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")
        return kubectl_get_tool.description

    @pytest.mark.parametrize("resource_keyword", [
        "roles",
        "clusterroles",
        "rolebindings",
        "clusterrolebindings",
        "storageclasses",
        "csidrivers",
        "limitranges",
        "endpointslices",
        "priorityclasses",
        "runtimeclasses",
    ])
    def test_description_mentions_new_resources(self, resource_keyword: str) -> None:
        """kubectl_get description must mention each newly added resource type."""
        description = self._get_tool_description()
        assert resource_keyword in description, (
            f"kubectl_get description must mention '{resource_keyword}' "
            f"so the LLM knows it's available"
        )
