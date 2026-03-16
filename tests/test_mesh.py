"""Tests for mesh.py — Phase 1 helpers, Phase 2 overview, Phase 3 config."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.tools.base import ToolResult
from vaig.tools.gke.mesh import (
    _ISTIO_CRD_VERSIONS,
    _ISTIO_NETWORKING_GROUP,
    _ISTIO_SECURITY_GROUP,
    _KIND_PLURALS,
    _MAX_RESOURCES_PER_TYPE,
    _MESH_CACHE_TTL,
    _detect_mesh_presence,
    _format_authorization_policy,
    _format_destination_rule,
    _format_gateway,
    _format_injection_table,
    _format_mesh_status,
    _format_peer_authentication,
    _format_request_authentication,
    _format_sidecar_table,
    _format_virtual_service,
    _get_istio_version,
    _kind_to_plural,
    _read_custom_resources,
    _resolve_crd_version,
    clear_crd_version_cache,
    get_mesh_config,
    get_mesh_overview,
    get_mesh_security,
    get_sidecar_status,
)


@pytest.fixture(autouse=True)
def _clear_crd_cache() -> None:
    """Clear the CRD version cache before each test."""
    clear_crd_version_cache()


def _make_api_exception(status: int, reason: str = "") -> Exception:
    """Create an ApiException compatible with both real kubernetes SDK and test patches.

    When running in the full test suite, test_gke_tools.py may patch
    kubernetes.client.exceptions.ApiException with a simple class that
    doesn't accept keyword arguments.  This helper uses the real class
    with positional args to avoid that issue.
    """
    from kubernetes.client.exceptions import ApiException

    try:
        exc = ApiException(status=status, reason=reason)
    except TypeError:
        # Fallback: class was patched with class attrs instead of __init__
        exc = ApiException()
        exc.status = status  # type: ignore[attr-defined]
        exc.reason = reason  # type: ignore[attr-defined]
    return exc


# ── Constants ────────────────────────────────────────────────


class TestConstants:
    def test_mesh_cache_ttl_is_30(self) -> None:
        assert _MESH_CACHE_TTL == 30

    def test_max_resources_per_type_is_50(self) -> None:
        assert _MAX_RESOURCES_PER_TYPE == 50

    def test_istio_networking_group(self) -> None:
        assert _ISTIO_NETWORKING_GROUP == "networking.istio.io"

    def test_istio_security_group(self) -> None:
        assert _ISTIO_SECURITY_GROUP == "security.istio.io"

    def test_crd_versions_has_all_expected_kinds(self) -> None:
        expected = {
            "VirtualService", "DestinationRule", "Gateway",
            "PeerAuthentication", "AuthorizationPolicy", "RequestAuthentication",
        }
        assert set(_ISTIO_CRD_VERSIONS.keys()) == expected

    def test_crd_versions_start_with_v1(self) -> None:
        for kind, versions in _ISTIO_CRD_VERSIONS.items():
            assert versions[0] == "v1", f"{kind} version chain should start with v1"


# ── _kind_to_plural ──────────────────────────────────────────


class TestKindToPlural:
    def test_known_kinds(self) -> None:
        assert _kind_to_plural("VirtualService") == "virtualservices"
        assert _kind_to_plural("DestinationRule") == "destinationrules"
        assert _kind_to_plural("Gateway") == "gateways"
        assert _kind_to_plural("PeerAuthentication") == "peerauthentications"
        assert _kind_to_plural("AuthorizationPolicy") == "authorizationpolicies"
        assert _kind_to_plural("RequestAuthentication") == "requestauthentications"
        assert _kind_to_plural("ServiceEntry") == "serviceentries"
        assert _kind_to_plural("Sidecar") == "sidecars"
        assert _kind_to_plural("EnvoyFilter") == "envoyfilters"

    def test_unknown_kind_falls_back_to_naive_plural(self) -> None:
        assert _kind_to_plural("SomethingNew") == "somethingnews"

    def test_all_kind_plurals_map_entries(self) -> None:
        for kind, plural in _KIND_PLURALS.items():
            assert _kind_to_plural(kind) == plural


# ── _detect_mesh_presence ────────────────────────────────────


class TestDetectMeshPresence:
    def test_mesh_present_with_istiod(self) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()

        # Namespace exists (no exception = found)
        core_v1.read_namespace.return_value = MagicMock()

        # istiod deployment found with healthy replicas
        istiod = MagicMock()
        istiod.status.ready_replicas = 2
        istiod.spec.replicas = 2
        apps_v1.read_namespaced_deployment.return_value = istiod

        # No managed ASM
        ns_obj = MagicMock()
        ns_obj.metadata.labels = {}
        ns_list = MagicMock()
        ns_list.items = [ns_obj]
        core_v1.list_namespace.return_value = ns_list

        result = _detect_mesh_presence(core_v1, apps_v1)

        assert result["installed"] is True
        assert result["istiod_found"] is True
        assert result["managed"] is False
        assert result["namespace"] == "istio-system"
        assert result["warnings"] == []

    def test_mesh_not_present_404(self) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()

        core_v1.read_namespace.side_effect = _make_api_exception(404, "Not Found")

        # No managed ASM namespaces
        ns_obj = MagicMock()
        ns_obj.metadata.labels = {}
        ns_list = MagicMock()
        ns_list.items = [ns_obj]
        core_v1.list_namespace.return_value = ns_list

        result = _detect_mesh_presence(core_v1, apps_v1)

        assert result["installed"] is False
        assert result["istiod_found"] is False
        assert result["managed"] is False

    def test_managed_asm_detected(self) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()

        # Namespace exists
        core_v1.read_namespace.return_value = MagicMock()

        # istiod NOT found (managed ASM doesn't always have it)
        apps_v1.read_namespaced_deployment.side_effect = _make_api_exception(404, "Not Found")

        # Namespace with asm-managed label
        ns_obj = MagicMock()
        ns_obj.metadata.labels = {"istio.io/rev": "asm-managed-rapid"}
        ns_list = MagicMock()
        ns_list.items = [ns_obj]
        core_v1.list_namespace.return_value = ns_list

        result = _detect_mesh_presence(core_v1, apps_v1)

        assert result["installed"] is True
        assert result["managed"] is True
        assert result["istiod_found"] is False

    def test_rbac_403_adds_warning(self) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()

        core_v1.read_namespace.side_effect = _make_api_exception(403, "Forbidden")

        # list_namespace also fails
        core_v1.list_namespace.side_effect = _make_api_exception(403, "Forbidden")

        result = _detect_mesh_presence(core_v1, apps_v1)

        assert result["installed"] is False
        assert len(result["warnings"]) >= 1
        assert any("403" in w for w in result["warnings"])

    def test_istiod_unhealthy_warns(self) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()

        core_v1.read_namespace.return_value = MagicMock()

        # istiod unhealthy: 0/2 replicas
        istiod = MagicMock()
        istiod.status.ready_replicas = 0
        istiod.spec.replicas = 2
        apps_v1.read_namespaced_deployment.return_value = istiod

        ns_list = MagicMock()
        ns_list.items = []
        core_v1.list_namespace.return_value = ns_list

        result = _detect_mesh_presence(core_v1, apps_v1)

        assert result["installed"] is True
        assert result["istiod_found"] is True
        assert any("0/2" in w for w in result["warnings"])


# ── _get_istio_version ───────────────────────────────────────


class TestGetIstioVersion:
    def test_version_from_istiod_image_tag(self) -> None:
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        container = MagicMock()
        container.image = "gcr.io/istio-release/pilot:1.20.3"

        istiod = MagicMock()
        istiod.spec.template.spec.containers = [container]
        apps_v1.read_namespaced_deployment.return_value = istiod

        assert _get_istio_version(apps_v1, custom_api) == "1.20.3"

    def test_version_with_v_prefix(self) -> None:
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        container = MagicMock()
        container.image = "gcr.io/istio-release/pilot:v1.19.0"

        istiod = MagicMock()
        istiod.spec.template.spec.containers = [container]
        apps_v1.read_namespaced_deployment.return_value = istiod

        assert _get_istio_version(apps_v1, custom_api) == "v1.19.0"

    def test_fallback_to_istiooperator_cr(self) -> None:
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        # istiod deployment not found
        apps_v1.read_namespaced_deployment.side_effect = _make_api_exception(404, "Not Found")

        # IstioOperator CR available
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [{"spec": {"tag": "1.18.2"}}],
        }

        assert _get_istio_version(apps_v1, custom_api) == "1.18.2"

    def test_returns_unknown_when_nothing_available(self) -> None:
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        apps_v1.read_namespaced_deployment.side_effect = _make_api_exception(404, "Not Found")
        custom_api.list_namespaced_custom_object.side_effect = _make_api_exception(404, "Not Found")

        assert _get_istio_version(apps_v1, custom_api) == "unknown"

    def test_version_from_istiooperator_version_field(self) -> None:
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        apps_v1.read_namespaced_deployment.side_effect = _make_api_exception(404, "Not Found")

        custom_api.list_namespaced_custom_object.return_value = {
            "items": [{"spec": {"version": "1.17.0"}}],
        }

        assert _get_istio_version(apps_v1, custom_api) == "1.17.0"

    def test_non_version_tag_skipped(self) -> None:
        """Tags like 'latest' or 'debug' should not be returned."""
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        container = MagicMock()
        container.image = "gcr.io/istio-release/pilot:latest"

        istiod = MagicMock()
        istiod.spec.template.spec.containers = [container]
        apps_v1.read_namespaced_deployment.return_value = istiod

        # No IstioOperator fallback
        custom_api.list_namespaced_custom_object.side_effect = Exception("not found")

        assert _get_istio_version(apps_v1, custom_api) == "unknown"


# ── _resolve_crd_version ─────────────────────────────────────


class TestResolveCrdVersion:
    def test_resolves_v1_first(self) -> None:
        custom_api = MagicMock()
        # v1 list succeeds
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result == "v1"

    def test_falls_back_to_v1beta1(self) -> None:
        custom_api = MagicMock()
        # v1 fails, v1beta1 succeeds
        custom_api.list_cluster_custom_object.side_effect = [
            _make_api_exception(404, "Not Found"),
            {"items": []},
        ]

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result == "v1beta1"

    def test_falls_back_to_v1alpha3(self) -> None:
        custom_api = MagicMock()
        # v1 and v1beta1 fail, v1alpha3 succeeds
        custom_api.list_cluster_custom_object.side_effect = [
            _make_api_exception(404, "Not Found"),
            _make_api_exception(404, "Not Found"),
            {"items": []},
        ]

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result == "v1alpha3"

    def test_returns_none_when_all_fail(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.side_effect = _make_api_exception(404, "Not Found")

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result is None

    def test_caches_resolved_version(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")

        # Second call should use cache
        custom_api.reset_mock()
        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result == "v1"
        custom_api.list_cluster_custom_object.assert_not_called()

    def test_403_skips_version(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.side_effect = [
            _make_api_exception(403, "Forbidden"),  # v1 forbidden
            {"items": []},                            # v1beta1 works
        ]

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "VirtualService")
        assert result == "v1beta1"

    def test_security_group_kinds(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = _resolve_crd_version(custom_api, _ISTIO_SECURITY_GROUP, "PeerAuthentication")
        assert result == "v1"

    def test_unknown_kind_uses_default_chain(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = _resolve_crd_version(custom_api, _ISTIO_NETWORKING_GROUP, "SomeNewKind")
        assert result == "v1"


# ── _read_custom_resources ───────────────────────────────────


class TestReadCustomResources:
    def test_reads_namespaced_resources(self) -> None:
        custom_api = MagicMock()
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {"metadata": {"name": "vs1"}},
                {"metadata": {"name": "vs2"}},
            ],
        }

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
            namespace="default",
        )

        assert len(result) == 2
        assert result[0]["metadata"]["name"] == "vs1"
        custom_api.list_namespaced_custom_object.assert_called_once_with(
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            namespace="default",
            plural="virtualservices",
        )

    def test_reads_cluster_scoped_resources(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.return_value = {
            "items": [{"metadata": {"name": "gw1"}}],
        }

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="Gateway",
            namespace=None,
        )

        assert len(result) == 1
        custom_api.list_cluster_custom_object.assert_called_once_with(
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            plural="gateways",
        )

    def test_truncates_at_max_resources(self) -> None:
        custom_api = MagicMock()
        items = [{"metadata": {"name": f"vs{i}"}} for i in range(_MAX_RESOURCES_PER_TYPE + 20)]
        custom_api.list_cluster_custom_object.return_value = {"items": items}

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
        )

        assert len(result) == _MAX_RESOURCES_PER_TYPE

    def test_403_returns_empty_list(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.side_effect = _make_api_exception(403, "Forbidden")

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
        )

        assert result == []

    def test_404_returns_empty_list(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.side_effect = _make_api_exception(404, "Not Found")

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
        )

        assert result == []

    def test_non_api_exception_returns_empty_list(self) -> None:
        custom_api = MagicMock()
        custom_api.list_cluster_custom_object.side_effect = ConnectionError("network down")

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
        )

        assert result == []

    def test_exact_max_not_truncated(self) -> None:
        custom_api = MagicMock()
        items = [{"metadata": {"name": f"vs{i}"}} for i in range(_MAX_RESOURCES_PER_TYPE)]
        custom_api.list_cluster_custom_object.return_value = {"items": items}

        result = _read_custom_resources(
            custom_api,
            group=_ISTIO_NETWORKING_GROUP,
            version="v1",
            kind="VirtualService",
        )

        assert len(result) == _MAX_RESOURCES_PER_TYPE


# ── Formatter helpers ────────────────────────────────────────


class TestFormatMeshStatus:
    def test_no_mesh(self) -> None:
        presence = {"installed": False, "managed": False, "namespace": "istio-system",
                     "istiod_found": False, "warnings": []}
        assert _format_mesh_status(presence, "unknown") == "No service mesh detected."

    def test_istio_with_istiod(self) -> None:
        presence = {"installed": True, "managed": False, "namespace": "istio-system",
                     "istiod_found": True, "warnings": []}
        result = _format_mesh_status(presence, "1.20.3")
        assert "Mesh type: Istio" in result
        assert "Version: 1.20.3" in result
        assert "istiod): running" in result

    def test_managed_asm(self) -> None:
        presence = {"installed": True, "managed": True, "namespace": "istio-system",
                     "istiod_found": False, "warnings": []}
        result = _format_mesh_status(presence, "1.20.3")
        assert "Anthos Service Mesh (managed)" in result
        assert "Google-managed" in result

    def test_istio_without_istiod(self) -> None:
        presence = {"installed": True, "managed": False, "namespace": "istio-system",
                     "istiod_found": False, "warnings": []}
        result = _format_mesh_status(presence, "unknown")
        assert "istiod): not found" in result


class TestFormatInjectionTable:
    def test_empty_list(self) -> None:
        result = _format_injection_table([])
        assert "(no namespaces found)" in result

    def test_mixed_injection(self) -> None:
        namespaces = [
            {"name": "default", "injection": "enabled", "revision": ""},
            {"name": "kube-system", "injection": "not set", "revision": ""},
            {"name": "app-ns", "injection": "enabled", "revision": "asm-managed"},
        ]
        result = _format_injection_table(namespaces)
        assert "default" in result
        assert "enabled" in result
        assert "asm-managed" in result
        assert "NAMESPACE" in result  # header


class TestFormatVirtualService:
    def test_basic_vs(self) -> None:
        vs = {
            "metadata": {"name": "reviews-route", "namespace": "default"},
            "spec": {
                "hosts": ["reviews.default.svc.cluster.local"],
                "http": [{
                    "route": [
                        {"destination": {"host": "reviews", "subset": "v1"}, "weight": 80},
                        {"destination": {"host": "reviews", "subset": "v2"}, "weight": 20},
                    ],
                }],
            },
        }
        result = _format_virtual_service(vs)
        assert "reviews-route" in result
        assert "reviews.default.svc.cluster.local" in result
        assert "reviews/v1" in result
        assert "80%" in result
        assert "20%" in result

    def test_vs_with_match_rules(self) -> None:
        vs = {
            "metadata": {"name": "test-vs", "namespace": "ns1"},
            "spec": {
                "hosts": ["test.example.com"],
                "http": [{
                    "match": [{"uri": {"prefix": "/api"}}],
                    "route": [{"destination": {"host": "api-svc", "port": {"number": 8080}}}],
                }],
            },
        }
        result = _format_virtual_service(vs)
        assert "prefix=/api" in result
        assert "api-svc" in result
        assert "8080" in result

    def test_vs_with_gateways(self) -> None:
        vs = {
            "metadata": {"name": "gw-vs", "namespace": "default"},
            "spec": {
                "hosts": ["*.example.com"],
                "gateways": ["my-gateway"],
                "http": [{"route": [{"destination": {"host": "web"}}]}],
            },
        }
        result = _format_virtual_service(vs)
        assert "my-gateway" in result

    def test_vs_with_tcp_routes(self) -> None:
        vs = {
            "metadata": {"name": "tcp-vs", "namespace": "default"},
            "spec": {
                "hosts": ["db.local"],
                "tcp": [{"route": [{"destination": {"host": "db"}}]}],
            },
        }
        result = _format_virtual_service(vs)
        assert "TCP routes: 1" in result

    def test_vs_with_timeout_retries(self) -> None:
        vs = {
            "metadata": {"name": "retry-vs", "namespace": "default"},
            "spec": {
                "hosts": ["svc"],
                "http": [{
                    "route": [{"destination": {"host": "svc"}}],
                    "timeout": "5s",
                    "retries": {"attempts": 3},
                }],
            },
        }
        result = _format_virtual_service(vs)
        assert "timeout=5s" in result
        assert "retries=3" in result

    def test_vs_empty_spec(self) -> None:
        vs = {"metadata": {"name": "empty", "namespace": "default"}, "spec": {}}
        result = _format_virtual_service(vs)
        assert "empty" in result


class TestFormatDestinationRule:
    def test_basic_dr(self) -> None:
        dr = {
            "metadata": {"name": "reviews-dr", "namespace": "default"},
            "spec": {
                "host": "reviews.default.svc.cluster.local",
                "subsets": [
                    {"name": "v1", "labels": {"version": "v1"}},
                    {"name": "v2", "labels": {"version": "v2"}},
                ],
            },
        }
        result = _format_destination_rule(dr)
        assert "reviews-dr" in result
        assert "reviews.default.svc.cluster.local" in result
        assert "v1(version=v1)" in result
        assert "v2(version=v2)" in result

    def test_dr_with_traffic_policy(self) -> None:
        dr = {
            "metadata": {"name": "policy-dr", "namespace": "default"},
            "spec": {
                "host": "my-svc",
                "trafficPolicy": {
                    "loadBalancer": {"simple": "ROUND_ROBIN"},
                    "connectionPool": {
                        "tcp": {"maxConnections": 100},
                        "http": {"h2UpgradePolicy": "UPGRADE"},
                    },
                    "outlierDetection": {
                        "consecutive5xxErrors": 5,
                        "interval": "10s",
                        "baseEjectionTime": "30s",
                    },
                },
            },
        }
        result = _format_destination_rule(dr)
        assert "ROUND_ROBIN" in result
        assert "maxConn=100" in result
        assert "5xx=5" in result
        assert "eject=30s" in result

    def test_dr_with_tls_policy(self) -> None:
        dr = {
            "metadata": {"name": "tls-dr", "namespace": "default"},
            "spec": {
                "host": "my-svc",
                "trafficPolicy": {
                    "tls": {"mode": "ISTIO_MUTUAL"},
                },
            },
        }
        result = _format_destination_rule(dr)
        assert "ISTIO_MUTUAL" in result

    def test_dr_empty_spec(self) -> None:
        dr = {"metadata": {"name": "empty", "namespace": "default"}, "spec": {}}
        result = _format_destination_rule(dr)
        assert "empty" in result


class TestFormatGateway:
    def test_basic_gateway(self) -> None:
        gw = {
            "metadata": {"name": "my-gw", "namespace": "istio-system"},
            "spec": {
                "selector": {"istio": "ingressgateway"},
                "servers": [{
                    "port": {"number": 443, "name": "https", "protocol": "HTTPS"},
                    "hosts": ["*.example.com"],
                    "tls": {"mode": "SIMPLE"},
                }],
            },
        }
        result = _format_gateway(gw)
        assert "my-gw" in result
        assert "istio=ingressgateway" in result
        assert "443/HTTPS" in result
        assert "*.example.com" in result
        assert "SIMPLE" in result

    def test_gateway_multiple_servers(self) -> None:
        gw = {
            "metadata": {"name": "multi-gw", "namespace": "default"},
            "spec": {
                "servers": [
                    {"port": {"number": 80, "protocol": "HTTP"}, "hosts": ["app.local"]},
                    {"port": {"number": 443, "protocol": "HTTPS"}, "hosts": ["app.local"],
                     "tls": {"mode": "MUTUAL"}},
                ],
            },
        }
        result = _format_gateway(gw)
        assert "80/HTTP" in result
        assert "443/HTTPS" in result
        assert "MUTUAL" in result

    def test_gateway_empty_spec(self) -> None:
        gw = {"metadata": {"name": "empty", "namespace": "default"}, "spec": {}}
        result = _format_gateway(gw)
        assert "empty" in result


# ── get_mesh_overview (Phase 2) ──────────────────────────────


def _mock_gke_config() -> MagicMock:
    """Create a mock GKEConfig for testing."""
    config = MagicMock()
    config.kubeconfig_path = ""
    config.context = ""
    config.proxy_url = ""
    config.default_namespace = "default"
    return config


class TestGetMeshOverview:
    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_no_mesh_detected(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # No istio-system namespace
        core_v1.read_namespace.side_effect = _make_api_exception(404, "Not Found")

        # No managed ASM
        ns_obj = MagicMock()
        ns_obj.metadata.labels = {}
        ns_list = MagicMock()
        ns_list.items = [ns_obj]
        core_v1.list_namespace.return_value = ns_list

        result = get_mesh_overview(gke_config=_mock_gke_config())

        assert result.error is False
        assert "No service mesh detected" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_istio_mesh_with_injection(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # istio-system namespace exists
        core_v1.read_namespace.return_value = MagicMock()

        # istiod healthy
        istiod = MagicMock()
        istiod.status.ready_replicas = 2
        istiod.spec.replicas = 2
        apps_v1.read_namespaced_deployment.return_value = istiod

        # Namespaces for injection status
        ns_default = MagicMock()
        ns_default.metadata.labels = {"istio-injection": "enabled"}
        ns_default.metadata.name = "default"

        ns_kube = MagicMock()
        ns_kube.metadata.labels = {}
        ns_kube.metadata.name = "kube-system"

        ns_app = MagicMock()
        ns_app.metadata.labels = {"istio.io/rev": "asm-managed"}
        ns_app.metadata.name = "app-ns"

        ns_list = MagicMock()
        ns_list.items = [ns_default, ns_kube, ns_app]
        core_v1.list_namespace.return_value = ns_list

        # Version
        container = MagicMock()
        container.image = "gcr.io/istio-release/pilot:1.20.3"
        istiod.spec.template.spec.containers = [container]

        result = get_mesh_overview(gke_config=_mock_gke_config())

        assert result.error is False
        assert "Mesh Overview" in result.output
        assert "Istio" in result.output or "Anthos" in result.output
        assert "1.20.3" in result.output
        assert "default" in result.output
        assert "enabled" in result.output
        assert "2/3 namespaces" in result.output or "Injection enabled:" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_overview_with_namespace_filter(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Mesh exists
        core_v1.read_namespace.return_value = MagicMock()
        istiod = MagicMock()
        istiod.status.ready_replicas = 1
        istiod.spec.replicas = 1
        apps_v1.read_namespaced_deployment.return_value = istiod

        # Filtered namespace
        ns_obj = MagicMock()
        ns_obj.metadata.labels = {"istio-injection": "enabled"}
        ns_obj.metadata.name = "my-app"

        # read_namespace for the specific ns (used by injection check)
        # The first call to read_namespace is for istio-system (mesh detection),
        # the second is for the specific namespace filter
        core_v1.read_namespace.side_effect = [MagicMock(), ns_obj]

        # list_namespace for managed ASM check
        ns_list = MagicMock()
        ns_list.items = [ns_obj]
        core_v1.list_namespace.return_value = ns_list

        container = MagicMock()
        container.image = "pilot:1.20.0"
        istiod.spec.template.spec.containers = [container]

        result = get_mesh_overview(gke_config=_mock_gke_config(), namespace="my-app")
        assert result.error is False
        assert "my-app" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_overview_rbac_403_on_namespaces(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Mesh exists (namespace read OK, but listing fails)
        core_v1.read_namespace.return_value = MagicMock()
        istiod = MagicMock()
        istiod.status.ready_replicas = 1
        istiod.spec.replicas = 1
        apps_v1.read_namespaced_deployment.return_value = istiod

        container = MagicMock()
        container.image = "pilot:1.20.0"
        istiod.spec.template.spec.containers = [container]

        # list_namespace fails with 403
        core_v1.list_namespace.side_effect = _make_api_exception(403, "Forbidden")

        result = get_mesh_overview(gke_config=_mock_gke_config())
        assert result.error is False
        assert "403" in result.output or "RBAC" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    def test_overview_k8s_client_failure(
        self, mock_create_clients: MagicMock,
    ) -> None:
        mock_create_clients.return_value = ToolResult(
            output="Failed to configure Kubernetes client", error=True,
        )
        result = get_mesh_overview(gke_config=_mock_gke_config())
        assert result.error is True
        assert "Failed" in result.output

    @patch("vaig.tools.gke.mesh._cache._get_cached")
    def test_overview_uses_cache(self, mock_get_cached: MagicMock) -> None:
        mock_get_cached.return_value = "cached mesh overview"
        result = get_mesh_overview(gke_config=_mock_gke_config())
        assert result.output == "cached mesh overview"
        assert result.error is False

    @patch("vaig.tools.gke.mesh._K8S_AVAILABLE", False)
    def test_overview_k8s_unavailable(self) -> None:
        result = get_mesh_overview(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_overview_managed_asm(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # istio-system exists
        core_v1.read_namespace.return_value = MagicMock()
        # No istiod
        apps_v1.read_namespaced_deployment.side_effect = _make_api_exception(404, "Not Found")

        # Managed ASM namespace
        ns_managed = MagicMock()
        ns_managed.metadata.labels = {"istio.io/rev": "asm-managed-rapid"}
        ns_managed.metadata.name = "app-ns"

        ns_list = MagicMock()
        ns_list.items = [ns_managed]
        core_v1.list_namespace.return_value = ns_list

        # Version unknown (no istiod, no operator)
        custom_api.list_namespaced_custom_object.side_effect = Exception("not found")

        result = get_mesh_overview(gke_config=_mock_gke_config())
        assert result.error is False
        assert "Anthos Service Mesh" in result.output
        assert "Google-managed" in result.output


# ── get_mesh_config (Phase 3) ────────────────────────────────


class TestGetMeshConfig:
    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_config_with_all_resources(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # All CRD versions resolve to v1
        custom_api.list_cluster_custom_object.side_effect = [
            # version resolution calls (limit=1)
            {"items": []},  # VirtualService v1
            # actual list calls
            {"items": [
                {"metadata": {"name": "vs1", "namespace": "default"},
                 "spec": {"hosts": ["svc1"], "http": [{"route": [{"destination": {"host": "svc1"}}]}]}},
            ]},
            # DestinationRule version resolution
            {"items": []},
            # DestinationRule list
            {"items": [
                {"metadata": {"name": "dr1", "namespace": "default"},
                 "spec": {"host": "svc1", "subsets": [{"name": "v1", "labels": {"version": "v1"}}]}},
            ]},
            # Gateway version resolution
            {"items": []},
            # Gateway list
            {"items": [
                {"metadata": {"name": "gw1", "namespace": "istio-system"},
                 "spec": {"servers": [{"port": {"number": 443, "protocol": "HTTPS"},
                                       "hosts": ["*.example.com"]}]}},
            ]},
        ]

        result = get_mesh_config(gke_config=_mock_gke_config())

        assert result.error is False
        assert "Mesh Traffic Configuration" in result.output
        assert "VirtualServices (1)" in result.output
        assert "DestinationRules (1)" in result.output
        assert "Gateways (1)" in result.output
        assert "vs1" in result.output
        assert "dr1" in result.output
        assert "gw1" in result.output
        assert "Total resources: 3" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_config_no_crds_available(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # All version resolutions fail
        custom_api.list_cluster_custom_object.side_effect = _make_api_exception(404, "Not Found")

        result = get_mesh_config(gke_config=_mock_gke_config())

        assert result.error is False
        assert "CRD not available" in result.output
        assert "Total resources: 0" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_config_partial_failure(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        """One CRD type fails but others succeed — fail-open behavior."""
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        call_count = [0]

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            # First 3 calls: VirtualService version resolve (fail all 3 versions)
            if call_count[0] <= 3:
                raise _make_api_exception(404, "Not Found")
            # DestinationRule version resolve succeeds
            if call_count[0] == 4:
                return {"items": []}
            # DestinationRule list
            if call_count[0] == 5:
                return {"items": [
                    {"metadata": {"name": "dr1", "namespace": "default"},
                     "spec": {"host": "svc1"}},
                ]}
            # Gateway version resolve succeeds
            if call_count[0] == 6:
                return {"items": []}
            # Gateway list
            if call_count[0] == 7:
                return {"items": [
                    {"metadata": {"name": "gw1", "namespace": "default"},
                     "spec": {"servers": []}},
                ]}
            return {"items": []}

        custom_api.list_cluster_custom_object.side_effect = side_effect

        result = get_mesh_config(gke_config=_mock_gke_config())

        assert result.error is False
        assert "CRD not available" in result.output  # VirtualService failed
        assert "dr1" in result.output  # DestinationRule succeeded
        assert "gw1" in result.output  # Gateway succeeded

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_config_with_namespace_filter(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Version resolution
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        # Namespaced list
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {"metadata": {"name": "vs1", "namespace": "my-app"},
                 "spec": {"hosts": ["svc"], "http": []}},
            ],
        }

        result = get_mesh_config(gke_config=_mock_gke_config(), namespace="my-app")

        assert result.error is False
        assert "my-app" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    def test_config_k8s_client_failure(
        self, mock_create_clients: MagicMock,
    ) -> None:
        mock_create_clients.return_value = ToolResult(
            output="Failed to configure Kubernetes client", error=True,
        )
        result = get_mesh_config(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._cache._get_cached")
    def test_config_uses_cache(self, mock_get_cached: MagicMock) -> None:
        mock_get_cached.return_value = "cached mesh config"
        result = get_mesh_config(gke_config=_mock_gke_config())
        assert result.output == "cached mesh config"

    @patch("vaig.tools.gke.mesh._K8S_AVAILABLE", False)
    def test_config_k8s_unavailable(self) -> None:
        result = get_mesh_config(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_config_empty_resources(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # All CRDs resolve but return empty
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = get_mesh_config(gke_config=_mock_gke_config())

        assert result.error is False
        assert "(none)" in result.output
        assert "Total resources: 0" in result.output


# ── Phase 4: Security formatter tests ────────────────────────


class TestFormatPeerAuthentication:
    def test_strict_mtls(self) -> None:
        pa = {
            "metadata": {"name": "default", "namespace": "istio-system"},
            "spec": {"mtls": {"mode": "STRICT"}},
        }
        output = _format_peer_authentication(pa)
        assert "default (ns: istio-system)" in output
        assert "mTLS: STRICT" in output

    def test_permissive_with_port_override(self) -> None:
        pa = {
            "metadata": {"name": "my-pa", "namespace": "prod"},
            "spec": {
                "mtls": {"mode": "PERMISSIVE"},
                "portLevelMtls": {
                    "8080": {"mode": "STRICT"},
                    "9090": {"mode": "DISABLE"},
                },
            },
        }
        output = _format_peer_authentication(pa)
        assert "mTLS: PERMISSIVE" in output
        assert "Port overrides:" in output
        assert "8080=STRICT" in output
        assert "9090=DISABLE" in output

    def test_no_mtls_spec(self) -> None:
        pa = {
            "metadata": {"name": "empty", "namespace": "default"},
            "spec": {},
        }
        output = _format_peer_authentication(pa)
        assert "mTLS: UNSET" in output


class TestFormatAuthorizationPolicy:
    def test_allow_with_rules(self) -> None:
        ap = {
            "metadata": {"name": "allow-frontend", "namespace": "prod"},
            "spec": {
                "action": "ALLOW",
                "rules": [{
                    "from": [{"source": {
                        "principals": ["cluster.local/ns/prod/sa/frontend"],
                        "namespaces": ["prod"],
                    }}],
                    "to": [{"operation": {
                        "paths": ["/api/*"],
                        "methods": ["GET", "POST"],
                    }}],
                }],
            },
        }
        output = _format_authorization_policy(ap)
        assert "allow-frontend (ns: prod)" in output
        assert "Action: ALLOW" in output
        assert "principals:" in output
        assert "paths: /api/*" in output
        assert "methods: GET, POST" in output

    def test_deny_all(self) -> None:
        ap = {
            "metadata": {"name": "deny-all", "namespace": "default"},
            "spec": {"action": "DENY", "rules": []},
        }
        output = _format_authorization_policy(ap)
        assert "Action: DENY" in output
        assert "(deny-all)" in output

    def test_allow_all(self) -> None:
        ap = {
            "metadata": {"name": "allow-all", "namespace": "default"},
            "spec": {"action": "ALLOW"},
        }
        output = _format_authorization_policy(ap)
        assert "Action: ALLOW" in output
        assert "(allow-all)" in output

    def test_custom_action(self) -> None:
        ap = {
            "metadata": {"name": "ext-auth", "namespace": "default"},
            "spec": {"action": "CUSTOM", "rules": []},
        }
        output = _format_authorization_policy(ap)
        assert "Action: CUSTOM" in output

    def test_rule_with_conditions(self) -> None:
        ap = {
            "metadata": {"name": "cond-policy", "namespace": "default"},
            "spec": {
                "action": "ALLOW",
                "rules": [{
                    "when": [
                        {"key": "request.headers[x-token]", "values": ["abc"]},
                        {"key": "source.ip", "notValues": ["10.0.0.0/8"]},
                    ],
                }],
            },
        }
        output = _format_authorization_policy(ap)
        assert "conditions: 2" in output


class TestFormatRequestAuthentication:
    def test_single_jwt_rule(self) -> None:
        ra = {
            "metadata": {"name": "jwt-policy", "namespace": "prod"},
            "spec": {
                "jwtRules": [{
                    "issuer": "https://accounts.google.com",
                    "audiences": ["my-app.example.com"],
                }],
            },
        }
        output = _format_request_authentication(ra)
        assert "jwt-policy (ns: prod)" in output
        assert "JWT: issuer=https://accounts.google.com" in output
        assert "audiences=my-app.example.com" in output

    def test_no_jwt_rules(self) -> None:
        ra = {
            "metadata": {"name": "empty-jwt", "namespace": "default"},
            "spec": {"jwtRules": []},
        }
        output = _format_request_authentication(ra)
        assert "(no JWT rules configured)" in output

    def test_multiple_jwt_rules(self) -> None:
        ra = {
            "metadata": {"name": "multi-jwt", "namespace": "default"},
            "spec": {
                "jwtRules": [
                    {"issuer": "https://issuer1.com", "audiences": ["aud1"]},
                    {"issuer": "https://issuer2.com", "audiences": ["aud2", "aud3"]},
                ],
            },
        }
        output = _format_request_authentication(ra)
        assert "issuer=https://issuer1.com" in output
        assert "issuer=https://issuer2.com" in output
        assert "audiences=aud2, aud3" in output

    def test_jwt_no_audiences(self) -> None:
        ra = {
            "metadata": {"name": "no-aud", "namespace": "default"},
            "spec": {
                "jwtRules": [{"issuer": "https://example.com"}],
            },
        }
        output = _format_request_authentication(ra)
        assert "audiences=(any)" in output


# ── Phase 4: get_mesh_security tests ─────────────────────────


class TestGetMeshSecurity:
    @patch("vaig.tools.gke.mesh._K8S_AVAILABLE", False)
    def test_security_k8s_unavailable(self) -> None:
        result = get_mesh_security(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._cache._get_cached")
    def test_security_cache_hit(self, mock_get_cached: MagicMock) -> None:
        mock_get_cached.return_value = "cached mesh security"
        result = get_mesh_security(gke_config=_mock_gke_config())
        assert result.output == "cached mesh security"

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    def test_security_client_failure(self, mock_create_clients: MagicMock) -> None:
        mock_create_clients.return_value = ToolResult(
            output="Failed to configure Kubernetes client", error=True,
        )
        result = get_mesh_security(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_no_crds_available(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # All CRD version probes return 404
        custom_api.list_cluster_custom_object.side_effect = _make_api_exception(404, "Not Found")

        result = get_mesh_security(gke_config=_mock_gke_config())

        assert result.error is False
        assert result.output.count("(CRD not available)") == 3
        assert "Total resources: 0" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_peer_auth_and_authz_policy(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        call_count = [0]

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            # PeerAuthentication version resolve (v1 succeeds)
            if call_count[0] == 1:
                return {"items": []}
            # PeerAuthentication list
            if call_count[0] == 2:
                return {"items": [
                    {"metadata": {"name": "default-pa", "namespace": "istio-system"},
                     "spec": {"mtls": {"mode": "STRICT"}}},
                ]}
            # AuthorizationPolicy version resolve (v1 succeeds)
            if call_count[0] == 3:
                return {"items": []}
            # AuthorizationPolicy list
            if call_count[0] == 4:
                return {"items": [
                    {"metadata": {"name": "deny-all", "namespace": "prod"},
                     "spec": {"action": "DENY", "rules": []}},
                ]}
            # RequestAuthentication version resolve (v1 succeeds)
            if call_count[0] == 5:
                return {"items": []}
            # RequestAuthentication list
            if call_count[0] == 6:
                return {"items": []}
            return {"items": []}

        custom_api.list_cluster_custom_object.side_effect = side_effect

        result = get_mesh_security(gke_config=_mock_gke_config())

        assert result.error is False
        assert "Mesh Security Configuration" in result.output
        assert "PeerAuthentications (1)" in result.output
        assert "default-pa" in result.output
        assert "mTLS: STRICT" in result.output
        assert "AuthorizationPolicys (1)" in result.output
        assert "deny-all" in result.output
        assert "RequestAuthentications (0)" in result.output
        assert "Total resources: 2" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_partial_failure(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        """One CRD type fails but others succeed — fail-open behavior."""
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        call_count = [0]

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            # PeerAuthentication version resolve — fail all versions (v1, v1beta1)
            if call_count[0] <= 2:
                raise _make_api_exception(404, "Not Found")
            # AuthorizationPolicy version resolve (v1 succeeds)
            if call_count[0] == 3:
                return {"items": []}
            # AuthorizationPolicy list
            if call_count[0] == 4:
                return {"items": [
                    {"metadata": {"name": "my-ap", "namespace": "default"},
                     "spec": {"action": "ALLOW"}},
                ]}
            # RequestAuthentication version resolve (v1 succeeds)
            if call_count[0] == 5:
                return {"items": []}
            # RequestAuthentication list
            if call_count[0] == 6:
                return {"items": []}
            return {"items": []}

        custom_api.list_cluster_custom_object.side_effect = side_effect

        result = get_mesh_security(gke_config=_mock_gke_config())

        assert result.error is False
        assert "CRD not available" in result.output  # PeerAuth failed
        assert "my-ap" in result.output  # AuthzPolicy succeeded

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_namespace_filter(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Version resolution
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        # Namespaced list
        custom_api.list_namespaced_custom_object.return_value = {
            "items": [
                {"metadata": {"name": "pa1", "namespace": "prod"},
                 "spec": {"mtls": {"mode": "STRICT"}}},
            ],
        }

        result = get_mesh_security(gke_config=_mock_gke_config(), namespace="prod")

        assert result.error is False
        assert "prod" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_security_empty_resources(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # All CRDs resolve but return empty
        custom_api.list_cluster_custom_object.return_value = {"items": []}

        result = get_mesh_security(gke_config=_mock_gke_config())

        assert result.error is False
        assert "(none)" in result.output
        assert "Total resources: 0" in result.output


# ── Phase 5: Sidecar status tests ────────────────────────────


class TestFormatSidecarTable:
    def test_empty_rows(self) -> None:
        output = _format_sidecar_table([])
        assert "(no pods found)" in output

    def test_pod_with_sidecar(self) -> None:
        rows = [{
            "pod": "frontend-abc123",
            "namespace": "prod",
            "has_sidecar": True,
            "sidecar_version": "1.20.3",
            "owner": "ReplicaSet/frontend-abc",
            "anomaly": "",
        }]
        output = _format_sidecar_table(rows)
        assert "frontend-abc123" in output
        assert "prod" in output
        assert "yes" in output
        assert "1.20.3" in output
        assert "ReplicaSet/frontend-abc" in output

    def test_pod_without_sidecar_anomaly(self) -> None:
        rows = [{
            "pod": "backend-xyz",
            "namespace": "prod",
            "has_sidecar": False,
            "sidecar_version": "",
            "owner": "ReplicaSet/backend",
            "anomaly": "MISSING",
        }]
        output = _format_sidecar_table(rows)
        assert "no" in output
        assert "MISSING" in output

    def test_multiple_rows(self) -> None:
        rows = [
            {"pod": "pod-a", "namespace": "ns1", "has_sidecar": True,
             "sidecar_version": "1.20", "owner": "-", "anomaly": ""},
            {"pod": "pod-b", "namespace": "ns2", "has_sidecar": False,
             "sidecar_version": "", "owner": "-", "anomaly": "MISSING"},
        ]
        output = _format_sidecar_table(rows)
        assert "pod-a" in output
        assert "pod-b" in output
        assert "POD" in output  # Header present


def _make_mock_pod(
    name: str,
    namespace: str,
    containers: list[dict[str, str]],
    init_containers: list[dict[str, str]] | None = None,
    owner_kind: str = "",
    owner_name: str = "",
) -> MagicMock:
    """Create a mock pod object for sidecar status tests."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = namespace

    # Owner references
    if owner_kind and owner_name:
        ref = MagicMock()
        ref.kind = owner_kind
        ref.name = owner_name
        pod.metadata.owner_references = [ref]
    else:
        pod.metadata.owner_references = []

    # Containers
    mock_containers = []
    for c in containers:
        mc = MagicMock()
        mc.name = c["name"]
        mc.image = c.get("image", "")
        mock_containers.append(mc)
    pod.spec.containers = mock_containers

    # Init containers
    if init_containers:
        mock_init = []
        for c in init_containers:
            mc = MagicMock()
            mc.name = c["name"]
            mc.image = c.get("image", "")
            mock_init.append(mc)
        pod.spec.init_containers = mock_init
    else:
        pod.spec.init_containers = []

    return pod


def _make_mock_namespace(name: str, labels: dict[str, str] | None = None) -> MagicMock:
    """Create a mock namespace object."""
    ns = MagicMock()
    ns.metadata.name = name
    ns.metadata.labels = labels or {}
    return ns


class TestGetSidecarStatus:
    @patch("vaig.tools.gke.mesh._K8S_AVAILABLE", False)
    def test_k8s_unavailable(self) -> None:
        result = get_sidecar_status(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._cache._get_cached")
    def test_cache_hit(self, mock_get_cached: MagicMock) -> None:
        mock_get_cached.return_value = "cached sidecar status"
        result = get_sidecar_status(gke_config=_mock_gke_config())
        assert result.output == "cached sidecar status"

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    def test_client_failure(self, mock_create_clients: MagicMock) -> None:
        mock_create_clients.return_value = ToolResult(
            output="Failed to configure Kubernetes client", error=True,
        )
        result = get_sidecar_status(gke_config=_mock_gke_config())
        assert result.error is True

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_pods_with_sidecar(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Namespace with injection enabled
        core_v1.list_namespace.return_value.items = [
            _make_mock_namespace("prod", {"istio-injection": "enabled"}),
        ]

        # Pod with sidecar
        core_v1.list_pod_for_all_namespaces.return_value.items = [
            _make_mock_pod(
                "frontend-abc", "prod",
                containers=[
                    {"name": "app", "image": "myapp:1.0"},
                    {"name": "istio-proxy", "image": "istio/proxyv2:1.20.3"},
                ],
                owner_kind="ReplicaSet", owner_name="frontend",
            ),
        ]

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "Sidecar Injection Status" in result.output
        assert "frontend-abc" in result.output
        assert "yes" in result.output
        assert "1.20.3" in result.output
        assert "Pods with sidecar: 1/1" in result.output
        assert "Anomalies: none" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_missing_sidecar_anomaly(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Namespace with injection enabled
        core_v1.list_namespace.return_value.items = [
            _make_mock_namespace("prod", {"istio-injection": "enabled"}),
        ]

        # Pod WITHOUT sidecar in injection-enabled namespace
        core_v1.list_pod_for_all_namespaces.return_value.items = [
            _make_mock_pod(
                "backend-xyz", "prod",
                containers=[{"name": "app", "image": "myapp:1.0"}],
            ),
        ]

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "MISSING" in result.output
        assert "Anomalies: 1 (missing: 1, unexpected: 0)" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_unexpected_sidecar_anomaly(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        # Namespace WITHOUT injection enabled
        core_v1.list_namespace.return_value.items = [
            _make_mock_namespace("default", {}),
        ]

        # Pod WITH sidecar in non-injected namespace
        core_v1.list_pod_for_all_namespaces.return_value.items = [
            _make_mock_pod(
                "rogue-pod", "default",
                containers=[
                    {"name": "app", "image": "myapp:1.0"},
                    {"name": "istio-proxy", "image": "istio/proxyv2:1.20.3"},
                ],
            ),
        ]

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "UNEXPECTED" in result.output
        assert "Anomalies: 1 (missing: 0, unexpected: 1)" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_namespace_filter(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        core_v1.list_namespace.return_value.items = [
            _make_mock_namespace("prod", {"istio-injection": "enabled"}),
        ]

        core_v1.list_namespaced_pod.return_value.items = [
            _make_mock_pod(
                "web-abc", "prod",
                containers=[
                    {"name": "app", "image": "myapp:1.0"},
                    {"name": "istio-proxy", "image": "istio/proxyv2:1.20.3"},
                ],
            ),
        ]

        result = get_sidecar_status(gke_config=_mock_gke_config(), namespace="prod")

        assert result.error is False
        assert "prod" in result.output
        # Should use list_namespaced_pod, not list_pod_for_all_namespaces
        core_v1.list_namespaced_pod.assert_called_once_with("prod")

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_no_pods_found(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        core_v1.list_namespace.return_value.items = []
        core_v1.list_pod_for_all_namespaces.return_value.items = []

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "(no pods found)" in result.output
        assert "Pods with sidecar: 0/0" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_rbac_403_on_pods(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        core_v1.list_namespace.return_value.items = []
        core_v1.list_pod_for_all_namespaces.side_effect = _make_api_exception(403, "Forbidden")

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "RBAC: cannot list pods" in result.output

    @patch("vaig.tools.gke.mesh._clients._create_k8s_clients")
    @patch("vaig.tools.gke.mesh._cache._get_cached", return_value=None)
    @patch("vaig.tools.gke.mesh._cache._set_cache")
    def test_asm_managed_revision_detection(
        self, mock_set_cache: MagicMock, mock_get_cached: MagicMock,
        mock_create_clients: MagicMock,
    ) -> None:
        """Namespaces with istio.io/rev label should be injection-enabled."""
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()
        api_client = MagicMock()
        mock_create_clients.return_value = (core_v1, apps_v1, custom_api, api_client)

        core_v1.list_namespace.return_value.items = [
            _make_mock_namespace("managed-ns", {"istio.io/rev": "asm-managed-rapid"}),
        ]

        # Pod without sidecar in managed namespace → MISSING anomaly
        core_v1.list_pod_for_all_namespaces.return_value.items = [
            _make_mock_pod(
                "no-proxy-pod", "managed-ns",
                containers=[{"name": "app", "image": "myapp:1.0"}],
            ),
        ]

        result = get_sidecar_status(gke_config=_mock_gke_config())

        assert result.error is False
        assert "MISSING" in result.output
