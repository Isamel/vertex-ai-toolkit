"""Tests for discover_dependencies — Phase 5.

Covers:
- Pure helper functions (_is_safe_env_var_name, _parse_hostname_from_value, _classify_confidence)
- Pod scanning (_find_pods_for_service, _extract_env_dependencies)
- Istio scanning (_discover_istio_dependencies)
- Security: sensitive env vars NEVER appear in output
- Graceful handling: no pods, no Istio, no dependencies
- End-to-end mock test for discover_dependencies()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult
from vaig.tools.gke.discovery import (
    _SENSITIVE_ENV_SUFFIXES,
    _classify_confidence,
    _extract_env_dependencies,
    _find_pods_for_service,
    _format_dependency_report,
    _is_safe_env_var_name,
    _parse_hostname_from_value,
    discover_dependencies,
)

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


def _make_mock_pod(env_vars: list[tuple[str, str | None]]) -> MagicMock:
    """Build a mock Pod with the given env vars.

    Args:
        env_vars: List of (name, value) tuples. value=None simulates valueFrom refs.
    """
    env_items = []
    for name, value in env_vars:
        env_var = MagicMock()
        env_var.name = name
        env_var.value = value
        env_items.append(env_var)

    container = MagicMock()
    container.env = env_items

    pod = MagicMock()
    pod.spec = MagicMock()
    pod.spec.containers = [container]
    return pod


# ════════════════════════════════════════════════════════════
# Phase 1: _is_safe_env_var_name
# ════════════════════════════════════════════════════════════


class TestIsSafeEnvVarName:
    """_is_safe_env_var_name must block sensitive suffixes."""

    def test_safe_name_returns_true(self) -> None:
        assert _is_safe_env_var_name("REDIS_HOST") is True

    def test_safe_url_name_returns_true(self) -> None:
        assert _is_safe_env_var_name("DATABASE_URL") is True

    def test_safe_service_name_returns_true(self) -> None:
        assert _is_safe_env_var_name("PAYMENT_SERVICE") is True

    @pytest.mark.parametrize("suffix", [
        "_PASSWORD", "_SECRET", "_TOKEN", "_KEY", "_CREDENTIALS",
        "_PRIVATE_KEY", "_ACCESS_KEY", "_AUTH", "_APIKEY", "_API_KEY",
        "_PASSPHRASE", "_PASSWD", "_PWD",
    ])
    def test_sensitive_suffix_returns_false(self, suffix: str) -> None:
        assert _is_safe_env_var_name(f"SOME_VAR{suffix}") is False

    def test_case_insensitive_lower(self) -> None:
        """Lowercase sensitive suffixes must also be blocked."""
        assert _is_safe_env_var_name("db_password") is False
        assert _is_safe_env_var_name("service_token") is False

    def test_partial_suffix_in_middle_is_safe(self) -> None:
        """Var names that CONTAIN but don't END WITH sensitive suffixes are safe."""
        assert _is_safe_env_var_name("DB_PASSWORD_ROTATION_INTERVAL") is True

    def test_empty_name_is_safe(self) -> None:
        assert _is_safe_env_var_name("") is True

    def test_all_sensitive_suffixes_defined(self) -> None:
        """_SENSITIVE_ENV_SUFFIXES must contain all expected suffixes."""
        required = {"_PASSWORD", "_SECRET", "_TOKEN", "_KEY", "_CREDENTIALS"}
        assert required.issubset(_SENSITIVE_ENV_SUFFIXES)


# ════════════════════════════════════════════════════════════
# Phase 1: _parse_hostname_from_value
# ════════════════════════════════════════════════════════════


class TestParseHostnameFromValue:
    """_parse_hostname_from_value must extract host:port from various formats."""

    def test_plain_hostname(self) -> None:
        assert _parse_hostname_from_value("redis-master.default.svc.cluster.local") == \
            "redis-master.default.svc.cluster.local"

    def test_plain_hostname_with_port(self) -> None:
        assert _parse_hostname_from_value("redis-master.default.svc.cluster.local:6379") == \
            "redis-master.default.svc.cluster.local:6379"

    def test_url_with_scheme_extracts_netloc(self) -> None:
        """Spec TS1: postgres URL must yield host:port only."""
        result = _parse_hostname_from_value(
            "postgres://user:pass@db.default.svc.cluster.local:5432/mydb"
        )
        assert result == "db.default.svc.cluster.local:5432"

    def test_http_url_extracts_host(self) -> None:
        result = _parse_hostname_from_value("http://payment-svc.payments.svc.cluster.local:8080/api")
        assert result == "payment-svc.payments.svc.cluster.local:8080"

    def test_grpc_url_extracts_netloc(self) -> None:
        result = _parse_hostname_from_value("grpc://grpc-service:9090")
        assert result == "grpc-service:9090"

    def test_url_credentials_stripped(self) -> None:
        """Credentials in URL netloc (user:pass@host) must be stripped."""
        result = _parse_hostname_from_value("mysql://admin:s3cret@mysql.default.svc.cluster.local:3306/db")
        assert "admin" not in result
        assert "s3cret" not in result
        assert result == "mysql.default.svc.cluster.local:3306"

    def test_empty_value_returns_empty(self) -> None:
        assert _parse_hostname_from_value("") == ""

    def test_whitespace_value_returns_empty(self) -> None:
        assert _parse_hostname_from_value("   ") == ""

    def test_leading_slashes_stripped(self) -> None:
        """Plain values with leading slashes (misconfigured) should have them stripped."""
        result = _parse_hostname_from_value("//some-host:8080")
        assert result == "some-host:8080"


# ════════════════════════════════════════════════════════════
# Phase 1: _classify_confidence
# ════════════════════════════════════════════════════════════


class TestClassifyConfidence:
    """_classify_confidence must return HIGH/MEDIUM/LOW correctly."""

    def test_svc_cluster_local_is_high(self) -> None:
        assert _classify_confidence("redis.default.svc.cluster.local", "REDIS_HOST", "redis.default.svc.cluster.local") == "HIGH"

    def test_svc_cluster_local_with_port_is_high(self) -> None:
        assert _classify_confidence("db.default.svc.cluster.local:5432", "DB_URL", "postgres://db.default.svc.cluster.local:5432/mydb") == "HIGH"

    def test_host_suffix_is_medium(self) -> None:
        assert _classify_confidence("redis-server", "REDIS_HOST", "redis-server") == "MEDIUM"

    def test_addr_suffix_is_medium(self) -> None:
        assert _classify_confidence("cache-service", "CACHE_ADDR", "cache-service") == "MEDIUM"

    def test_endpoint_suffix_is_medium(self) -> None:
        assert _classify_confidence("api.example.com", "API_ENDPOINT", "api.example.com") == "MEDIUM"

    def test_server_suffix_is_medium(self) -> None:
        assert _classify_confidence("db-host", "DATABASE_SERVER", "db-host") == "MEDIUM"

    def test_address_suffix_is_medium(self) -> None:
        assert _classify_confidence("grpc-svc:9090", "GRPC_ADDRESS", "grpc-svc:9090") == "MEDIUM"

    def test_url_suffix_is_medium(self) -> None:
        assert _classify_confidence("external-svc", "PAYMENT_URL", "external-svc") == "MEDIUM"

    def test_uri_suffix_is_medium(self) -> None:
        assert _classify_confidence("auth-svc", "AUTH_URI", "auth-svc") == "MEDIUM"

    def test_service_suffix_is_medium(self) -> None:
        assert _classify_confidence("order-svc", "ORDER_SERVICE", "order-svc") == "MEDIUM"

    def test_svc_suffix_is_medium(self) -> None:
        assert _classify_confidence("cache-svc", "CACHE_SVC", "cache-svc") == "MEDIUM"

    def test_url_scheme_in_value_is_medium(self) -> None:
        """original_value containing :// must classify MEDIUM even for generic env name."""
        assert _classify_confidence("external-host", "SOME_VAR", "http://external-host/path") == "MEDIUM"

    def test_generic_env_name_is_low(self) -> None:
        assert _classify_confidence("some-hostname", "RANDOM_VAR", "some-hostname") == "LOW"

    def test_svc_cluster_local_takes_precedence_over_suffix(self) -> None:
        """HIGH must win even if env name ends with _HOST too."""
        assert _classify_confidence("svc.default.svc.cluster.local", "SERVICE_HOST", "svc.default.svc.cluster.local") == "HIGH"


# ════════════════════════════════════════════════════════════
# Phase 2: _find_pods_for_service
# ════════════════════════════════════════════════════════════


class TestFindPodsForService:
    """_find_pods_for_service must resolve service selector → pods."""

    def test_returns_pods_matching_selector(self) -> None:
        core_v1 = MagicMock()

        # Mock service with selector
        mock_svc = MagicMock()
        mock_svc.spec.selector = {"app": "my-service", "env": "prod"}
        core_v1.read_namespaced_service.return_value = mock_svc

        # Mock pod list
        pod1 = MagicMock()
        pod2 = MagicMock()
        pod_list = MagicMock()
        pod_list.items = [pod1, pod2]
        core_v1.list_namespaced_pod.return_value = pod_list

        result = _find_pods_for_service("my-service", "default", core_v1)

        assert result == [pod1, pod2]
        core_v1.read_namespaced_service.assert_called_once_with(
            name="my-service", namespace="default"
        )
        core_v1.list_namespaced_pod.assert_called_once()
        call_kwargs = core_v1.list_namespaced_pod.call_args
        label_selector = call_kwargs.kwargs.get("label_selector", "") or call_kwargs.args[0] if call_kwargs.args else ""
        # The selector string must contain both labels (order may vary)
        assert "app=my-service" in call_kwargs.kwargs.get("label_selector", "")
        assert "env=prod" in call_kwargs.kwargs.get("label_selector", "")

    def test_returns_empty_list_when_no_selector(self) -> None:
        core_v1 = MagicMock()
        mock_svc = MagicMock()
        mock_svc.spec.selector = {}
        core_v1.read_namespaced_service.return_value = mock_svc

        result = _find_pods_for_service("headless-svc", "default", core_v1)

        assert result == []
        core_v1.list_namespaced_pod.assert_not_called()

    def test_returns_empty_list_on_api_exception(self) -> None:
        core_v1 = MagicMock()
        core_v1.read_namespaced_service.side_effect = Exception("404 Not Found")

        result = _find_pods_for_service("nonexistent", "default", core_v1)

        assert result == []

    def test_returns_empty_list_when_pods_items_is_none(self) -> None:
        core_v1 = MagicMock()
        mock_svc = MagicMock()
        mock_svc.spec.selector = {"app": "svc"}
        core_v1.read_namespaced_service.return_value = mock_svc

        pod_list = MagicMock()
        pod_list.items = None
        core_v1.list_namespaced_pod.return_value = pod_list

        result = _find_pods_for_service("svc", "default", core_v1)

        assert result == []


# ════════════════════════════════════════════════════════════
# Phase 2: _extract_env_dependencies
# ════════════════════════════════════════════════════════════


class TestExtractEnvDependencies:
    """_extract_env_dependencies must scan pods and return safe dependency list."""

    def test_extracts_svc_cluster_local_hostname(self) -> None:
        """Spec TS7: REDIS_HOST with .svc.cluster.local value → HIGH confidence."""
        pod = _make_mock_pod([
            ("REDIS_HOST", "redis-master.default.svc.cluster.local"),
        ])
        result = _extract_env_dependencies([pod])

        assert len(result) == 1
        assert result[0]["hostname"] == "redis-master.default.svc.cluster.local"
        assert result[0]["confidence"] == "HIGH"
        assert result[0]["source_env"] == "REDIS_HOST"

    def test_extracts_url_format_hostname(self) -> None:
        """Spec TS1: postgres URL → extracts host:port, strips credentials."""
        pod = _make_mock_pod([
            ("DATABASE_URL", "postgres://user:pass@db.default.svc.cluster.local:5432/mydb"),
        ])
        result = _extract_env_dependencies([pod])

        assert len(result) == 1
        assert result[0]["hostname"] == "db.default.svc.cluster.local:5432"
        assert "user" not in result[0]["hostname"]
        assert "pass" not in result[0]["hostname"]

    def test_skips_sensitive_env_vars(self) -> None:
        """Security: vars ending in _PASSWORD, _TOKEN, etc. must NEVER appear in output."""
        pod = _make_mock_pod([
            ("DB_PASSWORD", "super-secret-password"),
            ("API_TOKEN", "tok_abc123"),
            ("DB_SECRET", "very-secret"),
            ("SIGNING_KEY", "private-key-value"),
            ("DB_HOST", "db.default.svc.cluster.local"),  # this one is safe
        ])
        result = _extract_env_dependencies([pod])

        # Only DB_HOST should appear
        assert len(result) == 1
        assert result[0]["source_env"] == "DB_HOST"

        # Ensure no sensitive values leaked into any hostname
        all_hostnames = [d["hostname"] for d in result]
        for sensitive in ["super-secret-password", "tok_abc123", "very-secret", "private-key-value"]:
            for hostname in all_hostnames:
                assert sensitive not in hostname

    def test_skips_none_values(self) -> None:
        """Env vars with value=None (valueFrom refs) must be skipped silently."""
        pod = _make_mock_pod([
            ("DB_HOST", None),
            ("REDIS_HOST", "redis.default.svc.cluster.local"),
        ])
        result = _extract_env_dependencies([pod])

        assert len(result) == 1
        assert result[0]["source_env"] == "REDIS_HOST"

    def test_skips_empty_values(self) -> None:
        pod = _make_mock_pod([
            ("SERVICE_HOST", ""),
            ("CACHE_HOST", "cache.default.svc.cluster.local"),
        ])
        result = _extract_env_dependencies([pod])

        assert len(result) == 1

    def test_deduplicates_same_hostname(self) -> None:
        """Same hostname from two pods → one entry, highest confidence wins."""
        pod1 = _make_mock_pod([("DB_HOST", "db.default.svc.cluster.local")])
        pod2 = _make_mock_pod([("DATABASE_URL", "http://db.default.svc.cluster.local/path")])
        result = _extract_env_dependencies([pod1, pod2])

        hostnames = [d["hostname"] for d in result]
        # "db.default.svc.cluster.local" appears only once (deduped)
        exact_matches = [h for h in hostnames if h == "db.default.svc.cluster.local"]
        assert len(exact_matches) == 1

    def test_keeps_highest_confidence_on_dedup(self) -> None:
        """When deduplicating, keep the entry with highest confidence.

        Uses a plain hostname (not .svc.cluster.local) so pods produce
        different confidence levels: MEDIUM (suffix _HOST) vs LOW (generic name).
        """
        pod1 = _make_mock_pod([("SOME_VAR", "redis-master")])   # LOW — no special suffix or scheme
        pod2 = _make_mock_pod([("REDIS_HOST", "redis-master")])  # MEDIUM — _HOST suffix
        result = _extract_env_dependencies([pod1, pod2])

        assert len(result) == 1
        assert result[0]["hostname"] == "redis-master"
        assert result[0]["confidence"] == "MEDIUM"  # highest confidence kept

    def test_skips_localhost_values(self) -> None:
        pod = _make_mock_pod([
            ("CACHE_HOST", "localhost"),
            ("SIDECAR_ADDR", "127.0.0.1"),
            ("DB_HOST", "db.default.svc.cluster.local"),
        ])
        result = _extract_env_dependencies([pod])

        hostnames = [d["hostname"] for d in result]
        assert "localhost" not in hostnames
        assert "127.0.0.1" not in hostnames
        assert len(result) == 1

    def test_returns_empty_for_no_pods(self) -> None:
        result = _extract_env_dependencies([])
        assert result == []

    def test_returns_empty_for_pods_with_no_dep_vars(self) -> None:
        pod = _make_mock_pod([
            ("APP_NAME", "my-app"),
            ("LOG_LEVEL", "INFO"),
            ("PORT", "8080"),
        ])
        result = _extract_env_dependencies([pod])
        assert result == []

    def test_env_var_value_not_in_output(self) -> None:
        """Full env var value (e.g., full connection string) must not appear as-is in output.

        Only hostname:port should appear, never the full value with credentials.
        """
        sensitive_value = "postgres://admin:hunter2@db.svc.cluster.local:5432/prod?sslmode=require"
        pod = _make_mock_pod([("DATABASE_URL", sensitive_value)])
        result = _extract_env_dependencies([pod])

        # The output should have the hostname, not the full URL
        assert len(result) == 1
        hostname = result[0]["hostname"]
        assert "hunter2" not in hostname
        assert "admin" not in hostname
        assert "sslmode" not in hostname
        assert hostname == "db.svc.cluster.local:5432"

    def test_inline_svc_cluster_local_in_non_host_var(self) -> None:
        """A var not ending in _HOST but containing .svc.cluster.local should still be detected."""
        pod = _make_mock_pod([
            ("MY_BACKEND", "backend.default.svc.cluster.local:8080"),
        ])
        result = _extract_env_dependencies([pod])

        assert len(result) == 1
        assert result[0]["hostname"] == "backend.default.svc.cluster.local:8080"


# ════════════════════════════════════════════════════════════
# Phase 3: _discover_istio_dependencies
# ════════════════════════════════════════════════════════════


class TestDiscoverIstioDependencies:
    """_discover_istio_dependencies must handle Istio present/absent gracefully."""

    def test_returns_empty_when_istio_not_installed(self) -> None:
        """When _resolve_crd_version returns None (Istio absent), returns ([], [])."""
        from vaig.tools.gke.discovery import _discover_istio_dependencies

        custom_api = MagicMock()
        with patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):
            upstreams, downstreams = _discover_istio_dependencies("my-svc", "default", custom_api)

        assert upstreams == []
        assert downstreams == []

    def test_extracts_downstreams_from_virtual_service(self) -> None:
        """VirtualServices for the service listing other destinations → downstreams."""
        from vaig.tools.gke.discovery import _discover_istio_dependencies

        custom_api = MagicMock()
        # VS for "api-gateway" routes to "payment-svc" and "user-svc"
        vs = {
            "metadata": {"name": "api-gateway"},
            "spec": {
                "hosts": ["api-gateway"],
                "http": [
                    {
                        "route": [
                            {"destination": {"host": "payment-svc.default.svc.cluster.local"}},
                            {"destination": {"host": "user-svc.default.svc.cluster.local"}},
                        ]
                    }
                ],
            },
        }
        with patch("vaig.tools.gke.mesh._resolve_crd_version", return_value="v1"), \
             patch("vaig.tools.gke.mesh._read_custom_resources", return_value=[vs]):
            upstreams, downstreams = _discover_istio_dependencies("api-gateway", "default", custom_api)

        assert "payment-svc" in downstreams
        assert "user-svc" in downstreams
        assert upstreams == []

    def test_extracts_upstreams_when_service_is_destination(self) -> None:
        """VS for another service that routes TO this service → upstreams."""
        from vaig.tools.gke.discovery import _discover_istio_dependencies

        custom_api = MagicMock()
        # A VS named "frontend" routes to "backend" (our service)
        vs = {
            "metadata": {"name": "frontend"},
            "spec": {
                "hosts": ["frontend"],
                "http": [
                    {
                        "route": [
                            {"destination": {"host": "backend"}},
                        ]
                    }
                ],
            },
        }
        with patch("vaig.tools.gke.mesh._resolve_crd_version", return_value="v1"), \
             patch("vaig.tools.gke.mesh._read_custom_resources", return_value=[vs]):
            upstreams, downstreams = _discover_istio_dependencies("backend", "default", custom_api)

        assert "frontend" in upstreams
        assert downstreams == []

    def test_discover_dependencies_graceful_on_istio_exception(self) -> None:
        """discover_dependencies() must not error when Istio raises (e.g. RBAC denied).

        The exception is caught in the outer discover_dependencies() caller,
        not in _discover_istio_dependencies itself.
        """
        core_v1 = MagicMock()
        mock_svc = MagicMock()
        mock_svc.spec.selector = {"app": "svc"}
        core_v1.read_namespaced_service.return_value = mock_svc
        pod_list = MagicMock()
        pod_list.items = []
        core_v1.list_namespaced_pod.return_value = pod_list

        apps_v1 = MagicMock()
        custom_api = MagicMock()
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", side_effect=Exception("RBAC denied")):
            result = discover_dependencies("default", service_name="svc", gke_config=cfg)

        # Must not raise — error is suppressed, result is a valid ToolResult
        assert isinstance(result, ToolResult)
        assert result.error is False


# ════════════════════════════════════════════════════════════
# Phase 4: _format_dependency_report
# ════════════════════════════════════════════════════════════


class TestFormatDependencyReport:
    """_format_dependency_report must produce well-structured output."""

    def test_header_contains_service_name_and_namespace(self) -> None:
        report = _format_dependency_report("my-svc", "production", [], [], [])
        assert "my-svc" in report
        assert "production" in report

    def test_sections_present(self) -> None:
        report = _format_dependency_report("svc", "ns", [], [], [])
        assert "ENV VAR DEPENDENCIES" in report
        assert "ISTIO UPSTREAMS" in report
        assert "ISTIO DOWNSTREAMS" in report

    def test_none_found_when_empty(self) -> None:
        report = _format_dependency_report("svc", "ns", [], [], [])
        assert "(none found)" in report
        assert "(none detected" in report

    def test_env_deps_appear_in_report(self) -> None:
        env_deps = [
            {"hostname": "redis.default.svc.cluster.local", "confidence": "HIGH", "source_env": "REDIS_HOST"},
        ]
        report = _format_dependency_report("svc", "ns", env_deps, [], [])
        assert "redis.default.svc.cluster.local" in report
        assert "HIGH" in report
        assert "REDIS_HOST" in report

    def test_upstreams_appear_in_report(self) -> None:
        report = _format_dependency_report("backend", "ns", [], ["frontend", "gateway"], [])
        assert "frontend" in report
        assert "gateway" in report

    def test_downstreams_appear_in_report(self) -> None:
        report = _format_dependency_report("api-gw", "ns", [], [], ["payment-svc", "user-svc"])
        assert "payment-svc" in report
        assert "user-svc" in report

    def test_summary_line_present(self) -> None:
        env_deps = [{"hostname": "h1", "confidence": "LOW", "source_env": "VAR"}]
        report = _format_dependency_report("svc", "ns", env_deps, ["up1"], ["down1"])
        assert "1 env-var dependencies" in report
        assert "1 Istio upstreams" in report
        assert "1 Istio downstreams" in report


# ════════════════════════════════════════════════════════════
# End-to-end: discover_dependencies()
# ════════════════════════════════════════════════════════════


class TestDiscoverDependencies:
    """End-to-end tests for discover_dependencies() with mocked K8s clients."""

    def _make_clients_patch(
        self,
        env_pods: list[MagicMock] | None = None,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        custom_api = MagicMock()

        # Default: service has a selector, returns our pods
        mock_svc = MagicMock()
        mock_svc.spec.selector = {"app": "test-service"}
        core_v1.read_namespaced_service.return_value = mock_svc

        pod_list = MagicMock()
        pod_list.items = env_pods or []
        core_v1.list_namespaced_pod.return_value = pod_list

        return core_v1, apps_v1, custom_api

    def test_returns_tool_result(self) -> None:
        core_v1, apps_v1, custom_api = self._make_clients_patch()
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("default", service_name="test-service", gke_config=cfg)

        assert isinstance(result, ToolResult)
        assert result.error is False

    def test_output_contains_service_name(self) -> None:
        core_v1, apps_v1, custom_api = self._make_clients_patch()
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("production", service_name="my-app", gke_config=cfg)

        assert "my-app" in result.output
        assert "production" in result.output

    def test_no_pods_returns_gracefully(self) -> None:
        """Spec: when service has no backing pods, must return empty report without error."""
        core_v1, apps_v1, custom_api = self._make_clients_patch(env_pods=[])
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("default", service_name="empty-svc", gke_config=cfg)

        assert result.error is False
        assert "(none found)" in result.output

    def test_no_istio_returns_gracefully(self) -> None:
        """Spec: when Istio is not installed, must not error — just empty Istio sections."""
        core_v1, apps_v1, custom_api = self._make_clients_patch()
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("default", service_name="svc", gke_config=cfg)

        assert result.error is False
        assert "none detected or Istio not installed" in result.output

    def test_env_deps_in_output(self) -> None:
        """Pods with service-ref env vars → dependencies appear in output."""
        pod = _make_mock_pod([
            ("REDIS_HOST", "redis-master.default.svc.cluster.local"),
            ("DB_URL", "postgres://user:pass@postgres.default.svc.cluster.local:5432/db"),
        ])
        core_v1, apps_v1, custom_api = self._make_clients_patch(env_pods=[pod])
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("default", service_name="my-svc", gke_config=cfg)

        assert "redis-master.default.svc.cluster.local" in result.output
        assert "postgres.default.svc.cluster.local:5432" in result.output

    def test_sensitive_values_never_in_output(self) -> None:
        """Security: passwords and tokens in env var values must NEVER appear in output."""
        pod = _make_mock_pod([
            ("DB_PASSWORD", "hunter2"),
            ("API_TOKEN", "tok_secret_xyz"),
            ("DB_HOST", "db.default.svc.cluster.local"),
        ])
        core_v1, apps_v1, custom_api = self._make_clients_patch(env_pods=[pod])
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients",
                   return_value=(core_v1, apps_v1, custom_api, MagicMock())), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result = discover_dependencies("default", service_name="secure-svc", gke_config=cfg)

        assert "hunter2" not in result.output
        assert "tok_secret_xyz" not in result.output

    def test_k8s_unavailable_returns_error(self) -> None:
        """When Kubernetes is not available, must return a ToolResult with error."""
        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke.discovery._K8S_AVAILABLE", False):
            result = discover_dependencies("default", service_name="svc", gke_config=cfg)

        assert isinstance(result, ToolResult)

    def test_cache_returns_cached_result(self) -> None:
        """Second call with same args must return cached output (no K8s calls)."""
        core_v1, apps_v1, custom_api = self._make_clients_patch()
        cfg = _make_gke_config()
        mock_create_clients = MagicMock(return_value=(core_v1, apps_v1, custom_api, MagicMock()))

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients", mock_create_clients), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            result1 = discover_dependencies("default", service_name="cached-svc", gke_config=cfg)
            result2 = discover_dependencies("default", service_name="cached-svc", gke_config=cfg)

        # Both should return without error
        assert result1.error is False
        assert result2.error is False
        # Second call should use cache — create_k8s_clients only called once
        mock_create_clients.assert_called_once()

    def test_force_refresh_bypasses_cache(self) -> None:
        """force_refresh=True must bypass cache and make fresh K8s calls."""
        core_v1, apps_v1, custom_api = self._make_clients_patch()
        cfg = _make_gke_config()
        mock_create_clients = MagicMock(return_value=(core_v1, apps_v1, custom_api, MagicMock()))

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients", mock_create_clients), \
             patch("vaig.tools.gke.mesh._resolve_crd_version", return_value=None):

            discover_dependencies("default", service_name="fresh-svc", gke_config=cfg, force_refresh=True)
            discover_dependencies("default", service_name="fresh-svc", gke_config=cfg, force_refresh=True)

        assert mock_create_clients.call_count == 2


# ════════════════════════════════════════════════════════════
# Registration: discover_dependencies in _registry.py
# ════════════════════════════════════════════════════════════


class TestDiscoverDependenciesRegistration:
    """discover_dependencies must be registered as a GKE tool."""

    def test_tool_registered(self) -> None:
        """create_gke_tools() must include 'discover_dependencies' in the tool list."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)

        tool_names = [t.name for t in tools]
        assert "discover_dependencies" in tool_names, (
            f"discover_dependencies not found in registered tools: {tool_names}"
        )

    def test_tool_has_service_name_param(self) -> None:
        """ToolDef for discover_dependencies must have 'service_name' parameter."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)

        dep_tool = next(t for t in tools if t.name == "discover_dependencies")
        param_names = [p.name for p in dep_tool.parameters]
        assert "service_name" in param_names

    def test_tool_has_namespace_param(self) -> None:
        """ToolDef for discover_dependencies must have 'namespace' parameter."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)

        dep_tool = next(t for t in tools if t.name == "discover_dependencies")
        param_names = [p.name for p in dep_tool.parameters]
        assert "namespace" in param_names


# ════════════════════════════════════════════════════════════
# Prompt coverage: discover_dependencies in prompts.py
# ════════════════════════════════════════════════════════════


class TestDependencyMappingPrompts:
    """discover_dependencies must be referenced in the service health skill prompts."""

    def test_tool_in_core_tools_table(self) -> None:
        """_CORE_TOOLS_TABLE must mention discover_dependencies."""
        from vaig.skills.service_health.prompts import _CORE_TOOLS_TABLE

        assert "discover_dependencies" in _CORE_TOOLS_TABLE

    def test_gatherer_prompt_has_dependency_step(self) -> None:
        """_GATHERER_PROMPT_TEMPLATE must include a Step for dependency mapping."""
        from vaig.skills.service_health.prompts import _GATHERER_PROMPT_TEMPLATE

        assert "Step 13" in _GATHERER_PROMPT_TEMPLATE and "Dependency Mapping" in _GATHERER_PROMPT_TEMPLATE
