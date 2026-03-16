"""Tests for Helm release introspection tools — helm.py."""

from __future__ import annotations

import base64
import gzip
import json
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs) -> GKEConfig:
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


def _encode_helm_release(release_dict: dict) -> str:
    """Encode a release dict the same way Helm does (for mock secrets).

    The kubernetes Python client already base64-decodes .data fields,
    so mock secrets should provide a value that is: base64(gzip(json)).
    That's what the K8s client would hand us after its own decode step.
    """
    json_bytes = json.dumps(release_dict).encode("utf-8")
    compressed = gzip.compress(json_bytes)
    return base64.b64encode(compressed).decode("ascii")


def _make_helm_release_data(
    name: str = "my-release",
    chart_name: str = "my-chart",
    chart_version: str = "1.0.0",
    app_version: str = "2.0.0",
    status: str = "deployed",
    revision: int = 1,
    description: str = "Install complete",
    config: dict | None = None,
    chart_values: dict | None = None,
    notes: str = "",
) -> dict:
    """Build a realistic Helm release data dict."""
    return {
        "name": name,
        "version": revision,
        "info": {
            "status": status,
            "first_deployed": "2025-01-01T00:00:00Z",
            "last_deployed": "2025-06-15T12:00:00Z",
            "description": description,
            "notes": notes,
        },
        "chart": {
            "metadata": {
                "name": chart_name,
                "version": chart_version,
                "appVersion": app_version,
            },
            "values": chart_values or {"replicaCount": 1, "image": {"tag": "latest"}},
        },
        "config": config if config is not None else {"replicaCount": 3},
    }


def _make_helm_secret(
    release_name: str = "my-release",
    revision: int = 1,
    status: str = "deployed",
    chart: str = "my-chart-1.0.0",
    app_version: str = "2.0.0",
    release_data: dict | None = None,
) -> MagicMock:
    """Create a mock K8s secret that looks like a Helm release secret."""
    secret = MagicMock()
    secret.metadata.name = f"sh.helm.release.v1.{release_name}.v{revision}"
    secret.metadata.namespace = "default"
    secret.metadata.labels = {
        "owner": "helm",
        "status": status,
        "chart": chart,
        "app_version": app_version,
        "version": str(revision),
        "name": release_name,
    }
    secret.type = "helm.sh/release.v1"

    # Encode release data
    if release_data is None:
        release_data = _make_helm_release_data(
            name=release_name,
            revision=revision,
            status=status,
            app_version=app_version,
        )
    secret.data = {"release": _encode_helm_release(release_data)}
    return secret


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear all caches before each test."""
    from vaig.tools.gke._cache import clear_discovery_cache
    from vaig.tools.gke._clients import clear_k8s_client_cache

    clear_k8s_client_cache()
    clear_discovery_cache()


# ── Test: _decode_helm_release ───────────────────────────────


class TestDecodeHelmRelease:
    """Test the Helm release decode pipeline."""

    def test_decode_helm_release(self) -> None:
        from vaig.tools.gke.helm import _decode_helm_release

        original = {
            "name": "test-release",
            "version": 1,
            "info": {"status": "deployed"},
            "config": {"key": "value"},
        }
        encoded = _encode_helm_release(original)
        decoded = _decode_helm_release(encoded)

        assert decoded["name"] == "test-release"
        assert decoded["version"] == 1
        assert decoded["info"]["status"] == "deployed"
        assert decoded["config"]["key"] == "value"

    def test_decode_roundtrip_complex_data(self) -> None:
        from vaig.tools.gke.helm import _decode_helm_release

        original = _make_helm_release_data(
            name="complex",
            chart_name="big-chart",
            chart_version="3.2.1",
            config={"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]},
        )
        encoded = _encode_helm_release(original)
        decoded = _decode_helm_release(encoded)

        assert decoded["config"]["nested"]["deep"]["value"] == 42
        assert decoded["config"]["list"] == [1, 2, 3]
        assert decoded["chart"]["metadata"]["version"] == "3.2.1"

    def test_decode_invalid_data_raises(self) -> None:
        from vaig.tools.gke.helm import _decode_helm_release

        with pytest.raises(Exception):
            _decode_helm_release("not-valid-base64!!!")


# ── Test: helm_list_releases ─────────────────────────────────


class TestHelmListReleases:
    """Tests for helm_list_releases function."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_list_releases(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        # Two releases, one with multiple revisions
        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("nginx", revision=1, status="superseded"),
            _make_helm_secret("nginx", revision=2, status="deployed"),
            _make_helm_secret("redis", revision=1, status="deployed", chart="redis-7.0.0"),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_list_releases(gke_config=cfg)

        assert result.error is False
        assert "nginx" in result.output
        assert "redis" in result.output
        assert "Total releases: 2" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_list_releases_empty(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = []
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_list_releases(gke_config=cfg)

        assert result.error is False
        assert "No Helm releases found" in result.output
        assert "Total releases: 0" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_list_releases_shows_latest_only(self, mock_clients: MagicMock) -> None:
        """Only the latest revision per release should appear."""
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("app", revision=1, status="superseded"),
            _make_helm_secret("app", revision=2, status="superseded"),
            _make_helm_secret("app", revision=3, status="deployed"),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_list_releases(gke_config=cfg)

        assert result.error is False
        assert "Total releases: 1" in result.output
        # The status should be from revision 3 (deployed)
        assert "deployed" in result.output


# ── Test: helm_release_status ────────────────────────────────


class TestHelmReleaseStatus:
    """Tests for helm_release_status function."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_status(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        release_data = _make_helm_release_data(
            name="nginx",
            chart_name="nginx",
            chart_version="15.0.0",
            app_version="1.25.0",
            status="deployed",
            revision=3,
            description="Upgrade complete",
            notes="Access at http://nginx.default.svc",
        )
        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("nginx", revision=3, status="deployed", release_data=release_data),
            _make_helm_secret("nginx", revision=2, status="superseded"),
            _make_helm_secret("nginx", revision=1, status="superseded"),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_status(gke_config=cfg, release_name="nginx")

        assert result.error is False
        assert "nginx" in result.output
        assert "deployed" in result.output
        assert "15.0.0" in result.output
        assert "1.25.0" in result.output
        assert "Upgrade complete" in result.output
        assert "http://nginx.default.svc" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_status_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_status

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = []
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_status(gke_config=cfg, release_name="nonexistent")

        assert result.error is True
        assert "not found" in result.output


# ── Test: helm_release_history ───────────────────────────────


class TestHelmReleaseHistory:
    """Tests for helm_release_history function."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_history(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_history

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret(
                "myapp", revision=1, status="superseded",
                release_data=_make_helm_release_data(
                    name="myapp", revision=1, status="superseded",
                    description="Install complete",
                ),
            ),
            _make_helm_secret(
                "myapp", revision=2, status="superseded",
                release_data=_make_helm_release_data(
                    name="myapp", revision=2, status="superseded",
                    description="Upgrade to v2",
                ),
            ),
            _make_helm_secret(
                "myapp", revision=3, status="deployed",
                release_data=_make_helm_release_data(
                    name="myapp", revision=3, status="deployed",
                    description="Upgrade to v3",
                ),
            ),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_history(gke_config=cfg, release_name="myapp")

        assert result.error is False
        assert "myapp" in result.output
        assert "Total revisions: 3" in result.output
        assert "Install complete" in result.output
        assert "Upgrade to v3" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_history_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_history

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = []
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_history(gke_config=cfg, release_name="ghost")

        assert result.error is True
        assert "not found" in result.output


# ── Test: helm_release_values ────────────────────────────────


class TestHelmReleaseValues:
    """Tests for helm_release_values function."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_values_overrides(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_values

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        release_data = _make_helm_release_data(
            name="myapp",
            config={"replicaCount": 5, "service": {"type": "LoadBalancer"}},
            chart_values={"replicaCount": 1, "image": {"tag": "latest"}},
        )
        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("myapp", revision=1, release_data=release_data),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_values(gke_config=cfg, release_name="myapp")

        assert result.error is False
        assert "user overrides" in result.output
        assert "replicaCount" in result.output
        assert "LoadBalancer" in result.output
        # Should NOT contain chart defaults when all_values=False
        # (image.tag is a chart default, not in user overrides)

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_values_all(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_values

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        release_data = _make_helm_release_data(
            name="myapp",
            config={"replicaCount": 5},
            chart_values={"replicaCount": 1, "image": {"tag": "latest"}},
        )
        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("myapp", revision=1, release_data=release_data),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_values(
                gke_config=cfg, release_name="myapp", all_values=True,
            )

        assert result.error is False
        assert "all (computed)" in result.output
        assert "replicaCount" in result.output
        # Chart defaults should be present when all_values=True
        assert "image" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_values_empty_overrides(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_values

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        release_data = _make_helm_release_data(
            name="myapp",
            config={},
            chart_values={"replicaCount": 1},
        )
        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("myapp", revision=1, release_data=release_data),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_values(gke_config=cfg, release_name="myapp")

        assert result.error is False
        assert "(no values)" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_helm_release_values_not_found(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_values

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = []
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_values(gke_config=cfg, release_name="ghost")

        assert result.error is True
        assert "not found" in result.output


# ── Test: k8s_not_available ──────────────────────────────────


class TestK8sNotAvailable:
    """Test graceful handling when kubernetes package is not installed."""

    def test_k8s_not_available_list(self) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = helm_list_releases(gke_config=cfg)
            assert result.error is True
            assert "kubernetes" in result.output.lower()

    def test_k8s_not_available_status(self) -> None:
        from vaig.tools.gke.helm import helm_release_status

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = helm_release_status(gke_config=cfg, release_name="test")
            assert result.error is True

    def test_k8s_not_available_history(self) -> None:
        from vaig.tools.gke.helm import helm_release_history

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = helm_release_history(gke_config=cfg, release_name="test")
            assert result.error is True

    def test_k8s_not_available_values(self) -> None:
        from vaig.tools.gke.helm import helm_release_values

        cfg = _make_gke_config()
        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_AVAILABLE", False), \
             patch("vaig.tools.gke._clients._K8S_IMPORT_ERROR", "kubernetes not installed"):
            result = helm_release_values(gke_config=cfg, release_name="test")
            assert result.error is True


# ── Test: cache hit ──────────────────────────────────────────


class TestCacheHit:
    """Test that cache is used on second call."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_cache_hit(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("cached-release", revision=1),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            # First call — should hit the K8s API
            result1 = helm_list_releases(gke_config=cfg)
            assert result1.error is False
            assert "cached-release" in result1.output

            # Second call — should use cache, NOT hit K8s API again
            result2 = helm_list_releases(gke_config=cfg)
            assert result2.error is False
            assert result2.output == result1.output

        # list_namespaced_secret should only be called once (second call used cache)
        assert core_v1.list_namespaced_secret.call_count == 1

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_force_refresh_bypasses_cache(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        core_v1 = MagicMock()
        mock_clients.return_value = (core_v1, MagicMock(), MagicMock(), MagicMock())

        secrets_list = MagicMock()
        secrets_list.items = [
            _make_helm_secret("refresh-test", revision=1),
        ]
        core_v1.list_namespaced_secret.return_value = secrets_list

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result1 = helm_list_releases(gke_config=cfg)
            result2 = helm_list_releases(gke_config=cfg, force_refresh=True)

            assert result1.output == result2.output

        # force_refresh should cause a second API call
        assert core_v1.list_namespaced_secret.call_count == 2


# ── Test: client creation error ──────────────────────────────


class TestClientCreationError:
    """Test that client creation errors are propagated correctly."""

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_error_list(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_list_releases

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Failed to configure", error=True)

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_list_releases(gke_config=cfg)

        assert result.error is True
        assert "Failed to configure" in result.output

    @patch("vaig.tools.gke._clients._create_k8s_clients")
    def test_client_error_status(self, mock_clients: MagicMock) -> None:
        from vaig.tools.gke.helm import helm_release_status

        cfg = _make_gke_config()
        mock_clients.return_value = ToolResult(output="Auth failed", error=True)

        with patch("vaig.tools.gke.helm._K8S_AVAILABLE", True):
            result = helm_release_status(gke_config=cfg, release_name="test")

        assert result.error is True
        assert "Auth failed" in result.output
