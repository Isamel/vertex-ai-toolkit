"""Tests for dual-auth impersonation — get_gke_credentials, credential threading, config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import AuthConfig, AuthMode, GKEConfig, Settings

# _reset_settings is provided by conftest.py (autouse)


# ══════════════════════════════════════════════════════════════
# 4.1 — get_gke_credentials() fallback chain
# ══════════════════════════════════════════════════════════════


class TestGetGkeCredentials:
    """Test the 3-tier fallback chain in get_gke_credentials."""

    def _make_settings(
        self,
        *,
        gke_sa: str = "",
        auth_mode: str = "adc",
        auth_sa: str = "",
    ) -> Settings:
        """Build minimal Settings for testing credential resolution."""
        return Settings(
            gke=GKEConfig(impersonate_sa=gke_sa),
            auth=AuthConfig(mode=AuthMode(auth_mode), impersonate_sa=auth_sa),
        )

    @patch("vaig.core.auth._get_impersonated_credentials")
    def test_gke_sa_set_uses_gke_sa(self, mock_imp: MagicMock) -> None:
        """When gke.impersonate_sa is set, it should be used."""
        from vaig.core.auth import get_gke_credentials

        mock_imp.return_value = MagicMock(name="gke-creds")
        settings = self._make_settings(gke_sa="gke-sa@proj.iam.gserviceaccount.com")

        result = get_gke_credentials(settings)

        mock_imp.assert_called_once_with("gke-sa@proj.iam.gserviceaccount.com")
        assert result is not None

    @patch("vaig.core.auth._get_impersonated_credentials")
    def test_auth_mode_impersonate_falls_back(self, mock_imp: MagicMock) -> None:
        """When gke.impersonate_sa empty + auth.mode=impersonate, use auth SA."""
        from vaig.core.auth import get_gke_credentials

        mock_imp.return_value = MagicMock(name="auth-creds")
        settings = self._make_settings(
            auth_mode="impersonate",
            auth_sa="vertex-sa@proj.iam.gserviceaccount.com",
        )

        result = get_gke_credentials(settings)

        mock_imp.assert_called_once_with("vertex-sa@proj.iam.gserviceaccount.com")
        assert result is not None

    def test_adc_mode_returns_none(self) -> None:
        """When gke.impersonate_sa empty + auth.mode=adc, return None."""
        from vaig.core.auth import get_gke_credentials

        settings = self._make_settings(auth_mode="adc")

        result = get_gke_credentials(settings)

        assert result is None

    @patch("vaig.core.auth._get_impersonated_credentials")
    def test_gke_sa_wins_over_auth_sa(self, mock_imp: MagicMock) -> None:
        """When both gke.impersonate_sa and auth.impersonate_sa are set, gke wins."""
        from vaig.core.auth import get_gke_credentials

        mock_imp.return_value = MagicMock(name="gke-creds")
        settings = self._make_settings(
            gke_sa="gke-sa@proj.iam.gserviceaccount.com",
            auth_mode="impersonate",
            auth_sa="vertex-sa@proj.iam.gserviceaccount.com",
        )

        result = get_gke_credentials(settings)

        # gke.impersonate_sa should win — called with gke SA, not auth SA
        mock_imp.assert_called_once_with("gke-sa@proj.iam.gserviceaccount.com")
        assert result is not None

    def test_auth_mode_impersonate_no_sa_returns_none(self) -> None:
        """auth.mode=impersonate but auth.impersonate_sa empty → None."""
        from vaig.core.auth import get_gke_credentials

        settings = self._make_settings(auth_mode="impersonate", auth_sa="")

        result = get_gke_credentials(settings)

        assert result is None


# ══════════════════════════════════════════════════════════════
# 4.2 — _get_impersonated_credentials with explicit scopes
# ══════════════════════════════════════════════════════════════


class TestGetImpersonatedCredentials:
    """Test _get_impersonated_credentials with scopes parameter."""

    @patch("vaig.core.auth.google.auth.default")
    @patch("vaig.core.auth.impersonated_credentials.Credentials")
    def test_default_scopes(self, mock_cred_class: MagicMock, mock_default: MagicMock) -> None:
        """Without explicit scopes, should use _VERTEX_SCOPES."""
        from vaig.core.auth import _VERTEX_SCOPES, _get_impersonated_credentials

        mock_default.return_value = (MagicMock(), "proj")

        _get_impersonated_credentials("sa@proj.iam.gserviceaccount.com")

        mock_cred_class.assert_called_once()
        call_kwargs = mock_cred_class.call_args
        assert call_kwargs.kwargs["target_scopes"] == _VERTEX_SCOPES

    @patch("vaig.core.auth.google.auth.default")
    @patch("vaig.core.auth.impersonated_credentials.Credentials")
    def test_custom_scopes(self, mock_cred_class: MagicMock, mock_default: MagicMock) -> None:
        """With explicit scopes, those should be used instead of defaults."""
        from vaig.core.auth import _get_impersonated_credentials

        mock_default.return_value = (MagicMock(), "proj")
        custom_scopes = ["https://www.googleapis.com/auth/logging.read"]

        _get_impersonated_credentials("sa@proj.iam.gserviceaccount.com", scopes=custom_scopes)

        mock_cred_class.assert_called_once()
        call_kwargs = mock_cred_class.call_args
        assert call_kwargs.kwargs["target_scopes"] == custom_scopes

    def test_empty_sa_raises(self) -> None:
        """Empty target SA should raise ValueError."""
        from vaig.core.auth import _get_impersonated_credentials

        with pytest.raises(ValueError, match="VAIG_IMPERSONATE_SA"):
            _get_impersonated_credentials("")


# ══════════════════════════════════════════════════════════════
# 4.3 — Credential threading in create_gcloud_tools
# ══════════════════════════════════════════════════════════════


class TestCreateGcloudToolsCredentials:
    """Test that credentials are properly threaded through create_gcloud_tools."""

    def test_credentials_none_by_default(self) -> None:
        """create_gcloud_tools with no credentials arg should work (backward compat)."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools(project="test-proj")
        assert len(tools) == 2

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_logging_tool_passes_credentials(self, mock_client_fn: MagicMock) -> None:
        """Logging tool lambda should pass credentials to _get_logging_client."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        mock_creds = MagicMock(name="creds")
        mock_client = MagicMock()
        mock_client.list_entries.return_value = []
        mock_client_fn.return_value = (mock_client, None)

        tools = create_gcloud_tools(project="proj", credentials=mock_creds)
        logging_tool = next(t for t in tools if t.name == "gcloud_logging_query")

        # Execute the tool — it should pass credentials through
        logging_tool.execute("severity>=ERROR")

        # _get_logging_client should have been called with credentials
        mock_client_fn.assert_called_once()
        call_kwargs = mock_client_fn.call_args
        assert call_kwargs.kwargs.get("credentials") is mock_creds or call_kwargs[1].get("credentials") is mock_creds

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_monitoring_tool_passes_credentials(self, mock_client_fn: MagicMock) -> None:
        """Monitoring tool lambda should pass credentials to _get_monitoring_client."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        mock_creds = MagicMock(name="creds")
        mock_client_fn.return_value = (None, "SDK not installed")

        tools = create_gcloud_tools(project="proj", credentials=mock_creds)
        monitoring_tool = next(t for t in tools if t.name == "gcloud_monitoring_query")

        # Execute the tool — it should pass credentials through
        monitoring_tool.execute("kubernetes.io/container/cpu/core_usage_time")

        mock_client_fn.assert_called_once()
        call_kwargs = mock_client_fn.call_args
        assert call_kwargs.kwargs.get("credentials") is mock_creds or call_kwargs[1].get("credentials") is mock_creds


# ══════════════════════════════════════════════════════════════
# 4.4 — Credential threading in autopilot detection
# ══════════════════════════════════════════════════════════════


class TestAutopilotDetectionCredentials:
    """Test that credentials are passed to the autopilot detection GKE API call."""

    def setup_method(self) -> None:
        """Clear caches before each test."""
        from vaig.tools.gke._clients import clear_autopilot_cache

        clear_autopilot_cache()

    @patch("vaig.tools.gke._clients._query_autopilot_status")
    def test_credentials_passed_to_query(self, mock_query: MagicMock) -> None:
        """detect_autopilot should forward credentials to _query_autopilot_status."""
        from vaig.tools.gke._clients import detect_autopilot

        mock_creds = MagicMock(name="creds")
        mock_query.return_value = True

        config = GKEConfig(
            project_id="proj",
            location="us-central1",
            cluster_name="my-cluster",
        )

        result = detect_autopilot(config, credentials=mock_creds)

        assert result is True
        mock_query.assert_called_once_with(
            "proj", "us-central1", "my-cluster", credentials=mock_creds,
        )

    @patch("vaig.tools.gke._clients._query_autopilot_status")
    def test_credentials_none_by_default(self, mock_query: MagicMock) -> None:
        """detect_autopilot without credentials should pass None."""
        from vaig.tools.gke._clients import detect_autopilot

        mock_query.return_value = False

        config = GKEConfig(
            project_id="proj",
            location="us-central1",
            cluster_name="my-cluster",
        )

        result = detect_autopilot(config)

        assert result is False
        mock_query.assert_called_once_with(
            "proj", "us-central1", "my-cluster", credentials=None,
        )


# ══════════════════════════════════════════════════════════════
# 4.5 — Backward compatibility (credentials=None default)
# ══════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """All functions work correctly with credentials=None (default)."""

    def test_create_gcloud_tools_no_credentials(self) -> None:
        """create_gcloud_tools() with no credentials arg returns tools normally."""
        from vaig.tools.gcloud_tools import create_gcloud_tools

        tools = create_gcloud_tools()
        assert len(tools) == 2

    @patch("vaig.tools.gcloud_tools._get_logging_client")
    def test_logging_query_no_credentials(self, mock_client_fn: MagicMock) -> None:
        """gcloud_logging_query() without credentials should work."""
        from vaig.tools.gcloud_tools import gcloud_logging_query

        mock_client = MagicMock()
        mock_client.list_entries.return_value = []
        mock_client_fn.return_value = (mock_client, None)

        result = gcloud_logging_query("severity>=ERROR")
        assert result.error is False

        # credentials should default to None
        mock_client_fn.assert_called_once()
        call_args = mock_client_fn.call_args
        assert call_args[1].get("credentials") is None or call_args[0][-1] is None

    @patch("vaig.tools.gcloud_tools._get_monitoring_client")
    def test_monitoring_query_no_credentials(self, mock_client_fn: MagicMock) -> None:
        """gcloud_monitoring_query() without credentials should work."""
        from vaig.tools.gcloud_tools import gcloud_monitoring_query

        mock_client_fn.return_value = (None, "SDK not installed")

        result = gcloud_monitoring_query("kubernetes.io/container/cpu")
        assert result.error is True
        assert "not installed" in result.output

    def test_get_gke_credentials_returns_none_for_adc(self) -> None:
        """get_gke_credentials with default settings returns None (ADC)."""
        from vaig.core.auth import get_gke_credentials

        settings = Settings()
        result = get_gke_credentials(settings)
        assert result is None


# ══════════════════════════════════════════════════════════════
# 4.6 — Config: GKEConfig.impersonate_sa field
# ══════════════════════════════════════════════════════════════


class TestGKEConfigImpersonateSa:
    """Test GKEConfig.impersonate_sa field validation and env override."""

    def test_default_is_empty_string(self) -> None:
        cfg = GKEConfig()
        assert cfg.impersonate_sa == ""

    def test_accepts_valid_sa_email(self) -> None:
        cfg = GKEConfig(impersonate_sa="my-sa@my-project.iam.gserviceaccount.com")
        assert cfg.impersonate_sa == "my-sa@my-project.iam.gserviceaccount.com"

    def test_round_trip_serialization(self) -> None:
        cfg = GKEConfig(impersonate_sa="test-sa@proj.iam.gserviceaccount.com")
        data = cfg.model_dump()
        assert data["impersonate_sa"] == "test-sa@proj.iam.gserviceaccount.com"

        cfg2 = GKEConfig(**data)
        assert cfg2.impersonate_sa == cfg.impersonate_sa

    def test_settings_gke_has_impersonate_sa(self) -> None:
        """Settings.gke should expose impersonate_sa field."""
        settings = Settings()
        assert hasattr(settings.gke, "impersonate_sa")
        assert settings.gke.impersonate_sa == ""

    @patch.dict("os.environ", {"VAIG_GKE__IMPERSONATE_SA": "env-sa@proj.iam.gserviceaccount.com"})
    def test_env_var_override(self) -> None:
        """VAIG_GKE__IMPERSONATE_SA should override the default."""
        settings = Settings()
        assert settings.gke.impersonate_sa == "env-sa@proj.iam.gserviceaccount.com"


# ══════════════════════════════════════════════════════════════
# _build_gke_config carries impersonate_sa
# ══════════════════════════════════════════════════════════════


class TestBuildGkeConfigImpersonateSa:
    """Test that _build_gke_config threads impersonate_sa through."""

    def test_impersonate_sa_from_settings(self) -> None:
        """_build_gke_config should carry gke.impersonate_sa from settings."""
        from vaig.cli.app import _build_gke_config

        settings = Settings(
            gke=GKEConfig(impersonate_sa="gke-sa@proj.iam.gserviceaccount.com"),
        )

        gke_config = _build_gke_config(settings)
        assert gke_config.impersonate_sa == "gke-sa@proj.iam.gserviceaccount.com"

    def test_impersonate_sa_empty_by_default(self) -> None:
        """_build_gke_config with default settings should have empty impersonate_sa."""
        from vaig.cli.app import _build_gke_config

        settings = Settings()

        gke_config = _build_gke_config(settings)
        assert gke_config.impersonate_sa == ""
