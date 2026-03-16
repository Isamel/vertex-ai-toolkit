"""Tests for error handling — exception hierarchy, formatting, and error boundaries.

Covers Phase 1 (P0) error handling improvements:
- Custom exception hierarchy (VaigAuthError, GCPAuthError, GCPPermissionError, K8sAuthError)
- format_error_for_user() utility for user-friendly Rich-markup output
- handle_cli_error() CLI error boundary
- GeminiClient.initialize() error wrapping
- auth.py credential error wrapping
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.exceptions import (
    GCPAuthError,
    GCPPermissionError,
    GeminiClientError,
    K8sAuthError,
    VAIGError,
    VaigAuthError,
    format_error_for_user,
)


# ── Exception hierarchy ──────────────────────────────────────


class TestExceptionHierarchy:
    """Verify exception inheritance and attributes."""

    def test_vaig_auth_error_is_vaig_error(self) -> None:
        assert issubclass(VaigAuthError, VAIGError)

    def test_gcp_auth_error_is_vaig_auth_error(self) -> None:
        assert issubclass(GCPAuthError, VaigAuthError)

    def test_gcp_permission_error_is_vaig_auth_error(self) -> None:
        assert issubclass(GCPPermissionError, VaigAuthError)

    def test_k8s_auth_error_is_vaig_auth_error(self) -> None:
        assert issubclass(K8sAuthError, VaigAuthError)

    def test_gcp_auth_error_has_fix_suggestion(self) -> None:
        exc = GCPAuthError("No credentials", fix_suggestion="Run gcloud auth login")
        assert exc.fix_suggestion == "Run gcloud auth login"
        assert str(exc) == "No credentials"

    def test_gcp_auth_error_default_fix_suggestion(self) -> None:
        exc = GCPAuthError("No credentials")
        assert exc.fix_suggestion == ""

    def test_gcp_permission_error_has_required_permissions(self) -> None:
        exc = GCPPermissionError(
            "Permission denied",
            required_permissions=["roles/aiplatform.user"],
            fix_suggestion="Grant the role",
        )
        assert exc.required_permissions == ["roles/aiplatform.user"]
        assert exc.fix_suggestion == "Grant the role"
        assert str(exc) == "Permission denied"

    def test_gcp_permission_error_defaults(self) -> None:
        exc = GCPPermissionError("Permission denied")
        assert exc.required_permissions == []
        assert exc.fix_suggestion == ""

    def test_k8s_auth_error_message(self) -> None:
        exc = K8sAuthError("Cluster unreachable")
        assert str(exc) == "Cluster unreachable"
        assert isinstance(exc, VaigAuthError)


# ── format_error_for_user ────────────────────────────────────


class TestFormatErrorForUser:
    """Verify Rich-markup formatting for different exception types."""

    def test_gcp_auth_error_with_fix(self) -> None:
        exc = GCPAuthError("Credentials not found", fix_suggestion="Run gcloud auth login")
        result = format_error_for_user(exc)
        assert "[red]Authentication Error:[/red]" in result
        assert "Credentials not found" in result
        assert "[yellow]Fix:[/yellow]" in result
        assert "Run gcloud auth login" in result

    def test_gcp_auth_error_without_fix(self) -> None:
        exc = GCPAuthError("Credentials not found")
        result = format_error_for_user(exc)
        assert "[red]Authentication Error:[/red]" in result
        assert "[yellow]Fix:[/yellow]" not in result

    def test_gcp_permission_error_with_permissions(self) -> None:
        exc = GCPPermissionError(
            "Access denied",
            required_permissions=["roles/aiplatform.user", "roles/viewer"],
            fix_suggestion="Grant the roles",
        )
        result = format_error_for_user(exc)
        assert "[red]Permission Denied:[/red]" in result
        assert "Access denied" in result
        assert "[yellow]Required permissions:[/yellow]" in result
        assert "roles/aiplatform.user" in result
        assert "roles/viewer" in result
        assert "[yellow]Fix:[/yellow]" in result

    def test_gcp_permission_error_without_permissions(self) -> None:
        exc = GCPPermissionError("Access denied")
        result = format_error_for_user(exc)
        assert "[red]Permission Denied:[/red]" in result
        assert "[yellow]Required permissions:[/yellow]" not in result

    def test_k8s_auth_error(self) -> None:
        exc = K8sAuthError("Cannot reach cluster")
        result = format_error_for_user(exc)
        assert "[red]Kubernetes Auth Error:[/red]" in result
        assert "Cannot reach cluster" in result
        assert "kubectl config current-context" in result

    def test_vaig_auth_error(self) -> None:
        exc = VaigAuthError("Some auth failure")
        result = format_error_for_user(exc)
        assert "[red]Authentication Error:[/red]" in result
        assert "Some auth failure" in result

    def test_vaig_error(self) -> None:
        exc = VAIGError("Something went wrong")
        result = format_error_for_user(exc)
        assert "[red]Error:[/red]" in result
        assert "Something went wrong" in result

    def test_generic_exception(self) -> None:
        exc = RuntimeError("Unexpected crash")
        result = format_error_for_user(exc)
        assert "[red]Unexpected Error:[/red]" in result
        assert "RuntimeError" in result
        assert "Unexpected crash" in result

    def test_debug_false_shows_hint(self) -> None:
        exc = GCPAuthError("No creds")
        result = format_error_for_user(exc, debug=False)
        assert "Use --debug for full traceback" in result

    def test_debug_true_shows_traceback(self) -> None:
        try:
            raise GCPAuthError("No creds", fix_suggestion="Login")
        except GCPAuthError as exc:
            result = format_error_for_user(exc, debug=True)
            assert "[dim]Full traceback:[/dim]" in result
            assert "GCPAuthError" in result
            # Should NOT show the "Use --debug" hint
            assert "Use --debug for full traceback" not in result


# ── handle_cli_error ─────────────────────────────────────────


class TestHandleCliError:
    """Verify the CLI error boundary function."""

    def test_handle_cli_error_raises_typer_exit(self) -> None:
        import typer

        from vaig.cli._helpers import handle_cli_error

        with pytest.raises(typer.Exit) as exc_info:
            handle_cli_error(GCPAuthError("No creds"), debug=False)
        assert exc_info.value.exit_code == 1

    def test_handle_cli_error_prints_formatted_error(self) -> None:
        import typer

        from vaig.cli._helpers import err_console, handle_cli_error

        with patch.object(err_console, "print") as mock_print, pytest.raises(typer.Exit):
            handle_cli_error(GCPAuthError("No creds", fix_suggestion="Login"), debug=False)

        # The formatted error should have been printed
        call_args = mock_print.call_args[0][0]
        assert "Authentication Error" in call_args
        assert "No creds" in call_args


# ── GeminiClient.initialize() error wrapping ─────────────────


class TestClientInitializeErrors:
    """Verify GeminiClient.initialize() wraps auth errors correctly."""

    def _make_settings(self) -> MagicMock:
        """Create minimal mock settings for client tests."""
        from vaig.core.config import (
            GCPConfig,
            GenerationConfig,
            ModelInfo,
            ModelsConfig,
            Settings,
        )

        return Settings(
            gcp=GCPConfig(project_id="test-project", location="us-central1"),
            generation=GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
                top_p=0.95,
                top_k=40,
            ),
            models=ModelsConfig(
                default="gemini-2.5-pro",
                fallback="gemini-2.5-flash",
                available=[
                    ModelInfo(id="gemini-2.5-pro", description="Pro"),
                    ModelInfo(id="gemini-2.5-flash", description="Flash"),
                ],
            ),
        )

    def test_initialize_catches_default_credentials_error(self) -> None:
        """DefaultCredentialsError should become GCPAuthError."""
        import google.auth.exceptions

        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        with (
            patch(
                "vaig.core.client.get_credentials",
                side_effect=google.auth.exceptions.DefaultCredentialsError("No ADC"),
            ),
            pytest.raises(GCPAuthError, match="credentials not configured"),
        ):
            client.initialize()

    def test_initialize_catches_refresh_error(self) -> None:
        """RefreshError should become GCPAuthError."""
        from google.auth import exceptions as auth_exc

        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        with (
            patch(
                "vaig.core.client.get_credentials",
                side_effect=auth_exc.RefreshError("Token expired"),
            ),
            pytest.raises(GCPAuthError, match="expired or invalid"),
        ):
            client.initialize()

    def test_initialize_catches_permission_error(self) -> None:
        """Permission-related errors should become GCPPermissionError."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        with (
            patch(
                "vaig.core.client.get_credentials",
                side_effect=Exception("403 Permission denied for project"),
            ),
            pytest.raises(GCPPermissionError, match="Insufficient permissions"),
        ):
            client.initialize()

    def test_initialize_catches_generic_error(self) -> None:
        """Generic errors should become GeminiClientError."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        with (
            patch(
                "vaig.core.client.get_credentials",
                side_effect=Exception("Something totally unexpected"),
            ),
            pytest.raises(GeminiClientError, match="Failed to initialize"),
        ):
            client.initialize()

    def test_initialize_propagates_gcp_auth_error(self) -> None:
        """GCPAuthError from auth.py should propagate unchanged."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        original = GCPAuthError("Already wrapped", fix_suggestion="Do this")
        with (
            patch("vaig.core.client.get_credentials", side_effect=original),
            pytest.raises(GCPAuthError, match="Already wrapped") as exc_info,
        ):
            client.initialize()

        assert exc_info.value.fix_suggestion == "Do this"


# ── auth.py error wrapping ───────────────────────────────────


class TestAuthErrors:
    """Verify auth.py wraps credential errors into custom exceptions."""

    def test_adc_fallback_runtime_error_becomes_gcp_auth_error(self) -> None:
        """When ADC fails and gcloud CLI also fails, should raise GCPAuthError."""
        import google.auth.exceptions

        from vaig.core.auth import _get_adc_credentials

        with (
            patch(
                "vaig.core.auth.google.auth.default",
                side_effect=google.auth.exceptions.DefaultCredentialsError("No ADC"),
            ),
            patch(
                "vaig.core.auth._get_gcloud_token_credentials",
                side_effect=RuntimeError("gcloud not found"),
            ),
            pytest.raises(GCPAuthError, match="No GCP credentials available"),
        ):
            _get_adc_credentials()

    def test_impersonate_missing_source_credentials(self) -> None:
        """When source credentials fail, should raise GCPAuthError."""
        import google.auth.exceptions

        from vaig.core.auth import _get_impersonated_credentials

        with (
            patch(
                "vaig.core.auth.google.auth.default",
                side_effect=google.auth.exceptions.DefaultCredentialsError("No ADC"),
            ),
            pytest.raises(GCPAuthError, match="source credentials"),
        ):
            _get_impersonated_credentials("sa@project.iam.gserviceaccount.com")

    def test_impersonate_permission_error(self) -> None:
        """Permission errors during impersonation should raise GCPPermissionError."""
        from vaig.core.auth import _get_impersonated_credentials

        mock_creds = MagicMock()
        with (
            patch("vaig.core.auth.google.auth.default", return_value=(mock_creds, "project")),
            patch(
                "vaig.core.auth.impersonated_credentials.Credentials",
                side_effect=Exception("403 Permission denied on IAM"),
            ),
            pytest.raises(GCPPermissionError, match="Cannot impersonate SA"),
        ):
            _get_impersonated_credentials("sa@project.iam.gserviceaccount.com")

    def test_impersonate_missing_sa_raises_value_error(self) -> None:
        """Empty SA email should raise ValueError (not our custom type)."""
        from vaig.core.auth import _get_impersonated_credentials

        with pytest.raises(ValueError, match="VAIG_IMPERSONATE_SA"):
            _get_impersonated_credentials("")


# ══════════════════════════════════════════════════════════════
# Phase 2 (P1) — Agent sanitize, REPL debug state, REPL error display
# ══════════════════════════════════════════════════════════════


class TestSanitizeErrorForAgent:
    """Verify BaseAgent.sanitize_error_for_agent() strips internals."""

    @staticmethod
    def _sanitize(exc: Exception) -> str:
        from vaig.agents.base import BaseAgent

        return BaseAgent.sanitize_error_for_agent(exc)

    def test_strips_grpc_status_details(self) -> None:
        exc = RuntimeError(
            "StatusCode.UNAVAILABLE\nDebug info: some debug\n"
            "grpc._channel._InactiveRpcError\nService not responding"
        )
        result = self._sanitize(exc)
        assert result.startswith("API Error:")
        assert "grpc" not in result.lower()
        assert "Debug" not in result

    def test_strips_grpc_returns_fallback_when_no_clean_line(self) -> None:
        exc = RuntimeError("grpc error only\ngrpc._channel stuff\nDebug: x")
        result = self._sanitize(exc)
        assert result == "API Error: Service unavailable. Check your credentials and network."

    def test_strips_protobuf_wire_format(self) -> None:
        exc = RuntimeError("proto field_123 error\ndetailed line 2\ndetailed line 3")
        result = self._sanitize(exc)
        assert result.startswith("API Error:")
        # Only the first line should be kept, capped at 200 chars
        assert "detailed line 2" not in result

    def test_handles_gcp_auth_error_with_fix(self) -> None:
        exc = GCPAuthError("No credentials found", fix_suggestion="Run gcloud auth login")
        result = self._sanitize(exc)
        assert "Authentication failed" in result
        assert "No credentials found" in result
        assert "Run gcloud auth login" in result

    def test_handles_gcp_auth_error_without_fix(self) -> None:
        exc = GCPAuthError("No credentials found")
        result = self._sanitize(exc)
        assert "Authentication failed" in result
        assert "No credentials found" in result

    def test_handles_gcp_permission_error(self) -> None:
        exc = GCPPermissionError(
            "Access denied",
            required_permissions=["roles/aiplatform.user"],
        )
        result = self._sanitize(exc)
        assert "Permission denied" in result
        assert "roles/aiplatform.user" in result

    def test_caps_long_messages_at_500_chars(self) -> None:
        exc = RuntimeError("x" * 600)
        result = self._sanitize(exc)
        assert len(result) == 500

    def test_passes_short_generic_messages_unchanged(self) -> None:
        exc = RuntimeError("Simple error")
        result = self._sanitize(exc)
        assert result == "Simple error"


class TestREPLStateDebug:
    """Verify REPLState accepts and exposes the debug flag."""

    def test_repl_state_debug_default_false(self) -> None:
        from vaig.cli.repl import REPLState

        state = REPLState(
            settings=MagicMock(),
            client=MagicMock(),
            orchestrator=MagicMock(),
            session_manager=MagicMock(),
            context_builder=MagicMock(),
            skill_registry=MagicMock(),
        )
        assert state.debug is False

    def test_repl_state_debug_true(self) -> None:
        from vaig.cli.repl import REPLState

        state = REPLState(
            settings=MagicMock(),
            client=MagicMock(),
            orchestrator=MagicMock(),
            session_manager=MagicMock(),
            context_builder=MagicMock(),
            skill_registry=MagicMock(),
            debug=True,
        )
        assert state.debug is True


class TestREPLErrorDisplay:
    """Verify REPL catch blocks use format_error_for_user instead of raw strings."""

    async def test_async_direct_chat_uses_format_error(self) -> None:
        """_async_handle_direct_chat should call format_error_for_user on error."""
        from vaig.cli import repl as repl_module

        state = MagicMock()
        state.debug = False
        state.stream_enabled = True
        state.settings = MagicMock()

        exc = GCPAuthError("No creds", fix_suggestion="Login")

        # Make the orchestrator raise
        state.orchestrator.async_execute_single_stream = MagicMock(side_effect=exc)
        state.orchestrator.async_execute_single = MagicMock(side_effect=exc)

        with patch.object(repl_module, "console") as mock_console, \
             patch.object(repl_module, "_check_budget", return_value=True), \
             patch.object(repl_module, "format_error_for_user", wraps=format_error_for_user) as mock_fmt:
            await repl_module._async_handle_direct_chat(state, "hello", "")
            mock_fmt.assert_called_once_with(exc, debug=False)
            # Console should print the formatted output (not raw f-string)
            printed = mock_console.print.call_args_list[-1][0][0]
            assert "Authentication Error" in printed

    def test_sync_direct_chat_uses_format_error(self) -> None:
        """_handle_direct_chat should call format_error_for_user on error."""
        from vaig.cli import repl as repl_module

        state = MagicMock()
        state.debug = True
        state.stream_enabled = False
        state.settings = MagicMock()

        exc = GCPPermissionError("Denied", required_permissions=["roles/viewer"])

        state.orchestrator.execute_single = MagicMock(side_effect=exc)

        with patch.object(repl_module, "console") as mock_console, \
             patch.object(repl_module, "_check_budget", return_value=True), \
             patch.object(repl_module, "_try_chunked_chat", return_value=False), \
             patch.object(repl_module, "format_error_for_user", wraps=format_error_for_user) as mock_fmt:
            repl_module._handle_direct_chat(state, "hello", "")
            mock_fmt.assert_called_once_with(exc, debug=True)
            printed = mock_console.print.call_args_list[-1][0][0]
            assert "Permission Denied" in printed


# ══════════════════════════════════════════════════════════════
# Phase 3 (P2) — count_tokens, orchestrator boundaries, gcloud_tools
# ══════════════════════════════════════════════════════════════


class TestCountTokensErrorHandling:
    """Verify count_tokens() and async_count_tokens() return None on error."""

    def _make_settings(self) -> Any:
        """Create minimal mock settings for client tests."""
        from vaig.core.config import (
            GCPConfig,
            GenerationConfig,
            ModelInfo,
            ModelsConfig,
            Settings,
        )

        return Settings(
            gcp=GCPConfig(project_id="test-project", location="us-central1"),
            generation=GenerationConfig(
                temperature=0.7,
                max_output_tokens=8192,
                top_p=0.95,
                top_k=40,
            ),
            models=ModelsConfig(
                default="gemini-2.5-pro",
                fallback="gemini-2.5-flash",
                available=[
                    ModelInfo(id="gemini-2.5-pro", description="Pro"),
                    ModelInfo(id="gemini-2.5-flash", description="Flash"),
                ],
            ),
        )

    def test_count_tokens_returns_none_on_api_error(self) -> None:
        """count_tokens() should catch exceptions and return None."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        # Pre-set state so _get_client() works without initialize()
        mock_genai = MagicMock()
        mock_genai.models.count_tokens.side_effect = RuntimeError("API unavailable")
        client._client = mock_genai
        client._initialized = True
        client._current_model_id = "gemini-2.5-pro"

        result = client.count_tokens("hello world")
        assert result is None

    def test_count_tokens_returns_token_count_on_success(self) -> None:
        """count_tokens() should return the token count on success."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        mock_genai = MagicMock()
        mock_response = MagicMock()
        mock_response.total_tokens = 42
        mock_genai.models.count_tokens.return_value = mock_response
        client._client = mock_genai
        client._initialized = True
        client._current_model_id = "gemini-2.5-pro"

        result = client.count_tokens("hello world")
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_count_tokens_returns_none_on_api_error(self) -> None:
        """async_count_tokens() should catch exceptions and return None."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        mock_genai = MagicMock()
        mock_genai.aio.models.count_tokens = MagicMock(
            side_effect=RuntimeError("API unavailable"),
        )
        client._client = mock_genai
        client._initialized = True
        client._current_model_id = "gemini-2.5-pro"

        result = await client.async_count_tokens("hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_count_tokens_returns_token_count_on_success(self) -> None:
        """async_count_tokens() should return the token count on success."""
        from vaig.core.client import GeminiClient

        settings = self._make_settings()
        client = GeminiClient(settings)

        mock_response = MagicMock()
        mock_response.total_tokens = 99

        async def mock_count(*a: Any, **kw: Any) -> MagicMock:
            return mock_response

        mock_genai = MagicMock()
        mock_genai.aio.models.count_tokens = mock_count
        client._client = mock_genai
        client._initialized = True
        client._current_model_id = "gemini-2.5-pro"

        result = await client.async_count_tokens("hello world")
        assert result == 99


class TestOrchestratorErrorBoundaries:
    """Verify orchestrator entry points wrap unknown exceptions in VAIGError."""

    def _make_orchestrator(self) -> Any:
        """Create an Orchestrator with mocked dependencies."""
        from vaig.agents.orchestrator import Orchestrator

        mock_client = MagicMock()
        mock_settings = MagicMock()
        return Orchestrator(client=mock_client, settings=mock_settings)

    def test_execute_skill_phase_wraps_unknown_error(self) -> None:
        """Unknown exceptions should become VAIGError."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        with patch.object(
            orch, "execute_sequential", side_effect=RuntimeError("boom"),
        ), pytest.raises(VAIGError, match="Pipeline execution failed"):
            orch.execute_skill_phase(mock_skill, MagicMock(), "ctx", "input")

    def test_execute_skill_phase_passes_through_vaig_auth_error(self) -> None:
        """VaigAuthError should propagate unchanged."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        original = VaigAuthError("Auth failed")
        with patch.object(
            orch, "execute_sequential", side_effect=original,
        ), pytest.raises(VaigAuthError, match="Auth failed"):
            orch.execute_skill_phase(mock_skill, MagicMock(), "ctx", "input")

    def test_execute_skill_phase_passes_through_vaig_error(self) -> None:
        """VAIGError should propagate unchanged (not double-wrapped)."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        original = VAIGError("Already a VAIGError")
        with patch.object(
            orch, "execute_sequential", side_effect=original,
        ), pytest.raises(VAIGError, match="Already a VAIGError"):
            orch.execute_skill_phase(mock_skill, MagicMock(), "ctx", "input")

    def test_execute_with_tools_wraps_unknown_error(self) -> None:
        """Unknown exceptions should become VAIGError."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        with patch.object(
            orch, "_execute_with_tools_impl", side_effect=RuntimeError("boom"),
        ), pytest.raises(VAIGError, match="Pipeline execution failed"):
            orch.execute_with_tools("q", mock_skill, MagicMock())

    def test_execute_with_tools_passes_through_vaig_auth_error(self) -> None:
        """VaigAuthError should propagate unchanged."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()

        original = VaigAuthError("Auth failed")
        with patch.object(
            orch, "_execute_with_tools_impl", side_effect=original,
        ), pytest.raises(VaigAuthError, match="Auth failed"):
            orch.execute_with_tools("q", mock_skill, MagicMock())

    @pytest.mark.asyncio
    async def test_async_execute_skill_phase_wraps_unknown_error(self) -> None:
        """Unknown exceptions should become VAIGError (async)."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        with patch.object(
            orch, "async_execute_sequential", side_effect=RuntimeError("boom"),
        ), pytest.raises(VAIGError, match="Pipeline execution failed"):
            await orch.async_execute_skill_phase(mock_skill, MagicMock(), "ctx", "input")

    @pytest.mark.asyncio
    async def test_async_execute_skill_phase_passes_through_vaig_auth_error(self) -> None:
        """VaigAuthError should propagate unchanged (async)."""
        orch = self._make_orchestrator()
        mock_skill = MagicMock()
        mock_skill.get_metadata.return_value.name = "test"

        original = VaigAuthError("Auth failed")
        with patch.object(
            orch, "async_execute_sequential", side_effect=original,
        ), pytest.raises(VaigAuthError, match="Auth failed"):
            await orch.async_execute_skill_phase(mock_skill, MagicMock(), "ctx", "input")

    @pytest.mark.asyncio
    async def test_async_execute_with_tools_wraps_unknown_error(self) -> None:
        """Unknown exceptions should become VAIGError (async)."""
        orch = self._make_orchestrator()

        with patch.object(
            orch, "_async_execute_with_tools_impl", side_effect=RuntimeError("boom"),
        ), pytest.raises(VAIGError, match="Pipeline execution failed"):
            await orch.async_execute_with_tools("q", MagicMock(), MagicMock())

    @pytest.mark.asyncio
    async def test_async_execute_with_tools_passes_through_vaig_auth_error(self) -> None:
        """VaigAuthError should propagate unchanged (async)."""
        orch = self._make_orchestrator()

        original = VaigAuthError("Auth failed")
        with patch.object(
            orch, "_async_execute_with_tools_impl", side_effect=original,
        ), pytest.raises(VaigAuthError, match="Auth failed"):
            await orch.async_execute_with_tools("q", MagicMock(), MagicMock())


class TestHandleGCPApiError:
    """Verify _handle_gcp_api_error() classifies errors correctly."""

    def test_permission_denied_formatting(self) -> None:
        """PermissionDenied should produce a permission-specific message."""
        from google.api_core.exceptions import PermissionDenied

        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = PermissionDenied("User lacks access")
        result = _handle_gcp_api_error(exc, service="Cloud Logging")
        assert "Permission denied" in result
        assert "Cloud Logging" in result
        assert "IAM" in result

    def test_forbidden_formatting(self) -> None:
        """Forbidden should produce a permission-specific message."""
        from google.api_core.exceptions import Forbidden

        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = Forbidden("API disabled")
        result = _handle_gcp_api_error(exc, service="Cloud Monitoring")
        assert "Permission denied" in result
        assert "Cloud Monitoring" in result

    def test_resource_exhausted_formatting(self) -> None:
        """ResourceExhausted should mention quota."""
        from google.api_core.exceptions import ResourceExhausted

        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = ResourceExhausted("Quota limit reached")
        result = _handle_gcp_api_error(exc, service="Cloud Logging")
        assert "quota exceeded" in result.lower()
        assert "Cloud Logging" in result

    def test_invalid_argument_formatting(self) -> None:
        """InvalidArgument should mention invalid request."""
        from google.api_core.exceptions import InvalidArgument

        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = InvalidArgument("Bad filter syntax")
        result = _handle_gcp_api_error(exc, service="Cloud Logging")
        assert "Invalid request" in result
        assert "Cloud Logging" in result

    def test_generic_exception_fallback(self) -> None:
        """Unknown exception types should use the generic fallback."""
        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = RuntimeError("Something went wrong")
        result = _handle_gcp_api_error(exc, service="Cloud Logging")
        assert "Error querying Cloud Logging" in result
        assert "Something went wrong" in result

    def test_generic_exception_truncates_long_message(self) -> None:
        """Generic fallback should truncate message at 300 chars."""
        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = RuntimeError("x" * 500)
        result = _handle_gcp_api_error(exc, service="Cloud Logging")
        # The str(exc) part should be capped at 300 chars
        assert len(result) < 400

    def test_handles_missing_google_api_core(self) -> None:
        """When google.api_core is missing, should fall through to generic."""
        from vaig.tools.gcloud_tools import _handle_gcp_api_error

        exc = RuntimeError("Some error")
        with patch.dict("sys.modules", {"google.api_core.exceptions": None}):
            # Force ImportError on the lazy import
            result = _handle_gcp_api_error(exc, service="Cloud Logging")
        assert "Error querying Cloud Logging" in result
        assert "Some error" in result
