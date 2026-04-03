"""Tests for the headless skill runner."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig, Settings


@dataclass
class _FakeOrchestratorResult:
    """Minimal stand-in for OrchestratorResult to avoid heavy imports."""

    skill_name: str = "discovery"
    phase: str = "execute"
    synthesized_output: str = "test output"
    success: bool = True
    run_cost_usd: float = 0.001
    structured_report: Any = None
    agent_results: list[Any] = field(default_factory=list)


class _FakeSkill:
    """Minimal stand-in for BaseSkill."""

    def get_metadata(self) -> _FakeSkillMeta:
        return _FakeSkillMeta()


class _FakeSkillMeta:
    name: str = "discovery"
    display_name: str = "Discovery"


class _FakeToolRegistry:
    def list_tools(self) -> list[str]:
        return ["kubectl_get_pods", "kubectl_get_events", "kubectl_get_logs"]


# Patches target the original module where the import resolves, because
# headless.py uses inline imports (lazy to avoid circular dependency).
_P_REGISTER = "vaig.core.gke.register_live_tools"
_P_ORCHESTRATOR = "vaig.agents.orchestrator.Orchestrator"
_P_CLIENT = "vaig.core.client.GeminiClient"
_P_CREDS = "vaig.core.auth.get_gke_credentials"


class TestExecuteSkillHeadless:
    """Unit tests for execute_skill_headless()."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_returns_orchestrator_result(
        self,
        mock_creds: MagicMock,
        mock_register: MagicMock,
        mock_orch_cls: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        from vaig.core.headless import execute_skill_headless

        expected = _FakeOrchestratorResult()
        mock_register.return_value = _FakeToolRegistry()
        mock_orch_cls.return_value.execute_with_tools.return_value = expected

        settings = Settings()
        gke_config = GKEConfig(cluster_name="test-cluster")
        skill = _FakeSkill()

        result = execute_skill_headless(settings, skill, "test query", gke_config)

        assert result is expected
        assert result.success is True
        assert result.skill_name == "discovery"

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_no_stdout_output(
        self,
        mock_creds: MagicMock,
        mock_register: MagicMock,
        mock_orch_cls: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        """Headless runner must not write to stdout or stderr."""
        from vaig.core.headless import execute_skill_headless

        mock_register.return_value = _FakeToolRegistry()
        mock_orch_cls.return_value.execute_with_tools.return_value = (
            _FakeOrchestratorResult()
        )

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            execute_skill_headless(
                Settings(),
                _FakeSkill(),
                "test query",
                GKEConfig(cluster_name="test"),
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        assert captured_stdout.getvalue() == ""
        assert captured_stderr.getvalue() == ""

    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_raises_on_empty_registry(
        self,
        mock_creds: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        """Should raise RuntimeError when no tools are registered."""
        from vaig.core.headless import execute_skill_headless

        empty_registry = MagicMock()
        empty_registry.list_tools.return_value = []
        mock_register.return_value = empty_registry

        with pytest.raises(RuntimeError, match="No live tools registered"):
            execute_skill_headless(
                Settings(),
                _FakeSkill(),
                "test query",
                GKEConfig(cluster_name="test"),
            )

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_passes_gke_config_to_orchestrator(
        self,
        mock_creds: MagicMock,
        mock_register: MagicMock,
        mock_orch_cls: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        from vaig.core.headless import execute_skill_headless

        mock_register.return_value = _FakeToolRegistry()
        mock_orch_cls.return_value.execute_with_tools.return_value = (
            _FakeOrchestratorResult()
        )

        gke_config = GKEConfig(
            cluster_name="prod-us",
            default_namespace="payments",
            location="us-central1",
        )

        execute_skill_headless(Settings(), _FakeSkill(), "check health", gke_config)

        call_kwargs = mock_orch_cls.return_value.execute_with_tools.call_args
        assert call_kwargs.kwargs["gke_namespace"] == "payments"
        assert call_kwargs.kwargs["gke_location"] == "us-central1"
        assert call_kwargs.kwargs["gke_cluster_name"] == "prod-us"

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_passes_tool_call_store(
        self,
        mock_creds: MagicMock,
        mock_register: MagicMock,
        mock_orch_cls: MagicMock,
        mock_client_cls: MagicMock,
    ) -> None:
        from vaig.core.headless import execute_skill_headless

        mock_register.return_value = _FakeToolRegistry()
        mock_orch_cls.return_value.execute_with_tools.return_value = (
            _FakeOrchestratorResult()
        )

        fake_store = MagicMock()
        execute_skill_headless(
            Settings(),
            _FakeSkill(),
            "test",
            GKEConfig(cluster_name="test"),
            tool_call_store=fake_store,
        )

        call_kwargs = mock_orch_cls.return_value.execute_with_tools.call_args
        assert call_kwargs.kwargs["tool_call_store"] is fake_store

    def test_timeout_parameter_removed(self) -> None:
        """The unused ``timeout`` parameter should no longer be accepted."""
        import inspect

        from vaig.core.headless import execute_skill_headless

        sig = inspect.signature(execute_skill_headless)
        assert "timeout" not in sig.parameters
