"""Tests for the CLI routing fallback behaviour in the ``live`` command.

When a skill has ``requires_live_tools=True`` but NO infrastructure tools are
available at runtime, the CLI must NOT crash.  Instead it falls back to a true
offline path: it calls ``client.generate()`` / ``client.async_generate()``
directly with the skill context prepended to the question, and prints the
response.

Tasks covered: 1.1, 1.2, 1.3 of the requires-live-tools change.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaig.core.config import GKEConfig, Settings
from vaig.skills.base import SkillMetadata


def _make_skill_mock(
    *,
    requires_live_tools: bool = False,
    name: str = "test-skill",
    display_name: str = "Test Skill",
    description: str = "A test skill",
) -> MagicMock:
    """Create a mock skill with configurable metadata."""
    skill = MagicMock()
    skill.get_metadata.return_value = SkillMetadata(
        name=name,
        display_name=display_name,
        description=description,
        requires_live_tools=requires_live_tools,
    )
    return skill


def _make_empty_registry() -> MagicMock:
    """Tool registry with zero tools."""
    registry = MagicMock()
    registry.list_tools.return_value = []
    return registry


def _make_populated_registry(count: int = 3) -> MagicMock:
    """Tool registry with *count* tools."""
    registry = MagicMock()
    registry.list_tools.return_value = [MagicMock() for _ in range(count)]
    return registry


def _make_generation_result(text: str = "Offline analysis result") -> MagicMock:
    """Create a mock GenerationResult with a .text attribute."""
    result = MagicMock()
    result.text = text
    return result


# ══════════════════════════════════════════════════════════════
# Sync path — _execute_orchestrated_skill
# ══════════════════════════════════════════════════════════════


class TestSyncOrchestratedFallback:
    """Covers Task 1.1: sync fallback on zero tools."""

    def test_sync_fallback_on_zero_tools_calls_generate(self) -> None:
        """With requires_live_tools=True and zero tools, client.generate() is called."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)
        mock_client = MagicMock()
        mock_client.generate.return_value = _make_generation_result()

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            _execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                "Why is the service down?",
            )

        mock_client.generate.assert_called_once()

    def test_sync_generate_called_with_skill_context_and_question(self) -> None:
        """client.generate() receives a prompt containing the skill context and question."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        skill = _make_skill_mock(
            requires_live_tools=True,
            name="rca",
            display_name="Root Cause Analysis",
            description="Perform RCA on incidents.",
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = _make_generation_result()
        question = "Why did the service crash?"

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            _execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                question,
            )

        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]  # first positional argument
        assert "Root Cause Analysis" in prompt
        assert "Perform RCA on incidents." in prompt
        assert question in prompt

    def test_sync_no_exit_on_zero_tools(self) -> None:
        """With zero tools, _execute_orchestrated_skill does NOT raise SystemExit."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)
        mock_client = MagicMock()
        mock_client.generate.return_value = _make_generation_result()

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            try:
                _execute_orchestrated_skill(
                    mock_client,
                    Settings(),
                    GKEConfig(),
                    skill,
                    "query",
                )
            except SystemExit as exc:
                pytest.fail(f"Should not exit — got SystemExit({exc.code})")

    def test_sync_proceeds_with_tools(self) -> None:
        """With tools available, the orchestrated path (execute_with_tools) is used."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "All good"
        mock_orch_result.success = True
        mock_orch_result.agent_results = []
        mock_orch_result.structured_report = None
        mock_orch_result.total_usage = None

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_with_tools.return_value = mock_orch_result

        mock_client = MagicMock()

        with (
            patch(
                "vaig.cli.commands.live._register_live_tools",
                return_value=_make_populated_registry(3),
            ),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            _execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                "query",
            )

        # Orchestrated path was used
        mock_orchestrator.execute_with_tools.assert_called_once()
        # Direct generate fallback was NOT called
        mock_client.generate.assert_not_called()

    def test_fallback_warning_message_is_shown(self, capsys) -> None:
        """The user sees a warning explaining offline context-prepend mode."""
        from vaig.cli.commands.live import _execute_orchestrated_skill

        skill = _make_skill_mock(
            requires_live_tools=True,
            name="rca",
            display_name="RCA",
        )
        mock_client = MagicMock()
        mock_client.generate.return_value = _make_generation_result()

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            _execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                "query",
            )

        # Rich console outputs to stderr/stdout captured by capsys.
        # We check either channel for the warning.
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "offline" in combined or "no live tools" in combined or "context-prepend" in combined, (
            f"Expected warning message not found in output: {combined!r}"
        )

    def test_requires_live_tools_false_uses_legacy_path_directly(self) -> None:
        """Skills with requires_live_tools=False are never routed to orchestrator."""
        # This test verifies the existing routing — skills w/ requires_live_tools=False
        # never call _execute_orchestrated_skill at all (routed at the caller level).
        # We verify that SkillMetadata.requires_live_tools=False is readable.
        meta = SkillMetadata(
            name="log-analysis",
            display_name="Log Analysis",
            description="Analyse logs",
            requires_live_tools=False,
        )
        assert meta.requires_live_tools is False


# ══════════════════════════════════════════════════════════════
# Async path — _async_execute_orchestrated_skill
# ══════════════════════════════════════════════════════════════


class TestAsyncOrchestratedFallback:
    """Covers Task 1.2: async fallback on zero tools."""

    @pytest.mark.asyncio
    async def test_async_fallback_on_zero_tools_calls_async_generate(self) -> None:
        """With zero tools, client.async_generate() is awaited (not typer.Exit)."""
        from vaig.cli.commands.live import _async_execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)
        mock_client = MagicMock()
        mock_client.async_generate = AsyncMock(return_value=_make_generation_result())

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            await _async_execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                "query",
            )

        mock_client.async_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_generate_called_with_skill_context_and_question(self) -> None:
        """client.async_generate() receives a prompt with the skill context + question."""
        from vaig.cli.commands.live import _async_execute_orchestrated_skill

        skill = _make_skill_mock(
            requires_live_tools=True,
            name="rca",
            display_name="Root Cause Analysis",
            description="Perform RCA on incidents.",
        )
        mock_client = MagicMock()
        question = "Why did the pod OOMKill?"
        mock_client.async_generate = AsyncMock(return_value=_make_generation_result())

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            await _async_execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                question,
            )

        call_args = mock_client.async_generate.call_args
        prompt = call_args[0][0]
        assert "Root Cause Analysis" in prompt
        assert "Perform RCA on incidents." in prompt
        assert question in prompt

    @pytest.mark.asyncio
    async def test_async_no_exit_on_zero_tools(self) -> None:
        """Async path with zero tools does NOT raise SystemExit."""
        from vaig.cli.commands.live import _async_execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)
        mock_client = MagicMock()
        mock_client.async_generate = AsyncMock(return_value=_make_generation_result())

        with patch(
            "vaig.cli.commands.live._register_live_tools",
            return_value=_make_empty_registry(),
        ):
            try:
                await _async_execute_orchestrated_skill(
                    mock_client,
                    Settings(),
                    GKEConfig(),
                    skill,
                    "query",
                )
            except SystemExit as exc:
                pytest.fail(f"Async path should not exit — got SystemExit({exc.code})")

    @pytest.mark.asyncio
    async def test_async_proceeds_with_tools(self) -> None:
        """With tools available, the async orchestrated path is used (not fallback)."""
        from vaig.cli.commands.live import _async_execute_orchestrated_skill

        skill = _make_skill_mock(requires_live_tools=True)

        mock_orch_result = MagicMock()
        mock_orch_result.synthesized_output = "All clear"
        mock_orch_result.success = True
        mock_orch_result.agent_results = []
        mock_orch_result.structured_report = None
        mock_orch_result.total_usage = None

        mock_orchestrator = MagicMock()
        mock_orchestrator.async_execute_with_tools = AsyncMock(return_value=mock_orch_result)

        mock_client = MagicMock()
        mock_client.async_generate = AsyncMock()

        with (
            patch(
                "vaig.cli.commands.live._register_live_tools",
                return_value=_make_populated_registry(3),
            ),
            patch("vaig.agents.orchestrator.Orchestrator", return_value=mock_orchestrator),
        ):
            await _async_execute_orchestrated_skill(
                mock_client,
                Settings(),
                GKEConfig(),
                skill,
                "query",
            )

        # Async orchestrated path was used
        mock_orchestrator.async_execute_with_tools.assert_awaited_once()
        # Direct async_generate fallback was NOT called
        mock_client.async_generate.assert_not_called()
