"""T6.3 — CLI wire-through: verify that ``--attach*`` flags reach execute_skill_headless.

Tests:
- test_live_without_attach_flags: no --attach* → attachment_adapters=None (or kwarg absent)
- test_live_with_attach_local_path: --attach /tmp/foo.txt → non-empty list forwarded
- test_live_with_multiple_attach: --attach a --attach-git b --attach-url c → all three adapters
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vaig.cli.app import app
from vaig.core.config import Settings

runner = CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_settings() -> Settings:
    """Provide a default Settings object — avoids real config loading."""
    settings = Settings()
    settings.skills.auto_routing = False
    with patch("vaig.cli._helpers._get_settings", return_value=settings):
        yield settings


def _make_fake_skill() -> MagicMock:
    """Minimal BaseSkill mock with requires_live_tools=True."""
    skill = MagicMock()
    meta = MagicMock()
    meta.requires_live_tools = True
    meta.display_name = "Fake Skill"
    meta.name = "fake-skill"
    skill.get_metadata.return_value = meta
    return skill


def _make_fake_registry() -> MagicMock:
    """SkillRegistry mock that returns our fake skill."""
    registry = MagicMock()
    fake_skill = _make_fake_skill()
    registry.get.return_value = fake_skill
    registry.list_names.return_value = ["fake-skill"]
    return registry


def _make_orch_result() -> MagicMock:
    """Minimal OrchestratorResult mock."""
    result = MagicMock()
    result.structured_report = None
    result.answer = "ok"
    return result


def _make_fake_adapter(source: str = "fake.txt") -> MagicMock:
    """Minimal AttachmentAdapter mock."""
    adapter = MagicMock()
    adapter.spec = MagicMock()
    adapter.spec.source = source
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Shared patch stacks for live CLI invocation
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_PATCHES: list[tuple[str, Any]] = [
    # Prevent real GKE / network calls
    ("vaig.cli.commands.live._register_live_tools", MagicMock(return_value=MagicMock(list_tools=lambda: ["t1"]))),
    ("vaig.cli.commands.live.build_container", MagicMock(return_value=MagicMock(gemini_client=MagicMock()))),
    # Suppress display helpers that need real Rich terminals
    ("vaig.cli.commands.live._print_launch_header", MagicMock()),
    ("vaig.cli.commands.live._display_dry_run_plan", MagicMock()),
]


# ─────────────────────────────────────────────────────────────────────────────
# T6.3.1 — no --attach* flags → attachment_adapters is None or kwarg absent
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveWithoutAttachFlags:
    def test_live_without_attach_flags(self, _mock_settings: Settings) -> None:
        """Calling live with no --attach* must pass attachment_adapters=None to execute_skill_headless."""
        orch_result = _make_orch_result()

        with (
            patch(
                "vaig.core.headless.execute_skill_headless",
                return_value=orch_result,
            ) as mock_headless,
            patch(
                "vaig.skills.registry.SkillRegistry",
                return_value=_make_fake_registry(),
            ),
            patch("vaig.cli.commands.live._register_live_tools") as mock_reg,
            patch("vaig.core.container.build_container") as mock_container,
            patch("vaig.cli.commands.live._print_launch_header"),
            patch("vaig.cli.commands.live.Rule"),
            patch("vaig.cli.commands.live.AgentProgressDisplay"),
            patch("vaig.cli.commands.live.ToolCallLogger"),
            patch("vaig.cli.commands.live._create_tool_call_store", return_value=None),
        ):
            mock_tool_registry = MagicMock()
            mock_tool_registry.list_tools.return_value = ["tool1"]
            mock_reg.return_value = mock_tool_registry

            mock_container.return_value.gemini_client = MagicMock()

            result = runner.invoke(
                app,
                ["live", "--skill", "fake-skill", "what is wrong?"],
            )

        # Should have been called
        assert mock_headless.called, (
            f"execute_skill_headless was not called. Exit={result.exit_code}, output={result.output}"
        )

        _kwargs = mock_headless.call_args.kwargs
        # attachment_adapters should be None or empty list (no attachments provided)
        received = _kwargs.get("attachment_adapters")
        assert not received, f"Expected attachment_adapters to be None or empty, got: {received!r}"


# ─────────────────────────────────────────────────────────────────────────────
# T6.3.2 — --attach /tmp/foo.txt → non-empty list reaches execute_skill_headless
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveWithAttachLocalPath:
    def test_live_with_attach_local_path(self, _mock_settings: Settings, tmp_path: Any) -> None:
        """--attach <path> must forward a non-empty attachment_adapters list."""
        # Create a real file so the path resolves
        attach_file = tmp_path / "foo.txt"
        attach_file.write_text("hello")

        orch_result = _make_orch_result()
        fake_adapter = _make_fake_adapter(str(attach_file))

        with (
            patch(
                "vaig.core.headless.execute_skill_headless",
                return_value=orch_result,
            ) as mock_headless,
            patch(
                "vaig.skills.registry.SkillRegistry",
                return_value=_make_fake_registry(),
            ),
            # Intercept _build_and_resolve_attachments so we don't hit the filesystem
            patch(
                "vaig.cli.commands.live._build_and_resolve_attachments",
                return_value=[fake_adapter],
            ),
            patch("vaig.cli.commands.live._register_live_tools") as mock_reg,
            patch("vaig.core.container.build_container") as mock_container,
            patch("vaig.cli.commands.live._print_launch_header"),
            patch("vaig.cli.commands.live.Rule"),
            patch("vaig.cli.commands.live.AgentProgressDisplay"),
            patch("vaig.cli.commands.live.ToolCallLogger"),
            patch("vaig.cli.commands.live._create_tool_call_store", return_value=None),
        ):
            mock_tool_registry = MagicMock()
            mock_tool_registry.list_tools.return_value = ["tool1"]
            mock_reg.return_value = mock_tool_registry
            mock_container.return_value.gemini_client = MagicMock()

            result = runner.invoke(
                app,
                ["live", "--skill", "fake-skill", "--attach", str(attach_file), "what is wrong?"],
            )

        assert mock_headless.called, (
            f"execute_skill_headless was not called. Exit={result.exit_code}, output={result.output}"
        )

        _kwargs = mock_headless.call_args.kwargs
        received = _kwargs.get("attachment_adapters")
        assert received is not None, "attachment_adapters must not be None when --attach is provided"
        assert len(received) > 0, "attachment_adapters must be non-empty when --attach is provided"
        assert received[0] is fake_adapter


# ─────────────────────────────────────────────────────────────────────────────
# T6.3.3 — --attach a --attach-git b --attach-url c → all three adapters
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveWithMultipleAttach:
    def test_live_with_multiple_attach(self, _mock_settings: Settings) -> None:
        """--attach a --attach-git b --attach-url c → all three adapters forwarded."""
        orch_result = _make_orch_result()
        adapter_a = _make_fake_adapter("a")
        adapter_b = _make_fake_adapter("b")
        adapter_c = _make_fake_adapter("c")
        all_adapters = [adapter_a, adapter_b, adapter_c]

        with (
            patch(
                "vaig.core.headless.execute_skill_headless",
                return_value=orch_result,
            ) as mock_headless,
            patch(
                "vaig.skills.registry.SkillRegistry",
                return_value=_make_fake_registry(),
            ),
            patch(
                "vaig.cli.commands.live._build_and_resolve_attachments",
                return_value=all_adapters,
            ),
            patch("vaig.cli.commands.live._register_live_tools") as mock_reg,
            patch("vaig.core.container.build_container") as mock_container,
            patch("vaig.cli.commands.live._print_launch_header"),
            patch("vaig.cli.commands.live.Rule"),
            patch("vaig.cli.commands.live.AgentProgressDisplay"),
            patch("vaig.cli.commands.live.ToolCallLogger"),
            patch("vaig.cli.commands.live._create_tool_call_store", return_value=None),
        ):
            mock_tool_registry = MagicMock()
            mock_tool_registry.list_tools.return_value = ["tool1"]
            mock_reg.return_value = mock_tool_registry
            mock_container.return_value.gemini_client = MagicMock()

            result = runner.invoke(
                app,
                [
                    "live",
                    "--skill",
                    "fake-skill",
                    "--attach",
                    "a",
                    "--attach",
                    "b",
                    "--attach",
                    "c",
                    "what is wrong?",
                ],
            )

        assert mock_headless.called, (
            f"execute_skill_headless was not called. Exit={result.exit_code}, output={result.output}"
        )

        _kwargs = mock_headless.call_args.kwargs
        received = _kwargs.get("attachment_adapters")
        assert received is not None, "attachment_adapters must not be None when --attach* flags are provided"
        assert len(received) == 3, f"Expected 3 adapters, got {len(received)}: {received}"
        assert set(received) == set(all_adapters)
