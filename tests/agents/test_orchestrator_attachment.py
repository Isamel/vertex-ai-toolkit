from __future__ import annotations

"""Unit tests for ATT-10 G4: attachment_context propagation through orchestrator.py."""

from unittest.mock import MagicMock, patch

import pytest

MARKER = "MARKER_ATT10_G4"
ATTACHMENT_HEADER = "## Attached Context"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator() -> Orchestrator:  # noqa: F821
    from vaig.agents.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._gemini_client = MagicMock()
    return orch


def _make_skill_mock(marker: str | None = None) -> MagicMock:
    """Return a BaseSkill mock whose get_agents_config captures kwargs."""
    skill = MagicMock()
    skill.get_metadata.return_value = MagicMock(name="test_skill")

    captured: dict[str, object] = {}

    def _get_agents_config(**kwargs: object) -> list[dict]:
        captured.update(kwargs)
        si = ATTACHMENT_HEADER + "\n" + marker if marker else "plain prompt"
        return [
            {
                "name": "test_agent",
                "role": "Test",
                "requires_tools": False,
                "system_instruction": si,
            }
        ]

    skill.get_agents_config.side_effect = _get_agents_config
    skill._captured = captured
    return skill


def _make_tool_registry() -> MagicMock:
    tr = MagicMock()
    tr.get_tools_for_categories.return_value = []
    return tr


# ---------------------------------------------------------------------------
# T4.2-A: execute_with_tools forwards attachment_context
# ---------------------------------------------------------------------------


def test_execute_with_tools_forwards_attachment_context() -> None:
    """`execute_with_tools` must accept and forward `attachment_context` to
    `skill.get_agents_config`."""
    orch = _make_orchestrator()
    skill = _make_skill_mock(marker=MARKER)

    # Stub out the heavy internals so we don't need real GCP creds.
    with patch.object(
        orch,
        "_execute_with_tools_impl",
        wraps=lambda *a, **kw: _stub_impl(skill, **kw),
    ):
        orch.execute_with_tools(
            "test query",
            skill,
            _make_tool_registry(),
            attachment_context=MARKER,
        )

    assert "attachment_context" in skill._captured, "attachment_context was not forwarded to skill.get_agents_config"
    assert skill._captured["attachment_context"] == MARKER


def _stub_impl(skill: MagicMock, **kwargs: object) -> MagicMock:
    """Minimal stub that calls skill.get_agents_config and returns a mock result."""
    skill.get_agents_config(
        namespace=kwargs.get("gke_namespace", ""),
        location=kwargs.get("gke_location", ""),
        cluster_name=kwargs.get("gke_cluster_name", ""),
        attachment_context=kwargs.get("attachment_context"),
    )
    result = MagicMock()
    result.output = "stub"
    return result


# ---------------------------------------------------------------------------
# T4.2-B: async_execute_with_tools forwards attachment_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_execute_with_tools_forwards_attachment_context() -> None:
    """`async_execute_with_tools` must accept and forward `attachment_context`."""
    orch = _make_orchestrator()
    skill = _make_skill_mock(marker=MARKER)

    async def _async_stub(*args: object, **kwargs: object) -> MagicMock:
        _stub_impl(skill, **kwargs)
        result = MagicMock()
        result.output = "stub"
        return result

    with patch.object(orch, "_async_execute_with_tools_impl", side_effect=_async_stub):
        await orch.async_execute_with_tools(
            "test query",
            skill,
            _make_tool_registry(),
            attachment_context=MARKER,
        )

    assert "attachment_context" in skill._captured, (
        "attachment_context was not forwarded to skill.get_agents_config (async path)"
    )
    assert skill._captured["attachment_context"] == MARKER


# ---------------------------------------------------------------------------
# T4.2-C: no attachment_context → get_agents_config called without it
# ---------------------------------------------------------------------------


def test_execute_with_tools_no_attachment_context() -> None:
    """When `attachment_context` is omitted, `skill.get_agents_config` must
    NOT receive a truthy `attachment_context` value."""
    orch = _make_orchestrator()
    skill = _make_skill_mock()

    with patch.object(
        orch,
        "_execute_with_tools_impl",
        wraps=lambda *a, **kw: _stub_impl(skill, **kw),
    ):
        orch.execute_with_tools(
            "test query",
            skill,
            _make_tool_registry(),
        )

    ctx = skill._captured.get("attachment_context")
    assert not ctx, f"Expected attachment_context to be falsy when not passed, got: {ctx!r}"
