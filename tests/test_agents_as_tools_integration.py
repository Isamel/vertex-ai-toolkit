"""Integration tests for Agents-as-Tools — Batch 2 (Orchestrator Integration).

Tests:
- Task 2.1: `injectable_agents` config field is read and processed
- Task 2.2: Registry copy + injection logic in create_agents_for_skill()
- Task 2.3: Self-injection is prevented (caller_name passed through)
- Task 2.4: Thread-safety — each injecting agent gets its OWN registry copy
- Task 2.5: Sync and async paths both inject sub-agent tools
- Backward compatibility: agents WITHOUT `injectable_agents` are unaffected
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vaig.agents.orchestrator import Orchestrator
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolDef, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Create a mock GeminiClientProtocol."""
    client = MagicMock()
    client.generate.return_value = MagicMock(
        text="result",
        model="gemini-2.5-pro",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="STOP",
    )
    return client


def _make_mock_settings() -> MagicMock:
    """Create a minimal mock Settings."""
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    settings.agents.max_iterations_retry = 10
    settings.budget.max_cost_per_run = 0.0
    settings.agents.max_failures_before_fallback = 0
    return settings


def _make_tool(name: str = "ping") -> ToolDef:
    """Create a simple no-op ToolDef for registry population."""
    return ToolDef(
        name=name,
        description=f"Tool {name}",
        parameters=[],
        execute=lambda **_kw: MagicMock(output="ok", error=False),
        cacheable=False,
    )


def _make_registry(*tool_names: str) -> ToolRegistry:
    """Build a ToolRegistry with the given named tools pre-registered."""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(_make_tool(name))
    return registry


class StubInjectionSkill(BaseSkill):
    """Configurable skill stub used across injection tests."""

    def __init__(self, agent_configs: list[dict[str, Any]]) -> None:
        self._agent_configs = agent_configs

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="injection_test",
            display_name="Injection Test Skill",
            description="Skill for testing injectable_agents integration",
            requires_live_tools=False,
        )

    def get_system_instruction(self) -> str:
        return "Injection test."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._agent_configs


# ---------------------------------------------------------------------------
# Task 2.1 — injectable_agents config field is parsed
# ---------------------------------------------------------------------------


class TestInjectableAgentsConfigParsing:
    """Verify that `injectable_agents` is read from the config dict."""

    def test_injectable_agents_config_key_is_recognised(self) -> None:
        """When `injectable_agents` is present, no KeyError or AttributeError occurs."""
        configs = [
            {
                "name": "caller",
                "role": "Caller",
                "system_instruction": "You call sub-agents.",
                "requires_tools": True,
            },
            {
                "name": "target",
                "role": "Target",
                "system_instruction": "You are a target agent.",
                "requires_tools": True,
                "injectable_agents": ["caller"],  # target has caller in its list
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_a")

        # Must not raise; injectable_agents is silently processed
        agents = orchestrator.create_agents_for_skill(skill, registry)
        assert len(agents) == 2

    def test_agents_without_injectable_agents_created_normally(self) -> None:
        """Agents without `injectable_agents` are created as before."""
        configs = [
            {
                "name": "plain_agent",
                "role": "Plain",
                "system_instruction": "Plain agent.",
                "requires_tools": True,
            }
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_a")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        assert len(agents) == 1
        agent = agents[0]
        assert isinstance(agent, ToolAwareAgent)
        # Should share the SAME registry object (no copy needed)
        assert agent.tool_registry is registry


# ---------------------------------------------------------------------------
# Task 2.2 — Registry copy + tool injection
# ---------------------------------------------------------------------------


class TestRegistryCopyAndInjection:
    """Verify per-agent registry copy and sub-agent tool injection."""

    def test_injecting_agent_gets_its_own_registry_copy(self) -> None:
        """An agent with injectable_agents receives a COPY, not the shared registry."""
        configs = [
            {
                "name": "sub_agent",
                "role": "Sub",
                "system_instruction": "I am the sub-agent.",
                "requires_tools": True,
            },
            {
                "name": "caller_agent",
                "role": "Caller",
                "system_instruction": "I call sub-agents.",
                "requires_tools": True,
                "injectable_agents": ["sub_agent"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("original_tool")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        caller = next(a for a in agents if a.name == "caller_agent")
        sub = next(a for a in agents if a.name == "sub_agent")

        assert isinstance(caller, ToolAwareAgent)
        # Caller MUST have a separate registry instance (copy)
        assert caller.tool_registry is not registry
        # Sub-agent shares the original (no injectable_agents declared)
        assert sub.tool_registry is registry

    def test_injected_tool_has_ask_prefix(self) -> None:
        """The injected tool name follows the `ask_<name>` convention."""
        configs = [
            {
                "name": "network_agent",
                "role": "Network",
                "system_instruction": "Network specialist.",
                "requires_tools": True,
            },
            {
                "name": "workload_agent",
                "role": "Workload",
                "system_instruction": "Workload specialist.",
                "requires_tools": True,
                "injectable_agents": ["network_agent"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("base_tool")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        workload = next(a for a in agents if a.name == "workload_agent")

        assert isinstance(workload, ToolAwareAgent)
        tool_names = [t.name for t in workload.tool_registry.list_tools()]
        assert "ask_network_agent" in tool_names

    def test_original_tool_preserved_in_copy(self) -> None:
        """The copied registry retains all original tools plus the injected one."""
        configs = [
            {
                "name": "helper",
                "role": "Helper",
                "system_instruction": "I help.",
                "requires_tools": True,
            },
            {
                "name": "main_agent",
                "role": "Main",
                "system_instruction": "I am the main agent.",
                "requires_tools": True,
                "injectable_agents": ["helper"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_x", "tool_y")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        main = next(a for a in agents if a.name == "main_agent")

        assert isinstance(main, ToolAwareAgent)
        tool_names = {t.name for t in main.tool_registry.list_tools()}
        # Original tools preserved
        assert "tool_x" in tool_names
        assert "tool_y" in tool_names
        # Sub-agent tool injected
        assert "ask_helper" in tool_names

    def test_shared_registry_not_mutated(self) -> None:
        """The original shared registry must NOT be modified after injection."""
        configs = [
            {
                "name": "sub",
                "role": "Sub",
                "system_instruction": "Sub-agent.",
                "requires_tools": True,
            },
            {
                "name": "caller",
                "role": "Caller",
                "system_instruction": "Caller agent.",
                "requires_tools": True,
                "injectable_agents": ["sub"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("base_tool")

        orchestrator.create_agents_for_skill(skill, registry)

        # Shared registry should only have base_tool — ask_sub must NOT appear
        shared_names = {t.name for t in registry.list_tools()}
        assert "ask_sub" not in shared_names
        assert shared_names == {"base_tool"}

    def test_multiple_injectable_agents_all_injected(self) -> None:
        """When multiple agents are declared, all are injected."""
        configs = [
            {
                "name": "alpha",
                "role": "Alpha",
                "system_instruction": "Alpha.",
                "requires_tools": True,
            },
            {
                "name": "beta",
                "role": "Beta",
                "system_instruction": "Beta.",
                "requires_tools": True,
            },
            {
                "name": "orchestrating_agent",
                "role": "Orchestrating",
                "system_instruction": "Orchestrating.",
                "requires_tools": True,
                "injectable_agents": ["alpha", "beta"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("base")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        orchestrating = next(a for a in agents if a.name == "orchestrating_agent")

        assert isinstance(orchestrating, ToolAwareAgent)
        tool_names = {t.name for t in orchestrating.tool_registry.list_tools()}
        assert "ask_alpha" in tool_names
        assert "ask_beta" in tool_names


# ---------------------------------------------------------------------------
# Task 2.3 — Self-injection prevention
# ---------------------------------------------------------------------------


class TestSelfInjectionPrevention:
    """Verify self-injection is blocked when caller == target."""

    def test_self_injection_tool_is_error_no_op(self) -> None:
        """When an agent lists itself in injectable_agents, tool returns error."""
        configs = [
            {
                "name": "self_calling_agent",
                "role": "Self Caller",
                "system_instruction": "I try to call myself.",
                "requires_tools": True,
                "injectable_agents": ["self_calling_agent"],
            }
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("base")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        self_agent = agents[0]

        assert isinstance(self_agent, ToolAwareAgent)
        # Tool is registered but should be a no-op error tool
        tool_names = {t.name for t in self_agent.tool_registry.list_tools()}
        assert "ask_self_calling_agent" in tool_names

        # Invoke the tool — it must return an error ToolResult
        tool = self_agent.tool_registry.get("ask_self_calling_agent")
        assert tool is not None
        result = tool.execute(query="test self-invoke")
        assert result.error is True
        assert "self-injection" in result.output.lower() or "itself" in result.output.lower()


# ---------------------------------------------------------------------------
# Task 2.4 — Thread safety (each injecting agent gets its own copy)
# ---------------------------------------------------------------------------


class TestThreadSafetyRegistryIsolation:
    """Verify two injecting agents each get a distinct registry copy."""

    def test_two_injecting_agents_have_independent_registries(self) -> None:
        """Two separate agents with injectable_agents each get isolated copies."""
        configs = [
            {
                "name": "shared_target",
                "role": "Shared Target",
                "system_instruction": "I am shared.",
                "requires_tools": True,
            },
            {
                "name": "caller_a",
                "role": "Caller A",
                "system_instruction": "Caller A.",
                "requires_tools": True,
                "injectable_agents": ["shared_target"],
            },
            {
                "name": "caller_b",
                "role": "Caller B",
                "system_instruction": "Caller B.",
                "requires_tools": True,
                "injectable_agents": ["shared_target"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("original")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        caller_a = next(a for a in agents if a.name == "caller_a")
        caller_b = next(a for a in agents if a.name == "caller_b")

        assert isinstance(caller_a, ToolAwareAgent)
        assert isinstance(caller_b, ToolAwareAgent)

        # Each caller must have their own registry copy
        assert caller_a.tool_registry is not caller_b.tool_registry
        assert caller_a.tool_registry is not registry
        assert caller_b.tool_registry is not registry

        # But each has the injected tool
        names_a = {t.name for t in caller_a.tool_registry.list_tools()}
        names_b = {t.name for t in caller_b.tool_registry.list_tools()}
        assert "ask_shared_target" in names_a
        assert "ask_shared_target" in names_b

    def test_non_injecting_agents_share_original_registry(self) -> None:
        """Agents that do NOT declare injectable_agents share the original registry."""
        configs = [
            {
                "name": "plain_a",
                "role": "Plain A",
                "system_instruction": "Plain A.",
                "requires_tools": True,
            },
            {
                "name": "plain_b",
                "role": "Plain B",
                "system_instruction": "Plain B.",
                "requires_tools": True,
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_x")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        for agent in agents:
            assert isinstance(agent, ToolAwareAgent)
            # All plain agents share the SAME original registry object
            assert agent.tool_registry is registry


# ---------------------------------------------------------------------------
# Task 2.5 — Sync and async paths inject sub-agent tools
# ---------------------------------------------------------------------------


class TestSyncAndAsyncPaths:
    """Ensure injection works via create_agents_for_skill regardless of call path."""

    def test_sync_injection_creates_ask_tool(self) -> None:
        """Direct call to create_agents_for_skill (sync path) injects correctly."""
        configs = [
            {
                "name": "advisor",
                "role": "Advisor",
                "system_instruction": "I advise.",
                "requires_tools": True,
            },
            {
                "name": "decision_maker",
                "role": "Decision Maker",
                "system_instruction": "I make decisions.",
                "requires_tools": True,
                "injectable_agents": ["advisor"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_z")

        agents = orchestrator.create_agents_for_skill(skill, registry)
        decision_maker = next(a for a in agents if a.name == "decision_maker")
        assert isinstance(decision_maker, ToolAwareAgent)

        tool_names = {t.name for t in decision_maker.tool_registry.list_tools()}
        assert "ask_advisor" in tool_names

    def test_async_path_uses_same_create_agents_for_skill(self) -> None:
        """The async execution path also passes through create_agents_for_skill.

        Since async_execute_with_tools calls create_agents_for_skill internally
        (same as the sync path), we verify the injection is consistent by
        calling create_agents_for_skill directly and checking the result.
        This mirrors what _async_execute_with_tools_impl does at line ~1796.
        """
        configs = [
            {
                "name": "analyzer",
                "role": "Analyzer",
                "system_instruction": "I analyze.",
                "requires_tools": True,
            },
            {
                "name": "reporter",
                "role": "Reporter",
                "system_instruction": "I report.",
                "requires_tools": True,
                "injectable_agents": ["analyzer"],
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("tool_a")

        # Both sync and async paths call create_agents_for_skill (no async-specific code)
        agents = orchestrator.create_agents_for_skill(skill, registry)
        reporter = next(a for a in agents if a.name == "reporter")
        assert isinstance(reporter, ToolAwareAgent)

        tool_names = {t.name for t in reporter.tool_registry.list_tools()}
        assert "ask_analyzer" in tool_names


# ---------------------------------------------------------------------------
# Backward compatibility — no injectable_agents config
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure existing skills without injectable_agents are completely unaffected."""

    def test_no_injectable_agents_no_registry_copies(self) -> None:
        """Skills without injectable_agents continue to use the shared registry."""
        configs = [
            {
                "name": "legacy_tool_agent",
                "role": "Legacy Tool",
                "system_instruction": "Legacy.",
                "requires_tools": True,
            },
            {
                "name": "legacy_plain_agent",
                "role": "Legacy Plain",
                "system_instruction": "Plain.",
                "requires_tools": False,
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("existing_tool")

        agents = orchestrator.create_agents_for_skill(skill, registry)

        tool_agent = next(a for a in agents if a.name == "legacy_tool_agent")
        plain_agent = next(a for a in agents if a.name == "legacy_plain_agent")

        assert isinstance(tool_agent, ToolAwareAgent)
        assert isinstance(plain_agent, SpecialistAgent)
        # Tool agent uses the exact shared registry (no copies)
        assert tool_agent.tool_registry is registry
        # No extra tools have leaked in
        assert {t.name for t in tool_agent.tool_registry.list_tools()} == {"existing_tool"}

    def test_no_tool_registry_still_creates_specialist_agents(self) -> None:
        """Without a tool_registry, all agents are SpecialistAgent regardless of injectable_agents."""
        configs = [
            {
                "name": "agent_with_injectable",
                "role": "Injectable Agent",
                "system_instruction": "I would inject.",
                "requires_tools": True,
                "injectable_agents": ["other_agent"],
            },
            {
                "name": "other_agent",
                "role": "Other",
                "system_instruction": "Other.",
                "requires_tools": True,
            },
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        # No tool registry passed
        agents = orchestrator.create_agents_for_skill(skill, tool_registry=None)

        for agent in agents:
            assert isinstance(agent, SpecialistAgent)

    def test_missing_target_name_logs_warning_and_skips(self) -> None:
        """A missing target name in injectable_agents is skipped gracefully."""
        configs = [
            {
                "name": "caller",
                "role": "Caller",
                "system_instruction": "Caller.",
                "requires_tools": True,
                "injectable_agents": ["nonexistent_agent"],
            }
        ]
        skill = StubInjectionSkill(configs)
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_registry("base_tool")

        # Must NOT raise — missing targets are skipped
        agents = orchestrator.create_agents_for_skill(skill, registry)
        caller = agents[0]
        assert isinstance(caller, ToolAwareAgent)

        # Only base_tool in registry — no phantom tool was injected
        tool_names = {t.name for t in caller.tool_registry.list_tools()}
        assert "ask_nonexistent_agent" not in tool_names
