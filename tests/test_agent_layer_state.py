"""Unit tests for Batch 2 — Agent Layer state integration (Phase 2: Infrastructure).

Covers:
- AgentResult.state_patch field (Task 2.1)
- ToolAwareAgent.execute() accepts state parameter (Task 2.2)
- SpecialistAgent.execute() accepts state parameter (Task 2.3)
- BaseSkill.get_initial_state() default returns None (Task 2.4)
- BaseSkill subclass can override get_initial_state() (Task 2.4)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from vaig.agents.base import AgentConfig, AgentResult
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent
from vaig.core.models import PipelineState
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


class _MinimalSkill(BaseSkill):
    """Minimal concrete BaseSkill for testing (does not override get_initial_state)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="test-skill",
            display_name="Test Skill",
            description="For unit tests.",
        )

    def get_system_instruction(self) -> str:
        return "You are a test agent."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Phase: {phase}, input: {user_input}"


class _StateSkill(BaseSkill):
    """Concrete BaseSkill that overrides get_initial_state with a real PipelineState."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="state-skill",
            display_name="State Skill",
            description="Returns initial state.",
        )

    def get_system_instruction(self) -> str:
        return "You are a stateful agent."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Phase: {phase}, input: {user_input}"

    def get_initial_state(self) -> PipelineState:
        return PipelineState(metrics={"skill": "state-skill"})


class _SamplePatch(BaseModel):
    """Sample Pydantic model used as a state_patch value."""

    key: str
    value: int


# ── Task 2.1: AgentResult.state_patch ────────────────────────────────────────


class TestAgentResultStatePatch:
    """AgentResult carries state_patch field — default None, accepts dict or BaseModel."""

    def test_state_patch_defaults_to_none(self) -> None:
        result = AgentResult(agent_name="a", content="done")
        assert result.state_patch is None

    def test_state_patch_accepts_none_explicitly(self) -> None:
        result = AgentResult(agent_name="a", content="done", state_patch=None)
        assert result.state_patch is None

    def test_state_patch_accepts_plain_dict(self) -> None:
        patch: dict = {"errors": ["something failed"], "metrics": {"cpu": 95}}
        result = AgentResult(agent_name="a", content="done", state_patch=patch)
        assert result.state_patch == patch
        assert result.state_patch["metrics"]["cpu"] == 95  # type: ignore[index]

    def test_state_patch_accepts_pydantic_base_model(self) -> None:
        patch = _SamplePatch(key="hello", value=42)
        result = AgentResult(agent_name="a", content="done", state_patch=patch)
        assert result.state_patch is patch
        assert isinstance(result.state_patch, BaseModel)

    def test_state_patch_accepts_pipeline_state(self) -> None:
        """PipelineState is also a BaseModel — must be accepted."""
        patch = PipelineState(errors=["oops"])
        result = AgentResult(agent_name="a", content="done", state_patch=patch)
        assert isinstance(result.state_patch, PipelineState)
        assert result.state_patch.errors == ["oops"]  # type: ignore[union-attr]

    def test_other_fields_unaffected(self) -> None:
        """Existing fields still work correctly after adding state_patch."""
        result = AgentResult(
            agent_name="agent",
            content="output",
            success=False,
            usage={"total_tokens": 10},
            metadata={"model": "gemini-2.5-pro"},
            state_patch={"errors": ["err"]},
        )
        assert result.agent_name == "agent"
        assert result.success is False
        assert result.usage["total_tokens"] == 10
        assert result.metadata["model"] == "gemini-2.5-pro"


# ── Task 2.2: ToolAwareAgent.execute accepts state ───────────────────────────


class TestToolAwareAgentStateSignature:
    """ToolAwareAgent.execute() accepts optional state kwarg without error."""

    def _make_agent(self) -> ToolAwareAgent:
        """Build a ToolAwareAgent backed by a mocked client."""
        mock_client = MagicMock()
        mock_client.generate_with_tools.return_value = MagicMock(
            text="result text",
            function_calls=[],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            model="gemini-2.5-pro",
            finish_reason="stop",
        )
        registry = ToolRegistry()
        return ToolAwareAgent(
            system_instruction="You are a test agent.",
            tool_registry=registry,
            model="gemini-2.5-pro",
            name="test-tool-agent",
            client=mock_client,
            max_iterations=2,
        )

    def test_execute_accepts_state_none(self) -> None:
        """execute() can be called with state=None (explicit)."""
        agent = self._make_agent()
        result = agent.execute("do something", state=None)
        assert isinstance(result, AgentResult)

    def test_execute_accepts_empty_pipeline_state(self) -> None:
        """execute() can be called with state=PipelineState()."""
        agent = self._make_agent()
        state = PipelineState()
        result = agent.execute("do something", state=state)
        assert isinstance(result, AgentResult)

    def test_execute_accepts_populated_pipeline_state(self) -> None:
        """execute() can be called with a populated state."""
        agent = self._make_agent()
        state = PipelineState(
            findings=[{"issue": "high cpu"}],
            metrics={"cpu": 95},
            errors=[],
        )
        result = agent.execute("do something", state=state)
        assert isinstance(result, AgentResult)

    def test_execute_without_state_still_works(self) -> None:
        """execute() without state kwarg is fully backward-compatible."""
        agent = self._make_agent()
        result = agent.execute("do something")
        assert isinstance(result, AgentResult)


# ── Task 2.3: SpecialistAgent.execute accepts state ──────────────────────────


class TestSpecialistAgentStateSignature:
    """SpecialistAgent.execute() accepts optional state kwarg without error."""

    def _make_agent(self) -> SpecialistAgent:
        """Build a SpecialistAgent backed by a mocked client."""
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(
            text="specialist output",
            usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
            model="gemini-2.5-pro",
            finish_reason="stop",
        )
        config = AgentConfig(
            name="specialist-test",
            role="analyzer",
            system_instruction="You analyze things.",
        )
        return SpecialistAgent(config, mock_client)

    def test_execute_accepts_state_none(self) -> None:
        """execute() can be called with state=None (explicit)."""
        agent = self._make_agent()
        result = agent.execute("analyze this", state=None)
        assert isinstance(result, AgentResult)

    def test_execute_accepts_empty_pipeline_state(self) -> None:
        """execute() can be called with state=PipelineState()."""
        agent = self._make_agent()
        state = PipelineState()
        result = agent.execute("analyze this", state=state)
        assert isinstance(result, AgentResult)

    def test_execute_accepts_populated_pipeline_state(self) -> None:
        """execute() can be called with a populated state."""
        agent = self._make_agent()
        state = PipelineState(errors=["prior failure"], metrics={"count": 3})
        result = agent.execute("analyze this", state=state)
        assert isinstance(result, AgentResult)

    def test_execute_without_state_still_works(self) -> None:
        """execute() without state kwarg is fully backward-compatible."""
        agent = self._make_agent()
        result = agent.execute("analyze this")
        assert isinstance(result, AgentResult)


# ── Task 2.4: BaseSkill.get_initial_state ────────────────────────────────────


class TestBaseSkillGetInitialState:
    """BaseSkill.get_initial_state() defaults to None; subclasses can override."""

    def test_default_returns_none(self) -> None:
        """Concrete subclass that does NOT override returns None."""
        skill = _MinimalSkill()
        assert skill.get_initial_state() is None

    def test_default_returns_none_multiple_calls_are_consistent(self) -> None:
        """Multiple calls all return None — no side effects."""
        skill = _MinimalSkill()
        assert skill.get_initial_state() is None
        assert skill.get_initial_state() is None

    def test_subclass_override_returns_pipeline_state(self) -> None:
        """Subclass that overrides get_initial_state returns a PipelineState."""
        skill = _StateSkill()
        state = skill.get_initial_state()
        assert state is not None
        assert isinstance(state, PipelineState)

    def test_subclass_override_returns_correct_values(self) -> None:
        """Overridden method returns the correct seeded values."""
        skill = _StateSkill()
        state = skill.get_initial_state()
        assert state is not None
        assert state.metrics == {"skill": "state-skill"}
        assert state.findings == []
        assert state.errors == []

    def test_method_is_not_abstract(self) -> None:
        """get_initial_state is a concrete method — can be instantiated without override."""
        skill = _MinimalSkill()
        # If it were abstract, instantiating _MinimalSkill would raise TypeError
        # The fact that we reach here proves it's not abstract
        result = skill.get_initial_state()
        assert result is None
