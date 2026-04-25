"""Tests for agent base classes and data structures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from vaig.agents.base import AgentConfig, AgentMessage, AgentResult, AgentRole
from vaig.agents.specialist import SpecialistAgent


class TestAgentRole:
    def test_values(self) -> None:
        assert AgentRole.ORCHESTRATOR == "orchestrator"
        assert AgentRole.SPECIALIST == "specialist"
        assert AgentRole.ASSISTANT == "assistant"

    def test_is_str_enum(self) -> None:
        assert isinstance(AgentRole.ORCHESTRATOR, str)


class TestAgentConfig:
    def test_required_fields(self) -> None:
        cfg = AgentConfig(
            name="test-agent",
            role="analyzer",
            system_instruction="You analyze data.",
        )
        assert cfg.name == "test-agent"
        assert cfg.role == "analyzer"
        assert cfg.system_instruction == "You analyze data."

    def test_defaults(self) -> None:
        cfg = AgentConfig(name="a", role="b", system_instruction="c")
        # model defaults to "" sentinel — callers resolve via effective_model()
        assert cfg.model == ""
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 16384

    def test_effective_model_with_default_sentinel(self) -> None:
        """effective_model() returns the passed default when model is empty."""
        cfg = AgentConfig(name="a", role="b", system_instruction="c")
        assert cfg.effective_model("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_custom_values(self) -> None:
        cfg = AgentConfig(
            name="fast-agent",
            role="summarizer",
            system_instruction="Summarize.",
            model="gemini-2.5-flash",
            temperature=0.0,
            max_output_tokens=1024,
        )
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.temperature == 0.0
        assert cfg.max_output_tokens == 1024


class TestAgentMessage:
    def test_basic(self) -> None:
        msg = AgentMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.agent_name == ""
        assert msg.metadata == {}

    def test_with_agent_name(self) -> None:
        msg = AgentMessage(
            role="agent",
            content="Response",
            agent_name="log_analyzer",
            metadata={"tokens": 42},
        )
        assert msg.agent_name == "log_analyzer"
        assert msg.metadata["tokens"] == 42


class TestAgentResult:
    def test_success(self) -> None:
        result = AgentResult(
            agent_name="test-agent",
            content="Analysis complete.",
        )
        assert result.agent_name == "test-agent"
        assert result.content == "Analysis complete."
        assert result.success is True
        assert result.usage == {}
        assert result.metadata == {}

    def test_failure(self) -> None:
        result = AgentResult(
            agent_name="broken",
            content="Error: API quota exceeded",
            success=False,
            metadata={"error_code": "QUOTA_EXCEEDED"},
        )
        assert result.success is False
        assert result.metadata["error_code"] == "QUOTA_EXCEEDED"

    def test_with_usage(self) -> None:
        result = AgentResult(
            agent_name="agent",
            content="Done",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert result.usage["total_tokens"] == 150


class TestSpecialistAgentFromConfigDict:
    """Tests for SpecialistAgent.from_config_dict defensive type checking."""

    def test_valid_config_dict(self) -> None:
        mock_client = MagicMock()
        config = {
            "name": "analyzer",
            "role": "log analyzer",
            "system_instruction": "Analyze logs.",
            "model": "gemini-2.5-flash",
            "temperature": 0.3,
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.name == "analyzer"
        assert agent.role == "log analyzer"
        assert agent.model == "gemini-2.5-flash"

    def test_config_dict_with_defaults(self) -> None:
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "helper",
            "system_instruction": "Help.",
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        # Default is sentinel "" — stored in _config.model.
        # agent.model falls back to client.current_model when sentinel is set,
        # so we assert on the raw config field to verify the sentinel is stored.
        assert agent._config.model == ""

    def test_sentinel_empty_string_passes_through(self) -> None:
        """Sentinel '' is stored as-is; resolution is the orchestrator's responsibility."""
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "helper",
            "system_instruction": "Help.",
            "model": "",
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent._config.model == ""

    def test_raises_type_error_on_list_input(self) -> None:
        """The .get() AttributeError bug — if a list is passed instead of a dict."""
        mock_client = MagicMock()
        with pytest.raises(TypeError, match="Expected dict"):
            SpecialistAgent.from_config_dict(["not", "a", "dict"], mock_client)  # type: ignore[arg-type]

    def test_raises_type_error_on_string_input(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(TypeError, match="Expected dict"):
            SpecialistAgent.from_config_dict("not a dict", mock_client)  # type: ignore[arg-type]

    def test_raises_key_error_on_missing_required_keys(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(KeyError):
            SpecialistAgent.from_config_dict({"name": "agent"}, mock_client)

    def test_reads_frequency_penalty(self) -> None:
        """frequency_penalty was previously dropped by from_config_dict — verify fix."""
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "helper",
            "system_instruction": "Help.",
            "frequency_penalty": 0.5,
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.config.frequency_penalty == 0.5

    def test_frequency_penalty_defaults_to_none(self) -> None:
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "helper",
            "system_instruction": "Help.",
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.config.frequency_penalty is None

    def test_reads_response_schema(self) -> None:
        """from_config_dict passes response_schema through to AgentConfig."""

        class MySchema(BaseModel):
            title: str

        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "reporter",
            "system_instruction": "Report.",
            "response_schema": MySchema,
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.config.response_schema is MySchema

    def test_reads_response_mime_type(self) -> None:
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "reporter",
            "system_instruction": "Report.",
            "response_mime_type": "application/json",
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.config.response_mime_type == "application/json"

    def test_schema_fields_default_to_none(self) -> None:
        """When not provided, response_schema and response_mime_type are None."""
        mock_client = MagicMock()
        config = {
            "name": "agent",
            "role": "helper",
            "system_instruction": "Help.",
        }
        agent = SpecialistAgent.from_config_dict(config, mock_client)
        assert agent.config.response_schema is None
        assert agent.config.response_mime_type is None


class TestAgentConfigSchemaFields:
    """Tests for response_schema and response_mime_type on AgentConfig."""

    def test_defaults_are_none(self) -> None:
        cfg = AgentConfig(name="a", role="b", system_instruction="c")
        assert cfg.response_schema is None
        assert cfg.response_mime_type is None

    def test_accepts_pydantic_model_class(self) -> None:
        class Report(BaseModel):
            summary: str

        cfg = AgentConfig(
            name="reporter",
            role="reporter",
            system_instruction="Report.",
            response_schema=Report,
        )
        assert cfg.response_schema is Report

    def test_accepts_response_mime_type(self) -> None:
        cfg = AgentConfig(
            name="reporter",
            role="reporter",
            system_instruction="Report.",
            response_mime_type="application/json",
        )
        assert cfg.response_mime_type == "application/json"

    def test_both_fields_set_together(self) -> None:
        class Report(BaseModel):
            summary: str

        cfg = AgentConfig(
            name="reporter",
            role="reporter",
            system_instruction="Report.",
            response_schema=Report,
            response_mime_type="application/json",
        )
        assert cfg.response_schema is Report
        assert cfg.response_mime_type == "application/json"


class TestSpecialistAgentSchemaForwarding:
    """Tests for execute() and async_execute() forwarding response_schema/mime_type."""

    def _make_agent(
        self,
        *,
        response_schema: type[BaseModel] | None = None,
        response_mime_type: str | None = None,
        frequency_penalty: float | None = None,
    ) -> SpecialistAgent:
        mock_client = MagicMock()
        config = AgentConfig(
            name="test-agent",
            role="reporter",
            system_instruction="Report.",
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            frequency_penalty=frequency_penalty,
        )
        return SpecialistAgent(config, mock_client)

    def _mock_generate_result(self) -> MagicMock:
        result = MagicMock()
        result.text = "ok"
        result.usage = {"total_tokens": 10}
        result.model = "gemini-2.5-pro"
        result.finish_reason = "STOP"
        return result

    def test_execute_forwards_schema_params(self) -> None:
        class MySchema(BaseModel):
            title: str

        agent = self._make_agent(
            response_schema=MySchema,
            response_mime_type="application/json",
        )
        agent._client.generate.return_value = self._mock_generate_result()

        agent.execute("test prompt")

        call_kwargs = agent._client.generate.call_args
        assert call_kwargs.kwargs["response_schema"] is MySchema
        assert call_kwargs.kwargs["response_mime_type"] == "application/json"

    def test_execute_omits_schema_params_when_none(self) -> None:
        agent = self._make_agent()
        agent._client.generate.return_value = self._mock_generate_result()

        agent.execute("test prompt")

        call_kwargs = agent._client.generate.call_args
        assert "response_schema" not in call_kwargs.kwargs
        assert "response_mime_type" not in call_kwargs.kwargs

    def test_execute_forwards_frequency_penalty(self) -> None:
        agent = self._make_agent(frequency_penalty=0.8)
        agent._client.generate.return_value = self._mock_generate_result()

        agent.execute("test prompt")

        call_kwargs = agent._client.generate.call_args
        assert call_kwargs.kwargs["frequency_penalty"] == 0.8

    def test_execute_omits_frequency_penalty_when_none(self) -> None:
        agent = self._make_agent()
        agent._client.generate.return_value = self._mock_generate_result()

        agent.execute("test prompt")

        call_kwargs = agent._client.generate.call_args
        assert "frequency_penalty" not in call_kwargs.kwargs

    async def test_async_execute_forwards_schema_params(self) -> None:
        class MySchema(BaseModel):
            title: str

        agent = self._make_agent(
            response_schema=MySchema,
            response_mime_type="application/json",
        )
        from unittest.mock import AsyncMock

        agent._client.async_generate = AsyncMock(return_value=self._mock_generate_result())

        await agent.async_execute("test prompt")

        call_kwargs = agent._client.async_generate.call_args
        assert call_kwargs.kwargs["response_schema"] is MySchema
        assert call_kwargs.kwargs["response_mime_type"] == "application/json"

    async def test_async_execute_omits_schema_params_when_none(self) -> None:
        agent = self._make_agent()
        from unittest.mock import AsyncMock

        agent._client.async_generate = AsyncMock(return_value=self._mock_generate_result())

        await agent.async_execute("test prompt")

        call_kwargs = agent._client.async_generate.call_args
        assert "response_schema" not in call_kwargs.kwargs
        assert "response_mime_type" not in call_kwargs.kwargs


# ── Sentinel model resolution via Orchestrator ────────────────


class TestOrchestratorSentinelResolution:
    """Verify that sentinel '' in agent configs resolves to Settings-driven model names.

    These tests exercise the full path:
      skill.get_agents_config() → orchestrator.create_agents_for_skill()
      → agent.model == settings.agents.specialist_model
    """

    def _make_orchestrator(self, specialist_model: str = "gemini-2.5-flash") -> object:
        from vaig.agents.orchestrator import Orchestrator
        from vaig.core.config import Settings

        client = MagicMock()
        settings = Settings()
        settings.agents.specialist_model = specialist_model
        settings.agents.orchestrator_model = "gemini-2.5-pro"
        return Orchestrator(client, settings)

    def _make_sentinel_skill(self, *, requires_tools: bool = False) -> object:
        """Return a minimal BaseSkill stub whose agent configs use sentinel ''."""
        from vaig.skills.base import BaseSkill

        class _SentinelSkill(BaseSkill):
            def get_agents_config(self) -> list[dict]:
                return [
                    {
                        "name": "analyst",
                        "role": "analyst",
                        "system_instruction": "Analyze.",
                        "model": "",
                        "requires_tools": requires_tools,
                    }
                ]

            async def execute(self, *a, **kw):  # type: ignore[override]
                pass

            def get_metadata(self):  # type: ignore[override]
                from vaig.skills.base import SkillMetadata

                return SkillMetadata(name="sentinel-skill", description="stub")

            def get_system_instruction(self) -> str:
                return "stub"

            def get_phase_prompt(self, phase, context, user_input) -> str:  # type: ignore[override]
                return "stub"

        return _SentinelSkill()

    def test_specialist_agent_sentinel_resolves_to_specialist_model(self) -> None:
        """SpecialistAgent created from sentinel '' uses settings.agents.specialist_model."""
        orch = self._make_orchestrator(specialist_model="gemini-2.5-flash")
        skill = self._make_sentinel_skill(requires_tools=False)
        agents = orch.create_agents_for_skill(skill)  # type: ignore[arg-type]
        assert len(agents) == 1
        assert agents[0].model == "gemini-2.5-flash"

    def test_specialist_agent_sentinel_reflects_custom_settings(self) -> None:
        """Resolution uses the actual Settings value, not a hardcoded string."""
        orch = self._make_orchestrator(specialist_model="gemini-2.0-flash")
        skill = self._make_sentinel_skill(requires_tools=False)
        agents = orch.create_agents_for_skill(skill)  # type: ignore[arg-type]
        assert agents[0].model == "gemini-2.0-flash"

    def test_explicit_model_in_config_is_not_overridden(self) -> None:
        """An explicit non-empty model in the skill config must be honoured."""
        from vaig.agents.orchestrator import Orchestrator
        from vaig.core.config import Settings
        from vaig.skills.base import BaseSkill

        class _ExplicitModelSkill(BaseSkill):
            def get_agents_config(self) -> list[dict]:
                return [
                    {
                        "name": "analyst",
                        "role": "analyst",
                        "system_instruction": "Analyze.",
                        "model": "gemini-2.5-pro",
                    }
                ]

            async def execute(self, *a, **kw):  # type: ignore[override]
                pass

            def get_metadata(self):  # type: ignore[override]
                from vaig.skills.base import SkillMetadata

                return SkillMetadata(name="explicit-skill", description="stub")

            def get_system_instruction(self) -> str:
                return "stub"

            def get_phase_prompt(self, phase, context, user_input) -> str:  # type: ignore[override]
                return "stub"

        client = MagicMock()
        settings = Settings()
        settings.agents.specialist_model = "gemini-2.5-flash"
        orch = Orchestrator(client, settings)
        agents = orch.create_agents_for_skill(_ExplicitModelSkill())  # type: ignore[arg-type]
        assert agents[0].model == "gemini-2.5-pro"
