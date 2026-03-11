"""Tests for agent base classes and data structures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
        assert cfg.model == "gemini-2.5-pro"
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 8192

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
        assert agent.model == "gemini-2.5-pro"

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
