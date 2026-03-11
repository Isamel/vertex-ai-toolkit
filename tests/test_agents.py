"""Tests for agent base classes and data structures."""

from __future__ import annotations

from vaig.agents.base import AgentConfig, AgentMessage, AgentResult, AgentRole


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
