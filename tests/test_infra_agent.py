"""Tests for InfraAgent — initialization, tool registration, execute, and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.agents.base import AgentResult, AgentRole
from vaig.core.config import GKEConfig
from vaig.tools.base import ToolDef, ToolParam, ToolRegistry, ToolResult

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


def _make_mock_client() -> MagicMock:
    """Create a mock GeminiClient with sensible defaults."""
    client = MagicMock()
    client.current_model = "gemini-2.5-pro"
    return client


# ── InfraAgent initialization ────────────────────────────────


class TestInfraAgentInit:
    """Tests for InfraAgent construction and configuration."""

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_basic_initialization(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg)

        assert agent.name == "infra-agent"
        assert agent.gke_config is cfg
        assert agent._max_iterations == 25

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_custom_max_iterations(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg, max_tool_iterations=10)
        assert agent._max_iterations == 10

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_custom_model_id(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg, model_id="gemini-2.5-flash")
        assert agent._config.model == "gemini-2.5-flash"

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_config_has_low_temperature(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg)
        assert agent._config.temperature == 0.2  # Low for precision

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_role_is_sre(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg)
        assert agent._config.role == AgentRole.SRE

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_registry_is_tool_registry(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        agent = InfraAgent(client, cfg)
        assert isinstance(agent.registry, ToolRegistry)


# ── _register_tools ──────────────────────────────────────────


class TestRegisterTools:
    """Tests for tool registration with optional dependencies."""

    def test_register_gke_tools_import_error(self) -> None:
        """When kubernetes is not installed, GKE tools should be skipped gracefully."""
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()

        with patch("vaig.agents.infra_agent.InfraAgent._register_tools") as mock_reg:
            agent = InfraAgent(client, cfg)
            mock_reg.assert_called_once()

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_gke_config_exposed(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config(cluster_name="prod-cluster")

        agent = InfraAgent(client, cfg)
        assert agent.gke_config.cluster_name == "prod-cluster"


# ── _deduplicate_response ────────────────────────────────────


class TestDeduplicateResponse:
    """Tests for the static _deduplicate_response method."""

    def test_empty_text(self) -> None:
        from vaig.agents.infra_agent import InfraAgent

        assert InfraAgent._deduplicate_response("") == ""

    def test_no_repetition(self) -> None:
        from vaig.agents.infra_agent import InfraAgent

        text = "Line one\nLine two\nLine three"
        result = InfraAgent._deduplicate_response(text)
        assert result == text

    def test_removes_excessive_repetition(self) -> None:
        from vaig.agents.infra_agent import InfraAgent

        # Same long line repeated 10 times — should be truncated
        line = "This is a repeated analysis finding."
        text = "\n".join([line] * 10)
        result = InfraAgent._deduplicate_response(text, threshold=3)
        assert "[truncated" in result
        # Should only keep 3 occurrences
        assert result.count(line) == 3

    def test_short_lines_ignored(self) -> None:
        from vaig.agents.infra_agent import InfraAgent

        # Short lines (<=10 chars) should not trigger dedup
        text = "\n".join(["---"] * 20)
        result = InfraAgent._deduplicate_response(text)
        assert "[truncated" not in result

    def test_threshold_respected(self) -> None:
        from vaig.agents.infra_agent import InfraAgent

        line = "This is a repeated line for testing."
        text = "\n".join([line] * 5)
        result = InfraAgent._deduplicate_response(text, threshold=5)
        assert "[truncated" not in result  # 5 <= threshold of 5


# ── _build_prompt ────────────────────────────────────────────


class TestBuildPrompt:
    """Tests for _build_prompt helper."""

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_without_context(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        result = agent._build_prompt("List pods", context="")
        assert result == "List pods"

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_with_context(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        result = agent._build_prompt("List pods", context="Production cluster")
        assert "## Context" in result
        assert "Production cluster" in result
        assert "## Task" in result
        assert "List pods" in result


# ── _execute_tool ────────────────────────────────────────────


class TestExecuteTool:
    """Tests for _execute_tool with error handling."""

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_unknown_tool(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        result = agent._execute_tool("nonexistent_tool", {})
        assert result.error is True
        assert "Unknown tool" in result.output

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_tool_execution_success(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        # Register a mock tool
        tool = ToolDef(
            name="test_tool",
            description="A test tool",
            execute=lambda: ToolResult(output="success"),
        )
        agent._registry.register(tool)

        result = agent._execute_tool("test_tool", {})
        assert result.error is False
        assert result.output == "success"

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_tool_type_error(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        def bad_tool(required_arg: str) -> ToolResult:
            return ToolResult(output=required_arg)

        tool = ToolDef(
            name="bad_tool",
            description="A bad tool",
            parameters=[ToolParam(name="required_arg", type="string", description="required")],
            execute=bad_tool,
        )
        agent._registry.register(tool)

        # Call without required arg — should be caught as TypeError
        result = agent._execute_tool("bad_tool", {})
        assert result.error is True
        assert "Invalid arguments" in result.output

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_tool_unexpected_error(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        def crashing_tool() -> ToolResult:
            msg = "boom"
            raise RuntimeError(msg)

        tool = ToolDef(
            name="crash_tool",
            description="A crashing tool",
            execute=crashing_tool,
        )
        agent._registry.register(tool)

        result = agent._execute_tool("crash_tool", {})
        assert result.error is True
        assert "Tool execution error" in result.output


# ── execute (tool-use loop) ──────────────────────────────────


class TestExecuteLoop:
    """Tests for the execute method's tool-use loop."""

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_execute_text_response(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        # Mock generate_with_tools to return text (no function calls)
        mock_result = MagicMock()
        mock_result.function_calls = []
        mock_result.text = "All pods are healthy."
        mock_result.model = "gemini-2.5-pro"
        mock_result.finish_reason = "STOP"
        mock_result.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        client.generate_with_tools.return_value = mock_result

        result = agent.execute("Check pod status")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "All pods are healthy" in result.content
        assert result.agent_name == "infra-agent"

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_execute_api_error(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        client.generate_with_tools.side_effect = Exception("API connection failed")

        result = agent.execute("Check pods")

        assert result.success is False
        assert "Error during API call" in result.content

    @patch("vaig.agents.infra_agent.InfraAgent._register_tools")
    def test_execute_stream_fallback(self, mock_register: MagicMock) -> None:
        from vaig.agents.infra_agent import InfraAgent

        client = _make_mock_client()
        cfg = _make_gke_config()
        agent = InfraAgent(client, cfg)

        mock_result = MagicMock()
        mock_result.function_calls = []
        mock_result.text = "Streaming result"
        mock_result.model = "gemini-2.5-pro"
        mock_result.finish_reason = "STOP"
        mock_result.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        client.generate_with_tools.return_value = mock_result

        chunks = list(agent.execute_stream("Check pods"))
        assert len(chunks) == 1
        assert "Streaming result" in chunks[0]


# ── INFRA_SYSTEM_PROMPT ──────────────────────────────────────


class TestInfraSystemPrompt:
    """Tests for the system prompt constant."""

    def test_prompt_exists(self) -> None:
        from vaig.agents.infra_agent import INFRA_SYSTEM_PROMPT

        assert isinstance(INFRA_SYSTEM_PROMPT, str)
        assert len(INFRA_SYSTEM_PROMPT) > 100

    def test_prompt_mentions_read_only(self) -> None:
        from vaig.agents.infra_agent import INFRA_SYSTEM_PROMPT

        assert "read-only" in INFRA_SYSTEM_PROMPT.lower() or "READ-ONLY" in INFRA_SYSTEM_PROMPT

    def test_prompt_mentions_tools(self) -> None:
        from vaig.agents.infra_agent import INFRA_SYSTEM_PROMPT

        assert "kubectl_get" in INFRA_SYSTEM_PROMPT
        assert "kubectl_describe" in INFRA_SYSTEM_PROMPT
        assert "kubectl_logs" in INFRA_SYSTEM_PROMPT
        assert "kubectl_top" in INFRA_SYSTEM_PROMPT

    def test_prompt_mentions_gcloud_tools(self) -> None:
        from vaig.agents.infra_agent import INFRA_SYSTEM_PROMPT

        assert "gcloud_logging_query" in INFRA_SYSTEM_PROMPT
        assert "gcloud_monitoring_query" in INFRA_SYSTEM_PROMPT
