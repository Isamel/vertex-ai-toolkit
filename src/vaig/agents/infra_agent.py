"""Infrastructure agent — autonomous SRE agent using Gemini function calling for GKE/GCP inspection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.agents.mixins import OnToolCall, ToolLoopMixin
from vaig.agents.utils import deduplicate_response
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS, GKEConfig, Settings
from vaig.core.exceptions import MaxIterationsError
from vaig.tools import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from vaig.core.cache import ToolResultCache
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.core.tool_call_store import ToolCallStore

logger = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────

INFRA_SYSTEM_PROMPT = """\
You are an expert SRE and infrastructure analyst with deep expertise in Google \
Kubernetes Engine (GKE) and Google Cloud Platform (GCP). Your goal is to help \
engineers investigate, diagnose, and understand their production infrastructure.

## Capabilities

You have access to **read-only** tools for inspecting live GKE clusters and \
GCP observability data:

### Kubernetes (GKE) Tools
- **kubectl_get**: List or get Kubernetes resources (pods, deployments, services, \
nodes, configmaps, secrets, hpa, ingress, etc.)
- **kubectl_describe**: Get detailed information about a specific resource including \
labels, annotations, spec, status, conditions, and events.
- **kubectl_logs**: Retrieve pod logs with support for tail, since, and container \
selection. Automatically fetches previous container logs on CrashLoopBackOff.
- **kubectl_top**: Show CPU and memory usage for pods or nodes (requires metrics-server).
- **get_events**: List Kubernetes events filtered by type (Warning/Normal) and \
involved object. Events reveal WHY pods fail, WHY nodes have issues, etc.
- **get_rollout_status**: Check whether a deployment rollout is progressing, \
complete, stalled, or failed. Shows replica counts, conditions, and strategy.
- **get_rollout_history**: Show the revision history of a deployment. Lists all \
revisions with images, creation time, and status (active/scaled-down). Optionally \
show detailed pod template for a specific revision. Before recommending a rollback, \
always check rollout history first to understand what changed in each revision.
- **get_node_conditions**: Show node health conditions, resource pressure, taints, \
and capacity. Lists all nodes with CPU/memory capacity, or shows detailed node \
conditions (MemoryPressure, DiskPressure, PIDPressure) that kubectl_get hides.
- **get_container_status**: Show detailed container-level status for all containers \
in a pod (init, regular, ephemeral). Reveals per-container state, restart count, \
last termination info, resource requests/limits, and env var sources.
- **exec_command**: Execute read-only diagnostic commands inside a running container \
(cat, ls, ps, df, curl, etc.). SECURITY: Disabled by default — requires \
gke.exec_enabled=true. Commands are validated against denylist (no shell injection, \
no destructive ops) and allowlist (read-only diagnostics only). Use for inspecting \
config files, checking processes, network debugging, and runtime diagnostics.
- **check_rbac**: Check if a service account or current user has permission to \
perform a specific action on a Kubernetes resource. Use BEFORE operations that \
might fail with permission errors, or when troubleshooting RBAC issues.

### GCP Observability Tools
- **gcloud_logging_query**: Query Cloud Logging with filter expressions. Supports \
all Cloud Logging filter syntax.
- **gcloud_monitoring_query**: Query Cloud Monitoring time series for metrics like \
CPU utilization, memory usage, restart counts, and custom metrics.

## Workflow — Follow This Order

### Phase 1: Understand the Situation
Before diving into details, you MUST:
1. Ask clarifying questions if the request is ambiguous.
2. Start with broad resource listing (kubectl_get) to understand the landscape.
3. Identify which namespaces, deployments, or pods are relevant.

### Phase 2: Investigate Systematically
When diagnosing issues:
1. **Check resource status** — List pods/deployments to identify unhealthy resources.
2. **Check events** — Use get_events to see warnings and errors in the namespace.
3. **Examine details** — Describe resources to see events, conditions, and spec.
4. **Review logs** — Check pod logs for error messages and stack traces.
5. **Check rollout status** — Use get_rollout_status for deployment rollout issues.
6. **Check rollout history** — Use get_rollout_history to see revision history \
BEFORE recommending a rollback. Shows what changed in each revision.
7. **Check node health** — Use get_node_conditions for node pressure, taints, capacity.
8. **Inspect containers** — Use get_container_status for multi-container pod debugging.
9. **Check RBAC** — Use check_rbac to verify permissions BEFORE operations that might fail.
10. **Execute diagnostics** — Use exec_command to inspect config files, processes, or \
network inside containers (requires exec_enabled=true).
11. **Check metrics** — Use kubectl_top and Cloud Monitoring for resource pressure.
12. **Correlate** — Cross-reference Cloud Logging for broader context.

### Phase 3: Report Findings
After investigation:
1. Summarise what you found with specific evidence (pod names, error messages, metrics).
2. Identify the root cause or most likely candidates.
3. Suggest actionable next steps (but remember: your tools are READ-ONLY).
4. Highlight any concerning trends (high restart counts, resource pressure, etc.).

## Important Rules
- You have **READ-ONLY** access. You cannot modify, delete, or create resources.
- Always specify namespaces explicitly — don't assume "default".
- When checking pod health, look at: status, restart count, events, and logs.
- For performance issues, correlate kubectl_top with Cloud Monitoring metrics.
- Use label selectors to narrow results when clusters have many resources.
- Truncate large log outputs and focus on the most recent/relevant entries.

## Error Handling
If a tool returns an error:
- Authentication errors: suggest checking kubeconfig or GKE credentials.
- Permission errors: use check_rbac to verify which RBAC roles might be needed.
- Not found errors: verify the namespace and resource name.
- Metrics unavailable: suggest installing metrics-server.
"""


# ── InfraAgent class ─────────────────────────────────────────


class InfraAgent(BaseAgent, ToolLoopMixin):
    """Autonomous infrastructure agent that uses Gemini function calling to inspect GKE/GCP.

    The agent owns the tool-use loop:
    1. Sends the user prompt to Gemini with tool declarations
    2. If Gemini returns function calls, executes them and sends results back
    3. Repeats until Gemini returns a text response or max iterations is hit

    All tools are READ-ONLY — no cluster modifications are possible.

    Architecture decisions:
    - Follows CodingAgent pattern exactly (same loop structure, same error handling)
    - No confirmation callback needed — all tools are read-only
    - Max 25 tool iterations per turn (safety cap, configurable via GKEConfig)
    - Tool-use loop delegated to ToolLoopMixin for reuse across agents
    """

    def __init__(
        self,
        client: GeminiClientProtocol,
        gke_config: GKEConfig,
        *,
        settings: Settings | None = None,
        max_tool_iterations: int = 25,
        model_id: str | None = None,
    ) -> None:
        """Initialize the infrastructure agent.

        Args:
            client: The Gemini client (protocol) for API calls.
            gke_config: GKE configuration (cluster connection, defaults, etc.).
            settings: Full application settings (used for plugin tool loading).
            max_tool_iterations: Maximum tool-use loop iterations per turn.
            model_id: Override the default model for this agent.
        """
        config = AgentConfig(
            name="infra-agent",
            role=AgentRole.SRE,
            system_instruction=INFRA_SYSTEM_PROMPT,
            model=model_id or client.current_model,
            temperature=0.2,  # Low temperature for precise infrastructure analysis
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        super().__init__(config, client)

        self._gke_config = gke_config
        self._settings = settings
        self._max_iterations = max_tool_iterations

        # Build tool registry with GKE and GCP tools
        self._registry = ToolRegistry()
        self._register_tools()

        logger.info(
            "InfraAgent initialized — cluster=%s, project=%s, max_iterations=%d, tools=%d",
            gke_config.cluster_name or "(default kubeconfig)",
            gke_config.project_id or "(auto-detect)",
            self._max_iterations,
            len(self._registry.list_tools()),
        )

    def _register_tools(self) -> None:
        """Register GKE and GCP tools, handling missing optional dependencies gracefully."""
        # Resolve GKE-specific credentials (SA impersonation or ADC)
        gke_credentials = None
        if self._settings is not None:
            from vaig.core.auth import get_gke_credentials

            gke_credentials = get_gke_credentials(self._settings)

        # GKE tools — requires 'kubernetes' package
        try:
            from vaig.tools.gke_tools import create_gke_tools  # noqa: WPS433

            for tool in create_gke_tools(self._gke_config):
                self._registry.register(tool)
            logger.debug("Registered GKE tools (kubernetes SDK available)")
        except ImportError:
            logger.warning(
                "kubernetes package not installed — GKE tools unavailable. "
                "Install with: pip install vertex-ai-toolkit[live]"
            )

        # GCP observability tools — requires google-cloud-logging / google-cloud-monitoring
        try:
            from vaig.tools.gcloud_tools import create_gcloud_tools  # noqa: WPS433

            for tool in create_gcloud_tools(
                project=self._gke_config.project_id,
                log_limit=self._gke_config.log_limit,
                metrics_interval_minutes=self._gke_config.metrics_interval_minutes,
                credentials=gke_credentials,
            ):
                self._registry.register(tool)
            logger.debug("Registered GCP observability tools")
        except ImportError:
            logger.warning(
                "google-cloud-logging/monitoring packages not installed — "
                "GCP observability tools unavailable. "
                "Install with: pip install google-cloud-logging google-cloud-monitoring"
            )

        # Plugin tools — MCP auto-registration and Python module plugins
        if self._settings is not None:
            try:
                from vaig.tools.plugin_loader import load_all_plugin_tools  # noqa: WPS433

                for tool in load_all_plugin_tools(self._settings):
                    self._registry.register(tool)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load plugin tools for InfraAgent. Skipping.",
                    exc_info=True,
                )

    @property
    def gke_config(self) -> GKEConfig:
        """The GKE configuration for this agent."""
        return self._gke_config

    @property
    def registry(self) -> ToolRegistry:
        """The tool registry for this agent."""
        return self._registry

    # ── Tool-use loop ────────────────────────────────────────

    def execute(
        self,
        prompt: str,
        *,
        context: str = "",
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        tool_result_cache: ToolResultCache | None = None,
    ) -> AgentResult:
        """Execute an infrastructure investigation using the tool-use loop.

        Sends the prompt to Gemini with tool declarations. If the model
        returns function calls, executes them and feeds results back.
        Repeats until the model returns a text response or the iteration
        limit is reached.

        Args:
            prompt: The infrastructure question or investigation task.
            context: Optional additional context (cluster info, incident details, etc.).
            on_tool_call: Optional callback invoked after each tool
                execution with ``(tool_name, tool_args, duration_secs,
                success)``.
            tool_call_store: Optional persistent store for tool call
                records (used for telemetry / debugging).
            tool_result_cache: Optional TTL cache for deduplicating
                identical tool calls within the same session.

        Returns:
            AgentResult with the final text response and metadata.

        Raises:
            MaxIterationsError: If the tool loop exceeds max_iterations.
        """
        full_prompt = self._build_prompt(prompt, context)
        self._add_to_conversation("user", full_prompt)

        history = self._build_chat_history()

        logger.debug(
            "InfraAgent.execute() — starting tool loop (max=%d)",
            self._max_iterations,
        )

        try:
            loop_result = self._run_tool_loop(
                client=self._client,
                prompt=full_prompt,
                tool_registry=self._registry,
                system_instruction=self._config.system_instruction,
                history=history,
                max_iterations=self._max_iterations,
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                frequency_penalty=0.15,
                on_tool_call=on_tool_call,
                agent_name=self.name,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
            )
        except MaxIterationsError:
            raise
        except Exception as exc:
            logger.exception("InfraAgent API call failed")
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
                success=False,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"error": str(exc)},
            )

        clean_text = self._deduplicate_response(loop_result.text)
        logger.info(
            "InfraAgent completed — %d iterations, %d tool calls, %s total tokens",
            loop_result.iterations,
            len(loop_result.tools_executed),
            loop_result.usage.get("total_tokens", "?"),
        )
        self._add_to_conversation("agent", clean_text)
        return AgentResult(
            agent_name=self.name,
            content=clean_text,
            success=True,
            usage=loop_result.usage,
            metadata={
                "model": loop_result.model,
                "finish_reason": loop_result.finish_reason,
                "iterations": loop_result.iterations,
                "tools_executed": loop_result.tools_executed,
            },
        )

    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Streaming is not supported for the infrastructure agent.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to non-streaming execute and yields the result.
        """
        logger.debug("InfraAgent.execute_stream() — falling back to non-streaming")
        result = self.execute(prompt, context=context)
        yield result.content

    # ── Async methods ────────────────────────────────────────

    async def async_execute(
        self,
        prompt: str,
        *,
        context: str = "",
        on_tool_call: OnToolCall | None = None,
        tool_call_store: ToolCallStore | None = None,
        tool_result_cache: ToolResultCache | None = None,
    ) -> AgentResult:
        """Execute an infrastructure investigation using the async tool-use loop.

        Async version of :meth:`execute`.  Delegates to
        :meth:`ToolLoopMixin._async_run_tool_loop` for non-blocking LLM
        calls and tool execution.

        Args:
            prompt: The infrastructure question or investigation task.
            context: Optional additional context (cluster info, incident details, etc.).
            on_tool_call: Optional callback invoked after each tool
                execution with ``(tool_name, tool_args, duration_secs,
                success)``.
            tool_call_store: Optional persistent store for tool call
                records (used for telemetry / debugging).
            tool_result_cache: Optional TTL cache for deduplicating
                identical tool calls within the same session.

        Returns:
            AgentResult with the final text response and metadata.

        Raises:
            MaxIterationsError: If the tool loop exceeds max_iterations.
        """
        full_prompt = self._build_prompt(prompt, context)
        self._add_to_conversation("user", full_prompt)

        history = self._build_chat_history()

        logger.debug(
            "InfraAgent.async_execute() — starting async tool loop (max=%d)",
            self._max_iterations,
        )

        try:
            loop_result = await self._async_run_tool_loop(
                client=self._client,
                prompt=full_prompt,
                tool_registry=self._registry,
                system_instruction=self._config.system_instruction,
                history=history,
                max_iterations=self._max_iterations,
                model=self._config.model,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                frequency_penalty=0.15,
                on_tool_call=on_tool_call,
                agent_name=self.name,
                tool_call_store=tool_call_store,
                tool_result_cache=tool_result_cache,
            )
        except MaxIterationsError:
            raise
        except Exception as exc:
            logger.exception("InfraAgent async API call failed")
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
                success=False,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"error": str(exc)},
            )

        clean_text = self._deduplicate_response(loop_result.text)
        logger.info(
            "InfraAgent async completed — %d iterations, %d tool calls, %s total tokens",
            loop_result.iterations,
            len(loop_result.tools_executed),
            loop_result.usage.get("total_tokens", "?"),
        )
        self._add_to_conversation("agent", clean_text)
        return AgentResult(
            agent_name=self.name,
            content=clean_text,
            success=True,
            usage=loop_result.usage,
            metadata={
                "model": loop_result.model,
                "finish_reason": loop_result.finish_reason,
                "iterations": loop_result.iterations,
                "tools_executed": loop_result.tools_executed,
            },
        )

    async def async_execute_stream(
        self,
        prompt: str,
        *,
        context: str = "",
    ) -> AsyncIterator[str]:
        """Async streaming — falls back to async_execute.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to :meth:`async_execute` and yields the result.
        """
        logger.debug(
            "InfraAgent.async_execute_stream() — falling back to async non-streaming",
        )
        result = await self.async_execute(prompt, context=context)
        yield result.content

    # ── Tool execution with error handling ───────────────────

    # The core tool execution logic lives in ToolLoopMixin._execute_single_tool().
    # InfraAgent uses it as-is — all infrastructure tools are read-only,
    # so no confirmation callback or special handling is needed.

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call with error handling.

        Delegates to :meth:`ToolLoopMixin._execute_single_tool` using
        this agent's tool registry.  Kept as a convenience method so
        callers don't need to pass the registry explicitly.
        """
        return self._execute_single_tool(self._registry, tool_name, tool_args)

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _deduplicate_response(text: str, *, threshold: int = 3) -> str:
        """Remove repeated lines — delegates to shared ``deduplicate_response``."""
        return deduplicate_response(text, threshold=threshold)
