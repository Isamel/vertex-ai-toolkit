"""Infrastructure agent — autonomous SRE agent using Gemini function calling for GKE/GCP inspection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from google.genai import types

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.core.client import GeminiClient, ToolCallResult
from vaig.core.config import GKEConfig
from vaig.core.exceptions import MaxIterationsError
from vaig.tools import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterator

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
2. **Examine details** — Describe resources to see events, conditions, and spec.
3. **Review logs** — Check pod logs for error messages and stack traces.
4. **Check metrics** — Use kubectl_top and Cloud Monitoring for resource pressure.
5. **Correlate** — Cross-reference Cloud Logging for broader context.

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
- Permission errors: note which RBAC roles might be needed.
- Not found errors: verify the namespace and resource name.
- Metrics unavailable: suggest installing metrics-server.
"""


# ── InfraAgent class ─────────────────────────────────────────


class InfraAgent(BaseAgent):
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
    """

    def __init__(
        self,
        client: GeminiClient,
        gke_config: GKEConfig,
        *,
        max_tool_iterations: int = 25,
        model_id: str | None = None,
    ) -> None:
        """Initialize the infrastructure agent.

        Args:
            client: The GeminiClient for API calls.
            gke_config: GKE configuration (cluster connection, defaults, etc.).
            max_tool_iterations: Maximum tool-use loop iterations per turn.
            model_id: Override the default model for this agent.
        """
        config = AgentConfig(
            name="infra-agent",
            role=AgentRole.SRE,
            system_instruction=INFRA_SYSTEM_PROMPT,
            model=model_id or client.current_model,
            temperature=0.2,  # Low temperature for precise infrastructure analysis
            max_output_tokens=65536,
        )
        super().__init__(config, client)

        self._gke_config = gke_config
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
            ):
                self._registry.register(tool)
            logger.debug("Registered GCP observability tools")
        except ImportError:
            logger.warning(
                "google-cloud-logging/monitoring packages not installed — "
                "GCP observability tools unavailable. "
                "Install with: pip install google-cloud-logging google-cloud-monitoring"
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

    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute an infrastructure investigation using the tool-use loop.

        Sends the prompt to Gemini with tool declarations. If the model
        returns function calls, executes them and feeds results back.
        Repeats until the model returns a text response or the iteration
        limit is reached.

        Args:
            prompt: The infrastructure question or investigation task.
            context: Optional additional context (cluster info, incident details, etc.).

        Returns:
            AgentResult with the final text response and metadata.

        Raises:
            MaxIterationsError: If the tool loop exceeds max_iterations.
        """
        full_prompt = self._build_prompt(prompt, context)
        self._add_to_conversation("user", full_prompt)

        declarations = self._registry.to_function_declarations()
        history = self._build_chat_history()

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        tools_executed: list[dict[str, Any]] = []
        iteration = 0

        logger.debug(
            "InfraAgent.execute() — starting tool loop (max=%d)",
            self._max_iterations,
        )

        while iteration < self._max_iterations:
            iteration += 1
            logger.debug("Tool loop iteration %d/%d", iteration, self._max_iterations)

            # Call Gemini with tool declarations
            try:
                result: ToolCallResult = self._client.generate_with_tools(
                    full_prompt if iteration == 1 else [],
                    tool_declarations=declarations,
                    system_instruction=self._config.system_instruction,
                    history=history,
                    model_id=self._config.model,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_output_tokens,
                    frequency_penalty=0.15,
                )
            except Exception as exc:
                logger.exception("InfraAgent API call failed on iteration %d", iteration)
                return AgentResult(
                    agent_name=self.name,
                    content=f"Error during API call: {exc}",
                    success=False,
                    usage=total_usage,
                    metadata={"error": str(exc), "iterations": iteration},
                )

            # Accumulate token usage
            for key in total_usage:
                total_usage[key] += result.usage.get(key, 0)

            # Case 1: Model returned text (no function calls) — we're done
            if not result.function_calls:
                clean_text = self._deduplicate_response(result.text)
                logger.info(
                    "InfraAgent completed — %d iterations, %d tool calls, %s total tokens",
                    iteration,
                    len(tools_executed),
                    total_usage.get("total_tokens", "?"),
                )
                self._add_to_conversation("agent", clean_text)
                return AgentResult(
                    agent_name=self.name,
                    content=clean_text,
                    success=True,
                    usage=total_usage,
                    metadata={
                        "model": result.model,
                        "finish_reason": result.finish_reason,
                        "iterations": iteration,
                        "tools_executed": tools_executed,
                    },
                )

            # Case 2: Model returned function calls — execute them
            # First, add the model's response (with function calls) to history
            # so the next turn sees what the model requested
            fc_parts: list[types.Part] = []
            for fc in result.function_calls:
                fc_parts.append(
                    types.Part.from_function_call(
                        name=fc["name"],
                        args=fc["args"],
                    )
                )
            history.append(
                types.Content(role="model", parts=fc_parts)
            )

            # Execute each function call
            function_responses: list[dict[str, Any]] = []

            for fc in result.function_calls:
                tool_name = fc["name"]
                tool_args = fc["args"]

                tool_result = self._execute_tool(tool_name, tool_args)

                tools_executed.append({
                    "name": tool_name,
                    "args": tool_args,
                    "output": tool_result.output[:200],  # Truncate for metadata
                    "error": tool_result.error,
                })

                function_responses.append({
                    "name": tool_name,
                    "response": {
                        "output": tool_result.output,
                        "error": tool_result.error,
                    },
                })

            # Add function responses to history for next turn
            response_parts = GeminiClient.build_function_response_parts(function_responses)
            history.append(
                types.Content(role="user", parts=response_parts)
            )

            # Continue the loop — send empty prompt (history carries context).
            # generate_with_tools handles this by popping the last history
            # entry (the function responses) and using it as the prompt
            # for send_message(), since the SDK rejects empty prompts.

        # If we get here, max iterations exceeded
        msg = (
            f"Tool-use loop exceeded maximum iterations ({self._max_iterations}). "
            f"Executed {len(tools_executed)} tool calls."
        )
        logger.warning(msg)
        raise MaxIterationsError(msg, iterations=self._max_iterations)

    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Streaming is not supported for the infrastructure agent.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to non-streaming execute and yields the result.
        """
        logger.debug("InfraAgent.execute_stream() — falling back to non-streaming")
        result = self.execute(prompt, context=context)
        yield result.content

    # ── Tool execution with error handling ───────────────────

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call with error handling.

        All infrastructure tools are read-only, so no confirmation is needed.
        Catches ALL exceptions and returns them as ToolResult (never raises)
        so the model can see errors and self-correct.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments to pass to the tool.

        Returns:
            ToolResult with output or error message.
        """
        tool = self._registry.get(tool_name)

        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return ToolResult(
                output=f"Unknown tool: {tool_name}. Available tools: "
                f"{', '.join(t.name for t in self._registry.list_tools())}",
                error=True,
            )

        # Execute the tool — catch ALL exceptions to let model self-correct
        try:
            logger.debug("Executing tool: %s(%s)", tool_name, tool_args)
            result = tool.execute(**tool_args)
            logger.debug(
                "Tool %s result: error=%s, output_len=%d",
                tool_name,
                result.error,
                len(result.output),
            )
            return result
        except TypeError as exc:
            # Most common: wrong argument names from model
            logger.warning("Tool %s type error: %s", tool_name, exc)
            return ToolResult(
                output=f"Invalid arguments for {tool_name}: {exc}. "
                f"Expected parameters: {', '.join(p.name for p in tool.parameters)}",
                error=True,
            )
        except Exception as exc:
            # Catch-all: let the model see the error and try again
            logger.warning("Tool %s unexpected error: %s", tool_name, exc)
            return ToolResult(
                output=f"Tool execution error ({tool_name}): {exc}",
                error=True,
            )

    # ── Internal helpers ─────────────────────────────────────

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Build the full prompt with optional context."""
        if context:
            return f"## Context\n\n{context}\n\n## Task\n\n{prompt}"
        return prompt

    @staticmethod
    def _deduplicate_response(text: str, *, threshold: int = 3) -> str:
        """Remove repeated sentences/lines from model output.

        Gemini can sometimes produce pathological repetition — the same
        sentence hundreds of times in a single response, especially at low
        temperature with high ``max_output_tokens``.  This method acts as
        a safety net: it scans the text line by line and truncates once a
        line has been seen more than *threshold* consecutive times.

        The algorithm is intentionally conservative:
        - It only counts **consecutive** repetitions (not scattered ones).
        - Short lines (<=10 chars) are ignored to avoid false positives on
          blank lines, bullets, braces, etc.
        - A ``[truncated — repeated text removed]`` marker is appended when
          truncation occurs so the user knows something was cut.

        Args:
            text: The raw model response text.
            threshold: How many consecutive identical lines to allow before
                       truncating.  Default is 3 (keeps first 3 occurrences).

        Returns:
            The cleaned text, possibly truncated.
        """
        if not text:
            return text

        lines = text.split("\n")
        result: list[str] = []
        prev_line: str | None = None
        repeat_count = 0
        truncated = False

        for line in lines:
            stripped = line.strip()

            # Skip short-line tracking — too many false positives
            if len(stripped) <= 10:
                result.append(line)
                prev_line = None
                repeat_count = 0
                continue

            if stripped == prev_line:
                repeat_count += 1
                if repeat_count > threshold:
                    truncated = True
                    continue  # Drop this repeated line
                result.append(line)
            else:
                prev_line = stripped
                repeat_count = 1
                result.append(line)

        cleaned = "\n".join(result)
        if truncated:
            cleaned = cleaned.rstrip() + "\n\n[truncated — repeated text removed]"
            logger.warning(
                "Deduplicated model response — removed repeated lines "
                "(threshold=%d)",
                threshold,
            )
        return cleaned

    def _build_chat_history(self) -> list[Any]:
        """Convert agent conversation history to Gemini Content list.

        Unlike SpecialistAgent which uses ChatMessage, InfraAgent works
        directly with Content objects because the history may contain
        function call/response Parts (not just text).
        """
        contents: list[types.Content] = []
        for msg in self._conversation:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
            )
        return contents
