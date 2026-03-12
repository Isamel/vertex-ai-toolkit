"""Coding agent — autonomous file-editing agent using Gemini function calling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.core.client import ChatMessage, GeminiClient, ToolCallResult
from vaig.core.config import CodingConfig
from vaig.core.exceptions import MaxIterationsError
from vaig.tools import ToolRegistry, ToolResult, create_file_tools, create_shell_tools

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)


# ── Task 4.3 — System prompt ────────────────────────────────

CODING_SYSTEM_PROMPT = """\
You are an expert coding assistant with access to filesystem tools.

## Rules — STRICT
1. Write COMPLETE, production-ready code. Never use TODO, FIXME, placeholder, \
pass, or `...` as a substitute for real implementation.
2. When editing files, always use the edit_file tool with exact string matching. \
Provide enough surrounding context in old_string to uniquely identify the target.
3. Before writing or editing a file, read it first to understand its current state.
4. When creating new files, include all necessary imports and complete implementations.
5. Use search_files to find relevant code before making changes.
6. Explain what you're doing and why before each tool call.
7. After making changes, verify them by reading the modified file.

## Tool Usage
- read_file: Read file contents. Always read before editing.
- write_file: Create a new file or overwrite entirely. Use for new files.
- edit_file: Apply exact string replacement. Use for modifying existing files.
- list_files: List directory contents to understand project structure.
- search_files: Search file contents with regex. Use to find code patterns.
- run_command: Execute shell commands (e.g., tests, linting).

## Error Handling
If a tool returns an error, analyze the error message and try a different approach. \
Do NOT repeat the same failing call. Common fixes:
- edit_file "not found": re-read the file, copy the EXACT string including whitespace.
- edit_file "multiple matches": include more surrounding lines for uniqueness.
- Path errors: use paths relative to the workspace root.
"""


# ── Task 4.4 — Destructive tool names (need confirmation) ───

_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "run_command",
})


def _default_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Default confirmation callback — always approves (used in non-interactive mode)."""
    return True


# ── Task 4.1 — CodingAgent class ────────────────────────────


class CodingAgent(BaseAgent):
    """Autonomous coding agent that uses Gemini function calling to edit files.

    The agent owns the tool-use loop:
    1. Sends the user prompt to Gemini with tool declarations
    2. If Gemini returns function calls, executes them and sends results back
    3. Repeats until Gemini returns a text response or max iterations is hit

    Architecture decisions:
    - AD-02: CodingAgent owns the tool-use loop (not client, not orchestrator)
    - AD-03: Confirmation is an injectable callback (confirm_fn) — UI-agnostic
    - AD-07: Max 25 tool iterations per turn (safety cap, configurable)
    """

    def __init__(
        self,
        client: GeminiClient,
        coding_config: CodingConfig,
        *,
        confirm_fn: Callable[[str, dict[str, Any]], bool] | None = None,
        model_id: str | None = None,
    ) -> None:
        """Initialize the coding agent.

        Args:
            client: The GeminiClient for API calls.
            coding_config: Coding-specific configuration (workspace, limits, etc.).
            confirm_fn: Optional callback for confirming destructive operations.
                        Signature: (tool_name, args) -> bool. If None or if
                        confirm_actions is False, all actions are auto-approved.
            model_id: Override the default model for this agent.
        """
        config = AgentConfig(
            name="coding-agent",
            role=AgentRole.CODER,
            system_instruction=CODING_SYSTEM_PROMPT,
            model=model_id or client.current_model,
            temperature=0.2,  # Low temperature for precise code generation
            max_output_tokens=8192,
        )
        super().__init__(config, client)

        self._coding_config = coding_config
        self._workspace = Path(coding_config.workspace_root).resolve()
        self._max_iterations = coding_config.max_tool_iterations

        # Confirmation: use provided callback, or auto-approve if confirm disabled
        if coding_config.confirm_actions and confirm_fn is not None:
            self._confirm_fn = confirm_fn
        else:
            self._confirm_fn = _default_confirm

        # Build tool registry with all tools bound to workspace
        self._registry = ToolRegistry()
        for tool in create_file_tools(self._workspace):
            self._registry.register(tool)
        for tool in create_shell_tools(
            self._workspace,
            allowed_commands=coding_config.allowed_commands or None,
        ):
            self._registry.register(tool)

        logger.info(
            "CodingAgent initialized — workspace=%s, max_iterations=%d, "
            "confirm=%s, tools=%d",
            self._workspace,
            self._max_iterations,
            coding_config.confirm_actions,
            len(self._registry.list_tools()),
        )

    @property
    def workspace(self) -> Path:
        """The resolved workspace root path."""
        return self._workspace

    @property
    def registry(self) -> ToolRegistry:
        """The tool registry for this agent."""
        return self._registry

    # ── Task 4.2 — Tool-use loop ─────────────────────────────

    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a coding task using the tool-use loop.

        Sends the prompt to Gemini with tool declarations. If the model
        returns function calls, executes them and feeds results back.
        Repeats until the model returns a text response or the iteration
        limit is reached.

        Args:
            prompt: The coding task or question.
            context: Optional additional context (file contents, etc.).

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
            "CodingAgent.execute() — starting tool loop (max=%d)",
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
                )
            except Exception as exc:
                logger.exception("CodingAgent API call failed on iteration %d", iteration)
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
                logger.info(
                    "CodingAgent completed — %d iterations, %d tool calls, %s total tokens",
                    iteration,
                    len(tools_executed),
                    total_usage.get("total_tokens", "?"),
                )
                self._add_to_conversation("agent", result.text)
                return AgentResult(
                    agent_name=self.name,
                    content=result.text,
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
            from vertexai.generative_models import Content, Part

            fc_parts: list[Part] = []
            for fc in result.function_calls:
                fc_parts.append(
                    Part.from_dict({
                        "function_call": {
                            "name": fc["name"],
                            "args": fc["args"],
                        }
                    })
                )
            history.append(
                Content(role="model", parts=fc_parts)
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
                Content(role="user", parts=response_parts)
            )

            # Continue the loop — send empty prompt (history carries context)
            # The prompt for subsequent iterations is empty because
            # the conversation history already contains the full context.
            # generate_with_tools will use history to continue the conversation.

        # If we get here, max iterations exceeded
        msg = (
            f"Tool-use loop exceeded maximum iterations ({self._max_iterations}). "
            f"Executed {len(tools_executed)} tool calls."
        )
        logger.warning(msg)
        raise MaxIterationsError(msg, iterations=self._max_iterations)

    def execute_stream(self, prompt: str, *, context: str = "") -> Iterator[str]:
        """Streaming is not supported for the coding agent.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to non-streaming execute and yields the result.
        """
        logger.debug("CodingAgent.execute_stream() — falling back to non-streaming")
        result = self.execute(prompt, context=context)
        yield result.content

    # ── Task 4.5 — Tool execution with error handling ────────

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call with confirmation and error handling.

        - Checks confirmation for destructive operations
        - Catches ALL exceptions and returns them as ToolResult (never raises)
        - The model sees error results and can self-correct

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

        # Confirmation check for destructive operations
        if tool_name in _DESTRUCTIVE_TOOLS:
            if not self._confirm_fn(tool_name, tool_args):
                logger.info("User declined tool execution: %s", tool_name)
                return ToolResult(
                    output=f"User declined {tool_name} operation. "
                    "Try a different approach or explain what you want to do.",
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

    def _build_chat_history(self) -> list[Any]:
        """Convert agent conversation history to Vertex AI Content list.

        Unlike SpecialistAgent which uses ChatMessage, CodingAgent works
        directly with Content objects because the history may contain
        function call/response Parts (not just text).
        """
        from vertexai.generative_models import Content, Part

        contents: list[Content] = []
        for msg in self._conversation:
            role = "user" if msg.role == "user" else "model"
            contents.append(
                Content(role=role, parts=[Part.from_text(msg.content)])
            )
        return contents
