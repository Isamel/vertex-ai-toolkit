"""Coding agent — autonomous file-editing agent using Gemini function calling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vaig.agents.base import AgentConfig, AgentResult, AgentRole, BaseAgent
from vaig.agents.mixins import ToolLoopMixin
from vaig.agents.utils import deduplicate_response
from vaig.core.client import GeminiClient
from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS, CodingConfig, Settings
from vaig.core.exceptions import MaxIterationsError
from vaig.tools import ToolRegistry, ToolResult, create_file_tools, create_shell_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

logger = logging.getLogger(__name__)


# ── Task 4.3 — System prompt ────────────────────────────────

CODING_SYSTEM_PROMPT = """\
You are an expert coding assistant with access to filesystem tools. Your goal is to \
produce COMPLETE, production-ready code that integrates seamlessly with the existing \
codebase.

## Workflow — Follow This Order

### Phase 1: Understand Before Coding
Before writing ANY code, you MUST:
1. Read existing files to understand the current codebase style, patterns, and conventions.
2. Use search_files to find related code, imports, and usage patterns.
3. Identify the architecture (module structure, naming conventions, error handling patterns).
4. Plan your changes — list what files need to be created or modified and why.

### Phase 2: Implement With Quality
When writing code, follow these rules strictly:
1. **Complete implementations only** — NEVER use TODO, FIXME, placeholder, `pass`, \
or `...` as a substitute for real logic. Every function must have a real body.
2. **Match existing code style** — Use the same naming conventions, formatting, \
import ordering, docstring style, and patterns found in the project.
3. **All imports included** — Every file must have all necessary imports. Never \
assume an import exists without verifying.
4. **Type hints required** — Add type annotations to all function signatures and \
return types. Use modern Python typing (PEP 604 unions, etc.) matching the project.
5. **Docstrings required** — Add docstrings to all public functions, classes, and \
modules following the project's existing docstring format.
6. **Error handling** — Handle edge cases and errors appropriately. Don't let \
exceptions pass silently.

### Phase 3: Verify Your Work
After making changes:
1. Re-read every modified file to confirm the changes are correct.
2. Check that imports resolve and there are no obvious syntax errors.
3. If tests exist, suggest running them to verify correctness.

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


class CodingAgent(BaseAgent, ToolLoopMixin):
    """Autonomous coding agent that uses Gemini function calling to edit files.

    Inherits the generic tool-use loop from :class:`ToolLoopMixin` and
    adds CodingAgent-specific behaviour:

    - Confirmation callback for destructive tools (write_file, edit_file,
      run_command) via an overridden ``_execute_single_tool``.
    - Response deduplication to remove pathological repetition.
    - Graceful error handling that returns ``AgentResult(success=False)``
      instead of propagating API exceptions.

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
        settings: Settings | None = None,
        confirm_fn: Callable[[str, dict[str, Any]], bool] | None = None,
        model_id: str | None = None,
    ) -> None:
        """Initialize the coding agent.

        Args:
            client: The GeminiClient for API calls.
            coding_config: Coding-specific configuration (workspace, limits, etc.).
            settings: Full application settings (used for plugin tool loading).
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
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
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
            denied_commands=coding_config.denied_commands or None,
        ):
            self._registry.register(tool)

        # Plugin tools — MCP auto-registration and Python module plugins
        if settings is not None:
            try:
                from vaig.tools.plugin_loader import load_all_plugin_tools  # noqa: WPS433

                for tool in load_all_plugin_tools(settings):
                    self._registry.register(tool)
            except Exception:
                logger.warning(
                    "Failed to load plugin tools for CodingAgent. Skipping.",
                    exc_info=True,
                )

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

    # ── Task 4.2 — Tool-use loop (via ToolLoopMixin) ───────────

    def execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a coding task using the tool-use loop.

        Delegates to :meth:`ToolLoopMixin._run_tool_loop` for the loop
        mechanics.  Wraps the result in an ``AgentResult`` and applies
        response deduplication.

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

        history = self._build_chat_history()

        logger.debug(
            "CodingAgent.execute() — starting tool loop (max=%d)",
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
            )
        except MaxIterationsError:
            raise  # Let the caller handle iteration exhaustion
        except Exception as exc:
            logger.exception("CodingAgent API call failed")
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
                success=False,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"error": str(exc), "iterations": 0},
            )

        clean_text = self._deduplicate_response(loop_result.text)
        logger.info(
            "CodingAgent completed — %d iterations, %d tool calls, %s total tokens",
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
        """Streaming is not supported for the coding agent.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to non-streaming execute and yields the result.
        """
        logger.debug("CodingAgent.execute_stream() — falling back to non-streaming")
        result = self.execute(prompt, context=context)
        yield result.content

    # ── Async methods ────────────────────────────────────────

    async def async_execute(self, prompt: str, *, context: str = "") -> AgentResult:
        """Execute a coding task using the async tool-use loop.

        Async version of :meth:`execute`.  Delegates to
        :meth:`ToolLoopMixin._async_run_tool_loop` for non-blocking LLM
        calls and tool execution.

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

        history = self._build_chat_history()

        logger.debug(
            "CodingAgent.async_execute() — starting async tool loop (max=%d)",
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
            )
        except MaxIterationsError:
            raise  # Let the caller handle iteration exhaustion
        except Exception as exc:
            logger.exception("CodingAgent async API call failed")
            return AgentResult(
                agent_name=self.name,
                content=f"Error during API call: {self.sanitize_error_for_agent(exc)}",
                success=False,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={"error": str(exc), "iterations": 0},
            )

        clean_text = self._deduplicate_response(loop_result.text)
        logger.info(
            "CodingAgent async completed — %d iterations, %d tool calls, %s total tokens",
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
        self, prompt: str, *, context: str = "",
    ) -> AsyncIterator[str]:
        """Async streaming — falls back to async_execute.

        Tool-use loops are inherently non-streamable because the model
        needs to receive function execution results between turns.
        Falls back to :meth:`async_execute` and yields the result.
        """
        logger.debug(
            "CodingAgent.async_execute_stream() — falling back to async non-streaming",
        )
        result = await self.async_execute(prompt, context=context)
        yield result.content

    # ── Tool execution override (adds confirmation) ──────────

    def _execute_single_tool(
        self,
        tool_registry: ToolRegistry,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with confirmation for destructive operations.

        Overrides ``ToolLoopMixin._execute_single_tool`` to add a
        confirmation callback before write_file / edit_file / run_command.
        """
        # Confirmation check for destructive operations
        if tool_name in _DESTRUCTIVE_TOOLS:
            if not self._confirm_fn(tool_name, tool_args):
                logger.info("User declined tool execution: %s", tool_name)
                return ToolResult(
                    output=f"User declined {tool_name} operation. "
                    "Try a different approach or explain what you want to do.",
                    error=True,
                )

        # Delegate to the base implementation for actual execution
        return super()._execute_single_tool(tool_registry, tool_name, tool_args)

    async def _async_execute_single_tool(
        self,
        tool_registry: ToolRegistry,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """Async tool execution with confirmation for destructive operations.

        Overrides ``ToolLoopMixin._async_execute_single_tool`` to add the
        same confirmation callback used by the sync version before
        write_file / edit_file / run_command.
        """
        # Confirmation check for destructive operations (sync callback is fine —
        # confirmation is a quick UI interaction, not I/O-bound)
        if tool_name in _DESTRUCTIVE_TOOLS:
            if not self._confirm_fn(tool_name, tool_args):
                logger.info("User declined async tool execution: %s", tool_name)
                return ToolResult(
                    output=f"User declined {tool_name} operation. "
                    "Try a different approach or explain what you want to do.",
                    error=True,
                )

        # Delegate to the async base implementation for actual execution
        return await super()._async_execute_single_tool(
            tool_registry, tool_name, tool_args,
        )

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _deduplicate_response(text: str, *, threshold: int = 3) -> str:
        """Remove repeated lines — delegates to shared ``deduplicate_response``."""
        return deduplicate_response(text, threshold=threshold)
