"""Base tool types — core dataclasses and registry for agent tool use."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from google.genai import types

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolParam:
    """Schema for a single tool parameter."""

    name: str
    type: str  # "string" | "integer" | "boolean" | "number" | "array" | "object"
    description: str
    required: bool = True


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    output: str
    error: bool = False


@dataclass
class ToolCallRecord:
    """Record of a single tool call execution for metrics and feedback."""

    tool_name: str
    tool_args: dict[str, Any]
    output: str  # Full, untruncated output
    output_size_bytes: int  # len(output.encode('utf-8'))
    error: bool
    error_type: str  # Exception class name, empty if no error
    error_message: str  # Error details, empty if no error
    duration_s: float  # Wall clock seconds
    timestamp: str  # ISO 8601 UTC
    agent_name: str  # Which agent made the call (gatherer, analyzer, etc.)
    run_id: str  # UUID for the execution run
    iteration: int  # Which iteration of the tool loop
    cached: bool = False  # True when result came from ToolResultCache
    redactions: int = 0  # Number of sensitive values redacted from output

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "output": self.output,
            "output_size_bytes": self.output_size_bytes,
            "error": self.error,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration_s": round(self.duration_s, 4),
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "cached": self.cached,
            "redactions": self.redactions,
        }


@dataclass
class ToolDef:
    """Definition of a tool that an agent can invoke.

    Attributes:
        cache_ttl_seconds: Per-entry TTL in seconds.  ``0`` means no
            expiration — the entry lives for the lifetime of the cache
            instance (which is one pipeline run).
        categories: Domain categories this tool belongs to.  Used by
            :meth:`ToolRegistry.filter_by_categories` to select only the
            tools relevant for a given agent.  Defaults to
            ``frozenset({"uncategorized"})`` so untagged tools are still
            discoverable.  Use constants from :mod:`vaig.tools.categories`
            (e.g. ``KUBERNETES``, ``HELM``, ``DATADOG``).
    """

    name: str
    description: str
    parameters: list[ToolParam] | None = None
    execute: Callable[..., ToolResult] = field(default=lambda **_: ToolResult(output=""))
    cacheable: bool = True
    cache_ttl_seconds: int = 0
    categories: frozenset[str] = field(default_factory=lambda: frozenset({"uncategorized"}))


class ToolRegistry:
    """Registry that holds tool definitions and converts them to Vertex AI declarations."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> ToolDef | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        """Return all registered tools."""
        return list(self._tools.values())

    def copy(self) -> ToolRegistry:
        """Return a shallow copy of this registry.

        The new registry contains the same :class:`ToolDef` references as the
        original.  Mutating the copy (registering or removing tools) does not
        affect the source registry.

        Returns:
            A new :class:`ToolRegistry` with the same tool definitions.
        """
        new_registry = ToolRegistry()
        new_registry._tools = self._tools.copy()
        return new_registry

    def filter_by_categories(self, categories: frozenset[str]) -> ToolRegistry:
        """Return a new registry containing only tools whose categories intersect *categories*.

        A tool is included when ``tool.categories & categories`` is non-empty —
        i.e. the tool belongs to at least one of the requested categories.

        Unlike :meth:`register`, this method assigns tools directly without
        emitting per-tool DEBUG logs (which would fire once per agent that
        receives a filtered copy of the registry).  A single summary log line
        is emitted instead.

        Args:
            categories: The set of category names to keep.  Use constants from
                :mod:`vaig.tools.categories` (e.g.
                ``frozenset({KUBERNETES, HELM})``).

        Returns:
            A new :class:`ToolRegistry` with the filtered subset of tools.
            The original registry is not modified.
        """
        filtered = ToolRegistry()
        for tool in self._tools.values():
            if tool.categories & categories:
                filtered._tools[tool.name] = tool  # direct assign — avoids per-tool log noise
        logger.debug(
            "Filtered tool registry: %d/%d tools for categories=%s",
            len(filtered._tools),
            len(self._tools),
            sorted(categories),
        )
        return filtered

    def to_function_declarations(self) -> list[types.FunctionDeclaration]:
        """Convert registered tools to google-genai FunctionDeclaration objects."""
        declarations: list[types.FunctionDeclaration] = []
        for tool in self._tools.values():
            params = tool.parameters or []
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {param.name: {"type": param.type, "description": param.description} for param in params},
                "required": [p.name for p in params if p.required],
            }
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=schema,
                )
            )
        return declarations
