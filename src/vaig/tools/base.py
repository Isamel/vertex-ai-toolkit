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
    output: str                    # Full, untruncated output
    output_size_bytes: int         # len(output.encode('utf-8'))
    error: bool
    error_type: str                # Exception class name, empty if no error
    error_message: str             # Error details, empty if no error
    duration_s: float              # Wall clock seconds
    timestamp: str                 # ISO 8601 UTC
    agent_name: str                # Which agent made the call (gatherer, analyzer, etc.)
    run_id: str                    # UUID for the execution run
    iteration: int                 # Which iteration of the tool loop
    cached: bool = False           # True when result came from ToolResultCache

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
        }


@dataclass
class ToolDef:
    """Definition of a tool that an agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParam] = field(default_factory=list)
    execute: Callable[..., ToolResult] = field(default=lambda **_: ToolResult(output=""))
    cacheable: bool = True
    cache_ttl_seconds: int = 60


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

    def to_function_declarations(self) -> list[types.FunctionDeclaration]:
        """Convert registered tools to google-genai FunctionDeclaration objects."""
        declarations: list[types.FunctionDeclaration] = []
        for tool in self._tools.values():
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    param.name: {"type": param.type, "description": param.description}
                    for param in tool.parameters
                },
                "required": [p.name for p in tool.parameters if p.required],
            }
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=schema,
                )
            )
        return declarations
