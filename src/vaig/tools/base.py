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
class ToolDef:
    """Definition of a tool that an agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParam] = field(default_factory=list)
    execute: Callable[..., ToolResult] = field(default=lambda **_: ToolResult(output=""))


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
