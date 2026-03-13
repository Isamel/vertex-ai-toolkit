"""Agents package — multi-agent orchestration for skill-based tasks."""

import logging

from vaig.agents.base import AgentConfig, AgentMessage, AgentResult, AgentRole, BaseAgent
from vaig.agents.mixins import ToolLoopMixin, ToolLoopResult
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.agents.registry import AgentRegistry
from vaig.agents.specialist import SpecialistAgent
from vaig.agents.tool_aware import ToolAwareAgent

_logger = logging.getLogger(__name__)

# Coding agent (Phase 4 — available after implementation)
try:
    from vaig.agents.coding import CodingAgent
except ImportError:
    _logger.debug("CodingAgent not available — optional dependency missing")

# Infra agent — requires kubernetes / google-cloud packages
try:
    from vaig.agents.infra_agent import InfraAgent
except ImportError:
    _logger.debug("InfraAgent not available — kubernetes package not installed")

__all__ = [
    "AgentConfig",
    "AgentMessage",
    "AgentRegistry",
    "AgentResult",
    "AgentRole",
    "BaseAgent",
    "CodingAgent",
    "InfraAgent",
    "Orchestrator",
    "OrchestratorResult",
    "SpecialistAgent",
    "ToolAwareAgent",
    "ToolLoopMixin",
    "ToolLoopResult",
]
