"""Agents package — multi-agent orchestration for skill-based tasks."""

from vaig.agents.base import AgentConfig, AgentMessage, AgentResult, AgentRole, BaseAgent
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.agents.registry import AgentRegistry
from vaig.agents.specialist import SpecialistAgent

# Coding agent (Phase 4 — available after implementation)
try:
    from vaig.agents.coding import CodingAgent
except ImportError:
    pass

__all__ = [
    "AgentConfig",
    "AgentMessage",
    "AgentRegistry",
    "AgentResult",
    "AgentRole",
    "BaseAgent",
    "CodingAgent",
    "Orchestrator",
    "OrchestratorResult",
    "SpecialistAgent",
]
