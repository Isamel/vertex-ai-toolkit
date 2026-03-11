"""Agents package — multi-agent orchestration for skill-based tasks."""

from vaig.agents.base import AgentConfig, AgentMessage, AgentResult, AgentRole, BaseAgent
from vaig.agents.orchestrator import Orchestrator, OrchestratorResult
from vaig.agents.registry import AgentRegistry
from vaig.agents.specialist import SpecialistAgent

__all__ = [
    "AgentConfig",
    "AgentMessage",
    "AgentRegistry",
    "AgentResult",
    "AgentRole",
    "BaseAgent",
    "Orchestrator",
    "OrchestratorResult",
    "SpecialistAgent",
]
