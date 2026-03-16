"""Runbook Generator Skill — operational runbook creation and maintenance."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.runbook_generator.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class RunbookGeneratorSkill(BaseSkill):
    """Runbook Generator skill for creating operational runbooks for production systems.

    Supports multi-agent execution with specialized agents:
    - Procedure Analyst: Analyzes system context and identifies operational procedures needed
    - Step Writer: Generates detailed step-by-step instructions with verification and rollback
    - Runbook Lead: Synthesizes analysis into a complete, production-ready runbook
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="runbook-generator",
            display_name="Runbook Generator",
            description="Generate operational runbooks with step-by-step procedures, decision trees, rollback plans, and escalation paths",
            version="1.0.0",
            tags=["runbook", "sre", "operations", "incident", "playbook", "procedures", "on-call"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "procedure_analyst",
                "role": "Procedure Analyst",
                "system_instruction": (
                    "You are an operational procedure analysis specialist. Your job is to "
                    "understand the system under review, identify the procedures that need "
                    "to be documented, assess risks and prerequisites, and map out decision "
                    "points and failure modes. You focus on who will execute the runbook, "
                    "what can go wrong, and what access and tools are required."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "step_writer",
                "role": "Step Writer",
                "system_instruction": (
                    "You are a runbook step writing specialist. Your job is to convert "
                    "operational procedures into clear, unambiguous, step-by-step instructions "
                    "that someone under stress can follow at 3 AM. Every step includes exact "
                    "commands, expected outputs, verification checks, and rollback instructions. "
                    "You write decision trees with clear if/then/else branching and include "
                    "time estimates for each step."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "runbook_lead",
                "role": "Runbook Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
            },
        ]
