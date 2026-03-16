"""Test Generation Skill — automated test suite generation."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.test_generation.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class TestGenerationSkill(BaseSkill):
    """Test Generation skill for creating comprehensive test suites from source code.

    Supports multi-agent execution with specialized agents:
    - Test Planner: Analyzes source code to identify testable units and create test plans
    - Test Writer: Generates production-ready test code with proper fixtures and assertions
    - Coverage Analyzer: Reviews tests for coverage gaps and quality issues
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="test-generation",
            display_name="Test Generation",
            description="Generate comprehensive test suites from source code with unit, integration, and edge case coverage",
            version="1.0.0",
            tags=["testing", "test-generation", "quality", "coverage", "tdd", "automation"],
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
                "name": "test_planner",
                "role": "Test Planner",
                "system_instruction": (
                    "You are a test planning specialist. Your job is to analyze source code "
                    "and identify every testable unit — functions, methods, classes, and API "
                    "endpoints. For each unit, you map input/output contracts, side effects, "
                    "error conditions, boundary values, state transitions, and dependencies "
                    "that require mocking or stubbing. You create structured test plans with "
                    "prioritized test cases covering happy paths, edge cases, error paths, "
                    "and integration points. You assess risk and complexity to guide test "
                    "effort allocation."
                ),
                "model": "gemini-2.5-flash",
            },
            {
                "name": "test_writer",
                "role": "Test Writer",
                "system_instruction": (
                    "You are a senior test automation engineer. Your job is to generate "
                    "production-ready test code following the conventions of the detected "
                    "language and testing framework (pytest, jest, JUnit 5, Go testing, etc.). "
                    "You write clean, maintainable tests with proper fixtures, mocks, and "
                    "assertions. Every test follows Arrange-Act-Assert structure with "
                    "descriptive names (test_should_X_when_Y). You use parameterized tests "
                    "to eliminate duplication, mock at boundaries not internals, and include "
                    "assertion messages that explain the expectation. Your tests are "
                    "deterministic, independent, and fast."
                ),
                "model": "gemini-2.5-pro",
            },
            {
                "name": "coverage_analyzer",
                "role": "Coverage Analyzer",
                "system_instruction": (
                    "You are a test coverage and quality analyst. Your job is to review "
                    "generated test suites for coverage gaps — uncovered branches, missing "
                    "edge cases, untested error paths, and weak assertions. You identify "
                    "test smells (duplicate logic, tight implementation coupling, magic "
                    "values, missing boundary tests) and suggest specific additional test "
                    "cases to close gaps. You evaluate mutation testing opportunities where "
                    "arithmetic, conditional, or return-value mutations could survive the "
                    "current suite. You produce quantified coverage reports with actionable "
                    "improvement priorities."
                ),
                "model": "gemini-2.5-flash",
            },
        ]
