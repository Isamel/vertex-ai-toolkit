"""Tests for dynamic language detection and injection.

Validates:
- detect_language() correctly identifies Spanish vs English
- build_language_instruction() produces appropriate instructions
- inject_language_into_config() modifies agent configs in place
- Orchestrator.execute_with_tools() integrates language detection
"""

from __future__ import annotations

import pytest

from vaig.core.language import (
    build_autopilot_instruction,
    build_language_instruction,
    detect_language,
    inject_autopilot_into_config,
    inject_language_into_config,
)


# ===========================================================================
# detect_language() — heuristic-based detection
# ===========================================================================


class TestDetectLanguageEnglish:
    """Queries that should be classified as English."""

    def test_english_simple(self) -> None:
        assert detect_language("Check the health of my cluster") == "en"

    def test_english_technical(self) -> None:
        assert detect_language("Show me pods in CrashLoopBackOff") == "en"

    def test_english_question(self) -> None:
        assert detect_language("What is the status of the production namespace?") == "en"

    def test_english_imperative(self) -> None:
        assert detect_language("Run a health check on namespace staging") == "en"

    def test_english_short(self) -> None:
        assert detect_language("cluster health") == "en"

    def test_english_with_k8s_terms(self) -> None:
        assert detect_language("List all deployments with unavailable replicas") == "en"


class TestDetectLanguageSpanish:
    """Queries that should be classified as Spanish."""

    def test_spanish_simple(self) -> None:
        assert detect_language("Revisá el estado de los servicios") == "es"

    def test_spanish_question(self) -> None:
        assert detect_language("¿Cuál es el estado del cluster?") == "es"

    def test_spanish_inverted_exclamation(self) -> None:
        assert detect_language("¡Dame el reporte de salud!") == "es"

    def test_spanish_formal(self) -> None:
        assert detect_language("Muestra el estado de salud de los pods en producción") == "es"

    def test_spanish_imperative(self) -> None:
        assert detect_language("Dame un reporte de los servicios") == "es"

    def test_spanish_with_namespace(self) -> None:
        assert detect_language("Verificar la salud del namespace production") == "es"

    def test_spanish_casual(self) -> None:
        assert detect_language("Necesito saber si los pods están bien") == "es"

    def test_spanish_mixed_technical(self) -> None:
        """Spanish sentence with English technical terms."""
        assert detect_language("Analizá los pods en CrashLoopBackOff del namespace staging") == "es"

    def test_spanish_short_question(self) -> None:
        assert detect_language("¿Cómo está el cluster?") == "es"

    def test_spanish_without_accents(self) -> None:
        """Users often skip accent marks — should still detect Spanish."""
        assert detect_language("Como esta el cluster de produccion") == "es"


class TestDetectLanguageEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self) -> None:
        assert detect_language("") == "en"

    def test_whitespace_only(self) -> None:
        assert detect_language("   ") == "en"

    def test_single_word_english(self) -> None:
        assert detect_language("health") == "en"

    def test_single_spanish_indicator_below_threshold(self) -> None:
        """A single Spanish word should NOT be enough (threshold is 2)."""
        assert detect_language("pods en production") == "en"

    def test_inverted_question_mark_alone(self) -> None:
        """Even with just ¿, classify as Spanish."""
        assert detect_language("¿status?") == "es"

    def test_numbers_and_symbols(self) -> None:
        assert detect_language("3 pods, 2 nodes, 100% CPU") == "en"

    def test_kubectl_command(self) -> None:
        """Pure kubectl commands should be English."""
        assert detect_language("kubectl get pods -n production") == "en"


# ===========================================================================
# build_language_instruction()
# ===========================================================================


class TestBuildLanguageInstruction:
    """Tests for the language instruction builder."""

    def test_english_returns_empty(self) -> None:
        """English needs no language instruction — prompts are already English."""
        assert build_language_instruction("en") == ""

    def test_spanish_returns_instruction(self) -> None:
        instruction = build_language_instruction("es")
        assert "LANGUAGE INSTRUCTION" in instruction
        assert "Spanish" in instruction
        assert "MUST respond entirely in Spanish" in instruction

    def test_spanish_preserves_technical_terms(self) -> None:
        """The instruction should tell agents to keep technical terms in English."""
        instruction = build_language_instruction("es")
        assert "Technical terms" in instruction
        assert "English" in instruction

    def test_unknown_language_uses_code(self) -> None:
        """Unknown language codes should still produce an instruction."""
        instruction = build_language_instruction("fr")
        assert "LANGUAGE INSTRUCTION" in instruction
        assert "fr" in instruction


# ===========================================================================
# inject_language_into_config()
# ===========================================================================


class TestInjectLanguageIntoConfig:
    """Tests for runtime language injection into agent configs."""

    def _make_agent_configs(self) -> list[dict]:
        """Create a sample 4-agent config similar to ServiceHealthSkill."""
        return [
            {
                "name": "gatherer",
                "role": "Gatherer",
                "system_prompt": "You are a gatherer.",
                "system_instruction": "You are a gatherer.",
                "requires_tools": True,
            },
            {
                "name": "analyzer",
                "role": "Analyzer",
                "system_instruction": "You are an analyzer.",
                "requires_tools": False,
            },
            {
                "name": "verifier",
                "role": "Verifier",
                "system_prompt": "You are a verifier.",
                "system_instruction": "You are a verifier.",
                "requires_tools": True,
            },
            {
                "name": "reporter",
                "role": "Reporter",
                "system_instruction": "You are a reporter.",
                "requires_tools": False,
            },
        ]

    def test_english_no_modification(self) -> None:
        """English should not modify any configs."""
        configs = self._make_agent_configs()
        original_prompts = [c.get("system_prompt", c.get("system_instruction")) for c in configs]
        inject_language_into_config(configs, "en")
        new_prompts = [c.get("system_prompt", c.get("system_instruction")) for c in configs]
        assert original_prompts == new_prompts

    def test_spanish_prepends_to_all_agents(self) -> None:
        """Spanish instruction should be prepended to ALL 4 agents."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        for config in configs:
            if "system_instruction" in config:
                assert config["system_instruction"].startswith("## LANGUAGE INSTRUCTION")
            if "system_prompt" in config:
                assert config["system_prompt"].startswith("## LANGUAGE INSTRUCTION")

    def test_spanish_preserves_original_content(self) -> None:
        """Original prompt content should still be present after injection."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        assert "You are a gatherer." in configs[0]["system_prompt"]
        assert "You are an analyzer." in configs[1]["system_instruction"]
        assert "You are a verifier." in configs[2]["system_prompt"]
        assert "You are a reporter." in configs[3]["system_instruction"]

    def test_returns_same_list(self) -> None:
        """inject_language_into_config returns the same list reference."""
        configs = self._make_agent_configs()
        result = inject_language_into_config(configs, "es")
        assert result is configs

    def test_empty_config_list(self) -> None:
        """Empty config list should not raise."""
        configs: list[dict] = []
        result = inject_language_into_config(configs, "es")
        assert result == []

    def test_both_keys_updated(self) -> None:
        """Agents with both system_prompt AND system_instruction should have BOTH updated."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        # Gatherer has both keys
        gatherer = configs[0]
        assert gatherer["system_prompt"].startswith("## LANGUAGE INSTRUCTION")
        assert gatherer["system_instruction"].startswith("## LANGUAGE INSTRUCTION")


# ===========================================================================
# Integration: Orchestrator.execute_with_tools language detection
# ===========================================================================


class TestOrchestratorLanguageIntegration:
    """Test that the orchestrator detects language and injects it into agents."""

    def test_spanish_query_injects_language(self) -> None:
        """When the query is in Spanish, agents should receive language-injected prompts."""
        from unittest.mock import MagicMock, patch

        from vaig.agents.base import AgentResult
        from vaig.agents.orchestrator import Orchestrator
        from vaig.tools.base import ToolRegistry

        client = MagicMock()
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        orchestrator = Orchestrator(client, settings)
        registry = ToolRegistry()

        # Track what configs are passed to create_agents_for_skill
        captured_configs: list[dict] = []

        original_create = orchestrator.create_agents_for_skill

        def mock_create(skill, tool_registry=None, *, agent_configs=None):
            if agent_configs:
                captured_configs.extend(agent_configs)
            # Return mock agents to avoid actual Gemini calls
            agent = MagicMock()
            agent.name = "mock"
            agent.role = "Mock"
            agent.execute.return_value = AgentResult(
                agent_name="mock", content="OK", success=True, usage={},
            )
            return [agent]

        # Use a stub skill
        from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase

        class StubSkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(
                    name="test", display_name="Test", description="Test",
                )

            def get_system_instruction(self) -> str:
                return "Test instruction"

            def get_phase_prompt(self, phase, context, user_input) -> str:
                return "test"

            def get_agents_config(self) -> list[dict]:
                return [
                    {
                        "name": "test-agent",
                        "role": "Tester",
                        "system_instruction": "You are a tester.",
                        "system_prompt": "You are a tester.",
                        "requires_tools": True,
                    },
                ]

        with patch.object(orchestrator, "create_agents_for_skill", side_effect=mock_create):
            orchestrator.execute_with_tools(
                "¿Cuál es el estado de los servicios?",
                StubSkill(),
                registry,
            )

        # Verify language instruction was injected
        assert len(captured_configs) == 1
        assert "LANGUAGE INSTRUCTION" in captured_configs[0]["system_instruction"]
        assert "Spanish" in captured_configs[0]["system_instruction"]
        assert "LANGUAGE INSTRUCTION" in captured_configs[0]["system_prompt"]

    def test_english_query_no_injection(self) -> None:
        """When the query is in English, no language injection should happen."""
        from unittest.mock import MagicMock, patch

        from vaig.agents.base import AgentResult
        from vaig.agents.orchestrator import Orchestrator
        from vaig.tools.base import ToolRegistry

        client = MagicMock()
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        orchestrator = Orchestrator(client, settings)
        registry = ToolRegistry()

        captured_configs: list[dict] = []

        def mock_create(skill, tool_registry=None, *, agent_configs=None):
            if agent_configs:
                captured_configs.extend(agent_configs)
            agent = MagicMock()
            agent.name = "mock"
            agent.role = "Mock"
            agent.execute.return_value = AgentResult(
                agent_name="mock", content="OK", success=True, usage={},
            )
            return [agent]

        from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase

        class StubSkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(
                    name="test", display_name="Test", description="Test",
                )

            def get_system_instruction(self) -> str:
                return "Test instruction"

            def get_phase_prompt(self, phase, context, user_input) -> str:
                return "test"

            def get_agents_config(self) -> list[dict]:
                return [
                    {
                        "name": "test-agent",
                        "role": "Tester",
                        "system_instruction": "You are a tester.",
                        "system_prompt": "You are a tester.",
                        "requires_tools": True,
                    },
                ]

        with patch.object(orchestrator, "create_agents_for_skill", side_effect=mock_create):
            orchestrator.execute_with_tools(
                "Check the health of my cluster",
                StubSkill(),
                registry,
            )

        # Verify NO language injection for English
        assert len(captured_configs) == 1
        assert captured_configs[0]["system_instruction"] == "You are a tester."
        assert "LANGUAGE INSTRUCTION" not in captured_configs[0]["system_instruction"]


# ===========================================================================
# Integration: ServiceHealthSkill + language detection
# ===========================================================================


class TestServiceHealthLanguageIntegration:
    """Verify language injection works with the real ServiceHealthSkill configs."""

    def test_service_health_spanish_injection(self) -> None:
        """All 4 service-health agents should get Spanish language instructions."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()

        # Inject Spanish
        inject_language_into_config(configs, "es")

        # All 4 agents should have the language instruction
        assert len(configs) == 4
        for config in configs:
            if "system_instruction" in config:
                assert "LANGUAGE INSTRUCTION" in config["system_instruction"]
                assert "Spanish" in config["system_instruction"]
            if "system_prompt" in config:
                assert "LANGUAGE INSTRUCTION" in config["system_prompt"]
                assert "Spanish" in config["system_prompt"]

    def test_service_health_english_no_injection(self) -> None:
        """English should NOT modify service-health configs."""
        from vaig.skills.service_health.prompts import (
            HEALTH_ANALYZER_PROMPT,
            HEALTH_GATHERER_PROMPT,
            HEALTH_REPORTER_PROMPT,
            HEALTH_VERIFIER_PROMPT,
        )
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()

        inject_language_into_config(configs, "en")

        # Prompts should be unchanged
        assert configs[0]["system_prompt"] == HEALTH_GATHERER_PROMPT
        assert configs[1]["system_instruction"] == HEALTH_ANALYZER_PROMPT
        assert configs[2]["system_prompt"] == HEALTH_VERIFIER_PROMPT
        assert configs[3]["system_instruction"] == HEALTH_REPORTER_PROMPT

    def test_prompt_constants_not_mutated(self) -> None:
        """Prompt constants must NEVER be mutated by language injection.

        This is critical — get_agents_config() returns new dicts each call,
        so mutating them should not affect the module-level constants.
        """
        from vaig.skills.service_health.prompts import (
            HEALTH_GATHERER_PROMPT,
            HEALTH_REPORTER_PROMPT,
        )
        from vaig.skills.service_health.skill import ServiceHealthSkill

        # Capture original values
        original_gatherer = HEALTH_GATHERER_PROMPT
        original_reporter = HEALTH_REPORTER_PROMPT

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()
        inject_language_into_config(configs, "es")

        # Module-level constants must be untouched
        from vaig.skills.service_health import prompts

        assert prompts.HEALTH_GATHERER_PROMPT == original_gatherer
        assert prompts.HEALTH_REPORTER_PROMPT == original_reporter
        assert "LANGUAGE INSTRUCTION" not in prompts.HEALTH_GATHERER_PROMPT
        assert "LANGUAGE INSTRUCTION" not in prompts.HEALTH_REPORTER_PROMPT


# ══════════════════════════════════════════════════════════════
# Autopilot context injection tests
# ══════════════════════════════════════════════════════════════


class TestBuildAutopilotInstruction:
    """Tests for build_autopilot_instruction()."""

    def test_returns_empty_for_false(self) -> None:
        assert build_autopilot_instruction(False) == ""

    def test_returns_empty_for_none(self) -> None:
        assert build_autopilot_instruction(None) == ""

    def test_returns_instruction_for_true(self) -> None:
        result = build_autopilot_instruction(True)
        assert "GKE AUTOPILOT CLUSTER" in result
        assert "confirmed as GKE Autopilot" in result
        assert "node-level ACTIONS" in result

    def test_instruction_mentions_mandatory_requests(self) -> None:
        result = build_autopilot_instruction(True)
        assert "mandatory" in result.lower()

    def test_instruction_mentions_google_manages_scaling(self) -> None:
        result = build_autopilot_instruction(True)
        assert "Google manages node scaling" in result


class TestInjectAutopilotIntoConfig:
    """Tests for inject_autopilot_into_config()."""

    def test_no_modification_when_not_autopilot(self) -> None:
        configs = [
            {"system_prompt": "original prompt", "system_instruction": "original instruction"},
        ]
        result = inject_autopilot_into_config(configs, False)
        assert result[0]["system_prompt"] == "original prompt"
        assert result[0]["system_instruction"] == "original instruction"

    def test_no_modification_when_none(self) -> None:
        configs = [
            {"system_prompt": "original prompt"},
        ]
        result = inject_autopilot_into_config(configs, None)
        assert result[0]["system_prompt"] == "original prompt"

    def test_prepends_instruction_when_autopilot(self) -> None:
        configs = [
            {"system_prompt": "You are an SRE agent.", "system_instruction": "You are an SRE agent."},
        ]
        result = inject_autopilot_into_config(configs, True)
        assert result[0]["system_prompt"].startswith("## GKE AUTOPILOT CLUSTER")
        assert result[0]["system_instruction"].startswith("## GKE AUTOPILOT CLUSTER")
        assert "You are an SRE agent." in result[0]["system_prompt"]

    def test_mutates_in_place(self) -> None:
        configs = [{"system_prompt": "test"}]
        result = inject_autopilot_into_config(configs, True)
        assert result is configs

    def test_handles_multiple_agents(self) -> None:
        configs = [
            {"system_prompt": "agent 1"},
            {"system_instruction": "agent 2"},
            {"system_prompt": "agent 3", "system_instruction": "agent 3"},
        ]
        inject_autopilot_into_config(configs, True)
        assert "GKE AUTOPILOT" in configs[0]["system_prompt"]
        assert "GKE AUTOPILOT" in configs[1]["system_instruction"]
        assert "GKE AUTOPILOT" in configs[2]["system_prompt"]
        assert "GKE AUTOPILOT" in configs[2]["system_instruction"]

    def test_does_not_corrupt_original_prompt_constants(self) -> None:
        """Injecting Autopilot must not modify module-level prompt constants."""
        from vaig.skills.service_health.prompts import HEALTH_GATHERER_PROMPT
        from vaig.skills.service_health.skill import ServiceHealthSkill

        original = HEALTH_GATHERER_PROMPT

        skill = ServiceHealthSkill()
        configs = skill.get_agents_config()
        inject_autopilot_into_config(configs, True)

        from vaig.skills.service_health import prompts
        assert prompts.HEALTH_GATHERER_PROMPT == original
        assert "GKE AUTOPILOT" not in prompts.HEALTH_GATHERER_PROMPT
