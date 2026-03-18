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


class TestDetectLanguagePortuguese:
    """Queries that should be classified as Portuguese."""

    def test_portuguese_simple(self) -> None:
        assert detect_language("Verifique o estado dos serviços") == "pt"

    def test_portuguese_question(self) -> None:
        assert detect_language("Qual é o estado do cluster?") == "pt"

    def test_portuguese_imperative(self) -> None:
        assert detect_language("Mostre o relatório de saúde dos pods") == "pt"

    def test_portuguese_casual(self) -> None:
        assert detect_language("Preciso saber se os pods estão bem") == "pt"

    def test_portuguese_formal(self) -> None:
        assert detect_language("Você pode analisar o serviço para mim?") == "pt"

    def test_portuguese_with_negation(self) -> None:
        assert detect_language("Não consigo ver os serviços no namespace") == "pt"


class TestDetectLanguageFrench:
    """Queries that should be classified as French."""

    def test_french_simple(self) -> None:
        assert detect_language("Vérifiez l'état des services") == "fr"

    def test_french_question(self) -> None:
        assert detect_language("Quel est l'état du cluster?") == "fr"

    def test_french_imperative(self) -> None:
        assert detect_language("Montrez le rapport de santé des pods") == "fr"

    def test_french_casual(self) -> None:
        assert detect_language("Je veux voir les services dans le namespace") == "fr"

    def test_french_polite(self) -> None:
        assert detect_language("Pouvez-vous analyser les problèmes pour nous?") == "fr"

    def test_french_with_comment(self) -> None:
        assert detect_language("Comment est la santé du cluster maintenant?") == "fr"


class TestDetectLanguageGerman:
    """Queries that should be classified as German."""

    def test_german_simple(self) -> None:
        assert detect_language("Zeige den Zustand der Dienste") == "de"

    def test_german_question(self) -> None:
        assert detect_language("Wie ist der Zustand des Clusters?") == "de"

    def test_german_imperative(self) -> None:
        assert detect_language("Überprüfen Sie den Bericht der Pods") == "de"

    def test_german_casual(self) -> None:
        assert detect_language("Ich muss wissen ob die Pods in Ordnung sind") == "de"

    def test_german_with_bitte(self) -> None:
        assert detect_language("Bitte zeige mir alle Fehler im Cluster") == "de"

    def test_german_formal(self) -> None:
        assert detect_language("Können Sie den Dienst analysieren?") == "de"


class TestDetectLanguageItalian:
    """Queries that should be classified as Italian."""

    def test_italian_simple(self) -> None:
        assert detect_language("Verifica lo stato dei servizi") == "it"

    def test_italian_question(self) -> None:
        assert detect_language("Qual è lo stato del cluster?") == "it"

    def test_italian_imperative(self) -> None:
        assert detect_language("Mostrami il rapporto di stato dei pods") == "it"

    def test_italian_casual(self) -> None:
        assert detect_language("Ho bisogno di sapere se i pods sono tutti in ordine") == "it"

    def test_italian_polite(self) -> None:
        assert detect_language("Può verificare il servizio per favore?") == "it"

    def test_italian_with_grazie(self) -> None:
        assert detect_language("Grazie per il rapporto sui servizi") == "it"


class TestDetectLanguageJapanese:
    """Queries that should be classified as Japanese."""

    def test_japanese_hiragana(self) -> None:
        assert detect_language("クラスターの状態を確認してください") == "ja"

    def test_japanese_katakana(self) -> None:
        assert detect_language("ポッドのステータスを見せて") == "ja"

    def test_japanese_mixed(self) -> None:
        assert detect_language("サービスの健全性レポートをお願いします") == "ja"

    def test_japanese_with_english(self) -> None:
        """Japanese text mixed with English technical terms."""
        assert detect_language("CrashLoopBackOff の pod を確認して") == "ja"

    def test_japanese_short(self) -> None:
        assert detect_language("状態は？") == "ja"


class TestDetectLanguageChinese:
    """Queries that should be classified as Chinese."""

    def test_chinese_simple(self) -> None:
        assert detect_language("检查集群的健康状态") == "zh"

    def test_chinese_question(self) -> None:
        assert detect_language("集群的状态是什么？") == "zh"

    def test_chinese_imperative(self) -> None:
        assert detect_language("显示服务健康报告") == "zh"

    def test_chinese_with_english(self) -> None:
        """Chinese text mixed with English technical terms."""
        assert detect_language("查看 namespace production 中的 pod 状态") == "zh"

    def test_chinese_traditional(self) -> None:
        assert detect_language("檢查叢集的健康狀態") == "zh"


class TestDetectLanguageKorean:
    """Queries that should be classified as Korean."""

    def test_korean_simple(self) -> None:
        assert detect_language("클러스터 상태를 확인해 주세요") == "ko"

    def test_korean_question(self) -> None:
        assert detect_language("클러스터의 상태는 어떤가요?") == "ko"

    def test_korean_imperative(self) -> None:
        assert detect_language("서비스 건강 보고서를 보여주세요") == "ko"

    def test_korean_with_english(self) -> None:
        """Korean text mixed with English technical terms."""
        assert detect_language("CrashLoopBackOff 상태인 pod를 확인해") == "ko"

    def test_korean_short(self) -> None:
        assert detect_language("상태 확인") == "ko"


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
        instruction = build_language_instruction("sw")
        assert "LANGUAGE INSTRUCTION" in instruction
        assert "sw" in instruction

    @pytest.mark.parametrize(
        ("lang_code", "lang_name"),
        [
            ("pt", "Portuguese"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("ja", "Japanese"),
            ("zh", "Chinese"),
            ("ko", "Korean"),
        ],
    )
    def test_new_languages_return_instruction(
        self,
        lang_code: str,
        lang_name: str,
    ) -> None:
        """All newly supported languages should produce a proper instruction."""
        instruction = build_language_instruction(lang_code)
        assert "LANGUAGE INSTRUCTION" in instruction
        assert lang_name in instruction
        assert f"MUST respond entirely in {lang_name}" in instruction


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
        original_prompts = [c.get("system_instruction") for c in configs]
        inject_language_into_config(configs, "en")
        new_prompts = [c.get("system_instruction") for c in configs]
        assert original_prompts == new_prompts

    def test_spanish_prepends_to_all_agents(self) -> None:
        """Spanish instruction should be prepended to ALL 4 agents."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        for config in configs:
            assert config["system_instruction"].startswith("## LANGUAGE INSTRUCTION")

    def test_spanish_preserves_original_content(self) -> None:
        """Original prompt content should still be present after injection."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        assert "You are a gatherer." in configs[0]["system_instruction"]
        assert "You are an analyzer." in configs[1]["system_instruction"]
        assert "You are a verifier." in configs[2]["system_instruction"]
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

    def test_system_instruction_key_updated(self) -> None:
        """Agents with system_instruction should have it updated."""
        configs = self._make_agent_configs()
        inject_language_into_config(configs, "es")

        # Gatherer has system_instruction
        gatherer = configs[0]
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
        settings.budget.max_cost_per_run = 0.0
        settings.agents.max_failures_before_fallback = 0
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
                agent_name="mock",
                content="OK",
                success=True,
                usage={},
            )
            return [agent]

        # Use a stub skill
        from vaig.skills.base import BaseSkill, SkillMetadata

        class StubSkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(
                    name="test",
                    display_name="Test",
                    description="Test",
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

    def test_english_query_no_injection(self) -> None:
        """When the query is in English, no language injection should happen."""
        from unittest.mock import MagicMock, patch

        from vaig.agents.base import AgentResult
        from vaig.agents.orchestrator import Orchestrator
        from vaig.tools.base import ToolRegistry

        client = MagicMock()
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        settings.budget.max_cost_per_run = 0.0
        settings.agents.max_failures_before_fallback = 0
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
                agent_name="mock",
                content="OK",
                success=True,
                usage={},
            )
            return [agent]

        from vaig.skills.base import BaseSkill, SkillMetadata

        class StubSkill(BaseSkill):
            def get_metadata(self) -> SkillMetadata:
                return SkillMetadata(
                    name="test",
                    display_name="Test",
                    description="Test",
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
        configs = skill.get_sequential_agents_config()

        # Inject Spanish
        inject_language_into_config(configs, "es")

        # All 4 agents should have the language instruction
        assert len(configs) == 4
        for config in configs:
            assert "LANGUAGE INSTRUCTION" in config["system_instruction"]
            assert "Spanish" in config["system_instruction"]

    def test_service_health_english_no_injection(self) -> None:
        """English should NOT modify service-health configs."""
        from vaig.skills.service_health.prompts import (
            HEALTH_ANALYZER_PROMPT,
            HEALTH_REPORTER_PROMPT,
            HEALTH_VERIFIER_PROMPT,
        )
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        configs = skill.get_sequential_agents_config()

        # Capture the gatherer prompt BEFORE language injection
        # (get_agents_config builds it dynamically based on settings)
        gatherer_before = configs[0]["system_instruction"]

        inject_language_into_config(configs, "en")

        # Prompts should be unchanged (English = no injection)
        assert configs[0]["system_instruction"] == gatherer_before
        assert configs[1]["system_instruction"] == HEALTH_ANALYZER_PROMPT
        assert configs[2]["system_instruction"] == HEALTH_VERIFIER_PROMPT
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

        assert original_gatherer == prompts.HEALTH_GATHERER_PROMPT
        assert original_reporter == prompts.HEALTH_REPORTER_PROMPT
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
        assert "GKE Autopilot cluster" in result
        assert "node-level actions" in result

    def test_instruction_mentions_mandatory_requests(self) -> None:
        result = build_autopilot_instruction(True)
        assert "mandatory" in result.lower()

    def test_instruction_mentions_google_manages_nodes(self) -> None:
        result = build_autopilot_instruction(True)
        assert (
            "fully managed by Google" in result
            or "Node infrastructure is fully managed by Google" in result
        )
        assert "CONTEXT ONLY" in result
        assert "node-level actions" in result


class TestInjectAutopilotIntoConfig:
    """Tests for inject_autopilot_into_config()."""

    def test_no_modification_when_not_autopilot(self) -> None:
        configs = [
            {"system_instruction": "original instruction"},
        ]
        result = inject_autopilot_into_config(configs, False)
        assert result[0]["system_instruction"] == "original instruction"

    def test_no_modification_when_none(self) -> None:
        configs = [
            {"system_instruction": "original prompt"},
        ]
        result = inject_autopilot_into_config(configs, None)
        assert result[0]["system_instruction"] == "original prompt"

    def test_prepends_instruction_when_autopilot(self) -> None:
        configs = [
            {"system_instruction": "You are an SRE agent."},
        ]
        result = inject_autopilot_into_config(configs, True)
        assert result[0]["system_instruction"].startswith("## GKE AUTOPILOT CLUSTER")
        assert "You are an SRE agent." in result[0]["system_instruction"]

    def test_mutates_in_place(self) -> None:
        configs = [{"system_instruction": "test"}]
        result = inject_autopilot_into_config(configs, True)
        assert result is configs

    def test_handles_multiple_agents(self) -> None:
        configs = [
            {"system_instruction": "agent 1"},
            {"system_instruction": "agent 2"},
            {"system_instruction": "agent 3"},
        ]
        inject_autopilot_into_config(configs, True)
        assert "GKE AUTOPILOT" in configs[0]["system_instruction"]
        assert "GKE AUTOPILOT" in configs[1]["system_instruction"]
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

        assert original == prompts.HEALTH_GATHERER_PROMPT
        assert "GKE AUTOPILOT" not in prompts.HEALTH_GATHERER_PROMPT
