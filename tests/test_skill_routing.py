"""Tests for conditional skill routing — suggest_skill, _tokenize_query, _score_skill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.skills.base import SkillMetadata, SkillPhase
from vaig.skills.registry import SkillRegistry, _score_skill, _tokenize_query

# ── _tokenize_query ──────────────────────────────────────────


class TestTokenizeQuery:
    def test_lowercases(self) -> None:
        assert _tokenize_query("PODS Crashing") == ["pods", "crashing"]

    def test_removes_stop_words(self) -> None:
        result = _tokenize_query("Why are my pods crashing")
        assert "why" not in result
        assert "are" not in result
        assert "my" not in result
        assert "pods" in result
        assert "crashing" in result

    def test_preserves_hyphens(self) -> None:
        result = _tokenize_query("error-budget is burning")
        assert "error-budget" in result

    def test_removes_single_char(self) -> None:
        result = _tokenize_query("I have a problem")
        # "i" and "a" are stop words AND single chars
        assert "i" not in result
        assert "a" not in result

    def test_empty_query(self) -> None:
        assert _tokenize_query("") == []

    def test_only_stop_words(self) -> None:
        assert _tokenize_query("why is my the a") == []

    def test_special_characters(self) -> None:
        result = _tokenize_query("pods!!! crashing??? @#$")
        assert result == ["pods", "crashing"]

    def test_numbers(self) -> None:
        result = _tokenize_query("pod 503 errors")
        assert "503" in result
        assert "errors" in result


# ── _score_skill ─────────────────────────────────────────────


def _make_meta(
    name: str = "test",
    tags: list[str] | None = None,
    description: str = "A test skill",
) -> SkillMetadata:
    """Helper to create SkillMetadata for tests."""
    return SkillMetadata(
        name=name,
        display_name=name.replace("-", " ").title(),
        description=description,
        tags=tags or [],
    )


class TestScoreSkill:
    def test_exact_tag_match_highest(self) -> None:
        meta = _make_meta(tags=["incident", "debugging", "logs"])
        score = _score_skill(["incident"], meta)
        assert score == 2.0

    def test_name_match(self) -> None:
        meta = _make_meta(name="log-analysis", tags=[])
        score = _score_skill(["log"], meta)
        assert score == 1.5

    def test_description_match(self) -> None:
        meta = _make_meta(description="Analyze pod crashes and errors")
        score = _score_skill(["crashes"], meta)
        assert score == 0.5

    def test_tag_beats_description(self) -> None:
        # "debugging" is a tag AND in description
        meta_with_tag = _make_meta(
            tags=["debugging"],
            description="helps with debugging",
        )
        meta_desc_only = _make_meta(
            tags=[],
            description="helps with debugging",
        )
        score_tag = _score_skill(["debugging"], meta_with_tag)
        score_desc = _score_skill(["debugging"], meta_desc_only)
        assert score_tag > score_desc

    def test_no_match_returns_zero(self) -> None:
        meta = _make_meta(tags=["security"], description="Security auditing")
        score = _score_skill(["pods", "crashing"], meta)
        assert score == 0.0

    def test_multiple_words_normalized(self) -> None:
        meta = _make_meta(tags=["incident", "logs"])
        # 2 words: "incident" matches tag (2.0), "pods" matches nothing (0.0)
        score = _score_skill(["incident", "pods"], meta)
        assert score == pytest.approx(1.0)  # 2.0 / 2

    def test_combined_tag_name_description(self) -> None:
        meta = _make_meta(
            name="log-analysis",
            tags=["logs", "sre"],
            description="Analyze application logs for patterns",
        )
        # "logs" -> tag(2.0) + desc(0.5), "analysis" -> name(1.5), "patterns" -> desc(0.5)
        score = _score_skill(["logs", "analysis", "patterns"], meta)
        expected = (2.0 + 0.5 + 1.5 + 0.5) / 3
        assert score == pytest.approx(expected)

    def test_case_insensitive_tags(self) -> None:
        meta = _make_meta(tags=["SRE", "Incident"])
        score = _score_skill(["sre"], meta)
        assert score == 2.0


# ── suggest_skill (integration with real skills) ─────────────


@pytest.fixture
def _mock_registry(monkeypatch: pytest.MonkeyPatch) -> SkillRegistry:
    """Create a SkillRegistry with mocked skills (no config/file loading)."""

    class _FakeSettings:
        class skills:
            enabled: list[str] = []
            custom_dir: str | None = None

    registry = SkillRegistry(_FakeSettings())  # type: ignore[arg-type]

    # Manually populate the metadata cache (skip actual loading)
    registry._metadata_cache = {
        "rca": SkillMetadata(
            name="rca",
            display_name="Root Cause Analysis",
            description="Perform root cause analysis on incidents and outages",
            tags=["incident", "sre", "debugging", "post-mortem", "logs", "metrics"],
        ),
        "log-analysis": SkillMetadata(
            name="log-analysis",
            display_name="Log Analysis",
            description="Analyze application and infrastructure logs for patterns",
            tags=["logs", "sre", "diagnostics", "patterns", "incident", "observability"],
        ),
        "cost-analysis": SkillMetadata(
            name="cost-analysis",
            display_name="Cost Analysis",
            description="Analyze cloud costs and find optimization opportunities",
            tags=["cost", "finops", "cloud", "optimization", "billing", "savings"],
        ),
        "config-audit": SkillMetadata(
            name="config-audit",
            display_name="Config Audit",
            description="Audit infrastructure configurations for security and reliability",
            tags=["config", "sre", "security", "audit", "compliance", "infrastructure", "reliability"],
        ),
        "code-review": SkillMetadata(
            name="code-review",
            display_name="Code Review",
            description="Review code for quality, security, and maintainability",
            tags=["code-review", "quality", "security", "maintainability", "best-practices"],
        ),
    }
    registry._loaded = True

    return registry


class TestSuggestSkill:
    def test_incident_debugging_suggests_rca_or_log_analysis(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("debugging an incident with logs")
        names = [s[0] for s in suggestions]
        assert len(names) > 0
        # rca has "debugging", "incident", "logs" tags
        # log-analysis has "logs", "incident" tags
        assert any(n in ("rca", "log-analysis") for n in names)

    def test_cost_optimization_suggests_cost_analysis(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("cost optimization for cloud resources")
        assert len(suggestions) > 0
        assert suggestions[0][0] == "cost-analysis"

    def test_unrelated_query_returns_empty(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("What is the meaning of life?")
        assert suggestions == []

    def test_respects_top_n(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("sre incident logs debugging", top_n=2)
        assert len(suggestions) <= 2

    def test_sorted_by_score_descending(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("sre incident logs")
        if len(suggestions) > 1:
            scores = [s[1] for s in suggestions]
            assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("")
        assert suggestions == []

    def test_stop_words_only_returns_empty(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("what is the")
        assert suggestions == []

    def test_code_review_query(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("review my code for security issues")
        assert len(suggestions) > 0
        names = [s[0] for s in suggestions]
        assert "code-review" in names

    def test_config_audit_query(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("audit infrastructure config for compliance")
        assert len(suggestions) > 0
        assert suggestions[0][0] == "config-audit"

    def test_scores_are_positive(self, _mock_registry: SkillRegistry) -> None:
        suggestions = _mock_registry.suggest_skill("sre incident logs metrics debugging")
        for _name, score in suggestions:
            assert score > 0

    def test_top_n_default_is_three(self, _mock_registry: SkillRegistry) -> None:
        # Query that matches many skills via "sre" tag
        suggestions = _mock_registry.suggest_skill("sre incident logs debugging metrics cost audit")
        assert len(suggestions) <= 3


# ── SkillsConfig auto_routing defaults ──────────────────────


class TestSkillsConfigAutoRouting:
    def test_auto_routing_default_enabled(self) -> None:
        from vaig.core.config import SkillsConfig

        cfg = SkillsConfig()
        assert cfg.auto_routing is True

    def test_auto_routing_default_threshold(self) -> None:
        from vaig.core.config import SkillsConfig

        cfg = SkillsConfig()
        assert cfg.auto_routing_threshold == 1.5

    def test_auto_routing_can_be_disabled(self) -> None:
        from vaig.core.config import SkillsConfig

        cfg = SkillsConfig(auto_routing=False)
        assert cfg.auto_routing is False

    def test_auto_routing_threshold_configurable(self) -> None:
        from vaig.core.config import SkillsConfig

        cfg = SkillsConfig(auto_routing_threshold=2.0)
        assert cfg.auto_routing_threshold == 2.0


# ── _try_auto_route_skill (REPL integration) ────────────────


def _make_repl_state(
    *,
    auto_routing: bool = True,
    threshold: float = 1.5,
) -> MagicMock:
    """Create a mock REPLState for auto-routing tests."""
    from vaig.core.config import SkillsConfig

    state = MagicMock()
    state.settings.skills = SkillsConfig(
        enabled=[],
        auto_routing=auto_routing,
        auto_routing_threshold=threshold,
    )
    state.active_skill = None
    state.current_phase = SkillPhase.ANALYZE
    state.live_mode = False
    return state


class TestTryAutoRouteSkill:
    """Tests for the _try_auto_route_skill REPL function."""

    def test_returns_false_when_disabled(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state(auto_routing=False)
        result = _try_auto_route_skill(state, "debug my incident logs")
        assert result is False

    def test_returns_false_when_no_suggestions(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state()
        state.skill_registry.suggest_skill.return_value = []
        result = _try_auto_route_skill(state, "meaning of life")
        assert result is False

    def test_returns_false_when_below_threshold(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state(threshold=2.0)
        state.skill_registry.suggest_skill.return_value = [("rca", 1.5)]
        result = _try_auto_route_skill(state, "debug pods")
        assert result is False
        # Should NOT have set active_skill
        assert state.active_skill is None

    def test_returns_true_and_sets_skill_when_above_threshold(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state(threshold=1.0)
        fake_skill = MagicMock()
        state.skill_registry.suggest_skill.return_value = [("rca", 2.0)]
        state.skill_registry.get.return_value = fake_skill

        result = _try_auto_route_skill(state, "debug incident logs")
        assert result is True
        assert state.active_skill is fake_skill
        assert state.current_phase == SkillPhase.ANALYZE

    def test_returns_false_when_skill_not_found(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state(threshold=1.0)
        state.skill_registry.suggest_skill.return_value = [("nonexistent", 2.0)]
        state.skill_registry.get.return_value = None

        result = _try_auto_route_skill(state, "something")
        assert result is False

    def test_uses_config_threshold(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        # Threshold is 1.5, score is 1.4 — should NOT route
        state = _make_repl_state(threshold=1.5)
        state.skill_registry.suggest_skill.return_value = [("rca", 1.4)]
        result = _try_auto_route_skill(state, "debug logs")
        assert result is False

        # Now threshold is 1.0, score is 1.4 — should route
        state2 = _make_repl_state(threshold=1.0)
        fake_skill = MagicMock()
        state2.skill_registry.suggest_skill.return_value = [("rca", 1.4)]
        state2.skill_registry.get.return_value = fake_skill
        result2 = _try_auto_route_skill(state2, "debug logs")
        assert result2 is True

    def test_does_not_override_explicit_skill(self) -> None:
        """When a user explicitly set a skill, _handle_chat should skip auto-routing."""
        # This tests the _handle_chat logic path — if active_skill is set,
        # it goes to _handle_skill_chat directly without calling _try_auto_route_skill
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state()
        # Simulate explicit skill already set — _try_auto_route_skill
        # should never be called in this case (tested via _handle_chat)
        # But if called anyway, it would still work normally
        state.skill_registry.suggest_skill.return_value = [("cost-analysis", 2.0)]
        fake_skill = MagicMock()
        state.skill_registry.get.return_value = fake_skill
        result = _try_auto_route_skill(state, "cloud cost optimization")
        assert result is True

    def test_picks_highest_scoring_skill(self) -> None:
        from vaig.cli.repl import _try_auto_route_skill

        state = _make_repl_state(threshold=1.0)
        fake_skill = MagicMock()
        # suggest_skill returns sorted by score desc — first is best
        state.skill_registry.suggest_skill.return_value = [
            ("rca", 2.5),
            ("log-analysis", 2.0),
            ("config-audit", 1.0),
        ]
        state.skill_registry.get.return_value = fake_skill

        result = _try_auto_route_skill(state, "incident debugging with logs")
        assert result is True
        # Should have called get with the FIRST (highest scoring) skill
        state.skill_registry.get.assert_called_once_with("rca")


# ── Auto-routing in _handle_chat (integration) ──────────────


class TestHandleChatAutoRouting:
    """Integration tests verifying auto-routing wiring in _handle_chat."""

    def test_auto_route_activates_skill_then_clears(self) -> None:
        """Auto-routed skill should be active during chat, cleared after."""
        from vaig.cli.repl import _handle_chat

        state = _make_repl_state(threshold=1.0)
        fake_skill = MagicMock()
        fake_skill.get_metadata.return_value = SkillMetadata(
            name="rca", display_name="RCA", description="Root cause analysis",
        )
        state.skill_registry.suggest_skill.return_value = [("rca", 2.0)]
        state.skill_registry.get.return_value = fake_skill
        state.code_mode = False
        state.context_builder.bundle.file_count = 0

        with patch("vaig.cli.repl._handle_skill_chat") as mock_skill_chat, \
             patch("vaig.cli.repl._handle_direct_chat") as mock_direct_chat:
            _handle_chat(state, "incident debugging logs")

            # Should have used skill chat, NOT direct chat
            mock_skill_chat.assert_called_once()
            mock_direct_chat.assert_not_called()

        # After _handle_chat, skill should be cleared
        assert state.active_skill is None
        assert state.current_phase == SkillPhase.ANALYZE

    def test_no_auto_route_when_disabled(self) -> None:
        """When auto_routing is False, should fall through to direct chat."""
        from vaig.cli.repl import _handle_chat

        state = _make_repl_state(auto_routing=False)
        state.code_mode = False
        state.context_builder.bundle.file_count = 0

        with patch("vaig.cli.repl._handle_direct_chat") as mock_direct, \
             patch("vaig.cli.repl._handle_skill_chat") as mock_skill:
            _handle_chat(state, "incident debugging logs")

            mock_direct.assert_called_once()
            mock_skill.assert_not_called()

    def test_explicit_skill_not_overridden(self) -> None:
        """When user already activated a skill, auto-routing should NOT interfere."""
        from vaig.cli.repl import _handle_chat

        state = _make_repl_state(threshold=1.0)
        explicit_skill = MagicMock()
        explicit_skill.get_metadata.return_value = SkillMetadata(
            name="cost-analysis", display_name="Cost Analysis",
            description="Analyze costs",
        )
        state.active_skill = explicit_skill  # User set this explicitly
        state.code_mode = False
        state.context_builder.bundle.file_count = 0

        with patch("vaig.cli.repl._handle_skill_chat") as mock_skill, \
             patch("vaig.cli.repl._try_auto_route_skill") as mock_auto:
            _handle_chat(state, "some query")

            # Should use skill chat with the explicit skill
            mock_skill.assert_called_once()
            # Should NOT have called auto-route at all
            mock_auto.assert_not_called()

    def test_low_score_falls_to_direct_chat(self) -> None:
        """When auto-routing scores below threshold, should use direct chat."""
        from vaig.cli.repl import _handle_chat

        state = _make_repl_state(threshold=1.5)
        state.skill_registry.suggest_skill.return_value = [("rca", 0.5)]
        state.code_mode = False
        state.context_builder.bundle.file_count = 0

        with patch("vaig.cli.repl._handle_direct_chat") as mock_direct, \
             patch("vaig.cli.repl._handle_skill_chat") as mock_skill:
            _handle_chat(state, "what is kubernetes")

            mock_direct.assert_called_once()
            mock_skill.assert_not_called()
