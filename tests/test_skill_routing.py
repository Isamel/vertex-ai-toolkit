"""Tests for conditional skill routing — suggest_skill, _tokenize_query, _score_skill."""

from __future__ import annotations

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
