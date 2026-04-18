"""Tests for the HypothesisLibrary (SPEC-X-03).

Covers:
- Built-in OOM template matches symptom
- User YAML overrides built-in
- Missing YAML file is fail-safe
- Regex pattern matching
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from vaig.core.config import HypothesisConfig
from vaig.core.hypothesis_library import HypothesisLibrary, HypothesisTemplate


class TestHypothesisLibrary:
    __test__ = True  # explicit — class starts with "Test" so make sure pytest picks it up

    def test_default_has_at_least_ten_templates(self) -> None:
        library = HypothesisLibrary.default()
        assert len(library) >= 10

    def test_builtin_oom_matches_oomkilled(self) -> None:
        """Scenario: Built-in OOM template matches symptom."""
        library = HypothesisLibrary.default()
        matches = library.match(["OOMKilled"])
        ids = [t.id for t in matches]
        assert any("oom" in tid for tid in ids), f"No OOM template matched. Got: {ids}"

    def test_builtin_oom_matches_out_of_memory(self) -> None:
        library = HypothesisLibrary.default()
        matches = library.match(["out of memory"])
        assert any("oom" in t.id for t in matches)

    def test_user_yaml_overrides_builtin(self, tmp_path: Path) -> None:
        """Scenario: User YAML overrides built-in."""
        yaml_content = dedent("""\
            - id: "oom"
              symptom_pattern: "OOMKilled"
              hypothesis_text: "Custom OOM hypothesis from user YAML"
              investigation_strategy: "kubectl_describe"
              confidence_modifier: 0.5
        """)
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text(yaml_content)

        config = HypothesisConfig(custom_templates_path=yaml_file)
        library = HypothesisLibrary(config)
        matches = library.match(["OOMKilled"])
        oom_matches = [t for t in matches if t.id == "oom"]
        assert oom_matches, "Expected OOM template in matches"
        assert oom_matches[0].hypothesis_text == "Custom OOM hypothesis from user YAML"

    def test_missing_yaml_file_is_failsafe(self, tmp_path: Path) -> None:
        """Scenario: Missing YAML file is fail-safe."""
        config = HypothesisConfig(custom_templates_path=tmp_path / "nonexistent.yaml")
        # Should NOT raise
        library = HypothesisLibrary(config)
        # Built-ins should still be present
        assert len(library) >= 10
        matches = library.match(["OOMKilled"])
        assert any("oom" in t.id for t in matches)

    def test_malformed_yaml_is_failsafe(self, tmp_path: Path) -> None:
        """Malformed YAML file falls back to built-ins gracefully."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("{{invalid: yaml: [}")
        config = HypothesisConfig(custom_templates_path=yaml_file)
        library = HypothesisLibrary(config)
        assert len(library) >= 10

    def test_regex_pattern_matching(self) -> None:
        """Scenario: Regex pattern matching."""
        library = HypothesisLibrary.default()
        # pod_restarts template has regex: pod.*restart
        matches = library.match(["pod CrashLoopBackOff restart count 5"])
        ids = [t.id for t in matches]
        assert "pod_restarts" in ids, f"Expected pod_restarts in matches, got: {ids}"

    def test_case_insensitive_match(self) -> None:
        library = HypothesisLibrary.default()
        matches_lower = library.match(["oomkilled"])
        matches_upper = library.match(["OOMKILLED"])
        assert len(matches_lower) > 0
        assert len(matches_upper) > 0

    def test_match_returns_unique_templates(self) -> None:
        """Each template appears at most once even if multiple symptoms match."""
        library = HypothesisLibrary.default()
        # Two symptoms that could both match oom
        matches = library.match(["OOMKilled", "out of memory"])
        oom_matches = [t for t in matches if t.id == "oom"]
        assert len(oom_matches) == 1, "OOM template should appear exactly once"

    def test_match_empty_symptoms(self) -> None:
        library = HypothesisLibrary.default()
        assert library.match([]) == []

    def test_register_override(self) -> None:
        library = HypothesisLibrary.default()
        custom = HypothesisTemplate(
            id="oom",
            symptom_pattern="OOMKilled",
            hypothesis_text="Overridden via register()",
            investigation_strategy="kubectl_logs",
            confidence_modifier=0.9,
        )
        library.register(custom)
        matches = library.match(["OOMKilled"])
        oom = next((t for t in matches if t.id == "oom"), None)
        assert oom is not None
        assert oom.hypothesis_text == "Overridden via register()"

    def test_register_new_template(self) -> None:
        library = HypothesisLibrary.default()
        initial_count = len(library)
        new_tmpl = HypothesisTemplate(
            id="custom_new",
            symptom_pattern="my_custom_symptom",
            hypothesis_text="Custom hypothesis",
            investigation_strategy="kubectl_get",
            confidence_modifier=0.0,
        )
        library.register(new_tmpl)
        assert len(library) == initial_count + 1
        matches = library.match(["my_custom_symptom"])
        assert any(t.id == "custom_new" for t in matches)

    def test_hypothesis_template_model_fields(self) -> None:
        tmpl = HypothesisTemplate(
            id="test_tmpl",
            symptom_pattern="crash",
            hypothesis_text="App crashed",
            investigation_strategy="kubectl_logs",
            confidence_modifier=0.5,
        )
        assert tmpl.id == "test_tmpl"
        assert tmpl.symptom_pattern == "crash"
        assert tmpl.confidence_modifier == 0.5

    def test_hypothesis_config_auto_enable(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text("[]")
        config = HypothesisConfig(custom_templates_path=yaml_file)
        assert config.enabled is True

    def test_hypothesis_config_default_disabled(self) -> None:
        config = HypothesisConfig()
        assert config.enabled is False
        assert config.custom_templates_path is None

    def test_yaml_with_non_list_root_is_failsafe(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad_root.yaml"
        yaml_file.write_text("key: value\n")
        config = HypothesisConfig(custom_templates_path=yaml_file)
        library = HypothesisLibrary(config)
        assert len(library) >= 10
