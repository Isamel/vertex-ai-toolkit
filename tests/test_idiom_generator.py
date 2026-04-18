"""Tests for IdiomGenerator and the 3-tier _load_idiom_map fallback (CM-07)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_mock_client(response_text: str) -> MagicMock:
    """Build a minimal GeminiClient mock that returns ``response_text``."""
    client = MagicMock()
    result = MagicMock()
    result.text = response_text
    client.generate.return_value = result
    return client


_SAMPLE_YAML = """\
source_lang: rust
target_lang: go
idioms:
  - source_pattern: "ownership"
    target_pattern: "garbage collection"
    description: "Rust ownership vs Go GC"
    example_before: |
      let s = String::from("hello");
    example_after: |
      s := "hello"
  - source_pattern: "match expression"
    target_pattern: "switch statement"
    description: "Pattern matching"
    example_before: |
      match x { 1 => foo(), _ => bar() }
    example_after: |
      switch x { case 1: foo() }
dependencies:
  serde: "encoding/json"
"""


# ── IdiomGenerator tests ─────────────────────────────────────────────────────

class TestIdiomGeneratorCacheReadWrite:
    def test_generate_calls_llm_on_cache_miss(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        result = gen.generate("rust", "go")

        client.generate.assert_called_once()
        assert "source_lang: rust" in result

    def test_generate_writes_cache_file(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)
        gen.generate("rust", "go")

        cache_file = tmp_path / "rust_to_go.yaml"
        assert cache_file.exists()
        assert "source_lang: rust" in cache_file.read_text()

    def test_generate_reads_cache_on_second_call(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        gen.generate("rust", "go")
        gen.generate("rust", "go")  # second call — should use cache

        # LLM should only be called once
        assert client.generate.call_count == 1

    def test_generate_cache_miss_then_hit(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        first = gen.generate("rust", "go")
        second = gen.generate("rust", "go")

        assert first == second

    def test_is_cached_false_before_generate(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        assert not gen.is_cached("rust", "go")

    def test_is_cached_true_after_generate(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client(_SAMPLE_YAML)
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        gen.generate("rust", "go")

        assert gen.is_cached("rust", "go")

    def test_cache_path_returns_expected_file(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        path = gen.cache_path("rust", "go")

        assert path == tmp_path / "rust_to_go.yaml"

    def test_generate_raises_runtime_error_on_llm_failure(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = MagicMock()
        client.generate.side_effect = RuntimeError("API down")
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        with pytest.raises(RuntimeError, match="LLM call failed"):
            gen.generate("rust", "go")

    def test_parse_yaml_returns_dict(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        result = gen.parse_yaml(_SAMPLE_YAML)

        assert result is not None
        assert isinstance(result, dict)
        assert result["source_lang"] == "rust"
        assert len(result["idioms"]) == 2

    def test_parse_yaml_returns_none_on_invalid(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        result = gen.parse_yaml("not: valid: yaml: [broken")

        assert result is None

    def test_preexisting_cache_file_is_loaded(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        cache_file = tmp_path / "rust_to_go.yaml"
        cache_file.write_text(_SAMPLE_YAML, encoding="utf-8")

        client = _make_mock_client("should not be called")
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        result = gen.generate("rust", "go")

        client.generate.assert_not_called()
        assert "source_lang: rust" in result


# ── 3-tier fallback tests ─────────────────────────────────────────────────────

class TestLoadIdiomMapFallback:
    """Test _load_idiom_map 3-tier fallback: bundled → cache → LLM."""

    def test_tier1_bundled_map_found(self) -> None:
        """Tier 1: bundled map is returned when it exists."""
        from vaig.skills.code_migration.skill import _load_idiom_map

        result = _load_idiom_map("python", "go")

        assert result is not None
        assert "idioms" in result

    def test_tier1_returns_none_for_unknown_pair(self) -> None:
        """Tier 1 miss with no config → None (no further tiers)."""
        from vaig.skills.code_migration.skill import _load_idiom_map

        result = _load_idiom_map("rust", "fortran")

        assert result is None

    def test_tier2_user_cache_loaded(self, tmp_path: Path) -> None:
        """Tier 2: user cache file is loaded when bundled map is absent."""
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        cache_file = tmp_path / "rust_to_go.yaml"
        cache_file.write_text(_SAMPLE_YAML, encoding="utf-8")

        config = IdiomConfig(enabled=True, auto_generate=False, cache_dir=str(tmp_path))
        result = _load_idiom_map("rust", "go", idiom_config=config)

        assert result is not None
        assert result["source_lang"] == "rust"

    def test_tier2_skipped_when_no_idiom_config(self) -> None:
        """Without idiom_config, tiers 2 and 3 are skipped."""
        from vaig.skills.code_migration.skill import _load_idiom_map

        # "ruby" → "cobol" has no bundled map; without config → None
        result = _load_idiom_map("ruby", "cobol", idiom_config=None)

        assert result is None

    def test_tier3_llm_called_when_auto_generate(self, tmp_path: Path) -> None:
        """Tier 3: LLM is invoked when auto_generate=True and no map exists."""
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        client = _make_mock_client(_SAMPLE_YAML)
        config = IdiomConfig(enabled=True, auto_generate=True, cache_dir=str(tmp_path))

        result = _load_idiom_map("rust", "go", idiom_config=config, client=client)

        client.generate.assert_called_once()
        assert result is not None

    def test_tier3_skipped_when_auto_generate_false(self, tmp_path: Path) -> None:
        """Tier 3: LLM is NOT invoked when auto_generate=False."""
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        client = _make_mock_client(_SAMPLE_YAML)
        config = IdiomConfig(enabled=True, auto_generate=False, cache_dir=str(tmp_path))

        result = _load_idiom_map("ruby", "cobol", idiom_config=config, client=client)

        client.generate.assert_not_called()
        assert result is None

    def test_tier3_skipped_when_client_is_none(self, tmp_path: Path) -> None:
        """Tier 3: LLM is NOT invoked when client is None."""
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        config = IdiomConfig(enabled=True, auto_generate=True, cache_dir=str(tmp_path))

        result = _load_idiom_map("ruby", "cobol", idiom_config=config, client=None)

        assert result is None

    def test_tier3_graceful_degradation_on_llm_error(self, tmp_path: Path) -> None:
        """Tier 3: returns None gracefully when LLM generation fails."""
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        client = MagicMock()
        client.generate.side_effect = RuntimeError("API down")
        config = IdiomConfig(enabled=True, auto_generate=True, cache_dir=str(tmp_path))

        result = _load_idiom_map("ruby", "cobol", idiom_config=config, client=client)

        assert result is None


# ── Bundled YAML map validity tests ──────────────────────────────────────────

class TestBundledYamlMaps:
    """Verify that all 7 bundled YAML maps can be loaded and are structurally valid."""

    @pytest.mark.parametrize(
        "source_lang,target_lang",
        [
            ("python", "go"),
            ("pentaho", "glue"),
            ("java", "kotlin"),
            ("python2", "python3"),
            ("javascript", "typescript"),
            ("angular", "react"),
            ("express", "fastapi"),
        ],
    )
    def test_bundled_map_loads_successfully(self, source_lang: str, target_lang: str) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        result = _load_idiom_map(source_lang, target_lang)

        assert result is not None, f"Expected bundled map for {source_lang}→{target_lang} to load"
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "source_lang,target_lang",
        [
            ("python", "go"),
            ("pentaho", "glue"),
            ("java", "kotlin"),
            ("python2", "python3"),
            ("javascript", "typescript"),
            ("angular", "react"),
            ("express", "fastapi"),
        ],
    )
    def test_bundled_map_has_idioms_list(self, source_lang: str, target_lang: str) -> None:
        from vaig.skills.code_migration.skill import _load_idiom_map

        result = _load_idiom_map(source_lang, target_lang)

        assert result is not None
        assert "idioms" in result
        assert isinstance(result["idioms"], list)
        assert len(result["idioms"]) >= 8, (
            f"Expected ≥8 idioms in {source_lang}→{target_lang}, got {len(result['idioms'])}"
        )

    @pytest.mark.parametrize(
        "source_lang,target_lang",
        [
            ("java", "kotlin"),
            ("python2", "python3"),
            ("javascript", "typescript"),
            ("angular", "react"),
            ("express", "fastapi"),
        ],
    )
    def test_new_bundled_maps_have_required_fields(self, source_lang: str, target_lang: str) -> None:
        """New CM-07 maps must have source_pattern, target_pattern, description, example_before, example_after."""
        from vaig.skills.code_migration.skill import _load_idiom_map

        result = _load_idiom_map(source_lang, target_lang)
        assert result is not None

        for i, idiom in enumerate(result["idioms"]):
            assert "source_pattern" in idiom, f"idiom[{i}] missing source_pattern"
            assert "target_pattern" in idiom, f"idiom[{i}] missing target_pattern"
            assert "description" in idiom, f"idiom[{i}] missing description"
            assert "example_before" in idiom, f"idiom[{i}] missing example_before"
            assert "example_after" in idiom, f"idiom[{i}] missing example_after"


# ── IdiomConfig.enabled gate tests ───────────────────────────────────────────

class TestIdiomConfigEnabledGate:
    """When idiom_config.enabled=False, tiers 2 and 3 must be skipped."""

    def test_disabled_config_skips_tier2(self, tmp_path: Path) -> None:
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        cache_file = tmp_path / "rust_to_go.yaml"
        cache_file.write_text(_SAMPLE_YAML, encoding="utf-8")

        config = IdiomConfig(enabled=False, auto_generate=False, cache_dir=str(tmp_path))
        result = _load_idiom_map("rust", "go", idiom_config=config)

        assert result is None

    def test_disabled_config_skips_tier3(self, tmp_path: Path) -> None:
        from vaig.core.config import IdiomConfig
        from vaig.skills.code_migration.skill import _load_idiom_map

        client = _make_mock_client(_SAMPLE_YAML)
        config = IdiomConfig(enabled=False, auto_generate=True, cache_dir=str(tmp_path))
        result = _load_idiom_map("rust", "go", idiom_config=config, client=client)

        client.generate.assert_not_called()
        assert result is None


# ── Cache validation tests ────────────────────────────────────────────────────

class TestCacheValidation:
    """LLM output must be validated before writing to cache."""

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)

        with pytest.raises(ValueError, match="not valid YAML"):
            gen._save_to_cache("rust", "go", "not: valid: yaml: [broken")

    def test_non_dict_yaml_raises_value_error(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)

        with pytest.raises(ValueError, match="unexpected top-level type"):
            gen._save_to_cache("rust", "go", "- item1\n- item2\n")

    def test_invalid_yaml_not_written_to_disk(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)

        with pytest.raises(ValueError):
            gen._save_to_cache("rust", "go", "not: valid: yaml: [broken")

        assert not (tmp_path / "rust_to_go.yaml").exists()

    def test_generate_raises_on_invalid_llm_output(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        client = _make_mock_client("not: valid: yaml: [broken")
        gen = IdiomGenerator(client, cache_dir=tmp_path)

        with pytest.raises(ValueError, match="not valid YAML"):
            gen.generate("rust", "go")


# ── Path sanitization tests ───────────────────────────────────────────────────

class TestCachePathSanitization:
    """Language tokens must be sanitised to prevent path traversal."""

    def test_path_traversal_sanitized(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        path = gen.cache_path("../../../etc/passwd", "go")

        # Traversal sequences must be gone; path must stay within cache_dir
        assert ".." not in path.parts
        assert path.parent == tmp_path

    def test_special_chars_sanitized(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        path = gen.cache_path("rust!@#$%", "go")

        # Filename should contain only safe characters
        assert "!" not in path.name
        assert "@" not in path.name

    def test_normal_languages_unchanged(self, tmp_path: Path) -> None:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        gen = IdiomGenerator(_make_mock_client(""), cache_dir=tmp_path)
        path = gen.cache_path("python3", "go")

        assert path == tmp_path / "python3_to_go.yaml"


# ── _format_idiom_map extended fields test ────────────────────────────────────

class TestFormatIdiomMapExtendedFields:
    """_format_idiom_map must render description, example_before, example_after."""

    def test_format_renders_description(self) -> None:
        from vaig.skills.code_migration.skill import _format_idiom_map

        data = {
            "source_lang": "rust",
            "target_lang": "go",
            "idioms": [
                {
                    "source_pattern": "ownership",
                    "target_pattern": "garbage collection",
                    "description": "Rust ownership vs Go GC",
                    "example_before": "let s = String::from(\"hello\");",
                    "example_after": 's := "hello"',
                }
            ],
        }
        rendered = _format_idiom_map(data)

        assert "Rust ownership vs Go GC" in rendered

    def test_format_renders_example_before_and_after(self) -> None:
        from vaig.skills.code_migration.skill import _format_idiom_map

        data = {
            "source_lang": "rust",
            "target_lang": "go",
            "idioms": [
                {
                    "source_pattern": "ownership",
                    "target_pattern": "garbage collection",
                    "description": "Ownership vs GC",
                    "example_before": "let s = String::from(\"hello\");",
                    "example_after": 's := "hello"',
                }
            ],
        }
        rendered = _format_idiom_map(data)

        assert "Before:" in rendered
        assert "After:" in rendered
        assert 'let s = String::from("hello");' in rendered
        assert 's := "hello"' in rendered

    def test_format_skips_extended_section_when_fields_absent(self) -> None:
        from vaig.skills.code_migration.skill import _format_idiom_map

        data = {
            "source_lang": "python",
            "target_lang": "go",
            "idioms": [
                {
                    "source_pattern": "list comprehension",
                    "target_pattern": "for loop",
                    "notes": "Go has no list comprehension",
                }
            ],
        }
        rendered = _format_idiom_map(data)

        # Extended section should NOT appear — no description/example fields
        assert "Idiom Details" not in rendered
        # But the table row must still appear
        assert "list comprehension" in rendered
