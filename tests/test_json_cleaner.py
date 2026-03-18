"""Tests for vaig.utils.json_cleaner.clean_llm_json."""

from __future__ import annotations

from vaig.utils.json_cleaner import clean_llm_json


class TestCleanLlmJson:
    """Unit tests for the clean_llm_json helper."""

    # ── Passthrough ────────────────────────────────────────

    def test_valid_json_passthrough(self) -> None:
        """Plain JSON with no artefacts is returned unchanged."""
        raw = '{"key": "value", "num": 42}'
        result = clean_llm_json(raw)
        assert result == raw

    def test_empty_string_passthrough(self) -> None:
        """Empty input is returned unchanged."""
        assert clean_llm_json("") == ""

    def test_whitespace_only_passthrough(self) -> None:
        """Whitespace-only input is returned unchanged."""
        assert clean_llm_json("   ") == "   "

    # ── Markdown fences ─────────────────────────────────────

    def test_strips_json_code_fence(self) -> None:
        """Markdown ```json...``` fences are removed."""
        raw = '```json\n{"key": "value"}\n```'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_plain_code_fence(self) -> None:
        """Markdown ```...``` fences without language tag are removed."""
        raw = '```\n{"key": "value"}\n```'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_fence_with_surrounding_text(self) -> None:
        """Fences with surrounding text — content inside fence is used."""
        raw = 'Here is the output:\n```json\n{"status": "ok"}\n```\nEnd of output.'
        result = clean_llm_json(raw)
        assert result == '{"status": "ok"}'

    # ── Conversational preamble / postamble ─────────────────

    def test_strips_text_prefix(self) -> None:
        """Conversational preamble before the first '{' is discarded."""
        raw = 'Sure! Here is your report: {"key": "value"}'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_text_suffix(self) -> None:
        """Trailing text after the last '}' is discarded."""
        raw = '{"key": "value"} Hope this helps!'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_both_prefix_and_suffix(self) -> None:
        """Both leading and trailing garbage are stripped."""
        raw = 'Certainly:\n{"key": "value"}\n\nLet me know if you need more.'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_no_brace_returns_original(self) -> None:
        """If there is no '{' at all, the original raw value is returned."""
        raw = "This is plain text with no JSON object."
        result = clean_llm_json(raw)
        assert result == raw

    # ── Truncated JSON repair ────────────────────────────────

    def test_truncated_object_gets_closed(self) -> None:
        """A truncated object missing its closing '}' is repaired."""
        raw = '{"key": "value"'
        result = clean_llm_json(raw)
        assert result == '{"key": "value"}'

    def test_truncated_nested_object_gets_closed(self) -> None:
        """Nested unclosed objects are all repaired (LIFO order)."""
        raw = '{"outer": {"inner": "val"'
        result = clean_llm_json(raw)
        assert result == '{"outer": {"inner": "val"}}'

    def test_truncated_array_gets_closed(self) -> None:
        """An unclosed array is repaired."""
        raw = '{"items": [1, 2, 3'
        result = clean_llm_json(raw)
        assert result == '{"items": [1, 2, 3]}'

    def test_truncated_array_and_object_get_closed(self) -> None:
        """Both unclosed array and object are repaired in correct LIFO order."""
        raw = '{"items": [{"a": 1'
        result = clean_llm_json(raw)
        assert result == '{"items": [{"a": 1}]}'

    def test_complete_json_not_modified_by_repair(self) -> None:
        """A complete JSON object is not altered by the repair step."""
        raw = '{"key": "value", "arr": [1, 2, 3]}'
        result = clean_llm_json(raw)
        assert result == raw

    # ── Strings with braces inside them ─────────────────────

    def test_brace_inside_string_not_counted(self) -> None:
        """Braces inside JSON string values are not treated as structure."""
        raw = '{"msg": "hello {world}"}'
        result = clean_llm_json(raw)
        assert result == raw

    def test_escaped_quote_inside_string(self) -> None:
        """Escaped quotes inside strings are handled correctly."""
        raw = r'{"msg": "he said \"hi\""}'
        result = clean_llm_json(raw)
        assert result == raw
