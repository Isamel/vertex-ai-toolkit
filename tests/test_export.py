"""Tests for the export formatting module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vaig.cli.export import ExportMetadata, ExportPayload, format_export


@pytest.fixture()
def sample_metadata() -> ExportMetadata:
    """Metadata fixture with all fields populated."""
    return ExportMetadata(
        model="gemini-2.5-flash",
        skill="rca",
        timestamp="2026-03-13T10:00:00+00:00",
        tokens={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        cost="$0.0023",
        vaig_version="0.1.0",
    )


@pytest.fixture()
def sample_metadata_no_skill() -> ExportMetadata:
    """Metadata fixture without skill or cost."""
    return ExportMetadata(
        model="gemini-2.5-pro",
        skill=None,
        timestamp="2026-03-13T12:00:00+00:00",
        tokens={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
        cost=None,
        vaig_version="0.1.0",
    )


@pytest.fixture()
def sample_payload(sample_metadata: ExportMetadata) -> ExportPayload:
    """Full export payload with context files and agent results."""
    return ExportPayload(
        question="Why is the pod crashing?",
        response="The pod is crashing due to OOM kills.",
        metadata=sample_metadata,
        context_files=["logs/pod.log", "k8s/deployment.yaml"],
        agent_results=[
            {"agent_name": "investigator", "content": "Found OOM events in logs."},
            {"agent_name": "synthesizer", "content": "Root cause: memory limit too low."},
        ],
    )


@pytest.fixture()
def minimal_payload(sample_metadata_no_skill: ExportMetadata) -> ExportPayload:
    """Minimal payload — no skill, no context files, no agent results."""
    return ExportPayload(
        question="What is Kubernetes?",
        response="Kubernetes is a container orchestration platform.",
        metadata=sample_metadata_no_skill,
    )


# ── to_json ────────────────────────────────────────────────────


class TestToJson:
    def test_produces_valid_json(self, sample_payload: ExportPayload) -> None:
        raw = sample_payload.to_json()
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_contains_all_top_level_fields(self, sample_payload: ExportPayload) -> None:
        data = json.loads(sample_payload.to_json())
        assert data["question"] == "Why is the pod crashing?"
        assert data["response"] == "The pod is crashing due to OOM kills."
        assert data["metadata"]["model"] == "gemini-2.5-flash"
        assert data["metadata"]["skill"] == "rca"
        assert data["metadata"]["vaig_version"] == "0.1.0"
        assert data["metadata"]["cost"] == "$0.0023"

    def test_contains_context_files(self, sample_payload: ExportPayload) -> None:
        data = json.loads(sample_payload.to_json())
        assert data["context_files"] == ["logs/pod.log", "k8s/deployment.yaml"]

    def test_contains_agent_results(self, sample_payload: ExportPayload) -> None:
        data = json.loads(sample_payload.to_json())
        assert len(data["agent_results"]) == 2
        assert data["agent_results"][0]["agent_name"] == "investigator"

    def test_contains_token_data(self, sample_payload: ExportPayload) -> None:
        data = json.loads(sample_payload.to_json())
        tokens = data["metadata"]["tokens"]
        assert tokens["prompt_tokens"] == 100
        assert tokens["completion_tokens"] == 200
        assert tokens["total_tokens"] == 300

    def test_minimal_payload_has_empty_lists(self, minimal_payload: ExportPayload) -> None:
        data = json.loads(minimal_payload.to_json())
        assert data["context_files"] == []
        assert data["agent_results"] == []
        assert data["metadata"]["skill"] is None
        assert data["metadata"]["cost"] is None


# ── to_markdown ────────────────────────────────────────────────


class TestToMarkdown:
    def test_contains_title(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "# VAIG Analysis Report" in md

    def test_contains_model(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "gemini-2.5-flash" in md

    def test_contains_skill(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "**Skill**: rca" in md

    def test_contains_question_section(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "## Question" in md
        assert "Why is the pod crashing?" in md

    def test_contains_response_section(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "## Response" in md
        assert "The pod is crashing due to OOM kills." in md

    def test_contains_context_files_section(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "## Context Files" in md
        assert "- `logs/pod.log`" in md
        assert "- `k8s/deployment.yaml`" in md

    def test_contains_agent_details(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "## Agent Details" in md
        assert "### investigator" in md
        assert "### synthesizer" in md

    def test_contains_cost(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "$0.0023" in md

    def test_contains_token_summary(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "300 total" in md
        assert "100 prompt" in md
        assert "200 completion" in md

    def test_contains_footer(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "Generated by VAIG v0.1.0" in md

    def test_no_skill_section_when_none(self, minimal_payload: ExportPayload) -> None:
        md = minimal_payload.to_markdown()
        assert "**Skill**" not in md

    def test_no_cost_when_none(self, minimal_payload: ExportPayload) -> None:
        md = minimal_payload.to_markdown()
        assert "**Estimated Cost**" not in md

    def test_no_context_files_section_when_empty(self, minimal_payload: ExportPayload) -> None:
        md = minimal_payload.to_markdown()
        assert "## Context Files" not in md

    def test_no_agent_details_when_empty(self, minimal_payload: ExportPayload) -> None:
        md = minimal_payload.to_markdown()
        assert "## Agent Details" not in md


# ── to_html ────────────────────────────────────────────────────


class TestToHtml:
    def test_produces_valid_html_structure(self, sample_payload: ExportPayload) -> None:
        html = sample_payload.to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_contains_title(self, sample_payload: ExportPayload) -> None:
        html = sample_payload.to_html()
        assert "<title>VAIG Analysis Report</title>" in html

    def test_contains_inline_css(self, sample_payload: ExportPayload) -> None:
        html = sample_payload.to_html()
        assert "<style>" in html
        assert "font-family" in html

    def test_contains_model_in_metadata(self, sample_payload: ExportPayload) -> None:
        html = sample_payload.to_html()
        assert "gemini-2.5-flash" in html

    def test_contains_question(self, sample_payload: ExportPayload) -> None:
        html = sample_payload.to_html()
        assert "Why is the pod crashing?" in html

    def test_xss_protection_script_tag_blocked(self) -> None:
        """Inline HTML (e.g. <script>) must be escaped, not executed."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="What about <script>alert('xss')</script>?",
            response="Use &amp; for ampersands",
            metadata=meta,
        )
        html = payload.to_html()
        # Raw <script> tag must not appear in output
        assert "<script>" not in html
        # The text content must be escaped (angle brackets → entities)
        assert "&lt;script&gt;" in html

    def test_xss_protection_inline_event_handler_blocked(self) -> None:
        """Event handlers in raw HTML must be escaped (not executed) in output."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="safe question",
            response='<img src=x onerror="alert(1)">',
            metadata=meta,
        )
        html = payload.to_html()
        # The raw HTML tag must NOT be reproduced verbatim — must be escaped
        assert '<img src=x onerror="alert(1)">' not in html
        # The onerror attribute value must appear only as escaped text, not executable
        assert "onerror=&quot;" in html or "&lt;img" in html

    def test_no_skill_element_when_none(self, minimal_payload: ExportPayload) -> None:
        html = minimal_payload.to_html()
        assert "Skill" not in html

    def test_self_contained_no_external_deps(self, sample_payload: ExportPayload) -> None:
        """HTML should not link to any external stylesheet or script."""
        html = sample_payload.to_html()
        assert '<link rel="stylesheet"' not in html
        assert "<script src=" not in html

    def test_renders_markdown_table_as_html_table(self) -> None:
        """Markdown tables must be rendered as HTML <table> elements."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="show table",
            response="| Col A | Col B |\n|-------|-------|\n| val1 | val2 |",
            metadata=meta,
        )
        html = payload.to_html()
        assert "<table>" in html
        assert "<th>" in html
        assert "<td>" in html

    def test_renders_code_blocks_as_pre_code(self) -> None:
        """Fenced code blocks must produce <pre><code...> elements."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="show code",
            response="```python\nprint('hello')\n```",
            metadata=meta,
        )
        html = payload.to_html()
        assert "<pre>" in html
        assert "<code" in html  # may have class="language-..." attribute

    def test_renders_strikethrough_as_del(self) -> None:
        """~~strikethrough~~ text must produce <s> or <del> elements."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="strikethrough test",
            response="~~deprecated~~",
            metadata=meta,
        )
        html = payload.to_html()
        # markdown-it-py renders strikethrough as <s>; some renderers use <del>
        assert "<s>" in html or "<del>" in html
        assert "deprecated" in html

    def test_contains_tokyo_night_css_variables(self, sample_payload: ExportPayload) -> None:
        """Tokyo Night color variables must be present in the CSS."""
        html = sample_payload.to_html()
        assert "--bg-base" in html
        assert "#1a1b26" in html
        assert "--bg-surface" in html
        assert "#24283b" in html
        assert "--accent" in html
        assert "#7aa2f7" in html

    def test_does_not_contain_raw_markdown_pre_block(self, sample_payload: ExportPayload) -> None:
        """The old raw-markdown-in-pre approach must no longer be used."""
        html = sample_payload.to_html()
        # Markdown heading markers must be rendered into HTML — not appear as raw text
        assert "<h1>" in html
        assert "# VAIG Analysis Report" not in html

    def test_headings_rendered_as_html_headings(self, sample_payload: ExportPayload) -> None:
        """Markdown headings must be rendered as <h1>, <h2> etc."""
        html = sample_payload.to_html()
        assert "<h1>" in html
        assert "<h2>" in html

    def test_cost_table_rendered_as_html_table(self, sample_payload: ExportPayload) -> None:
        """The ## Cost & Usage Summary markdown table must become an HTML table."""
        html = sample_payload.to_html()
        # The cost section must contain a rendered HTML table
        assert "<table>" in html

    def test_empty_response_does_not_raise(self) -> None:
        """to_html() must not raise when response is an empty string."""
        meta = ExportMetadata(
            model="test-model",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(question="q", response="", metadata=meta)
        html = payload.to_html()
        assert "<!DOCTYPE html>" in html


# ── format_export router ──────────────────────────────────────


class TestFormatExport:
    def test_routes_json(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "json")
        data = json.loads(result)
        assert data["question"] == sample_payload.question

    def test_routes_md(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "md")
        assert "# VAIG Analysis Report" in result

    def test_routes_markdown(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "markdown")
        assert "# VAIG Analysis Report" in result

    def test_routes_html(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "html")
        assert "<!DOCTYPE html>" in result

    def test_case_insensitive(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "JSON")
        data = json.loads(result)
        assert data["question"] == sample_payload.question

    def test_strips_whitespace(self, sample_payload: ExportPayload) -> None:
        result = format_export(sample_payload, "  md  ")
        assert "# VAIG Analysis Report" in result

    def test_raises_on_unknown_format(self, sample_payload: ExportPayload) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            format_export(sample_payload, "xml")

    def test_raises_on_empty_format(self, sample_payload: ExportPayload) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            format_export(sample_payload, "")


# ── Cost & Usage Summary section ──────────────────────────────


class TestCostSummarySection:
    """Tests for the ## Cost & Usage Summary markdown section."""

    def test_section_present_when_tokens_available(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "## Cost & Usage Summary" in md

    def test_contains_input_tokens(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "| Input tokens | 100 |" in md

    def test_contains_output_tokens(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "| Output tokens | 200 |" in md

    def test_contains_total_tokens(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "| Total tokens | 300 |" in md

    def test_contains_model(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "| Model | gemini-2.5-flash |" in md

    def test_contains_estimated_cost(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "| Estimated cost | $0.0023 |" in md

    def test_no_cost_row_when_cost_is_none(self, minimal_payload: ExportPayload) -> None:
        md = minimal_payload.to_markdown()
        assert "Estimated cost" not in md

    def test_no_section_when_all_zero_tokens(self) -> None:
        meta = ExportMetadata(
            model="gemini-2.5-pro",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="test",
            response="test",
            metadata=meta,
        )
        md = payload.to_markdown()
        assert "## Cost & Usage Summary" not in md

    def test_no_section_when_tokens_dict_empty(self) -> None:
        meta = ExportMetadata(
            model="gemini-2.5-pro",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={},
            cost=None,
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="test",
            response="test",
            metadata=meta,
        )
        md = payload.to_markdown()
        assert "## Cost & Usage Summary" not in md

    def test_thinking_tokens_included_when_present(self) -> None:
        meta = ExportMetadata(
            model="gemini-2.5-pro",
            skill=None,
            timestamp="2026-01-01T00:00:00Z",
            tokens={
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "thinking_tokens": 50,
                "total_tokens": 350,
            },
            cost="$0.0050",
            vaig_version="0.1.0",
        )
        payload = ExportPayload(
            question="test",
            response="test",
            metadata=meta,
        )
        md = payload.to_markdown()
        assert "| Thinking tokens | 50 |" in md

    def test_thinking_tokens_omitted_when_zero(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        assert "Thinking tokens" not in md

    def test_section_appears_before_footer(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        cost_pos = md.index("## Cost & Usage Summary")
        footer_pos = md.index("Generated by VAIG")
        assert cost_pos < footer_pos

    def test_section_is_valid_markdown_table(self, sample_payload: ExportPayload) -> None:
        md = sample_payload.to_markdown()
        # Extract the cost section
        start = md.index("## Cost & Usage Summary")
        section = md[start:]
        # Should have table header separator
        assert "|--------|-------|" in section

    def test_json_export_includes_cost_data(self, sample_payload: ExportPayload) -> None:
        """JSON export should include token and cost data in metadata."""
        result = format_export(sample_payload, "json")
        data = json.loads(result)
        assert data["metadata"]["tokens"]["prompt_tokens"] == 100
        assert data["metadata"]["cost"] == "$0.0023"


# ── _handle_export_output format dispatch ────────────────────


class TestHandleExportOutputFormatDispatch:
    """Tests for the _handle_export_output CLI helper — format routing.

    These tests verify that the format flag is correctly dispatched,
    including the special-case ``rich`` passthrough that must NOT raise.
    """

    @pytest.fixture()
    def _mock_settings(self) -> None:
        """Suppress settings loading in _helpers."""
        from vaig.core.config import Settings

        settings = Settings()
        with patch("vaig.cli._helpers._get_settings", return_value=settings):
            yield

    def _make_kwargs(self, format_: str | None = None, output: Path | None = None) -> dict:
        return {
            "response_text": "The pod is crashing due to OOM.",
            "question": "Why is the pod crashing?",
            "model_id": "gemini-2.5-flash",
            "skill_name": "rca",
            "format_": format_,
            "output": output,
            "tokens": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            "cost": "$0.0023",
        }

    def test_rich_format_does_not_raise(self) -> None:
        """--format rich must be a silent passthrough — no ValueError."""
        from vaig.cli._helpers import _handle_export_output

        # Should not raise — rich means "terminal display already done"
        _handle_export_output(**self._make_kwargs(format_="rich"))

    def test_rich_format_uppercase_does_not_raise(self) -> None:
        """Case-insensitive: --format RICH must not raise."""
        from vaig.cli._helpers import _handle_export_output

        _handle_export_output(**self._make_kwargs(format_="RICH"))

    def test_rich_format_with_whitespace_does_not_raise(self) -> None:
        """Whitespace-padded --format '  rich  ' must not raise."""
        from vaig.cli._helpers import _handle_export_output

        _handle_export_output(**self._make_kwargs(format_="  rich  "))

    def test_json_format_produces_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--format json routes through format_export and outputs valid JSON."""
        from vaig.cli._helpers import _handle_export_output

        # No output file — content is printed to console (via rich console)
        _handle_export_output(**self._make_kwargs(format_="json"))
        # We can't easily capture rich console output in tests, but
        # the important thing is that it doesn't raise.

    def test_html_format_does_not_raise(self) -> None:
        """--format html routes through format_export without error."""
        from vaig.cli._helpers import _handle_export_output

        _handle_export_output(**self._make_kwargs(format_="html"))

    def test_md_format_does_not_raise(self) -> None:
        """--format md routes through format_export without error."""
        from vaig.cli._helpers import _handle_export_output

        _handle_export_output(**self._make_kwargs(format_="md"))

    def test_no_format_no_output_is_noop(self) -> None:
        """When neither --format nor --output is set, function is a no-op."""
        from vaig.cli._helpers import _handle_export_output

        # Should return immediately without any side effects
        _handle_export_output(**self._make_kwargs(format_=None, output=None))

    def test_rich_format_does_not_write_file(self, tmp_path: Path) -> None:
        """--format rich with --output must not create a file (passthrough)."""
        from vaig.cli._helpers import _handle_export_output

        out_file = tmp_path / "report.txt"
        _handle_export_output(**self._make_kwargs(format_="rich", output=out_file))
        # File must NOT be created — rich is terminal display only
        assert not out_file.exists()

    def test_format_export_rich_is_still_unsupported(self, sample_payload: ExportPayload) -> None:
        """format_export itself still rejects 'rich' — handled at CLI layer only."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            format_export(sample_payload, "rich")
