"""Tests for the fix-forward loop helpers in CodingSkillOrchestrator."""

from __future__ import annotations

from vaig.agents.coding_pipeline import CodingSkillOrchestrator

# ── _build_feedback_context ───────────────────────────────────


def test_build_feedback_context_contains_iteration() -> None:
    xml = CodingSkillOrchestrator._build_feedback_context(iteration=2, issues=["foo", "bar"])
    assert "<iteration>2</iteration>" in xml


def test_build_feedback_context_contains_issues() -> None:
    xml = CodingSkillOrchestrator._build_feedback_context(
        iteration=1, issues=["missing import", "wrong type"]
    )
    assert "missing import" in xml
    assert "wrong type" in xml


def test_build_feedback_context_caps_issues_at_5() -> None:
    issues = [f"issue {i}" for i in range(10)]
    xml = CodingSkillOrchestrator._build_feedback_context(iteration=1, issues=issues)
    # Only the first 5 issues should appear
    for i in range(5):
        assert f"issue {i}" in xml
    for i in range(5, 10):
        assert f"issue {i}" not in xml


def test_build_feedback_context_empty_issues() -> None:
    xml = CodingSkillOrchestrator._build_feedback_context(iteration=1, issues=[])
    assert "<fix_forward_feedback>" in xml
    assert "<iteration>1</iteration>" in xml


def test_build_feedback_context_xml_structure() -> None:
    xml = CodingSkillOrchestrator._build_feedback_context(
        iteration=3, issues=["a problem"]
    )
    assert xml.startswith("<fix_forward_feedback>")
    assert xml.rstrip().endswith("</fix_forward_feedback>")
    assert "<failed_checks>" in xml
    assert "</failed_checks>" in xml
    assert "<instruction>" in xml


# ── _extract_issues ───────────────────────────────────────────


def test_extract_issues_from_json_report() -> None:
    import json

    report_json = json.dumps({
        "success": False,
        "issues": ["Missing docstring on foo()", "Import error in bar.py"],
        "details": {},
    })
    issues = CodingSkillOrchestrator._extract_issues(report_json)
    assert "Missing docstring on foo()" in issues
    assert "Import error in bar.py" in issues


def test_extract_issues_from_json_fenced_markdown() -> None:
    import json

    report_json = json.dumps({
        "success": False,
        "issues": ["SyntaxError in main.py"],
        "details": {},
    })
    fenced = f"```json\n{report_json}\n```"
    issues = CodingSkillOrchestrator._extract_issues(fenced)
    assert "SyntaxError in main.py" in issues


def test_extract_issues_heuristic_from_plain_text() -> None:
    text = (
        "Overall: FAIL ❌\n"
        "main.py: placeholder found — FAIL\n"
        "tests/test_foo.py: PASS ✅\n"
        "ISSUE: missing return type annotation\n"
    )
    issues = CodingSkillOrchestrator._extract_issues(text)
    assert len(issues) > 0
    # Lines with FAIL or issue should be captured
    assert any("FAIL" in i or "issue" in i.lower() for i in issues)


def test_extract_issues_returns_empty_for_clean_pass() -> None:
    text = "Overall: PASS ✅\nAll files verified."
    issues = CodingSkillOrchestrator._extract_issues(text)
    # "PASS" line doesn't match FAIL/failure/issue/error — list should be empty or minimal
    fail_issues = [i for i in issues if re.search(r"\bFAIL\b|\bfailure\b|\bissue\b|\berror\b", i, re.IGNORECASE)]
    assert len(fail_issues) == 0


import re  # noqa: E402  (needed for inline use above)

# ── _parse_success_structured / _parse_success ────────────────


def test_parse_success_structured_from_json() -> None:
    import json

    report = json.dumps({"success": True, "issues": [], "details": {}})
    assert CodingSkillOrchestrator._parse_success_structured(report) is True


def test_parse_success_structured_false_from_json() -> None:
    import json

    report = json.dumps({"success": False, "issues": ["foo"], "details": {}})
    assert CodingSkillOrchestrator._parse_success_structured(report) is False


def test_parse_success_structured_falls_back_to_heuristic() -> None:
    # Plain text, not JSON — falls back to _parse_success
    assert CodingSkillOrchestrator._parse_success_structured("Overall: PASS ✅") is True
    assert CodingSkillOrchestrator._parse_success_structured("Overall: FAIL ❌") is False


def test_parse_success_no_verdict_optimistic() -> None:
    # No PASS/FAIL markers → optimistic default
    assert CodingSkillOrchestrator._parse_success("Verification complete.") is True


def test_parse_success_fail_word_boundary() -> None:
    # "No failures detected" should NOT be treated as FAIL
    assert CodingSkillOrchestrator._parse_success("No failures detected.") is True


def test_parse_success_fail_emoji() -> None:
    assert CodingSkillOrchestrator._parse_success("Result: ❌") is False


def test_parse_success_pass_emoji() -> None:
    assert CodingSkillOrchestrator._parse_success("Result: ✅") is True


# ── _emit_fix_forward_event ───────────────────────────────────


def test_emit_fix_forward_event_does_not_raise() -> None:
    """Even if EventBus fails, the helper must swallow exceptions."""
    from unittest.mock import patch

    with patch("vaig.agents.coding_pipeline.EventBus") as mock_bus:
        mock_bus.get.side_effect = RuntimeError("bus is broken")
        # Should not raise
        CodingSkillOrchestrator._emit_fix_forward_event(2)


def test_emit_fix_forward_event_calls_event_bus() -> None:
    from unittest.mock import MagicMock, patch

    mock_bus = MagicMock()
    with patch("vaig.agents.coding_pipeline.EventBus") as mock_cls:
        mock_cls.get.return_value = mock_bus
        CodingSkillOrchestrator._emit_fix_forward_event(3)

    mock_bus.emit.assert_called_once()
    event = mock_bus.emit.call_args[0][0]
    assert event.loop_type == "fix_forward"
    assert event.iteration == 3
    assert event.skill == "coding-pipeline"
