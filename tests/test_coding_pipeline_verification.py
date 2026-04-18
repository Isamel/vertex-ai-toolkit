"""Tests for VerificationReport integration in CodingSkillOrchestrator (CM-02).

Covers _parse_success_structured:
- structured JSON success path
- structured JSON failure path
- markdown-fenced JSON is stripped and parsed
- unstructured PASS text falls back to regex → True
- unstructured FAIL text falls back to regex → False
- unstructured text with no verdict falls back to regex → True (optimistic)
"""

from __future__ import annotations

from vaig.agents.coding_pipeline import CodingSkillOrchestrator

parse = CodingSkillOrchestrator._parse_success_structured


# ── Structured JSON paths ─────────────────────────────────────────────────────


def test_structured_json_success_true():
    raw = '{"success": true, "files_verified": [], "issues": [], "summary": "ok"}'
    assert parse(raw) is True


def test_structured_json_success_false():
    raw = '{"success": false, "issues": ["broken import"]}'
    assert parse(raw) is False


def test_structured_json_minimal_true():
    assert parse('{"success": true}') is True


def test_structured_json_minimal_false():
    assert parse('{"success": false}') is False


# ── Markdown fence stripping ──────────────────────────────────────────────────


def test_markdown_fenced_json_success():
    raw = '```json\n{"success": true}\n```'
    assert parse(raw) is True


def test_markdown_fenced_json_failure():
    raw = '```json\n{"success": false, "issues": ["oops"]}\n```'
    assert parse(raw) is False


def test_plain_fenced_json_success():
    raw = '```\n{"success": true}\n```'
    assert parse(raw) is True


# ── Fallback to regex heuristics ──────────────────────────────────────────────


def test_unstructured_pass_returns_true():
    assert parse("Verification complete. Overall: PASS ✅") is True


def test_unstructured_fail_returns_false():
    assert parse("Verification complete. Overall: FAIL ❌") is False


def test_unstructured_pass_emoji_only():
    assert parse("Everything looks good ✅") is True


def test_unstructured_fail_emoji_only():
    assert parse("Something broke ❌") is False


def test_unstructured_no_verdict_returns_true():
    # Optimistic default — no PASS/FAIL signal → True
    assert parse("The code looks reasonable but I cannot fully verify.") is True


def test_fail_word_boundary_no_false_positive():
    # "failures" should NOT trigger FAIL word-boundary match
    assert parse("No failures detected in the codebase.") is True
