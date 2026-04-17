"""Tests for VerificationReport Pydantic schema (CM-02)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaig.core.schemas import VerificationReport


def test_full_json_parses_correctly():
    raw = '{"success": true, "files_verified": ["a.py", "b.py"], "issues": [], "summary": "All good"}'
    report = VerificationReport.model_validate_json(raw)
    assert report.success is True
    assert report.files_verified == ["a.py", "b.py"]
    assert report.issues == []
    assert report.summary == "All good"


def test_minimal_json_success_true():
    raw = '{"success": true}'
    report = VerificationReport.model_validate_json(raw)
    assert report.success is True
    assert report.files_verified == []
    assert report.issues == []
    assert report.summary == ""


def test_minimal_json_success_false():
    raw = '{"success": false}'
    report = VerificationReport.model_validate_json(raw)
    assert report.success is False


def test_extra_keys_are_ignored():
    raw = '{"success": true, "unexpected_field": "ignored"}'
    report = VerificationReport.model_validate_json(raw)
    assert report.success is True


def test_missing_success_raises_validation_error():
    raw = '{"files_verified": ["a.py"]}'
    with pytest.raises(ValidationError):
        VerificationReport.model_validate_json(raw)


def test_issues_list_is_preserved():
    raw = '{"success": false, "issues": ["Import error in a.py", "Type mismatch in b.py"]}'
    report = VerificationReport.model_validate_json(raw)
    assert len(report.issues) == 2
    assert "Import error in a.py" in report.issues
