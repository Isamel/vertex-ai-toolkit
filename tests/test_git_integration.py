"""Tests for GitManager and git_integration helpers (CM-05)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from vaig.core.config import GitConfig
from vaig.core.git_integration import (
    GitDirtyError,
    GitManager,
    GitSafetyError,
    _sanitize_branch_name,
)

# ── Fixtures ─────────────────────────────────────────────────


def _make_manager(
    *,
    enabled: bool = True,
    auto_branch: bool = True,
    auto_commit: bool = True,
    auto_pr: bool = False,
    commit_signoff: bool = False,
    branch_prefix: str = "vaig/",
    pr_provider: str = "gh",
    workspace: Path | None = None,
) -> GitManager:
    config = GitConfig(
        enabled=enabled,
        auto_branch=auto_branch,
        auto_commit=auto_commit,
        auto_pr=auto_pr,
        commit_signoff=commit_signoff,
        branch_prefix=branch_prefix,
        pr_provider=pr_provider,
    )
    return GitManager(config, workspace=workspace or Path("/fake/workspace"))


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# ── Enabled=False: all methods are no-ops ────────────────────


def test_disabled_check_clean_returns_true() -> None:
    manager = _make_manager(enabled=False)
    assert manager.check_clean() is True


def test_disabled_current_branch_returns_empty() -> None:
    manager = _make_manager(enabled=False)
    assert manager.current_branch() == ""


def test_disabled_create_branch_no_subprocess() -> None:
    manager = _make_manager(enabled=False)
    with patch("subprocess.run") as mock_run:
        manager.create_branch("vaig/test")
        mock_run.assert_not_called()


def test_disabled_commit_all_no_subprocess() -> None:
    manager = _make_manager(enabled=False)
    with patch("subprocess.run") as mock_run:
        manager.commit_all("feat: something")
        mock_run.assert_not_called()


def test_disabled_push_no_subprocess() -> None:
    manager = _make_manager(enabled=False)
    with patch("subprocess.run") as mock_run:
        manager.push()
        mock_run.assert_not_called()


def test_disabled_create_pr_returns_empty() -> None:
    manager = _make_manager(enabled=False)
    with patch("subprocess.run") as mock_run:
        url = manager.create_pr("My PR")
        mock_run.assert_not_called()
        assert url == ""


# ── check_clean ───────────────────────────────────────────────


def test_check_clean_returns_true_when_porcelain_empty() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed(stdout="")) as mock_run:
        assert manager.check_clean() is True
        mock_run.assert_called_once()


def test_check_clean_returns_false_when_dirty() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed(stdout=" M src/foo.py\n")):
        assert manager.check_clean() is False


# ── current_branch ────────────────────────────────────────────


def test_current_branch_returns_branch_name() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed(stdout="vaig/my-feature\n")):
        assert manager.current_branch() == "vaig/my-feature"


# ── create_branch ─────────────────────────────────────────────


def test_create_branch_calls_git_checkout_b() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed()) as mock_run:
        manager.create_branch("vaig/new-feature")
        args = mock_run.call_args[0][0]
        assert args == ["git", "checkout", "-b", "vaig/new-feature"]


# ── commit_all ────────────────────────────────────────────────


def test_commit_all_raises_git_safety_error_on_main() -> None:
    manager = _make_manager()
    with patch(
        "subprocess.run",
        return_value=_completed(stdout="main\n"),
    ):
        with pytest.raises(GitSafetyError) as exc_info:
            manager.commit_all("feat: something")
        assert exc_info.value.branch == "main"


def test_commit_all_raises_git_safety_error_on_master() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed(stdout="master\n")):
        with pytest.raises(GitSafetyError):
            manager.commit_all("feat: something")


def test_commit_all_stages_and_commits_on_feature_branch() -> None:
    manager = _make_manager()
    responses = [
        _completed(stdout="vaig/feature\n"),  # current_branch call
        _completed(),  # git add -A
        _completed(),  # git commit
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        manager.commit_all("feat: add retry logic")

    calls = [c[0][0] for c in mock_run.call_args_list]
    assert calls[1] == ["git", "add", "-A"]
    assert calls[2][:3] == ["git", "commit", "-m"]
    assert "feat: add retry logic" in calls[2]


def test_commit_all_with_signoff_appends_flag() -> None:
    manager = _make_manager(commit_signoff=True)
    responses = [
        _completed(stdout="vaig/feature\n"),
        _completed(),
        _completed(),
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        manager.commit_all("feat: signed commit")

    commit_cmd = mock_run.call_args_list[2][0][0]
    assert "--signoff" in commit_cmd


# ── push ──────────────────────────────────────────────────────


def test_push_raises_git_safety_error_on_main() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed(stdout="main\n")):
        with pytest.raises(GitSafetyError) as exc_info:
            manager.push()
        assert exc_info.value.branch == "main"


def test_push_with_set_upstream() -> None:
    manager = _make_manager()
    responses = [
        _completed(stdout="vaig/feature\n"),  # current_branch
        _completed(),  # push
    ]
    with patch("subprocess.run", side_effect=responses) as mock_run:
        manager.push(set_upstream=True)

    push_cmd = mock_run.call_args_list[1][0][0]
    assert push_cmd == ["git", "push", "-u", "origin", "vaig/feature"]


def test_push_explicit_branch_without_upstream() -> None:
    manager = _make_manager()
    with patch("subprocess.run", return_value=_completed()) as mock_run:
        manager.push("vaig/my-branch", set_upstream=False)

    push_cmd = mock_run.call_args[0][0]
    assert push_cmd == ["git", "push", "origin", "vaig/my-branch"]


# ── create_pr ────────────────────────────────────────────────


def test_create_pr_calls_gh_cli() -> None:
    manager = _make_manager()
    with patch(
        "subprocess.run",
        return_value=_completed(stdout="https://github.com/owner/repo/pull/42\n"),
    ) as mock_run:
        url = manager.create_pr("Add retry logic", body="PR body", base="main")

    assert url == "https://github.com/owner/repo/pull/42"
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "gh"
    assert "pr" in cmd
    assert "create" in cmd


def test_create_pr_raises_for_unsupported_provider() -> None:
    manager = _make_manager(pr_provider="gitlab")
    with pytest.raises(RuntimeError, match="Unsupported PR provider"):
        manager.create_pr("My PR")


# ── _sanitize_branch_name ─────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "prefix", "expected"),
    [
        ("Add retry logic to GCS upload", "vaig/", "vaig/add-retry-logic-to-gcs-upload"),
        ("Fix: broken test!!!", "vaig/", "vaig/fix-broken-test"),
        ("   spaces   ", "feat/", "feat/spaces"),
        ("a" * 100, "vaig/", "vaig/" + "a" * 60),
    ],
)
def test_sanitize_branch_name(raw: str, prefix: str, expected: str) -> None:
    assert _sanitize_branch_name(raw, prefix) == expected


# ── GitSafetyError / GitDirtyError ───────────────────────────


def test_git_safety_error_message_contains_branch() -> None:
    exc = GitSafetyError("main", "commit")
    assert "main" in str(exc)
    assert exc.branch == "main"
    assert exc.operation == "commit"


def test_git_dirty_error_message() -> None:
    exc = GitDirtyError("M src/foo.py")
    assert "dirty" in str(exc).lower()
    assert "src/foo.py" in exc.details
