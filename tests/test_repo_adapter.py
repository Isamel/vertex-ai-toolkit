"""Tests for SPEC-V2-REPO-09 — GitLab support + remote/local repo coexistence.

Acceptance criteria:
1. get_adapter("owner/repo")                         → GitHubAdapter
2. get_adapter("https://github.com/owner/repo")      → GitHubAdapter
3. get_adapter("https://gitlab.example.com/acme/c")  → GitLabAdapter (correct base_url)
4. get_adapter("/tmp/gateway-configs")               → LocalFilesystemAdapter
5. get_adapter("file:///tmp/clone")                  → LocalFilesystemAdapter
6. get_adapter("malformed-scheme://foo")             → ValueError with clear message
7. LocalFilesystemAdapter.list_tree()                → list[FileMeta]
8. LocalFilesystemAdapter.fetch_file()               → file content
9. LocalFilesystemAdapter.link_for()                 → file:// URL
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from vaig.core.repo_adapter import (
    GitHubAdapter,
    GitLabAdapter,
    LocalFilesystemAdapter,
    get_adapter,
)
from vaig.core.repo_pipeline import FileMeta

# ── Factory tests ─────────────────────────────────────────────────────────────


def test_get_adapter_owner_repo_shorthand():
    adapter = get_adapter("owner/repo")
    assert isinstance(adapter, GitHubAdapter)
    assert adapter.owner == "owner"
    assert adapter.repo == "repo"


def test_get_adapter_github_https_url():
    adapter = get_adapter("https://github.com/owner/repo")
    assert isinstance(adapter, GitHubAdapter)
    assert adapter.owner == "owner"
    assert adapter.repo == "repo"


def test_get_adapter_github_https_url_with_git_suffix():
    adapter = get_adapter("https://github.com/owner/repo.git")
    assert isinstance(adapter, GitHubAdapter)
    assert adapter.repo == "repo"


def test_get_adapter_gitlab_dot_com():
    adapter = get_adapter("https://gitlab.com/acme/configs")
    assert isinstance(adapter, GitLabAdapter)
    assert adapter.base_url == "https://gitlab.com"
    assert adapter.project_path == "acme/configs"


def test_get_adapter_gitlab_custom_host():
    adapter = get_adapter("https://gitlab.example.com/acme/configs")
    assert isinstance(adapter, GitLabAdapter)
    assert adapter.base_url == "https://gitlab.example.com"
    assert adapter.project_path == "acme/configs"


def test_get_adapter_local_absolute_path(tmp_path: Path):
    adapter = get_adapter(str(tmp_path))
    assert isinstance(adapter, LocalFilesystemAdapter)
    assert adapter.root == tmp_path.resolve()


def test_get_adapter_file_url(tmp_path: Path):
    adapter = get_adapter(f"file://{tmp_path}")
    assert isinstance(adapter, LocalFilesystemAdapter)


def test_get_adapter_malformed_scheme_raises():
    with pytest.raises(ValueError, match="Unrecognized repo spec"):
        get_adapter("malformed-scheme://foo")


def test_get_adapter_malformed_scheme_message_names_schemes():
    with pytest.raises(ValueError, match="GitHub") as exc_info:
        get_adapter("svn+ssh://foo/bar")
    msg = str(exc_info.value)
    assert "GitLab" in msg
    assert "local filesystem" in msg


# ── GitHubAdapter unit tests ──────────────────────────────────────────────────


def test_github_adapter_link_for_no_line_range():
    adapter = GitHubAdapter(owner="acme", repo="configs")
    link = adapter.link_for("main", "charts/gateway/values.yaml")
    assert link == "https://github.com/acme/configs/blob/main/charts/gateway/values.yaml"


def test_github_adapter_link_for_with_line_range():
    adapter = GitHubAdapter(owner="acme", repo="configs")
    link = adapter.link_for("main", "values.yaml", line_range=(10, 20))
    assert "L10-L20" in link


def test_github_adapter_list_tree_raises():
    adapter = GitHubAdapter(owner="acme", repo="configs")
    with pytest.raises(NotImplementedError):
        adapter.list_tree("main")


# ── GitLabAdapter unit tests ──────────────────────────────────────────────────


def test_gitlab_adapter_link_for():
    adapter = GitLabAdapter(project_path="acme/configs", base_url="https://gitlab.example.com")
    link = adapter.link_for("main", "values.yaml")
    assert link == "https://gitlab.example.com/acme/configs/-/blob/main/values.yaml"


def test_gitlab_adapter_link_for_with_line_range():
    adapter = GitLabAdapter(project_path="acme/configs", base_url="https://gitlab.com")
    link = adapter.link_for("main", "values.yaml", line_range=(5, 15))
    assert "L5-15" in link


def test_gitlab_adapter_headers_with_token():
    adapter = GitLabAdapter(project_path="acme/configs", token="tok123")
    assert adapter._headers() == {"PRIVATE-TOKEN": "tok123"}


def test_gitlab_adapter_headers_without_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    adapter = GitLabAdapter(project_path="acme/configs", token=None)
    assert adapter._headers() == {}


# ── LocalFilesystemAdapter integration tests ─────────────────────────────────


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one commit and two files."""
    subprocess.run(["git", "init", "--initial-branch=main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True, capture_output=True)

    (tmp_path / "values.yaml").write_text("key: value\n")
    subdir = tmp_path / "charts"
    subdir.mkdir()
    (subdir / "Chart.yaml").write_text(textwrap.dedent("""\
        apiVersion: v2
        name: gateway
        version: 0.1.0
    """))

    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial commit"],
        check=True,
        capture_output=True,
    )
    return tmp_path


def test_local_adapter_init_raises_on_missing_dir():
    with pytest.raises(ValueError, match="Not a directory"):
        LocalFilesystemAdapter("/nonexistent/path/that/does/not/exist")


def test_local_adapter_list_tree(git_repo: Path):
    adapter = LocalFilesystemAdapter(git_repo)
    files = adapter.list_tree("main")
    paths = {f.path for f in files}
    assert "values.yaml" in paths
    assert "charts/Chart.yaml" in paths
    assert all(isinstance(f, FileMeta) for f in files)


def test_local_adapter_list_tree_subdirectory(git_repo: Path):
    adapter = LocalFilesystemAdapter(git_repo)
    files = adapter.list_tree("main", path="charts")
    paths = {f.path for f in files}
    assert "charts/Chart.yaml" in paths
    assert "values.yaml" not in paths


def test_local_adapter_fetch_file(git_repo: Path):
    adapter = LocalFilesystemAdapter(git_repo)
    content = adapter.fetch_file("main", "values.yaml")
    assert "key: value" in content


def test_local_adapter_link_for(git_repo: Path):
    adapter = LocalFilesystemAdapter(git_repo)
    link = adapter.link_for("main", "values.yaml")
    assert link.startswith("file://")
    assert "values.yaml" in link


def test_local_adapter_commit_history(git_repo: Path):
    adapter = LocalFilesystemAdapter(git_repo)
    commits = adapter.commit_history("main", "values.yaml", limit=5)
    assert len(commits) >= 1
    c = commits[0]
    assert c.sha  # non-empty
    assert "initial commit" in c.message
    assert c.author == "Test"
    assert c.date  # ISO timestamp
