"""Tests for JiraClient — issue creation, dedup, auth errors, project validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from vaig.core.config import JiraConfig
from vaig.integrations.jira import JiraClient

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def jira_config() -> JiraConfig:
    """Return a fully configured JiraConfig for tests."""
    return JiraConfig(
        enabled=True,
        base_url="https://myorg.atlassian.net",
        email="test@example.com",
        api_token="test-token",
        project_key="OPS",
        issue_type="Bug",
    )


@pytest.fixture()
def client(jira_config: JiraConfig) -> JiraClient:
    """Return a JiraClient with full config."""
    return JiraClient(jira_config)


# ── Config auto-enable tests ────────────────────────────────


class TestJiraConfigAutoEnable:
    """Tests for JiraConfig auto-enable validator."""

    def test_auto_enable_when_base_url_set(self) -> None:
        config = JiraConfig(base_url="https://myorg.atlassian.net")
        assert config.enabled is True

    def test_stays_disabled_without_base_url(self) -> None:
        config = JiraConfig()
        assert config.enabled is False

    def test_explicit_enabled_true_stays(self) -> None:
        config = JiraConfig(enabled=True, base_url="https://myorg.atlassian.net")
        assert config.enabled is True

    def test_enabled_without_base_url_disables(self) -> None:
        """enabled=True but no base_url → warn and disable."""
        config = JiraConfig(enabled=True, base_url="")
        assert config.enabled is False

    def test_severity_mapping_defaults(self) -> None:
        config = JiraConfig()
        assert config.severity_field_mapping["CRITICAL"] == "Highest"
        assert config.severity_field_mapping["HIGH"] == "High"
        assert config.severity_field_mapping["MEDIUM"] == "Medium"
        assert config.severity_field_mapping["LOW"] == "Low"
        assert config.severity_field_mapping["INFO"] == "Lowest"

    def test_credentials_not_in_repr(self) -> None:
        config = JiraConfig(
            base_url="https://myorg.atlassian.net",
            email="secret@test.com",
            api_token="super-secret",
        )
        r = repr(config)
        assert "secret@test.com" not in r
        assert "super-secret" not in r


# ── Create issue tests ──────────────────────────────────────


class TestJiraClientCreateIssue:
    """Tests for JiraClient.create_issue."""

    @patch("vaig.integrations.jira.requests.post")
    def test_create_issue_success(self, mock_post: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": "10001",
            "key": "OPS-42",
            "self": "https://myorg.atlassian.net/rest/api/3/issue/10001",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = client.create_issue(
            summary="CrashLoop in payment-svc",
            description="Pod is crash-looping due to OOM.",
            priority="High",
            labels=["pod-health", "crashloop-payment"],
        )

        assert result["key"] == "OPS-42"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["fields"]["project"]["key"] == "OPS"
        assert payload["fields"]["issuetype"]["name"] == "Bug"
        assert payload["fields"]["priority"]["name"] == "High"
        assert "crashloop-payment" in payload["fields"]["labels"]

    @patch("vaig.integrations.jira.requests.post")
    def test_create_issue_auth_error(self, mock_post: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            client.create_issue(
                summary="test", description="test", priority="High", labels=[]
            )

    @patch("vaig.integrations.jira.requests.post")
    def test_create_issue_forbidden(self, mock_post: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            client.create_issue(
                summary="test", description="test", priority="High", labels=[]
            )


# ── Dedup / search existing tests ───────────────────────────


class TestJiraClientSearchExisting:
    """Tests for JiraClient._search_existing."""

    @patch("vaig.integrations.jira.requests.get")
    def test_finds_existing_issue(self, mock_get: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "issues": [{"key": "OPS-42", "id": "10001"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = client._search_existing("crashloop-payment")
        assert result == "OPS-42"

    @patch("vaig.integrations.jira.requests.get")
    def test_no_existing_issue(self, mock_get: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"issues": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = client._search_existing("nonexistent")
        assert result is None

    @patch("vaig.integrations.jira.requests.get")
    def test_search_error_returns_none(self, mock_get: MagicMock, client: JiraClient) -> None:
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        result = client._search_existing("any-finding")
        assert result is None


# ── Validate project tests ──────────────────────────────────


class TestJiraClientValidateProject:
    """Tests for JiraClient.validate_project."""

    @patch("vaig.integrations.jira.requests.get")
    def test_valid_project(self, mock_get: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        assert client.validate_project() is True

    @patch("vaig.integrations.jira.requests.get")
    def test_invalid_project(self, mock_get: MagicMock, client: JiraClient) -> None:
        mock_get.side_effect = requests.exceptions.HTTPError("Not Found")

        assert client.validate_project() is False


# ── Add comment tests ───────────────────────────────────────


class TestJiraClientAddComment:
    """Tests for JiraClient.add_comment."""

    @patch("vaig.integrations.jira.requests.post")
    def test_add_comment_success(self, mock_post: MagicMock, client: JiraClient) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # Should not raise
        client.add_comment("OPS-42", "Evidence from diagnosis")
        mock_post.assert_called_once()

    @patch("vaig.integrations.jira.requests.post")
    def test_add_comment_error_does_not_raise(
        self, mock_post: MagicMock, client: JiraClient
    ) -> None:
        mock_post.side_effect = requests.exceptions.ConnectionError("fail")

        # Should not raise — add_comment swallows errors
        client.add_comment("OPS-42", "Evidence")


# ── Issue URL tests ─────────────────────────────────────────


class TestJiraClientIssueUrl:
    """Tests for JiraClient.issue_url."""

    def test_issue_url_format(self, client: JiraClient) -> None:
        url = client.issue_url("OPS-42")
        assert url == "https://myorg.atlassian.net/browse/OPS-42"
