"""Jira Cloud REST API v3 integration — create issues from health-report findings."""

from __future__ import annotations

import logging
from typing import Any

import requests

from vaig.core.config import JiraConfig

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


class JiraClient:
    """Jira Cloud REST API v3 client for issue creation and search.

    Uses HTTP Basic Auth (email + API token) consistent with
    Atlassian Cloud's authentication model.
    """

    def __init__(self, config: JiraConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.email = config.email
        self.api_token = config.api_token
        self.project_key = config.project_key
        self.issue_type = config.issue_type
        self.severity_field_mapping = config.severity_field_mapping

    # ── Public API ───────────────────────────────────────────

    def create_issue(
        self,
        summary: str,
        description: str,
        priority: str,
        labels: list[str],
        components: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a Jira issue via REST API v3.

        Args:
            summary: Issue summary (title).
            description: Plain-text description.
            priority: Jira priority name (e.g. "High").
            labels: List of labels to apply.
            components: Optional list of component names.

        Returns:
            Dict with ``key``, ``id``, and ``self`` from the Jira response.

        Raises:
            requests.exceptions.HTTPError: On API errors.
        """
        payload = self._build_issue_payload(
            summary=summary,
            description=description,
            priority=priority,
            labels=labels,
            components=components,
        )

        try:
            resp = requests.post(
                f"{self.base_url}/rest/api/3/issue",
                json=payload,
                auth=(self.email, self.api_token),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            logger.info("Created Jira issue %s", data.get("key"))
            return data
        except (KeyboardInterrupt, SystemExit):
            raise
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 401:
                logger.error(
                    "Jira authentication failed (401). "
                    "Verify VAIG_JIRA__EMAIL and VAIG_JIRA__API_TOKEN."
                )
            elif status == 403:
                logger.error(
                    "Jira authorization failed (403). "
                    "Check project permissions for %s.",
                    self.project_key,
                )
            elif status == 404:
                logger.error(
                    "Jira resource not found (404). "
                    "Verify base_url (%s) and project_key (%s).",
                    self.base_url,
                    self.project_key,
                )
            raise

    def add_comment(self, issue_key: str, body: str) -> None:
        """Add a plain-text comment to an existing Jira issue.

        Args:
            issue_key: The Jira issue key (e.g. ``OPS-123``).
            body: Comment body as plain text.
        """
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": body[:32767]}],
                    }
                ],
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/rest/api/3/issue/{issue_key}/comment",
                json=payload,
                auth=(self.email, self.api_token),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("Added comment to Jira issue %s", issue_key)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("Failed to add comment to Jira issue %s", issue_key)

    def validate_project(self) -> bool:
        """Pre-flight check that the configured project exists.

        Returns:
            ``True`` if the project is accessible, ``False`` otherwise.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/rest/api/3/project/{self.project_key}",
                auth=(self.email, self.api_token),
                headers={"Accept": "application/json"},
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return True
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.warning(
                "Jira project validation failed for %s", self.project_key
            )
            return False

    def _search_existing(self, finding_id: str) -> str | None:
        """Search for an existing Jira issue with the given finding ID as a label.

        Returns:
            The issue key if found, or ``None``.
        """
        jql = f'project = "{self.project_key}" AND labels = "{finding_id}"'
        try:
            resp = requests.get(
                f"{self.base_url}/rest/api/3/search",
                params={"jql": jql, "maxResults": 1, "fields": "key"},
                auth=(self.email, self.api_token),
                headers={"Accept": "application/json"},
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            issues = data.get("issues", [])
            if issues:
                return str(issues[0]["key"])
            return None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception(
                "Failed to search Jira for existing finding %s", finding_id
            )
            return None

    # ── Private helpers ──────────────────────────────────────

    def _build_issue_payload(
        self,
        summary: str,
        description: str,
        priority: str,
        labels: list[str],
        components: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON payload for issue creation."""
        fields: dict[str, Any] = {
            "project": {"key": self.project_key},
            "summary": summary[:255],  # Jira summary limit
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description[:32767]}],
                    }
                ],
            },
            "issuetype": {"name": self.issue_type},
            "priority": {"name": priority},
            "labels": labels,
        }
        if components:
            fields["components"] = [{"name": c} for c in components]
        return {"fields": fields}

    def issue_url(self, issue_key: str) -> str:
        """Build the browse URL for an issue."""
        return f"{self.base_url}/browse/{issue_key}"
