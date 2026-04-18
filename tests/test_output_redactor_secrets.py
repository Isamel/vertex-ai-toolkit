"""Tests for new secret patterns added to output_redactor.py."""

from __future__ import annotations

import pytest

from vaig.core.output_redactor import redact_sensitive_output


class TestAWSAccessKeyRedaction:
    """AKIA-prefixed AWS access key IDs should be redacted."""

    def test_aws_access_key_is_redacted(self) -> None:
        output = "key: AKIAIOSFODNN7EXAMPLE1234"
        result, count = redact_sensitive_output(output)
        assert "AKIAIOSFODNN7EXAMPLE1234" not in result
        assert count >= 1

    def test_aws_access_key_prefix_preserved(self) -> None:
        output = "AKIAIOSFODNN7EXAMPLE1234"
        result, count = redact_sensitive_output(output)
        assert "AKIA" in result
        assert count >= 1


class TestAWSSecretKeyRedaction:
    """AWS secret access keys (keyword-prefixed + 40 chars) should be redacted."""

    def test_aws_secret_key_is_redacted(self) -> None:
        secret = "aws_secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result, count = redact_sensitive_output(secret)
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result
        assert count >= 1


class TestGitHubTokenRedaction:
    """GitHub PAT tokens (ghp_ and gho_) should be redacted."""

    def test_ghp_token_is_redacted(self) -> None:
        token = "ghp_" + "A" * 36
        output = f"token={token}"
        result, count = redact_sensitive_output(output)
        assert token not in result
        assert count >= 1

    def test_ghp_prefix_preserved(self) -> None:
        token = "ghp_" + "B" * 36
        result, count = redact_sensitive_output(token)
        assert "ghp_" in result

    def test_gho_token_is_redacted(self) -> None:
        token = "gho_" + "C" * 36
        output = f"Authorization: {token}"
        result, count = redact_sensitive_output(output)
        assert token not in result
        assert count >= 1

    def test_gho_prefix_preserved(self) -> None:
        token = "gho_" + "D" * 36
        result, count = redact_sensitive_output(token)
        assert "gho_" in result


class TestSlackWebhookRedaction:
    """Slack webhook URLs should be redacted."""

    def test_slack_webhook_url_is_redacted(self) -> None:
        webhook = "https://hooks.slack.com/services/T01234567/B01234567/abcdefghijklmnopqrstuvwxyz12"
        output = f"webhook_url: {webhook}"
        result, count = redact_sensitive_output(output)
        assert "abcdefghijklmnopqrstuvwxyz12" not in result
        assert count >= 1

    def test_slack_webhook_prefix_preserved(self) -> None:
        webhook = "https://hooks.slack.com/services/T01234567/B01234567/abcdefghijklmnopqrstuvwxyz12"
        result, _ = redact_sensitive_output(webhook)
        assert "https://hooks.slack.com/services/T" in result


class TestDBConnectionStringRedaction:
    """Database connection strings (postgres/mysql/mongodb) should be redacted."""

    @pytest.mark.parametrize("scheme", ["postgres", "mysql", "mongodb"])
    def test_db_connection_string_redacted(self, scheme: str) -> None:
        conn = f"{scheme}://user:password@host:5432/dbname"
        output = f"DATABASE_URL={conn}"
        result, count = redact_sensitive_output(output)
        assert "password@host" not in result
        assert count >= 1

    def test_db_scheme_prefix_preserved(self) -> None:
        conn = "postgres://user:secret@db.example.com:5432/mydb"
        result, _ = redact_sensitive_output(conn)
        assert "postgres://" in result


class TestNoFalsePositives:
    """Short or safe strings should NOT be redacted."""

    def test_clean_output_unchanged(self) -> None:
        output = "No secrets here, just normal text."
        result, count = redact_sensitive_output(output)
        assert result == output
        assert count == 0

    def test_empty_string(self) -> None:
        result, count = redact_sensitive_output("")
        assert result == ""
        assert count == 0
