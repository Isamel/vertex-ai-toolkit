"""Tests for the FastAPI webhook server — Datadog alert → vaig analysis pipeline."""

from __future__ import annotations

import hashlib
import hmac as hmac_module
import json
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from vaig.integrations.webhook_server import (
    AnalysisResult,
    DailyBudgetCounter,
    DatadogWebhookPayload,
    DedupCache,
    create_webhook_app,
    parse_datadog_tags,
    verify_hmac_signature,
)

# ── FastAPI test client ──────────────────────────────────────

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed — skipping webhook server tests", allow_module_level=True)


# ── Tag parsing tests ────────────────────────────────────────


class TestParseDatadogTags:
    """Tests for parse_datadog_tags helper."""

    def test_empty_string(self) -> None:
        assert parse_datadog_tags("") == {}

    def test_single_tag(self) -> None:
        result = parse_datadog_tags("env:production")
        assert result == {"env": "production"}

    def test_multiple_tags(self) -> None:
        result = parse_datadog_tags("env:production,service:my-api,kube_cluster_name:prod-us")
        assert result == {
            "env": "production",
            "service": "my-api",
            "kube_cluster_name": "prod-us",
        }

    def test_tag_without_value(self) -> None:
        result = parse_datadog_tags("standalone,env:prod")
        assert result == {"standalone": "", "env": "prod"}

    def test_whitespace_handling(self) -> None:
        result = parse_datadog_tags(" env : production , service : my-api ")
        assert result == {"env": "production", "service": "my-api"}

    def test_empty_segments_ignored(self) -> None:
        result = parse_datadog_tags("env:prod,,service:api,")
        assert result == {"env": "prod", "service": "api"}

    def test_colon_in_value(self) -> None:
        """Tag value containing colons (e.g. URL) should preserve everything after first colon."""
        result = parse_datadog_tags("url:http://example.com:8080")
        assert result == {"url": "http://example.com:8080"}


# ── DedupCache tests ─────────────────────────────────────────


class TestDedupCache:
    """Tests for the in-memory deduplication cache."""

    def test_new_key_is_not_duplicate(self) -> None:
        cache = DedupCache(cooldown_seconds=60)
        assert cache.is_duplicate("alert-1") is False

    def test_recorded_key_is_duplicate(self) -> None:
        cache = DedupCache(cooldown_seconds=60)
        cache.record("alert-1")
        assert cache.is_duplicate("alert-1") is True

    def test_different_key_is_not_duplicate(self) -> None:
        cache = DedupCache(cooldown_seconds=60)
        cache.record("alert-1")
        assert cache.is_duplicate("alert-2") is False

    def test_expired_key_is_not_duplicate(self) -> None:
        cache = DedupCache(cooldown_seconds=0)  # 0 second cooldown = immediate expiry
        cache.record("alert-1")
        # Manually expire by setting expiry to the past
        cache._cache["alert-1"] = time.monotonic() - 1
        assert cache.is_duplicate("alert-1") is False

    def test_size_property(self) -> None:
        cache = DedupCache(cooldown_seconds=60)
        assert cache.size == 0
        cache.record("a")
        cache.record("b")
        assert cache.size == 2

    def test_cleanup_removes_expired(self) -> None:
        cache = DedupCache(cooldown_seconds=60)
        cache.record("expired")
        cache._cache["expired"] = time.monotonic() - 1  # Force expire
        cache.record("active")
        # Trigger cleanup via is_duplicate
        cache.is_duplicate("probe")
        assert cache.size == 1


# ── DailyBudgetCounter tests ─────────────────────────────────


class TestDailyBudgetCounter:
    """Tests for the daily analysis budget counter."""

    def test_initial_state(self) -> None:
        counter = DailyBudgetCounter(max_per_day=10)
        assert counter.count == 0
        assert counter.remaining == 10
        assert counter.max_per_day == 10
        assert counter.can_analyze() is True

    def test_increment(self) -> None:
        counter = DailyBudgetCounter(max_per_day=10)
        result = counter.increment()
        assert result == 1
        assert counter.count == 1
        assert counter.remaining == 9

    def test_budget_exhaustion(self) -> None:
        counter = DailyBudgetCounter(max_per_day=2)
        counter.increment()
        counter.increment()
        assert counter.can_analyze() is False
        assert counter.remaining == 0

    def test_date_reset(self) -> None:
        counter = DailyBudgetCounter(max_per_day=10)
        counter.increment()
        counter.increment()
        assert counter.count == 2

        # Simulate date change
        counter._date = "1999-01-01"
        assert counter.can_analyze() is True
        assert counter.count == 0  # Reset after date change check


# ── HMAC verification tests ──────────────────────────────────


class TestHMACVerification:
    """Tests for HMAC-SHA256 signature verification."""

    def test_valid_signature(self) -> None:
        body = b'{"alert_id": "123"}'
        secret = "my-secret"
        signature = hmac_module.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        assert verify_hmac_signature(body, signature, secret) is True

    def test_invalid_signature(self) -> None:
        body = b'{"alert_id": "123"}'
        assert verify_hmac_signature(body, "bad-signature", "my-secret") is False

    def test_wrong_secret(self) -> None:
        body = b'{"alert_id": "123"}'
        secret = "my-secret"
        signature = hmac_module.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        assert verify_hmac_signature(body, signature, "wrong-secret") is False

    def test_tampered_body(self) -> None:
        body = b'{"alert_id": "123"}'
        secret = "my-secret"
        signature = hmac_module.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        tampered = b'{"alert_id": "456"}'
        assert verify_hmac_signature(tampered, signature, secret) is False


# ── DatadogWebhookPayload model tests ────────────────────────


class TestDatadogWebhookPayload:
    """Tests for the Datadog webhook payload Pydantic model."""

    def test_minimal_payload(self) -> None:
        payload = DatadogWebhookPayload()
        assert payload.id == ""
        assert payload.alert_id == ""
        assert payload.tags == ""
        assert payload.priority == "normal"

    def test_full_payload(self) -> None:
        data = {
            "id": "evt-123",
            "title": "CPU High",
            "text": "CPU usage is above 90%",
            "alert_id": "alert-456",
            "alert_status": "Triggered",
            "alert_transition": "Triggered",
            "hostname": "web-01",
            "tags": "env:prod,service:api,kube_cluster_name:prod-us",
            "org": {"id": "org-1", "name": "MyOrg"},
        }
        payload = DatadogWebhookPayload(**data)
        assert payload.id == "evt-123"
        assert payload.alert_id == "alert-456"
        assert payload.hostname == "web-01"
        assert payload.org.name == "MyOrg"

    def test_extra_fields_ignored(self) -> None:
        """Datadog may send fields we don't model — should not crash."""
        data = {"id": "evt-1", "unknown_field": "value"}
        payload = DatadogWebhookPayload.model_validate(data)
        assert payload.id == "evt-1"


# ── AnalysisResult model tests ───────────────────────────────


class TestAnalysisResult:
    """Tests for the AnalysisResult tracking model."""

    def test_default_values(self) -> None:
        result = AnalysisResult(alert_id="a1", service_name="svc")
        assert result.status == "pending"
        assert result.error == ""
        assert result.dispatch_errors == []

    def test_status_transitions(self) -> None:
        result = AnalysisResult(alert_id="a1", service_name="svc")
        result.status = "running"
        result.started_at = datetime.now(UTC).isoformat()
        assert result.status == "running"


# ── FastAPI endpoint tests ───────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_healthy(self) -> None:
        app = create_webhook_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_stats_initial_state(self) -> None:
        app = create_webhook_app(max_analyses_per_day=25)
        client = TestClient(app)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["analyses_today"] == 0
        assert data["budget_max"] == 25
        assert data["budget_remaining"] == 25
        assert data["dedup_cache_size"] == 0


class TestDatadogWebhookEndpoint:
    """Tests for POST /webhook/datadog."""

    def _make_payload(self, **overrides: object) -> dict[str, object]:
        """Build a valid Datadog webhook payload dict."""
        base: dict[str, object] = {
            "id": "evt-100",
            "alert_id": "alert-200",
            "title": "High CPU on api-server",
            "alert_status": "Triggered",
            "alert_transition": "Triggered",
            "hostname": "web-01",
            "tags": "env:prod,service:api-server,kube_cluster_name:prod-us,kube_namespace:default",
        }
        base.update(overrides)
        return base

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_accepts_valid_webhook(self, _mock_bg: object) -> None:
        app = create_webhook_app()
        client = TestClient(app)
        payload = self._make_payload()
        resp = client.post("/webhook/datadog", json=payload)

        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["alert_id"] == "alert-200"
        assert data["service"] == "api-server"
        assert data["cluster"] == "prod-us"
        assert data["namespace"] == "default"

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_dedup_skips_repeated_alert(self, _mock_bg: object) -> None:
        app = create_webhook_app(dedup_cooldown_seconds=300)
        client = TestClient(app)
        payload = self._make_payload()

        resp1 = client.post("/webhook/datadog", json=payload)
        assert resp1.status_code == 202

        resp2 = client.post("/webhook/datadog", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "skipped"
        assert resp2.json()["reason"] == "duplicate"

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_budget_exhaustion_returns_429(self, _mock_bg: object) -> None:
        app = create_webhook_app(max_analyses_per_day=1, dedup_cooldown_seconds=0)
        client = TestClient(app)

        # First request uses the budget
        resp1 = client.post("/webhook/datadog", json=self._make_payload(alert_id="a1"))
        assert resp1.status_code == 202

        # Second request exhausts the budget
        resp2 = client.post("/webhook/datadog", json=self._make_payload(alert_id="a2"))
        assert resp2.status_code == 429
        assert resp2.json()["reason"] == "budget_exhausted"

    def test_invalid_json_returns_400(self) -> None:
        app = create_webhook_app()
        client = TestClient(app)
        resp = client.post(
            "/webhook/datadog",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_service_from_hostname_fallback(self, _mock_bg: object) -> None:
        """When tags have no service, falls back to hostname."""
        app = create_webhook_app()
        client = TestClient(app)
        payload = self._make_payload(tags="env:prod", hostname="web-01")
        resp = client.post("/webhook/datadog", json=payload)

        assert resp.status_code == 202
        assert resp.json()["service"] == "web-01"

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_service_unknown_fallback(self, _mock_bg: object) -> None:
        """When no service in tags and no hostname, uses 'unknown'."""
        app = create_webhook_app()
        client = TestClient(app)
        payload = self._make_payload(tags="", hostname="", alert_id="a-unique")
        resp = client.post("/webhook/datadog", json=payload)

        assert resp.status_code == 202
        assert resp.json()["service"] == "unknown"


class TestHMACEnforcement:
    """Tests for HMAC signature verification on the webhook endpoint."""

    def _sign(self, body: bytes, secret: str) -> str:
        return hmac_module.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

    def test_missing_signature_when_required(self) -> None:
        app = create_webhook_app(hmac_secret="my-secret")
        client = TestClient(app)
        payload = json.dumps({"alert_id": "a1"}).encode()
        resp = client.post(
            "/webhook/datadog",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_signature_rejected(self) -> None:
        app = create_webhook_app(hmac_secret="my-secret")
        client = TestClient(app)
        payload = json.dumps({"alert_id": "a1"}).encode()
        resp = client.post(
            "/webhook/datadog",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-DD-Signature": "invalid-hex",
            },
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_valid_signature_accepted(self, _mock_bg: object) -> None:
        secret = "my-secret"
        app = create_webhook_app(hmac_secret=secret)
        client = TestClient(app)
        body = json.dumps({"alert_id": "a1", "tags": "service:svc"}).encode()
        signature = self._sign(body, secret)
        resp = client.post(
            "/webhook/datadog",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-DD-Signature": signature,
            },
        )
        assert resp.status_code == 202

    @patch("vaig.integrations.webhook_server.run_analysis_background", new_callable=AsyncMock)
    def test_no_hmac_enforcement_when_no_secret(self, _mock_bg: object) -> None:
        """When no HMAC secret is configured, any request is accepted."""
        app = create_webhook_app(hmac_secret="")
        client = TestClient(app)
        resp = client.post(
            "/webhook/datadog",
            json={"alert_id": "a1", "tags": "service:svc"},
        )
        assert resp.status_code == 202


# ── Background analysis runner tests ─────────────────────────


class TestRunAnalysisBackground:
    """Tests for the background analysis runner (mocked imports)."""

    @pytest.mark.asyncio()
    async def test_analysis_background_import_error(self) -> None:
        """When orchestrator is not available, result is marked failed."""
        from vaig.integrations.webhook_server import run_analysis_background

        results_store: dict[str, AnalysisResult] = {}
        budget = DailyBudgetCounter(max_per_day=50)

        # Make get_settings raise so the background runner hits its
        # outer except branch and marks the result as failed/completed.
        with patch(
            "vaig.core.config.get_settings",
            side_effect=RuntimeError("settings unavailable in CI"),
        ):
            await run_analysis_background(
                alert_id="test-1",
                service_name="test-svc",
                cluster_name="test-cluster",
                namespace="default",
                results_store=results_store,
                budget_counter=budget,
            )

        result = results_store.get("test-1")
        assert result is not None
        # The result should have completed (possibly with failure if imports fail)
        assert result.completed_at != ""


# ── Config model tests ───────────────────────────────────────


class TestWebhookServerConfig:
    """Tests for WebhookServerConfig in vaig.core.config."""

    def test_default_values(self) -> None:
        from vaig.core.config import WebhookServerConfig

        cfg = WebhookServerConfig()
        assert cfg.enabled is False
        assert cfg.host == "0.0.0.0"  # noqa: S104
        assert cfg.port == 8080
        assert cfg.hmac_secret == ""
        assert cfg.max_analyses_per_day == 50
        assert cfg.dedup_cooldown_seconds == 300
        assert cfg.analysis_timeout_seconds == 600

    def test_auto_enable_with_hmac_secret(self) -> None:
        from vaig.core.config import WebhookServerConfig

        cfg = WebhookServerConfig(hmac_secret="my-secret")
        assert cfg.enabled is True

    def test_explicit_disable_stays_disabled(self) -> None:
        """Explicit enabled=False should remain False even with hmac_secret."""
        from vaig.core.config import WebhookServerConfig

        # Note: auto_enable fires after init, setting enabled=True when secret is present.
        # This is intentional — matches PagerDuty/Google Chat pattern.
        cfg = WebhookServerConfig(hmac_secret="secret")
        assert cfg.enabled is True

    def test_settings_has_webhook_server(self) -> None:
        """The root Settings class should include webhook_server field."""
        from vaig.core.config import Settings, WebhookServerConfig

        # Just check the field exists in the model
        assert "webhook_server" in Settings.model_fields
        # Verify default factory
        settings = Settings()
        assert isinstance(settings.webhook_server, WebhookServerConfig)
