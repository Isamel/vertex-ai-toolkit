"""Tests for SPEC-GEP-01 / SPEC-GEP-02 / SPEC-GEP-03 (Vertex AI global endpoint with fallback).

Covers:
- New :class:`~vaig.core.config.GCPConfig` fields (``endpoint_mode``, probe TTL).
- ``resolve_endpoint_location`` dispatch logic for each mode.
- Probe cache read/write round-trip.
- CLI ``--endpoint`` flag validation.
- Runtime global → regional flip on persistent 429s (GEP-03).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from vaig.core.config import GCPConfig
from vaig.core.endpoint_probe import (
    _CacheEntry,
    _credentials_fingerprint,
    _load_cache_entry,
    _write_cache_entry,
    invalidate_probe_cache,
    resolve_endpoint_location,
)

if TYPE_CHECKING:
    pass


class TestGCPConfigDefaults:
    """New SPEC-GEP-01 fields exist with the right defaults."""

    def test_default_location_is_global(self) -> None:
        cfg = GCPConfig()
        assert cfg.location == "global"

    def test_default_fallback_is_regional(self) -> None:
        cfg = GCPConfig()
        assert cfg.fallback_location == "us-central1"

    def test_default_endpoint_mode(self) -> None:
        cfg = GCPConfig()
        assert cfg.endpoint_mode == "auto"

    def test_probe_timeout_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0.1"):
            GCPConfig(global_probe_timeout_s=0.0)

    def test_probe_ttl_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            GCPConfig(global_probe_cache_ttl_s=-1)

    def test_endpoint_mode_values(self) -> None:
        # Accept the three documented values.
        assert GCPConfig(endpoint_mode="auto").endpoint_mode == "auto"
        assert GCPConfig(endpoint_mode="global").endpoint_mode == "global"
        assert GCPConfig(endpoint_mode="regional").endpoint_mode == "regional"
        # Anything else rejected by Literal.
        with pytest.raises(ValueError):
            GCPConfig(endpoint_mode="invalid")  # type: ignore[arg-type]


class TestResolveEndpointLocation:
    """``resolve_endpoint_location`` branches by mode without hitting the network."""

    def test_regional_mode_skips_probe(self) -> None:
        cfg = GCPConfig(
            project_id="test",
            endpoint_mode="regional",
            location="global",
            fallback_location="us-east1",
        )
        with patch("vaig.core.endpoint_probe._probe_global_endpoint") as probe:
            result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "us-east1"
        probe.assert_not_called()

    def test_regional_mode_empty_fallback_defaults(self) -> None:
        cfg = GCPConfig(
            project_id="test",
            endpoint_mode="regional",
            fallback_location="",
        )
        result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "us-central1"  # default when fallback unset

    def test_auto_mode_with_explicit_region_skips_probe(self) -> None:
        """Auto + region pinned → region as-is, no probe."""
        cfg = GCPConfig(
            project_id="test",
            endpoint_mode="auto",
            location="europe-west1",
        )
        with patch("vaig.core.endpoint_probe._probe_global_endpoint") as probe:
            result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "europe-west1"
        probe.assert_not_called()

    def test_global_mode_probe_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(project_id="test", endpoint_mode="global")
        with patch(
            "vaig.core.endpoint_probe._probe_global_endpoint",
            return_value=True,
        ):
            assert resolve_endpoint_location(cfg, credentials=None) == "global"

    def test_global_mode_probe_failure_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(project_id="test", endpoint_mode="global")
        with (
            patch(
                "vaig.core.endpoint_probe._probe_global_endpoint",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="unreachable"),
        ):
            resolve_endpoint_location(cfg, credentials=None)

    def test_auto_probe_success_returns_global(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(project_id="test", endpoint_mode="auto", location="global")
        with patch(
            "vaig.core.endpoint_probe._probe_global_endpoint",
            return_value=True,
        ) as probe:
            result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "global"
        probe.assert_called_once()

    def test_auto_probe_failure_returns_regional(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(
            project_id="test",
            endpoint_mode="auto",
            location="global",
            fallback_location="us-central1",
        )
        with patch(
            "vaig.core.endpoint_probe._probe_global_endpoint",
            return_value=False,
        ):
            result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "us-central1"


class TestProbeCache:
    """The probe cache persists across calls and is fingerprint-aware."""

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "endpoint-probe.json"
        entry = _CacheEntry(
            project_id="proj",
            credentials_hash="abc123",
            global_available=True,
            probed_at=1_000_000.0,
        )
        _write_cache_entry(path, entry)
        loaded = _load_cache_entry(path)
        assert loaded == entry

    def test_missing_cache_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "does-not-exist.json"
        assert _load_cache_entry(path) is None

    def test_corrupt_cache_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not valid JSON", encoding="utf-8")
        assert _load_cache_entry(path) is None

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"project_id": "foo"}), encoding="utf-8")
        assert _load_cache_entry(path) is None

    def test_cache_hit_within_ttl_skips_probe(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time

        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(
            project_id="proj",
            endpoint_mode="auto",
            location="global",
            global_probe_cache_ttl_s=3600,
        )
        # Pre-populate the cache with a recent "available" entry.
        entry = _CacheEntry(
            project_id="proj",
            credentials_hash="",  # matches fingerprint of credentials=None
            global_available=True,
            probed_at=time.time() - 60,  # 60 s ago, well within 1-h TTL
        )
        _write_cache_entry(tmp_path / "endpoint-probe.json", entry)

        with patch("vaig.core.endpoint_probe._probe_global_endpoint") as probe:
            result = resolve_endpoint_location(cfg, credentials=None)
        assert result == "global"
        probe.assert_not_called()  # TTL hit ⇒ no probe.

    def test_ttl_expired_re_probes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time

        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(
            project_id="proj",
            endpoint_mode="auto",
            location="global",
            global_probe_cache_ttl_s=10,
        )
        entry = _CacheEntry(
            project_id="proj",
            credentials_hash="",
            global_available=True,
            probed_at=time.time() - 100,  # older than TTL (10 s)
        )
        _write_cache_entry(tmp_path / "endpoint-probe.json", entry)

        with patch(
            "vaig.core.endpoint_probe._probe_global_endpoint",
            return_value=True,
        ) as probe:
            resolve_endpoint_location(cfg, credentials=None)
        probe.assert_called_once()

    def test_credentials_change_invalidates_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Different credentials ⇒ different fingerprint ⇒ cache miss."""
        import time

        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        cfg = GCPConfig(
            project_id="proj",
            endpoint_mode="auto",
            location="global",
            global_probe_cache_ttl_s=3600,
        )
        # Entry for a *different* credential fingerprint.
        entry = _CacheEntry(
            project_id="proj",
            credentials_hash="SOMETHING_ELSE",
            global_available=True,
            probed_at=time.time(),
        )
        _write_cache_entry(tmp_path / "endpoint-probe.json", entry)

        with patch(
            "vaig.core.endpoint_probe._probe_global_endpoint",
            return_value=False,
        ) as probe:
            result = resolve_endpoint_location(cfg, credentials=None)
        # Because fingerprints don't match, we probe fresh; probe returned
        # False so we fall back to the regional.
        probe.assert_called_once()
        assert result == "us-central1"

    def test_invalidate_removes_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAIG_CACHE_DIR", str(tmp_path))
        path = tmp_path / "endpoint-probe.json"
        path.write_text("{}", encoding="utf-8")
        assert path.exists()
        invalidate_probe_cache()
        assert not path.exists()


class TestCredentialsFingerprint:
    """``_credentials_fingerprint`` produces stable, length-bounded keys."""

    def test_none_returns_empty(self) -> None:
        assert _credentials_fingerprint(None) == ""

    def test_same_input_same_output(self) -> None:
        class FakeCreds:
            service_account_email = "svc@example.iam.gserviceaccount.com"

        a = _credentials_fingerprint(FakeCreds())  # type: ignore[arg-type]
        b = _credentials_fingerprint(FakeCreds())  # type: ignore[arg-type]
        assert a == b and len(a) == 16

    def test_different_input_different_output(self) -> None:
        class A:
            service_account_email = "a@example.com"

        class B:
            service_account_email = "b@example.com"

        assert _credentials_fingerprint(A()) != _credentials_fingerprint(B())  # type: ignore[arg-type]

    def test_fallback_to_type_name(self) -> None:
        class Opaque:
            pass

        fp = _credentials_fingerprint(Opaque())  # type: ignore[arg-type]
        assert fp and len(fp) == 16


# ── TestGEP03RuntimeFlip ──────────────────────────────────────────────────────


class TestGEP03RuntimeFlip:
    """GEP-03: global → regional fallback triggered by persistent 429s."""

    def _make_client_on_global(self):  # noqa: ANN201  — local import, type resolved at call site
        """Return a GeminiClient pre-initialized on the global endpoint."""
        from unittest.mock import MagicMock, patch

        from vaig.core.client import GeminiClient
        from vaig.core.config import GCPConfig, ModelsConfig, RetryConfig, Settings

        settings = Settings(
            gcp=GCPConfig(
                project_id="proj",
                location="global",
                fallback_location="us-central1",
                endpoint_mode="auto",
            ),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
            retry=RetryConfig(
                max_retries=5,
                initial_delay=0.0,
                max_delay=0.0,
                backoff_multiplier=1.0,
            ),
        )
        with (
            patch("vaig.core.endpoint_probe._probe_global_endpoint", return_value=True),
            patch("vaig.core.client.genai.Client") as mock_cls,
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            client = GeminiClient(settings)
            client.initialize()
            client._mock_genai_cls = mock_cls  # stash for call-count assertions
        return client

    def test_flip_recorded_after_two_consecutive_429s(self) -> None:
        """Two consecutive 429s on global → EndpointFlip appended."""
        from unittest.mock import MagicMock, patch

        from google.genai import errors as genai_errors

        from vaig.core.client import EndpointFlip

        client = self._make_client_on_global()
        assert client._active_location == "global"

        api_429 = genai_errors.ClientError(429, "rate limited")
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            # First two calls raise 429; third call (on regional) succeeds.
            if call_count <= 2:
                raise api_429
            return "ok"

        with (
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            result = client._retry_with_backoff(fn)

        assert result == "ok"
        assert len(client.endpoint_flips) == 1
        flip = client.endpoint_flips[0]
        assert isinstance(flip, EndpointFlip)
        assert flip.from_location == "global"
        assert flip.to_location == "us-central1"
        assert flip.reason == "persistent_429"
        assert client._using_fallback is True
        assert client._active_location == "us-central1"

    def test_flip_does_not_occur_on_single_429(self) -> None:
        """A single 429 on global should NOT trigger a flip — just retry."""
        from unittest.mock import MagicMock, patch

        from google.genai import errors as genai_errors

        client = self._make_client_on_global()

        api_429 = genai_errors.ClientError(429, "rate limited")
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise api_429
            return "ok"

        with (
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            result = client._retry_with_backoff(fn)

        assert result == "ok"
        assert client.endpoint_flips == []
        assert client._active_location == "global"

    def test_consecutive_count_resets_on_success(self) -> None:
        """_consecutive_429_count resets to 0 after a successful call."""
        from unittest.mock import MagicMock, patch

        from google.genai import errors as genai_errors

        client = self._make_client_on_global()

        api_429 = genai_errors.ClientError(429, "rate limited")
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise api_429
            return "ok"

        with (
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            client._retry_with_backoff(fn)

        assert client._consecutive_429_count == 0

    def test_no_flip_when_already_on_regional(self) -> None:
        """No EndpointFlip when the client is already on a regional endpoint."""
        from unittest.mock import MagicMock, patch

        from google.genai import errors as genai_errors

        from vaig.core.client import GeminiClient
        from vaig.core.config import GCPConfig, ModelsConfig, RetryConfig, Settings

        settings = Settings(
            gcp=GCPConfig(
                project_id="proj",
                location="us-central1",
                fallback_location="us-central1",
                endpoint_mode="regional",
            ),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
            retry=RetryConfig(
                max_retries=5,
                initial_delay=0.0,
                max_delay=0.0,
                backoff_multiplier=1.0,
            ),
        )
        with (
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            client = GeminiClient(settings)
            client.initialize()

        api_429 = genai_errors.ClientError(429, "rate limited")
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise api_429
            return "ok"

        with (
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            result = client._retry_with_backoff(fn)

        assert result == "ok"
        # No flip: we were already regional
        assert client.endpoint_flips == []

    def test_endpoint_flips_property_returns_copy(self) -> None:
        """endpoint_flips property returns a copy — mutation does not affect internal list."""
        from vaig.core.client import GeminiClient
        from vaig.core.config import GCPConfig, ModelsConfig, Settings

        settings = Settings(
            gcp=GCPConfig(project_id="proj"),
            models=ModelsConfig(default="gemini-2.5-pro", fallback="gemini-2.5-flash"),
        )
        client = GeminiClient(settings)
        copy1 = client.endpoint_flips
        copy1.append("garbage")  # type: ignore[arg-type]
        assert client.endpoint_flips == []

    def test_flip_log_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Flip emits WARNING with from/to/reason in message."""
        import logging
        from unittest.mock import MagicMock, patch

        from google.genai import errors as genai_errors

        client = self._make_client_on_global()
        api_429 = genai_errors.ClientError(429, "rate limited")
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise api_429
            return "ok"

        # Use root-level caplog so the result is not affected by logger-level
        # mutations from other tests running in the full suite.
        with (
            caplog.at_level(logging.WARNING),
            patch("vaig.core.client.genai.Client"),
            patch("vaig.core.client.get_credentials", return_value=MagicMock()),
        ):
            client._retry_with_backoff(fn)

        flip_logs = [r for r in caplog.records if r.name == "vaig.core.client" and "Endpoint flipped" in r.message]
        assert flip_logs, "Expected at least one 'Endpoint flipped' WARNING log"
        assert "global" in flip_logs[0].message
        assert "us-central1" in flip_logs[0].message
        assert "persistent_429" in flip_logs[0].message
