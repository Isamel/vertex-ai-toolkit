"""Tests for ObservationFingerprint."""
from __future__ import annotations

import pytest

from vaig.core.memory.fingerprint import ObservationFingerprint, _normalise


class TestNormalise:
    def test_strips_timestamps(self) -> None:
        result = _normalise("Error at 2024-03-15T10:30:00Z")
        assert "2024" not in result
        assert "<ts>" in result

    def test_strips_uuids(self) -> None:
        result = _normalise("id=550e8400-e29b-41d4-a716-446655440000")
        assert "550e8400" not in result
        assert "<uuid>" in result

    def test_strips_ips(self) -> None:
        result = _normalise("host 10.0.0.1 failed")
        assert "10.0.0.1" not in result
        assert "<ip>" in result

    def test_strips_pod_hashes(self) -> None:
        result = _normalise("pod payment-svc-7d4f8b-xkz9p crashed")
        assert "7d4f8b" not in result

    def test_strips_counters(self) -> None:
        result = _normalise("restarted 42 times")
        assert "42" not in result
        assert "<n>" in result

    def test_lowercases(self) -> None:
        assert _normalise("CrashLoopBackOff") == _normalise("crashloopbackoff")


class TestObservationFingerprint:
    def test_compute_returns_16_hex_chars(self) -> None:
        fp = ObservationFingerprint.compute("pod-health", "Deployment", "my-svc", "OOMKilled")
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_compute_is_deterministic(self) -> None:
        fp1 = ObservationFingerprint.compute("pod-health", "Deployment", "my-svc", "OOMKilled")
        fp2 = ObservationFingerprint.compute("pod-health", "Deployment", "my-svc", "OOMKilled")
        assert fp1 == fp2

    def test_different_kinds_produce_different_fingerprints(self) -> None:
        fp1 = ObservationFingerprint.compute("pod-health", "Deployment", "svc", "OOMKilled")
        fp2 = ObservationFingerprint.compute("networking", "Deployment", "svc", "OOMKilled")
        assert fp1 != fp2

    def test_pod_hash_variants_produce_same_fingerprint(self) -> None:
        """Two pod names differing only in their ephemeral hash should share a fingerprint."""
        fp1 = ObservationFingerprint.compute(
            "pod-health", "Deployment", "payment-7d4f8b-xkz9p", "CrashLoop"
        )
        fp2 = ObservationFingerprint.compute(
            "pod-health", "Deployment", "payment-abc12-xyz99", "CrashLoop"
        )
        assert fp1 == fp2

    def test_from_finding_maps_fields(self) -> None:
        fp_direct = ObservationFingerprint.compute(
            "pod-health", "payment", "OOMKilled", "pod is OOM"
        )
        fp_finding = ObservationFingerprint.from_finding(
            category="pod-health",
            service="payment",
            title="OOMKilled",
            description="pod is OOM",
        )
        assert fp_direct == fp_finding
