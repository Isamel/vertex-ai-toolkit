"""Sprint 3 acceptance tests for SPEC-ATT-08: AttachmentCache.

Coverage:
- put → get roundtrip
- config_hash mismatch → miss
- TTL expired → miss
- Corrupted manifest → miss + DEBUG log (graceful)
- Missing chunks file → miss
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaig.core.attachment_cache import AttachmentCache

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cache(tmp_path: Path, *, config_hash: str = "abc123", ttl: int = 3600) -> AttachmentCache:
    return AttachmentCache(tmp_path / "cache", ttl_seconds=ttl, config_hash=config_hash)


# ─────────────────────────────────────────────────────────────────────────────
# put → get roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    manifest = [{"path": "file.txt", "size": 42}]
    chunks = [{"id": "c1", "text": "hello"}]

    cache.put("fp1", manifest, chunks)
    result = cache.get("fp1")

    assert result is not None
    got_manifest, got_chunks = result
    assert got_manifest == manifest
    assert got_chunks == chunks


def test_get_miss_no_entry(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache.get("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# config_hash mismatch → miss
# ─────────────────────────────────────────────────────────────────────────────


def test_config_hash_mismatch_returns_none(tmp_path: Path) -> None:
    writer = _cache(tmp_path, config_hash="hash_v1")
    reader = _cache(tmp_path, config_hash="hash_v2")

    writer.put("fp2", [{"file": "a"}], [{"chunk": "b"}])
    result = reader.get("fp2")
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TTL expired → miss
# ─────────────────────────────────────────────────────────────────────────────


def test_ttl_expired_returns_none(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    cache = _cache(tmp_path, ttl=3600)
    cache.put("fp3", [{"a": 1}], [{"b": 2}])

    # Manually rewrite meta with an old timestamp
    key = cache._make_key("fp3")
    _, _, meta_path = cache._paths(key)
    old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    meta_data = json.loads(meta_path.read_text())
    meta_data["created_at"] = old_time
    meta_path.write_text(json.dumps(meta_data))

    result = cache.get("fp3")
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Corrupted manifest → miss + DEBUG log (graceful)
# ─────────────────────────────────────────────────────────────────────────────


def test_corrupted_manifest_graceful(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    cache = _cache(tmp_path)
    cache.put("fp4", [{"file": "x"}], [{"chunk": "y"}])

    # Corrupt the manifest file
    key = cache._make_key("fp4")
    manifest_path, _, _ = cache._paths(key)
    manifest_path.write_text("NOT VALID JSON !!!", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="vaig.core.attachment_cache"):
        result = cache.get("fp4")

    assert result is None
    # Should have logged a DEBUG message
    assert any("miss" in r.message.lower() or "read" in r.message.lower() for r in caplog.records)


def test_missing_chunks_file_graceful(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.put("fp5", [{"file": "z"}], [{"chunk": "w"}])

    # Delete the chunks file
    key = cache._make_key("fp5")
    _, chunks_path, _ = cache._paths(key)
    chunks_path.unlink()

    result = cache.get("fp5")
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# invalidate
# ─────────────────────────────────────────────────────────────────────────────


def test_invalidate_removes_entry(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.put("fp6", [{"a": 1}], [{"b": 2}])
    assert cache.get("fp6") is not None

    cache.invalidate("fp6")
    assert cache.get("fp6") is None


# ─────────────────────────────────────────────────────────────────────────────
# File permissions (SA-7)
# ─────────────────────────────────────────────────────────────────────────────


def test_cache_files_have_mode_0600(tmp_path: Path) -> None:
    import os
    import stat

    cache = _cache(tmp_path)
    cache.put("fp7", [{"file": "p"}], [{"chunk": "q"}])

    key = cache._make_key("fp7")
    manifest_path, chunks_path, meta_path = cache._paths(key)

    for path in (manifest_path, chunks_path, meta_path):
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"{path} has mode {oct(mode)}, expected 0o600"
