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


def test_corrupted_manifest_graceful(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.put("fp4", [{"file": "x"}], [{"chunk": "y"}])

    # Corrupt the manifest file
    key = cache._make_key("fp4")
    manifest_path, _, _ = cache._paths(key)
    manifest_path.write_text("NOT VALID JSON !!!", encoding="utf-8")

    # Corrupted manifest must be treated as a cache miss (return None, no crash)
    result = cache.get("fp4")
    assert result is None


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


def test_invalidate_nonexistent_returns_false(tmp_path: Path) -> None:
    """invalidate() on a fingerprint that was never cached should return False."""
    cache = _cache(tmp_path)
    result = cache.invalidate("never-existed")
    assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# _write_atomic helper — security and atomicity guarantees
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteAtomic:
    """Verify the _write_atomic helper meets the three contract requirements."""

    def test_file_ends_with_0600_mode(self, tmp_path: Path) -> None:
        """Written file must have exactly 0o600 permissions (SA-7)."""
        import os
        import stat

        from vaig.core.attachment_cache import _write_atomic

        dest = tmp_path / "secret.json"
        _write_atomic(dest, b'{"x": 1}')
        mode = stat.S_IMODE(os.stat(dest).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_no_tempfile_left_behind_after_success(self, tmp_path: Path) -> None:
        """After a successful write there must be no leftover temp files."""
        from vaig.core.attachment_cache import _write_atomic

        dest = tmp_path / "clean.json"
        _write_atomic(dest, b"data")
        files = list(tmp_path.iterdir())
        assert files == [dest], f"unexpected files in tmp_path: {files}"

    def test_original_not_modified_if_replace_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If os.replace raises, the original file must be untouched and no temp files remain."""
        import os

        from vaig.core.attachment_cache import _write_atomic

        # Write original content
        dest = tmp_path / "original.json"
        original_content = b"original"
        dest.write_bytes(original_content)

        real_replace = os.replace

        def failing_replace(src: str, dst: str) -> None:  # type: ignore[return]
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="simulated replace failure"):
            _write_atomic(dest, b"new content")

        # Original must be intact
        assert dest.read_bytes() == original_content

        # No temp file remains
        leftover = [f for f in tmp_path.iterdir() if f != dest]
        assert not leftover, f"temp file(s) not cleaned up: {leftover}"
