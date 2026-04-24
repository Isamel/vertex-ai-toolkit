"""Sprint 3 acceptance tests for SPEC-ATT-08: AttachmentSession.

Coverage:
- save → load roundtrip
- merge-not-replace (same source twice)
- missing session file → empty list
"""

from __future__ import annotations

from pathlib import Path

from vaig.core.attachment_cache import AttachmentSession

# ─────────────────────────────────────────────────────────────────────────────
# save → load roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path) -> None:
    session = AttachmentSession(tmp_path / "sessions", "sess1")
    session.load()  # empty initially
    session.add(
        source="https://example.com/file.txt",
        name="file.txt",
        fingerprint="abc123",
        kind="url",
    )
    session.save()

    # Re-load from disk
    session2 = AttachmentSession(tmp_path / "sessions", "sess1")
    attachments = session2.load()

    assert len(attachments) == 1
    a = attachments[0]
    assert a.source == "https://example.com/file.txt"
    assert a.name == "file.txt"
    assert a.fingerprint == "abc123"
    assert a.kind == "url"
    assert a.added_at  # non-empty ISO 8601 string


# ─────────────────────────────────────────────────────────────────────────────
# merge-not-replace (same source twice)
# ─────────────────────────────────────────────────────────────────────────────


def test_add_same_source_updates_not_duplicates(tmp_path: Path) -> None:
    session = AttachmentSession(tmp_path / "sessions", "sess2")
    session.load()
    session.add(source="https://example.com/f.txt", name=None, fingerprint="v1", kind="url")
    first_added_at = session._attachments[0].added_at

    # Add the same source again with new fingerprint
    session.add(source="https://example.com/f.txt", name="renamed", fingerprint="v2", kind="url")

    assert len(session._attachments) == 1
    a = session._attachments[0]
    assert a.fingerprint == "v2"
    assert a.name == "renamed"
    # added_at should be updated (or at least not earlier than before)


def test_add_two_different_sources(tmp_path: Path) -> None:
    session = AttachmentSession(tmp_path / "sessions", "sess3")
    session.load()
    session.add(source="https://a.com/1.txt", name=None, fingerprint="fp1", kind="url")
    session.add(source="https://b.com/2.txt", name=None, fingerprint="fp2", kind="url")
    session.save()

    session2 = AttachmentSession(tmp_path / "sessions", "sess3")
    attachments = session2.load()
    assert len(attachments) == 2
    sources = {a.source for a in attachments}
    assert "https://a.com/1.txt" in sources
    assert "https://b.com/2.txt" in sources


# ─────────────────────────────────────────────────────────────────────────────
# missing session file → empty list
# ─────────────────────────────────────────────────────────────────────────────


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    session = AttachmentSession(tmp_path / "sessions", "nonexistent-session")
    attachments = session.load()
    assert attachments == []


# ─────────────────────────────────────────────────────────────────────────────
# corrupted file → empty list (graceful)
# ─────────────────────────────────────────────────────────────────────────────


def test_load_corrupted_file_returns_empty(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    bad_file = session_dir / "bad_sess.json"
    bad_file.write_text("NOT JSON!!", encoding="utf-8")

    session = AttachmentSession(session_dir, "bad_sess")
    attachments = session.load()
    assert attachments == []


# ─────────────────────────────────────────────────────────────────────────────
# multiple save-load cycles (idempotency)
# ─────────────────────────────────────────────────────────────────────────────


def test_multiple_saves_idempotent(tmp_path: Path) -> None:
    session = AttachmentSession(tmp_path / "sessions", "idem")
    session.load()
    session.add(source="https://example.com/a.txt", name="a", fingerprint="fa", kind="url")
    session.save()
    session.save()  # save twice

    session2 = AttachmentSession(tmp_path / "sessions", "idem")
    attachments = session2.load()
    assert len(attachments) == 1
