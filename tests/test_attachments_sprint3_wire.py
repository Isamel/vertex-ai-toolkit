"""Sprint 3 tests: verify that settings.attachments.cache_dir / session_dir are
wired through _build_and_resolve_attachments and AttachmentCache.

When settings.attachments.cache_dir is set to a custom tmp path the cache writes
must go there — not to the default `.vaig/attachments-cache`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_adapter(source: str = "fake.zip") -> MagicMock:
    """Return a minimal mock adapter that satisfies _build_and_resolve_attachments."""
    adapter = MagicMock()
    adapter.spec.source = source
    adapter.spec.name = None
    adapter.spec.kind = "archive"
    adapter.list_files.return_value = iter([])
    adapter.fingerprint.return_value = "a" * 64
    return adapter


def _call_build(
    tmp_path: Path,
    *,
    cache_dir: Path | None,
    captured_dirs: list[Path],
) -> None:
    """Call _build_and_resolve_attachments with a fake adapter, capturing AttachmentCache init dirs."""
    from vaig.cli.commands.live import _build_and_resolve_attachments
    from vaig.core.attachment_cache import AttachmentCache

    fake = _fake_adapter()
    original_init = AttachmentCache.__init__

    def tracking_init(self: AttachmentCache, cd: Path, **kwargs: Any) -> None:
        captured_dirs.append(Path(cd))
        original_init(self, cd, **kwargs)

    # resolve_attachment is imported inside the function body; patch it at the source
    with (
        patch("vaig.core.attachment_adapter.resolve_attachment", return_value=fake),
        patch("vaig.cli.commands.live.resolve_attachment", return_value=fake, create=True),
        patch.object(AttachmentCache, "__init__", tracking_init),
    ):
        from vaig.core.config import AttachmentsConfig

        _build_and_resolve_attachments(
            base_config=AttachmentsConfig(cache_dir=cache_dir),
            attach_sources=["fake.zip"],
            attach_names=[],
            max_files=100,
            unlimited_files=False,
            max_depth=5,
            follow_symlinks=False,
            use_default_excludes=True,
            include_everything=False,
            max_bytes_absolute=10_000_000,
            allow_http=False,
            url_allowlist=[],
            session_id="dummy",
            cache_enabled=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: custom cache_dir is passed to AttachmentCache
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheDirWiring:
    def test_custom_cache_dir_is_used_not_default(self, tmp_path: Path) -> None:
        """When cache_dir is provided, AttachmentCache must be constructed with it."""
        custom_cache_dir = tmp_path / "my_custom_cache"
        captured_dirs: list[Path] = []

        _call_build(tmp_path, cache_dir=custom_cache_dir, captured_dirs=captured_dirs)

        assert any(d == custom_cache_dir for d in captured_dirs), (
            f"Expected AttachmentCache to be initialised with {custom_cache_dir}, but got: {captured_dirs}"
        )

    def test_default_cache_dir_used_when_none(self, tmp_path: Path) -> None:
        """When cache_dir is None, the fallback .vaig/attachments-cache is used."""
        captured_dirs: list[Path] = []

        _call_build(tmp_path, cache_dir=None, captured_dirs=captured_dirs)

        assert any(d == Path(".vaig/attachments-cache") for d in captured_dirs), (
            f"Expected fallback .vaig/attachments-cache, but got: {captured_dirs}"
        )
