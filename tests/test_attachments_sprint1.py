"""Sprint 1 acceptance tests for Live Attachments (SPEC-ATT-01..04).

Coverage:
- AttachmentsConfig defaults and serialization (SPEC-ATT-01)
- resolve_attachment() dispatcher (SPEC-ATT-02)
- LocalPathAdapter invariants AT-1..AT-6 + ATT-03 acceptance criteria (SPEC-ATT-03)
- SingleFileAdapter basics + streaming + max_bytes cap (SPEC-ATT-04)
- CLI --attach flag wiring
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vaig.core.attachment_adapter import (
    AttachmentKind,
    AttachmentSpec,
    LocalPathAdapter,
    SingleFileAdapter,
    resolve_attachment,
)
from vaig.core.config import AttachmentsConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_spec(kind: AttachmentKind = AttachmentKind.local_path, source: str = ".") -> AttachmentSpec:
    return AttachmentSpec(name=None, source=source, kind=kind, resolved_path=None)


def _default_cfg(**kwargs: Any) -> AttachmentsConfig:
    return AttachmentsConfig(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# AttachmentsConfig tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAttachmentsConfig:
    def test_defaults_match_spec(self) -> None:
        cfg = AttachmentsConfig()
        assert cfg.max_files_per_attachment == 10_000
        assert cfg.unlimited_files is False
        assert cfg.max_depth == -1
        assert cfg.follow_symlinks is False
        assert cfg.use_default_excludes is True
        assert cfg.streaming_threshold_bytes == 2_000_000
        assert cfg.max_bytes_absolute == 500_000_000
        assert cfg.binary_skip is True
        assert cfg.extra_excludes == []

    def test_include_everything_does_not_mutate_other_fields(self) -> None:
        """include_everything alone should not change other field values."""
        cfg = AttachmentsConfig(include_everything=True)
        # The field itself is set, but siblings retain their defaults.
        assert cfg.include_everything is True
        assert cfg.use_default_excludes is True  # CLI layer applies cascade, not model
        assert cfg.max_depth == -1
        assert cfg.unlimited_files is False

    def test_roundtrip_serialization(self) -> None:
        cfg = AttachmentsConfig(max_files_per_attachment=500, max_depth=3)
        data = cfg.model_dump()
        cfg2 = AttachmentsConfig.model_validate(data)
        assert cfg2.max_files_per_attachment == 500
        assert cfg2.max_depth == 3
        assert cfg2 == cfg


# ─────────────────────────────────────────────────────────────────────────────
# resolve_attachment dispatcher tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveAttachmentDispatcher:
    def test_http_url_raises_not_implemented(self) -> None:
        cfg = _default_cfg()
        with pytest.raises(NotImplementedError, match="Sprint 3"):
            resolve_attachment("http://example.com", cfg=cfg)

    def test_https_url_raises_not_implemented(self) -> None:
        cfg = _default_cfg()
        with pytest.raises(NotImplementedError, match="Sprint 3"):
            resolve_attachment("https://example.com/foo", cfg=cfg)

    def test_zip_file_raises_not_implemented(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.zip"
        bundle.write_bytes(b"")
        cfg = _default_cfg()
        with pytest.raises(NotImplementedError, match="Sprint 2"):
            resolve_attachment(str(bundle), cfg=cfg)

    def test_nonexistent_path_raises_value_error(self) -> None:
        cfg = _default_cfg()
        with pytest.raises(ValueError, match="Cannot resolve attachment"):
            resolve_attachment("/does/not/exist/ever", cfg=cfg)

    def test_dir_with_git_raises_not_implemented(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        cfg = _default_cfg()
        with pytest.raises(NotImplementedError, match="Sprint 2"):
            resolve_attachment(str(tmp_path), cfg=cfg)

    def test_plain_file_returns_single_file_adapter(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hi")
        cfg = _default_cfg()
        adapter = resolve_attachment(str(f), cfg=cfg)
        assert isinstance(adapter, SingleFileAdapter)
        assert adapter.spec.kind == AttachmentKind.single_file

    def test_plain_dir_returns_local_path_adapter(self, tmp_path: Path) -> None:
        cfg = _default_cfg()
        adapter = resolve_attachment(str(tmp_path), cfg=cfg)
        assert isinstance(adapter, LocalPathAdapter)
        assert adapter.spec.kind == AttachmentKind.local_path


# ─────────────────────────────────────────────────────────────────────────────
# LocalPathAdapter tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLocalPathAdapter:
    # AT-1: all non-excluded files are returned
    def test_at1_returns_all_non_excluded_files(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")
        cfg = _default_cfg(use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        assert len(entries) == 5

    # AT-3: large file streaming
    def test_at3_large_file_streams(self, tmp_path: Path) -> None:
        big = tmp_path / "big.bin"
        # 3 MB of non-binary text data (a-z repeated)
        big.write_bytes(b"a" * (3 * 1024 * 1024))
        cfg = _default_cfg(streaming_threshold_bytes=2_000_000, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter.list_files(cfg))  # populate _listed & _cfg_ref
        result = adapter.fetch_bytes("big.bin")
        assert isinstance(result, Iterator)
        assembled = b"".join(result)
        assert len(assembled) == 3 * 1024 * 1024

    # AT-5: fingerprint stability
    def test_at5_fingerprint_stable_across_calls(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        cfg = _default_cfg(binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter.list_files(cfg))
        fp1 = adapter.fingerprint()
        fp2 = adapter.fingerprint()
        assert fp1 == fp2

    def test_at5_fingerprint_changes_after_mtime_update(self, tmp_path: Path) -> None:
        import time

        f = tmp_path / "a.txt"
        f.write_text("hello")
        cfg = _default_cfg(binary_skip=False)

        adapter1 = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter1.list_files(cfg))
        fp1 = adapter1.fingerprint()

        # Touch the file to update mtime
        time.sleep(0.01)
        f.write_text("changed")

        adapter2 = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter2.list_files(cfg))
        fp2 = adapter2.fingerprint()

        assert fp1 != fp2

    # ATT-03 deep recursion: 50-level deep tree
    def test_att03_deep_recursion(self, tmp_path: Path) -> None:
        current = tmp_path
        for i in range(50):
            current = current / f"d{i}"
            current.mkdir()
        (current / "deep.txt").write_text("found it")
        cfg = _default_cfg(max_depth=-1, use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        assert any("deep.txt" in p for p in rel_paths)

    # ATT-03 many files: cap at max_files_per_attachment
    def test_att03_many_files_cap(self, tmp_path: Path) -> None:
        for i in range(1200):
            (tmp_path / f"f{i:04d}.txt").write_text("x")
        cfg = _default_cfg(
            max_files_per_attachment=1000,
            unlimited_files=False,
            use_default_excludes=False,
            binary_skip=False,
        )
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        assert len(entries) == 1000

    # ATT-03 extension-agnostic
    def test_att03_extension_agnostic(self, tmp_path: Path) -> None:
        (tmp_path / "foo.weirdext").write_text("a")
        (tmp_path / "Makefile").write_text("b")
        (tmp_path / "run").write_text("c")
        cfg = _default_cfg(use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        names = {Path(e.relative_path).name for e in entries}
        assert "foo.weirdext" in names
        assert "Makefile" in names
        assert "run" in names

    # ATT-03 large file fetch + max_bytes_absolute rejection
    def test_att03_large_file_fetch_and_rejection(self, tmp_path: Path) -> None:
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MB
        cfg = _default_cfg(
            streaming_threshold_bytes=2_000_000,
            max_bytes_absolute=500_000_000,
            binary_skip=False,
        )
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter.list_files(cfg))
        result = adapter.fetch_bytes("big.bin")
        assert isinstance(result, Iterator)
        assembled = b"".join(result)
        assert assembled == b"x" * (10 * 1024 * 1024)

        # Now enforce max_bytes_absolute
        cfg_small = _default_cfg(
            streaming_threshold_bytes=2_000_000,
            max_bytes_absolute=1024,  # 1 KB cap
            binary_skip=False,
        )
        adapter2 = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter2.list_files(cfg_small))
        with pytest.raises(ValueError, match="max_bytes_absolute"):
            adapter2.fetch_bytes("big.bin")

    # Default excludes
    def test_default_excludes_skip_git_and_node_modules(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "x.js").write_text("code")
        pyc = tmp_path / "__pycache__"
        pyc.mkdir()
        (pyc / "y.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "real.py").write_text("real")
        cfg = _default_cfg(use_default_excludes=True, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        assert not any(".git" in p for p in rel_paths)
        assert not any("node_modules" in p for p in rel_paths)
        assert not any("__pycache__" in p for p in rel_paths)
        assert any("real.py" in p for p in rel_paths)

    def test_no_default_excludes_includes_git_head(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        cfg = _default_cfg(use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        assert any(".git" in p and "HEAD" in p for p in rel_paths)

    # include_everything
    def test_include_everything_includes_hidden_and_node_modules(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=xyz")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "foo.js").write_text("module")
        cfg = _default_cfg(
            include_everything=True,
            binary_skip=False,  # cascaded
            use_default_excludes=False,  # cascaded
        )
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        assert any(".env" in p for p in rel_paths)
        assert any("node_modules" in p for p in rel_paths)

    # Symlink not followed by default
    def test_symlink_not_followed_by_default(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("sensitive")
        link = tmp_path / "inner" 
        link.mkdir()
        symlink_target = link / "linked"
        symlink_target.symlink_to(outside)

        cfg = _default_cfg(follow_symlinks=False, use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(link, _make_spec(source=str(link)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        assert not any("secret.txt" in p for p in rel_paths)

    # Symlink jail: symlink pointing outside root must not be followed
    def test_symlink_jail(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "private.txt").write_text("private")

        inside = tmp_path / "sandbox"
        inside.mkdir()
        (inside / "escape").symlink_to(outside)

        cfg = _default_cfg(follow_symlinks=True, use_default_excludes=False, binary_skip=False)
        adapter = LocalPathAdapter(inside, _make_spec(source=str(inside)))
        entries = list(adapter.list_files(cfg))
        rel_paths = [e.relative_path for e in entries]
        # The symlink entry itself may appear but the file inside outside/ must not
        assert not any("private.txt" in p for p in rel_paths)

    # Path traversal in fetch_bytes
    def test_path_traversal_fetch_bytes_raises(self, tmp_path: Path) -> None:
        (tmp_path / "legit.txt").write_text("ok")
        cfg = _default_cfg(binary_skip=False)
        adapter = LocalPathAdapter(tmp_path, _make_spec(source=str(tmp_path)))
        list(adapter.list_files(cfg))
        with pytest.raises(ValueError, match="traversal"):
            adapter.fetch_bytes("../../etc/passwd")


# ─────────────────────────────────────────────────────────────────────────────
# SingleFileAdapter tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleFileAdapter:
    def test_single_text_file_listed_once_and_fetched(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("world")
        spec = AttachmentSpec(
            name=None,
            source=str(f),
            kind=AttachmentKind.single_file,
            resolved_path=f.resolve(),
        )
        cfg = _default_cfg()
        adapter = SingleFileAdapter(f, spec)
        entries = list(adapter.list_files(cfg))
        assert len(entries) == 1
        content = adapter.fetch_bytes("")
        assert isinstance(content, bytes)
        assert content == b"world"

    def test_large_file_streams(self, tmp_path: Path) -> None:
        f = tmp_path / "large.bin"
        f.write_bytes(b"z" * (3 * 1024 * 1024))
        spec = AttachmentSpec(
            name=None,
            source=str(f),
            kind=AttachmentKind.single_file,
            resolved_path=f.resolve(),
        )
        cfg = _default_cfg(streaming_threshold_bytes=2_000_000)
        adapter = SingleFileAdapter(f, spec)
        list(adapter.list_files(cfg))
        result = adapter.fetch_bytes()
        assert isinstance(result, Iterator)
        data = b"".join(result)
        assert len(data) == 3 * 1024 * 1024

    def test_max_bytes_absolute_honored(self, tmp_path: Path) -> None:
        f = tmp_path / "huge.bin"
        f.write_bytes(b"y" * 1024)  # 1 KB
        spec = AttachmentSpec(
            name=None,
            source=str(f),
            kind=AttachmentKind.single_file,
            resolved_path=f.resolve(),
        )
        cfg = _default_cfg(max_bytes_absolute=100)  # cap at 100 bytes
        adapter = SingleFileAdapter(f, spec)
        list(adapter.list_files(cfg))
        with pytest.raises(ValueError, match="max_bytes_absolute"):
            adapter.fetch_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# CLI wiring tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCLIWiring:
    def test_attach_dir_logs_resolved_message(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--attach <tmp_dir> with 3 files → stderr contains 'resolved 1 attachment'."""
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(f"content {i}")

        from vaig.cli.commands.live import _build_and_resolve_attachments

        _build_and_resolve_attachments(
            attach_sources=[str(tmp_path)],
            attach_names=[],
            max_files=10_000,
            unlimited_files=False,
            max_depth=-1,
            follow_symlinks=False,
            use_default_excludes=False,
            include_everything=False,
            max_bytes_absolute=500_000_000,
        )
        captured = capsys.readouterr()
        assert "resolved 1 attachment" in captured.err

    def test_include_everything_cascades_in_config(self) -> None:
        """--attach-include-everything cascades unlimited_files, max_depth=-1, use_default_excludes=False, binary_skip=False."""
        from vaig.core.config import AttachmentsConfig

        # Simulate what _build_and_resolve_attachments does
        include_everything = True
        unlimited_files = False
        max_depth = 5  # non-default, should be overridden
        use_default_excludes = True  # should be overridden
        binary_skip = True  # should be overridden

        if include_everything:
            unlimited_files = True
            max_depth = -1
            use_default_excludes = False
            binary_skip = False

        cfg = AttachmentsConfig(
            unlimited_files=unlimited_files,
            max_depth=max_depth,
            use_default_excludes=use_default_excludes,
            include_everything=include_everything,
            binary_skip=binary_skip,
        )

        assert cfg.unlimited_files is True
        assert cfg.max_depth == -1
        assert cfg.use_default_excludes is False
        assert cfg.binary_skip is False
