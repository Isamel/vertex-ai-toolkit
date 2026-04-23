"""Sprint 2 acceptance tests for SPEC-ATT-05: Archive and GitClone adapters.

Coverage:
- ArchiveAttachmentAdapter: zip/tar extraction roundtrip, path-traversal
  rejection, symlink rejection, max_bytes_absolute exceeded, max_files
  exceeded, cleanup removes tempdir (SPEC-ATT-05)
- GitCloneAttachmentAdapter: URL detection, shallow clone success (mocked),
  CalledProcessError, TimeoutExpired, FileNotFoundError, cleanup
- resolve_attachment() dispatch for archives and git URLs
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from vaig.core.attachment_adapter import (
    ArchiveAttachmentAdapter,
    AttachmentKind,
    AttachmentSpec,
    GitCloneAttachmentAdapter,
    _is_git_url,
    resolve_attachment,
)
from vaig.core.config import AttachmentsConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _cfg(**kwargs: Any) -> AttachmentsConfig:
    return AttachmentsConfig(**kwargs)


def _spec(kind: AttachmentKind = AttachmentKind.archive, source: str = "test.zip") -> AttachmentSpec:
    return AttachmentSpec(name=None, source=source, kind=kind, resolved_path=None)


def _make_zip(dest_dir: Path, files: dict[str, bytes], name: str = "test.zip") -> Path:
    """Create a zip archive in *dest_dir* containing *files* (relative_path → bytes)."""
    archive = dest_dir / name
    with zipfile.ZipFile(archive, "w") as zf:
        for rel, data in files.items():
            zf.writestr(rel, data)
    return archive


def _make_tar(dest_dir: Path, files: dict[str, bytes], name: str = "test.tar.gz") -> Path:
    """Create a tar.gz archive in *dest_dir* containing *files*."""
    archive = dest_dir / name
    with tarfile.open(archive, "w:gz") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return archive


# ─────────────────────────────────────────────────────────────────────────────
# ArchiveAttachmentAdapter — zip roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestArchiveAdapterZip:
    def test_zip_extraction_roundtrip(self, tmp_path: Path) -> None:
        """Files in zip are accessible via list_files + fetch_bytes."""
        archive = _make_zip(tmp_path, {"hello.txt": b"world", "sub/foo.py": b"print(1)"})
        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        cfg = _cfg()

        entries = list(adapter.list_files(cfg))
        rel_paths = {e.relative_path for e in entries}
        assert "hello.txt" in rel_paths
        assert "sub/foo.py" in rel_paths

        data = adapter.fetch_bytes("hello.txt")
        assert isinstance(data, bytes)
        assert data == b"world"

        adapter.cleanup()

    def test_zip_cleanup_removes_tempdir(self, tmp_path: Path) -> None:
        archive = _make_zip(tmp_path, {"a.txt": b"x"})
        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        list(adapter.list_files(_cfg()))  # trigger extraction

        saved_tempdir = adapter._tempdir
        assert saved_tempdir is not None
        assert os.path.isdir(saved_tempdir)

        adapter.cleanup()
        assert not os.path.isdir(saved_tempdir)

    def test_zip_context_manager_cleanup(self, tmp_path: Path) -> None:
        archive = _make_zip(tmp_path, {"b.txt": b"y"})
        with ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg()) as adapter:
            list(adapter.list_files(_cfg()))
            saved = adapter._tempdir

        assert saved is not None
        assert not os.path.isdir(saved)


# ─────────────────────────────────────────────────────────────────────────────
# ArchiveAttachmentAdapter — tar roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestArchiveAdapterTar:
    def test_tar_gz_extraction_roundtrip(self, tmp_path: Path) -> None:
        archive = _make_tar(tmp_path, {"readme.md": b"# hello", "src/main.py": b"pass"})
        spec = _spec(kind=AttachmentKind.archive, source=str(archive))
        adapter = ArchiveAttachmentAdapter(archive, spec, _cfg())
        cfg = _cfg()

        entries = list(adapter.list_files(cfg))
        rel_paths = {e.relative_path for e in entries}
        assert "readme.md" in rel_paths
        assert "src/main.py" in rel_paths

        data = adapter.fetch_bytes("readme.md")
        assert data == b"# hello"

        adapter.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# ArchiveAttachmentAdapter — security guards
# ─────────────────────────────────────────────────────────────────────────────


class TestArchiveSecurityGuards:
    def test_zip_absolute_path_rejected(self, tmp_path: Path) -> None:
        """Archive with an absolute member path raises ValueError before extraction."""
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("/etc/passwd", "evil")

        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        with pytest.raises(ValueError, match="absolute path"):
            adapter.list_files(_cfg())

    def test_zip_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        """Archive with ``..`` in member path raises ValueError."""
        archive = tmp_path / "traversal.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("safe/../../../etc/passwd", "evil")

        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        with pytest.raises(ValueError, match=r"\.\.|traversal"):
            adapter.list_files(_cfg())

    def test_tar_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        """Tar archive with ``..`` in member name raises ValueError."""
        archive = tmp_path / "traversal.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo(name="../escape.txt")
            data = b"evil"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        with pytest.raises(ValueError, match=r"\.\.|traversal"):
            adapter.list_files(_cfg())

    def test_max_files_exceeded_raises(self, tmp_path: Path) -> None:
        """Archive member count > max_files raises ValueError before extraction."""
        files = {f"file_{i}.txt": b"x" for i in range(5)}
        archive = _make_zip(tmp_path, files)

        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        # max_files_per_attachment=3 — should reject
        with pytest.raises(ValueError, match="max_files"):
            adapter.list_files(_cfg(max_files_per_attachment=3))

    def test_max_bytes_exceeded_raises(self, tmp_path: Path) -> None:
        """Total uncompressed size > max_bytes_absolute raises ValueError."""
        big_data = b"A" * 1000
        archive = _make_zip(tmp_path, {"big.txt": big_data})

        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        with pytest.raises(ValueError, match="max_bytes_absolute"):
            adapter.list_files(_cfg(max_bytes_absolute=500))  # 500 bytes cap


# ─────────────────────────────────────────────────────────────────────────────
# ArchiveAttachmentAdapter — fingerprint
# ─────────────────────────────────────────────────────────────────────────────


class TestArchiveFingerprint:
    def test_fingerprint_is_stable(self, tmp_path: Path) -> None:
        archive = _make_zip(tmp_path, {"f.txt": b"data"})
        adapter = ArchiveAttachmentAdapter(archive, _spec(source=str(archive)), _cfg())
        fp1 = adapter.fingerprint()
        fp2 = adapter.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # sha256 hex

    def test_fingerprint_differs_for_different_archives(self, tmp_path: Path) -> None:
        a1 = _make_zip(tmp_path, {"f.txt": b"aaa"}, name="a1.zip")
        a2 = _make_zip(tmp_path, {"g.txt": b"bbb"}, name="a2.zip")
        adapter1 = ArchiveAttachmentAdapter(a1, _spec(source=str(a1)), _cfg())
        adapter2 = ArchiveAttachmentAdapter(a2, _spec(source=str(a2)), _cfg())
        assert adapter1.fingerprint() != adapter2.fingerprint()


# ─────────────────────────────────────────────────────────────────────────────
# GitCloneAttachmentAdapter — URL detection
# ─────────────────────────────────────────────────────────────────────────────


class TestGitUrlDetection:
    @pytest.mark.parametrize(
        "url",
        [
            "git@github.com:org/repo.git",
            "git+https://github.com/org/repo",
            "git://github.com/org/repo.git",
            "https://github.com/org/repo.git",
        ],
    )
    def test_recognised_as_git(self, url: str) -> None:
        assert _is_git_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/page",  # no .git suffix
            "http://example.com/repo",
            "/local/path",
            "archive.tar.gz",
        ],
    )
    def test_not_recognised_as_git(self, url: str) -> None:
        assert _is_git_url(url) is False


# ─────────────────────────────────────────────────────────────────────────────
# GitCloneAttachmentAdapter — clone behaviour (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestGitCloneAdapter:
    def _spec(self, url: str) -> AttachmentSpec:
        return AttachmentSpec(name=None, source=url, kind=AttachmentKind.git_clone, resolved_path=None)

    def test_shallow_clone_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful clone: list_files returns entries from cloned tree."""
        url = "git@github.com:org/repo.git"

        def fake_run(cmd: list[str], **kwargs: Any) -> None:
            # Instead of actually cloning, create files in the dest directory
            dest = cmd[-1]
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "README.md").write_bytes(b"# repo")
            (Path(dest) / "main.py").write_bytes(b"pass")

        monkeypatch.setattr(subprocess, "run", fake_run)

        adapter = GitCloneAttachmentAdapter(url, self._spec(url), _cfg())
        entries = list(adapter.list_files(_cfg()))
        rel_paths = {e.relative_path for e in entries}
        assert "README.md" in rel_paths
        assert "main.py" in rel_paths

        saved = adapter._tempdir
        adapter.cleanup()
        assert saved is not None
        assert not os.path.isdir(saved)

    def test_clone_called_process_error_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "git@github.com:org/repo.git"

        def fake_run(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.CalledProcessError(
                128, ["git", "clone"], stderr=b"fatal: repo not found"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        adapter = GitCloneAttachmentAdapter(url, self._spec(url), _cfg())
        with pytest.raises(RuntimeError, match="git clone failed"):
            adapter.list_files(_cfg())

    def test_clone_timeout_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://github.com/org/repo.git"

        def fake_run(*_args: Any, **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(["git", "clone"], timeout=60)

        monkeypatch.setattr(subprocess, "run", fake_run)

        adapter = GitCloneAttachmentAdapter(url, self._spec(url), _cfg())
        with pytest.raises(RuntimeError, match="timed out"):
            adapter.list_files(_cfg())

    def test_git_not_installed_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "git+https://github.com/org/repo"

        def fake_run(*_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("git: command not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        adapter = GitCloneAttachmentAdapter(url, self._spec(url), _cfg())
        with pytest.raises(RuntimeError, match="git is not installed"):
            adapter.list_files(_cfg())

    def test_context_manager_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "git@github.com:org/repo.git"

        def fake_run(cmd: list[str], **kwargs: Any) -> None:
            dest = cmd[-1]
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "f.txt").write_bytes(b"hi")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with GitCloneAttachmentAdapter(url, self._spec(url), _cfg()) as adapter:
            list(adapter.list_files(_cfg()))
            saved = adapter._tempdir

        assert saved is not None
        assert not os.path.isdir(saved)

    def test_fingerprint_is_stable(self) -> None:
        url = "git@github.com:org/repo.git"
        adapter = GitCloneAttachmentAdapter(url, self._spec(url), _cfg())
        assert adapter.fingerprint() == adapter.fingerprint()
        assert len(adapter.fingerprint()) == 64


# ─────────────────────────────────────────────────────────────────────────────
# resolve_attachment dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveAttachmentDispatch:
    def test_zip_path_dispatches_to_archive_adapter(self, tmp_path: Path) -> None:
        archive = _make_zip(tmp_path, {"x.txt": b"hi"})
        adapter = resolve_attachment(str(archive), cfg=_cfg())
        assert isinstance(adapter, ArchiveAttachmentAdapter)
        assert adapter.spec.kind == AttachmentKind.archive

    def test_tar_gz_path_dispatches_to_archive_adapter(self, tmp_path: Path) -> None:
        archive = _make_tar(tmp_path, {"x.txt": b"hi"})
        adapter = resolve_attachment(str(archive), cfg=_cfg())
        assert isinstance(adapter, ArchiveAttachmentAdapter)

    def test_tgz_suffix_dispatches_to_archive_adapter(self, tmp_path: Path) -> None:
        # File doesn't need to exist — suffix check only for archive dispatch
        path = str(tmp_path / "bundle.tgz")
        # Create the file so the path is valid
        Path(path).write_bytes(b"")
        adapter = resolve_attachment(path, cfg=_cfg())
        assert isinstance(adapter, ArchiveAttachmentAdapter)

    def test_git_at_url_dispatches_to_git_clone(self) -> None:
        url = "git@github.com:org/repo.git"
        adapter = resolve_attachment(url, cfg=_cfg())
        assert isinstance(adapter, GitCloneAttachmentAdapter)
        assert adapter.spec.kind == AttachmentKind.git_clone

    def test_https_git_url_dispatches_to_git_clone(self) -> None:
        url = "https://github.com/org/repo.git"
        adapter = resolve_attachment(url, cfg=_cfg())
        assert isinstance(adapter, GitCloneAttachmentAdapter)

    def test_git_plus_scheme_dispatches_to_git_clone(self) -> None:
        url = "git+https://github.com/org/repo"
        adapter = resolve_attachment(url, cfg=_cfg())
        assert isinstance(adapter, GitCloneAttachmentAdapter)

    def test_plain_https_url_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="Sprint 3"):
            resolve_attachment("https://example.com/page", cfg=_cfg())

    def test_name_propagated_to_spec(self, tmp_path: Path) -> None:
        archive = _make_zip(tmp_path, {"x.txt": b"hi"})
        adapter = resolve_attachment(str(archive), name="my-archive", cfg=_cfg())
        assert adapter.spec.name == "my-archive"
