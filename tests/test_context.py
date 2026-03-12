"""Tests for the context system — ContextBuilder, ContextBundle, loaders, and filters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.table import Table

from vaig.context.builder import ContextBuilder, ContextBundle
from vaig.context.filters import build_file_filter, is_binary_file, should_include_file
from vaig.context.loader import FileType, LoadedFile, classify_file, load_file
from vaig.core.config import ContextConfig, Settings


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def settings() -> Settings:
    """Settings configured with extensions and ignore patterns for testing."""
    return Settings(
        context=ContextConfig(
            max_file_size_mb=50,
            supported_extensions={
                "code": [".py", ".js", ".ts", ".tsx", ".jsx"],
                "text": [".md", ".txt", ".csv", ".log", ".rst"],
                "data": [".json", ".yaml", ".yml", ".xml", ".toml"],
                "etl": [".ktr", ".kjb", ".kdb"],
                "media": [".png", ".jpg", ".jpeg", ".gif", ".mp3", ".wav", ".mp4", ".pdf"],
            },
            ignore_patterns=["__pycache__", "*.pyc", ".git", "node_modules"],
        ),
    )


@pytest.fixture()
def builder(settings: Settings) -> ContextBuilder:
    """A fresh ContextBuilder for each test."""
    return ContextBuilder(settings)


@pytest.fixture()
def sample_text_loaded_file() -> LoadedFile:
    """A pre-built LoadedFile representing a text file."""
    return LoadedFile(
        path=Path("notes.txt"),
        file_type=FileType.TEXT,
        content="Hello, world!",
        size_bytes=13,
        mime_type="text/plain",
        token_estimate=3,
    )


@pytest.fixture()
def sample_binary_loaded_file() -> LoadedFile:
    """A pre-built LoadedFile representing a binary image."""
    mock_part = MagicMock()
    return LoadedFile(
        path=Path("image.png"),
        file_type=FileType.IMAGE,
        part=mock_part,
        size_bytes=1024,
        mime_type="image/png",
        token_estimate=256,
    )


# ── TestFileType ──────────────────────────────────────────


class TestFileType:
    """Verify all expected FileType enum members exist."""

    @pytest.mark.parametrize(
        "member,value",
        [
            ("TEXT", "text"),
            ("CODE", "code"),
            ("PDF", "pdf"),
            ("IMAGE", "image"),
            ("AUDIO", "audio"),
            ("VIDEO", "video"),
            ("ETL", "etl"),
        ],
    )
    def test_member_exists_with_correct_value(self, member: str, value: str) -> None:
        assert hasattr(FileType, member)
        assert FileType[member] == value

    def test_total_member_count(self) -> None:
        assert len(FileType) == 7


# ── TestClassifyFile ──────────────────────────────────────


class TestClassifyFile:
    """classify_file() maps extensions to the correct FileType."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            # Code
            ("main.py", FileType.CODE),
            ("app.js", FileType.CODE),
            ("index.ts", FileType.CODE),
            ("Component.tsx", FileType.CODE),
            ("style.css", FileType.CODE),
            ("query.sql", FileType.CODE),
            # Text
            ("README.txt", FileType.TEXT),
            ("CHANGELOG.md", FileType.TEXT),
            ("data.csv", FileType.TEXT),
            ("output.log", FileType.TEXT),
            # PDF
            ("report.pdf", FileType.PDF),
            # Image
            ("photo.png", FileType.IMAGE),
            ("photo.jpg", FileType.IMAGE),
            ("photo.jpeg", FileType.IMAGE),
            ("icon.gif", FileType.IMAGE),
            # Audio
            ("song.mp3", FileType.AUDIO),
            ("recording.wav", FileType.AUDIO),
            ("track.ogg", FileType.AUDIO),
            # Video
            ("clip.mp4", FileType.VIDEO),
            ("movie.avi", FileType.VIDEO),
            # ETL (Pentaho)
            ("transform.ktr", FileType.ETL),
            ("job.kjb", FileType.ETL),
            ("db.kdb", FileType.ETL),
            # Unknown → TEXT fallback
            ("data.xyz", FileType.TEXT),
            ("noext", FileType.TEXT),
        ],
    )
    def test_extension_classification(self, filename: str, expected: FileType) -> None:
        assert classify_file(Path(filename)) == expected

    def test_case_insensitive_extension(self) -> None:
        assert classify_file(Path("script.PY")) == FileType.CODE
        assert classify_file(Path("photo.PNG")) == FileType.IMAGE
        assert classify_file(Path("doc.PDF")) == FileType.PDF


# ── TestLoadedFile ────────────────────────────────────────


class TestLoadedFile:
    def test_basic_construction(self) -> None:
        lf = LoadedFile(
            path=Path("src/main.py"),
            file_type=FileType.CODE,
            content="print('hi')",
            size_bytes=11,
            token_estimate=2,
        )
        assert lf.path == Path("src/main.py")
        assert lf.file_type == FileType.CODE
        assert lf.content == "print('hi')"
        assert lf.size_bytes == 11
        assert lf.token_estimate == 2
        assert lf.part is None
        assert lf.mime_type == "text/plain"

    def test_display_name_returns_string_path(self) -> None:
        lf = LoadedFile(path=Path("src/utils/helpers.py"), file_type=FileType.CODE)
        assert lf.display_name == "src/utils/helpers.py"

    def test_defaults(self) -> None:
        lf = LoadedFile(path=Path("f.txt"), file_type=FileType.TEXT)
        assert lf.content is None
        assert lf.part is None
        assert lf.size_bytes == 0
        assert lf.mime_type == "text/plain"
        assert lf.token_estimate == 0


# ── TestLoadFile ──────────────────────────────────────────


class TestLoadFile:
    """load_file() reads real temp files and returns correct LoadedFile."""

    def test_load_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("Line one\nLine two\n", encoding="utf-8")

        result = load_file(f)

        assert result.file_type == FileType.TEXT
        assert result.content == "Line one\nLine two\n"
        assert result.part is None
        assert result.size_bytes == f.stat().st_size
        assert result.token_estimate == len("Line one\nLine two\n") // 4

    def test_load_code_file_wrapped_in_fences(self, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        source = "def hello():\n    return 'hi'\n"
        f.write_text(source, encoding="utf-8")

        result = load_file(f)

        assert result.file_type == FileType.CODE
        assert result.content is not None
        assert result.content.startswith("```py (app.py)\n")
        assert result.content.endswith("\n```")
        assert source in result.content

    def test_load_etl_file_wrapped_in_fences(self, tmp_path: Path) -> None:
        f = tmp_path / "transform.ktr"
        xml_content = "<transformation><step>Extract</step></transformation>"
        f.write_text(xml_content, encoding="utf-8")

        result = load_file(f)

        assert result.file_type == FileType.ETL
        assert result.content is not None
        assert result.content.startswith("```ktr (transform.ktr)\n")
        assert result.content.endswith("\n```")
        assert xml_content in result.content

    def test_token_estimate_for_text(self, tmp_path: Path) -> None:
        content = "a" * 400
        f = tmp_path / "big.txt"
        f.write_text(content, encoding="utf-8")

        result = load_file(f)

        assert result.token_estimate == 400 // 4  # 100

    @patch("vaig.context.loader.types.Part.from_bytes")
    def test_load_image_creates_binary_part(self, mock_from_bytes: MagicMock, tmp_path: Path) -> None:
        mock_part = MagicMock()
        mock_from_bytes.return_value = mock_part

        f = tmp_path / "image.png"
        # Write valid PNG magic bytes + some data
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = load_file(f)

        assert result.file_type == FileType.IMAGE
        assert result.content is None
        assert result.part is mock_part
        assert result.size_bytes == f.stat().st_size
        mock_from_bytes.assert_called_once()

    @patch("vaig.context.loader.types.Part.from_bytes")
    def test_load_pdf_creates_binary_part(self, mock_from_bytes: MagicMock, tmp_path: Path) -> None:
        mock_part = MagicMock()
        mock_from_bytes.return_value = mock_part

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")

        result = load_file(f)

        assert result.file_type == FileType.PDF
        assert result.content is None
        assert result.part is mock_part
        assert result.mime_type == "application/pdf"

    @patch("vaig.context.loader.types.Part.from_bytes")
    def test_load_audio_creates_binary_part(self, mock_from_bytes: MagicMock, tmp_path: Path) -> None:
        mock_part = MagicMock()
        mock_from_bytes.return_value = mock_part

        f = tmp_path / "clip.mp3"
        f.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)

        result = load_file(f)

        assert result.file_type == FileType.AUDIO
        assert result.part is mock_part
        assert result.content is None


# ── TestContextBundle ─────────────────────────────────────


class TestContextBundle:
    def test_add_file_updates_totals(
        self, sample_text_loaded_file: LoadedFile
    ) -> None:
        bundle = ContextBundle()
        bundle.add_file(sample_text_loaded_file)

        assert bundle.total_tokens_estimate == 3
        assert bundle.total_size_bytes == 13
        assert len(bundle.files) == 1

    def test_add_multiple_files_accumulates(self) -> None:
        bundle = ContextBundle()
        f1 = LoadedFile(path=Path("a.txt"), file_type=FileType.TEXT, content="aaa", size_bytes=3, token_estimate=1)
        f2 = LoadedFile(path=Path("b.txt"), file_type=FileType.TEXT, content="bbb", size_bytes=3, token_estimate=1)
        bundle.add_file(f1)
        bundle.add_file(f2)

        assert bundle.total_tokens_estimate == 2
        assert bundle.total_size_bytes == 6

    def test_file_count_property(self) -> None:
        bundle = ContextBundle()
        assert bundle.file_count == 0

        bundle.add_file(LoadedFile(path=Path("a.txt"), file_type=FileType.TEXT, size_bytes=1, token_estimate=0))
        assert bundle.file_count == 1

        bundle.add_file(LoadedFile(path=Path("b.txt"), file_type=FileType.TEXT, size_bytes=1, token_estimate=0))
        assert bundle.file_count == 2

    def test_to_context_string_text_files(self) -> None:
        bundle = ContextBundle()
        bundle.add_file(
            LoadedFile(path=Path("a.txt"), file_type=FileType.TEXT, content="AAA", size_bytes=3, token_estimate=0)
        )
        bundle.add_file(
            LoadedFile(path=Path("b.txt"), file_type=FileType.TEXT, content="BBB", size_bytes=3, token_estimate=0)
        )

        ctx = bundle.to_context_string()

        assert "## File: a.txt" in ctx
        assert "AAA" in ctx
        assert "## File: b.txt" in ctx
        assert "BBB" in ctx
        assert "---" in ctx  # separator between files

    def test_to_context_string_binary_shows_placeholder(
        self, sample_binary_loaded_file: LoadedFile
    ) -> None:
        bundle = ContextBundle()
        bundle.add_file(sample_binary_loaded_file)

        ctx = bundle.to_context_string()

        assert "## File: image.png" in ctx
        assert "binary" in ctx.lower()
        assert "image/png" in ctx
        assert "1024 bytes" in ctx

    def test_to_parts_text_concatenated(self) -> None:
        bundle = ContextBundle()
        bundle.add_file(
            LoadedFile(path=Path("a.txt"), file_type=FileType.TEXT, content="AAA", size_bytes=3, token_estimate=0)
        )
        bundle.add_file(
            LoadedFile(path=Path("b.txt"), file_type=FileType.TEXT, content="BBB", size_bytes=3, token_estimate=0)
        )

        parts = bundle.to_parts()

        # Two text files should be concatenated into a single string part
        assert len(parts) == 1
        assert isinstance(parts[0], str)
        assert "AAA" in parts[0]
        assert "BBB" in parts[0]

    def test_to_parts_includes_binary(self, sample_binary_loaded_file: LoadedFile) -> None:
        bundle = ContextBundle()
        # Add a text file first, then a binary
        bundle.add_file(
            LoadedFile(path=Path("readme.md"), file_type=FileType.TEXT, content="# README", size_bytes=8, token_estimate=2)
        )
        bundle.add_file(sample_binary_loaded_file)

        parts = bundle.to_parts()

        # Should be: text string, binary label string, binary part
        assert len(parts) == 3
        assert isinstance(parts[0], str)
        assert "# README" in parts[0]
        assert isinstance(parts[1], str)
        assert "image.png" in parts[1]
        # parts[2] is the mock Part

    def test_to_parts_binary_between_text(self, sample_binary_loaded_file: LoadedFile) -> None:
        bundle = ContextBundle()
        bundle.add_file(
            LoadedFile(path=Path("a.txt"), file_type=FileType.TEXT, content="AAA", size_bytes=3, token_estimate=0)
        )
        bundle.add_file(sample_binary_loaded_file)
        bundle.add_file(
            LoadedFile(path=Path("b.txt"), file_type=FileType.TEXT, content="BBB", size_bytes=3, token_estimate=0)
        )

        parts = bundle.to_parts()

        # text → flushed, binary label, binary part, remaining text
        assert len(parts) == 4
        assert isinstance(parts[0], str)
        assert "AAA" in parts[0]
        assert "image.png" in parts[1]
        assert isinstance(parts[3], str)
        assert "BBB" in parts[3]

    def test_clear_resets_everything(self, sample_text_loaded_file: LoadedFile) -> None:
        bundle = ContextBundle()
        bundle.add_file(sample_text_loaded_file)
        assert bundle.file_count == 1

        bundle.clear()

        assert bundle.file_count == 0
        assert bundle.total_tokens_estimate == 0
        assert bundle.total_size_bytes == 0
        assert bundle.files == []

    def test_summary_table_returns_rich_table(self, sample_text_loaded_file: LoadedFile) -> None:
        bundle = ContextBundle()
        bundle.add_file(sample_text_loaded_file)

        table = bundle.summary_table()

        assert isinstance(table, Table)


# ── TestContextBuilder ────────────────────────────────────


class TestContextBuilder:
    def test_add_file_single(self, builder: ContextBuilder, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")

        loaded = builder.add_file(f)

        assert loaded.file_type == FileType.TEXT
        assert loaded.content == "hello world"
        assert builder.bundle.file_count == 1

    def test_add_text_raw(self, builder: ContextBuilder) -> None:
        loaded = builder.add_text("raw error log output", label="error.log")

        assert loaded.file_type == FileType.TEXT
        assert loaded.content == "raw error log output"
        assert loaded.path == Path("error.log")
        assert builder.bundle.file_count == 1
        assert builder.bundle.total_tokens_estimate == len("raw error log output") // 4

    def test_add_directory_loads_supported_files(self, builder: ContextBuilder, tmp_path: Path) -> None:
        # Create files with supported extensions
        (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
        (tmp_path / "readme.txt").write_text("# Hello", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

        count = builder.add_directory(tmp_path)

        assert count == 3
        assert builder.bundle.file_count == 3

    def test_add_directory_skips_gitignored_patterns(self, builder: ContextBuilder, tmp_path: Path) -> None:
        # Create supported files
        (tmp_path / "main.py").write_text("ok", encoding="utf-8")

        # Create a __pycache__ directory (in ignore_patterns)
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_bytes(b"\x00" * 10)

        count = builder.add_directory(tmp_path)

        # Only main.py should be loaded (pyc not in supported_extensions, __pycache__ is ignored)
        assert count == 1
        assert builder.bundle.file_count == 1

    def test_add_directory_respects_gitignore_file(self, tmp_path: Path) -> None:
        """A .gitignore in the directory should cause matching files to be excluded."""
        settings = Settings(
            context=ContextConfig(
                supported_extensions={"code": [".py"], "text": [".txt", ".log"]},
                ignore_patterns=[],
            )
        )
        b = ContextBuilder(settings)

        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("pass", encoding="utf-8")
        (tmp_path / "debug.log").write_text("log line", encoding="utf-8")

        count = b.add_directory(tmp_path)

        assert count == 1  # only app.py
        names = [str(f.path) for f in b.bundle.files]
        assert any("app.py" in n for n in names)
        assert not any("debug.log" in n for n in names)

    def test_add_directory_recursive(self, builder: ContextBuilder, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        (tmp_path / "top.py").write_text("top", encoding="utf-8")
        (sub / "nested.py").write_text("nested", encoding="utf-8")

        count = builder.add_directory(tmp_path, recursive=True)

        assert count == 2

    def test_add_directory_non_recursive(self, builder: ContextBuilder, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        (tmp_path / "top.py").write_text("top", encoding="utf-8")
        (sub / "nested.py").write_text("nested", encoding="utf-8")

        count = builder.add_directory(tmp_path, recursive=False)

        assert count == 1  # only top.py

    def test_add_file_nonexistent_raises(self, builder: ContextBuilder) -> None:
        with pytest.raises(FileNotFoundError, match="Not a file"):
            builder.add_file("/nonexistent/path/file.txt")

    def test_add_directory_nonexistent_raises(self, builder: ContextBuilder) -> None:
        with pytest.raises(FileNotFoundError, match="Not a directory"):
            builder.add_directory("/nonexistent/directory")

    def test_clear_empties_bundle(self, builder: ContextBuilder, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("data", encoding="utf-8")
        builder.add_file(f)
        assert builder.bundle.file_count == 1

        builder.clear()

        assert builder.bundle.file_count == 0
        assert builder.bundle.total_tokens_estimate == 0

    def test_bundle_property_returns_context_bundle(self, builder: ContextBuilder) -> None:
        assert isinstance(builder.bundle, ContextBundle)

    def test_add_directory_uses_relative_paths(self, builder: ContextBuilder, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("pass", encoding="utf-8")

        builder.add_directory(tmp_path)

        # Paths in the bundle should be relative to the added directory
        paths = [str(f.path) for f in builder.bundle.files]
        assert any("src/main.py" in p for p in paths)
        # Should NOT contain the tmp_path prefix
        assert not any(str(tmp_path) in p for p in paths)


# ── TestFileFilters ───────────────────────────────────────


class TestIsBinaryFile:
    def test_png_is_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        assert is_binary_file(f) is True

    def test_jpeg_is_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        assert is_binary_file(f) is True

    def test_pdf_is_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        assert is_binary_file(f) is True

    def test_zip_is_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 50)
        assert is_binary_file(f) is True

    def test_elf_is_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "executable"
        f.write_bytes(b"\x7fELF" + b"\x00" * 50)
        assert is_binary_file(f) is True

    def test_null_bytes_detected_as_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "weird.dat"
        f.write_bytes(b"some text\x00more text")
        assert is_binary_file(f) is True

    def test_text_file_is_not_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("This is plain text content.\nSecond line.", encoding="utf-8")
        assert is_binary_file(f) is False

    def test_python_file_is_not_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "script.py"
        f.write_text("def main():\n    print('hello')\n", encoding="utf-8")
        assert is_binary_file(f) is False

    def test_empty_file_is_not_binary(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert is_binary_file(f) is False

    def test_nonexistent_file_returns_true(self) -> None:
        # OSError branch → returns True
        assert is_binary_file(Path("/nonexistent/file.bin")) is True


class TestShouldIncludeFile:
    def test_normal_supported_file_included(self, settings: Settings, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("pass", encoding="utf-8")
        spec = build_file_filter(settings, tmp_path)

        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is True

    def test_unsupported_extension_excluded(self, settings: Settings, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("stuff", encoding="utf-8")
        spec = build_file_filter(settings, tmp_path)

        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is False

    def test_gitignore_pattern_excludes_file(self, settings: Settings, tmp_path: Path) -> None:
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        f = cache_dir / "module.cpython-312.pyc"
        f.write_bytes(b"\x00")
        spec = build_file_filter(settings, tmp_path)

        # __pycache__ is in ignore_patterns, and .pyc also matches *.pyc
        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is False

    def test_git_directory_excluded(self, settings: Settings, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        f = git_dir / "config"
        f.write_text("[core]\n", encoding="utf-8")
        spec = build_file_filter(settings, tmp_path)

        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is False

    def test_media_file_allowed_even_if_binary(self, settings: Settings, tmp_path: Path) -> None:
        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        spec = build_file_filter(settings, tmp_path)

        # .png is in media extensions → should be included despite being binary
        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is True

    def test_oversized_file_excluded(self, tmp_path: Path) -> None:
        small_settings = Settings(
            context=ContextConfig(
                max_file_size_mb=0,  # 0 MB limit → everything is "too large"
                supported_extensions={"code": [".py"]},
                ignore_patterns=[],
            )
        )
        f = tmp_path / "big.py"
        f.write_text("x" * 100, encoding="utf-8")
        spec = build_file_filter(small_settings, tmp_path)

        assert should_include_file(f, settings=small_settings, spec=spec, root_dir=tmp_path) is False

    def test_binary_non_media_excluded(self, settings: Settings, tmp_path: Path) -> None:
        """A .py file that contains null bytes should be rejected as binary."""
        f = tmp_path / "corrupt.py"
        f.write_bytes(b"import os\x00\x00\x00")
        spec = build_file_filter(settings, tmp_path)

        assert should_include_file(f, settings=settings, spec=spec, root_dir=tmp_path) is False


class TestBuildFileFilter:
    def test_loads_gitignore_patterns(self, settings: Settings, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\nbuild/\n", encoding="utf-8")

        spec = build_file_filter(settings, tmp_path)

        assert spec.match_file("app.log") is True
        assert spec.match_file("build/output.js") is True
        assert spec.match_file("main.py") is False

    def test_includes_config_ignore_patterns(self, settings: Settings, tmp_path: Path) -> None:
        spec = build_file_filter(settings, tmp_path)

        # __pycache__ and *.pyc are in settings.context.ignore_patterns
        assert spec.match_file("__pycache__/mod.pyc") is True
        assert spec.match_file("src/main.py") is False

    def test_no_gitignore_still_works(self, settings: Settings, tmp_path: Path) -> None:
        # No .gitignore file exists — should still build from config patterns
        spec = build_file_filter(settings, tmp_path)

        assert spec.match_file("node_modules/pkg/index.js") is True
        assert spec.match_file("src/app.py") is False
