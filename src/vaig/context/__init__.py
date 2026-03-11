"""Context package — file ingestion, loading, and filtering."""

from vaig.context.builder import ContextBuilder, ContextBundle
from vaig.context.filters import build_file_filter, is_binary_file, should_include_file
from vaig.context.loader import FileType, LoadedFile, load_file

__all__ = [
    "ContextBuilder",
    "ContextBundle",
    "FileType",
    "LoadedFile",
    "build_file_filter",
    "is_binary_file",
    "load_file",
    "should_include_file",
]
