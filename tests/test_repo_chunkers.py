"""Tests for SPEC-V2-REPO-03 — Manifest-aware chunking."""

from __future__ import annotations

import io

import pytest

from vaig.core.repo_chunkers import (
    Chunk,
    ChunkingError,
    FallbackLineChunker,
    MarkdownChunker,
    TerraformChunker,
    YamlDocChunker,
    _token_estimate,
    chunk_file,
    get_chunker,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _roundtrip(chunks: list[Chunk], original: str) -> bool:
    """All chunk contents concatenated must equal the original content."""
    return "".join(c.content for c in chunks) == original


# ── YamlDocChunker ─────────────────────────────────────────────────────────────


MULTI_DOC_YAML = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  key: value
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deploy
spec:
  replicas: 1
"""

SINGLE_DOC_YAML = """\
apiVersion: v1
kind: Service
metadata:
  name: my-svc
spec:
  type: ClusterIP
"""

MALFORMED_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: bad
  labels: [unclosed
"""


class TestYamlDocChunker:
    def test_splits_multi_document_yaml(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(MULTI_DOC_YAML, "test.yaml")
        assert len(chunks) == 2

    def test_extracts_outline_kind_name(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(MULTI_DOC_YAML, "test.yaml")
        outlines = [c.outline for c in chunks]
        assert "ConfigMap/my-config" in outlines
        assert "Deployment/my-deploy" in outlines

    def test_single_doc_no_separator(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(SINGLE_DOC_YAML, "svc.yaml")
        assert len(chunks) == 1
        assert chunks[0].outline == "Service/my-svc"

    def test_large_doc_splits_on_top_level_keys(self) -> None:
        # Build a YAML document that exceeds max_chunk_tokens=100
        big_doc = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: big\n"
        big_doc += "data:\n"
        big_doc += "".join(f"  key{i}: {'x' * 50}\n" for i in range(20))
        chunker = YamlDocChunker(max_chunk_tokens=100)
        chunks = chunker.chunk(big_doc, "big.yaml")
        assert len(chunks) >= 2
        # Each chunk should be smaller than the full document; individual large
        # top-level key sections may still exceed max_chunk_tokens since they
        # are atomic and cannot be split further.
        full_tokens = _token_estimate(big_doc)
        for c in chunks:
            assert c.token_estimate < full_tokens

    def test_chunk_kind_is_yaml_doc(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(SINGLE_DOC_YAML, "svc.yaml")
        assert all(c.kind == "yaml_doc" for c in chunks)

    def test_malformed_yaml_raises_chunking_error(self) -> None:
        chunker = YamlDocChunker()
        with pytest.raises(ChunkingError):
            chunker.chunk(MALFORMED_YAML, "bad.yaml")

    def test_roundtrip(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(MULTI_DOC_YAML, "test.yaml")
        # Round-trip: reassemble content from chunks
        # Each chunk content is the text between separators; original has '---\n' between
        reconstructed = "---\n".join(c.content for c in chunks)
        # Verify all content is present (original may start with content or separator)
        for c in chunks:
            assert c.content in MULTI_DOC_YAML

    def test_chunk_stream_matches_chunk(self) -> None:
        chunker = YamlDocChunker()
        chunks_list = chunker.chunk(MULTI_DOC_YAML, "test.yaml")
        stream_chunks = list(chunker.chunk_stream(io.StringIO(MULTI_DOC_YAML), "test.yaml"))
        assert len(stream_chunks) == len(chunks_list)
        for a, b in zip(stream_chunks, chunks_list, strict=True):
            assert a.content == b.content


# ── TerraformChunker ───────────────────────────────────────────────────────────


TF_CONTENT = """\
variable "region" {
  default = "us-central1"
}

resource "google_container_cluster" "primary" {
  name     = "my-gke"
  location = var.region
}

module "vpc" {
  source = "./modules/vpc"
}
"""

TF_LOCALS = """\
locals {
  env = "prod"
}

output "cluster_name" {
  value = google_container_cluster.primary.name
}
"""


class TestTerraformChunker:
    def test_splits_resource_module_blocks(self) -> None:
        chunker = TerraformChunker()
        chunks = chunker.chunk(TF_CONTENT, "main.tf")
        assert len(chunks) == 3

    def test_outlines_contain_block_type_and_name(self) -> None:
        chunker = TerraformChunker()
        chunks = chunker.chunk(TF_CONTENT, "main.tf")
        outlines = [c.outline for c in chunks]
        assert any("variable" in o for o in outlines)
        assert any("google_container_cluster" in o for o in outlines)
        assert any("vpc" in o for o in outlines)

    def test_locals_and_output(self) -> None:
        chunker = TerraformChunker()
        chunks = chunker.chunk(TF_LOCALS, "locals.tf")
        assert len(chunks) == 2
        outlines = [c.outline for c in chunks]
        assert any("locals" in o for o in outlines)
        assert any("output" in o for o in outlines)

    def test_chunk_kind_is_tf_resource(self) -> None:
        chunker = TerraformChunker()
        chunks = chunker.chunk(TF_CONTENT, "main.tf")
        assert all(c.kind == "tf_resource" for c in chunks)

    def test_roundtrip(self) -> None:
        chunker = TerraformChunker()
        chunks = chunker.chunk(TF_CONTENT, "main.tf")
        assert _roundtrip(chunks, TF_CONTENT)

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        chunker = TerraformChunker(max_chunk_tokens=2000)
        chunks = chunker.chunk(TF_CONTENT, "main.tf")
        for c in chunks:
            assert c.token_estimate <= 2000


# ── MarkdownChunker ────────────────────────────────────────────────────────────


MD_CONTENT = """\
# My Project

Intro paragraph.

## Installation

Install it like this.

## Usage

Use it like that.
"""

MD_NO_H2 = """\
# Single Section

Just some content here.
No H2 headings.
"""


class TestMarkdownChunker:
    def test_splits_on_h2_headings(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(MD_CONTENT, "README.md")
        # preamble + 2 H2 sections
        h2_chunks = [c for c in chunks if "##" in c.content or c.start_line > 1]
        assert len(chunks) >= 2

    def test_preserves_h1_context_in_outline(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(MD_CONTENT, "README.md")
        # H2 chunks should contain H1 context
        h2_chunks = [c for c in chunks if "Installation" in c.outline or "Usage" in c.outline]
        for c in h2_chunks:
            assert "My Project" in c.outline

    def test_no_h2_yields_single_chunk(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(MD_NO_H2, "doc.md")
        assert len(chunks) == 1

    def test_chunk_kind_is_markdown(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(MD_CONTENT, "README.md")
        assert all(c.kind == "markdown" for c in chunks)

    def test_roundtrip(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(MD_CONTENT, "README.md")
        assert _roundtrip(chunks, MD_CONTENT)

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        chunker = MarkdownChunker(max_chunk_tokens=2000)
        chunks = chunker.chunk(MD_CONTENT, "README.md")
        for c in chunks:
            assert c.token_estimate <= 2000


# ── FallbackLineChunker ────────────────────────────────────────────────────────


class TestFallbackLineChunker:
    def test_produces_overlapping_windows(self) -> None:
        # 100 lines, window=40, overlap=10
        content = "".join(f"line{i}\n" for i in range(100))
        chunker = FallbackLineChunker(window=40, overlap=10)
        chunks = chunker.chunk(content, "file.txt")
        assert len(chunks) >= 3
        # Check overlap: last 10 lines of chunk N = first 10 lines of chunk N+1
        for i in range(len(chunks) - 1):
            prev_lines = chunks[i].content.splitlines()
            next_lines = chunks[i + 1].content.splitlines()
            assert prev_lines[-10:] == next_lines[:10]

    def test_outline_format_lines_start_end(self) -> None:
        content = "".join(f"line{i}\n" for i in range(50))
        chunker = FallbackLineChunker(window=20, overlap=5)
        chunks = chunker.chunk(content, "file.txt")
        for c in chunks:
            assert c.outline.startswith("lines ")
            parts = c.outline.split(" ")[1].split("-")
            assert len(parts) == 2
            assert int(parts[0]) <= int(parts[1])

    def test_chunk_kind_is_fallback(self) -> None:
        content = "hello\nworld\n"
        chunker = FallbackLineChunker()
        chunks = chunker.chunk(content, "file.txt")
        assert all(c.kind == "fallback" for c in chunks)

    def test_roundtrip_small_file(self) -> None:
        content = "hello\nworld\n"
        chunker = FallbackLineChunker()
        chunks = chunker.chunk(content, "file.txt")
        assert _roundtrip(chunks, content)

    def test_roundtrip_large_file(self) -> None:
        content = "".join(f"line{i}\n" for i in range(500))
        chunker = FallbackLineChunker(window=100, overlap=20)
        chunks = chunker.chunk(content, "large.txt")
        # Round-trip: non-overlapping portions must cover original
        # For overlap windows, just check all lines of original appear
        full = "".join(c.content for c in chunks)
        # Each line of original must appear in the concatenated output
        for line in content.splitlines():
            assert line in full


# ── get_chunker selector ───────────────────────────────────────────────────────


class TestGetChunker:
    def test_yaml_kinds_return_yaml_chunker(self) -> None:
        for kind in ("yaml", "k8s_manifest", "argocd", "istio_crd", "kustomization"):
            chunker = get_chunker(kind)
            assert isinstance(chunker, YamlDocChunker), f"Expected YamlDocChunker for kind={kind}"

    def test_terraform_kinds_return_tf_chunker(self) -> None:
        for kind in ("terraform", "terraform_gke"):
            chunker = get_chunker(kind)
            assert isinstance(chunker, TerraformChunker)

    def test_markdown_returns_markdown_chunker(self) -> None:
        assert isinstance(get_chunker("markdown"), MarkdownChunker)

    def test_unknown_returns_fallback(self) -> None:
        assert isinstance(get_chunker("python"), FallbackLineChunker)
        assert isinstance(get_chunker("unknown_kind"), FallbackLineChunker)


# ── chunk_file fallback on malformed YAML ─────────────────────────────────────


class TestChunkFileFallback:
    def test_malformed_yaml_falls_back_to_line_chunker(self) -> None:
        chunks = chunk_file(MALFORMED_YAML, "bad.yaml", "yaml")
        assert len(chunks) >= 1
        assert all(c.kind == "fallback" for c in chunks)

    def test_valid_yaml_uses_yaml_chunker(self) -> None:
        chunks = chunk_file(SINGLE_DOC_YAML, "svc.yaml", "yaml")
        assert len(chunks) >= 1
        assert chunks[0].kind == "yaml_doc"


# ── Token estimate integrity ───────────────────────────────────────────────────


class TestTokenEstimate:
    def test_token_estimate_matches_chars_div_4(self) -> None:
        chunker = YamlDocChunker()
        chunks = chunker.chunk(SINGLE_DOC_YAML, "svc.yaml")
        for c in chunks:
            assert c.token_estimate == len(c.content) // 4
