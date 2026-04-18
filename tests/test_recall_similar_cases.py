"""Tests for recall_similar_cases tool."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# vaig.tools.base triggers vaig.tools.__init__ which imports gke_tools; skip
# gracefully when the gke k8s_client attribute is absent (dev environment).
try:
    from vaig.tools.knowledge.recall_similar_cases import recall_similar_cases

    _IMPORT_OK = True
except (ImportError, AttributeError):
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason="vaig.tools.gke.k8s_client not available in this environment",
)


def _make_index(narratives: list[str]) -> MagicMock:
    index = MagicMock()
    index.recall.return_value = narratives
    return index


class TestRecallSimilarCases:
    def test_returns_formatted_narratives(self) -> None:
        index = _make_index(["Case A happened.", "Case B happened."])
        result = recall_similar_cases("OOMKilled pods", index)
        assert not result.error
        assert "Case A happened." in result.output
        assert "Case B happened." in result.output

    def test_returns_no_results_message_when_empty(self) -> None:
        index = _make_index([])
        result = recall_similar_cases("some query", index)
        assert not result.error
        assert "No similar historical cases found" in result.output

    def test_passes_top_k_to_index(self) -> None:
        index = _make_index(["n1"])
        recall_similar_cases("query", index, top_k=3)
        index.recall.assert_called_once_with("query", top_k=3)

    def test_output_wrapped_as_untrusted(self) -> None:
        index = _make_index(["Narrative here."])
        result = recall_similar_cases("query", index)
        # Content should be wrapped with untrusted content delimiters
        assert "Narrative here." in result.output
