"""Tests for summarise_text in vaig.core.summariser."""

from __future__ import annotations

from unittest.mock import MagicMock

from vaig.core.summariser import _UNAVAILABLE, summarise_text


class TestSummariseTextNormal:
    """summarise_text returns the client's generated summary."""

    def test_returns_summarised_text(self) -> None:
        client = MagicMock()
        client.generate.return_value = "This is the summary."

        result = summarise_text("Some long text to summarise", client)

        assert result == "This is the summary."

    def test_client_called_with_prompt_and_system(self) -> None:
        client = MagicMock()
        client.generate.return_value = "Summary."

        summarise_text("Input text", client)

        call_kwargs = client.generate.call_args.kwargs
        assert "prompt" in call_kwargs
        assert "system" in call_kwargs
        assert "temperature" in call_kwargs
        # The prompt should wrap the original text
        assert "Input text" in call_kwargs["prompt"]

    def test_max_output_tokens_forwarded(self) -> None:
        client = MagicMock()
        client.generate.return_value = "Short."

        summarise_text("text", client, max_output_tokens=256)

        call_kwargs = client.generate.call_args.kwargs
        assert call_kwargs["max_output_tokens"] == 256


class TestSummariseTextClientFailure:
    """summarise_text falls back gracefully when the LLM client raises."""

    def test_returns_unavailable_marker_on_exception(self) -> None:
        client = MagicMock()
        client.generate.side_effect = RuntimeError("model unavailable")

        result = summarise_text("Some text", client)

        assert result == _UNAVAILABLE

    def test_does_not_propagate_exception(self) -> None:
        client = MagicMock()
        client.generate.side_effect = ValueError("bad request")

        # Should not raise
        result = summarise_text("text", client)

        assert result == _UNAVAILABLE
