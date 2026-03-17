"""Tests for Q2: token-aware history summarization with lazy trigger.

Covers:
- Config defaults and validation
- Rough token estimation (chars/4)
- HistorySummarizer prompt content and anti-hallucination rules
- Summarization trigger logic (threshold crossing)
- Recent message preservation during summarization
- Summarizer with mocked client
- SessionManager._check_and_summarize integration
- Edge cases: empty history, single message, below threshold
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.core.client import ChatMessage, GenerationResult
from vaig.core.config import SessionConfig, Settings
from vaig.session.manager import _RECENT_MESSAGES_TO_KEEP, SessionManager
from vaig.session.summarizer import (
    SUMMARIZATION_PROMPT,
    HistorySummarizer,
    estimate_history_tokens,
    estimate_tokens,
)

# ══════════════════════════════════════════════════════════════
# Config defaults
# ══════════════════════════════════════════════════════════════


class TestSessionConfigDefaults:
    """Verify new SessionConfig fields have correct defaults."""

    def test_max_history_tokens_default(self) -> None:
        cfg = SessionConfig()
        assert cfg.max_history_tokens == 28_000

    def test_summarization_threshold_default(self) -> None:
        cfg = SessionConfig()
        assert cfg.summarization_threshold == 0.8

    def test_summary_target_tokens_default(self) -> None:
        cfg = SessionConfig()
        assert cfg.summary_target_tokens == 4_000

    def test_custom_values(self) -> None:
        cfg = SessionConfig(
            max_history_tokens=50_000,
            summarization_threshold=0.9,
            summary_target_tokens=2_000,
        )
        assert cfg.max_history_tokens == 50_000
        assert cfg.summarization_threshold == 0.9
        assert cfg.summary_target_tokens == 2_000


# ══════════════════════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════════════════════


class TestTokenEstimation:
    """Test the rough chars/4 token estimation."""

    def test_estimate_tokens_basic(self) -> None:
        # 100 chars / 4 = 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_estimate_tokens_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_tokens_custom_ratio(self) -> None:
        text = "a" * 100
        # 100 chars / 2.0 = 50 tokens
        assert estimate_tokens(text, chars_per_token=2.0) == 50

    def test_estimate_history_tokens_sums_messages(self) -> None:
        messages = [
            ChatMessage(role="user", content="a" * 100),
            ChatMessage(role="model", content="b" * 200),
        ]
        # (100 + 200) / 4 = 75
        assert estimate_history_tokens(messages) == 75

    def test_estimate_history_tokens_empty_list(self) -> None:
        assert estimate_history_tokens([]) == 0

    def test_estimate_history_tokens_with_content_objects(self) -> None:
        """estimate_history_tokens must handle google-genai Content objects.

        Content objects have ``.parts`` (list of Part-like objects with
        ``.text``) instead of a ``.content`` string.  This is the exact
        scenario that caused the original AttributeError bug.
        """
        from types import SimpleNamespace

        # Simulate google.genai.types.Content with .parts containing .text
        content_objs = [
            SimpleNamespace(
                role="user",
                parts=[SimpleNamespace(text="a" * 100)],
            ),
            SimpleNamespace(
                role="model",
                parts=[
                    SimpleNamespace(text="b" * 80),
                    SimpleNamespace(text="c" * 120),
                ],
            ),
        ]
        # 100 + (80 + 1 (space joiner) + 120) = 301 chars → 301 / 4 = 75
        assert estimate_history_tokens(content_objs) == 75  # type: ignore[arg-type]

    def test_estimate_history_tokens_content_with_non_text_parts(self) -> None:
        """Content objects with function_call parts (no .text) should not crash."""
        from types import SimpleNamespace

        content_objs = [
            SimpleNamespace(
                role="model",
                parts=[SimpleNamespace(function_call={"name": "get_pods"})],
            ),
        ]
        # No .text attribute → 0 tokens
        assert estimate_history_tokens(content_objs) == 0  # type: ignore[arg-type]

    def test_estimate_history_tokens_content_with_empty_parts(self) -> None:
        """Content with empty or None parts should return 0."""
        from types import SimpleNamespace

        content_objs = [
            SimpleNamespace(role="model", parts=None),
            SimpleNamespace(role="user", parts=[]),
        ]
        assert estimate_history_tokens(content_objs) == 0  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════
# Summarization prompt
# ══════════════════════════════════════════════════════════════


class TestSummarizationPrompt:
    """Verify the summarization prompt content and anti-hallucination rules."""

    def test_anti_hallucination_rules_present(self) -> None:
        assert "Do NOT invent information" in SUMMARIZATION_PROMPT
        assert "Do NOT add details" in SUMMARIZATION_PROMPT
        assert "Do NOT fabricate" in SUMMARIZATION_PROMPT

    def test_prompt_contains_preservation_guidance(self) -> None:
        assert "Key user requests" in SUMMARIZATION_PROMPT
        assert "Decisions made" in SUMMARIZATION_PROMPT
        assert "Tool invocations" in SUMMARIZATION_PROMPT
        assert "Error states" in SUMMARIZATION_PROMPT

    def test_prompt_has_target_token_placeholder(self) -> None:
        assert "{target_tokens}" in SUMMARIZATION_PROMPT
        assert "{target_chars}" in SUMMARIZATION_PROMPT


# ══════════════════════════════════════════════════════════════
# HistorySummarizer
# ══════════════════════════════════════════════════════════════


class TestHistorySummarizer:
    """Test HistorySummarizer with mocked GeminiClient."""

    def _make_mock_client(self, summary_text: str = "Summary of conversation.") -> MagicMock:
        client = MagicMock()
        client.generate.return_value = GenerationResult(
            text=summary_text,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        )
        return client

    def test_summarize_returns_chat_message(self) -> None:
        client = self._make_mock_client()
        summarizer = HistorySummarizer(summary_target_tokens=4_000)
        messages = [
            ChatMessage(role="user", content="Deploy the app"),
            ChatMessage(role="model", content="Deploying to production..."),
        ]
        result = summarizer.summarize(messages, client)
        assert isinstance(result, ChatMessage)
        assert result.role == "user"
        assert "[CONVERSATION SUMMARY]" in result.content

    def test_summarize_adds_marker_if_missing(self) -> None:
        client = self._make_mock_client("Just a plain summary without marker.")
        summarizer = HistorySummarizer()
        messages = [ChatMessage(role="user", content="Hello")]
        result = summarizer.summarize(messages, client)
        assert result.content.startswith("[CONVERSATION SUMMARY]")

    def test_summarize_preserves_existing_marker(self) -> None:
        client = self._make_mock_client("[CONVERSATION SUMMARY]\nAlready has marker.")
        summarizer = HistorySummarizer()
        messages = [ChatMessage(role="user", content="Hello")]
        result = summarizer.summarize(messages, client)
        # Should not duplicate the marker
        assert result.content.count("[CONVERSATION SUMMARY]") == 1

    def test_summarize_empty_messages(self) -> None:
        client = self._make_mock_client()
        summarizer = HistorySummarizer()
        result = summarizer.summarize([], client)
        assert result.role == "user"
        assert "[CONVERSATION SUMMARY]" in result.content
        assert "No prior conversation" in result.content
        # Should NOT call the API for empty messages
        client.generate.assert_not_called()

    def test_summarize_calls_generate_with_correct_params(self) -> None:
        client = self._make_mock_client()
        summarizer = HistorySummarizer(
            model_name="gemini-2.5-flash",
            summary_target_tokens=4_000,
        )
        messages = [
            ChatMessage(role="user", content="Fix the bug"),
            ChatMessage(role="model", content="Found the issue in auth.py"),
        ]
        summarizer.summarize(messages, client)
        client.generate.assert_called_once()
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["model_id"] == "gemini-2.5-flash"
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_output_tokens"] == 4_000
        assert "Do NOT invent information" in call_kwargs.kwargs["system_instruction"]

    def test_summarize_handles_empty_response(self) -> None:
        client = self._make_mock_client("")
        summarizer = HistorySummarizer()
        messages = [ChatMessage(role="user", content="Hello")]
        result = summarizer.summarize(messages, client)
        assert "[CONVERSATION SUMMARY]" in result.content
        assert "could not be summarized" in result.content

    def test_summarization_prompt_property_fills_targets(self) -> None:
        summarizer = HistorySummarizer(summary_target_tokens=2_000)
        prompt = summarizer.summarization_prompt
        assert "2000" in prompt  # target_tokens filled in
        assert "8000" in prompt  # target_chars = 2000 * 4


# ══════════════════════════════════════════════════════════════
# SessionManager — get_token_estimate & _check_and_summarize
# ══════════════════════════════════════════════════════════════


def _make_settings(tmp_path: Path, **session_overrides: object) -> Settings:
    """Build a ``Settings`` with a temp DB and optional session config overrides."""
    session_kwargs: dict[str, object] = {
        "db_path": str(tmp_path / "test.db"),
        **session_overrides,
    }
    return Settings(
        session=SessionConfig(**session_kwargs),  # type: ignore[arg-type]
    )


class TestSessionManagerTokenEstimate:
    """Test SessionManager.get_token_estimate()."""

    def test_no_active_session(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        assert mgr.get_token_estimate() == 0

    def test_with_messages(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        mgr.new_session("test")
        mgr.add_message("user", "a" * 400)  # 100 tokens
        mgr.add_message("model", "b" * 800)  # 200 tokens
        assert mgr.get_token_estimate() == 300


class TestCheckAndSummarize:
    """Test SessionManager._check_and_summarize trigger logic."""

    def test_no_active_session_no_op(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        mgr = SessionManager(settings)
        client = MagicMock()
        # Should not raise
        mgr._check_and_summarize(client)

    def test_below_threshold_no_summarization(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, max_history_tokens=28_000)
        mgr = SessionManager(settings)
        mgr.new_session("test")
        # Add small messages — well below threshold
        mgr.add_message("user", "Hello")
        mgr.add_message("model", "Hi there")
        client = MagicMock()
        mgr._check_and_summarize(client)
        # No summarization should occur
        assert len(mgr.get_history()) == 2

    def test_above_threshold_triggers_summarization(self, tmp_path: Path) -> None:
        # Set a tiny token limit so we easily exceed it
        settings = _make_settings(
            tmp_path,
            max_history_tokens=200,  # 200 tokens = 800 chars
            summarization_threshold=0.5,  # trigger at 100 tokens = 400 chars
            summary_target_tokens=50,
        )
        mgr = SessionManager(settings)
        mgr.new_session("test")

        # Add enough messages to exceed threshold (> 100 tokens = 400 chars)
        # We need > _RECENT_MESSAGES_TO_KEEP messages total
        for i in range(15):
            role = "user" if i % 2 == 0 else "model"
            mgr.add_message(role, f"Message {i}: " + "x" * 40)  # ~12 tokens each

        assert len(mgr.get_history()) == 15

        # Mock the client for summarization
        mock_client = MagicMock()
        mock_client.generate.return_value = GenerationResult(
            text="[CONVERSATION SUMMARY]\nSummary of 5 older messages.",
            model="gemini-2.5-flash",
            usage={},
        )

        mgr._check_and_summarize(mock_client)

        # After summarization: 1 summary + _RECENT_MESSAGES_TO_KEEP recent
        assert len(mgr.get_history()) == 1 + _RECENT_MESSAGES_TO_KEEP
        assert "[CONVERSATION SUMMARY]" in mgr.get_history()[0].content

    def test_recent_messages_preserved(self, tmp_path: Path) -> None:
        settings = _make_settings(
            tmp_path,
            max_history_tokens=100,
            summarization_threshold=0.3,
            summary_target_tokens=20,
        )
        mgr = SessionManager(settings)
        mgr.new_session("test")

        # Add 15 messages with identifiable content
        for i in range(15):
            role = "user" if i % 2 == 0 else "model"
            mgr.add_message(role, f"MSG-{i:02d} " + "x" * 40)

        mock_client = MagicMock()
        mock_client.generate.return_value = GenerationResult(
            text="[CONVERSATION SUMMARY]\nOlder messages summarized.",
            model="gemini-2.5-flash",
            usage={},
        )

        mgr._check_and_summarize(mock_client)

        history = mgr.get_history()
        # The last _RECENT_MESSAGES_TO_KEEP messages should be the recent ones
        recent = history[1:]
        assert len(recent) == _RECENT_MESSAGES_TO_KEEP
        # They should be the LAST 10 original messages (MSG-05 through MSG-14)
        for idx, msg in enumerate(recent):
            expected_num = 15 - _RECENT_MESSAGES_TO_KEEP + idx
            assert f"MSG-{expected_num:02d}" in msg.content

    def test_too_few_messages_skips_summarization(self, tmp_path: Path) -> None:
        """When there are <= _RECENT_MESSAGES_TO_KEEP messages, skip summarization."""
        settings = _make_settings(
            tmp_path,
            max_history_tokens=10,  # Very low — would trigger
            summarization_threshold=0.1,
        )
        mgr = SessionManager(settings)
        mgr.new_session("test")

        # Add only 5 messages (< _RECENT_MESSAGES_TO_KEEP)
        for i in range(5):
            mgr.add_message("user", f"Message {i}" + "x" * 100)

        client = MagicMock()
        mgr._check_and_summarize(client)

        # Should not summarize — too few messages to split
        assert len(mgr.get_history()) == 5
        client.generate.assert_not_called()

    def test_summarization_failure_preserves_history(self, tmp_path: Path) -> None:
        """If summarization API call fails, original history is kept."""
        settings = _make_settings(
            tmp_path,
            max_history_tokens=100,
            summarization_threshold=0.3,
        )
        mgr = SessionManager(settings)
        mgr.new_session("test")

        for i in range(15):
            role = "user" if i % 2 == 0 else "model"
            mgr.add_message(role, f"Message {i}" + "x" * 40)

        original_count = len(mgr.get_history())

        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("API exploded")

        mgr._check_and_summarize(mock_client)

        # History should be unchanged after failure
        assert len(mgr.get_history()) == original_count

    def test_single_message_history(self, tmp_path: Path) -> None:
        """Single message should never trigger summarization."""
        settings = _make_settings(
            tmp_path,
            max_history_tokens=1,  # Absurdly low
            summarization_threshold=0.1,
        )
        mgr = SessionManager(settings)
        mgr.new_session("test")
        mgr.add_message("user", "Hello world" * 100)

        client = MagicMock()
        mgr._check_and_summarize(client)
        assert len(mgr.get_history()) == 1
        client.generate.assert_not_called()


# ══════════════════════════════════════════════════════════════
# GeminiClient warning log
# ══════════════════════════════════════════════════════════════


class TestClientHistoryWarning:
    """Verify GeminiClient logs warnings when history is large."""

    def test_warn_if_history_large_triggers(self) -> None:
        from vaig.core.client import GeminiClient

        history = [ChatMessage(role="user", content="x" * 200_000)]  # ~50k tokens
        with patch.object(GeminiClient, "__init__", lambda self, s: None):
            with patch("vaig.core.client.logger") as mock_logger:
                GeminiClient._warn_if_history_large(history, max_tokens=10_000)
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "exceeds max_history_tokens" in warning_msg

    def test_warn_if_history_small_no_warning(self) -> None:
        from vaig.core.client import GeminiClient

        history = [ChatMessage(role="user", content="short message")]
        with patch.object(GeminiClient, "__init__", lambda self, s: None):
            with patch("vaig.core.client.logger") as mock_logger:
                GeminiClient._warn_if_history_large(history, max_tokens=10_000)
                mock_logger.warning.assert_not_called()
