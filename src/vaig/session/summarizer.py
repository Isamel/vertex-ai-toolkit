"""History summarizer — compresses old conversation messages into a compact summary.

When the conversation history approaches the configured token budget, the
``HistorySummarizer`` condenses older messages into a single summary message
that preserves key decisions, tool results, error states, and user intent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.core.client import ChatMessage
from vaig.core.config import DEFAULT_CHARS_PER_TOKEN

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient

logger = logging.getLogger(__name__)

# ── Summarization system prompt ──────────────────────────────
# Anti-hallucination rules are embedded directly in the prompt.

SUMMARIZATION_PROMPT = """\
You are a conversation summarizer for a DevOps/SRE AI assistant.

Your task is to produce a concise summary of the conversation messages below.
The summary will REPLACE the original messages in the conversation history,
so it MUST preserve all information necessary for the assistant to continue
the conversation coherently.

## What to preserve
- Key user requests and stated intent
- Decisions made (by the user or the assistant)
- Tool invocations and their results (especially errors)
- Error states, exceptions, and how they were resolved
- File paths, resource names, and identifiers mentioned
- Configuration values or commands that were discussed

## What to omit
- Repetitive greetings or acknowledgments
- Verbose tool output that was already interpreted
- Intermediate reasoning that led to a final conclusion (keep the conclusion)

## Anti-hallucination rules — CRITICAL
- Do NOT invent information. Only summarize what is explicitly stated in the messages.
- Do NOT add details, conclusions, or recommendations that are not present in the original messages.
- If something is ambiguous or unclear in the messages, state it as ambiguous — do NOT resolve the ambiguity by guessing.
- Do NOT fabricate tool results, file paths, error messages, or any other factual claims.

## Output format
Produce a single block of text (no markdown headers) that reads as a chronological
narrative summary. Start with "[CONVERSATION SUMMARY]" on the first line.
Keep the summary under {target_tokens} tokens (roughly {target_chars} characters).
"""


def estimate_tokens(text: str, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from character length.

    Uses a simple ``len(text) / chars_per_token`` heuristic.  This is
    intentionally conservative (overestimates slightly) to avoid crossing
    the real token limit.

    Args:
        text: The text to estimate tokens for.
        chars_per_token: Characters per token ratio (default 4.0).

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0
    return max(0, int(len(text) / chars_per_token))


def estimate_history_tokens(
    messages: list[ChatMessage],
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimate the total token count for a list of chat messages.

    Sums the rough estimate for each message's content.

    Args:
        messages: List of ``ChatMessage`` objects.
        chars_per_token: Characters per token ratio.

    Returns:
        Total estimated token count.
    """
    return sum(estimate_tokens(m.content, chars_per_token=chars_per_token) for m in messages)


class HistorySummarizer:
    """Summarizes a list of chat messages into a single compact message.

    The summarizer calls the Gemini API to produce a concise summary that
    preserves actionable context while drastically reducing token count.

    Args:
        model_name: Model ID to use for summarization (e.g. ``gemini-2.5-flash``).
        summary_target_tokens: Target token budget for the output summary.
    """

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash",
        summary_target_tokens: int = 4_000,
    ) -> None:
        self._model_name = model_name
        self._summary_target_tokens = summary_target_tokens

    @property
    def summarization_prompt(self) -> str:
        """Return the full summarization system prompt with target size filled in."""
        target_chars = int(self._summary_target_tokens * DEFAULT_CHARS_PER_TOKEN)
        return SUMMARIZATION_PROMPT.format(
            target_tokens=self._summary_target_tokens,
            target_chars=target_chars,
        )

    def summarize(
        self,
        messages: list[ChatMessage],
        client: GeminiClient,
    ) -> ChatMessage:
        """Summarize a list of messages into a single summary message.

        Sends the messages to the model with a summarization system prompt
        and returns a ``ChatMessage`` with ``role="user"`` whose content is
        the generated summary prefixed with ``[CONVERSATION SUMMARY]``.

        Args:
            messages: The messages to summarize.
            client: An initialized ``GeminiClient`` instance.

        Returns:
            A single ``ChatMessage`` containing the summary.

        Raises:
            RuntimeError: If the model returns an empty summary.
        """
        if not messages:
            return ChatMessage(
                role="user",
                content="[CONVERSATION SUMMARY]\nNo prior conversation.",
            )

        # Build a textual representation of the messages for the model
        parts: list[str] = []
        for msg in messages:
            role_label = msg.role.upper()
            parts.append(f"[{role_label}]: {msg.content}")
        conversation_text = "\n\n".join(parts)

        logger.info(
            "Summarizing %d messages (~%d estimated tokens) with model %s",
            len(messages),
            estimate_history_tokens(messages),
            self._model_name,
        )

        result = client.generate(
            prompt=conversation_text,
            system_instruction=self.summarization_prompt,
            model_id=self._model_name,
            temperature=0.3,  # Low temperature for factual summary
            max_output_tokens=self._summary_target_tokens,
        )

        summary_text = result.text.strip()
        if not summary_text:
            logger.warning("Summarization returned empty text — using fallback")
            summary_text = "[CONVERSATION SUMMARY]\nPrevious conversation could not be summarized."

        # Ensure the summary starts with the marker
        if not summary_text.startswith("[CONVERSATION SUMMARY]"):
            summary_text = f"[CONVERSATION SUMMARY]\n{summary_text}"

        logger.info(
            "Summarization complete — result ~%d estimated tokens",
            estimate_tokens(summary_text),
        )

        return ChatMessage(role="user", content=summary_text)
