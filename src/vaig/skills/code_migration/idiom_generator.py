"""IdiomGenerator — LLM-based idiom map generation with filesystem caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
Generate a YAML idiom map for migrating code from {source_lang} to {target_lang}.

The map must follow this exact schema:

```yaml
source_lang: {source_lang}
target_lang: {target_lang}

idioms:
  - source_pattern: "description of the source idiom"
    target_pattern: "equivalent idiom in target language"
    description: "brief explanation of the transformation"
    example_before: |
      source language code example
    example_after: |
      target language code example

dependencies:
  "source-package": "target-package-equivalent"
```

Requirements:
- Include at least 10 common idioms that developers encounter when migrating
- Each idiom entry MUST have: source_pattern, target_pattern, description, example_before, example_after
- Focus on idiomatic patterns, not just syntax differences
- Include realistic, copy-paste-ready code examples
- Include 5-10 dependency substitutions in the dependencies section
- Output ONLY the YAML content — no markdown fences, no preamble, no explanation

Produce the complete YAML now:
"""


class IdiomGenerator:
    """Generate idiom maps for language pairs using an LLM, with local caching.

    Generated maps are stored under ``~/.vaig/idioms/`` as
    ``{source_lang}_to_{target_lang}.yaml`` files so subsequent calls avoid
    redundant LLM round-trips.

    Args:
        client: A :class:`~vaig.core.client.GeminiClient` instance used for
            LLM generation.  Must be provided; the generator does NOT import
            settings or create its own client.
        cache_dir: Root directory for cached idiom maps.  Defaults to
            ``~/.vaig/idioms/``.
        model_id: Override model for generation calls.  ``None`` uses the
            client's default model.
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        cache_dir: Path | str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else Path("~/.vaig/idioms").expanduser()
        self._model_id = model_id

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, source_lang: str, target_lang: str) -> str:
        """Return YAML content for the given language-pair idiom map.

        Checks the cache first.  On a cache miss, calls the LLM and persists
        the result before returning.

        Args:
            source_lang: Source programming language (e.g. ``"python"``).
            target_lang: Target programming language (e.g. ``"go"``).

        Returns:
            Raw YAML string representing the idiom map.

        Raises:
            RuntimeError: If the LLM call fails and no cached content is
                available.
        """
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()

        cached = self._load_from_cache(source_lang, target_lang)
        if cached is not None:
            logger.debug("IdiomGenerator: cache hit for %s→%s", source_lang, target_lang)
            return cached

        logger.info("IdiomGenerator: generating idiom map for %s→%s", source_lang, target_lang)
        yaml_content = self._call_llm(source_lang, target_lang)
        self._save_to_cache(source_lang, target_lang, yaml_content)
        return yaml_content

    def cache_path(self, source_lang: str, target_lang: str) -> Path:
        """Return the expected cache file path for a language pair.

        Args:
            source_lang: Source language (lowercased automatically).
            target_lang: Target language (lowercased automatically).

        Returns:
            Absolute :class:`~pathlib.Path` to the YAML cache file.
        """
        filename = f"{source_lang.lower()}_to_{target_lang.lower()}.yaml"
        return self._cache_dir / filename

    def is_cached(self, source_lang: str, target_lang: str) -> bool:
        """Return True if a cached map exists for the given language pair."""
        return self.cache_path(source_lang, target_lang).exists()

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_from_cache(self, source_lang: str, target_lang: str) -> str | None:
        """Load cached YAML content if available.

        Returns:
            YAML string from disk, or ``None`` if no cache entry exists.
        """
        path = self.cache_path(source_lang, target_lang)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("IdiomGenerator: failed to read cache %s: %s", path, exc)
            return None

    def _save_to_cache(self, source_lang: str, target_lang: str, content: str) -> None:
        """Persist YAML content to the cache directory.

        Creates parent directories as needed.  Failures are logged as
        warnings and do NOT propagate — caching is best-effort.
        """
        path = self.cache_path(source_lang, target_lang)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.debug("IdiomGenerator: cached %s→%s → %s", source_lang, target_lang, path)
        except OSError as exc:
            logger.warning("IdiomGenerator: failed to write cache %s: %s", path, exc)

    def _call_llm(self, source_lang: str, target_lang: str) -> str:
        """Invoke the LLM to generate an idiom map YAML.

        Args:
            source_lang: Source language (already lowercased).
            target_lang: Target language (already lowercased).

        Returns:
            Raw YAML string from the LLM response.

        Raises:
            RuntimeError: If the LLM call raises an exception.
        """
        prompt = _GENERATION_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
        )
        try:
            result = self._client.generate(
                prompt,
                model_id=self._model_id,
            )
            text: str = result.text.strip()
            return text
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"IdiomGenerator: LLM call failed for {source_lang}→{target_lang}: {exc}"
            ) from exc

    def parse_yaml(self, content: str) -> dict[str, Any] | None:
        """Parse and validate YAML content returned by the LLM or cache.

        Returns:
            Parsed dict on success, or ``None`` if parsing fails or the
            content is not a valid idiom map dict.
        """
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            logger.warning("IdiomGenerator: failed to parse YAML: %s", exc)
            return None

        if not isinstance(raw, dict):
            logger.warning(
                "IdiomGenerator: unexpected YAML top-level type %s (expected dict)",
                type(raw).__name__,
            )
            return None

        data: dict[str, Any] = raw
        return data
