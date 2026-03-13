"""Dynamic language detection for multi-agent pipelines.

Detects the language of user queries and generates language instructions
that can be prepended to agent system prompts at runtime.  This ensures
the entire pipeline responds in the same language the user used — without
modifying the underlying prompt constants.

Only ``"es"`` (Spanish) and ``"en"`` (English, default) are supported.
The detection uses a lightweight heuristic based on common words and
patterns — no external libraries required.
"""

from __future__ import annotations

import re

# ── Spanish indicator words ──────────────────────────────────
# Common words that strongly signal Spanish.  Includes question markers,
# articles, prepositions, pronouns, and verbs that rarely overlap with
# English.  The set is deliberately broad to catch casual queries like
# "revisá el estado de los pods".
_SPANISH_INDICATORS: frozenset[str] = frozenset({
    # Articles & prepositions
    "el", "la", "los", "las", "del", "al", "un", "una", "unos", "unas",
    "de", "en", "con", "por", "para", "sobre", "entre", "sin", "desde",
    "hacia", "hasta",
    # Pronouns & determiners
    "yo", "tú", "él", "ella", "nosotros", "ellos", "ellas", "esto",
    "esta", "ese", "esa", "estos", "estas", "esos", "esas",
    "qué", "cómo", "cuál", "cuáles", "dónde", "cuándo", "cuánto",
    "cuántos", "quién", "quiénes",
    # Question markers (without accents — users often skip them)
    "que", "como", "cual", "cuales", "donde", "cuando", "cuanto",
    "cuantos", "quien", "quienes",
    # Common verbs
    "es", "está", "son", "están", "tiene", "tienen", "hay",
    "puede", "pueden", "hacer", "dame", "dime", "muestra",
    "revisar", "verificar", "comprobar", "analizar", "mostrar",
    "necesito", "quiero", "puedes", "podrías", "hazme",
    # Service-health domain
    "salud", "estado", "servicios", "clúster",
    "reporte", "informe",
    # Conjunctions & misc
    "pero", "también", "ahora", "aquí", "sí", "no", "más",
    "todo", "todos", "todas", "cada", "otro", "otra",
    "si", "y", "o", "ni",
})

# Words that exist in both languages and should NOT count as Spanish
# indicators (e.g. "no", "en", "cluster" could be English too).
# We handle this by requiring a THRESHOLD of matches rather than a
# single match.

# Minimum number of Spanish indicator words required to classify as Spanish.
_SPANISH_THRESHOLD = 2

# ── Public API ───────────────────────────────────────────────

_LANGUAGE_NAMES: dict[str, str] = {
    "es": "Spanish",
    "en": "English",
}


def detect_language(query: str) -> str:
    """Detect whether a query is in Spanish or English.

    Uses a heuristic based on the presence of common Spanish words.
    Returns ``"es"`` for Spanish or ``"en"`` for English (default).

    The algorithm:
    1. Normalise the query to lowercase and tokenise on word boundaries.
    2. Count how many tokens appear in the Spanish indicator set.
    3. If the count meets or exceeds the threshold, classify as Spanish.
    4. Additionally, detect inverted question/exclamation marks (``¿``, ``¡``)
       which are uniquely Spanish — a single occurrence is sufficient.

    Args:
        query: The user's input text.

    Returns:
        ``"es"`` or ``"en"``.
    """
    if not query or not query.strip():
        return "en"

    # Inverted punctuation is a dead giveaway for Spanish
    if "¿" in query or "¡" in query:
        return "es"

    # Tokenise: split on non-alphanumeric characters (preserving accented chars)
    tokens = re.findall(r"[a-záéíóúüñ]+", query.lower())

    spanish_count = sum(1 for token in tokens if token in _SPANISH_INDICATORS)

    if spanish_count >= _SPANISH_THRESHOLD:
        return "es"

    return "en"


def build_language_instruction(lang: str) -> str:
    """Build a language instruction to prepend to agent system prompts.

    Args:
        lang: Language code (``"es"`` or ``"en"``).

    Returns:
        A system-level instruction telling the agent which language to use.
        Returns an empty string for English (the prompts are already in
        English, so no instruction is needed).
    """
    if lang == "en":
        # Prompts are already in English — no injection needed.
        return ""

    language_name = _LANGUAGE_NAMES.get(lang, lang)

    return (
        f"## LANGUAGE INSTRUCTION\n\n"
        f"The user's query is in {language_name}. You MUST respond entirely in "
        f"{language_name}. All section headers, descriptions, explanations, "
        f"findings, recommendations, and summaries MUST be written in "
        f"{language_name}. Technical terms (Kubernetes resource names, kubectl "
        f"commands, metric names, pod names) should remain in English as they "
        f"are proper nouns, but all surrounding text MUST be in {language_name}.\n\n"
    )


def inject_language_into_config(
    agent_configs: list[dict],
    lang: str,
) -> list[dict]:
    """Prepend a language instruction to every agent's system prompt.

    Modifies the config dicts **in place** — this is safe because the
    orchestrator creates fresh config lists per execution (via
    ``skill.get_agents_config()``).

    Args:
        agent_configs: List of agent configuration dicts (as returned by
            ``BaseSkill.get_agents_config()``).
        lang: Detected language code.

    Returns:
        The same list (mutated), for convenience.
    """
    instruction = build_language_instruction(lang)
    if not instruction:
        return agent_configs

    for config in agent_configs:
        if "system_prompt" in config:
            config["system_prompt"] = instruction + config["system_prompt"]
        if "system_instruction" in config:
            config["system_instruction"] = instruction + config["system_instruction"]

    return agent_configs
