"""Dynamic language detection for multi-agent pipelines.

Detects the language of user queries and generates language instructions
that can be prepended to agent system prompts at runtime.  This ensures
the entire pipeline responds in the same language the user used — without
modifying the underlying prompt constants.

Supported languages: ``"es"`` (Spanish), ``"pt"`` (Portuguese),
``"fr"`` (French), ``"de"`` (German), ``"it"`` (Italian),
``"ja"`` (Japanese), ``"zh"`` (Chinese), ``"ko"`` (Korean),
and ``"en"`` (English, default).

The detection uses a lightweight heuristic based on common words,
Unicode script ranges, and special punctuation — no external libraries
required.
"""

from __future__ import annotations

import re
from typing import Any

# ── CJK Unicode ranges ──────────────────────────────────────
# Used for script-based detection of Japanese, Chinese, and Korean.
# These are compiled once at import time for performance.

# Hiragana (U+3040–U+309F) and Katakana (U+30A0–U+30FF) are unique to Japanese.
_RE_JAPANESE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")

# Hangul Syllables (U+AC00–U+D7AF) and Hangul Jamo (U+1100–U+11FF,
# U+3130–U+318F) are unique to Korean.
_RE_KOREAN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")

# CJK Unified Ideographs (U+4E00–U+9FFF).  Shared by Chinese and Japanese,
# but Japanese also uses Hiragana/Katakana — so we check Japanese first.
_RE_CJK_IDEOGRAPHS = re.compile(r"[\u4E00-\u9FFF]")

# ── Latin-script indicator words ─────────────────────────────
# Each set contains common words that strongly signal a specific language.
# Words are chosen to minimise overlap with English AND with each other.
# The threshold approach (≥2 matches) handles the few inevitable overlaps.

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

_PORTUGUESE_INDICATORS: frozenset[str] = frozenset({
    # Articles & prepositions
    "o", "os", "as", "um", "uma", "uns", "umas",
    "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "com", "sem", "sobre", "entre", "desde", "até",
    # Pronouns & determiners — uniquely Portuguese forms
    "eu", "você", "vocês", "nós", "eles", "elas",
    "ele", "ela", "isso", "isto", "esse", "essa",
    "este", "esta", "estes", "estas", "esses", "essas",
    "qual", "quais", "quem", "onde", "quando", "quanto",
    # Common verbs — Portuguese-specific conjugations
    "é", "são", "está", "estão", "tem", "têm", "há",
    "pode", "podem", "fazer", "mostre", "mostra",
    "verificar", "analisar", "mostrar", "preciso",
    "quero", "consegue",
    # Portuguese-unique words (not shared with Spanish)
    "também", "agora", "aqui", "mais", "muito", "muitos",
    "tudo", "cada", "outro", "outra", "outros", "outras",
    "mas", "porém", "então", "ainda", "já", "sim", "não",
    "obrigado", "obrigada", "serviço", "serviços",
    "relatório", "saúde",
})

_FRENCH_INDICATORS: frozenset[str] = frozenset({
    # Articles & prepositions
    "le", "la", "les", "un", "une", "des",
    "du", "de", "au", "aux",
    "dans", "sur", "avec", "pour", "par", "entre", "sans",
    "chez", "vers", "depuis", "après", "avant",
    # Pronouns & determiners
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre",
    "quel", "quelle", "quels", "quelles", "qui", "quoi", "où",
    "comment", "combien", "pourquoi",
    # Common verbs
    "est", "sont", "suis", "avez", "ont", "fait", "faire",
    "peut", "peuvent", "veux", "veut", "faut",
    "vérifier", "analyser", "montrer", "montrez",
    # French-unique words
    "oui", "non", "mais", "aussi", "maintenant", "ici",
    "très", "plus", "tout", "tous", "toutes", "chaque",
    "autre", "autres", "bien", "encore", "toujours",
    "rapport", "état", "santé", "service", "services",
    "bonjour", "merci", "problème",
})

_GERMAN_INDICATORS: frozenset[str] = frozenset({
    # Articles & prepositions
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer",
    "in", "auf", "mit", "für", "von", "zu", "an",
    "bei", "nach", "über", "unter", "zwischen", "vor",
    # Pronouns & determiners
    "ich", "du", "er", "sie", "wir", "ihr",
    "mein", "dein", "sein", "unser", "euer",
    "dieser", "diese", "dieses", "welcher", "welche", "welches",
    "was", "wer", "wo", "wie", "wann", "warum",
    # Common verbs
    "ist", "sind", "bin", "hat", "haben", "kann", "können",
    "muss", "müssen", "soll", "sollen", "wird", "werden",
    "zeige", "zeigen", "prüfen", "überprüfen", "analysieren",
    # German-unique words
    "ja", "nein", "nicht", "auch", "jetzt", "hier",
    "sehr", "mehr", "alle", "alles", "jeder", "jede",
    "andere", "anderer", "gut", "noch", "immer",
    "bericht", "zustand", "dienst", "dienste",
    "bitte", "danke", "fehler",
})

_ITALIAN_INDICATORS: frozenset[str] = frozenset({
    # Articles & prepositions
    "il", "lo", "la", "gli", "le",
    "uno", "una", "dei", "delle", "degli",
    "del", "dello", "della", "nel", "nella", "nei", "nelle",
    "di", "da", "in", "con", "su", "per", "tra", "fra",
    # Pronouns & determiners
    "io", "tu", "lui", "lei", "noi", "voi", "loro",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "quale", "quali", "chi", "dove", "quando", "quanto",
    "perché",
    # Common verbs
    "è", "sono", "sei", "ha", "hanno", "può", "possono",
    "fare", "mostra", "mostrami", "verifica", "verificare",
    "analizzare", "controllare",
    # Italian-unique words
    "sì", "non", "anche", "adesso", "qui",
    "molto", "più", "tutto", "tutti", "tutte", "ogni",
    "altro", "altra", "altri", "altre", "bene", "ancora",
    "sempre", "rapporto", "stato", "servizio", "servizi",
    "grazie", "prego", "problema",
})

# Mapping from language code to its indicator set.  Checked in order;
# languages with more distinctive words are checked first to reduce
# false positives between closely related languages (es vs pt, fr vs it).
_INDICATOR_SETS: list[tuple[str, frozenset[str]]] = [
    ("es", _SPANISH_INDICATORS),
    ("pt", _PORTUGUESE_INDICATORS),
    ("de", _GERMAN_INDICATORS),
    ("fr", _FRENCH_INDICATORS),
    ("it", _ITALIAN_INDICATORS),
]

# Minimum number of indicator words required to classify a language.
_INDICATOR_THRESHOLD = 2

# ── Tokenisation regex ───────────────────────────────────────
# Matches Latin-script words including accented characters from all
# supported Latin-script languages (es, pt, fr, de, it).
_RE_LATIN_TOKEN = re.compile(
    r"[a-záàâãéèêíïîóòôõúùûüçñœæß]+",
    re.IGNORECASE,
)

# ── Public API ───────────────────────────────────────────────

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
}


def detect_language(query: str) -> str:
    """Detect the language of a user query.

    Uses lightweight heuristics — no external NLP libraries required:

    * **CJK scripts**: Japanese is detected via Hiragana/Katakana characters,
      Korean via Hangul, and Chinese via CJK Ideographs (when no
      Hiragana/Katakana are present).
    * **Latin-script languages**: Detected via indicator-word matching with
      a threshold of 2.  Spanish inverted punctuation (``¿``, ``¡``) is a
      special-case shortcut.

    Supported return values: ``"es"``, ``"pt"``, ``"fr"``, ``"de"``,
    ``"it"``, ``"ja"``, ``"zh"``, ``"ko"``, ``"en"`` (default).

    Args:
        query: The user's input text.

    Returns:
        A BCP-47 language code string.
    """
    if not query or not query.strip():
        return "en"

    # ── CJK script detection (highest priority) ─────────────
    # Japanese: presence of Hiragana or Katakana is unambiguous.
    if _RE_JAPANESE.search(query):
        return "ja"

    # Korean: Hangul syllables/jamo are unambiguous.
    if _RE_KOREAN.search(query):
        return "ko"

    # Chinese: CJK Ideographs without Japanese/Korean script markers.
    if _RE_CJK_IDEOGRAPHS.search(query):
        return "zh"

    # ── Spanish inverted punctuation shortcut ────────────────
    if "¿" in query or "¡" in query:
        return "es"

    # ── Latin-script indicator-word matching ─────────────────
    tokens = _RE_LATIN_TOKEN.findall(query.lower())

    # Score each language and pick the best match above threshold.
    best_lang = "en"
    best_count = 0

    for lang_code, indicators in _INDICATOR_SETS:
        count = sum(1 for token in tokens if token in indicators)
        if count > best_count:
            best_count = count
            best_lang = lang_code

    if best_count >= _INDICATOR_THRESHOLD:
        return best_lang

    return "en"


def build_language_instruction(lang: str) -> str:
    """Build a language instruction to prepend to agent system prompts.

    Args:
        lang: Language code — any of the supported codes (``"es"``,
            ``"pt"``, ``"fr"``, ``"de"``, ``"it"``, ``"ja"``, ``"zh"``,
            ``"ko"``, ``"en"``).

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
    agent_configs: list[dict[str, Any]],
    lang: str,
) -> list[dict[str, Any]]:
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
        if "system_instruction" in config:
            config["system_instruction"] = instruction + config["system_instruction"]

    return agent_configs


# ── Autopilot context injection ──────────────────────────────


def build_autopilot_instruction(is_autopilot: bool | None) -> str:
    """Build an Autopilot context instruction to prepend to agent system prompts.

    Args:
        is_autopilot: ``True`` if the cluster is GKE Autopilot, ``False``
            if Standard, ``None`` if detection was unavailable.

    Returns:
        A system-level instruction telling agents about the cluster mode.
        Returns an empty string when ``is_autopilot`` is ``False`` or ``None``.
    """
    if not is_autopilot:
        return ""

    return (
        "## GKE AUTOPILOT CLUSTER\n\n"
        "This is a GKE Autopilot cluster (confirmed via GKE API). "
        "Node infrastructure is fully managed by Google.\n\n"
        "**Autopilot rules — follow STRICTLY:**\n"
        "- Node data is CONTEXT ONLY — never create findings about node health.\n"
        "- NotReady nodes are NORMAL on Autopilot (Google recycles them). "
        "Do NOT report NotReady as a finding or warning.\n"
        "- kubectl_top(resource_type='nodes') is NOT available.\n"
        "- NEVER recommend node-level actions (scaling, draining, cordoning, "
        "machine types) — Google manages these.\n"
        "- Resource requests are MANDATORY on Autopilot — flag workloads missing them.\n"
        "- Focus ALL analysis on WORKLOAD-LEVEL health: pods, deployments, "
        "services, HPA, events, logs.\n\n"
    )


def inject_autopilot_into_config(
    agent_configs: list[dict[str, Any]],
    is_autopilot: bool | None,
) -> list[dict[str, Any]]:
    """Prepend an Autopilot context instruction to every agent's system prompt.

    Follows the same in-place mutation pattern as
    :func:`inject_language_into_config`.

    Args:
        agent_configs: List of agent configuration dicts.
        is_autopilot: Autopilot detection result from
            :func:`~vaig.tools.gke_tools.detect_autopilot`.

    Returns:
        The same list (mutated), for convenience.
    """
    instruction = build_autopilot_instruction(is_autopilot)
    if not instruction:
        return agent_configs

    for config in agent_configs:
        if "system_instruction" in config:
            config["system_instruction"] = instruction + config["system_instruction"]
    return agent_configs

