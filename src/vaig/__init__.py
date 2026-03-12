"""Vertex AI Gemini Toolkit — Multi-agent AI toolkit powered by Vertex AI."""

import warnings

# Suppress noisy third-party warnings that pollute CLI output:
# 1. requests vs urllib3/charset_normalizer version mismatch (cosmetic, no impact)
# 2. Vertex AI SDK deprecation (sunset June 2026, we'll migrate when ready)
warnings.filterwarnings(
    "ignore",
    message="urllib3.*doesn't match a supported version",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message="This feature is deprecated",
    category=UserWarning,
)

__version__ = "0.1.0"
