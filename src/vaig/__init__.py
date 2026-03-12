"""Vertex AI Gemini Toolkit — Multi-agent AI toolkit powered by Vertex AI."""

import warnings

# Suppress noisy third-party warnings that pollute CLI output:
# requests vs urllib3/charset_normalizer version mismatch (cosmetic, no impact)
warnings.filterwarnings(
    "ignore",
    message="urllib3.*doesn't match a supported version",
    category=Warning,
)

__version__ = "0.1.0"
