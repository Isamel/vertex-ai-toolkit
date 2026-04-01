"""Vertex AI Gemini Toolkit — Multi-agent AI toolkit powered by Vertex AI."""

import warnings
from importlib.metadata import PackageNotFoundError, version

# Suppress noisy third-party warnings that pollute CLI output:
# requests vs urllib3/charset_normalizer version mismatch (cosmetic, no impact)
warnings.filterwarnings(
    "ignore",
    message="urllib3.*doesn't match a supported version",
    category=Warning,
)

# urllib3 InsecureRequestWarning when verify=False (common with internal GKE
# endpoints behind custom CAs or skip-TLS proxies — safe to silence globally)
import urllib3.exceptions  # noqa: E402

warnings.filterwarnings(
    "ignore",
    category=urllib3.exceptions.InsecureRequestWarning,
)

# Single source of truth: pyproject.toml [project] version.
# importlib.metadata reads it from the installed package metadata.
# Fallback "0.0.0-dev" covers editable installs where metadata isn't available
# (e.g. running from source without `pip install -e .`).
try:
    __version__ = version("vertex-ai-toolkit")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
