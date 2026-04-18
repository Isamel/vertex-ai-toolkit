"""Document fetch tool — downloads a URL, converts HTML to Markdown."""

from __future__ import annotations

import re
import urllib.parse
from typing import TYPE_CHECKING

import html2text
import httpx

from vaig.core.exceptions import ToolExecutionError
from vaig.core.prompt_defense import wrap_untrusted_content

if TYPE_CHECKING:
    from vaig.core.config import DocFetchConfig

_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)


def fetch_doc(
    url: str,
    config: DocFetchConfig,
    allowed_domains: list[str],
    *,
    _run_counter: list[int] | None = None,
) -> str:
    """Fetch a document from an allowed domain and return it as Markdown.

    Args:
        url: The URL to fetch.
        config: Document fetch configuration (byte cap, timeout, per-run cap).
        allowed_domains: List of allowed hostnames.
        _run_counter: Optional mutable counter for per-run cap enforcement.

    Returns:
        Markdown content wrapped with untrusted content delimiters.

    Raises:
        ToolExecutionError: If the domain is not allowed, the per-run cap is
            exhausted, a timeout occurs, or a redirect targets a disallowed domain.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    if hostname not in allowed_domains:
        raise ToolExecutionError(f"fetch_doc: domain not allowed: {hostname}")

    if _run_counter is not None:
        if _run_counter[0] >= config.per_run_cap:
            raise ToolExecutionError(
                f"fetch_doc: per_run_cap exhausted (limit={config.per_run_cap})"
            )
        _run_counter[0] += 1

    try:
        response = httpx.get(url, follow_redirects=False, timeout=config.timeout_seconds)
    except httpx.TimeoutException as exc:
        raise ToolExecutionError(
            f"fetch_doc: request timed out after {config.timeout_seconds}s: {exc}"
        ) from exc

    # Handle redirects manually (max 1 hop)
    if 300 <= response.status_code < 400:
        location = response.headers.get("location", "")
        redirect_parsed = urllib.parse.urlparse(location)
        redirect_hostname = redirect_parsed.hostname or ""
        if redirect_hostname not in allowed_domains:
            raise ToolExecutionError(
                f"fetch_doc: redirect to disallowed domain: {redirect_hostname}"
            )
        try:
            response = httpx.get(
                location,
                follow_redirects=False,
                timeout=config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(
                f"fetch_doc: redirect request timed out after {config.timeout_seconds}s: {exc}"
            ) from exc

    body = response.content[: config.max_bytes]
    text = body.decode("utf-8", errors="replace")

    text = _SCRIPT_RE.sub("", text)
    text = _STYLE_RE.sub("", text)

    h = html2text.HTML2Text()
    h.ignore_links = False
    markdown = h.handle(text)

    return wrap_untrusted_content(markdown)
