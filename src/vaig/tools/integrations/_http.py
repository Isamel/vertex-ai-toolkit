"""Shared HTTP helpers for alert correlation tools."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from vaig.tools.base import ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: int = 10
_DEFAULT_MAX_RETRIES: int = 1


def api_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    service_name: str = "API",
) -> tuple[dict[str, Any] | None, ToolResult | None]:
    """Make an HTTP request with timeout and retry on 5xx.

    Returns ``(json_data, None)`` on success or ``(None, error_result)`` on
    failure.  The *service_name* is used only in error messages to identify
    which integration failed.
    """
    last_exc: Exception | None = None
    attempts = 1 + max_retries

    for attempt in range(attempts):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            if resp.status_code >= 500 and attempt < attempts - 1:
                time.sleep(1)
                continue
            if resp.status_code == 401:
                return None, ToolResult(
                    output=f"{service_name} authentication failed (401). Check credentials.",
                    error=True,
                )
            if resp.status_code == 403:
                return None, ToolResult(
                    output=f"{service_name} access denied (403). Check API key permissions.",
                    error=True,
                )
            if resp.status_code == 429:
                return None, ToolResult(
                    output=f"{service_name} rate limited (429). Try again later.",
                    error=True,
                )
            if resp.status_code >= 400:
                return None, ToolResult(
                    output=f"{service_name} API unavailable: {resp.status_code}",
                    error=True,
                )
            return resp.json(), None
        except requests.Timeout:
            last_exc = requests.Timeout(f"{service_name} request timed out after {timeout}s")
            if attempt < attempts - 1:
                time.sleep(1)
                continue
        except requests.ConnectionError:
            last_exc = requests.ConnectionError(f"{service_name} connection failed")
            if attempt < attempts - 1:
                time.sleep(1)
                continue
        except requests.RequestException as exc:
            return None, ToolResult(
                output=f"{service_name} request failed: {type(exc).__name__}",
                error=True,
            )

    # Exhausted retries
    error_msg = str(last_exc) if last_exc else f"{service_name} request failed"
    return None, ToolResult(output=error_msg, error=True)
