"""Async utilities — bridge between sync and async code for backward compatibility.

Provides:
- ``run_sync(coro)`` — Run an async coroutine from a synchronous context.
- ``to_async(fn)`` — Decorator to wrap a blocking function for ``asyncio.to_thread()``.
- ``gather_with_errors(*coros)`` — ``asyncio.gather()`` with timeout and better error handling.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous context.

    Uses ``asyncio.run()`` when no event loop is running.  If called from
    within an already-running loop (e.g. Jupyter, nested sync-from-async),
    creates a new loop in a background thread to avoid ``RuntimeError``.

    This is the primary bridge for backward compatibility — existing sync
    callers can keep calling the same API while the internals are async.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(coro)

    # Already inside a running loop — run in a new thread to avoid
    # "cannot call asyncio.run() while another loop is running".
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def to_async(fn: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator that wraps a synchronous (blocking) function to run in ``asyncio.to_thread()``.

    Useful for wrapping blocking I/O calls (e.g. kubernetes client, file I/O)
    so they don't block the event loop.

    Example::

        @to_async
        def get_pods(namespace: str) -> list[dict]:
            return k8s_client.list_namespaced_pod(namespace)

        # Now callable as: pods = await get_pods("default")

    Args:
        fn: A synchronous callable.

    Returns:
        An async wrapper that delegates to ``asyncio.to_thread()``.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(fn, *args, **kwargs)

    return wrapper


async def gather_with_errors(
    *coros: Coroutine[Any, Any, Any],
    return_exceptions: bool = False,
    timeout: float | None = None,
) -> list[Any]:
    """Wrapper around ``asyncio.gather()`` with optional timeout and better error handling.

    Args:
        *coros: Coroutines to run concurrently.
        return_exceptions: If ``True``, exceptions are returned as results
            instead of being raised (same as ``asyncio.gather``).
        timeout: Optional timeout in seconds. If exceeded, raises
            ``asyncio.TimeoutError``.

    Returns:
        List of results (or exceptions if ``return_exceptions=True``).

    Raises:
        asyncio.TimeoutError: If *timeout* is specified and exceeded.
        Exception: The first exception from any coroutine (when
            ``return_exceptions=False``).
    """
    if not coros:
        return []

    gathered = asyncio.gather(*coros, return_exceptions=return_exceptions)

    if timeout is not None:
        return list(await asyncio.wait_for(gathered, timeout=timeout))

    return list(await gathered)
