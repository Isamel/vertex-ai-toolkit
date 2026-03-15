"""Tests for async_utils — run_sync, to_async, gather_with_errors."""

from __future__ import annotations

import asyncio

import pytest

from vaig.core.async_utils import gather_with_errors, run_sync, to_async


# ══════════════════════════════════════════════════════════════
# run_sync
# ══════════════════════════════════════════════════════════════


class TestRunSync:
    """Tests for run_sync() — bridge from sync to async."""

    def test_runs_simple_coroutine(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b

        result = run_sync(add(3, 4))
        assert result == 7

    def test_returns_correct_type(self) -> None:
        async def greeting() -> str:
            return "hello"

        result = run_sync(greeting())
        assert isinstance(result, str)
        assert result == "hello"

    def test_propagates_exceptions(self) -> None:
        async def fail() -> None:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            run_sync(fail())

    def test_works_with_asyncio_sleep(self) -> None:
        async def delayed() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = run_sync(delayed())
        assert result == "done"

    def test_runs_from_within_running_loop(self) -> None:
        """run_sync should work even when called from inside an async context."""

        async def inner() -> int:
            return 42

        async def outer() -> int:
            # Calling run_sync from inside an async context
            return run_sync(inner())

        result = asyncio.run(outer())
        assert result == 42

    def test_handles_none_return(self) -> None:
        async def noop() -> None:
            pass

        result = run_sync(noop())
        assert result is None


# ══════════════════════════════════════════════════════════════
# to_async
# ══════════════════════════════════════════════════════════════


class TestToAsync:
    """Tests for to_async() — decorator wrapping sync functions."""

    async def test_wraps_sync_function(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        async_add = to_async(add)
        result = await async_add(2, 3)
        assert result == 5

    async def test_preserves_function_name(self) -> None:
        def my_func() -> str:
            return "test"

        wrapped = to_async(my_func)
        assert wrapped.__name__ == "my_func"

    async def test_handles_kwargs(self) -> None:
        def greet(name: str, prefix: str = "Hello") -> str:
            return f"{prefix}, {name}"

        async_greet = to_async(greet)
        result = await async_greet("World", prefix="Hi")
        assert result == "Hi, World"

    async def test_propagates_exception(self) -> None:
        def broken() -> None:
            msg = "sync error"
            raise RuntimeError(msg)

        async_broken = to_async(broken)
        with pytest.raises(RuntimeError, match="sync error"):
            await async_broken()

    async def test_used_as_decorator(self) -> None:
        @to_async
        def compute(x: int) -> int:
            return x * x

        result = await compute(5)
        assert result == 25

    async def test_runs_in_separate_thread(self) -> None:
        """Verify the function actually runs in a different thread."""
        import threading

        main_thread_id = threading.current_thread().ident

        def get_thread_id() -> int | None:
            return threading.current_thread().ident

        async_get_id = to_async(get_thread_id)
        worker_thread_id = await async_get_id()
        assert worker_thread_id != main_thread_id


# ══════════════════════════════════════════════════════════════
# gather_with_errors
# ══════════════════════════════════════════════════════════════


class TestGatherWithErrors:
    """Tests for gather_with_errors() — enhanced asyncio.gather."""

    async def test_gathers_multiple_coroutines(self) -> None:
        async def double(x: int) -> int:
            return x * 2

        results = await gather_with_errors(double(1), double(2), double(3))
        assert results == [2, 4, 6]

    async def test_empty_input_returns_empty_list(self) -> None:
        results = await gather_with_errors()
        assert results == []

    async def test_single_coroutine(self) -> None:
        async def single() -> str:
            return "only"

        results = await gather_with_errors(single())
        assert results == ["only"]

    async def test_propagates_first_exception(self) -> None:
        async def ok() -> str:
            return "ok"

        async def fail() -> str:
            msg = "error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="error"):
            await gather_with_errors(ok(), fail())

    async def test_return_exceptions_captures_errors(self) -> None:
        async def ok() -> str:
            return "ok"

        async def fail() -> str:
            msg = "captured"
            raise ValueError(msg)

        results = await gather_with_errors(ok(), fail(), return_exceptions=True)
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "captured"

    async def test_timeout_raises_on_slow_coroutine(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await gather_with_errors(slow(), timeout=0.05)

    async def test_timeout_succeeds_on_fast_coroutines(self) -> None:
        async def fast(x: int) -> int:
            await asyncio.sleep(0.01)
            return x

        results = await gather_with_errors(fast(1), fast(2), timeout=5.0)
        assert results == [1, 2]

    async def test_concurrent_execution(self) -> None:
        """Verify coroutines actually run concurrently (not sequentially)."""
        import time

        async def sleep_and_return(delay: float, val: int) -> int:
            await asyncio.sleep(delay)
            return val

        start = time.monotonic()
        results = await gather_with_errors(
            sleep_and_return(0.1, 1),
            sleep_and_return(0.1, 2),
            sleep_and_return(0.1, 3),
        )
        elapsed = time.monotonic() - start

        assert sorted(results) == [1, 2, 3]
        # If concurrent, total time ~ 0.1s; if sequential, ~ 0.3s
        assert elapsed < 0.25
