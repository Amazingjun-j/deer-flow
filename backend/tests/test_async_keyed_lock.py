"""Regression coverage for runtime per-thread lock-table reclamation (#5171)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import deerflow.runtime.goal as goal_module
import deerflow.runtime.runs.worker as worker_module
from deerflow.runtime.async_keyed_lock import AsyncKeyedLockTable

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _entry_count(table: Any) -> int:
    return sum(len(entries) for entries in table._entries_by_loop.values())


def _entry_refs(table: Any, key: str) -> int:
    entries = table._entries_by_loop.get(asyncio.get_running_loop(), {})
    entry = entries.get(key)
    return 0 if entry is None else entry.refs


async def _wait_for_refs(table: Any, key: str, expected: int) -> None:
    async with asyncio.timeout(1):
        while _entry_refs(table, key) != expected:
            await asyncio.sleep(0)


async def test_async_keyed_lock_reclaims_many_unique_keys() -> None:
    table = AsyncKeyedLockTable[str]()

    for index in range(1000):
        async with table.hold(f"thread-{index}"):
            pass

    assert _entry_count(table) == 0


async def test_async_keyed_lock_preserves_waiter_order_during_reclamation() -> None:
    table = AsyncKeyedLockTable[str]()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    order: list[str] = []

    async def holder() -> None:
        async with table.hold("same-thread"):
            order.append("holder")
            holder_entered.set()
            await release_holder.wait()

    async def contender(name: str) -> None:
        async with table.hold("same-thread"):
            order.append(name)

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(contender("waiter"))
    await _wait_for_refs(table, "same-thread", 2)

    late_task = asyncio.create_task(contender("late"))
    await _wait_for_refs(table, "same-thread", 3)

    release_holder.set()
    await asyncio.gather(holder_task, waiter_task, late_task)

    assert order == ["holder", "waiter", "late"]
    assert _entry_count(table) == 0


async def test_async_keyed_lock_cancelled_waiter_is_checked_in() -> None:
    table = AsyncKeyedLockTable[str]()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with table.hold("same-thread"):
            holder_entered.set()
            await release_holder.wait()

    async def waiter() -> None:
        async with table.hold("same-thread"):
            pytest.fail("cancelled waiter entered the critical section")

    holder_task = asyncio.create_task(holder())
    await holder_entered.wait()

    waiter_task = asyncio.create_task(waiter())
    await _wait_for_refs(table, "same-thread", 2)

    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    assert _entry_refs(table, "same-thread") == 1
    release_holder.set()
    await holder_task
    assert _entry_count(table) == 0


async def test_async_keyed_lock_keeps_different_keys_independent() -> None:
    table = AsyncKeyedLockTable[str]()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with table.hold("first"):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        await first_entered.wait()
        async with table.hold("second"):
            second_entered.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert _entry_count(table) == 0


@pytest.mark.parametrize(
    ("module", "lock_name", "table_name"),
    [
        (goal_module, "goal_thread_lock", "_goal_thread_locks"),
        (worker_module, "_checkpoint_thread_lock", "_checkpoint_thread_locks"),
    ],
)
async def test_runtime_thread_lock_wrappers_reclaim_historical_thread_ids(
    module: Any,
    lock_name: str,
    table_name: str,
) -> None:
    lock = getattr(module, lock_name)
    for index in range(500):
        async with lock(f"historical-thread-{index}"):
            pass

    table = getattr(module, table_name)
    assert _entry_count(table) == 0
