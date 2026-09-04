"""Regression tests for reclaimable runtime keyed locks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import pytest

from deerflow.runtime import goal as goal_runtime
from deerflow.runtime.runs import worker as run_worker

LockFactory = Callable[[str], AbstractAsyncContextManager[None]]

CASES = (
    pytest.param(goal_runtime.goal_thread_lock, goal_runtime._goal_thread_locks, id="goal"),
    pytest.param(run_worker._checkpoint_thread_lock, run_worker._checkpoint_thread_locks, id="checkpoint"),
)


def _participant_count(table, key: str) -> int:
    loop = asyncio.get_running_loop()
    with table._guard:
        entries = table._entries_by_loop.get(loop)
        if entries is None:
            return 0
        entry = entries.get(key)
        return entry.participants if entry is not None else 0


async def _wait_for_participants(table, key: str, expected: int) -> None:
    for _ in range(100):
        if _participant_count(table, key) == expected:
            return
        await asyncio.sleep(0)
    assert _participant_count(table, key) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("lock_factory", "table"), CASES)
async def test_idle_entries_are_reclaimed(lock_factory: LockFactory, table) -> None:
    for index in range(512):
        async with lock_factory(f"completed-{index}"):
            pass
    assert table._entry_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("lock_factory", "table"), CASES)
async def test_waiter_cannot_be_bypassed(lock_factory: LockFactory, table) -> None:
    key = "handoff"
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    release_second = asyncio.Event()
    third_entered = asyncio.Event()

    async def first() -> None:
        async with lock_factory(key):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with lock_factory(key):
            second_entered.set()
            await release_second.wait()

    async def third() -> None:
        async with lock_factory(key):
            third_entered.set()

    first_task = asyncio.create_task(first())
    await asyncio.wait_for(first_entered.wait(), 2)
    second_task = asyncio.create_task(second())
    await _wait_for_participants(table, key, 2)

    release_first.set()
    await asyncio.wait_for(first_task, 2)
    await asyncio.wait_for(second_entered.wait(), 2)
    assert _participant_count(table, key) == 1

    third_task = asyncio.create_task(third())
    await _wait_for_participants(table, key, 2)
    await asyncio.sleep(0)
    assert not third_entered.is_set()

    release_second.set()
    await asyncio.wait_for(asyncio.gather(second_task, third_task), 2)
    assert table._entry_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("lock_factory", "table"), CASES)
async def test_cancelled_waiter_checks_in(lock_factory: LockFactory, table) -> None:
    key = "cancel"
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with lock_factory(key):
            holder_entered.set()
            await release_holder.wait()

    holder_task = asyncio.create_task(holder())
    await asyncio.wait_for(holder_entered.wait(), 2)
    waiter_task = asyncio.create_task(_hold_once(lock_factory, key))
    await _wait_for_participants(table, key, 2)

    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task
    assert _participant_count(table, key) == 1

    release_holder.set()
    await asyncio.wait_for(holder_task, 2)
    assert table._entry_count() == 0


async def _hold_once(lock_factory: LockFactory, key: str) -> None:
    async with lock_factory(key):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(("lock_factory", "table"), CASES)
async def test_different_keys_remain_independent(lock_factory: LockFactory, table) -> None:
    release_first = asyncio.Event()
    first_entered = asyncio.Event()
    other_entered = asyncio.Event()

    async def first() -> None:
        async with lock_factory("a"):
            first_entered.set()
            await release_first.wait()

    async def other() -> None:
        async with lock_factory("b"):
            other_entered.set()

    first_task = asyncio.create_task(first())
    await asyncio.wait_for(first_entered.wait(), 2)
    other_task = asyncio.create_task(other())
    await asyncio.wait_for(other_entered.wait(), 2)
    await other_task
    release_first.set()
    await first_task
    assert table._entry_count() == 0
