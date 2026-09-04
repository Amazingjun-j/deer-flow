"""Async per-loop keyed serialization with idle-entry reclamation."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    refs: int = 0  # holders + queued waiters


class AsyncKeyedLockTable[KeyT: Hashable]:
    """Serialize same-key async work without retaining idle keys.

    ``asyncio.Lock`` becomes event-loop-affine once it has contended, so
    each loop gets an independent key table. A checkout increments
    ``refs`` before waiting for the lock; both successful holders and
    cancelled waiters check the same entry back in. The entry is removed
    only after the last participant leaves, preventing a later caller
    from creating a second lock while an earlier waiter is still queued.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[KeyT, _LockEntry]] = weakref.WeakKeyDictionary()

    @asynccontextmanager
    async def hold(self, key: KeyT) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        with self._guard:
            entries = self._entries_by_loop.get(loop)
            if entries is None:
                entries = {}
                self._entries_by_loop[loop] = entries
            entry = entries.get(key)
            if entry is None:
                entry = _LockEntry()
                entries[key] = entry
            entry.refs += 1

        try:
            async with entry.lock:
                yield
        finally:
            with self._guard:
                entry.refs -= 1
                if entry.refs == 0 and entries.get(key) is entry:
                    entries.pop(key, None)
                    if not entries and self._entries_by_loop.get(loop) is entries:
                        self._entries_by_loop.pop(loop, None)
