"""Event-loop-local keyed asyncio locks with deterministic idle cleanup."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    participants: int = 0


class AsyncKeyedLockTable[KeyT: Hashable]:
    """Serialize same-key coroutines without retaining idle keys."""

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
            entry.participants += 1

        try:
            async with entry.lock:
                yield
        finally:
            with self._guard:
                entry.participants -= 1
                entries = self._entries_by_loop.get(loop)
                if entry.participants == 0 and entries is not None and entries.get(key) is entry:
                    entries.pop(key, None)
                    if not entries:
                        self._entries_by_loop.pop(loop, None)

    def _entry_count(self, loop: asyncio.AbstractEventLoop | None = None) -> int:
        if loop is None:
            loop = asyncio.get_running_loop()
        with self._guard:
            entries = self._entries_by_loop.get(loop)
            return len(entries) if entries is not None else 0
