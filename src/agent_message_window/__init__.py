"""agent-message-window - sliding window of recent conversation turns.

`MessageWindow(max_turns=N)` keeps the last N messages. Each `add()`
drops the oldest entries until the count fits the cap. Pair-aware
eviction is the trick: if the oldest message is an assistant
`tool_use`, we never evict it without also being willing to evict the
matching `tool_result` that follows. Anthropic's API rejects a
conversation that has a tool_result whose tool_use is no longer in the
history (and vice versa), so the naive "drop the oldest" approach
breaks the next request.

    from agent_message_window import MessageWindow

    win = MessageWindow(max_turns=20, paired_protect=True)

    win.add({"role": "user", "content": "search for X"})
    win.add({"role": "assistant", "content": [
        {"type": "tool_use", "id": "u1", "name": "search", "input": {...}},
    ]})
    win.add({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "u1", "content": "..."},
    ]})

    win.messages          # current window, ordered oldest → newest
    win.evicted_count     # how many messages have been dropped to date

Without `paired_protect`, the window evicts the strict-oldest message
on each overflow. With `paired_protect=True` (the default), an
overflow that *would* leave a dangling tool_result also drops that
result (and so on transitively).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

__version__ = "0.1.0"
__all__ = [
    "MessageWindow",
]


# ---- block helpers --------------------------------------------------------


def _tool_use_ids(message: dict) -> set[str]:
    """Return the set of tool_use IDs in this assistant message."""
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    out: set[str] = set()
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tid = block.get("id")
            if isinstance(tid, str):
                out.add(tid)
    return out


def _tool_result_refs(message: dict) -> set[str]:
    """Return the set of tool_use_ids this user message references."""
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    out: set[str] = set()
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                out.add(tid)
    return out


# ---- main class -----------------------------------------------------------


class MessageWindow:
    """Sliding window over LLM conversation messages.

    Args:
        max_turns: max number of messages retained. Must be >= 1.
        paired_protect: if True, evict tool_results alongside their
            tool_use parents (and vice versa) when one is dropped, so
            the window never contains an unpaired tool_use / tool_result.
            Default: True.
    """

    def __init__(
        self,
        max_turns: int,
        *,
        paired_protect: bool = True,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._max = int(max_turns)
        self._paired = bool(paired_protect)
        self._buf: deque[dict] = deque()
        self._evicted = 0

    # ---- inspection --------------------------------------------------

    @property
    def max_turns(self) -> int:
        return self._max

    @property
    def messages(self) -> list[dict]:
        return list(self._buf)

    @property
    def evicted_count(self) -> int:
        return self._evicted

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)

    # ---- core --------------------------------------------------------

    def add(self, message: dict) -> None:
        """Append a message and evict-to-fit."""
        if not isinstance(message, dict):
            raise TypeError("message must be a dict")
        self._buf.append(dict(message))  # store a shallow copy
        self._evict_to_fit()

    def extend(self, messages: Iterable[dict]) -> None:
        for m in messages:
            self.add(m)

    def clear(self) -> None:
        self._buf.clear()
        self._evicted = 0

    # ---- internals ---------------------------------------------------

    def _evict_to_fit(self) -> None:
        """Drop oldest messages until `len <= max`, honoring paired_protect."""
        while len(self._buf) > self._max:
            oldest = self._buf.popleft()
            self._evicted += 1
            if not self._paired:
                continue
            # Pair removal: if the oldest had tool_use IDs, also drop any
            # remaining messages that reference those IDs in tool_result.
            tu_ids = _tool_use_ids(oldest)
            if tu_ids:
                self._drop_orphan_tool_results(tu_ids)
                continue
            # Conversely: if the oldest had tool_result refs and the
            # corresponding tool_use is no longer in the buffer, the
            # message is already removed so no action. (But if the tu_use
            # IS still in the buffer, the result is a paired sibling we
            # just dropped — that's the desired behavior.)
            # Nothing extra to do here.

    def _drop_orphan_tool_results(self, tu_ids: set[str]) -> None:
        """Remove any messages in the buffer that contain tool_results
        referencing the given tool_use IDs."""
        if not tu_ids:
            return
        kept: deque[dict] = deque()
        for msg in self._buf:
            refs = _tool_result_refs(msg)
            if refs & tu_ids:
                self._evicted += 1
                continue
            kept.append(msg)
        self._buf = kept
