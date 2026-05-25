"""Sliding message window that preserves tool_use/tool_result pairs.

The Anthropic Messages API rejects a message list where a ``tool_use``
content block appears without a subsequent ``tool_result`` block in the
next message. This module's :class:`AgentMessageWindow` keeps a rolling
window of the most recent messages while never breaking those pairs when
the window shrinks.

Example::

    from agent_message_window import AgentMessageWindow

    window = AgentMessageWindow(max_messages=10)
    window.add({"role": "user", "content": "Hello"})
    window.add({"role": "assistant", "content": "Hi!"})

    messages = window.messages  # safe to send to the API
"""

from __future__ import annotations

import copy
from typing import Any


class WindowOverflowError(RuntimeError):
    """Raised when a single atomic group is larger than the window."""


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
    """Return the set of tool_use ids present in *message*."""
    content = message.get("content", "")
    if isinstance(content, list):
        return {
            b["id"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
        }
    return set()


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    """Return the set of tool_use_id values in tool_result blocks."""
    content = message.get("content", "")
    if isinstance(content, list):
        return {
            b["tool_use_id"]
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and "tool_use_id" in b
        }
    return set()


def _group_messages(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group messages into atomic units that must not be split.

    A ``tool_use`` assistant message and the subsequent ``tool_result`` user
    message form a single group that must be kept or dropped together.
    Everything else is a singleton group.
    """
    groups: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        ids = _tool_use_ids(msg)
        if ids and i + 1 < len(messages):
            next_msg = messages[i + 1]
            result_ids = _tool_result_ids(next_msg)
            if ids & result_ids:
                # This pair is atomic
                groups.append([msg, next_msg])
                i += 2
                continue
        groups.append([msg])
        i += 1
    return groups


class AgentMessageWindow:
    """A rolling window over LLM messages that keeps tool pairs intact.

    Args:
        max_messages: Maximum number of messages in the window.  Defaults
                      to ``None`` (unbounded — acts like a plain list).
        system:       An optional system message prepended to every
                      :attr:`messages` result (not counted against the window).

    Raises:
        WindowOverflowError: If a single atomic group (tool_use + tool_result
                             pair) exceeds *max_messages*, making it impossible
                             to fit.
    """

    def __init__(
        self,
        max_messages: int | None = None,
        *,
        system: dict[str, Any] | None = None,
    ) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        self._max = max_messages
        self._system = copy.deepcopy(system) if system is not None else None
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core mutations
    # ------------------------------------------------------------------

    def add(self, message: dict[str, Any]) -> AgentMessageWindow:
        """Append *message* to the window, trimming oldest if needed.

        When trimming, atomic tool_use/tool_result pairs are always dropped
        together so the resulting list remains API-safe.

        Args:
            message: A ``{"role": ..., "content": ...}`` dict.

        Returns:
            ``self`` for chaining.

        Raises:
            WindowOverflowError: If the message cannot fit even after dropping
                                 all other messages.
        """
        self._messages.append(copy.deepcopy(message))
        self._trim()
        return self

    def add_many(self, messages: list[dict[str, Any]]) -> AgentMessageWindow:
        """Append multiple messages, trimming after each one.

        Args:
            messages: List of message dicts.

        Returns:
            ``self`` for chaining.
        """
        for m in messages:
            self.add(m)
        return self

    def clear(self) -> AgentMessageWindow:
        """Remove all messages from the window.

        Returns:
            ``self`` for chaining.
        """
        self._messages.clear()
        return self

    def replace(self, messages: list[dict[str, Any]]) -> AgentMessageWindow:
        """Replace the entire window with *messages* (deep copy).

        Args:
            messages: New message list.  Must fit within *max_messages*.

        Returns:
            ``self`` for chaining.

        Raises:
            WindowOverflowError: If the replacement list is too large.
        """
        self._messages = [copy.deepcopy(m) for m in messages]
        self._trim()
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The current window as a deep-copied list.

        If a *system* message was provided at construction, it is prepended.
        """
        result = []
        if self._system is not None:
            result.append(copy.deepcopy(self._system))
        result.extend(copy.deepcopy(self._messages))
        return result

    @property
    def count(self) -> int:
        """Number of non-system messages currently in the window."""
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        """``True`` if the window contains no non-system messages."""
        return len(self._messages) == 0

    @property
    def max_messages(self) -> int | None:
        """The configured window size, or ``None`` if unbounded."""
        return self._max

    def last(self, n: int = 1) -> list[dict[str, Any]]:
        """Return the last *n* messages (deep copy, no system prepend).

        Args:
            n: Number of messages to return.  Clamped to :attr:`count`.

        Returns:
            List of the most recent messages.
        """
        n = max(0, min(n, len(self._messages)))
        return copy.deepcopy(self._messages[-n:] if n else [])

    def first(self, n: int = 1) -> list[dict[str, Any]]:
        """Return the first *n* messages (deep copy, no system prepend).

        Args:
            n: Number of messages to return.  Clamped to :attr:`count`.

        Returns:
            List of the oldest messages.
        """
        n = max(0, min(n, len(self._messages)))
        return copy.deepcopy(self._messages[:n])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Trim oldest atomic groups until the window fits."""
        if self._max is None:
            return
        groups = _group_messages(self._messages)
        while sum(len(g) for g in groups) > self._max:
            if not groups:
                break
            groups.pop(0)
            if not groups:
                # Everything was dropped — remaining check happens below.
                break
        # Flatten back
        flat: list[dict[str, Any]] = []
        for g in groups:
            flat.extend(g)
        if self._max is not None and len(flat) > self._max:
            raise WindowOverflowError(
                f"Cannot fit messages in window of {self._max}: "
                f"atomic group of {len(flat)} messages is too large"
            )
        self._messages = flat

    def __repr__(self) -> str:
        return f"AgentMessageWindow(max_messages={self._max}, count={self.count})"

    def __len__(self) -> int:
        return self.count
