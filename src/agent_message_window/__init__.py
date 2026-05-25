"""Sliding message window that preserves tool_use/tool_result pairs."""

from __future__ import annotations

from agent_message_window.core import AgentMessageWindow, WindowOverflowError

__all__ = ["AgentMessageWindow", "WindowOverflowError"]
__version__ = "0.1.0"
