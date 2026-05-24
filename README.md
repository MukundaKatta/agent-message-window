# agent-message-window

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-message-window.svg)](https://pypi.org/project/agent-message-window/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sliding window of recent LLM conversation turns, with paired-protection: never drop a `tool_use` without its `tool_result` sibling.** Zero deps.

```python
from agent_message_window import MessageWindow

win = MessageWindow(max_turns=20, paired_protect=True)

win.add({"role": "user", "content": "search for X"})
win.add({"role": "assistant", "content": [
    {"type": "tool_use", "id": "u1", "name": "search", "input": {"q": "X"}},
]})
win.add({"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "u1", "content": "found Y"},
]})

win.messages          # current window, oldest → newest
win.evicted_count     # total messages dropped to date
```

## Why

A naive "keep the last N messages" loop breaks the moment the boundary falls between a `tool_use` and its `tool_result`. Anthropic's API rejects the request: a tool_result needs its tool_use in history (and a tool_use needs its tool_result on the next user turn).

`MessageWindow` is the smallest version of that bookkeeping. When the oldest message is an assistant `tool_use`, the eviction step also drops any `tool_result` messages that reference the evicted tool_use IDs. Pair-aware eviction is `paired_protect=True` by default; turn it off if your downstream sanitization handles orphans separately.

For *token-aware* truncation (vs turn-count), use [`agentfit`](https://github.com/MukundaKatta/agentfit). For full conversation persistence, use [`conversation-codec`](https://github.com/MukundaKatta/conversation-codec).

## Install

```bash
pip install agent-message-window
```

## API

```python
win = MessageWindow(max_turns: int, *, paired_protect: bool = True)

win.add(message: dict)
win.extend(messages: Iterable[dict])
win.clear()

win.messages -> list[dict]       # copy, oldest first
len(win); iter(win)
win.max_turns -> int
win.evicted_count -> int         # cumulative drops
```

`add()` rejects non-dict input with `TypeError`. Stored messages are shallow-copied so caller-side mutation doesn't corrupt the window.

## Companion libraries

- [`agentfit`](https://github.com/MukundaKatta/agentfit) — token-aware truncation when turn-count alone isn't enough.
- [`conversation-codec`](https://github.com/MukundaKatta/conversation-codec) — JSONL save/load with optional redaction + Fernet encryption.
- [`prompt-token-counter`](https://github.com/MukundaKatta/prompt-token-counter) — measure window cost before sending.

## License

MIT
