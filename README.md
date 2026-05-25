# agent-message-window

A sliding message window for LLM agent loops that keeps `tool_use`/`tool_result` pairs intact.

The Anthropic Messages API rejects a message list where a `tool_use` block appears without a matching `tool_result` in the next message. This library's `AgentMessageWindow` enforces that constraint automatically when trimming old messages.

## Install

```bash
pip install agent-message-window
```

## Usage

```python
from agent_message_window import AgentMessageWindow

window = AgentMessageWindow(
    max_messages=20,
    system={"role": "system", "content": "Be helpful."},
)

# Add messages as your agent loop runs
window.add({"role": "user", "content": "Search for climate data"})
window.add({
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}],
})
window.add({
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "..."}],
})

# Safe to send — tool pairs are never split when trimming
messages = window.messages
```

## API

### `AgentMessageWindow(max_messages=None, *, system=None)`

| Parameter | Description |
|-----------|-------------|
| `max_messages` | Maximum non-system messages. `None` = unbounded. |
| `system` | Optional system message prepended to `.messages` (not counted toward the window). |

### Mutations (all chainable)

| Method | Description |
|--------|-------------|
| `add(message)` | Append one message, trim oldest if needed. |
| `add_many(messages)` | Append multiple messages. |
| `clear()` | Remove all messages. |
| `replace(messages)` | Replace the entire window. |

### Queries

| Property/Method | Description |
|-----------------|-------------|
| `messages` | Deep copy of the current window (with system prepended if set). |
| `count` | Number of non-system messages. |
| `is_empty` | `True` when the window is empty. |
| `max_messages` | Configured window size. |
| `last(n=1)` | Last *n* messages (deep copy, no system). |
| `first(n=1)` | First *n* messages (deep copy, no system). |

### Exceptions

| Exception | When |
|-----------|------|
| `WindowOverflowError` | A single atomic group exceeds `max_messages`. |

## License

MIT
