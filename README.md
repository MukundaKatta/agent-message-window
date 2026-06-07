# agent-message-window

A sliding message window for LLM agent loops that keeps `tool_use`/`tool_result` pairs intact.

The Anthropic Messages API rejects a message list where a `tool_use` block appears without a matching `tool_result` in the next message. This library's `AgentMessageWindow` enforces that constraint automatically when trimming old messages.

## Install

```bash
pip install agent-message-window
```

## Why

When an LLM agent runs in a loop, the message history grows without bound. The
usual fix — drop the oldest messages once you cross a limit — is unsafe for
tool-calling agents: dropping an assistant `tool_use` message but keeping the
following `tool_result` (or vice versa) produces a list the API rejects.

`AgentMessageWindow` is a drop-in replacement for "just a list" that trims by
*atomic groups*: a `tool_use` message and its matching `tool_result` message are
always kept together or dropped together, so `.messages` is always a valid
request body.

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

### In an agent loop

```python
from agent_message_window import AgentMessageWindow

window = AgentMessageWindow(max_messages=40, system=SYSTEM_PROMPT)
window.add({"role": "user", "content": user_input})

while True:
    response = client.messages.create(model=MODEL, messages=window.messages, tools=TOOLS)
    window.add({"role": "assistant", "content": response.content})

    tool_uses = [b for b in response.content if b.type == "tool_use"]
    if not tool_uses:
        break

    # Return every result in a single user message so it stays paired.
    results = [
        {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b)}
        for b in tool_uses
    ]
    window.add({"role": "user", "content": results})
```

Because the window only ever trims whole `tool_use`/`tool_result` groups,
`window.messages` is always safe to pass straight back to the API — no matter
how long the loop runs.

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
| `WindowOverflowError` | A single atomic `tool_use`/`tool_result` group is larger than `max_messages`, so it can never fit. |

When `add(...)` or `replace(...)` raises `WindowOverflowError`, the window is
left **unchanged** — the failed operation is rolled back, so you can catch the
error and keep using the existing window. (Constructing with `max_messages < 1`
raises `ValueError`.)

## Development

The test suite uses only the standard library, so no extra packages are needed
to run it:

```bash
python3 -m unittest discover -s tests
```

Linting and formatting use [ruff](https://docs.astral.sh/ruff/) (installed via
the `dev` extra):

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
```

## License

MIT
