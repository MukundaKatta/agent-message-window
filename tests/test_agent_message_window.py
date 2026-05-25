"""Tests for agent_message_window."""

from __future__ import annotations

import pytest

from agent_message_window import AgentMessageWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(content: str = "Hello") -> dict:
    return {"role": "user", "content": content}


def _asst(content: str = "Hi!") -> dict:
    return {"role": "assistant", "content": content}


def _sys(content: str = "Be helpful.") -> dict:
    return {"role": "system", "content": content}


def _tool_use(tool_id: str = "t1", name: str = "search") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _tool_result(tool_id: str = "t1", output: str = "result") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": output}],
    }


# ---------------------------------------------------------------------------
# Constructor / repr / dunder
# ---------------------------------------------------------------------------


def test_repr():
    w = AgentMessageWindow(max_messages=10)
    assert "max_messages=10" in repr(w)
    assert "count=0" in repr(w)


def test_max_messages_property():
    assert AgentMessageWindow(max_messages=5).max_messages == 5


def test_unbounded_window():
    w = AgentMessageWindow()
    assert w.max_messages is None


def test_invalid_max_messages():
    with pytest.raises(ValueError):
        AgentMessageWindow(max_messages=0)


def test_len():
    w = AgentMessageWindow()
    w.add(_user())
    assert len(w) == 1


# ---------------------------------------------------------------------------
# add / count / is_empty
# ---------------------------------------------------------------------------


def test_add_returns_self():
    w = AgentMessageWindow()
    assert w.add(_user()) is w


def test_count_after_add():
    w = AgentMessageWindow()
    w.add(_user()).add(_asst())
    assert w.count == 2


def test_is_empty_initially():
    assert AgentMessageWindow().is_empty is True


def test_is_empty_after_add():
    w = AgentMessageWindow()
    w.add(_user())
    assert w.is_empty is False


def test_add_many():
    w = AgentMessageWindow()
    w.add_many([_user("a"), _asst("b"), _user("c")])
    assert w.count == 3


def test_add_many_returns_self():
    w = AgentMessageWindow()
    assert w.add_many([_user()]) is w


# ---------------------------------------------------------------------------
# clear / replace
# ---------------------------------------------------------------------------


def test_clear():
    w = AgentMessageWindow()
    w.add_many([_user(), _asst()])
    w.clear()
    assert w.count == 0
    assert w.is_empty is True


def test_clear_returns_self():
    w = AgentMessageWindow()
    assert w.clear() is w


def test_replace():
    w = AgentMessageWindow()
    w.add(_user("old"))
    w.replace([_user("new"), _asst("reply")])
    assert w.count == 2
    assert w.messages[-1]["content"] == "reply"


def test_replace_returns_self():
    w = AgentMessageWindow()
    assert w.replace([]) is w


# ---------------------------------------------------------------------------
# messages property — deep copy
# ---------------------------------------------------------------------------


def test_messages_deep_copy():
    w = AgentMessageWindow()
    w.add(_user("original"))
    msgs = w.messages
    msgs[0]["content"] = "mutated"
    # Re-read should still return original
    assert w.messages[0]["content"] == "original"


def test_messages_returns_list():
    w = AgentMessageWindow()
    w.add(_user())
    assert isinstance(w.messages, list)


# ---------------------------------------------------------------------------
# Sliding window — simple messages
# ---------------------------------------------------------------------------


def test_window_trims_oldest():
    w = AgentMessageWindow(max_messages=3)
    for i in range(5):
        w.add(_user(str(i)))
    assert w.count == 3
    contents = [m["content"] for m in w.messages]
    assert "0" not in contents
    assert "4" in contents


def test_window_at_exact_limit():
    w = AgentMessageWindow(max_messages=2)
    w.add(_user("a")).add(_asst("b"))
    assert w.count == 2


def test_unbounded_window_grows():
    w = AgentMessageWindow()
    for i in range(20):
        w.add(_user(str(i)))
    assert w.count == 20


# ---------------------------------------------------------------------------
# Tool-use / tool-result pair preservation
# ---------------------------------------------------------------------------


def test_tool_pair_kept_together_when_fits():
    w = AgentMessageWindow(max_messages=4)
    w.add(_user("q"))
    w.add(_tool_use("t1"))
    w.add(_tool_result("t1"))
    w.add(_asst("done"))
    assert w.count == 4


def test_tool_pair_dropped_atomically():
    # Window of 3 with 4 messages: 1 plain user + tool_use + tool_result + asst
    # Oldest droppable group is either the single user msg or the pair.
    # The single user is oldest -> drops first, leaving 3.
    w = AgentMessageWindow(max_messages=3)
    w.add(_user("first"))
    w.add(_tool_use("t1"))
    w.add(_tool_result("t1"))
    w.add(_asst("final"))
    # "first" should have been dropped; tool pair + asst remain
    msgs = w.messages
    assert w.count == 3
    # all three remaining should be present
    assert len(msgs) == 3


def test_tool_pair_never_split():
    # Window of 2: force dropping until only 2 remain.
    # If we have tool_use + tool_result that's size-2 -> fits exactly.
    w = AgentMessageWindow(max_messages=2)
    w.add(_user("a"))
    w.add(_user("b"))
    w.add(_tool_use("t1"))
    w.add(_tool_result("t1"))
    # After adding the pair, the pair = 2 msgs, so "a" and "b" were dropped
    assert w.count == 2
    # Both should be tool-related
    for m in w.messages:
        content = m.get("content", "")
        if isinstance(content, list):
            types = {b.get("type") for b in content if isinstance(b, dict)}
            assert types & {"tool_use", "tool_result"}


def test_multiple_tool_pairs():
    w = AgentMessageWindow(max_messages=4)
    w.add(_tool_use("t1"))
    w.add(_tool_result("t1"))
    w.add(_tool_use("t2"))
    w.add(_tool_result("t2"))
    assert w.count == 4
    # Add one more plain message — oldest pair should drop
    w.add(_asst("done"))
    assert w.count == 3


# ---------------------------------------------------------------------------
# system message prepended
# ---------------------------------------------------------------------------


def test_system_prepended():
    w = AgentMessageWindow(max_messages=5, system=_sys("System prompt"))
    w.add(_user("Hi"))
    msgs = w.messages
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "System prompt"
    assert msgs[1]["role"] == "user"


def test_system_not_counted_in_window():
    w = AgentMessageWindow(max_messages=2, system=_sys())
    w.add(_user("a")).add(_asst("b"))
    # count is only non-system messages
    assert w.count == 2
    # messages has system + 2 = 3 total
    assert len(w.messages) == 3


def test_system_deep_copied():
    sys_msg = _sys("original")
    w = AgentMessageWindow(system=sys_msg)
    sys_msg["content"] = "mutated"
    # Window should still have original
    assert w.messages[0]["content"] == "original"


# ---------------------------------------------------------------------------
# last / first
# ---------------------------------------------------------------------------


def test_last_default():
    w = AgentMessageWindow()
    w.add_many([_user("a"), _asst("b"), _user("c")])
    result = w.last()
    assert len(result) == 1
    assert result[0]["content"] == "c"


def test_last_n():
    w = AgentMessageWindow()
    w.add_many([_user("a"), _asst("b"), _user("c")])
    result = w.last(2)
    assert [m["content"] for m in result] == ["b", "c"]


def test_last_clamps():
    w = AgentMessageWindow()
    w.add(_user("only"))
    assert len(w.last(100)) == 1


def test_first_n():
    w = AgentMessageWindow()
    w.add_many([_user("a"), _asst("b"), _user("c")])
    result = w.first(2)
    assert [m["content"] for m in result] == ["a", "b"]


def test_first_zero():
    w = AgentMessageWindow()
    w.add(_user())
    assert w.first(0) == []


def test_last_zero():
    w = AgentMessageWindow()
    w.add(_user())
    assert w.last(0) == []
