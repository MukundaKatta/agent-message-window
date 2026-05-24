"""Tests for agent_message_window.MessageWindow."""

from __future__ import annotations

import pytest

from agent_message_window import MessageWindow


# ---- basic FIFO ---------------------------------------------------------


def test_keeps_up_to_max_turns():
    w = MessageWindow(max_turns=3)
    for i in range(5):
        w.add({"role": "user", "content": f"msg-{i}"})
    assert len(w) == 3
    assert [m["content"] for m in w.messages] == ["msg-2", "msg-3", "msg-4"]


def test_evicted_count_increments():
    w = MessageWindow(max_turns=2)
    w.add({"role": "user", "content": "a"})
    w.add({"role": "user", "content": "b"})
    w.add({"role": "user", "content": "c"})
    assert w.evicted_count == 1


def test_under_capacity_no_eviction():
    w = MessageWindow(max_turns=10)
    w.add({"role": "user", "content": "a"})
    w.add({"role": "assistant", "content": "b"})
    assert w.evicted_count == 0
    assert len(w) == 2


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError):
        MessageWindow(max_turns=0)
    with pytest.raises(ValueError):
        MessageWindow(max_turns=-1)


def test_add_rejects_non_dict():
    w = MessageWindow(max_turns=3)
    with pytest.raises(TypeError):
        w.add("not a dict")  # type: ignore[arg-type]


def test_add_stores_shallow_copy():
    w = MessageWindow(max_turns=3)
    m = {"role": "user", "content": "x"}
    w.add(m)
    m["content"] = "MUTATED"
    assert w.messages[0]["content"] == "x"


def test_extend_takes_iterable():
    w = MessageWindow(max_turns=3)
    w.extend([
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ])
    assert len(w) == 2


def test_clear_resets_buffer_and_counter():
    w = MessageWindow(max_turns=2)
    w.add({"role": "user", "content": "a"})
    w.add({"role": "user", "content": "b"})
    w.add({"role": "user", "content": "c"})  # evicted=1
    w.clear()
    assert len(w) == 0
    assert w.evicted_count == 0


def test_iter_yields_messages_oldest_first():
    w = MessageWindow(max_turns=3)
    w.add({"role": "user", "content": "1"})
    w.add({"role": "user", "content": "2"})
    contents = [m["content"] for m in w]
    assert contents == ["1", "2"]


def test_messages_property_returns_copy():
    w = MessageWindow(max_turns=3)
    w.add({"role": "user", "content": "x"})
    snap = w.messages
    snap.append({"role": "fake"})
    assert len(w) == 1


# ---- paired protection -------------------------------------------------


def _tu(tu_id, name="search", inp=None):
    return {"role": "assistant", "content": [{
        "type": "tool_use", "id": tu_id, "name": name, "input": inp or {}
    }]}


def _tr(tu_id, content="ok"):
    return {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": tu_id, "content": content,
    }]}


def test_paired_protect_drops_tool_result_when_tool_use_evicted():
    w = MessageWindow(max_turns=3, paired_protect=True)
    # window will hold up to 3
    w.add(_tu("u1"))                                  # 1 in window
    w.add(_tr("u1"))                                  # 2 in window
    w.add({"role": "user", "content": "next turn"})   # 3 in window
    w.add({"role": "user", "content": "follow up"})   # overflow → drop oldest
    # oldest was the tool_use(u1). It was evicted; paired_protect should
    # also drop the tool_result that referenced u1.
    messages = w.messages
    # Make sure no orphan tool_result remains
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                assert not (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == "u1"
                ), "orphan tool_result for u1 remains in window"


def test_paired_protect_off_leaves_orphans():
    w = MessageWindow(max_turns=3, paired_protect=False)
    w.add(_tu("u1"))
    w.add(_tr("u1"))
    w.add({"role": "user", "content": "x"})
    w.add({"role": "user", "content": "y"})
    # tool_result(u1) should now be in the window without its tool_use parent
    found_orphan = False
    for msg in w.messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    found_orphan = True
    assert found_orphan


def test_paired_protect_handles_multi_tool_use():
    w = MessageWindow(max_turns=4, paired_protect=True)
    # One assistant message with two tool_uses
    w.add({"role": "assistant", "content": [
        {"type": "tool_use", "id": "u1", "name": "a", "input": {}},
        {"type": "tool_use", "id": "u2", "name": "b", "input": {}},
    ]})
    w.add(_tr("u1"))
    w.add(_tr("u2"))
    w.add({"role": "user", "content": "filler-1"})
    # window is now full at 4
    w.add({"role": "user", "content": "filler-2"})  # overflow → evict oldest
    # Both tool_results should be gone (paired with the assistant message)
    for msg in w.messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                assert not (
                    isinstance(block, dict) and block.get("type") == "tool_result"
                )


def test_paired_protect_does_not_drop_unrelated_messages():
    w = MessageWindow(max_turns=3, paired_protect=True)
    w.add(_tu("u1"))
    w.add({"role": "user", "content": "unrelated"})  # NOT a tool_result
    w.add({"role": "assistant", "content": "ok"})
    w.add({"role": "user", "content": "newest"})
    # Should have evicted only the oldest. No paired drop needed.
    assert len(w.messages) == 3
    assert w.evicted_count == 1


def test_paired_protect_counts_paired_drops():
    w = MessageWindow(max_turns=2, paired_protect=True)
    w.add(_tu("u1"))
    w.add(_tr("u1"))
    w.add({"role": "user", "content": "next"})
    # When tool_use(u1) was evicted, tool_result(u1) was also dropped.
    # That's 2 evictions for the one overflow.
    assert w.evicted_count == 2


def test_window_with_max_1():
    w = MessageWindow(max_turns=1)
    w.add({"role": "user", "content": "a"})
    w.add({"role": "user", "content": "b"})
    assert w.messages == [{"role": "user", "content": "b"}]


def test_string_content_messages_pair_protect_unaffected():
    w = MessageWindow(max_turns=2, paired_protect=True)
    w.add({"role": "user", "content": "plain text 1"})
    w.add({"role": "user", "content": "plain text 2"})
    w.add({"role": "user", "content": "plain text 3"})
    # nothing fancy to pair; standard FIFO
    assert len(w) == 2
    assert w.evicted_count == 1


def test_max_turns_property():
    w = MessageWindow(max_turns=5)
    assert w.max_turns == 5
