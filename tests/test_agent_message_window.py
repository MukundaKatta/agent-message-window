"""Tests for :mod:`agent_message_window`.

Written with the standard-library :mod:`unittest` framework so the suite runs
on a bare interpreter with no third-party dependencies::

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

# Support running from a source checkout (``src/`` layout) without installing.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_message_window import AgentMessageWindow, WindowOverflowError  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(content: str = "Hello") -> dict[str, Any]:
    return {"role": "user", "content": content}


def _asst(content: str = "Hi!") -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def _sys(content: str = "Be helpful.") -> dict[str, Any]:
    return {"role": "system", "content": content}


def _tool_use(tool_id: str = "t1", name: str = "search") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _tool_result(tool_id: str = "t1", output: str = "result") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": output}
        ],
    }


# ---------------------------------------------------------------------------
# Constructor / repr / dunder
# ---------------------------------------------------------------------------


class ConstructorTests(unittest.TestCase):
    def test_repr(self) -> None:
        w = AgentMessageWindow(max_messages=10)
        self.assertIn("max_messages=10", repr(w))
        self.assertIn("count=0", repr(w))

    def test_max_messages_property(self) -> None:
        self.assertEqual(AgentMessageWindow(max_messages=5).max_messages, 5)

    def test_unbounded_window(self) -> None:
        self.assertIsNone(AgentMessageWindow().max_messages)

    def test_invalid_max_messages_zero(self) -> None:
        with self.assertRaises(ValueError):
            AgentMessageWindow(max_messages=0)

    def test_invalid_max_messages_negative(self) -> None:
        with self.assertRaises(ValueError):
            AgentMessageWindow(max_messages=-3)

    def test_len(self) -> None:
        w = AgentMessageWindow()
        w.add(_user())
        self.assertEqual(len(w), 1)


# ---------------------------------------------------------------------------
# add / count / is_empty
# ---------------------------------------------------------------------------


class AddTests(unittest.TestCase):
    def test_add_returns_self(self) -> None:
        w = AgentMessageWindow()
        self.assertIs(w.add(_user()), w)

    def test_count_after_add(self) -> None:
        w = AgentMessageWindow()
        w.add(_user()).add(_asst())
        self.assertEqual(w.count, 2)

    def test_is_empty_initially(self) -> None:
        self.assertTrue(AgentMessageWindow().is_empty)

    def test_is_empty_after_add(self) -> None:
        w = AgentMessageWindow()
        w.add(_user())
        self.assertFalse(w.is_empty)

    def test_add_many(self) -> None:
        w = AgentMessageWindow()
        w.add_many([_user("a"), _asst("b"), _user("c")])
        self.assertEqual(w.count, 3)

    def test_add_many_returns_self(self) -> None:
        w = AgentMessageWindow()
        self.assertIs(w.add_many([_user()]), w)

    def test_add_deep_copies_input(self) -> None:
        msg = _user("original")
        w = AgentMessageWindow()
        w.add(msg)
        msg["content"] = "mutated"
        self.assertEqual(w.messages[0]["content"], "original")


# ---------------------------------------------------------------------------
# clear / replace
# ---------------------------------------------------------------------------


class ClearReplaceTests(unittest.TestCase):
    def test_clear(self) -> None:
        w = AgentMessageWindow()
        w.add_many([_user(), _asst()])
        w.clear()
        self.assertEqual(w.count, 0)
        self.assertTrue(w.is_empty)

    def test_clear_returns_self(self) -> None:
        w = AgentMessageWindow()
        self.assertIs(w.clear(), w)

    def test_replace(self) -> None:
        w = AgentMessageWindow()
        w.add(_user("old"))
        w.replace([_user("new"), _asst("reply")])
        self.assertEqual(w.count, 2)
        self.assertEqual(w.messages[-1]["content"], "reply")

    def test_replace_returns_self(self) -> None:
        w = AgentMessageWindow()
        self.assertIs(w.replace([]), w)

    def test_replace_deep_copies_input(self) -> None:
        msgs = [_user("a"), _asst("b")]
        w = AgentMessageWindow()
        w.replace(msgs)
        msgs[0]["content"] = "mutated"
        self.assertEqual(w.messages[0]["content"], "a")


# ---------------------------------------------------------------------------
# messages property — deep copy
# ---------------------------------------------------------------------------


class MessagesPropertyTests(unittest.TestCase):
    def test_messages_deep_copy(self) -> None:
        w = AgentMessageWindow()
        w.add(_user("original"))
        msgs = w.messages
        msgs[0]["content"] = "mutated"
        # Re-read should still return the original.
        self.assertEqual(w.messages[0]["content"], "original")

    def test_messages_returns_list(self) -> None:
        w = AgentMessageWindow()
        w.add(_user())
        self.assertIsInstance(w.messages, list)

    def test_messages_empty(self) -> None:
        self.assertEqual(AgentMessageWindow().messages, [])


# ---------------------------------------------------------------------------
# Sliding window — simple messages
# ---------------------------------------------------------------------------


class SlidingWindowTests(unittest.TestCase):
    def test_window_trims_oldest(self) -> None:
        w = AgentMessageWindow(max_messages=3)
        for i in range(5):
            w.add(_user(str(i)))
        self.assertEqual(w.count, 3)
        contents = [m["content"] for m in w.messages]
        self.assertNotIn("0", contents)
        self.assertNotIn("1", contents)
        self.assertEqual(contents, ["2", "3", "4"])

    def test_window_at_exact_limit(self) -> None:
        w = AgentMessageWindow(max_messages=2)
        w.add(_user("a")).add(_asst("b"))
        self.assertEqual(w.count, 2)

    def test_unbounded_window_grows(self) -> None:
        w = AgentMessageWindow()
        for i in range(20):
            w.add(_user(str(i)))
        self.assertEqual(w.count, 20)


# ---------------------------------------------------------------------------
# Tool-use / tool-result pair preservation
# ---------------------------------------------------------------------------


class ToolPairTests(unittest.TestCase):
    def test_tool_pair_kept_together_when_fits(self) -> None:
        w = AgentMessageWindow(max_messages=4)
        w.add(_user("q"))
        w.add(_tool_use("t1"))
        w.add(_tool_result("t1"))
        w.add(_asst("done"))
        self.assertEqual(w.count, 4)

    def test_tool_pair_dropped_atomically(self) -> None:
        # Window of 3 with: plain user + tool_use + tool_result + asst.
        # The single user is oldest, so it drops first, leaving 3.
        w = AgentMessageWindow(max_messages=3)
        w.add(_user("first"))
        w.add(_tool_use("t1"))
        w.add(_tool_result("t1"))
        w.add(_asst("final"))
        self.assertEqual(w.count, 3)
        roles = [m["role"] for m in w.messages]
        self.assertEqual(roles, ["assistant", "user", "assistant"])

    def test_tool_pair_never_split(self) -> None:
        # Window of 2: adding the pair must drop both plain messages so the
        # tool_use/tool_result pair stays adjacent and intact.
        w = AgentMessageWindow(max_messages=2)
        w.add(_user("a"))
        w.add(_user("b"))
        w.add(_tool_use("t1"))
        w.add(_tool_result("t1"))
        self.assertEqual(w.count, 2)
        for m in w.messages:
            content = m.get("content", "")
            self.assertIsInstance(content, list)
            types = {b.get("type") for b in content if isinstance(b, dict)}
            self.assertTrue(types & {"tool_use", "tool_result"})

    def test_multiple_tool_pairs(self) -> None:
        w = AgentMessageWindow(max_messages=4)
        w.add(_tool_use("t1"))
        w.add(_tool_result("t1"))
        w.add(_tool_use("t2"))
        w.add(_tool_result("t2"))
        self.assertEqual(w.count, 4)
        # Adding one more message drops the oldest *pair* whole.
        w.add(_asst("done"))
        self.assertEqual(w.count, 3)
        # The surviving pair must still be the t2 pair, kept intact.
        roles = [m["role"] for m in w.messages]
        self.assertEqual(roles, ["assistant", "user", "assistant"])

    def test_multi_block_tool_use_in_single_message(self) -> None:
        # An assistant message may request several tools at once; the matching
        # user message returns several results. They still form one atomic pair.
        w = AgentMessageWindow(max_messages=2)
        w.add(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "s", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "s", "input": {}},
                ],
            }
        )
        w.add(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
                ],
            }
        )
        self.assertEqual(w.count, 2)

    def test_trailing_tool_use_without_result_is_allowed(self) -> None:
        # An assistant tool_use that has not yet received its result is a valid
        # intermediate state and must not be dropped just because it is unpaired.
        w = AgentMessageWindow(max_messages=2)
        w.add(_user("a"))
        w.add(_tool_use("t1"))
        self.assertEqual(w.count, 2)
        self.assertEqual(w.messages[-1]["content"][0]["type"], "tool_use")


# ---------------------------------------------------------------------------
# WindowOverflowError — oversized atomic group
# ---------------------------------------------------------------------------


class OverflowTests(unittest.TestCase):
    def test_oversized_pair_raises(self) -> None:
        # Regression: a tool_use/tool_result pair (size 2) cannot fit a window
        # of 1.  The library must raise rather than silently discard the pair.
        w = AgentMessageWindow(max_messages=1)
        w.add(_tool_use("t1"))
        with self.assertRaises(WindowOverflowError):
            w.add(_tool_result("t1"))

    def test_failed_add_leaves_window_unchanged(self) -> None:
        # Regression: a failed add must roll the window back to its prior state.
        w = AgentMessageWindow(max_messages=1)
        w.add(_tool_use("t1"))
        before = w.messages
        with self.assertRaises(WindowOverflowError):
            w.add(_tool_result("t1"))
        self.assertEqual(w.count, 1)
        self.assertEqual(w.messages, before)

    def test_replace_oversized_pair_raises(self) -> None:
        w = AgentMessageWindow(max_messages=1)
        with self.assertRaises(WindowOverflowError):
            w.replace([_tool_use("t1"), _tool_result("t1")])

    def test_failed_replace_leaves_window_unchanged(self) -> None:
        w = AgentMessageWindow(max_messages=1)
        w.add(_user("keep"))
        before = w.messages
        with self.assertRaises(WindowOverflowError):
            w.replace([_tool_use("t1"), _tool_result("t1")])
        self.assertEqual(w.messages, before)


# ---------------------------------------------------------------------------
# system message prepended
# ---------------------------------------------------------------------------


class SystemMessageTests(unittest.TestCase):
    def test_system_prepended(self) -> None:
        w = AgentMessageWindow(max_messages=5, system=_sys("System prompt"))
        w.add(_user("Hi"))
        msgs = w.messages
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "System prompt")
        self.assertEqual(msgs[1]["role"], "user")

    def test_system_not_counted_in_window(self) -> None:
        w = AgentMessageWindow(max_messages=2, system=_sys())
        w.add(_user("a")).add(_asst("b"))
        self.assertEqual(w.count, 2)
        # messages has system + 2 = 3 total.
        self.assertEqual(len(w.messages), 3)

    def test_system_deep_copied(self) -> None:
        sys_msg = _sys("original")
        w = AgentMessageWindow(system=sys_msg)
        sys_msg["content"] = "mutated"
        self.assertEqual(w.messages[0]["content"], "original")

    def test_system_survives_trim(self) -> None:
        w = AgentMessageWindow(max_messages=1, system=_sys("sys"))
        w.add(_user("a")).add(_user("b"))
        self.assertEqual(w.count, 1)
        self.assertEqual(w.messages[0]["role"], "system")
        self.assertEqual(w.messages[1]["content"], "b")


# ---------------------------------------------------------------------------
# last / first
# ---------------------------------------------------------------------------


class SliceAccessorTests(unittest.TestCase):
    def test_last_default(self) -> None:
        w = AgentMessageWindow()
        w.add_many([_user("a"), _asst("b"), _user("c")])
        result = w.last()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "c")

    def test_last_n(self) -> None:
        w = AgentMessageWindow()
        w.add_many([_user("a"), _asst("b"), _user("c")])
        result = w.last(2)
        self.assertEqual([m["content"] for m in result], ["b", "c"])

    def test_last_clamps(self) -> None:
        w = AgentMessageWindow()
        w.add(_user("only"))
        self.assertEqual(len(w.last(100)), 1)

    def test_first_n(self) -> None:
        w = AgentMessageWindow()
        w.add_many([_user("a"), _asst("b"), _user("c")])
        result = w.first(2)
        self.assertEqual([m["content"] for m in result], ["a", "b"])

    def test_first_zero(self) -> None:
        w = AgentMessageWindow()
        w.add(_user())
        self.assertEqual(w.first(0), [])

    def test_last_zero(self) -> None:
        w = AgentMessageWindow()
        w.add(_user())
        self.assertEqual(w.last(0), [])

    def test_last_is_deep_copy(self) -> None:
        w = AgentMessageWindow()
        w.add(_user("a"))
        snapshot = w.last(1)
        snapshot[0]["content"] = "mutated"
        self.assertEqual(w.messages[-1]["content"], "a")

    def test_negative_n_treated_as_zero(self) -> None:
        w = AgentMessageWindow()
        w.add(_user("a"))
        self.assertEqual(w.last(-5), [])
        self.assertEqual(w.first(-5), [])


if __name__ == "__main__":
    unittest.main()
