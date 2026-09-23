"""The conversation store. P1 proved this IS the conversation, not a mirror."""
import json

import pytest

from app.memory import store


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH", str(tmp_path / "c.db"))
    store._reset_for_tests()
    yield
    store._reset_for_tests()


def user_entry(text, uuid="u1", ts="2026-08-31T10:00:00Z"):
    return {"type": "user", "uuid": uuid, "timestamp": ts,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def assistant_entry(text, uuid="a1", ts="2026-08-31T10:00:01Z"):
    return {"type": "assistant", "uuid": uuid, "timestamp": ts,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def tool_entries(name="start_coding_job", uuid="t1", ts="2026-08-31T10:00:02Z"):
    return [
        {"type": "assistant", "uuid": uuid, "timestamp": ts,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "c1", "name": name, "input": {"text": "hi"}}]}},
        {"type": "user", "uuid": uuid + "r", "timestamp": ts,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "c1", "content": '{"ok": true}'}]}},
    ]


# ------------------------------------------------------------- relationships

def test_each_relationship_gets_its_own_session_id():
    a = store.session_id_for("telegram", "100000001")
    b = store.session_id_for("other", "someone")
    assert a != b
    assert len(a) == 36
    # Stable across calls: one real session id per conversation.
    assert store.session_id_for("telegram", "100000001") == a


def test_a_session_id_maps_back_to_its_relationship():
    sid = store.session_id_for("other", "group:x")
    assert store.relationship_of(sid) == ("other", "group:x")
    assert store.relationship_of("no-such-session") is None


# ------------------------------------------------------------------ entries

def test_entries_round_trip_verbatim():
    """Prompt fuel must come back byte-identical: a rewritten chain is not a chain."""
    entries = [user_entry("hello"), assistant_entry("hi there")]
    store.append_entries("telegram", "1", entries)
    assert store.load_entries("telegram", "1") == entries


def test_entries_are_isolated_per_relationship():
    store.append_entries("telegram", "1", [user_entry("mine")])
    store.append_entries("telegram", "2", [user_entry("yours")])
    assert len(store.load_entries("telegram", "1")) == 1
    assert store.load_entries("telegram", "2")[0]["message"]["content"][0]["text"] == "yours"


def test_append_returns_ids_so_the_writer_never_reads_back():
    ids = store.append_entries("telegram", "1", [user_entry("a"), assistant_entry("b")])
    assert len(ids) == 2 and ids[0] < ids[1]


def test_bookkeeping_entry_types_are_stored_but_not_projected():
    """P4 recorded these: mode, queue-operation, ai-title... they are not conversation."""
    noise = [{"type": t, "uuid": t, "timestamp": "2026-08-31T10:00:00Z"}
             for t in ("mode", "queue-operation", "ai-title", "last-prompt", "atis-latch")]
    store.append_entries("telegram", "1", noise + [user_entry("real")])
    assert len(store.load_entries("telegram", "1")) == 6      # all kept as fuel
    assert len(store.history("telegram", "1")) == 1           # only the real one shows


# ------------------------------------------------------------------ reading

def test_history_matches_the_shape_history_tools_returns():
    store.append_entries("telegram", "1", [user_entry("q"), assistant_entry("a")])
    out = store.history("telegram", "1")
    assert [m["from"] for m in out] == ["user", "assistant"]
    assert out[0]["text"] == "q"
    assert set(out[0]) == {"at", "from", "text"}


def test_history_is_newest_last_and_limited():
    for i in range(10):
        store.append_entries("telegram", "1", [user_entry("m%d" % i, uuid="u%d" % i)])
    out = store.history("telegram", "1", limit=3)
    assert [m["text"] for m in out] == ["m7", "m8", "m9"]


def test_tool_calls_are_hidden_unless_asked_for():
    store.append_entries("telegram", "1", [user_entry("go")] + tool_entries())
    assert len(store.history("telegram", "1")) == 1
    withtools = store.history("telegram", "1", include_tools=True)
    kinds = [set(m) - {"at", "from"} for m in withtools]
    assert {"called", "args"} in kinds
    assert {"result_of", "result"} in kinds


def test_conversations_lists_each_relationship_newest_first():
    store.append_entries("other", "old", [user_entry("x", ts="2026-08-01T00:00:00Z")])
    store.append_entries("telegram", "new", [user_entry("y", ts="2026-08-31T00:00:00Z")])
    convos = store.conversations()
    assert [c["user_id"] for c in convos] == ["new", "old"]
    assert convos[0]["platform"] == "telegram"
    assert convos[0]["messages"] == 1


# ----------------------------------------------------------------- subpaths

def test_a_nested_run_is_attributed_to_its_agent():
    """P3: nested runs arrive under subagents/<id>. read_my_chat_history must
    still show all seven authors after a handoff."""
    store.append_entries("telegram", "1", [user_entry("build me a thing")])
    store.append_entries("telegram", "1", [assistant_entry("on it", uuid="n1")],
                         subpath="subagents/agent-abc")
    authors = {m["from"] for m in store.history("telegram", "1")}
    assert "user" in authors
    assert "assistant:agent-abc" in authors
    assert store.subpaths("telegram", "1") == ["subagents/agent-abc"]


def test_nested_entries_do_not_leak_into_the_parent_chain():
    store.append_entries("telegram", "1", [user_entry("a")])
    store.append_entries("telegram", "1", [assistant_entry("b")], subpath="subagents/x")
    assert len(store.load_entries("telegram", "1")) == 1
    assert len(store.load_entries("telegram", "1", subpath="subagents/x")) == 1


# ----------------------------------------------------------------- deleting

def test_forget_turn_since_removes_a_suffix_not_a_hole():
    """Deleting a suffix keeps the remaining chain unbroken, so a cached
    session survives the prune. Removing from the middle would not."""
    store.append_entries("telegram", "1", [user_entry("first", uuid="u1"),
                                           assistant_entry("reply one", uuid="a1")])
    store.append_entries("telegram", "1", [user_entry("second", uuid="u2"),
                                           assistant_entry("reply two", uuid="a2")])
    removed = store.forget_turn_since("telegram", "1")
    assert removed == 2
    left = store.load_entries("telegram", "1")
    assert [e["uuid"] for e in left] == ["u1", "a1"]
    assert [m["text"] for m in store.history("telegram", "1")] == ["first", "reply one"]


def test_forget_turn_since_on_an_empty_conversation_is_a_no_op():
    assert store.forget_turn_since("telegram", "nobody") == 0


def test_reset_really_deletes():
    """/clear tells the owner it wipes everything, so it must."""
    store.session_id_for("telegram", "1")
    store.append_entries("telegram", "1", [user_entry("a")])
    store.append_entries("telegram", "1", [assistant_entry("b")], subpath="subagents/x")
    out = store.reset("telegram", "1")
    assert out["entries"] == 2
    assert store.load_entries("telegram", "1") == []
    assert store.load_entries("telegram", "1", subpath="subagents/x") == []
    assert store.history("telegram", "1") == []
    # A fresh session id is minted next time, not the old one.
    assert store.relationship_of("whatever") is None


def test_reset_does_not_touch_other_conversations():
    store.append_entries("telegram", "1", [user_entry("mine")])
    store.append_entries("telegram", "2", [user_entry("theirs")])
    store.reset("telegram", "1")
    assert len(store.load_entries("telegram", "2")) == 1
