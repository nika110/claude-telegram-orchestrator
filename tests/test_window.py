"""Windowing. P1 proved this slice IS what the model reads back."""
from app.memory import window


def user(text, ts="2026-08-31T10:00:00Z", uuid="u"):
    return {"type": "user", "uuid": uuid, "timestamp": ts,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def assistant(text, ts="2026-08-31T10:00:01Z", uuid="a"):
    return {"type": "assistant", "uuid": uuid, "timestamp": ts,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def call(name="send", cid="c1", ts="2026-08-31T10:00:02Z"):
    return {"type": "assistant", "uuid": "ac", "timestamp": ts,
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": cid, "name": name, "input": {}}]}}


def result(cid="c1", ts="2026-08-31T10:00:03Z"):
    return {"type": "user", "uuid": "ur", "timestamp": ts,
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": cid, "content": "ok"}]}}


def test_a_short_conversation_is_returned_whole():
    entries = [user("hi"), assistant("hello")]
    assert window.build(entries) == entries


def test_empty_in_empty_out():
    assert window.build([]) == []


def test_the_window_is_a_contiguous_suffix():
    entries = []
    for i in range(50):
        entries += [user("q%d" % i, uuid="u%d" % i), assistant("a%d" % i, uuid="a%d" % i)]
    out = window.build(entries, max_turns=5)
    assert out == entries[-len(out):], "must be a suffix, not a filtered subset"
    assert len(out) < len(entries)


def test_turn_cap_is_respected():
    entries = []
    for i in range(40):
        entries += [user("q%d" % i, uuid="u%d" % i), assistant("a%d" % i, uuid="a%d" % i)]
    out = window.build(entries, max_turns=5)
    starts = [e for e in out if window._starts_a_turn(e)]
    assert len(starts) <= 5


def test_char_budget_is_respected():
    big = "x" * 5000
    entries = []
    for i in range(20):
        entries += [user(big, uuid="u%d" % i), assistant(big, uuid="a%d" % i)]
    out = window.build(entries, max_chars=30_000)
    total = sum(len(str(e)) for e in out)
    assert total < 60_000


def test_a_long_silence_cuts_the_window():
    """A thread nobody has touched for a week is not context for today."""
    entries = [
        user("last month", ts="2026-08-01T22:00:00Z", uuid="u1"),
        assistant("ok", ts="2026-08-01T22:00:01Z", uuid="a1"),
        user("today", ts="2026-08-31T10:00:00Z", uuid="u2"),
        assistant("hi", ts="2026-08-31T10:00:01Z", uuid="a2"),
    ]
    out = window.build(entries)
    assert [e["uuid"] for e in out] == ["u2", "a2"]


def test_sleeping_does_not_end_a_conversation():
    """The reason MAX_GAP_SECONDS is three days and not six hours.

    At six hours every overnight gap read as a new conversation, and the
    owner's own thread -- 1,226 migrated entries -- came back as two.
    """
    entries = [
        user("last night", ts="2026-08-30T22:00:00Z", uuid="u1"),
        assistant("ok", ts="2026-08-30T22:00:01Z", uuid="a1"),
        user("this morning", ts="2026-08-31T10:00:00Z", uuid="u2"),
        assistant("hi", ts="2026-08-31T10:00:01Z", uuid="a2"),
    ]
    assert [e["uuid"] for e in window.build(entries)] == ["u1", "a1", "u2", "a2"]


def test_a_short_gap_does_not_cut():
    entries = [
        user("a", ts="2026-08-31T10:00:00Z", uuid="u1"),
        assistant("b", ts="2026-08-31T10:00:01Z", uuid="a1"),
        user("c", ts="2026-08-31T11:00:00Z", uuid="u2"),
    ]
    assert len(window.build(entries)) == 3


def test_a_tool_result_is_not_a_turn_boundary():
    """It is the back half of the assistant's turn, not a new message."""
    assert window._starts_a_turn(user("real")) is True
    assert window._starts_a_turn(result()) is False
    assert window._starts_a_turn(assistant("x")) is False


def test_the_window_never_opens_on_an_orphaned_tool_result():
    """Anthropic requires every tool_result to answer a visible tool_use."""
    entries = []
    for i in range(30):
        entries += [user("q%d" % i, uuid="u%d" % i), call(cid="c%d" % i),
                    result(cid="c%d" % i), assistant("a%d" % i, uuid="a%d" % i)]
    out = window.build(entries, max_turns=3)
    assert window.validate(out) == [], window.validate(out)
    assert window._starts_a_turn(out[0]) or out[0]["type"] != "user"


def test_validate_catches_a_dangling_tool_result():
    bad = [result(cid="missing"), assistant("done")]
    problems = window.validate(bad)
    assert problems and "not in the window" in problems[0]


def test_validate_passes_a_matched_pair():
    good = [user("go"), call(cid="c9"), result(cid="c9"), assistant("done")]
    assert window.validate(good) == []


def test_a_realistic_long_history_windows_to_something_legal():
    entries = []
    for i in range(200):
        entries += [user("message %d" % i, uuid="u%d" % i),
                    call(cid="c%d" % i), result(cid="c%d" % i),
                    assistant("reply %d" % i, uuid="a%d" % i)]
    out = window.build(entries)
    assert out == entries[-len(out):]
    assert window.validate(out) == []
    assert len(out) < len(entries)
