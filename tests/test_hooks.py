"""PreToolUse is the mandatory interception point. Prove it denies a repeat."""
import pytest

from app.effects import ledger, registry
from app.runtime import hooks

# Registration happens at module import; hoisted so it lands in the base dict
# and survives the per-test copy (a lesson from tests/test_effectful.py).
from app.tools import claude_code_tool  # noqa: F401,E402


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / "e.db"))
    ledger._reset_for_tests()
    yield
    ledger._reset_for_tests()


JOB = {"prompt": "build the thing", "working_dir": "/tmp/demo"}


def pre(tool_name, tool_input):
    return {"hook_event_name": "PreToolUse", "tool_name": tool_name,
            "tool_input": tool_input, "tool_use_id": "tu1"}


def test_the_mcp_prefix_is_stripped():
    assert hooks.bare_name("mcp__orchestrator__start_coding_job") == "start_coding_job"
    assert hooks.bare_name("Read") == "Read"
    assert hooks.bare_name("") == ""


async def test_a_tool_with_no_declared_effect_is_left_alone():
    out = await hooks.pre_tool_use(pre("mcp__orchestrator__read_file", {"path": "/x"}),
                                   "tu1", {})
    assert out == {}


async def test_a_first_start_is_allowed():
    out = await hooks.pre_tool_use(
        pre("mcp__orchestrator__start_coding_job", JOB),
        "tu1", {})
    assert out == {}


async def test_a_repeat_start_is_denied_with_the_first_result():
    name = "start_coding_job"
    kind = registry.registered()[name]
    mark = registry.fingerprint_for(name)((), JOB)
    with ledger.current_turn("t1"):
        row = ledger.record_intent(name, kind, mark)
        ledger.record_outcome(row, {"ok": True, "job_id": "job-1"}, ok=True)

    out = await hooks.pre_tool_use(
        pre("mcp__orchestrator__start_coding_job", JOB),
        "tu1", {})

    decision = out["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    reason = decision["permissionDecisionReason"]
    # The reason must carry the first result AND tell the model not to reword,
    # or it apologises and retries with a new body -- a new fingerprint, which
    # would go straight through.
    assert "job-1" in reason
    assert "reword" in reason.lower()


async def test_a_different_prompt_is_still_allowed():
    name = "start_coding_job"
    kind = registry.registered()[name]
    mark = registry.fingerprint_for(name)((), JOB)
    with ledger.current_turn("t1"):
        row = ledger.record_intent(name, kind, mark)
        ledger.record_outcome(row, {"ok": True}, ok=True)

    out = await hooks.pre_tool_use(
        pre("mcp__orchestrator__start_coding_job",
            dict(JOB, prompt="something else")), "tu1", {})
    assert out == {}


async def test_a_failed_start_is_not_denied_on_retry():
    """It had no effect, so retrying is right."""
    name = "start_coding_job"
    kind = registry.registered()[name]
    mark = registry.fingerprint_for(name)((), JOB)
    with ledger.current_turn("t1"):
        row = ledger.record_intent(name, kind, mark)
        ledger.record_outcome(row, {"ok": False, "error": "offline"}, ok=False)

    out = await hooks.pre_tool_use(
        pre("mcp__orchestrator__start_coding_job", JOB),
        "tu1", {})
    assert out == {}


async def test_post_tool_use_observes_without_writing_a_second_row():
    name = "start_coding_job"
    kind = registry.registered()[name]
    mark = registry.fingerprint_for(name)((), JOB)
    with ledger.current_turn("t1"):
        row = ledger.record_intent(name, kind, mark)
        ledger.record_outcome(row, {"ok": True}, ok=True)

    before = len(ledger.dangling()) + 1
    out = await hooks.post_tool_use(
        {"hook_event_name": "PostToolUse", "tool_name": "mcp__orchestrator__start_coding_job",
         "tool_input": JOB,
         "tool_response": {"ok": True}, "tool_use_id": "tu1"}, "tu1", {})
    assert out == {}
    # Two witnesses, one record.
    assert ledger.check(name, kind, mark) is not None
    assert len(ledger.dangling()) + 1 == before


def test_looks_ok_unwraps_an_mcp_envelope():
    assert hooks._looks_ok({"ok": True}) is True
    assert hooks._looks_ok({"ok": False}) is False
    assert hooks._looks_ok('{"ok": false}') is False
    assert hooks._looks_ok({"content": [{"type": "text", "text": '{"ok": false}'}]}) is False
    assert hooks._looks_ok(None) is False


def test_matchers_are_wired_for_both_events():
    m = hooks.matchers()
    assert set(m) == {"PreToolUse", "PostToolUse"}
    assert m["PreToolUse"][0].hooks == [hooks.pre_tool_use]
