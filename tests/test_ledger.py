"""The effect ledger. Everything about not sending the same DM twice."""
import pytest

from app.effects import ledger


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / "effects.db"))
    ledger._reset_for_tests()
    yield
    ledger._reset_for_tests()


def test_an_unseen_effect_is_not_a_repeat():
    assert ledger.check("start_coding_job", "spend", "claude|/tmp/a|build") is None


def test_a_completed_effect_is_returned_instead_of_repeating():
    with ledger.current_turn("t1"):
        row = ledger.record_intent("start_coding_job", "spend", "claude|/tmp/a|build")
        ledger.record_outcome(row, {"ok": True, "job_id": "job-1"}, ok=True)

    seen = ledger.check("start_coding_job", "spend", "claude|/tmp/a|build")
    assert seen is not None
    assert seen["result"] == {"ok": True, "job_id": "job-1"}


def test_a_different_fingerprint_is_not_a_repeat():
    with ledger.current_turn("t1"):
        row = ledger.record_intent("start_coding_job", "spend", "claude|/tmp/a|build")
        ledger.record_outcome(row, {"ok": True}, ok=True)
    assert ledger.check("start_coding_job", "spend", "claude|/tmp/a|test") is None


def test_a_failed_effect_may_be_retried():
    # A send that failed did not reach anyone, so blocking a retry would turn a
    # transient error into a lost message.
    with ledger.current_turn("t1"):
        row = ledger.record_intent("run_claude_code", "spend", "claude|/tmp/b|hi")
        ledger.record_outcome(row, {"ok": False, "error": "offline"}, ok=False)
    assert ledger.check("run_claude_code", "spend", "claude|/tmp/b|hi") is None


def test_an_effect_outside_the_window_is_not_a_repeat(monkeypatch):
    with ledger.current_turn("t1"):
        row = ledger.record_intent("start_coding_job", "spend", "claude|/tmp/a|build")
        ledger.record_outcome(row, {"ok": True}, ok=True)
    monkeypatch.setattr(ledger, "WINDOW_SECONDS", 0.0)
    assert ledger.check("start_coding_job", "spend", "claude|/tmp/a|build") is None


def test_intent_without_outcome_is_dangling_and_never_replayable():
    # The process died between the wire call and the result. It MAY have
    # shipped, so it must never be silently re-run -- but it is also not a
    # completed effect whose result can be handed back.
    with ledger.current_turn("t1"):
        ledger.record_intent("deploy_app", "deploy", "deploy|myapp")

    assert ledger.check("deploy_app", "deploy", "deploy|myapp") is None
    hanging = ledger.dangling()
    assert len(hanging) == 1
    assert hanging[0]["tool"] == "deploy_app"
    assert hanging[0]["kind"] == "deploy"


def test_dangling_can_be_filtered_by_kind():
    with ledger.current_turn("t1"):
        ledger.record_intent("deploy_app", "deploy", "d1")
        ledger.record_intent("start_coding_job", "spend", "s1")
    assert len(ledger.dangling(kind="deploy")) == 1
    assert len(ledger.dangling(kind="spend")) == 1
    assert len(ledger.dangling()) == 2


def test_the_ledger_survives_a_reopen():
    with ledger.current_turn("t1"):
        row = ledger.record_intent("start_coding_job", "spend", "ig|a|b")
        ledger.record_outcome(row, {"ok": True}, ok=True)
    ledger._reset_for_tests()          # drops the connection, keeps the file
    assert ledger.check("start_coding_job", "spend", "ig|a|b") is not None


def test_turn_id_is_recorded_and_defaults_when_unset():
    with ledger.current_turn("turn-abc"):
        row = ledger.record_intent("start_coding_job", "spend", "f1")
        ledger.record_outcome(row, {"ok": True}, ok=True)
    assert ledger.check("start_coding_job", "spend", "f1")["turn_id"] == "turn-abc"

    # Outside a turn the ledger still works rather than raising -- a tool called
    # from a script or a test must not blow up.
    row = ledger.record_intent("start_coding_job", "spend", "f2")
    ledger.record_outcome(row, {"ok": True}, ok=True)
    assert ledger.check("start_coding_job", "spend", "f2")["turn_id"] == "-"
