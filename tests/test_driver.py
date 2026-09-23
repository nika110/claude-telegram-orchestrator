"""The driver. The interesting half is what it does with an interrupted turn."""
import json
import subprocess
import time

import pytest

from app.effects import ledger, registry
from app.memory import store, window
from app.runtime import driver

# Registration happens at module import; hoisted so it lands in the base dict.
from app.tools import claude_code_tool  # noqa: F401


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH", str(tmp_path / "c.db"))
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / "e.db"))
    store._reset_for_tests()
    ledger._reset_for_tests()
    yield
    store._reset_for_tests()
    ledger._reset_for_tests()


def text_entry(role, body, uuid):
    return {"type": role, "uuid": uuid, "timestamp": "2026-09-01T10:00:00Z",
            "message": {"role": role, "content": [{"type": "text", "text": body}]}}


def call_entry(uuid, tool_use_id, name, tool_input):
    return {"type": "assistant", "uuid": uuid, "timestamp": "2026-09-01T10:00:01Z",
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}]}}


def result_entry(uuid, tool_use_id, body):
    return {"type": "user", "uuid": uuid, "timestamp": "2026-09-01T10:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": body}]}}


JOB = {"prompt": "build the thing", "working_dir": "/tmp/demo"}


def a_killed_start(platform="telegram", user_id="1"):
    """A chain that stops between a job starting and its result landing."""
    store.session_id_for(platform, user_id)
    store.append_entries(platform, user_id, [
        text_entry("user", "build the thing", "u1"),
        call_entry("a1", "toolu_01", "mcp__orchestrator__start_coding_job", JOB),
    ])
    return platform, user_id


def ledger_row(state, result=None):
    name, kind = "start_coding_job", registry.registered()["start_coding_job"]
    mark = registry.fingerprint_for(name)((), JOB)
    with ledger.current_turn("t1"):
        row = ledger.record_intent(name, kind, mark)
        if state != "intent":
            ledger.record_outcome(row, result or {"ok": True, "job_id": "job-1"},
                                  ok=(state == "done"))
    return mark


# ------------------------------------------------------- finding the loose end

def test_a_clean_chain_has_nothing_to_close():
    platform, user_id = "telegram", "1"
    store.session_id_for(platform, user_id)
    store.append_entries(platform, user_id, [
        text_entry("user", "hi", "u1"), text_entry("assistant", "hello", "a1")])
    assert driver.unanswered_tool_uses(store.load_entries(platform, user_id)) == []
    assert driver.resolve_dangling(platform, user_id) == 0


def test_an_answered_call_is_not_reopened():
    platform, user_id = a_killed_start()
    store.append_entries(platform, user_id, [result_entry("u2", "toolu_01", '{"ok": true}')])
    assert driver.unanswered_tool_uses(store.load_entries(platform, user_id)) == []


def test_the_unanswered_call_is_found():
    platform, user_id = a_killed_start()
    pending = driver.unanswered_tool_uses(store.load_entries(platform, user_id))
    assert len(pending) == 1
    tool_use_id, name, tool_input = pending[0]
    assert tool_use_id == "toolu_01"
    assert name.endswith("start_coding_job")
    assert tool_input == JOB


# -------------------------------------------- closing it WITHOUT re-running it

def test_a_completed_start_is_closed_from_the_ledger_not_re_run(monkeypatch):
    """The acceptance test for the whole design.

    The process died after the job started. On resume the tool must NOT run
    again -- the result comes out of the ledger.
    """
    platform, user_id = a_killed_start()
    ledger_row("done", {"ok": True, "job_id": "job-1", "status": "running"})

    def explode(*args, **kwargs):
        raise AssertionError("the tool was re-run; the job would have started twice")

    monkeypatch.setattr(claude_code_tool, "start_coding_job", explode)

    assert driver.resolve_dangling(platform, user_id) == 1

    entries = store.load_entries(platform, user_id)
    block = entries[-1]["message"]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_01"
    assert block["is_error"] is False
    payload = json.loads(block["content"])
    assert payload["recovered"] is True
    assert payload["result"]["job_id"] == "job-1"


def test_a_committed_but_unconfirmed_start_says_so_and_forbids_a_retry():
    """Intent with no outcome. It may have started, so it must not be replayed
    and must not be reported as failed either."""
    platform, user_id = a_killed_start()
    ledger_row("intent")

    assert driver.resolve_dangling(platform, user_id) == 1
    block = store.load_entries(platform, user_id)[-1]["message"]["content"][0]
    payload = json.loads(block["content"])
    assert payload["ok"] is None
    assert payload["interrupted"] is True
    assert "not run it again" in payload["note"].lower()
    # is_error would invite the model to retry.
    assert block["is_error"] is False


def test_a_call_with_no_ledger_row_at_all_is_safe_to_retry():
    platform, user_id = a_killed_start()
    assert driver.resolve_dangling(platform, user_id) == 1
    block = store.load_entries(platform, user_id)[-1]["message"]["content"][0]
    payload = json.loads(block["content"])
    assert payload["ok"] is False
    assert block["is_error"] is True


def test_a_repaired_chain_is_legal_to_send():
    """The point of repairing at all: Anthropic rejects an unanswered tool_use."""
    platform, user_id = a_killed_start()
    ledger_row("done")
    assert window.validate(store.load_entries(platform, user_id)), "should start illegal"
    driver.resolve_dangling(platform, user_id)
    assert window.validate(store.load_entries(platform, user_id)) == []


def test_the_repair_links_into_the_chain():
    """An unlinked record is unreachable when the SDK reconstructs the
    conversation, so a repair without parentUuid mends the chain on paper and
    leaves the resumed turn seeing nothing before it. Measured: 73 unlinked
    records, 1 reached the model."""
    platform, user_id = a_killed_start()
    ledger_row("done")
    driver.resolve_dangling(platform, user_id)

    entries = store.load_entries(platform, user_id)
    assert entries[-1]["parentUuid"] == entries[-2]["uuid"]


def test_repair_is_idempotent():
    platform, user_id = a_killed_start()
    ledger_row("done")
    assert driver.resolve_dangling(platform, user_id) == 1
    assert driver.resolve_dangling(platform, user_id) == 0


def test_several_parallel_calls_are_all_closed():
    platform, user_id = "telegram", "1"
    store.session_id_for(platform, user_id)
    store.append_entries(platform, user_id, [
        text_entry("user", "do both", "u1"),
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-09-01T10:00:01Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "t1", "name": "start_coding_job", "input": JOB},
             {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "/x"}}]}},
    ])
    assert driver.resolve_dangling(platform, user_id) == 2
    ids = {b["tool_use_id"] for b in store.load_entries(platform, user_id)[-1]["message"]["content"]}
    assert ids == {"t1", "t2"}


# ------------------------------------------------------------------- the cursor

def test_the_cursor_prefers_the_last_real_text():
    cursor = driver.TurnCursor()
    cursor.add_text("thinking out loud")
    cursor.add_text("   ")
    cursor.add_text("Started it.")
    assert cursor.reply() == "Started it."


def test_the_cursor_describes_a_silent_turn_from_its_tools():
    """An empty answer after real work is what made the owner ask twice."""
    cursor = driver.TurnCursor()
    cursor.add_tool_use("t1", "mcp__orchestrator__start_coding_job", JOB)
    cursor.add_tool_result("t1", json.dumps({"ok": True, "job_id": "job-1", "status": "running"}), False)
    assert cursor.reply() == "Coding job job-1 is running."


def test_the_cursor_reports_a_failure_rather_than_claiming_success():
    cursor = driver.TurnCursor()
    cursor.add_tool_use("t1", "mcp__orchestrator__start_coding_job", JOB)
    cursor.add_tool_result("t1", json.dumps({"ok": False, "error": "all slots busy"}), False)
    assert "all slots busy" in cursor.reply()


def test_the_cursor_unwraps_an_mcp_envelope():
    cursor = driver.TurnCursor()
    cursor.add_tool_use("t1", "mcp__orchestrator__start_coding_job", JOB)
    cursor.add_tool_result(
        "t1", [{"type": "text", "text": json.dumps({"ok": True, "job_id": "job-2"})}],
        False)
    assert cursor.reply() == "Coding job job-2 is running."


def test_a_truly_empty_turn_says_so():
    assert driver.TurnCursor().reply() == driver.NO_RESPONSE


# ------------------------------------------------------------------- the ladder

def test_only_a_history_problem_triggers_the_ladder():
    assert driver._is_history_problem(ValueError("messages: unexpected role"))
    assert driver._is_history_problem(RuntimeError("no such session"))
    assert not driver._is_history_problem(OSError("connection reset by peer"))
    assert not driver._is_history_problem(RuntimeError("Not logged in"))


# ------------------------------------------------------------------ auth wiring

def test_ensure_config_pins_a_dir_that_is_logged_in(tmp_path, monkeypatch):
    import os
    source = tmp_path / "src.json"
    source.write_text('{"token": "x"}')
    # ensure_config writes os.environ directly, so monkeypatch has to be told
    # about the key first or the tmp config dir leaks into every later test in
    # the session and they all fail with "Not logged in".
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "placeholder"))
    monkeypatch.setattr(driver, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(driver, "CREDENTIALS_SOURCE", str(source))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-removed")

    got = driver.ensure_config()
    assert os.environ["CLAUDE_CONFIG_DIR"] == got
    assert os.path.exists(os.path.join(got, ".credentials.json"))
    assert "ANTHROPIC_API_KEY" not in os.environ


def creds(hours_from_now, marker="t"):
    """A credentials file whose access token dies `hours_from_now`."""
    return json.dumps({"claudeAiOauth": {
        "accessToken": marker,
        "refreshToken": "r-%s" % marker,
        "expiresAt": int((time.time() + hours_from_now * 3600) * 1000),
    }})


def pin(tmp_path, monkeypatch):
    """Point the driver at a throwaway config dir and credentials source."""
    source = tmp_path / "src.json"
    # ensure_config writes os.environ directly, so monkeypatch has to be told
    # about the key first or the tmp config dir leaks into every later test in
    # the session and they all fail with "Not logged in".
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "placeholder"))
    monkeypatch.setattr(driver, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(driver, "CREDENTIALS_SOURCE", str(source))
    return source, tmp_path / "cfg" / ".credentials.json"


def test_refreshed_credentials_are_re_copied(tmp_path, monkeypatch):
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(1, "old"))
    driver.ensure_config()
    assert "old" in target.read_text()

    source.write_text(creds(8, "new"))
    driver.ensure_config()
    assert "new" in target.read_text(), "a stale token expires overnight"


def test_a_longer_lived_pinned_token_survives_a_newer_source(tmp_path, monkeypatch):
    """The 8 Sep outage: mtime said copy, the tokens said the opposite.

    shutil.copy2 preserves mtime, so once the pinned dir refreshes on its own
    the source is routinely both newer-looking and worse -- and copying it
    back installs a refresh token the CLI has already rotated past.
    """
    import os
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(1, "old"))
    driver.ensure_config()

    target.write_text(creds(8, "self-refreshed"))
    os.utime(source, (time.time() + 60, time.time() + 60))
    driver.ensure_config()
    assert "self-refreshed" in target.read_text()


def test_an_unreadable_pinned_file_is_replaced(tmp_path, monkeypatch):
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(8, "good"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{ truncated")
    driver.ensure_config()
    assert "good" in target.read_text()


def test_a_live_token_is_not_renewed(tmp_path, monkeypatch):
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(8))
    driver.ensure_config()
    monkeypatch.setattr(driver.shutil, "which",
                        lambda _: pytest.fail("renewed a token with hours left"))
    assert driver.refresh_credentials() is True


def test_a_dying_token_is_renewed_before_the_owner_needs_it(tmp_path, monkeypatch):
    source, target = pin(tmp_path, monkeypatch)
    # Inside the margin but not yet expired: nothing is broken yet, which is
    # the whole point of renewing here rather than mid-turn.
    source.write_text(creds(0.1))
    driver.ensure_config()

    seen = []

    def fake_run(cmd, **kwargs):
        # The CLI refreshes on the stored expiry alone, so what it sees here
        # is the whole mechanism: a token it considers due.
        seen.append(driver.token_expiry(str(target)))
        target.write_text(creds(8, "renewed"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(driver, "_renewed_at", 0.0)
    monkeypatch.setattr(driver.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    assert driver.refresh_credentials() is True
    assert "renewed" in target.read_text()
    assert seen and seen[0] < time.time(), "the CLI was asked to renew a live token"


def test_a_renewal_that_mints_nothing_leaves_the_old_token_working(tmp_path, monkeypatch):
    """Exit 0 is not the signal -- a token with a future expiry is.

    And a failed renewal must not leave the backdated expiry behind: that
    token still has hours on it, and the next turn should spend them.
    """
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(0.1, "still-good"))
    driver.ensure_config()
    monkeypatch.setattr(driver, "_renewed_at", 0.0)
    monkeypatch.setattr(driver.shutil, "which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    assert driver.refresh_credentials() is False
    assert "still-good" in target.read_text()
    assert driver.token_expiry(str(target)) > time.time()


def test_a_renewal_answers_for_the_turns_queued_behind_it(tmp_path, monkeypatch):
    """Two chats can want a renewal at once; only one may mint.

    A refresh rotates the refresh token, so a second renewal would rotate the
    first one's token away -- and the loser restores a file that is already
    dead over the winner's live one.
    """
    source, target = pin(tmp_path, monkeypatch)
    source.write_text(creds(-1))
    driver.ensure_config()
    monkeypatch.setattr(driver, "_renewed_at", time.time())
    monkeypatch.setattr(driver.shutil, "which",
                        lambda _: pytest.fail("renewed on top of a fresh renewal"))
    assert driver.refresh_credentials(force=True) is True


def test_the_401_the_cli_actually_returns_is_read_as_auth(tmp_path):
    assert driver._is_auth_problem(RuntimeError(
        "Claude Code returned an error result: Failed to authenticate. API "
        "Error: 401 OAuth access token has expired. Re-authenticate to continue."))
    assert not driver._is_auth_problem(RuntimeError("start_coding_job timed out"))
    # An auth failure must not be mistaken for a corrupt conversation: that
    # would spend the owner's history on a problem history did not cause.
    assert not driver._is_history_problem(RuntimeError("401 OAuth access token"))


async def test_an_expired_token_costs_a_retry_not_the_turn(monkeypatch):
    attempts = []

    async def flaky(prompt, options, cursor):
        attempts.append(prompt)
        if len(attempts) == 1:
            cursor.add_text("Failed to authenticate.")
            raise RuntimeError("401 OAuth access token has expired")
        cursor.add_text("Here is the build log.")

    monkeypatch.setattr(driver, "_stream", flaky)
    monkeypatch.setattr(driver, "refresh_credentials", lambda force=False: True)
    cursor = driver.TurnCursor()
    await driver._stream_with_reauth("show me the build log", None, cursor)

    assert len(attempts) == 2
    assert cursor.reply() == "Here is the build log."
    assert "Failed to authenticate." not in cursor.texts


async def test_an_unrenewable_token_still_surfaces(monkeypatch):
    async def always_401(prompt, options, cursor):
        raise RuntimeError("401 OAuth access token has expired")

    monkeypatch.setattr(driver, "_stream", always_401)
    monkeypatch.setattr(driver, "refresh_credentials", lambda force=False: False)
    with pytest.raises(RuntimeError):
        await driver._stream_with_reauth("hi", None, driver.TurnCursor())


# ------------------------------------------------------------- the tool surface

def test_every_tool_is_allowed_by_name():
    names = driver.allowed_tool_names()
    assert all(n.startswith("mcp__orchestrator__") for n in names)
    assert {n.split("__")[-1] for n in names} == {
        "start_coding_job", "check_coding_job", "list_coding_jobs",
        "cancel_coding_job", "run_claude_code", "run_shell_command",
        "read_file", "write_file", "replace_in_file", "list_directory",
    }


def test_the_options_pass_their_own_isolation_check(tmp_path, monkeypatch):
    from app.runtime import toolserver
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "placeholder"))
    monkeypatch.setattr(driver, "CONFIG_DIR", str(tmp_path / "cfg"))
    source = tmp_path / "src.json"
    source.write_text("{}")
    monkeypatch.setattr(driver, "CREDENTIALS_SOURCE", str(source))
    driver.ensure_config()
    options = driver.build_options("11111111-2222-3333-4444-555555555555")
    toolserver.assert_isolation(options)
