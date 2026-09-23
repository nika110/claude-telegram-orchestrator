"""The SessionStore adapter, including a live end-to-end resume through it."""
import pytest

from app.memory import store
from app.memory.sdk_store import ConversationStore


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH", str(tmp_path / "c.db"))
    store._reset_for_tests()
    yield
    store._reset_for_tests()


def entry(kind, text, uuid):
    return {"type": kind, "uuid": uuid, "timestamp": "2026-08-31T10:00:00Z",
            "message": {"role": kind, "content": [{"type": "text", "text": text}]}}


def key_for(sid, subpath=""):
    # SessionKey is a TypedDict -- a plain dict. Subscript it, never getattr.
    return {"project_key": "-home-ubuntu-agents-orchestrator",
            "session_id": sid, "subpath": subpath}


async def test_append_and_load_round_trip():
    sid = store.session_id_for("telegram", "100000001")
    adapter = ConversationStore(windowed=False)
    entries = [entry("user", "hello", "u1"), entry("assistant", "hi", "a1")]
    await adapter.append(key_for(sid), entries)
    assert await adapter.load(key_for(sid)) == entries


async def test_load_returns_none_for_an_empty_conversation():
    sid = store.session_id_for("telegram", "empty")
    assert await ConversationStore().load(key_for(sid)) is None


async def test_the_key_maps_to_the_right_relationship():
    a = store.session_id_for("telegram", "1")
    b = store.session_id_for("other", "someone")
    adapter = ConversationStore(windowed=False)
    await adapter.append(key_for(a), [entry("user", "mine", "u1")])
    await adapter.append(key_for(b), [entry("user", "theirs", "u2")])
    mine = await adapter.load(key_for(a))
    assert len(mine) == 1
    assert mine[0]["message"]["content"][0]["text"] == "mine"


async def test_an_unclaimed_session_is_filed_not_dropped():
    """A session nobody registered must not lose its entries silently."""
    adapter = ConversationStore(windowed=False)
    await adapter.append(key_for("11111111-2222-3333-4444-555555555555"),
                         [entry("user", "orphaned", "u1")])
    assert len(store.load_entries("__orphan__", "11111111-2222-3333-4444-555555555555")) == 1


async def test_subpath_entries_are_kept_apart():
    sid = store.session_id_for("telegram", "1")
    adapter = ConversationStore(windowed=False)
    await adapter.append(key_for(sid), [entry("user", "parent", "u1")])
    await adapter.append(key_for(sid, "subagents/agent-x"),
                         [entry("assistant", "child", "a1")])
    assert len(await adapter.load(key_for(sid))) == 1
    assert len(await adapter.load(key_for(sid, "subagents/agent-x"))) == 1
    assert await adapter.list_subkeys({"session_id": sid}) == ["subagents/agent-x"]


async def test_load_windows_a_long_conversation():
    sid = store.session_id_for("telegram", "1")
    adapter = ConversationStore(windowed=True)
    entries = []
    for i in range(120):
        entries += [entry("user", "q%d" % i, "u%d" % i),
                    entry("assistant", "a%d" % i, "a%d" % i)]
    await adapter.append(key_for(sid), entries)
    got = await adapter.load(key_for(sid))
    assert len(got) < len(entries)
    assert got == entries[-len(got):], "the window must be a suffix"


async def test_delete_really_removes():
    sid = store.session_id_for("telegram", "1")
    adapter = ConversationStore()
    await adapter.append(key_for(sid), [entry("user", "a", "u1")])
    await adapter.delete(key_for(sid))
    assert store.load_entries("telegram", "1") == []


async def test_a_real_resume_reads_back_through_this_store():
    """End to end: the SDK writes here, resumes from here, and recalls a fact.

    This is the assertion the whole design rests on -- P1 proved load() is
    authoritative in isolation; this proves OUR adapter satisfies it.
    """
    import os
    import shutil
    import tempfile

    from app.runtime import driver

    scratch = tempfile.mkdtemp(prefix="sdkstore-")
    config = os.path.join(scratch, "config")
    os.makedirs(config, exist_ok=True)
    # The credentials of record are the pinned dir's, not ~/.claude's. A
    # refresh rotates the refresh token, so the dir that refreshes is the only
    # one still holding a live one -- and the service has been refreshing the
    # pinned copy since it was seeded. Reading ~/.claude here failed with
    # "OAuth session expired and could not be refreshed" while the running
    # service was perfectly authenticated.
    src = driver.credentials_path()
    if driver.token_expiry(src) is None:
        src = os.path.expanduser("~/.claude/.credentials.json")
    if driver.token_expiry(src) is None:
        pytest.skip("no OAuth credentials on this box")
    # And do not run at all on a token near expiry: this scratch copy would
    # refresh, rotate the refresh token out from under the running service,
    # and take the owner's agent down to prove a point about session ids.
    import time
    if driver.token_expiry(src) - time.time() < driver.AUTH_MARGIN_SECONDS:
        pytest.skip("the live token is near expiry; refreshing a copy would orphan it")
    shutil.copy2(src, os.path.join(config, ".credentials.json"))

    old_config = os.environ.get("CLAUDE_CONFIG_DIR")
    os.environ["CLAUDE_CONFIG_DIR"] = config
    os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                                      ResultMessage, TextBlock, query)

        sid = store.session_id_for("telegram", "e2e")
        adapter = ConversationStore()

        async def run(prompt, resume=None, session_id=None):
            opts = ClaudeAgentOptions(
                model="claude-haiku-4-5-20251001", max_turns=2,
                allowed_tools=[], tools=[], setting_sources=[],
                session_store=adapter,
                **({"resume": resume} if resume else {}),
                **({"session_id": session_id} if session_id else {}),
            )
            text, result = [], None
            async for m in query(prompt=prompt, options=opts):
                if isinstance(m, AssistantMessage):
                    for b in m.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            text.append(b.text.strip())
                elif isinstance(m, ResultMessage):
                    result = m
            return " ".join(text), (result.session_id if result else None)

        # Hand the SDK the id we minted. If it honours it, the driver can key
        # a conversation by its own id; if not, the driver has to record the
        # SDK's id after the first turn instead. Either way the mapping must
        # exist before append() fires, or entries land under __orphan__.
        _, real_sid = await run("Remember the code word: WALRUS. Reply only: OK",
                                session_id=sid)
        assert real_sid == sid, (
            "the SDK did not honour our session_id (got %s, wanted %s); the driver "
            "must record the SDK's id after turn one" % (real_sid, sid))

        assert store.relationship_of(real_sid) == ("telegram", "e2e")
        assert store.load_entries("telegram", "e2e"), "nothing was stored"

        answer, _ = await run("What was the code word? One word only.", resume=real_sid)
        assert "walrus" in answer.lower(), answer
    finally:
        if old_config is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = old_config
        shutil.rmtree(scratch, ignore_errors=True)
