"""The decorator that makes a repeated side effect impossible to fire twice."""
import pytest

from app.effects import ledger, registry

# Imported at module scope on purpose. Decoration registers a tool the first
# time its module is imported, and Python caches that -- so if the first import
# happened inside a test, the registration would land in that test's copied
# _REGISTERED and be reverted at teardown, with the re-import a no-op.
from app.tools import claude_code_tool  # noqa: F401,E402


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / "e.db"))
    ledger._reset_for_tests()
    monkeypatch.setattr(registry, "_REGISTERED", dict(registry._REGISTERED))
    yield
    ledger._reset_for_tests()


def test_kind_is_required_and_validated():
    with pytest.raises(TypeError):
        registry.effectful()                      # no kind at all
    with pytest.raises(ValueError):
        registry.effectful(kind="nonsense")


def test_a_sync_tool_is_refused():
    with pytest.raises(TypeError):
        @registry.effectful(kind="spend", fingerprint=lambda a, k: "x")
        def sync_tool(text: str) -> dict:
            """Start. Args: text: what."""
            return {"ok": True}


async def test_the_same_effect_fires_once():
    calls = []

    @registry.effectful(kind="spend", fingerprint=lambda a, k: "job|%s|%s" % (a[0], a[1]))
    async def fake_start(prompt: str, working_dir: str) -> dict:
        """Start a job.

        Args:
            prompt: What.
            working_dir: Where.
        """
        calls.append((prompt, working_dir))
        return {"ok": True, "job_id": "job-1"}

    with ledger.current_turn("t1"):
        first = await fake_start("build it", "/tmp/a")
        second = await fake_start("build it", "/tmp/a")

    assert len(calls) == 1, "the job must start exactly once"
    assert first == {"ok": True, "job_id": "job-1"}
    assert second["job_id"] == "job-1"
    # The caller can tell the difference, so a report can say "already started"
    # rather than claiming a fresh one.
    assert second.get("deduplicated") is True
    assert first.get("deduplicated") is None


async def test_a_different_prompt_still_starts():
    calls = []

    @registry.effectful(kind="spend", fingerprint=lambda a, k: "job|%s" % a[0])
    async def fake_start(prompt: str) -> dict:
        """Start. Args: prompt: what."""
        calls.append(prompt)
        return {"ok": True}

    with ledger.current_turn("t1"):
        await fake_start("build it")
        await fake_start("test it")
    assert len(calls) == 2


async def test_a_failed_call_is_retried_not_blocked():
    attempts = []

    @registry.effectful(kind="spend", fingerprint=lambda a, k: "x")
    async def flaky(prompt: str) -> dict:
        """Start. Args: prompt: what."""
        attempts.append(prompt)
        return {"ok": False, "error": "busy"} if len(attempts) == 1 else {"ok": True}

    with ledger.current_turn("t1"):
        first = await flaky("hi")
        second = await flaky("hi")
    assert first["ok"] is False
    assert second["ok"] is True
    assert len(attempts) == 2


async def test_a_raising_tool_leaves_no_completed_row():
    @registry.effectful(kind="spend", fingerprint=lambda a, k: "boom")
    async def explodes(prompt: str) -> dict:
        """Start. Args: prompt: what."""
        raise RuntimeError("process died")

    with ledger.current_turn("t1"):
        with pytest.raises(RuntimeError):
            await explodes("hi")

    # It MAY have started, so it is dangling: not completed, not replayable.
    assert ledger.check("explodes", "spend", "boom") is None
    assert len(ledger.dangling(kind="spend")) == 1


async def test_an_unfingerprintable_call_still_fires():
    calls = []

    def boom(args, kwargs):
        raise ValueError("cannot fingerprint")

    @registry.effectful(kind="spend", fingerprint=boom)
    async def still_works(prompt: str) -> dict:
        """Start. Args: prompt: what."""
        calls.append(prompt)
        return {"ok": True}

    with ledger.current_turn("t1"):
        out = await still_works("hi")
    assert out == {"ok": True}
    assert calls == ["hi"]


def test_registry_records_the_kind():
    @registry.effectful(kind="deploy", fingerprint=lambda a, k: "d")
    async def deployer(name: str) -> dict:
        """Deploy. Args: name: what."""
        return {"ok": True}

    assert registry.registered()["deployer"] == "deploy"


def test_every_paid_tool_is_classified():
    kinds = registry.registered()
    assert kinds.get("start_coding_job") == "spend"
    assert kinds.get("run_claude_code") == "spend"
    assert set(kinds.values()) <= registry.KINDS


def test_a_dangling_effect_is_reported_not_replayed():
    """Process death mid-call: never re-run, surface it instead."""
    with ledger.current_turn("t9"):
        ledger.record_intent("start_coding_job", "spend", "claude|/srv/app||build")
    assert ledger.check("start_coding_job", "spend", "claude|/srv/app||build") is None
    hanging = ledger.dangling(kind="spend")
    assert [h["tool"] for h in hanging] == ["start_coding_job"]
