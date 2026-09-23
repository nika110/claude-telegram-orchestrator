"""The channel facade over the driver."""
import pytest

from app.channels import base
from app.runtime import driver


def test_it_keeps_the_names_the_channels_import():
    # telegram_channel and claude_code_tool import exactly these.
    assert callable(base.handle_incoming)
    assert callable(base.is_busy)
    assert base.BUSY_MESSAGE


async def test_a_turn_goes_through_the_driver(monkeypatch):
    seen = {}

    async def fake(platform, user_id, text):
        seen.update(platform=platform, user_id=user_id, text=text)
        return "done"

    monkeypatch.setattr(driver, "handle_incoming", fake)
    assert await base.handle_incoming("telegram", "1", "hi") == "done"
    assert seen == {"platform": "telegram", "user_id": "1", "text": "hi"}


async def test_a_usage_limit_is_explained_rather_than_reported_as_a_fault(monkeypatch):
    """The owner must not go looking for a bug, or reach for /clear."""
    async def limited(*args):
        raise RuntimeError("Claude API error: usage limit reached")

    monkeypatch.setattr(driver, "handle_incoming", limited)
    reply = await base.handle_incoming("telegram", "1", "hi")
    assert reply == base.USAGE_LIMIT_MESSAGE
    assert "/clear" in reply and "won't help" in reply


async def test_any_other_failure_still_reaches_the_channel(monkeypatch):
    """telegram_channel logs the traceback one frame out; swallowing it here
    would leave nothing to debug from."""
    async def broken(*args):
        raise ValueError("tool blew up")

    monkeypatch.setattr(driver, "handle_incoming", broken)
    with pytest.raises(ValueError):
        await base.handle_incoming("telegram", "1", "hi")


def test_busy_is_answered_from_the_driver_s_locks(monkeypatch):
    monkeypatch.setattr(driver, "is_busy", lambda p, u: p == "telegram")
    assert base.is_busy("telegram", "1") is True
    assert base.is_busy("other", "1") is False
