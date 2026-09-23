"""The Telegram side: who is answered, and how replies are cut up."""
from types import SimpleNamespace

from app.channels import telegram_channel as tc


def update_from(user_id):
    user = SimpleNamespace(id=user_id, username="someone") if user_id else None
    return SimpleNamespace(effective_user=user)


def test_only_the_allowed_user_is_answered(monkeypatch):
    monkeypatch.setattr(tc, "ALLOWED_TELEGRAM_USER_ID", "100000001")
    assert tc.is_authorized(update_from(100000001)) is True
    assert tc.is_authorized(update_from(100000002)) is False
    assert tc.is_authorized(update_from(None)) is False


def test_an_unset_allowlist_answers_nobody(monkeypatch):
    monkeypatch.setattr(tc, "ALLOWED_TELEGRAM_USER_ID", "")
    assert tc.is_authorized(update_from(100000001)) is False


def test_short_text_is_one_message():
    assert tc.split_for_telegram("hello") == ["hello"]


def test_long_text_splits_on_lines_under_the_limit():
    text = "\n".join("line %d %s" % (i, "x" * 50) for i in range(400))
    chunks = tc.split_for_telegram(text, limit=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text


def test_a_single_huge_line_is_split_hard_not_dropped():
    chunks = tc.split_for_telegram("y" * 2500, limit=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_empty_text_still_says_something():
    assert tc.split_for_telegram("") == ["(empty response)"]


def test_a_reply_carries_the_quoted_message_as_data():
    replied = SimpleNamespace(text="Job job-1 failed: tests red",
                              from_user=SimpleNamespace(is_bot=True))
    message = SimpleNamespace(reply_to_message=replied)
    out = tc._with_quoted_context(message, "fix this")
    assert out.endswith("fix this")
    assert "> Job job-1 failed: tests red" in out
    assert "YOUR OWN earlier message" in out
    assert "not a new instruction" in out


def test_a_plain_message_is_unchanged():
    assert tc._with_quoted_context(SimpleNamespace(reply_to_message=None), "hi") == "hi"
