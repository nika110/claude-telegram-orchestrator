import contextvars
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("orchestrator.progress")

# Set by whichever channel is handling the current turn, so tools deep in the
# call stack can report progress without knowing anything about Telegram.
_notifier: contextvars.ContextVar[Callable[[str], Awaitable[None]] | None] = (
    contextvars.ContextVar("progress_notifier", default=None)
)


# Same idea for sending a file back: tools produce images, only the channel
# knows how to deliver one.
_media_sender: contextvars.ContextVar[Callable[[str, str], Awaitable[None]] | None] = (
    contextvars.ContextVar("media_sender", default=None)
)

# Who the current turn is with. Tools that own per-person state (the reference
# job registry) need this, and nothing passes it down to plain functions.
_person: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("current_person", default=None)
)


def set_notifier(fn: Callable[[str], Awaitable[None]] | None) -> None:
    _notifier.set(fn)


def set_media_sender(fn: Callable[[str, str], Awaitable[None]] | None) -> None:
    _media_sender.set(fn)


def set_person(platform: str | None, user_id: str | None) -> None:
    _person.set((platform, user_id) if platform and user_id else None)


def current_person() -> tuple[str, str]:
    """The (platform, user_id) of the turn in progress.

    Falls back to the Telegram owner rather than raising: a tool that can't
    tell who it's talking to is more useful defaulting to the only person who
    can reach this bot than it is failing.
    """
    return _person.get() or ("telegram", "unknown")


async def send_media(file_path: str, caption: str = "") -> bool:
    """Deliver a file to the user through the current channel."""
    fn = _media_sender.get()
    if fn is None:
        return False
    try:
        await fn(file_path, caption)
        return True
    except Exception:
        logger.warning("Could not deliver media %s", file_path, exc_info=True)
        return False


async def notify(message: str) -> bool:
    """Send an interim progress update to the user, if the channel supports it.

    Returns whether it actually went out. Never raises: a failed progress ping
    must not take down the actual work.

    The return value is load-bearing for one caller. `_report_to_owner` does
    `_mark_reported(job, await notify(...))`, so while this returned None every
    coding job was filed as undelivered however well the send had gone -- and
    the sweeper then pushed the same result out again three minutes later,
    prefixed "this is late -- the message carrying it failed the first time",
    which was untrue. Seen twice on 19 Aug (job-4a64b864, job-7dec5f1b).
    The same bug was fixed in telegram_channel's own notify but not here, and
    this is the one the /claude path actually calls.
    """
    fn = _notifier.get()
    if fn is None:
        return False
    try:
        await fn(message)
        return True
    except Exception:
        logger.warning("Could not deliver progress update", exc_info=True)
        return False
