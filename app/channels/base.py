"""The channel entry point. One turn per counterparty, run by the driver.

Two things live here. The per-counterparty lock is in the driver (channels
deliver concurrently, and two turns sharing one session interleave their
entries); this module adds the exception contract on top of it:

  * a subscription usage limit is answered with an explanation, because it is
    not a fault and retrying or /clear does not move it;
  * every other failure RAISES, so app/channels/telegram_channel.py can log the
    real traceback one frame out instead of a sanitised sentence.
"""

import logging

from app.runtime import driver

logger = logging.getLogger("orchestrator.channels")

BUSY_MESSAGE = driver.BUSY_MESSAGE
NO_RESPONSE = driver.NO_RESPONSE
EMPTY_TURN_MESSAGE = driver.EMPTY_TURN_MESSAGE

# A subscription usage limit is not a bug and not a transient blip: it clears on
# its own window and no amount of retrying moves it. Saying "something went
# wrong" here would send the owner looking for a fault that does not exist --
# and worse, towards /reset, which would destroy this conversation to "fix" it.
USAGE_LIMIT_MESSAGE = (
    "I've hit the Claude usage limit on your plan, so I can't think until it "
    "resets. Nothing is broken and our conversation is intact -- don't use "
    "/clear, it won't help and it would wipe everything I remember. Send that "
    "again once the limit resets."
)


def is_busy(platform: str, user_id: str) -> bool:
    """True if an agent turn is currently running for this counterparty."""
    return driver.is_busy(platform, user_id)


def _is_usage_limit(exc: Exception) -> bool:
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return ("usage limit" in text or "rate limit" in text
            or "quota" in text or "429" in text)


async def handle_incoming(platform: str, user_id: str, text: str) -> str:
    try:
        return await driver.handle_incoming(platform, user_id, text)
    except Exception as exc:
        if _is_usage_limit(exc):
            logger.warning("Claude usage limit reached on %s/%s: %s",
                           platform, user_id, exc)
            return USAGE_LIMIT_MESSAGE
        # Everything else is a real failure and must reach the channel, which
        # logs the traceback. Swallowing it here would leave nothing but a
        # sanitised sentence to debug from.
        raise
