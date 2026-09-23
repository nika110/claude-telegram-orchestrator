"""Declaring what a tool does to the world.

`kind` has NO DEFAULT. Forgetting to classify a new side-effectful tool is a
TypeError at import, not a silent gap discovered when the same job runs twice.

The decorator records; app/runtime/hooks.py enforces from PreToolUse.

Wrapping must not change the tool's signature -- functools.wraps sets
__wrapped__ and inspect.signature follows it. That is pinned by
tests/test_decoration_is_transparent.py, because schemas/tools.json describes
the unwrapped signature.
"""

import functools
import inspect
import logging

from app.effects import ledger

logger = logging.getLogger("orchestrator.effects")

# spend  -- a paid model run starts (every Claude Code job)
# send   -- a person receives something under the owner's name
# deploy -- a service changes on a machine
# device -- the physical world
# Only "spend" is used by the tools that ship here; the others are for tools
# you add. There is deliberately no default kind.
KINDS = frozenset({"send", "deploy", "spend", "device"})

_REGISTERED = {}
# The same fingerprint function the wrapper uses, reachable by tool name so
# a PreToolUse hook can compute the identity of a call it has only seen as
# a keyword dict.
_FINGERPRINTS = {}


def registered() -> dict:
    """{tool name: kind} for everything decorated so far."""
    return dict(_REGISTERED)


def fingerprint_for(name: str):
    """The fingerprint function for a tool, or None if it is not effectful."""
    return _FINGERPRINTS.get(name)


def _default_fingerprint(name: str, args: tuple, kwargs: dict) -> str:
    parts = [repr(a) for a in args] + ["%s=%r" % (k, v) for k, v in sorted(kwargs.items())]
    return "%s|%s" % (name, "|".join(parts))


def effectful(*, kind: str, fingerprint=None):
    """Record this tool's effect in the ledger, and never fire it twice.

    Args:
        kind: One of KINDS. Required -- there is deliberately no default.
        fingerprint: (args, kwargs) -> str identifying "the same effect". For a
            send that is (platform, target, normalised text), so a reworded
            message is a different effect but a re-driven turn is not.
    """
    if kind not in KINDS:
        raise ValueError("unknown effect kind %r; expected one of %s" % (kind, sorted(KINDS)))

    def decorate(fn):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("@effectful expects an async tool; %s is sync" % fn.__name__)

        _REGISTERED[fn.__name__] = kind
        make_fingerprint = fingerprint or (
            lambda args, kwargs: _default_fingerprint(fn.__name__, args, kwargs)
        )
        _FINGERPRINTS[fn.__name__] = make_fingerprint

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                mark = make_fingerprint(args, kwargs)
            except Exception:
                # A fingerprint we cannot compute must not stop the tool from
                # working -- it only costs the dedupe guarantee for this call.
                logger.exception("Could not fingerprint %s; firing without dedupe", fn.__name__)
                return await fn(*args, **kwargs)

            already = ledger.check(fn.__name__, kind, mark)
            if already is not None:
                logger.info(
                    "Skipping a repeat %s (%s); returning the first result", fn.__name__, kind
                )
                result = already["result"]
                if isinstance(result, dict):
                    return dict(result, deduplicated=True)
                return {"ok": True, "deduplicated": True}

            rowid = ledger.record_intent(fn.__name__, kind, mark)
            result = await fn(*args, **kwargs)
            ok = not isinstance(result, dict) or result.get("ok", True)
            ledger.record_outcome(
                rowid, result if isinstance(result, dict) else {"ok": ok}, ok=ok
            )
            return result

        return wrapper

    return decorate
