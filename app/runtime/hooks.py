"""Binding the effect ledger to the SDK.

PreToolUse is the mandatory interception point: unlike allowed_tools or
permission_mode it cannot be skipped, and a denied call never enters the tool
body at all. That asymmetry is the whole reason to use it -- a denial emits Pre
and NO Post, so "intent with no outcome" stays a meaningful state.

Division of labour with @effectful, which already wraps every dangerous tool:

  PreToolUse   ENFORCES. If the ledger already holds a completed identical
               effect, deny the call and hand the model the first result. The
               tool body is never entered, so nothing double-records.
  @effectful   RECORDS. It owns intent and outcome, because it is the only
               layer that sees the real return value rather than a JSON string
               that has been through an MCP envelope.
  PostToolUse  WITNESSES. It observes independently and logs any disagreement
               with what the tool asserted. It deliberately does NOT insert a
               second row -- two witnesses, one record.

Tool names arrive prefixed (mcp__orchestrator__start_coding_job), so every
lookup strips that first.
"""

import json
import logging

from app.effects import ledger, registry

logger = logging.getLogger("orchestrator.hooks")

MCP_PREFIX = "mcp__"


def bare_name(tool_name: str) -> str:
    """start_coding_job from mcp__orchestrator__start_coding_job."""
    if not tool_name or not tool_name.startswith(MCP_PREFIX):
        return tool_name or ""
    return tool_name.split("__")[-1]


def _fingerprint(name: str, tool_input: dict):
    """The effect's identity, using the same function the decorator uses.

    A hook only ever sees keyword arguments, so the call is ((), tool_input) --
    which is exactly the shape the decorator's lambdas already handle.
    """
    make = registry.fingerprint_for(name)
    if make is None:
        return None
    try:
        return make((), dict(tool_input or {}))
    except Exception:
        logger.exception("Could not fingerprint %s in a hook", name)
        return None


async def pre_tool_use(payload, tool_use_id, context):
    """Deny a repeat of an effect that already completed."""
    name = bare_name(payload.get("tool_name"))
    kind = registry.registered().get(name)
    if not kind:
        return {}

    mark = _fingerprint(name, payload.get("tool_input"))
    if mark is None:
        return {}

    already = ledger.check(name, kind, mark)
    if already is None:
        return {}

    logger.warning("Denying a repeat %s (%s): already completed in turn %s",
                   name, kind, already.get("turn_id"))
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            # The reason is shown to the model, so it must say what happened AND
            # what to do about it. Without the second half the model tends to
            # apologise and try again with a reworded body, which is a new
            # fingerprint and would go through.
            "permissionDecisionReason": (
                "Already done. This exact %s was completed earlier and its result "
                "was: %s. Do NOT retry it or reword it -- treat it as sent and "
                "carry on." % (name, json.dumps(already.get("result"), default=str)[:400])
            ),
        }
    }


async def post_tool_use(payload, tool_use_id, context):
    """Second witness. Observes; never writes a competing row."""
    name = bare_name(payload.get("tool_name"))
    kind = registry.registered().get(name)
    if not kind:
        return {}

    mark = _fingerprint(name, payload.get("tool_input"))
    if mark is None:
        return {}

    recorded = ledger.check(name, kind, mark)
    response = payload.get("tool_response")

    # What the tool told the ledger vs what the transport saw come back.
    asserted_ok = recorded is not None
    observed_ok = _looks_ok(response)

    if asserted_ok != observed_ok:
        logger.warning(
            "Ledger disagreement on %s (%s): tool asserted ok=%s, transport observed "
            "ok=%s. Treating the union as done.", name, kind, asserted_ok, observed_ok)
    return {}


def _looks_ok(response) -> bool:
    """Best effort: the MCP envelope turns a dict into a JSON string."""
    if response is None:
        return False
    if isinstance(response, dict):
        if "ok" in response:
            return bool(response["ok"])
        content = response.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("text") if isinstance(content[0], dict) else None
            return _looks_ok(text)
        return True
    if isinstance(response, str):
        try:
            return _looks_ok(json.loads(response))
        except json.JSONDecodeError:
            return True
    return True


def matchers():
    """HookMatcher list for ClaudeAgentOptions(hooks=...)."""
    from claude_agent_sdk import HookMatcher
    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[post_tool_use])],
    }
