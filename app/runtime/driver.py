"""The turn driver: one Telegram message in, one Claude Agent SDK run, one reply out.

One call in -- run_turn(platform, user_id, text) -- one reply out, with the
conversation persisted through app.memory.store and every dangerous tool passed
through the effect ledger. app.channels.base is a thin caller over this.

Three things here are not obvious and were each learned the hard way.

RESUME IS NOT REPLAY. The SDK cannot re-run a stored turn, so resumption works
by handing back the session id and letting load() supply the history. That makes the stored chain authoritative,
which in turn makes an interrupted turn dangerous: if the process died between
an effectful tool (a coding job starting) completing and its tool_result being
written, the chain ends on an unanswered tool_use. Anthropic rejects that shape
outright, and the naive repair -- re-running the tool to get a result -- starts
the job twice.
resolve_dangling() closes it from the ledger instead, so the wire call is never
repeated.

THE SDK HONOURS A CALLER-SUPPLIED session_id. Measured in
tests/test_sdk_store.py: pass session_id= on the first turn and ResultMessage
comes back with exactly that id. Without it the SDK mints its own, our
relationship row never matches, and every entry lands under __orphan__.

PINNING CLAUDE_CONFIG_DIR MOVES WHERE CREDENTIALS ARE READ. Probe P1: an
unpopulated pinned dir fails with "Not logged in". So ensure_config() copies
the OAuth credentials in -- and then owns them, because once the pinned dir
starts refreshing its own token the two files diverge. Two rules fall out of
that, both paid for in a dead Telegram chat on 8 Sep: copy on which token
lives longer rather than which file is newer (shutil.copy2 preserves mtime,
so a copy of ~/.claude looks exactly as new as ~/.claude does), and renew the
pinned token ahead of expiry instead of letting the CLI discover it is dead
in the middle of the owner's turn.
"""

import asyncio
import contextvars
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid as uuidlib
from datetime import datetime, timezone

from app.effects import ledger, registry
from app.memory import store, window
from app.memory.sdk_store import ConversationStore
from app.runtime import hooks, toolserver
from app.runtime.prompt import system_prompt as owner_prompt

logger = logging.getLogger("orchestrator.driver")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The owner agent is the one place worth spending on: it routes everything,
# reads files, runs shell commands and decides what to say to real people.
# Overridable so a cheaper model can be A/B'd without a code change.
MODEL = os.environ.get("ORCHESTRATOR_MODEL", "claude-sonnet-5")

# A real request ("look at the failing service, fix the config, restart it and
# tell me what you changed") legitimately runs a dozen-plus tools, so this is
# set high enough not to truncate honest work and low enough to stop a loop
# burning the subscription overnight.
MAX_TURNS = int(os.environ.get("ORCHESTRATOR_MAX_TURNS", "60"))

CONFIG_DIR = os.environ.get("ORCHESTRATOR_CLAUDE_CONFIG",
                            os.path.join(ROOT, "data", "claude-config"))
CREDENTIALS_SOURCE = os.path.expanduser("~/.claude/.credentials.json")

# An access token lives eight hours, and the CLI only reaches for the refresh
# token once the access token is already dead -- so a refresh that fails fails
# a real message from the owner. Renewing this far ahead of expiry moves that
# risk off their message path entirely.
AUTH_MARGIN_SECONDS = int(os.environ.get("ORCHESTRATOR_AUTH_MARGIN", "1800"))

# The renewal call's answer is thrown away; only the token it mints matters.
AUTH_PROBE_MODEL = os.environ.get("ORCHESTRATOR_AUTH_PROBE_MODEL", "haiku")

# Channels deliver concurrently, so two chats can want a renewal at once. Only
# one may run: a refresh rotates the refresh token, so the loser of a race
# would be holding -- and restoring -- credentials that are already dead.
_renewal_lock = threading.Lock()
_renewed_at = 0.0

# How long a completed renewal answers for anyone queued behind it.
AUTH_RACE_SECONDS = 120

NO_RESPONSE = "(no response)"

BUSY_MESSAGE = (
    "I'm still working on your previous message -- give me a moment and send "
    "that again once I've replied."
)

EMPTY_TURN_MESSAGE = (
    "I came back with nothing that time and no tool ran, so I genuinely don't "
    "know what happened. Nothing was lost: ask me again and I'll look at what "
    "actually ran rather than guess."
)

# One in-flight turn per counterparty. Channels deliver concurrently, and two
# turns sharing one session interleave their entries and corrupt that
# conversation's history.
_turn_locks: dict = {}

_server = None
_allowed = None


# ------------------------------------------------------------------ auth setup

def credentials_path() -> str:
    return os.path.join(CONFIG_DIR, ".credentials.json")


def token_expiry(path: str):
    """When the OAuth access token in `path` dies, as a unix timestamp.

    None means the file is absent, unreadable, or not a credentials file --
    every caller here reads that as "no token worth keeping".
    """
    try:
        with open(path) as handle:
            oauth = json.load(handle)["claudeAiOauth"]
        return float(oauth["expiresAt"]) / 1000.0
    except (OSError, ValueError, TypeError, KeyError):
        return None


def ensure_config() -> str:
    """Pin an isolated CLAUDE_CONFIG_DIR that is actually logged in."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    target = credentials_path()
    if os.path.exists(CREDENTIALS_SOURCE):
        # Compare the tokens, not the files. copy2 preserves mtime, so the
        # copy is always exactly as new as its source and "source is newer"
        # decays into a coin toss the moment the pinned dir refreshes on its
        # own -- at which point copying can hand the CLI back a refresh token
        # it has already rotated past, which no retry recovers from.
        source_expiry = token_expiry(CREDENTIALS_SOURCE)
        target_expiry = token_expiry(target)
        if target_expiry is None or (source_expiry or 0) > target_expiry:
            shutil.copy2(CREDENTIALS_SOURCE, target)
            logger.info("Copied longer-lived OAuth credentials into %s", CONFIG_DIR)
    os.environ["CLAUDE_CONFIG_DIR"] = CONFIG_DIR
    # This system runs on the owner's subscription. A stray API key in the
    # environment would silently start billing per token instead.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return CONFIG_DIR


def refresh_credentials(force: bool = False) -> bool:
    """Renew the pinned OAuth token before a turn has to depend on it.

    Two measurements shape this. Nothing cheap renews a token -- `claude auth
    status` reports on the stored one and leaves it exactly as it found it --
    so the renewal is a real call, on the smallest model, with its answer
    thrown away. And the CLI decides whether to refresh purely from the stored
    expiresAt, so asking it to renew a token that still has time left is a
    no-op: the expiry is backdated first, which is the only way to say "treat
    this as due" to a process that owns the decision. If the refresh does not
    happen the original expiry goes back, and the token that was working a
    moment ago is still working.

    Returns whether the pinned token is usable afterwards.
    """
    global _renewed_at
    target = credentials_path()
    expiry = token_expiry(target)
    if not force and expiry is not None and expiry - time.time() > AUTH_MARGIN_SECONDS:
        return True

    with _renewal_lock:
        # Whoever held the lock may have just done this work. Their token is
        # the live one; minting a second would only rotate theirs out.
        if time.time() - _renewed_at < AUTH_RACE_SECONDS:
            return True
        renewed = _renew_once(target)
    if renewed:
        _renewed_at = time.time()
    return renewed


def _renew_once(target: str) -> bool:
    """The renewal itself. Callers hold _renewal_lock."""
    binary = shutil.which("claude")
    if not binary:
        logger.error("No claude CLI on PATH to renew the OAuth token with")
        return False

    try:
        with open(target) as handle:
            original = handle.read()
        stored = json.loads(original)
        stored["claudeAiOauth"]["expiresAt"] = int((time.time() - 60) * 1000)
        with open(target, "w") as handle:
            json.dump(stored, handle)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("Could not stage the pinned credentials for renewal: %s", exc)
        return False

    env = dict(os.environ, CLAUDE_CONFIG_DIR=CONFIG_DIR)
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        proc = subprocess.run([binary, "-p", "ok", "--model", AUTH_PROBE_MODEL],
                              env=env, capture_output=True, text=True, timeout=180)
        detail = (proc.stderr or proc.stdout or "").strip()
        code = proc.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        detail, code = str(exc), None

    renewed = token_expiry(target)
    if renewed is not None and renewed > time.time():
        logger.info("Renewed the pinned OAuth token; good for another %d min",
                    int((renewed - time.time()) / 60))
        return True

    # Nothing was minted, so the backdated expiry is a lie that would send the
    # next turn straight back here. Put the working token back as it was.
    try:
        with open(target, "w") as handle:
            handle.write(original)
    except OSError as exc:
        logger.error("Could not restore the pinned credentials: %s", exc)
    logger.error("OAuth renewal minted no token (exit %s): %s", code, detail[:500])
    return False


# ---------------------------------------------------------------- tool surface

def server():
    """The one in-process MCP server, built once."""
    global _server, _allowed
    if _server is None:
        built, missing = toolserver.build_tools()
        if missing:
            logger.error("Tools in schemas/tools.json with no callable: %s", ", ".join(missing))
        _server = toolserver.make_server(built)
        _allowed = ["mcp__orchestrator__%s" % name for name, _, _, _ in built]
        logger.info("Tool server built with %d tools", len(built))
    return _server


def allowed_tool_names() -> list:
    server()
    return list(_allowed)


def build_options(session_id: str, resume: str = None, system_prompt: str = None,
                  allowed_tools: list = None, max_turns: int = None):
    """Options for one run."""
    from claude_agent_sdk import ClaudeAgentOptions

    kwargs = dict(
        model=MODEL,
        max_turns=max_turns or MAX_TURNS,
        system_prompt=system_prompt if system_prompt is not None else owner_prompt(),
        mcp_servers={"orchestrator": server()},
        allowed_tools=allowed_tools if allowed_tools is not None else allowed_tool_names(),
        # Both are load-bearing and both are asserted before every start.
        # setting_sources=[] keeps the host's own settings out; tools=[] keeps
        # the CLI from advertising its built-ins and from deferring our MCP
        # tools behind a search step.
        tools=[],
        setting_sources=[],
        # The tool surface is already exactly ours, so there is nothing left
        # for a permission prompt to protect -- and a prompt in a headless
        # service is a hang, not a safeguard. The real gate is the PreToolUse
        # hook below, which bypassPermissions does not skip.
        permission_mode="bypassPermissions",
        hooks=hooks.matchers(),
        session_store=ConversationStore(),
    )
    if resume:
        kwargs["resume"] = resume
    else:
        kwargs["session_id"] = session_id
    return ClaudeAgentOptions(**kwargs)


# ------------------------------------------------------- interrupted-turn repair

def _iso(ts: float = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), timezone.utc).isoformat()


def _blocks(entry: dict) -> list:
    message = entry.get("message") or {}
    content = message.get("content")
    return content if isinstance(content, list) else []


def unanswered_tool_uses(entries: list) -> list:
    """[(tool_use_id, name, input)] the chain never returned a result for.

    Only the tail matters: an unanswered call in the middle would already have
    broken the turn it was in, and rewriting history that far back is how a
    conversation quietly loses a real answer.
    """
    answered = set()
    for entry in entries:
        for block in _blocks(entry):
            if block.get("type") == "tool_result" and block.get("tool_use_id"):
                answered.add(block["tool_use_id"])

    pending = []
    for entry in reversed(entries):
        if entry.get("type") != "assistant":
            continue
        calls = [b for b in _blocks(entry) if b.get("type") == "tool_use"]
        if not calls:
            # An assistant turn that ended in text is a clean stopping point.
            break
        pending = [(b.get("id"), b.get("name"), b.get("input") or {})
                   for b in calls if b.get("id") not in answered]
        break
    return [p for p in pending if p[0]]


def _closing_result(name: str, tool_input: dict):
    """What to hand back for a call that never returned. Never re-runs it."""
    # The stored chain carries the wire name (mcp__orchestrator__x); the
    # registry is keyed by the bare one.
    name = hooks.bare_name(name)
    kind = registry.registered().get(name)
    if kind:
        make = registry.fingerprint_for(name)
        mark = None
        try:
            mark = make((), dict(tool_input)) if make else None
        except Exception:
            logger.exception("Could not fingerprint %s while repairing history", name)
        if mark:
            done = ledger.check(name, kind, mark)
            if done is not None:
                return json.dumps({"ok": True, "recovered": True,
                                   "result": done.get("result")},
                                  ensure_ascii=False, default=str), False
            if any(row["tool"] == name and row["fingerprint"] == mark
                   for row in ledger.dangling(kind)):
                # The intent was committed and no outcome came back. It may
                # have shipped. Saying "it failed" would get it sent twice;
                # saying "it worked" would be a guess. Say exactly this.
                return json.dumps({
                    "ok": None,
                    "interrupted": True,
                    "note": ("This call was interrupted after it was committed. It MAY "
                             "have taken effect. Do NOT run it again -- check the result "
                             "first, or tell the owner you cannot confirm it."),
                }, ensure_ascii=False), False
    return json.dumps({
        "ok": False,
        "interrupted": True,
        "error": "The process stopped before this tool returned. It had no recorded "
                 "effect, so it is safe to try again if it still matters.",
    }, ensure_ascii=False), True


def resolve_dangling(platform: str, user_id: str) -> int:
    """Close any tool_use the chain never answered. Returns how many."""
    entries = store.load_entries(platform, user_id)
    pending = unanswered_tool_uses(entries)
    if not pending:
        return 0

    blocks = []
    for tool_use_id, name, tool_input in pending:
        content, is_error = _closing_result(name, tool_input)
        blocks.append({"type": "tool_result", "tool_use_id": tool_use_id,
                       "content": content, "is_error": is_error})
        logger.warning("Closing an unanswered %s from the ledger (error=%s)", name, is_error)

    store.append_entries(platform, user_id, [{
        "type": "user",
        "uuid": str(uuidlib.uuid4()),
        # Links to the entry it closes. An unlinked record is unreachable when
        # the SDK reconstructs the conversation, so a repair that skipped this
        # would leave the chain looking mended while the resumed turn saw
        # nothing before it.
        "parentUuid": entries[-1].get("uuid") if entries else None,
        "timestamp": _iso(),
        "message": {"role": "user", "content": blocks},
    }])
    return len(blocks)


# ---------------------------------------------------------------- the turn itself

class TurnCursor:
    """What one turn produced, assembled as the stream arrives."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget a failed attempt, so a retry is not read as a continuation."""
        self.texts = []
        self.tool_names = {}
        self.tool_results = []
        self.session_id = None
        self.cost_usd = None
        self.hit_limit = False

    def add_text(self, text: str) -> None:
        if text and text.strip():
            self.texts.append(text.strip())

    def add_tool_use(self, tool_use_id: str, name: str, tool_input) -> None:
        self.tool_names[tool_use_id] = hooks.bare_name(name)

    def add_tool_result(self, tool_use_id: str, content, is_error: bool) -> None:
        name = self.tool_names.get(tool_use_id, "?")
        self.tool_results.append((name, _as_dict(content), bool(is_error)))

    def finish(self, result_message) -> None:
        self.session_id = getattr(result_message, "session_id", None)
        self.cost_usd = getattr(result_message, "total_cost_usd", None)
        subtype = getattr(result_message, "subtype", "") or ""
        self.hit_limit = "max_turns" in subtype

    def reply(self) -> str:
        if self.texts:
            return self.texts[-1]
        described = describe_tool_outcome(self.tool_results)
        if described:
            logger.info("Turn ended without text; described it from %d tool results",
                        len(self.tool_results))
            return described
        return NO_RESPONSE


def _as_dict(content):
    """A tool result comes back as a JSON string inside an MCP envelope."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return _as_dict(first.get("text"))
        return _as_dict(first)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"text": content}
        return parsed if isinstance(parsed, dict) else {"text": content}
    return {}


def describe_tool_outcome(results: list) -> str:
    """Say what a turn did, from its tool results alone.

    A turn can do real work -- the job really was started -- and end without
    text, at which point "(no response)" makes the owner ask again and it
    starts twice. The tool result is authoritative about what happened.
    """
    for name, response, is_error in reversed(results):
        if name == "?" or not isinstance(response, dict):
            continue
        if is_error or response.get("ok") is False:
            return "That didn't work: %s" % response.get("error", "unknown error")
        if response.get("job_id"):
            return "Coding job %s is %s." % (response["job_id"],
                                             response.get("status", "running"))
        if response.get("ok"):
            return "Done (%s)." % name
    return ""


async def _stream(prompt: str, options, cursor: TurnCursor) -> None:
    from claude_agent_sdk import (AssistantMessage, ResultMessage, TextBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage, query)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    cursor.add_text(block.text)
                elif isinstance(block, ToolUseBlock):
                    cursor.add_tool_use(block.id, block.name, block.input)
        elif isinstance(message, UserMessage):
            content = getattr(message, "content", None)
            for block in content if isinstance(content, list) else []:
                if isinstance(block, ToolResultBlock):
                    cursor.add_tool_result(block.tool_use_id, block.content,
                                           getattr(block, "is_error", False))
        elif isinstance(message, ResultMessage):
            cursor.finish(message)


async def _stream_with_reauth(prompt: str, options, cursor: TurnCursor) -> None:
    """Run the turn, and if it died on authentication, renew and run it again.

    An expired token is the one failure that costs the owner a whole message
    for a reason that has nothing to do with what they asked -- and on 8 Sep
    it cost them three in a row, each answered with "something went wrong".
    One retry is the whole policy: if the renewal cannot mint a live token
    the error is real and belongs on the surface, not in a loop.
    """
    try:
        await _stream(prompt, options, cursor)
        return
    except Exception as exc:
        if not _is_auth_problem(exc):
            raise
        logger.error("Turn failed to authenticate (%s); renewing the pinned "
                     "token and running it once more", exc)
        if not await asyncio.to_thread(refresh_credentials, True):
            raise
    cursor.reset()
    await _stream(prompt, options, cursor)


async def run_turn(platform: str, user_id: str, text: str) -> str:
    """One agent turn. The resume ladder is the whole of the error handling."""
    ensure_config()
    # Off-thread: this is a no-op on all but one turn in eight hours, and that
    # one turn should not stall every other conversation on the event loop.
    await asyncio.to_thread(refresh_credentials)
    session_id = store.session_id_for(platform, user_id)
    resolve_dangling(platform, user_id)
    has_history = bool(store.load_entries(platform, user_id))
    turn_id = "%s:%s:%d" % (platform, user_id, time.time_ns())
    return await _ladder(platform, user_id, text, session_id, turn_id, has_history)


async def _ladder(platform, user_id, text, session_id, turn_id, has_history) -> str:
    """The resume ladder."""
    # Tier 1: resume the stored conversation.
    # Tier 2: it would not load -- drop the trailing turn and resume again.
    # Tier 3: start clean rather than answer nothing. Loud, because it means
    #         this conversation just lost its memory of everything before now.
    for tier in (1, 2, 3):
        resume = session_id if (has_history and tier < 3) else None
        options = build_options(session_id, resume=resume)
        toolserver.assert_isolation(options)
        cursor = TurnCursor()
        try:
            with ledger.current_turn(turn_id):
                await _stream_with_reauth(text, options, cursor)
        except Exception as exc:
            if tier == 3 or not _is_history_problem(exc):
                raise
            if tier == 1:
                removed = store.forget_turn_since(platform, user_id)
                logger.error("Resume failed (%s); dropped the last %d entries and "
                             "retrying", exc, removed)
                has_history = bool(store.load_entries(platform, user_id))
            else:
                logger.error("Resume failed again (%s); starting this conversation "
                             "clean", exc)
            continue

        if cursor.hit_limit:
            logger.warning("Turn for %s/%s hit the %d-turn ceiling",
                           platform, user_id, MAX_TURNS)
        reply = cursor.reply()
        return EMPTY_TURN_MESSAGE if reply == NO_RESPONSE else reply

    raise RuntimeError("unreachable")


def _is_auth_problem(exc: Exception) -> bool:
    """A token the API rejected, as the CLI words it.

    Deliberately broad: the cost of a false positive is one wasted renewal
    and one honest retry, and the cost of a false negative is the owner being
    told "something went wrong" about a token that could have been renewed.
    """
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(mark in text for mark in
               ("401", "oauth", "authenticate", "unauthorized"))


def _is_history_problem(exc: Exception) -> bool:
    """Only a malformed or missing conversation is worth re-trying differently.

    A tool blowing up or a network drop needs to surface; silently restarting
    the conversation would hide it and cost the owner their history for a
    reason that had nothing to do with history. An expired token is the one
    exception, and it is answered before this is ever asked -- see
    _stream_with_reauth.
    """
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(mark in text for mark in (
        "session", "resume", "tool_use", "tool_result", "invalid_request",
        "messages:", "unexpected role", "alternat",
    ))


# --------------------------------------------------------------- channel facade

def _get_turn_lock(platform: str, user_id: str) -> asyncio.Lock:
    key = (platform, user_id)
    if key not in _turn_locks:
        _turn_locks[key] = asyncio.Lock()
    return _turn_locks[key]


def is_busy(platform: str, user_id: str) -> bool:
    key = (platform, user_id)
    return key in _turn_locks and _turn_locks[key].locked()


async def handle_incoming(platform: str, user_id: str, text: str) -> str:
    lock = _get_turn_lock(platform, user_id)
    if lock.locked():
        logger.info("Rejected overlapping turn for %s/%s", platform, user_id)
        return BUSY_MESSAGE
    async with lock:
        return await run_turn(platform, user_id, text)
