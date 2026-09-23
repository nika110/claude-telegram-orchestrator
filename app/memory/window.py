"""Choosing how much of the conversation the model reads back.

The full re-encoded history is ~404k tokens. It is legal Anthropic input and it
does not fit in a context window, so something has to choose a slice. P1 settled
where that choice lives: load() IS the resume source, so the slice this module
returns is exactly what the model sees.

Three rules, in order of how often they bite:

  BUDGET   stop at roughly 20k tokens of conversation. Estimated, not counted --
           an exact count would need a tokenizer call per turn and the cost of
           being wrong here is a slightly shorter window, not a broken one.
  TURNS    at most 30 turns, so a long thread of one-word messages cannot
           crowd out the system prompt.
  GAP      cut at a silence longer than 6 hours. Yesterday's argument is not
           context for this morning's "what time?", and carrying it makes the
           model answer the wrong question.

Always a CONTIGUOUS SUFFIX, never a filtered subset. The entries form a parent
chain; a window with a hole in the middle is not a chain, and the first entry
having no live parent is fine (it becomes the root) while a gap in the middle
is not.

Cut on TURN boundaries. A window that starts with a tool_result whose tool_use
was trimmed away is illegal input -- Anthropic requires every tool_result to
answer a visible call.
"""

import json

MAX_TURNS = 30
MAX_CHARS = 80_000          # ~20k tokens at the usual 4 chars/token
# Three days, not six hours. At six the owner's own thread collapsed to two
# entries -- he sleeps -- and past three days MAX_CHARS binds first, so the
# gap rule stopped bounding the window and started deleting the memory.
MAX_GAP_SECONDS = 3 * 24 * 3600


def _timestamp(entry: dict) -> float:
    raw = str(entry.get("timestamp") or "")
    if not raw:
        return 0.0
    # ISO-8601, possibly with a trailing Z. Parsed by hand rather than with
    # dateutil: this runs on every turn and the format is the SDK's own.
    try:
        import datetime
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _size(entry: dict) -> int:
    return len(json.dumps(entry, ensure_ascii=False, default=str))


def _starts_a_turn(entry: dict) -> bool:
    """True for a user entry that is a real message, not a tool result.

    A user entry whose content is only tool_result blocks is the back half of
    the assistant's turn, not the start of a new one -- cutting there would
    orphan the tool_use it answers.
    """
    if entry.get("type") != "user":
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    blocks = [b for b in (content or []) if isinstance(b, dict)]
    if not blocks:
        return True
    return any(b.get("type") != "tool_result" for b in blocks)


def build(entries: list, max_turns: int = MAX_TURNS, max_chars: int = MAX_CHARS,
          max_gap: float = MAX_GAP_SECONDS) -> list:
    """The contiguous suffix of `entries` the model should read back."""
    if not entries:
        return []

    # Walk backwards, remembering the last legal place to cut.
    turns = 0
    chars = 0
    cut = 0                     # index to start the window at
    previous_ts = None

    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        chars += _size(entry)

        stamp = _timestamp(entry)
        if previous_ts and stamp and (previous_ts - stamp) > max_gap:
            # A long silence: everything older belongs to another conversation.
            cut = index + 1
            break
        if stamp:
            previous_ts = stamp

        if _starts_a_turn(entry):
            turns += 1
            if turns > max_turns or chars > max_chars:
                cut = index + 1
                break
            cut = index

    window = entries[cut:]
    # Never hand back a window that opens on an orphaned tool_result.
    while window and not _starts_a_turn(window[0]) and window[0].get("type") == "user":
        window = window[1:]
    return window


def validate(window: list) -> list:
    """Problems that would make this window illegal input. Empty means fine."""
    problems = []
    if not window:
        return problems

    open_calls = set()
    answered = set()
    for index, entry in enumerate(window):
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str) or content is None:
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                open_calls.add(block.get("id"))
            elif block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                answered.add(tid)
                if tid not in open_calls:
                    problems.append(
                        "entry %d answers tool call %r that is not in the window" % (index, tid))

    # The other half of the same invariant, and the one a killed process
    # actually leaves behind: every tool_use must be answered. The API rejects
    # this outright, so a window carrying it is not merely untidy.
    for tool_use_id in sorted(open_calls - answered, key=str):
        problems.append("tool call %r is never answered" % tool_use_id)
    return problems
