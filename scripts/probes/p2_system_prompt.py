#!/usr/bin/env python3
"""P2 — does a changed system_prompt take effect on resume?

This decides persistent-vs-per-turn client, and it is not a style question.
Three of the nine agents pass an instruction PROVIDER rather than a string, so
the system prompt is rebuilt on every single turn:

  root_agent.py:301      appends profile_text()
  instagram_agent.py:194 appends the DM-wording rules
  auto_reply_agent.py:702 assembles BASE + tool note + profile + NATURAL_VOICE
                          + SILENCE_RULE + corrections_text(platform) + ...

If resuming a session freezes the prompt it was created with, then a
long-lived client per conversation would serve a stale personality forever --
newly learned facts and newly taught reply corrections would never reach the
model. That would kill the persistent-client design outright.

The experiment: create a session under system prompt A, resume it under a
contradictory system prompt B, and ask which rule is in force.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="p2-")
CONFIG = os.path.join(SCRATCH, "config")
os.makedirs(CONFIG, exist_ok=True)
_src = os.path.expanduser("~/.claude/.credentials.json")
if os.path.exists(_src):
    shutil.copy2(_src, os.path.join(CONFIG, ".credentials.json"))
os.environ["CLAUDE_CONFIG_DIR"] = CONFIG
os.environ.pop("ANTHROPIC_API_KEY", None)

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

PROMPT_A = "You must always answer with exactly the single word: ALPHA"
PROMPT_B = "You must always answer with exactly the single word: BRAVO"


class Store:
    def __init__(self):
        self.entries = {}

    @staticmethod
    def _k(key):
        return (key.get("project_key"), key.get("session_id"), key.get("subpath") or "")

    async def append(self, key, entries):
        self.entries.setdefault(self._k(key), []).extend(entries)

    async def load(self, key):
        return list(self.entries.get(self._k(key), [])) or None


async def ask(prompt, system, store, resume=None):
    opts = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=2,
        allowed_tools=[],
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=[],
        system_prompt=system,
        session_store=store,
        **({"resume": resume} if resume else {}),
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


async def main():
    out = {"probe": "P2", "question": "does a changed system_prompt apply on resume?"}
    store = Store()

    first, sid = await ask("Say something.", PROMPT_A, store)
    out["turn1_under_A"] = {"reply": first[:80], "session_id": sid}

    second, _ = await ask("Say something.", PROMPT_B, store, resume=sid)
    out["turn2_resumed_under_B"] = {"reply": second[:80]}

    said_a = "ALPHA" in second.upper()
    said_b = "BRAVO" in second.upper()

    if said_b and not said_a:
        mode = "per_turn"
        verdict = ("APPLIED — the new system_prompt takes effect on resume. A "
                   "long-lived client per conversation is viable, and the three "
                   "instruction providers can be re-resolved before each turn.")
    elif said_a and not said_b:
        mode = "frozen"
        verdict = ("FROZEN — resume keeps the prompt the session was created "
                   "with. A persistent client would serve a stale personality: "
                   "newly learned facts and reply corrections would never land. "
                   "The runner must build a fresh client per turn.")
    else:
        mode = "inconclusive"
        verdict = ("INCONCLUSIVE — the reply matched both or neither marker; "
                   "rerun with stronger instructions.")

    if not (("ALPHA" in first.upper()) and not ("BRAVO" in first.upper())):
        mode = "inconclusive"
        verdict = ("INCONCLUSIVE — the control failed: turn 1 did not obey "
                   "prompt A, so turn 2 proves nothing.")

    out["SYSTEM_PROMPT_MODE"] = mode
    out["verdict"] = verdict
    print(json.dumps(out, indent=1, ensure_ascii=False))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
