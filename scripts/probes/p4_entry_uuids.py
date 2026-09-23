#!/usr/bin/env python3
"""P4 — are the entry uuids we see in append() usable as resume addresses?

The driver needs to know where it is in the transcript without reading the
store back and racing its own writer. Since the app implements append(), it
sees every entry as it is written -- but only if those entries actually carry
stable uuids, and only if resume_session_at accepts one.

That second half is also the mechanism the design leans on for pruning: cutting
the last turn is "resume at the uuid before it".

The experiment: run three turns, capture every entry uuid, then resume at the
uuid of an early entry and ask about a fact stated AFTER it. If resume_session_at
truncates the model's view, the fact is unknown.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="p4-")
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


class Store:
    def __init__(self):
        self.entries = {}
        self.append_log = []

    @staticmethod
    def _k(key):
        return (key.get("project_key"), key.get("session_id"), key.get("subpath") or "")

    async def append(self, key, entries):
        for e in entries:
            self.append_log.append({
                "subpath": self._k(key)[2],
                "type": e.get("type"),
                "uuid": e.get("uuid"),
                "parentUuid": e.get("parentUuid"),
                "has_ts": bool(e.get("timestamp")),
            })
        self.entries.setdefault(self._k(key), []).extend(entries)

    async def load(self, key):
        return list(self.entries.get(self._k(key), [])) or None


async def ask(prompt, store, resume=None, at=None):
    kw = {}
    if resume:
        kw["resume"] = resume
    if at:
        kw["resume_session_at"] = at
    opts = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=2,
        allowed_tools=[], tools=[], setting_sources=[],
        permission_mode="bypassPermissions",
        session_store=store,
        **kw,
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
    out = {"probe": "P4", "question": "are append() entry uuids stable and addressable?"}
    store = Store()

    _, sid = await ask("Fact one: the colour is CRIMSON. Reply only: OK", store)
    mark = len(store.append_log)
    await ask("Fact two: the animal is OTTER. Reply only: OK", store, resume=sid)

    uuids = [e["uuid"] for e in store.append_log if e["uuid"]]
    out["entries_seen"] = len(store.append_log)
    out["entries_with_uuid"] = len(uuids)
    out["uuids_unique"] = len(set(uuids)) == len(uuids)
    out["entry_types"] = sorted({e["type"] for e in store.append_log if e["type"]})
    out["subpaths_seen"] = sorted({e["subpath"] for e in store.append_log})
    out["parent_links_present"] = sum(
        1 for e in store.append_log if e.get("parentUuid")) > 0

    # Control: full resume must know BOTH facts.
    control, _ = await ask(
        "Name the colour and the animal. Two words, comma separated.", store, resume=sid)
    out["control"] = {
        "reply": control[:120],
        "knew_colour": "crimson" in control.lower(),
        "knew_animal": "otter" in control.lower(),
    }

    # The question: resume AT the last entry of turn 1. Fact two came later.
    anchor = None
    for e in store.append_log[:mark][::-1]:
        if e["uuid"]:
            anchor = e["uuid"]
            break
    out["anchor_uuid"] = anchor

    if anchor is None:
        out["UUID_MODE"] = "no_uuids"
        out["verdict"] = "No entry carried a uuid; the driver cannot address entries."
    else:
        try:
            trimmed, _ = await ask(
                "Name the animal. One word. If you do not know, reply exactly: UNKNOWN",
                store, resume=sid, at=anchor)
            knew = "otter" in trimmed.lower()
            out["resumed_at_anchor"] = {"reply": trimmed[:120], "still_knew_later_fact": knew}
            if not out["control"]["knew_animal"]:
                out["UUID_MODE"] = "inconclusive"
                out["verdict"] = "Control failed: the animal was not known even on a full resume."
            elif knew:
                out["UUID_MODE"] = "not_truncating"
                out["verdict"] = ("resume_session_at accepted the uuid but did NOT truncate "
                                  "the view. Pruning cannot be done by re-anchoring; the "
                                  "store must drop entries from load() instead.")
            else:
                out["UUID_MODE"] = "addressable"
                out["verdict"] = ("resume_session_at truncated the model's view at our uuid. "
                                  "Entry uuids from append() are usable resume addresses, so "
                                  "pruning the last turn is a re-anchor.")
        except Exception as exc:
            out["resumed_at_anchor"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            out["UUID_MODE"] = "rejected"
            out["verdict"] = ("resume_session_at rejected a uuid taken from append(). "
                              "Pruning must be done by editing what load() returns.")

    print(json.dumps(out, indent=1, ensure_ascii=False))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
