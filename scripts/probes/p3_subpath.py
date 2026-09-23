#!/usr/bin/env python3
"""P3 — does a nested agent run get its own subpath in the store?

W9 replaces the five ADK handoff shims and AgentTool(coding_agent) with nested
in-process runs. The design puts each nested transcript under a subpath key the
app owns, which is what makes ledger scoping across a handoff definable and
keeps read_my_chat_history able to show all seven authors.

That only works if subpath is real. SessionKey has the field, but P4 saw only
"" across a normal conversation, so this asks directly: when the SDK runs a
sub-agent via the `agents` option, does append() arrive under a non-empty
subpath, or is the nested transcript folded into the parent's?

Either answer is usable, but they are different designs:
  * non-empty subpath -> the SDK partitions nested runs for us; W9 mirrors them.
  * always ""         -> the app must partition, by driving nested runs under a
                         session key it mints itself.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="p3-")
CONFIG = os.path.join(SCRATCH, "config")
os.makedirs(CONFIG, exist_ok=True)
_src = os.path.expanduser("~/.claude/.credentials.json")
if os.path.exists(_src):
    shutil.copy2(_src, os.path.join(CONFIG, ".credentials.json"))
os.environ["CLAUDE_CONFIG_DIR"] = CONFIG
os.environ.pop("ANTHROPIC_API_KEY", None)

from claude_agent_sdk import (  # noqa: E402
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


class Store:
    def __init__(self):
        self.appends = []
        self.entries = {}

    @staticmethod
    def _k(key):
        return (key.get("project_key"), key.get("session_id"), key.get("subpath") or "")

    async def append(self, key, entries):
        k = self._k(key)
        for e in entries:
            self.appends.append({"subpath": k[2], "session_id": k[1], "type": e.get("type")})
        self.entries.setdefault(k, []).extend(entries)

    async def load(self, key):
        return list(self.entries.get(self._k(key), [])) or None

    async def list_subkeys(self, key):
        want = (key.get("project_key"), key.get("session_id"))
        return sorted({k[2] for k in self.entries if (k[0], k[1]) == want and k[2]})


async def main():
    out = {"probe": "P3", "question": "does a nested agent run get its own subpath?"}
    store = Store()

    researcher = AgentDefinition(
        description="Answers one trivia question and nothing else.",
        prompt="You answer with a single word and stop.",
        tools=[],
        permission_mode="bypassPermissions",
        model="haiku",
    )

    opts = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=6,
        allowed_tools=["Task"],
        # NOT tools=[] here. The first run of this probe passed it and the model
        # could not reach Task at all -- it printed a <function_calls> block as
        # prose instead, so the sub-agent never ran and the probe measured
        # nothing. tools=[] suppresses the built-ins, which is right for a leaf
        # call and wrong for anything that must delegate.
        setting_sources=[],
        session_store=store,
        agents={"researcher": researcher},
    )

    text, result = [], None
    try:
        async for m in query(
            prompt="Use the Task tool to ask the `researcher` subagent what the "
                   "capital of France is, then tell me its answer.",
            options=opts,
        ):
            if isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        text.append(b.text.strip())
            elif isinstance(m, ResultMessage):
                result = m
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)

    out["reply"] = " ".join(text)[:200]
    out["session_id"] = result.session_id if result else None
    out["subpaths_seen"] = sorted({a["subpath"] for a in store.appends})
    out["session_ids_seen"] = sorted({a["session_id"] for a in store.appends if a["session_id"]})
    out["appends"] = len(store.appends)
    nested = [s for s in out["subpaths_seen"] if s]
    out["task_actually_ran"] = any(
        a["type"] in ("user", "assistant") for a in store.appends) and (
        "function_calls" not in out["reply"])
    if not out["task_actually_ran"]:
        out["SUBPATH_MODE"] = "inconclusive"
        out["verdict"] = ("The model narrated a tool call instead of making one, so "
                          "no nested run happened and nothing was measured.")
        print(json.dumps(out, indent=1, ensure_ascii=False))
        shutil.rmtree(SCRATCH, ignore_errors=True)
        return 0

    if nested:
        out["SUBPATH_MODE"] = "sdk_partitions"
        out["verdict"] = ("Nested runs arrive under a non-empty subpath (%s). The SDK "
                          "partitions them for us; W9 mirrors those entries back as "
                          "author=<agent> rows." % nested)
    elif len(out["session_ids_seen"]) > 1:
        out["SUBPATH_MODE"] = "separate_session"
        out["verdict"] = ("Nested runs got their own SESSION id rather than a subpath. "
                          "W9 must track the child session id to scope the ledger.")
    else:
        out["SUBPATH_MODE"] = "folded_in"
        out["verdict"] = ("Everything landed under one key with an empty subpath. The "
                          "SDK does not partition nested runs, so the app must: drive "
                          "each nested run under a session key it mints itself.")

    print(json.dumps(out, indent=1, ensure_ascii=False))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
