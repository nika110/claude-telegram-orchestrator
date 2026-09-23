#!/usr/bin/env python3
"""P1 — is load() authoritative on resume, or only a mirror?

The whole Phase 2 store design rests on the app owning the conversation: the
window it returns from load() is what the model actually reads back. But the
SessionStore docstring says "Adapter for mirroring session transcripts to
external storage" and "the subprocess still writes to local disk", which would
mean the local JSONL is the authority and load() is a write-only copy.

Those two readings lead to different systems, so measure it.

The experiment: teach the model a code word in turn 1, then resume with a
load() that has censored the entry containing it.

  * If the model still knows the word -> load() is a mirror. The local
    transcript is authoritative and the app cannot edit what the model sees.
    Cold-start windowing has to be done some other way.
  * If the model does not know it   -> load() IS the resume source. The app
    owns the conversation and windowing is a matter of what load() returns.

CLAUDE_CONFIG_DIR is pinned to a scratch directory so a stale local transcript
cannot answer for us either way.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="p1-")
CONFIG = os.path.join(SCRATCH, "config")
os.makedirs(CONFIG, exist_ok=True)

# Pinning CLAUDE_CONFIG_DIR isolates the transcript -- but it also moves where
# the CLI looks for credentials, so an unpopulated scratch dir fails with
# "Not logged in". Copy the OAuth credentials across (never read or printed
# here, just placed) so the probe measures storage, not auth. This is a real
# constraint for the W6 isolation assertion, not just a probe detail.
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

CODE_WORD = "BANANA"


class RecordingStore:
    """Duck-typed SessionStore. Records everything and can censor on load."""

    def __init__(self):
        self.entries = {}          # key tuple -> list[entry]
        self.load_calls = []
        self.censor = None         # optional callable(list) -> list

    @staticmethod
    def _k(key):
        # SessionKey is a TypedDict, i.e. a plain dict -- not an object with
        # attributes. getattr() silently returns None on it, which is how the
        # first run of this probe recorded every key as (None, None, ()).
        if isinstance(key, dict):
            return (key.get("project_key"), key.get("session_id"), key.get("subpath") or "")
        return (getattr(key, "project_key", None), getattr(key, "session_id", None),
                getattr(key, "subpath", None) or "")

    async def append(self, key, entries):
        self.entries.setdefault(self._k(key), []).extend(entries)

    async def load(self, key):
        kept = list(self.entries.get(self._k(key), []))
        self.load_calls.append({"key": self._k(key), "returned": len(kept)})
        if self.censor is not None:
            kept = self.censor(kept)
            self.load_calls[-1]["after_censor"] = len(kept)
        return kept or None


async def ask(prompt, store, sid=None, resume=None):
    opts = ClaudeAgentOptions(
        model="claude-haiku-4-5-20251001",
        max_turns=2,
        allowed_tools=[],
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=[],
        session_store=store,
        **({"session_id": sid} if sid else {}),
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


def contains_word(text):
    return CODE_WORD.lower() in (text or "").lower()


async def main():
    out = {"probe": "P1", "question": "is load() authoritative on resume?"}
    store = RecordingStore()

    say, sid = await ask(
        f"Remember this code word exactly: {CODE_WORD}. Reply only: STORED", store)
    out["turn1"] = {"reply": say[:120], "session_id": sid}
    out["entries_appended"] = sum(len(v) for v in store.entries.values())
    out["store_keys"] = [str(k) for k in store.entries]

    # Control: resume with the full history. The model should recall it.
    store.censor = None
    say2, _ = await ask("What was the code word? Reply with the single word only.",
                        store, resume=sid)
    out["control_uncensored"] = {"reply": say2[:120], "knew_word": contains_word(say2)}

    # The real question: censor every entry mentioning the word, then resume.
    def drop_the_word(entries):
        kept = []
        for e in entries:
            blob = json.dumps(e, default=str)
            if CODE_WORD.lower() in blob.lower():
                continue
            kept.append(e)
        return kept

    store.censor = drop_the_word
    say3, _ = await ask("What was the code word? Reply with the single word only. "
                        "If you do not know, reply exactly: UNKNOWN",
                        store, resume=sid)
    out["censored"] = {"reply": say3[:160], "still_knew_word": contains_word(say3)}

    out["load_calls"] = store.load_calls[-4:]

    if out["censored"]["still_knew_word"]:
        verdict = ("MIRROR — load() did not gate the resume; the local transcript "
                   "is authoritative. The app CANNOT edit what the model reads back.")
        mode = "mirror"
    elif not out["control_uncensored"]["knew_word"]:
        verdict = ("INCONCLUSIVE — the control failed too: the model did not recall "
                   "the word even with full history, so the censor proved nothing.")
        mode = "inconclusive"
    else:
        verdict = ("AUTHORITATIVE — censoring load() removed the fact from the "
                   "model's view. The app owns the conversation; windowing is "
                   "simply what load() returns.")
        mode = "authoritative"

    out["COLD_START_MODE"] = mode
    out["verdict"] = verdict
    print(json.dumps(out, indent=1, ensure_ascii=False))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
