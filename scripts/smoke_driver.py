"""A real turn through the driver: does it answer, use a tool, and remember?

Deliberately read-only and on a throwaway relationship, so it cannot touch a
real conversation and cannot message anyone.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

from app.memory import store
from app.runtime import driver

PLATFORM, USER = "smoketest", "w7"


async def main():
    store.reset(PLATFORM, USER)

    print("\n===== TURN 1: does a tool actually run in-process? =====")
    reply = await driver.run_turn(
        PLATFORM, USER,
        "Run the shell command `uname -r` and reply with ONLY its output. "
        "Do not message anyone.")
    print("REPLY 1:", reply)

    entries = store.load_entries(PLATFORM, USER)
    kinds = []
    for entry in entries:
        for block in (entry.get("message") or {}).get("content") or []:
            if isinstance(block, dict):
                kinds.append(block.get("type"))
    print("ENTRIES:", len(entries), "BLOCKS:", kinds)
    assert any(k == "tool_use" for k in kinds), "no tool ran -- the server is not wired"

    print("\n===== TURN 2: does it resume and remember? =====")
    reply2 = await driver.run_turn(
        PLATFORM, USER,
        "Without running anything, what kernel version did you just report? "
        "Reply with only the version.")
    print("REPLY 2:", reply2)

    print("\n===== state =====")
    print("entries now:", len(store.load_entries(PLATFORM, USER)))
    print("session id :", store.session_id_for(PLATFORM, USER))
    print("dangling   :", len(__import__("app.effects.ledger", fromlist=["x"]).dangling()))

    store.reset(PLATFORM, USER)
    print("cleaned up.")


asyncio.run(main())
