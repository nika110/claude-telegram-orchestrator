"""The SessionStore the SDK actually calls.

A thin adapter, on purpose. It translates the SDK's SessionKey into this
system's own key -- (platform, user_id, subpath) -- and then gets out of the
way. All the judgement lives in app.memory.store and app.memory.window.

Two things worth knowing about the protocol, both measured rather than read:

  * SessionKey is a TypedDict, i.e. a plain dict. getattr() on it silently
    returns None, which is how the first run of probe P1 recorded every key as
    (None, None, ()). Always subscript it.
  * load() is NOT a mirror despite the protocol docstring calling it one.
    Probe P1 censored an entry out of load() and the model stopped knowing the
    fact, so what this returns IS the conversation the model reads back.

Only append() and load() are required; delete() is optional and the SDK probes
for it by duck-typing rather than isinstance, so this need not subclass
anything.
"""

import logging

from app.memory import store, window

logger = logging.getLogger("orchestrator.sdk_store")


class ConversationStore:
    """Duck-typed claude_agent_sdk.SessionStore over app.memory.store."""

    def __init__(self, windowed: bool = True):
        # Turned off in tests and in the migration, where the whole chain is
        # wanted rather than the slice a live turn should read.
        self.windowed = windowed

    # ------------------------------------------------------------- key mapping

    @staticmethod
    def _parts(key):
        session_id = key["session_id"]
        subpath = key.get("subpath") or ""
        relationship = store.relationship_of(session_id)
        if relationship is None:
            # The driver registers the mapping before it ever calls the SDK, so
            # this means a session nobody claimed. Keep the entries under a
            # synthetic relationship rather than dropping them on the floor.
            logger.warning("Unclaimed SDK session %s; filing under __orphan__", session_id)
            return ("__orphan__", session_id, subpath)
        return (relationship[0], relationship[1], subpath)

    # ---------------------------------------------------------------- required

    async def append(self, key, entries) -> None:
        platform, user_id, subpath = self._parts(key)
        store.append_entries(platform, user_id, list(entries), subpath=subpath)

    async def load(self, key):
        platform, user_id, subpath = self._parts(key)
        entries = store.load_entries(platform, user_id, subpath=subpath)
        if not entries:
            return None

        if not self.windowed:
            return entries

        sliced = window.build(entries)
        problems = window.validate(sliced)
        if problems:
            # A window that would be rejected as input is worse than a long one.
            logger.warning("Window for %s/%s was illegal (%s); sending the full chain",
                           platform, user_id, problems[0])
            return entries
        return sliced or None

    # ---------------------------------------------------------------- optional

    async def delete(self, key) -> None:
        platform, user_id, _ = self._parts(key)
        store.reset(platform, user_id)

    async def list_subkeys(self, key):
        platform, user_id, _ = self._parts(
            {"session_id": key["session_id"], "subpath": ""})
        return store.subpaths(platform, user_id)
