"""The app's authoritative conversation store.

P1 measured that load() IS what the SDK materializes a resume from -- censoring
it removed a fact from the model's view. So this is not a mirror of the CLI's
transcript; it is the conversation, and what load() returns is what the model
reads back.

Three tables, because they answer three different questions:

  relationships  one row per (platform, user_id). Claude wants a real session
                 id per conversation, and this is where that mapping lives.
  sdk_entries    the verbatim SDK entries, in order. This is prompt fuel: it
                 goes back to the SDK untouched, because an entry chain with a
                 rewritten middle is not a chain.
  messages       a projection with real columns -- author, text, tool name --
                 so reading a conversation back is a simple query instead of
                 JSON-parsing every row.

Real DELETEs, not tombstones: /clear and forget_turn_since have to actually
remove things, and the owner is told they do.
"""

import json
import os
import sqlite3
import threading
import time
import uuid as uuidlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_PATH = os.path.join(ROOT, "data", "conversations.db")

_conn = None
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS relationships (
    platform        TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    sdk_session_id  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (platform, user_id)
);
CREATE TABLE IF NOT EXISTS sdk_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    platform   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    subpath    TEXT NOT NULL DEFAULT '',
    uuid       TEXT,
    entry      TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_rel
    ON sdk_entries (platform, user_id, subpath, id);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    platform   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    at         TEXT NOT NULL,
    author     TEXT NOT NULL,
    kind       TEXT NOT NULL,          -- text | call | result
    text       TEXT,
    tool_name  TEXT,
    entry_id   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_rel ON messages (platform, user_id, id);
"""


def _db():
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
                conn = sqlite3.connect(STORE_PATH, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("pragma journal_mode=WAL")
                conn.executescript(SCHEMA)
                conn.commit()
                _conn = conn
    return _conn


def _reset_for_tests():
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


# ---------------------------------------------------------------- relationships

def session_id_for(platform: str, user_id: str) -> str:
    """The Claude session id for this conversation, minting one if needed."""
    db = _db()
    with _lock:
        row = db.execute(
            "select sdk_session_id from relationships where platform=? and user_id=?",
            (platform, user_id)).fetchone()
        if row:
            return row["sdk_session_id"]
        sid = str(uuidlib.uuid4())
        now = time.time()
        db.execute(
            "insert into relationships (platform,user_id,sdk_session_id,created_at,updated_at)"
            " values (?,?,?,?,?)", (platform, user_id, sid, now, now))
        db.commit()
        return sid


def relationship_of(session_id: str):
    """(platform, user_id) for an SDK session id, or None."""
    row = _db().execute(
        "select platform, user_id from relationships where sdk_session_id=?",
        (session_id,)).fetchone()
    return (row["platform"], row["user_id"]) if row else None


def conversations() -> list:
    """One row per relationship, newest activity first.

    """
    rows = _db().execute(
        "select m.platform, m.user_id, count(*) n, max(m.at) last "
        "from messages m group by m.platform, m.user_id order by last desc").fetchall()
    return [{"platform": r["platform"], "user_id": r["user_id"],
             "messages": r["n"], "last": r["last"]} for r in rows]


# ------------------------------------------------------------------- entries

def append_entries(platform: str, user_id: str, entries: list, subpath: str = "") -> list:
    """Store SDK entries verbatim and project the readable ones.

    Returns the rowids, so the caller knows every entry's identity without
    reading back and racing its own writer.
    """
    db = _db()
    now = time.time()
    ids = []
    with _lock:
        for entry in entries:
            cur = db.execute(
                "insert into sdk_entries (platform,user_id,subpath,uuid,entry,created_at)"
                " values (?,?,?,?,?,?)",
                (platform, user_id, subpath, entry.get("uuid"),
                 json.dumps(entry, ensure_ascii=False, default=str), now))
            ids.append(int(cur.lastrowid))
            for row in _project(entry, subpath):
                db.execute(
                    "insert into messages (platform,user_id,at,author,kind,text,tool_name,entry_id)"
                    " values (?,?,?,?,?,?,?,?)",
                    (platform, user_id, row["at"], row["author"], row["kind"],
                     row.get("text"), row.get("tool_name"), ids[-1]))
        db.execute("update relationships set updated_at=? where platform=? and user_id=?",
                   (now, platform, user_id))
        db.commit()
    return ids


def _project(entry: dict, subpath: str = "") -> list:
    """Readable rows for one SDK entry. Unknown entry types project to nothing.

    P4 recorded the entry vocabulary: user, assistant, attachment, mode,
    queue-operation, ai-title, last-prompt, atis-latch. Only the first two carry
    conversation; the rest are bookkeeping and would be noise in a transcript.
    """
    kind = entry.get("type")
    if kind not in ("user", "assistant"):
        return []

    # A nested sub-agent run is attributed to the agent, not to the owner.
    author = kind
    if subpath.startswith("subagents/"):
        author = "%s:%s" % (kind, subpath.split("/", 1)[1])

    at = str(entry.get("timestamp") or "")[:19]
    message = entry.get("message") or {}
    content = message.get("content")
    rows = []

    if isinstance(content, str):
        if content.strip():
            rows.append({"at": at, "author": author, "kind": "text", "text": content})
        return rows

    for block in (content or []):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and (block.get("text") or "").strip():
            rows.append({"at": at, "author": author, "kind": "text",
                         "text": block["text"]})
        elif btype == "tool_use":
            rows.append({"at": at, "author": author, "kind": "call",
                         "tool_name": block.get("name"),
                         "text": json.dumps(block.get("input"), ensure_ascii=False)[:600]})
        elif btype == "tool_result":
            body = block.get("content")
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False, default=str)
            rows.append({"at": at, "author": author, "kind": "result",
                         "tool_name": block.get("tool_use_id"), "text": body[:600]})
    return rows


def load_entries(platform: str, user_id: str, subpath: str = "") -> list:
    """Every stored entry for this conversation, in order."""
    rows = _db().execute(
        "select entry from sdk_entries where platform=? and user_id=? and subpath=?"
        " order by id", (platform, user_id, subpath)).fetchall()
    out = []
    for r in rows:
        try:
            out.append(json.loads(r["entry"]))
        except json.JSONDecodeError:
            continue
    return out


def subpaths(platform: str, user_id: str) -> list:
    rows = _db().execute(
        "select distinct subpath from sdk_entries where platform=? and user_id=?"
        " and subpath <> '' order by subpath", (platform, user_id)).fetchall()
    return [r["subpath"] for r in rows]


# -------------------------------------------------------------------- reading

def history(platform: str, user_id: str = "", limit: int = 40,
            include_tools: bool = False) -> list:
    """Newest last, shaped like history_tools._render already returns."""
    where, params = ["platform = ?"], [platform]
    if user_id:
        where.append("user_id = ?")
        params.append(user_id)
    if not include_tools:
        where.append("kind = 'text'")
    params.append(int(limit))
    rows = _db().execute(
        "select at, author, kind, text, tool_name from messages where %s"
        " order by id desc limit ?" % " and ".join(where), params).fetchall()

    out = []
    for r in reversed(rows):
        item = {"at": r["at"], "from": r["author"]}
        if r["kind"] == "text":
            item["text"] = (r["text"] or "")[:1500]
        elif r["kind"] == "call":
            item["called"] = r["tool_name"]
            item["args"] = r["text"]
        else:
            item["result_of"] = r["tool_name"]
            item["result"] = r["text"]
        out.append(item)
    return out


# ------------------------------------------------------------------- deleting

def forget_turn_since(platform: str, user_id: str) -> int:
    """Drop the trailing turn: everything after the last user entry.

    Used by auto-reply so a turn that decided to stay silent leaves no trace to
    react to next time. Deleting a SUFFIX is what keeps the entry chain valid --
    the remaining entries still form an unbroken chain, so the cached session
    survives a prune. Removing from the middle would not.
    """
    db = _db()
    with _lock:
        rows = db.execute(
            "select id, entry from sdk_entries where platform=? and user_id=? and subpath=''"
            " order by id", (platform, user_id)).fetchall()
        cut = None
        for r in rows:
            try:
                if json.loads(r["entry"]).get("type") == "user":
                    cut = r["id"]
            except json.JSONDecodeError:
                continue
        if cut is None:
            return 0
        removed = db.execute(
            "delete from sdk_entries where platform=? and user_id=? and subpath='' and id >= ?",
            (platform, user_id, cut)).rowcount
        db.execute("delete from messages where platform=? and user_id=? and entry_id >= ?",
                   (platform, user_id, cut))
        db.commit()
    return removed


def mark(platform: str, user_id: str, subpath: str = "") -> int:
    """Where this conversation ends right now. Pass it to forget_after()."""
    row = _db().execute(
        "select max(id) m from sdk_entries where platform=? and user_id=? and subpath=?",
        (platform, user_id, subpath)).fetchone()
    return int(row["m"] or 0)


def forget_after(platform: str, user_id: str, since: int, subpath: str = "") -> int:
    """Drop everything written after `since`. Returns how many entries went.

    A suffix again, so the remaining entries still form an unbroken chain and
    the cached session survives the prune. Auto-reply uses this when a turn
    delivered nothing: the notes the model writes to itself explaining a
    silence are what it reads back as the house style for that thread, so a
    turn nobody heard must leave no trace to imitate.
    """
    db = _db()
    with _lock:
        removed = db.execute(
            "delete from sdk_entries where platform=? and user_id=? and subpath=? and id > ?",
            (platform, user_id, subpath, since)).rowcount
        if not subpath:
            db.execute("delete from messages where platform=? and user_id=? and entry_id > ?",
                       (platform, user_id, since))
        db.commit()
    return removed


def reset(platform: str, user_id: str) -> dict:
    """/clear: really remove the conversation, including nested transcripts."""
    db = _db()
    with _lock:
        entries = db.execute(
            "delete from sdk_entries where platform=? and user_id=?",
            (platform, user_id)).rowcount
        msgs = db.execute(
            "delete from messages where platform=? and user_id=?",
            (platform, user_id)).rowcount
        db.execute("delete from relationships where platform=? and user_id=?",
                   (platform, user_id))
        db.commit()
    return {"entries": entries, "messages": msgs}
