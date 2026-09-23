"""A durable record of every side-effectful thing the agent does.

The agent starts paid Claude Code runs. A turn that fails partway is re-driven,
and that is only safe if the effects it already had are known -- so the
guarantee is a property of this system, not of whatever framework runs the
turn.

Two phases, because the dangerous window is between them:

  * record_intent() commits BEFORE the wire call. If the process dies after
    this and before the outcome, the effect MAY have shipped. That row is
    "dangling" -- it is never treated as completed (its result is unknown) and
    never silently re-run (it may already have reached a person).
  * record_outcome() completes the row once the call returns.

check() only ever matches a row that COMPLETED SUCCESSFULLY. A failed call had
no effect, so blocking its retry would turn a transient error into lost work.

The @effectful decorator records intent and outcome from inside each tool body;
the SDK's PreToolUse hook (app/runtime/hooks.py) enforces "never twice".
"""

import contextlib
import contextvars
import json
import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger("orchestrator.effects")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_PATH = os.path.join(ROOT, "data", "effects.db")

# How far back an identical effect still counts as the same effect. Long enough
# to cover a retried turn and a resumed one; short enough that deliberately
# sending the same short message again an hour later is not blocked.
WINDOW_SECONDS = 15 * 60

_turn_id: contextvars.ContextVar = contextvars.ContextVar("effect_turn_id", default="-")

_conn = None
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS effects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      TEXT    NOT NULL,
    tool         TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    fingerprint  TEXT    NOT NULL,
    state        TEXT    NOT NULL,          -- intent | done | failed
    result       TEXT,
    created_at   REAL    NOT NULL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_effects_lookup
    ON effects (tool, fingerprint, state, completed_at);
CREATE INDEX IF NOT EXISTS idx_effects_state ON effects (state, kind);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
                conn = sqlite3.connect(LEDGER_PATH, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # WAL: the tools run in a thread pool and a reader may be a
                # different thread mid-write.
                conn.execute("pragma journal_mode=WAL")
                conn.executescript(SCHEMA)
                conn.commit()
                _conn = conn
    return _conn


def _reset_for_tests() -> None:
    """Drop the cached connection so LEDGER_PATH can be repointed."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


@contextlib.contextmanager
def current_turn(turn_id: str):
    """Tag every effect recorded inside this block with one turn id."""
    token = _turn_id.set(turn_id or "-")
    try:
        yield
    finally:
        _turn_id.reset(token)


def record_intent(tool: str, kind: str, fingerprint: str) -> int:
    """Commit, before the wire call, that this is about to happen."""
    db = _db()
    with _lock:
        cursor = db.execute(
            "insert into effects (turn_id, tool, kind, fingerprint, state, created_at) "
            "values (?,?,?,?,'intent',?)",
            (_turn_id.get(), tool, kind, fingerprint, time.time()),
        )
        db.commit()
    return int(cursor.lastrowid)


def record_outcome(rowid: int, result: dict, ok: bool) -> None:
    """Complete the row the wire call was made under."""
    db = _db()
    with _lock:
        db.execute(
            "update effects set state=?, result=?, completed_at=? where id=?",
            (
                "done" if ok else "failed",
                json.dumps(result, ensure_ascii=False, default=str)[:8000],
                time.time(),
                rowid,
            ),
        )
        db.commit()


def check(tool: str, kind: str, fingerprint: str):
    """The result of an identical effect that already completed, or None.

    Only `done` rows match. A `failed` row means nobody received anything, and
    an `intent` row means we do not know -- neither may be handed back as if it
    were a successful result.
    """
    db = _db()
    row = db.execute(
        "select * from effects where tool=? and fingerprint=? and state='done' "
        "and completed_at >= ? order by completed_at desc limit 1",
        (tool, fingerprint, time.time() - WINDOW_SECONDS),
    ).fetchone()
    if row is None:
        return None
    try:
        result = json.loads(row["result"] or "null")
    except json.JSONDecodeError:
        result = None
    return {
        "turn_id": row["turn_id"],
        "tool": row["tool"],
        "kind": row["kind"],
        "fingerprint": row["fingerprint"],
        "result": result,
        "completed_at": row["completed_at"],
    }


def dangling(kind: str = None) -> list:
    """Intents with no outcome: they MAY have shipped. Never auto-replayed."""
    db = _db()
    if kind:
        rows = db.execute(
            "select * from effects where state='intent' and kind=? order by created_at",
            (kind,),
        ).fetchall()
    else:
        rows = db.execute(
            "select * from effects where state='intent' order by created_at"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "turn_id": r["turn_id"],
            "tool": r["tool"],
            "kind": r["kind"],
            "fingerprint": r["fingerprint"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
