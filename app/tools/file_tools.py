"""Reading and writing files directly, without Claude Code.

Claude Code is a subscription with a usage limit, and when that limit is hit
the whole system used to stop dead -- it was the only way any agent could put
code on disk. These tools remove that single point of failure: an agent with a
shell and these can still write a script, patch a config, or fix a bug on its
own. Shell heredocs could do it in theory, but quoting real source code through
/bin/sh is exactly the kind of thing that fails silently and corrupts a file.
"""

import logging
import os
import shutil
import time

from app.tools.secrets import looks_like_secrets_file, redact

logger = logging.getLogger("orchestrator.files")

MAX_READ_BYTES = 200_000
BACKUP_SUFFIX = ".bak"


async def read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> dict:
    """Read a text file and return its contents.

    Args:
        path: Absolute path to the file.
        max_bytes: Stop after this many bytes; the result says if it was cut.
    """
    if not os.path.isfile(path):
        return {"ok": False, "error": f"no such file: {path}"}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes + 1)
    except OSError as exc:
        return {"ok": False, "error": f"could not read {path}: {exc}"}

    truncated = len(content) > max_bytes
    result = {
        "ok": True,
        "path": path,
        # Secret values come back as ${THEIR_NAME}. The agent still learns which
        # credentials exist and what they are called, which is all it needs to
        # wire one up properly (EnvironmentFile=, an env var reference); the
        # literal never enters the conversation, so it can never be echoed into
        # a command, a unit file, or the permanent session log.
        "content": redact(content[:max_bytes]),
        "truncated": truncated,
        "lines": content.count("\n") + 1,
    }
    if looks_like_secrets_file(path):
        result["note"] = (
            "Secret values in this file are shown as ${VARIABLE_NAME}. That is "
            "deliberate and is not an error -- pass the credential by reference "
            "(EnvironmentFile= in a systemd unit, or load the .env in the "
            "program itself). Never try to obtain the literal another way."
        )
    return result


async def write_file(path: str, content: str, overwrite: bool = True) -> dict:
    """Write a text file, creating parent directories as needed.

    Use this to create scripts, config files or source code without Claude
    Code. An existing file is copied to <path>.bak first so a bad write is
    always recoverable.

    Args:
        path: Absolute path to write.
        content: The full file contents. This REPLACES the file; it does not
            append, so include everything the file should end up containing.
        overwrite: If False, refuse when the file already exists.
    """
    existed = os.path.isfile(path)
    if existed and not overwrite:
        return {"ok": False, "error": f"{path} already exists and overwrite is False"}

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        backup = ""
        if existed:
            backup = path + BACKUP_SUFFIX
            shutil.copy2(path, backup)
        # Write via a temp file in the same directory, then rename: a crash
        # mid-write must never leave a half-written source file behind.
        tmp = f"{path}.tmp-{int(time.time())}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except OSError as exc:
        logger.exception("Could not write %s", path)
        return {"ok": False, "error": f"could not write {path}: {exc}"}

    logger.info("Wrote %s (%d bytes, replaced=%s)", path, len(content), existed)
    return {"ok": True, "path": path, "bytes": len(content), "replaced": existed,
            "backup": backup}


async def replace_in_file(path: str, find: str, replace: str, count: int = 1) -> dict:
    """Replace exact text inside an existing file.

    Prefer this over rewriting a whole file for a small edit -- it cannot
    accidentally drop the parts you didn't mean to touch. Fails rather than
    guessing if `find` does not appear, or appears more often than `count`
    allows.

    Args:
        path: Absolute path to the file.
        find: The exact text to look for, including indentation.
        replace: What to put in its place.
        count: How many occurrences to replace. Use 0 to replace all.
    """
    if not os.path.isfile(path):
        return {"ok": False, "error": f"no such file: {path}"}
    try:
        with open(path, encoding="utf-8") as f:
            original = f.read()
    except OSError as exc:
        return {"ok": False, "error": f"could not read {path}: {exc}"}

    found = original.count(find)
    if found == 0:
        return {"ok": False, "error": f"text not found in {path} -- read it first and match exactly"}
    if count and found > count:
        return {
            "ok": False,
            "error": (
                f"{found} occurrences found but count={count}; make `find` more "
                "specific or raise count"
            ),
        }

    updated = original.replace(find, replace) if count == 0 else original.replace(find, replace, count)
    result = await write_file(path, updated)
    if not result["ok"]:
        return result
    return {"ok": True, "path": path, "replacements": found if count == 0 else min(found, count),
            "backup": result.get("backup", "")}


async def list_directory(path: str) -> dict:
    """List the entries in a directory, marking which are directories.

    Args:
        path: Absolute path to the directory.
    """
    if not os.path.isdir(path):
        return {"ok": False, "error": f"not a directory: {path}"}
    try:
        entries = sorted(os.listdir(path))
    except OSError as exc:
        return {"ok": False, "error": f"could not list {path}: {exc}"}

    items = []
    for name in entries[:500]:
        full = os.path.join(path, name)
        items.append({
            "name": name,
            "type": "dir" if os.path.isdir(full) else "file",
            "bytes": os.path.getsize(full) if os.path.isfile(full) else None,
        })
    return {"ok": True, "path": path, "entries": items, "count": len(entries)}
