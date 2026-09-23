import asyncio
import logging
import os
import signal

from app.runtime.paths import ORCHESTRATOR_ROOT
from app.tools.secrets import redact

logger = logging.getLogger("orchestrator.shell")

DEFAULT_WORKING_DIR = ORCHESTRATOR_ROOT
DEFAULT_TIMEOUT_SECONDS = 60


async def run_shell_command(
    command: str,
    working_dir: str = DEFAULT_WORKING_DIR,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Run a simple shell command directly and return its raw output.

    Use this for quick, simple operations -- listing files, reading a file,
    checking disk/process/service status, simple filesystem operations like
    creating or removing a directory. It just runs one command and returns
    the output, with no reasoning about code.

    For anything that involves actually writing or modifying source code,
    use start_coding_job instead -- it drives a real Claude Code session, not
    a bare shell.

    Args:
        command: The shell command to run (executed via /bin/sh -c).
        working_dir: Absolute path to run the command from.
        timeout_seconds: Give up if the command runs longer than this. Keep
            it short -- this tool is for quick commands, and a long-running
            or interactive one would otherwise block the conversation.
    """
    if not os.path.isdir(working_dir):
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"working_dir does not exist: {working_dir}",
        }

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Never inherit stdin: a command that reads it (cat, a prompt for
            # input) would otherwise wait forever for input that never comes.
            stdin=asyncio.subprocess.DEVNULL,
            # Own process group so a timeout can kill the whole tree, not just
            # the /bin/sh wrapper (which would orphan its children).
            start_new_session=True,
        )
    except OSError as exc:
        logger.exception("Could not start command %r", command)
        return {"returncode": -1, "stdout": "", "stderr": f"could not start command: {exc}"}

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_seconds)
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await proc.wait()
        logger.warning("Command timed out after %ss: %r", timeout_seconds, command)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": (
                f"command timed out after {timeout_seconds}s and was killed. "
                "If it needs to run longer, start it as a background service "
                "instead of waiting on it here."
            ),
        }

    # Redacted on the way out, so `cat .env` teaches the agent which secrets
    # exist without ever putting their values into the conversation -- and
    # therefore never into the permanent session log.
    return {
        "returncode": proc.returncode,
        "stdout": redact(stdout.decode(errors="replace")[-4000:]),
        "stderr": redact(stderr.decode(errors="replace")[-4000:]),
    }
