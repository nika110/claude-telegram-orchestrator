"""Deferred self-restart, so applying a code change doesn't kill the work that made it.

The orchestrator runs as personal-agent.service, and every Claude Code job runs
as a child process *inside that service's cgroup*. systemd's default
KillMode=control-group means `systemctl restart personal-agent.service` SIGTERMs
the entire cgroup -- so a coding job whose last step was "restart the service to
apply my changes" killed:

  * itself, mid-run (`claude exited 143` = 128 + SIGTERM),
  * every other coding job running at the time,
  * and the orchestrator task that was in the middle of pushing the finished
    job's result into Telegram ("Task was destroyed but it is pending!").

The owner saw a job start and then silence: five jobs in one day died this
way, including one that had already produced a 6.6k-character answer nobody
ever received.

So nothing restarts this service synchronously any more. A restart is *requested*
by dropping a flag file; the watcher below performs it only once no coding job is
running and every finished job's result has actually been delivered. The flag
lives on disk rather than in memory so a request survives the very restart it
asks for.
"""

import asyncio
import json
import logging
import os
import time
from typing import Awaitable, Callable

from app.runtime.paths import DATA_DIR

logger = logging.getLogger("orchestrator.restart")

FLAG_PATH = os.path.join(DATA_DIR, "restart_requested.json")
AGENT_UNIT = os.environ.get("ORCHESTRATOR_SERVICE", "personal-agent.service")

# How often the watcher re-checks whether it is safe to restart. Short enough
# that a code change lands promptly, long enough to be free.
POLL_SECONDS = 15

# A restart request that can never be satisfied would leave the agent running
# stale code forever, so past this point it goes ahead regardless of what is
# still in flight and says so.
MAX_DEFER_SECONDS = 30 * 60


def request_restart(reason: str = "", platform: str = "", user_id: str = "") -> dict:
    """Ask for the agent to restart itself as soon as it is safe to do so.

    Returns immediately -- the restart happens later, from the watcher, once no
    coding job would be killed by it.
    """
    payload = {
        "reason": reason or "code change",
        "requested_at": time.time(),
        "platform": platform,
        "user_id": user_id,
    }
    try:
        os.makedirs(os.path.dirname(FLAG_PATH), exist_ok=True)
        with open(FLAG_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        logger.warning("Could not record restart request", exc_info=True)
        return {"ok": False, "error": f"could not record restart request: {exc}"}
    logger.info("Restart requested: %s", payload["reason"])
    return {
        "ok": True,
        "restart": "deferred",
        "note": (
            "Restart queued. It runs automatically once no Claude Code job is "
            "in flight and every finished job's result has been delivered -- "
            "usually within seconds. Do NOT run `systemctl restart "
            f"{AGENT_UNIT}` yourself: that kills the running jobs and the "
            "pending result messages along with the service."
        ),
    }


def pending() -> dict | None:
    """The outstanding restart request, if there is one."""
    if not os.path.exists(FLAG_PATH):
        return None
    try:
        with open(FLAG_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # An unreadable flag still means somebody asked for a restart; losing
        # the reason is better than ignoring the request.
        logger.warning("Unreadable restart flag -- treating as a bare request", exc_info=True)
        return {"reason": "unknown", "requested_at": time.time()}


def clear() -> None:
    try:
        os.remove(FLAG_PATH)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not clear the restart flag", exc_info=True)


def _blockers() -> list[str]:
    """What would be destroyed by restarting right now."""
    # Lazy: claude_code_tool imports progress -> channels, so a module-level
    # import here would be a cycle through whichever channel started the job.
    from app.tools.claude_code_tool import running_job_ids, undelivered_report_ids

    blockers = []
    running = running_job_ids()
    if running:
        blockers.append(f"{len(running)} coding job(s) still running: {', '.join(running)}")
    undelivered = undelivered_report_ids()
    if undelivered:
        blockers.append(
            f"{len(undelivered)} finished job result(s) not yet delivered: "
            f"{', '.join(undelivered)}"
        )
    return blockers


async def _do_restart() -> None:
    proc = await asyncio.create_subprocess_exec(
        "sudo", "systemctl", "restart", AGENT_UNIT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    # Normally unreachable: the restart tears down this process first.
    if proc.returncode != 0:
        logger.error("Restart failed (%s): %s", proc.returncode, stderr.decode()[-500:])


async def run_restart_watcher(send: Callable[[str], Awaitable[None]]) -> None:
    """Perform any queued restart at the first moment it destroys nothing.

    `send` delivers a line to the owner's chat, so the restart is never silent
    -- an agent that goes away for two seconds with no explanation looks
    exactly like an agent that crashed.
    """
    logger.info("Restart watcher started")
    while True:
        try:
            await asyncio.sleep(POLL_SECONDS)
            request = pending()
            if not request:
                continue

            blockers = _blockers()
            waited = time.time() - float(request.get("requested_at") or 0)
            if blockers and waited < MAX_DEFER_SECONDS:
                logger.info("Restart deferred: %s", "; ".join(blockers))
                continue

            reason = request.get("reason") or "code change"
            # Cleared *before* restarting: if the new process came up and found
            # the flag still set it would restart again, forever.
            clear()

            if blockers:
                logger.warning(
                    "Restarting after %.0fs despite: %s", waited, "; ".join(blockers)
                )
                await send(
                    f"Restarting now to apply: {reason}. I waited "
                    f"{int(waited / 60)} minutes and this is still in flight, so "
                    "it will be cut off: " + "; ".join(blockers)
                )
            else:
                await send(f"Restarting to apply: {reason}. Back in a couple of seconds.")

            logger.info("Restarting %s (reason: %s)", AGENT_UNIT, reason)
            await _do_restart()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A watcher that dies on one bad iteration silently stops applying
            # every future code change, which is far worse than a noisy log.
            logger.exception("Restart watcher iteration failed")
