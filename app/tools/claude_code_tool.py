"""Headless Claude Code, run as background jobs.

A Claude Code run takes minutes to tens of minutes. Running one *inside* an
agent turn blocks the entire conversation: the turn stays open, the
per-person turn lock in app.channels.base then answers every later message with
"I'm still working on your previous message", and even /clear is refused. With
a Semaphore(1) and no bound on how many runs could queue behind it, several
30-minute runs stacked up into hours of a completely unusable bot.

So the normal path is start_coding_job: it returns a job id immediately and
pushes the result into the chat when the run finishes. run_claude_code is kept
for the one case that genuinely needs the answer inline, but with a much
shorter timeout and no unbounded queueing.
"""

import asyncio
import json
import logging
import os
import time
import uuid

from app.effects.registry import effectful
from app.channels.progress import current_person, notify
from app.runtime.paths import CODING_WORKSPACE_ROOT, DATA_DIR, ORCHESTRATOR_ROOT

logger = logging.getLogger("orchestrator.claude_code")

JOBS_PATH = os.path.join(DATA_DIR, "coding_jobs.json")

# ORCHESTRATOR_ROOT is this repo. Editing anything under it needs the service
# restarted to take effect, which is the single most dangerous thing a job can
# do to itself -- see RESTART_PREAMBLE below.

# 2 vCPU / 3.8GB with no swap: two concurrent `claude -p` subprocesses is the
# realistic ceiling before this box starts thrashing.
_semaphore = asyncio.Semaphore(2)

# How long to wait for a slot before giving up. Waiting forever is what turned
# a slow run into a multi-hour outage, so this fails fast and says so instead.
SLOT_WAIT_SECONDS = 30

FOREGROUND_MAX_SECONDS = 900
BACKGROUND_DEFAULT_SECONDS = 3600

# Job records live in memory; the file is only so status survives a restart --
# which happens routinely here, since the agent restarts its own service after
# changing its own code.
_jobs: dict[str, dict] = {}
_tasks: dict[str, asyncio.Task] = {}

LIMIT_HINT = (
    "Claude Code has hit its usage limit, so it cannot write code right now. "
    "This does NOT mean you are stuck: run_shell_command still works and can "
    "read files, run commands, edit with sed/python, restart services and "
    "inspect the database. Use it, and never tell the owner to SSH in and run "
    "commands themselves -- you have the same shell access they do."
)


# Prepended to every job that runs against this repo. A job here is a child
# process inside personal-agent.service's cgroup, so `systemctl restart` on that
# unit SIGTERMs the job itself (exit 143), its siblings, and the task that was
# about to send its result to the owner. Five jobs died that way in one day, and
# the owner just saw silence -- so the instruction has to be explicit, and the
# reason has to be stated or the next run will "helpfully" restart anyway.
RESTART_PREAMBLE = (
    "[Environment note, read before you plan]\n"
    "You are running as a child process of personal-agent.service, the very "
    "service this repository implements. NEVER run `systemctl restart "
    "personal-agent.service` (or stop/kill it) -- systemd kills the whole "
    "cgroup, which kills YOU mid-run, kills any sibling job, and destroys the "
    "message that would have delivered your result. That is not a theoretical "
    "risk: it is the single most common way work here is lost.\n"
    "When your changes need the service restarted to take effect, request it "
    "instead and finish normally:\n"
    "    python3 -c \"import sys; sys.path.insert(0, '"
    f"{ORCHESTRATOR_ROOT}'); from app.tools.restart_guard import "
    "request_restart; print(request_restart('what you changed'))\"\n"
    "The orchestrator then restarts itself as soon as no job is in flight and "
    "your result has reached the owner. Say in your final answer that you "
    "queued a restart.\n\n"
)


def _works_on_this_repo(working_dir: str) -> bool:
    """Whether a job is editing the orchestrator itself, not a project it builds.

    Projects live under CODING_WORKSPACE_ROOT, which by default sits inside this
    repo, so being under ORCHESTRATOR_ROOT alone does not decide it.
    """
    here = os.path.realpath(working_dir)
    root = os.path.realpath(ORCHESTRATOR_ROOT)
    workspace = os.path.realpath(CODING_WORKSPACE_ROOT)
    inside = here == root or here.startswith(root + os.sep)
    in_workspace = here == workspace or here.startswith(workspace + os.sep)
    return inside and not in_workspace


def _looks_like_usage_limit(*texts: str) -> bool:
    blob = " ".join(t.lower() for t in texts if t)
    return any(
        phrase in blob
        for phrase in ("usage limit", "limit reached", "weekly limit", "rate limit exceeded")
    )


def _save_jobs() -> None:
    try:
        os.makedirs(os.path.dirname(JOBS_PATH), exist_ok=True)
        with open(JOBS_PATH, "w") as f:
            json.dump(_jobs, f, indent=2)
    except OSError:
        logger.warning("Could not persist coding job registry", exc_info=True)


def _load_jobs() -> None:
    """Restore job records at startup, marking anything still 'running' as lost.

    A job's subprocess does not survive the restart, so leaving it as running
    would have the agent wait forever on a job that no longer exists.
    """
    global _jobs
    if not os.path.exists(JOBS_PATH):
        return
    try:
        with open(JOBS_PATH) as f:
            _jobs = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Unreadable coding job registry -- starting empty", exc_info=True)
        _jobs = {}
        return
    for job in _jobs.values():
        was_running = job.get("status") == "running"
        if was_running:
            job["status"] = "interrupted"
            job["error"] = "the service restarted while this job was running"
            job["finished"] = time.time()
        # Anything already on disk from before reports were tracked is treated
        # as delivered, so switching this on doesn't replay the whole backlog
        # into the owner's chat. A job the restart just killed is the one real
        # exception: nothing has ever told them about it.
        if was_running:
            job["reported"] = False
        else:
            job.setdefault("reported", True)
    _save_jobs()


# A standalone importer -- deployments/<job>/run_daily.py wants
# _invoke_claude and nothing else -- must not run the recovery pass above. It
# rewrites coding_jobs.json from a second process, which would stomp on the
# live service's in-flight records and could hand the owner a spurious
# "cut off by a restart" report for a job that is still running fine.
if os.environ.get("ORCHESTRATOR_SKIP_JOB_REGISTRY") != "1":
    _load_jobs()


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# What `claude -p` runs when the caller names no model. Left empty, the CLI
# picked whatever its own config defaulted to, which is not necessarily the
# strongest model available -- and every job started from here is real work on
# his machines. The owner asked on 2026-08-21 for Claude to start on Opus, so
# it is set here, at the one place that builds the argv, rather than in each of
# the tools that can start a job.
#
# A caller naming a model still wins: "sonnet", "claude-fable-5" and
# "opus high effort" all pass through untouched.
DEFAULT_MODEL = "opus"


def _split_model_spec(model: str) -> tuple[str, str]:
    """Split a spec like "opus high effort" into ("opus", "high").

    The CLI wants the model and the reasoning effort as two separate flags
    (--model / --effort), but callers name them as one phrase. Handing
    "opus high effort" straight to --model is rejected by the CLI and the whole
    run dies before it starts, so pull the effort word out here instead.

    Anything that is not a known effort level is left alone as the model name,
    so plain values ("opus", "claude-fable-5") still pass through untouched.
    """
    name_parts, effort = [], ""
    for token in model.split():
        lowered = token.lower()
        if lowered == "effort":
            continue
        if lowered in EFFORT_LEVELS:
            effort = lowered
            continue
        name_parts.append(token)
    return " ".join(name_parts), effort


def _summary(job: dict, include_result: bool = True) -> dict:
    out = {
        "job_id": job["job_id"],
        "status": job["status"],
        "working_dir": job["working_dir"],
        "task": job["task"][:200],
        "elapsed_seconds": int((job.get("finished") or time.time()) - job["started"]),
    }
    for key in ("error", "session_id", "total_cost_usd"):
        if job.get(key):
            out[key] = job[key]
    if include_result and job.get("result"):
        out["result"] = job["result"][:4000]
        if len(job["result"]) > 4000:
            out["result_truncated"] = (
                "Cut off at 4000 characters. Do not continue any list or "
                "reconstruct the rest from memory -- read the file instead."
            )
    return out


async def _invoke_claude(
    prompt: str, working_dir: str, allowed_tools: str, resume_session_id: str, timeout_seconds: int,
    model: str = "",
) -> dict:
    """Run one `claude -p` to completion. Never raises."""
    model_name, effort = _split_model_spec(model)
    # Applied after the split, so "high effort" on its own still means Opus at
    # high effort rather than an empty --model flag.
    model_name = model_name or DEFAULT_MODEL
    if _works_on_this_repo(working_dir):
        prompt = RESTART_PREAMBLE + prompt

    os.makedirs(working_dir, exist_ok=True)
    argv = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--allowedTools", allowed_tools,
        # Explicit on argv rather than left to settings.json: driver pins
        # CLAUDE_CONFIG_DIR to data/claude-config, which has no settings of
        # its own, so ~/.claude/settings.json is never read here and a job
        # would silently fall back to "default" -- a permission prompt in a
        # headless job is a hang until the timeout, not a safeguard.
        "--permission-mode", "bypassPermissions",
    ]
    if model_name:
        argv += ["--model", model_name]
    if effort:
        argv += ["--effort", effort]
    if resume_session_id:
        argv += ["--resume", resume_session_id]

    # Deliberately does NOT log the prompt text: prompts routinely carry
    # secrets (an .env edit, an API key) and the journal is readable by root
    # and persisted indefinitely.
    logger.info(
        "claude -p start: working_dir=%s allowed_tools=%s resume=%s model=%s "
        "effort=%s prompt_chars=%d",
        working_dir, allowed_tools, resume_session_id or None,
        model_name or None, effort or None, len(prompt),
    )

    try:
        await asyncio.wait_for(_semaphore.acquire(), SLOT_WAIT_SECONDS)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "result": "",
            "error": (
                "all Claude Code slots are busy. Check what is already running "
                "with list_coding_jobs rather than starting another."
            ),
        }

    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=working_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.exception("Could not start claude CLI")
            return {"ok": False, "result": "", "error": f"could not start claude: {exc}"}

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("claude -p timed out after %ss in %s", timeout_seconds, working_dir)
            return {"ok": False, "result": "", "error": f"timed out after {timeout_seconds}s"}
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
    finally:
        _semaphore.release()

    out_text = stdout.decode(errors="replace")
    err_text = stderr.decode(errors="replace")

    try:
        payload = json.loads(out_text)
    except json.JSONDecodeError:
        logger.error("claude -p returned non-JSON output: %s", out_text[:2000])
        error = "non-JSON output from claude -p"
        if _looks_like_usage_limit(out_text, err_text):
            error = LIMIT_HINT
        return {
            "ok": False, "result": "", "error": error,
            "stdout_tail": out_text[-2000:], "stderr_tail": err_text[-2000:],
        }

    result_text = payload.get("result", "")
    ok = proc.returncode == 0 and not payload.get("is_error", False)
    outcome = {
        "ok": ok,
        "result": result_text,
        "session_id": payload.get("session_id"),
        "total_cost_usd": payload.get("total_cost_usd"),
        "returncode": proc.returncode,
    }
    if not ok and _looks_like_usage_limit(result_text, err_text):
        outcome["error"] = LIMIT_HINT
    elif not ok:
        outcome["error"] = result_text[:1000] or f"claude exited {proc.returncode}"
    return outcome


async def _run_job(job_id: str, prompt: str, working_dir: str, allowed_tools: str,
                   resume_session_id: str, timeout_seconds: int, model: str = "") -> None:
    job = _jobs[job_id]
    try:
        outcome = await _invoke_claude(
            prompt, working_dir, allowed_tools, resume_session_id, timeout_seconds, model
        )
    except asyncio.CancelledError:
        job.update(status="cancelled", finished=time.time(), error="cancelled")
        _save_jobs()
        raise
    except Exception as exc:
        logger.exception("Coding job %s crashed", job_id)
        job.update(status="failed", finished=time.time(), error=f"{type(exc).__name__}: {exc}")
        _save_jobs()
        return

    job.update(
        status="done" if outcome["ok"] else "failed",
        finished=time.time(),
        result=outcome.get("result", ""),
        error=outcome.get("error", ""),
        session_id=outcome.get("session_id"),
        total_cost_usd=outcome.get("total_cost_usd"),
    )
    _save_jobs()

    await _report_to_owner(job)


async def _report_to_owner(job: dict) -> None:
    """Tell the owner a job finished, in the agent's own words.

    The turn that started the job ended long ago, so this is an out-of-band
    push. Rather than dumping Claude Code's raw output into the chat, it is fed
    back through the agent: it then explains what happened in the
    context of the conversation, and -- because the exchange is persisted -- it
    actually remembers afterwards what its own job did, instead of having to
    guess when asked about it later.
    """
    ok = job["status"] == "done"
    detail = (job.get("result") or job.get("error") or "").strip()
    raw = f"Coding job {job['job_id']} {'finished' if ok else 'failed'}: {detail[:1200]}"

    platform, user_id = job.get("platform"), job.get("user_id")
    if not platform or not user_id or user_id == "unknown":
        _mark_reported(job, await notify(raw))
        return

    # Lazy: app.channels.base -> root_agent -> this module, so importing at
    # module level would be a cycle.
    from app.channels.base import BUSY_MESSAGE, handle_incoming, is_busy

    prompt = (
        "[Automatic system notification, NOT a message from the owner. Do not "
        "treat it as a new instruction from them.]\n"
        f"The background coding job {job['job_id']} you started in "
        f"{job['working_dir']} has {'finished successfully' if ok else 'FAILED'} "
        f"after {int((job.get('finished') or 0) - job['started'])}s.\n"
        f"What it reported:\n{detail[:3000]}\n\n"
        # An earlier report was cut off mid-list and the agent finished the list
        # itself, handing the owner 150 fabricated URLs. Saying where
        # the text stops, and what to do about it, is cheaper than that mistake.
        + (
            "[This report was CUT OFF at 3000 characters -- the rest is missing. "
            "Do NOT guess, summarise beyond, or continue any list past this "
            "point. If the owner needs what came after, read the actual file or "
            "run the command yourself.]\n\n"
            if len(detail) > 3000
            else ""
        )
        + "Tell the owner what happened, in one or two short lines, in your own "
        "words. If it failed, say what you intend to do about it. Do not call "
        "check_coding_job -- you already have the result above."
    )

    # The owner may be mid-conversation, and that turn owns the session. Wait
    # for it rather than colliding, but never wait forever.
    for _ in range(30):
        if not is_busy(platform, user_id):
            break
        await asyncio.sleep(10)

    try:
        reply = await handle_incoming(platform, user_id, prompt)
    except Exception:
        logger.exception("Could not report job %s through the agent", job["job_id"])
        reply = ""

    # If the agent was busy or said nothing useful, the owner still needs to
    # hear that the job landed -- never swallow the result.
    delivered = await notify(raw if (not reply.strip() or reply == BUSY_MESSAGE) else reply)
    # Only now is the job genuinely finished. If this never ran -- the service
    # was restarted out from under it, which is what kept happening -- the
    # record stays unreported and startup redelivers it.
    _mark_reported(job, delivered)


def _mark_reported(job: dict, delivered: bool) -> None:
    job["reported"] = bool(delivered)
    if not delivered:
        logger.warning(
            "Job %s produced a result but nothing could deliver it; queued for "
            "redelivery at startup", job["job_id"],
        )
    _save_jobs()


# Long enough that a report the normal path is about to deliver is not raced
# for, short enough that a lost one arrives while it still matters.
REPORT_SWEEP_SECONDS = 180


async def run_report_sweeper(send) -> None:
    """Re-deliver any job result that never reached the owner, forever."""
    while True:
        await asyncio.sleep(REPORT_SWEEP_SECONDS)
        try:
            delivered = await deliver_pending_reports(send, after_restart=False)
            if delivered:
                logger.info(
                    "Swept up %d coding job report(s) that never went out", delivered
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Coding job report sweep failed")


def running_job_ids() -> list[str]:
    """Jobs a restart would kill right now."""
    return [j["job_id"] for j in _jobs.values() if j.get("status") == "running"]


def undelivered_report_ids() -> list[str]:
    """Finished jobs whose result has not reached the owner yet."""
    return [
        j["job_id"]
        for j in _jobs.values()
        if j.get("status") != "running" and not j.get("reported", True)
    ]


async def deliver_pending_reports(send, after_restart: bool = True) -> int:
    """Push out every job result that was never delivered, then mark it sent.

    Called at startup, and then on a timer -- see run_report_sweeper. Startup
    alone was not enough. A restart is only one of the ways delivery dies: the
    reporting turn itself can fail, and when it does, this record sits
    unreported for as long as the service happens to stay up -- one finished
    job sat there for hours with the owner never told, because nothing was
    going to look at it again until the next restart.
    """
    pending = [j for j in _jobs.values() if j.get("status") != "running"
               and not j.get("reported", True)]
    # A job with no requester came from a test or an internal caller, not from
    # him asking for something, so there is nobody waiting on the result.
    # Marked delivered rather than skipped, or the sweeper below would offer
    # them again every three minutes forever.
    orphans = [j for j in pending if str(j.get("user_id") or "") in ("", "unknown")]
    for job in orphans:
        job["reported"] = True
    if orphans:
        _save_jobs()
        logger.info("Marked %d ownerless coding job report(s) as delivered", len(orphans))
    pending = [j for j in pending if j not in orphans]
    pending.sort(key=lambda j: j.get("finished") or j.get("started") or 0)
    for job in pending:
        detail = (job.get("result") or job.get("error") or "").strip()
        header = {
            "done": f"Coding job {job['job_id']} finished",
            "interrupted": f"Coding job {job['job_id']} was cut off by a restart",
            "cancelled": f"Coding job {job['job_id']} was cancelled",
        }.get(job.get("status", ""), f"Coding job {job['job_id']} failed")
        # The reason matters to him: "the service restarted" is a fact he can
        # check, and saying it when nothing restarted is how a diagnostic
        # message becomes a misleading one.
        why = (
            "you never got this -- it was lost when the service restarted"
            if after_restart else
            "this is late -- the message carrying it failed the first time"
        )
        message = (
            f"{header} ({why}).\nTask: {(job.get('task') or '')[:300]}\n\n"
            f"{detail[:2500]}"
        )
        try:
            await send(message)
        except Exception:
            logger.exception("Could not redeliver report for job %s", job["job_id"])
            continue
        job["reported"] = True
    if pending:
        _save_jobs()
        logger.info("Redelivered %d undelivered coding job report(s)", len(pending))
    return len(pending)


@effectful(kind="spend",
           fingerprint=lambda a, k: "claude|%s|%s|%s" % (
               k.get("working_dir", a[1] if len(a) > 1 else ""),
               k.get("resume_session_id", ""),
               (k.get("prompt", a[0] if a else ""))[:400]))
async def start_coding_job(
    prompt: str,
    working_dir: str,
    allowed_tools: str = "Bash,Read,Edit,Write,Glob,Grep",
    resume_session_id: str = "",
    timeout_seconds: int = BACKGROUND_DEFAULT_SECONDS,
    model: str = "",
) -> dict:
    """Start a Claude Code session in the background and return a job id at once.

    This is the normal way to get code written. It does NOT block the
    conversation: you get a job id back immediately, you stay able to answer
    other questions, and the owner is sent the outcome automatically when the
    job finishes. Tell them the job id and that you'll report back -- then
    actually answer their next message instead of waiting.

    Use check_coding_job with the id if they ask how it's going. Never start a
    second job for the same task because the first one hasn't answered yet.

    Args:
        prompt: Full, self-contained task instructions for Claude Code.
        working_dir: Absolute path to treat as the project root.
        allowed_tools: Comma-separated Claude Code tool names to allow.
        resume_session_id: If set, continues that prior Claude Code session in
            the same working_dir instead of starting fresh.
        timeout_seconds: Give up on the run after this long.
        model: Which model to run, optionally with a reasoning effort level --
            "opus", "sonnet", "claude-fable-5", "opus high effort". Leave empty
            for Opus, which is what these jobs start on unless told otherwise.
    """
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    # Captured now, while the owner's turn is still on the stack: by the time
    # the job finishes there is no conversation context left to ask.
    platform, user_id = current_person()
    _jobs[job_id] = {
        "job_id": job_id,
        "platform": platform,
        "user_id": user_id,
        "status": "running",
        "task": prompt,
        "working_dir": working_dir,
        "model": model,
        "started": time.time(),
        "finished": None,
        "result": "",
        "error": "",
    }
    _save_jobs()

    _tasks[job_id] = asyncio.create_task(
        _run_job(
            job_id, prompt, working_dir, allowed_tools, resume_session_id, timeout_seconds,
            model,
        )
    )
    logger.info("Started coding job %s in %s (model=%s)",
                job_id, working_dir, model or "default")
    return {
        "ok": True,
        "job_id": job_id,
        "status": "running",
        "note": (
            "Running in the background. You are free to answer other messages "
            "now; the owner gets the result automatically when it lands."
        ),
    }


async def check_coding_job(job_id: str) -> dict:
    """Check how a background coding job is doing.

    Report what this actually returns. If the status is still "running", say it
    is still running and roughly how long it has been going -- do not guess at
    what it is doing internally, and do not claim it finished.

    Args:
        job_id: The id returned by start_coding_job.
    """
    job = _jobs.get(job_id)
    if not job:
        known = ", ".join(sorted(_jobs)) or "none"
        return {"ok": False, "error": f"no coding job {job_id!r} (known jobs: {known})"}
    return {"ok": True, **_summary(job)}


async def list_coding_jobs() -> dict:
    """List recent background coding jobs and their status.

    Use this when asked what is running, or before starting a job that may
    already be in flight.
    """
    jobs = sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)[:20]
    running = [j["job_id"] for j in jobs if j["status"] == "running"]
    return {
        "ok": True,
        "jobs": [_summary(j, include_result=False) for j in jobs],
        "running": running,
        "running_count": len(running),
    }


async def cancel_coding_job(job_id: str) -> dict:
    """Stop a background coding job that is still running.

    Args:
        job_id: The id returned by start_coding_job.
    """
    job = _jobs.get(job_id)
    if not job:
        return {"ok": False, "error": f"no coding job {job_id!r}"}
    if job["status"] != "running":
        return {"ok": False, "error": f"job {job_id} is already {job['status']}"}
    task = _tasks.get(job_id)
    if task:
        task.cancel()
    logger.info("Cancelled coding job %s", job_id)
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


@effectful(kind="spend",
           fingerprint=lambda a, k: "claude|%s|%s|%s" % (
               k.get("working_dir", a[1] if len(a) > 1 else ""),
               k.get("resume_session_id", ""),
               (k.get("prompt", a[0] if a else ""))[:400]))
async def run_claude_code(
    prompt: str,
    working_dir: str,
    allowed_tools: str = "Bash,Read,Edit,Write,Glob,Grep",
    resume_session_id: str = "",
    timeout_seconds: int = 600,
) -> dict:
    """Run a SHORT Claude Code session and wait for the result.

    Only use this when you genuinely cannot continue without the answer in this
    same turn -- it freezes the whole conversation while it runs, so the owner
    cannot even ask you a question until it finishes. For anything that might
    take more than a few minutes, use start_coding_job instead.

    Args:
        prompt: Full, self-contained task instructions for Claude Code.
        working_dir: Absolute path to treat as the project root.
        allowed_tools: Comma-separated Claude Code tool names to allow.
        resume_session_id: If set, continues that prior Claude Code session.
        timeout_seconds: Max seconds to wait, capped at 900.
    """
    timeout_seconds = min(timeout_seconds, FOREGROUND_MAX_SECONDS)
    await notify(
        f"Working on it - running Claude Code in {os.path.basename(working_dir) or working_dir}."
    )
    return await _invoke_claude(
        prompt, working_dir, allowed_tools, resume_session_id, timeout_seconds
    )
