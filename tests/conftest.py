"""Keep the suite off the live service's state.

app.tools.claude_code_tool reconciles the job registry at import: anything
still marked running is assumed to have died with the process, rewritten as
"the service restarted while this job was running", and queued for a report to
the owner -- against the real data/coding_jobs.json of the checkout the tests
run in. The module has an escape hatch for a second process importing it; the
suite is one.

Without this, one `pytest tests/` on the live box marked two genuinely-running
jobs as cut off by a restart that never happened. Set before any app import,
which is what conftest gets right and a fixture would not.
"""
import os

os.environ["ORCHESTRATOR_SKIP_JOB_REGISTRY"] = "1"
