"""One in-process MCP server holding every tool, built from schemas/tools.json.

The schemas are stored rather than generated at startup so that what the model
is told about each tool is reviewable in one place, and so a signature change
that forgets its schema fails a test instead of silently mis-describing the
tool (tests/test_tool_schemas.py).

IN-PROCESS is the whole point. A contextvar set before the call is visible
inside the tool, which is what keeps "who am I answering" alive: a coding job
started from a tool captures current_person() so its result can be pushed back
to the right chat later. A subprocess-per-tool design would silently degrade
that to ('telegram', 'unknown') and the result would have nowhere to go.

Zero-arg tools have no schema object, so they get an empty one here -- the one
substitution this module makes.
"""

import importlib
import inspect
import json
import logging
import os

logger = logging.getLogger("orchestrator.toolserver")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(ROOT, "schemas", "tools.json")

# Where the callables actually live. Resolution walks these in order.
TOOL_MODULES = [
    "app.tools.claude_code_tool",
    "app.tools.shell_tool",
    "app.tools.file_tools",
]

# A () -> dict function has no schema object. An MCP tool still needs one.
EMPTY_SCHEMA = {"type": "object", "properties": {}}


def load_schemas() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def resolve(name: str, modules=None):
    """The live callable for a tool name, or None."""
    for dotted in (modules or TOOL_MODULES):
        try:
            module = importlib.import_module(dotted)
        except Exception:                       # a module that cannot import is
            continue                            # not a reason to lose the rest
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def build_tools(schemas: dict = None, modules=None):
    """[(name, description, schema, callable)] for everything resolvable."""
    payload = schemas or load_schemas()
    built, missing = [], []
    for name, entry in sorted(payload["tools"].items()):
        fn = resolve(name, modules)
        if fn is None:
            missing.append(name)
            continue
        schema = entry.get("parameters_json_schema") or EMPTY_SCHEMA
        description = entry.get("description") or name
        built.append((name, description, schema, fn))
    return built, missing


def make_server(tools=None, name: str = "orchestrator"):
    """An in-process SDK MCP server exposing the tools.

    Every handler awaits the real callable in THIS process, so contextvars set
    by the channel before the turn are visible inside the tool.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    built = tools if tools is not None else build_tools()[0]
    decorated = []
    for tool_name, description, schema, fn in built:
        decorated.append(_wrap(tool, tool_name, description, schema, fn))
    return create_sdk_mcp_server(name=name, version="1.0.0", tools=decorated)


def _wrap(tool_decorator, tool_name, description, schema, fn):
    async def handler(args):
        try:
            result = fn(**(args or {}))
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.exception("Tool %s raised", tool_name)
            result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        return {"content": [{"type": "text",
                             "text": json.dumps(result, ensure_ascii=False, default=str)}]}

    handler.__name__ = tool_name
    # MCP descriptions are not the place for a 3000-character docstring, but the
    # enums live in there, so keep a generous slice rather than a summary.
    return tool_decorator(tool_name, description[:4000], schema)(handler)


def assert_isolation(options) -> None:
    """Refuse to start if the subprocess could reach tools we did not grant.

    Measured, not theoretical: the host harness's own messaging tools leaked
    into a subprocess and the model actually invoked one; allowed_tools did not
    stop it. setting_sources=[] and tools=[] are what close that, and tools=[]
    is load-bearing for a second reason -- without it the CLI defers MCP tools
    behind a search step and emits a reference the driver must not mistake for
    a real tool response.
    """
    problems = []
    if getattr(options, "setting_sources", None) != []:
        problems.append("setting_sources must be [] so no user or project settings load")
    if getattr(options, "tools", None) != []:
        problems.append("tools must be [] so the CLI advertises only our MCP tools")
    if not getattr(options, "allowed_tools", None):
        problems.append("allowed_tools must name exactly the tools this agent may use")
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        problems.append("CLAUDE_CONFIG_DIR must be pinned")
    else:
        # Probe P1: pinning the config dir also moves where the CLI reads
        # credentials, so an unpopulated one fails with "Not logged in".
        creds = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], ".credentials.json")
        if not os.path.exists(creds):
            problems.append("pinned CLAUDE_CONFIG_DIR has no .credentials.json; "
                            "copy it in or the subprocess is not logged in")
    if os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY is set; this system runs on subscription auth")

    if problems:
        raise RuntimeError("refusing to start: " + "; ".join(problems))
