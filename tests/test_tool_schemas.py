"""schemas/tools.json is what the model is told about each tool. Guard it.

A schema that promises an argument the function does not take fails the call;
a function argument the schema forgets is one the model can never pass. Both
are silent at runtime, so they are checked here.
"""
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas" / "tools.json"

EXPECTED = {
    "start_coding_job", "check_coding_job", "list_coding_jobs", "cancel_coding_job",
    "run_claude_code", "run_shell_command", "read_file", "write_file",
    "replace_in_file", "list_directory",
}


@pytest.fixture(scope="module")
def tools():
    return json.loads(SCHEMAS.read_text(encoding="utf-8"))["tools"]


def test_the_expected_tools_are_described(tools):
    assert set(tools) == EXPECTED


def test_every_described_tool_resolves():
    from app.runtime import toolserver

    built, missing = toolserver.build_tools()
    assert missing == [], "described but no callable: %s" % missing
    assert {name for name, _, _, _ in built} == EXPECTED


def test_schemas_match_the_signatures(tools):
    from app.runtime import toolserver

    for name, entry in tools.items():
        fn = toolserver.resolve(name)
        params = inspect.signature(fn).parameters
        schema = entry["parameters_json_schema"] or {"properties": {}}
        declared = set(schema.get("properties", {}))
        assert declared == set(params), (
            "%s: schema says %s, function takes %s" % (name, sorted(declared), sorted(params)))
        required = set(schema.get("required", []))
        no_default = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert required == no_default, name


def test_descriptions_are_the_docstrings(tools):
    """One source of truth for what a tool does: its docstring."""
    from app.runtime import toolserver

    for name, entry in tools.items():
        assert entry["description"] == inspect.getdoc(toolserver.resolve(name)), name


def test_every_schema_builds_into_a_claude_tool(tools):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    async def noop(args):
        return {"content": [{"type": "text", "text": "ok"}]}

    built = [
        tool(name, entry["description"][:1000] or name,
             entry["parameters_json_schema"] or {"type": "object", "properties": {}})(noop)
        for name, entry in tools.items()
    ]
    assert create_sdk_mcp_server(name="orchestrator", version="1.0.0", tools=built)
