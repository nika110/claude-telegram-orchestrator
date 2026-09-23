"""The in-process tool server. Every tool, from schemas/tools.json."""
import contextvars
import json
import os

import pytest

from app.runtime import toolserver


def test_every_tool_resolves():
    built, missing = toolserver.build_tools()
    assert missing == [], missing
    assert len(built) == len(toolserver.load_schemas()["tools"])


def test_the_server_builds_with_every_tool():
    """The whole bridge in one assertion."""
    server = toolserver.make_server()
    assert server is not None


def test_zero_arg_tools_get_an_empty_object_schema():
    built, _ = toolserver.build_tools()
    by_name = {b[0]: b for b in built}
    # list_coding_jobs takes no arguments, so it has no schema object at all.
    _, _, schema, _ = by_name["list_coding_jobs"]
    assert schema == toolserver.EMPTY_SCHEMA


async def test_a_tool_runs_in_process_and_sees_contextvars():
    """The contextvars carrying 'who am I answering' must survive.

    Out of process, current_person() degrades to ('telegram','unknown') and a
    coding job's result has no chat to go back to.
    """
    marker = contextvars.ContextVar("marker", default=None)

    async def fake_tool(text: str) -> dict:
        """Echo. Args: text: what."""
        return {"ok": True, "saw": marker.get(), "text": text}

    from claude_agent_sdk import tool
    wrapped = toolserver._wrap(
        tool, "fake_tool", "Echo.",
        {"type": "object", "properties": {"text": {"type": "string"}}}, fake_tool)

    marker.set("telegram:100000001")
    out = await wrapped.handler({"text": "hi"})
    payload = json.loads(out["content"][0]["text"])
    assert payload["saw"] == "telegram:100000001"
    assert payload["text"] == "hi"


async def test_a_raising_tool_returns_an_error_instead_of_killing_the_turn():
    async def explodes(x: str) -> dict:
        """Boom. Args: x: what."""
        raise RuntimeError("network died")

    from claude_agent_sdk import tool
    wrapped = toolserver._wrap(tool, "explodes", "Boom.",
                               {"type": "object", "properties": {"x": {"type": "string"}}},
                               explodes)
    out = await wrapped.handler({"x": "a"})
    payload = json.loads(out["content"][0]["text"])
    assert payload["ok"] is False
    assert "network died" in payload["error"]


async def test_a_sync_tool_is_supported():
    """The handler must not await a plain dict."""
    def plain(x: str) -> dict:
        """Sync. Args: x: what."""
        return {"ok": True, "x": x}

    from claude_agent_sdk import tool
    wrapped = toolserver._wrap(tool, "plain", "Sync.",
                               {"type": "object", "properties": {"x": {"type": "string"}}},
                               plain)
    out = await wrapped.handler({"x": "a"})
    assert json.loads(out["content"][0]["text"]) == {"ok": True, "x": "a"}


# ------------------------------------------------------------------ isolation

class Opts:
    def __init__(self, **kw):
        self.setting_sources = kw.get("setting_sources")
        self.tools = kw.get("tools")
        self.allowed_tools = kw.get("allowed_tools")


def test_isolation_refuses_without_setting_sources_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text("{}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        toolserver.assert_isolation(Opts(setting_sources=None, tools=[], allowed_tools=["x"]))
    assert "setting_sources" in str(exc.value)


def test_isolation_refuses_without_tools_empty(tmp_path, monkeypatch):
    """tools=[] is load-bearing: without it the CLI defers MCP tools behind a
    search step and emits a reference the driver must not mistake for a result."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text("{}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        toolserver.assert_isolation(Opts(setting_sources=[], tools=None, allowed_tools=["x"]))
    assert "tools must be []" in str(exc.value)


def test_isolation_refuses_a_config_dir_without_credentials(tmp_path, monkeypatch):
    """Probe P1: pinning CLAUDE_CONFIG_DIR moves where credentials are read."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        toolserver.assert_isolation(Opts(setting_sources=[], tools=[], allowed_tools=["x"]))
    assert "not logged in" in str(exc.value).lower()


def test_isolation_refuses_an_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text("{}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    with pytest.raises(RuntimeError) as exc:
        toolserver.assert_isolation(Opts(setting_sources=[], tools=[], allowed_tools=["x"]))
    assert "subscription" in str(exc.value)


def test_isolation_passes_when_everything_is_right(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text("{}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    toolserver.assert_isolation(Opts(setting_sources=[], tools=[], allowed_tools=["read_file"]))
