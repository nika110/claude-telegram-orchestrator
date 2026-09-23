"""How a coding job is turned into a `claude -p` command line."""
import asyncio
import json
import os

import pytest

from app.tools import claude_code_tool as cc


def test_a_model_phrase_splits_into_model_and_effort():
    assert cc._split_model_spec("opus high effort") == ("opus", "high")
    assert cc._split_model_spec("sonnet") == ("sonnet", "")
    assert cc._split_model_spec("max") == ("", "max")


def test_a_job_on_this_repo_is_recognised(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    workspace = root / "deployments"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(cc, "ORCHESTRATOR_ROOT", str(root))
    monkeypatch.setattr(cc, "CODING_WORKSPACE_ROOT", str(workspace))

    assert cc._works_on_this_repo(str(root)) is True
    assert cc._works_on_this_repo(str(root / "app")) is True
    # A project the bot builds lives inside the repo but is not the repo.
    assert cc._works_on_this_repo(str(workspace / "demo")) is False
    assert cc._works_on_this_repo(str(tmp_path / "elsewhere")) is False


class FakeProc:
    returncode = 0

    async def communicate(self, input=None):
        return json.dumps({"result": "done", "session_id": "s1"}).encode(), b""


@pytest.fixture
def captured(monkeypatch, tmp_path):
    seen = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        seen["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(cc, "ORCHESTRATOR_ROOT", str(tmp_path / "repo"))
    monkeypatch.setattr(cc, "CODING_WORKSPACE_ROOT", str(tmp_path / "repo" / "deployments"))
    return seen


async def test_the_command_line_is_headless_and_explicit(captured, tmp_path):
    out = await cc._invoke_claude("build it", str(tmp_path / "proj"), "Bash,Read", "", 30,
                                  "sonnet high")
    argv = captured["argv"]
    assert out["ok"] is True and out["result"] == "done"
    assert argv[:3] == ["claude", "-p", "build it"]
    # A permission prompt in a headless job is a hang until the timeout.
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "high"
    assert captured["cwd"] == str(tmp_path / "proj")
    assert os.path.isdir(tmp_path / "proj")


async def test_no_model_named_means_the_default(captured, tmp_path):
    await cc._invoke_claude("x", str(tmp_path / "p"), "Read", "", 30)
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == cc.DEFAULT_MODEL


async def test_a_job_on_the_bot_itself_is_told_not_to_restart_it(captured, tmp_path):
    await cc._invoke_claude("change the prompt", str(tmp_path / "repo"), "Read", "", 30)
    prompt = captured["argv"][2]
    assert prompt.startswith(cc.RESTART_PREAMBLE)
    assert prompt.endswith("change the prompt")


async def test_a_project_job_gets_the_prompt_untouched(captured, tmp_path):
    await cc._invoke_claude("build it", str(tmp_path / "repo" / "deployments" / "demo"),
                            "Read", "", 30)
    assert captured["argv"][2] == "build it"
