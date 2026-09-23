# claude-telegram-orchestrator

Talk to Claude from Telegram, and have it run Claude Code jobs on your server.

You send a message. A Claude agent (the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python),
running on your Claude subscription) answers it. When the request is real work,
it starts a headless Claude Code job in the background, tells you the job id,
and stays free to chat. When the job finishes, the result lands in the chat.

```
you (Telegram) ──> bot ──> Claude agent ──┬─> start_coding_job ──> claude -p  (background)
                   ▲                      ├─> run_shell_command / read_file / write_file ...
                   │                      └─> reply
                   └──────── job result pushed back when it finishes ──────────┘
```

## What it does

- **Chat with Claude** in one Telegram conversation that survives restarts
  (SQLite-backed session store; `/clear` wipes it).
- **Background Claude Code jobs.** `start_coding_job` runs `claude -p` in a
  project directory and reports back on its own. Check, list or cancel jobs from
  the chat. Two run at once; a third waits up to 30 seconds for a slot, then
  is refused rather than queued forever.
- **`/claude <prompt>`** sends a prompt straight to a Claude Code job, skipping
  the agent.
- **Quick work without Claude Code:** shell commands and file read/write/edit
  tools, so a one-line question does not cost a coding session and a usage limit
  does not block everything.
- **Files:** send a document or photo and it is saved to `data/incoming/`. The
  agent is told the path.
- **It can change itself safely.** A job that edits this bot queues a restart
  instead of killing its own service. The restart waits until no job is running
  and every result has been delivered.
- **Never twice.** Every job start is recorded in an effect ledger before it
  runs. If the process dies mid-turn, the interrupted call is closed from the
  ledger on resume and never re-run (`app/effects/`, `app/runtime/hooks.py`).
- **One user only.** Only the Telegram user id in `ALLOWED_TELEGRAM_USER_ID` is
  answered. Everyone else is ignored and logged.

## Setup

You need Linux, Python 3.12, a Telegram bot token from [@BotFather](https://t.me/BotFather),
and the `claude` CLI installed and logged in on the box (`claude` once,
interactively, as the user the service will run as).

```bash
git clone https://github.com/nika110/claude-telegram-orchestrator.git
cd claude-telegram-orchestrator
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN and ALLOWED_TELEGRAM_USER_ID
./venv/bin/python main.py
```

Your Telegram user id: message [@userinfobot](https://t.me/userinfobot).

It runs on the subscription login of the `claude` CLI, not an API key. On first
run the bot copies `~/.claude/.credentials.json` into its own pinned
`data/claude-config/` and renews that token itself before it expires. If
`ANTHROPIC_API_KEY` is set, it is removed from the bot's environment so nothing
bills per token by accident.

### As a service

`deploy/personal-agent.service` is a systemd unit. Edit the user and paths, then:

```bash
sudo cp deploy/personal-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now personal-agent
```

For self-restart after a job changes the bot, the service user needs
passwordless sudo for exactly that one command:

```
# /etc/sudoers.d/personal-agent
ubuntu ALL=(root) NOPASSWD: /usr/bin/systemctl restart personal-agent.service
```

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | | required |
| `ALLOWED_TELEGRAM_USER_ID` | | required; the only user answered |
| `ORCHESTRATOR_MODEL` | `claude-sonnet-5` | model for the chat agent |
| `ORCHESTRATOR_MAX_TURNS` | `60` | tool-use ceiling per message |
| `ORCHESTRATOR_WORKSPACE` | `./deployments` | where new projects are created |
| `CLAUDE_COMMAND_MODEL` | `opus high effort` | model for `/claude` jobs |
| `CLAUDE_COMMAND_DIR` | repo root | working directory for `/claude` jobs |
| `ORCHESTRATOR_SERVICE` | `personal-agent.service` | unit restarted after self-edits |

Coding jobs default to Opus; the agent can pass any model the CLI accepts,
optionally with an effort level (`"sonnet"`, `"opus high effort"`).

## Commands

| | |
| --- | --- |
| `/start` | help |
| `/claude <prompt>` | start a Claude Code job directly |
| `/jobs` | recent jobs and their status |
| `/clear` | forget this conversation |

## Security

Read this before running it. The agent can run shell commands and edit files as
the service user, and Claude Code jobs run with `--permission-mode
bypassPermissions`. That is what makes it useful from a phone, and it means:

- Only `ALLOWED_TELEGRAM_USER_ID` can talk to it. Keep your bot token secret.
- Anything the agent reads (web pages, files, command output) can carry a
  prompt injection. The system prompt tells it to treat all of that as data and
  to report injection attempts, but that is a mitigation, not a sandbox.
- Run it as an unprivileged user on a machine you are comfortable handing to
  it. Don't grant it broader sudo than the restart line above.
- Reads of `.env` files are redacted (`app/tools/secrets.py`), so the model sees
  `${VAR_NAME}` rather than secret values.

## Layout

```
main.py                       entry point
app/channels/telegram_channel.py   Telegram handlers, background result delivery
app/runtime/driver.py         one turn: SDK options, resume, OAuth renewal, repair
app/runtime/toolserver.py     in-process MCP server built from schemas/tools.json
app/runtime/prompt.py         the system prompt
app/tools/claude_code_tool.py background Claude Code jobs
app/tools/shell_tool.py, file_tools.py   direct tools
app/memory/                   SQLite conversation store + context window
app/effects/                  effect ledger ("never twice")
scripts/probes/               the SDK measurements the store design rests on
```

## Tests

```bash
./venv/bin/python -m pytest tests/ -q
```

## License

MIT
