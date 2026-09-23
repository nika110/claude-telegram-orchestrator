"""Keeping secret values out of the model's context, and therefore out of the log.

An agent that can read .env will, sooner or later, paste a password or a bot
token into a shell command or a systemd unit -- at which point it is sitting in
plain text in the session database, in the Telegram chat, and in anything
derived from either. It will even announce "I will not display the sensitive
information here" in the same turn it displays it.

The durable fix is not a rule about being careful. It is that the value never
reaches the model at all: reads of a secrets file come back with the values
replaced by their own variable names. That leaves the agent knowing exactly
which secrets exist and what they are called -- which is all it needs, because
the correct way to give a service a credential is EnvironmentFile= or an env
var reference, never a copied literal.
"""

import os
import re

from dotenv import dotenv_values

from app.runtime.paths import ORCHESTRATOR_ROOT

ORCHESTRATOR_ENV = os.path.join(ORCHESTRATOR_ROOT, ".env")

# Substring match on the variable name. Deliberately broad: a false positive
# costs a redacted value the agent can reference by name anyway, while a false
# negative is permanent.
SECRET_HINTS = ("PASSWORD", "PASSWD", "TOKEN", "SECRET", "APIKEY", "API_KEY", "_KEY", "CREDENTIAL")

# Short values would match far too much unrelated text.
MIN_SECRET_LENGTH = 8

_cache: dict | None = None


def secret_values() -> dict:
    """{value: NAME} for every secret-looking variable in the orchestrator env."""
    global _cache
    if _cache is None:
        _cache = {}
        try:
            values = dotenv_values(ORCHESTRATOR_ENV)
        except OSError:
            values = {}
        for name, value in {**values, **os.environ}.items():
            if not value or len(value) < MIN_SECRET_LENGTH:
                continue
            if any(hint in name.upper() for hint in SECRET_HINTS):
                _cache[value] = name
    return _cache


def redact(text: str) -> str:
    """Replace any known secret value with a reference to its variable name."""
    if not text:
        return text
    for value, name in secret_values().items():
        if value in text:
            text = text.replace(value, f"${{{name}}}")
    return text


def redact_all(payload):
    """Redact recursively through whatever a tool is about to return."""
    if isinstance(payload, str):
        return redact(payload)
    if isinstance(payload, dict):
        return {k: redact_all(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_all(v) for v in payload]
    return payload


def looks_like_secrets_file(path: str) -> bool:
    return bool(re.search(r"(^|/)\.env(\.|$)|credentials|\.pem$|id_rsa", path or ""))
