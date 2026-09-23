"""Decorating a tool must not change the signature its schema was derived from.

schemas/tools.json describes each tool's signature and docstring; if a wrapper
changes the signature, the schema quietly stops describing the real tool and
the model is told the wrong arguments.

inspect.signature follows __wrapped__ -- the attribute functools.wraps sets,
and the only reason @effectful can wrap a tool without invalidating its schema.
"""
import inspect
import json
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = json.loads((ROOT / "schemas" / "tools.json").read_text(encoding="utf-8"))


async def sample(username: str, text: str, count: int = 1) -> dict:
    """Send a thing.

    Args:
        username: Who to send it to.
        text: What to say.
        count: How many times.
    """
    return {"ok": True}


def test_wraps_preserves_the_signature_a_schema_is_read_from():
    def decorate(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
        return wrapper

    assert inspect.signature(decorate(sample)) == inspect.signature(sample)
    assert decorate(sample).__doc__ == sample.__doc__
    assert decorate(sample).__name__ == sample.__name__


def test_a_wrapper_without_wraps_would_break_it():
    """The failure mode this test exists to catch, demonstrated."""
    def bad_decorate(fn):
        async def wrapper(*args, **kwargs):   # no @wraps
            return await fn(*args, **kwargs)
        return wrapper

    broken = inspect.signature(bad_decorate(sample))
    assert broken != inspect.signature(sample)
    # It collapses to (*args, **kwargs), which describes every tool and none.
    assert list(broken.parameters) == ["args", "kwargs"]


def test_the_real_decorated_tools_kept_their_signatures():
    """Not a demonstration -- the tools @effectful actually wraps."""
    from app.effects import registry
    from app.runtime import toolserver

    for name in registry.registered():
        fn = toolserver.resolve(name)
        assert fn is not None, name
        assert getattr(fn, "__wrapped__", None) is not None, (
            "%s is decorated but did not keep __wrapped__" % name)
        params = set(inspect.signature(fn).parameters)
        assert params and params != {"args", "kwargs"}, name
        entry = toolserver.load_schemas()["tools"].get(name)
        assert entry is not None, "%s is decorated but has no schema anywhere" % name
        schema = entry["parameters_json_schema"] or {}
        for declared in schema.get("properties", {}):
            assert declared in params, (
                "%s: the schema promises an argument %r the function does not take"
                % (name, declared))


def test_the_paid_tools_have_schemas():
    tools = SCHEMAS["tools"]
    for name in ("start_coding_job", "run_claude_code"):
        assert name in tools, name
        assert tools[name]["parameters_json_schema"], name
