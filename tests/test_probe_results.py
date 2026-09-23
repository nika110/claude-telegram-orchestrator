"""The recorded probe results the Phase 2 store design is built on.

These are not live tests -- they pin what was MEASURED, so that if a future
SDK release changes the answer, the design constant and the recorded evidence
stop agreeing and somebody has to look.
"""
import json
from pathlib import Path

PROBES = Path(__file__).resolve().parents[1] / "scripts" / "probes"

# The decision P1 settled. The store design depends on this being "authoritative":
# it means the window load() returns IS what the model reads back on resume, so
# the app owns the conversation. If a future SDK makes load() a write-only
# mirror, windowing has to be redesigned and this test is the alarm.
COLD_START_MODE = "authoritative"


def test_p1_recorded_load_is_authoritative():
    result = json.loads((PROBES / "p1_result.json").read_text(encoding="utf-8"))
    assert result["COLD_START_MODE"] == COLD_START_MODE, result["verdict"]
    # The control must have passed, or the censored result proves nothing.
    assert result["control_uncensored"]["knew_word"] is True
    assert result["censored"]["still_knew_word"] is False


def test_p1_recorded_the_real_session_key_shape():
    """SessionKey is a TypedDict: project_key, session_id, subpath.

    The first run of this probe used getattr() on it and silently recorded
    every key as (None, None, ()). W5 and W9 both index on these fields, so
    the recorded shape has to be the real one.
    """
    result = json.loads((PROBES / "p1_result.json").read_text(encoding="utf-8"))
    key = result["load_calls"][0]["key"]
    project_key, session_id, subpath = key
    assert project_key and project_key.startswith("-"), project_key
    assert session_id and len(session_id) == 36, session_id
    assert subpath == "", "the main transcript uses an empty subpath"


# P2 settled persistent-vs-per-turn client. "per_turn" means a changed
# system_prompt DOES apply on resume, so a long-lived client is viable and the
# three instruction providers (profile, corrections, platform rules) can be
# re-resolved before every turn. If this ever became "frozen", a persistent
# client would serve a stale personality and newly learned facts would never land.
SYSTEM_PROMPT_MODE = "per_turn"

# P3 settled how nested sub-agent runs are stored. "sdk_partitions" means they
# arrive under a subagents/<id> subpath of the SAME session, so the store mirrors
# them rather than minting its own session keys.
SUBPATH_MODE = "sdk_partitions"

# P4 settled whether entry uuids seen in append() are resume addresses.
# "addressable" means pruning the last turn is a re-anchor via resume_session_at,
# not a rewrite of stored entries.
UUID_MODE = "addressable"


def _result(name):
    return json.loads((PROBES / name).read_text(encoding="utf-8"))


def test_p2_system_prompt_applies_on_resume():
    r = _result("p2_result.json")
    assert r["SYSTEM_PROMPT_MODE"] == SYSTEM_PROMPT_MODE, r["verdict"]
    # The control: turn 1 must have obeyed prompt A, or turn 2 proves nothing.
    assert "ALPHA" in r["turn1_under_A"]["reply"].upper()
    assert "BRAVO" in r["turn2_resumed_under_B"]["reply"].upper()


def test_p3_nested_runs_get_their_own_subpath():
    r = _result("p3_result.json")
    assert r["SUBPATH_MODE"] == SUBPATH_MODE, r["verdict"]
    # Refuse a result where the sub-agent never actually ran. The first run of
    # this probe had the model print a <function_calls> block as prose, which
    # would have recorded exactly the opposite conclusion.
    assert r["task_actually_ran"] is True
    nested = [s for s in r["subpaths_seen"] if s]
    assert nested and nested[0].startswith("subagents/"), nested
    # One session, partitioned -- not a second session.
    assert len(r["session_ids_seen"]) == 1


def test_p4_entry_uuids_are_stable_and_addressable():
    r = _result("p4_result.json")
    assert r["UUID_MODE"] == UUID_MODE, r["verdict"]
    assert r["uuids_unique"] is True
    assert r["parent_links_present"] is True
    assert r["control"]["knew_colour"] and r["control"]["knew_animal"]
    assert r["resumed_at_anchor"]["still_knew_later_fact"] is False
    # The entry vocabulary W5's store has to round-trip.
    for kind in ("user", "assistant"):
        assert kind in r["entry_types"], r["entry_types"]
