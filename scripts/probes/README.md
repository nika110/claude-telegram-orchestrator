# SDK probes

Measurements taken while designing the conversation store (app/memory/), when
this bot was part of a larger multi-agent system. Each probe asks the Claude
Agent SDK one question the design depends on and records the answer in
`p*_result.json`; `tests/test_probe_results.py` pins those answers so a future
SDK release that changes one fails loudly. The scripts mention agents that are
not in this repo; the measurements are about the SDK, not about them.
