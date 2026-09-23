"""Where things live.

Everything is derived from the checkout itself, so the bot runs from wherever it
is cloned. ORCHESTRATOR_WORKSPACE moves the directory coding jobs build new
projects in; by default that is deployments/ inside this repo.
"""

import os

ORCHESTRATOR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ORCHESTRATOR_ROOT, "data")
CODING_WORKSPACE_ROOT = os.environ.get(
    "ORCHESTRATOR_WORKSPACE", os.path.join(ORCHESTRATOR_ROOT, "deployments"))
