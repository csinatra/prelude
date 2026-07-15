"""runs.jsonl — append-only registry of every harness run.

One JSON object per line. A run appears once per lifecycle stage transition
(status: spec_built → agent_run → graded); the latest entry for a run_key is
authoritative. Append-only keeps the registry trivially mergeable across
machines (dev laptop builds specs; cloud box runs agents and grades).
"""

import json
from pathlib import Path

RUNS_PATH = Path("results/runs.jsonl")


def append_run(*, entry: dict) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_PATH.open(mode="a") as handle:
        handle.write(json.dumps(entry) + "\n")


def load_runs() -> dict[str, dict]:
    """Latest entry per run_key (later lines supersede earlier ones)."""
    if not RUNS_PATH.exists():
        return {}
    runs: dict[str, dict] = {}
    for line in RUNS_PATH.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            runs[entry["run_key"]] = entry
    return runs
