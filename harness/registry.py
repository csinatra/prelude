"""runs.jsonl — append-only registry of every harness run.

One JSON object per line. A run appears once per lifecycle stage transition
(status: spec_built → agent_run → graded); entries for a run_key are merged
in file order, later fields overriding earlier ones — so a transition entry
carries only its new fields and never erases spec-time ones. Append-only
keeps the registry trivially mergeable across machines (dev laptop builds
specs; cloud box runs agents and grades).
"""

import json
from pathlib import Path

RUNS_PATH = Path("results/runs.jsonl")


def append_run(*, entry: dict) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_PATH.open(mode="a") as handle:
        handle.write(json.dumps(entry) + "\n")


def load_runs() -> dict[str, dict]:
    """Entries merged per run_key in file order; later fields override."""
    if not RUNS_PATH.exists():
        return {}
    runs: dict[str, dict] = {}
    for line in RUNS_PATH.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            runs.setdefault(entry["run_key"], {}).update(entry)
    return runs
