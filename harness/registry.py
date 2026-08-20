"""Append-only registry of every harness run, one file per experiment stage.

One JSON object per line. A run appears once per lifecycle stage transition
(status: spec_built → agent_run → graded); entries for a run_key are merged
in file order, later fields overriding earlier ones — so a transition entry
carries only its new fields and never erases spec-time ones. Append-only
keeps the registry trivially mergeable across machines (dev laptop builds
specs; cloud box runs agents and grades).

Registries are SEPARATED BY STAGE — `results/runs_dev.jsonl`,
`results/runs_eval_v1.jsonl` — rather than accumulating in one file. run_key is
`{competition}_{condition}_{seed}` and carries no corpus identifier, so a run
rebuilt against a new corpus produces the SAME key as its stale predecessor and
load_runs() would merge the two silently. Separate files make that structurally
impossible instead of merely detectable. A stage boundary is anything that
invalidates prior rows: a corpus rebuild, a pipeline change, the dev-to-eval
transition.

Append-only is a property within a stage, not across stages. Retiring a stage
means starting a new file, never rewriting an existing one.

What the separation blocks is POOLING rows from different corpus generations
into one analysis, which is the invalid operation. Deliberate cross-stage
comparison stays available — pass `stage=` to load a named registry, e.g. to
read runs_eval_v1 beside runs_eval_v1_5. Report such a comparison as
REPLICATION across corpus generations, with the corpus difference stated; the
paired within-competition statistics in analysis/stats.py assume one corpus and
apply within a stage, not across.

The active stage comes from PRELUDE_REGISTRY_STAGE (matching the repo's
PRELUDE_SPEC_PATH convention) and defaults to `dev`, so writing into an eval
registry has to be asked for explicitly:

    PRELUDE_REGISTRY_STAGE=eval_v1 python -m harness.batch ...

Resolved per call, never bound at import — otherwise the path would freeze at
whatever the environment held when this module was first imported.
"""

import json
import os
from pathlib import Path

RESULTS_DIR = Path("results")
STAGE_ENV = "PRELUDE_REGISTRY_STAGE"
DEFAULT_STAGE = "dev"


def active_stage() -> str:
    return os.environ.get(STAGE_ENV) or DEFAULT_STAGE


def registry_path(*, stage: str | None = None) -> Path:
    """Path to a stage's registry; the active stage when none is named."""
    return RESULTS_DIR / f"runs_{stage or active_stage()}.jsonl"


def append_run(*, entry: dict, stage: str | None = None) -> None:
    path = registry_path(stage=stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="a") as handle:
        handle.write(json.dumps(entry) + "\n")


def load_runs(*, stage: str | None = None) -> dict[str, dict]:
    """Entries merged per run_key in file order; later fields override.

    Merging is per run_key WITHIN one stage. Never merge the return values of
    two stages into a single mapping — identical run_keys across corpus
    generations would silently overwrite each other, which is the failure the
    per-stage split exists to prevent. Compare them side by side instead.
    """
    path = registry_path(stage=stage)
    if not path.exists():
        return {}
    runs: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            runs.setdefault(entry["run_key"], {}).update(entry)
    return runs
