"""Load `.env` before any module reads it.

Every module reads credentials and configuration from `os.environ` (CLAUDE.md
core constraint 4), but loading them was left to the caller's shell. That made
the documented `python -m ...` commands fail from a clean shell, and across the
two-machine split it let a run start with `PRELUDE_REGISTRY_STAGE` unset when
the operator forgot to source the box's env file — not a loud failure but a
silent write to the wrong registry (docs/DATA.md).

Called from each package's `__init__`, not from `__main__` blocks: several
modules read env at import time (an API client, CHROMA_PATH, LLM_PROVIDER), and
those run before a `__main__` block ever executes. Package init is the earliest
point that precedes them all, so a constant added later cannot reintroduce the
gap.

One filename on both machines. The box's env is a least-privilege subset with
different values, so loading a second file would mean picking a precedence
order that is wrong on one machine or the other: a stray dev `.env` on the box
would win and reinstate the stage mismatch this is meant to prevent. The box
copies `.env.cloudbox.example` to `.env` instead.

The real environment always wins (`override=False`), so a one-off
`MODEL=claude-sonnet-5 python -m harness.runner ...` does what it says rather
than being overridden back to the `.env` value.

That direction has a residual risk in the other direction: a variable exported
in the shell — including by an old `set -a && . .env` — silently shadows the
file a reader assumes is authoritative. It is recorded rather than silent (the
run manifest carries model, provider, and stage), and the mitigation if it
becomes a real problem is a pre-run confirmation of the resolved values, not a
flip to `override=True`, which would remove one-off overrides entirely.
"""

from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(".env")

_loaded = False


def load_env() -> None:
    """Load `.env` from the working directory if present, once, without overriding."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
