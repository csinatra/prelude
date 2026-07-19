"""Local mirror of the spec injection performed inside the MLE-bench container.

The canonical injection happens in cloudbox/agents/aide-prelude/start.sh:
after the competition description, the per-run spec.md (mounted at
/home/spec/spec.md) is appended under SPEC_HEADER. This module replicates
that composition byte-for-byte so the injected document can be previewed,
token-counted, and tested locally without a container. A unit test asserts
the shell script and this module share the same header.
"""

SPEC_HEADER = "ADVISOR CONTEXT\n------"


def compose_injected_instructions(*, description: str, spec: str) -> str:
    """description.md content + delimited spec section, as the agent sees it."""
    return f"{description}\n{SPEC_HEADER}\n\n{spec}"
