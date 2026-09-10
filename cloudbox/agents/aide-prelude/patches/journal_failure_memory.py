"""Carry unresolved failures in the journal summary, not just successes.

aideml's `generate_summary` iterates `good_nodes`, so the Memory block that
seeds `_draft` and `_improve` contains only solutions that worked. Failure
knowledge does reach later steps, but only when a debug attempt *succeeded* —
the fixing node's own `plan`/`analysis` describes the bug it repaired. So a
lesson survives exactly when the run was lucky enough not to need it, and is
discarded when the run is stuck.

Two coupled facts make that a structural dead spot rather than a mild
asymmetry. `search_policy` returns to drafting only when `good_nodes` is empty,
and the Memory block is built only from `good_nodes` — so every draft triggered
by failure is guaranteed to arrive with an empty Memory. Observed 2026-09-07 on
uw-madison: `good_nodes` stayed empty for 17 of 18 steps while repeated drafts
re-derived the same id-parsing error, each with a blank Memory section.

This appends unresolved failures to the same block:

- **Buggy leaves only.** A buggy node with descendants was debugged; if a
  descendant succeeded, its summary already carries the lesson. Leaves are the
  failures nothing has fixed.
- **Diagnosis only, never code.** `analysis` is the reviewer's account of what
  went wrong. Including the buggy implementation would multiply context and
  risks the agent avoiding a sound approach because one broken node used it.
- **Deduplicated and capped.** Repeated attempts produce near-identical
  diagnoses; the seventh copy teaches nothing the first did. `agent.steps` is
  500, so an uncapped list would grow without bound — and prompt size is a
  throughput cost, measured at ~0.37 nodes/min on pizza.

This is a DESIGN CHANGE, not a bug fix — upstream's summary is good-nodes-only
by construction, and aideml's paper does not claim otherwise. Applied
identically to every condition, decided on a development holdout before any
eval run, expected direction stated in advance: it helps all conditions and
reduces variance, so it cannot be claimed afterwards as a treatment effect.
See docs/DECISIONS.md for the adoption rule this is being tested against.
"""

import aide.journal as m

FAILURE_CAP = 5

HELPER = '''
    def _unresolved_failures(self, cap: int = ''' + str(FAILURE_CAP) + '''):
        """Buggy leaves, newest first, deduplicated by diagnosis and capped.

        A buggy node with descendants was debugged; if any descendant became a
        good node, that node's summary already carries the lesson. Leaves are
        the failures nothing has fixed.
        """
        seen: set = set()
        out: list = []
        for n in reversed(self.buggy_nodes):
            if not n.is_leaf:
                continue
            key = (n.analysis or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(n)
            if len(out) >= cap:
                break
        return list(reversed(out))
'''

OLD_RETURN = '        return "\\n-------------------------------\\n".join(summary)'
NEW_RETURN = (
    '        summary += [\n'
    '            f"Failed attempt (not yet fixed — do not repeat): {n.analysis}"\n'
    '            for n in self._unresolved_failures()\n'
    '        ]\n'
    '        return "\\n-------------------------------\\n".join(summary)'
)

OLD_DEF = "    def generate_summary(self, include_code: bool = False) -> str:"


def main() -> None:
    src = open(m.__file__).read()

    assert src.count(OLD_RETURN) == 1, "aideml generate_summary return changed — re-check this patch"
    assert src.count(OLD_DEF) == 1, "aideml generate_summary signature changed — re-check this patch"
    assert "_unresolved_failures" not in src, "patch already applied"

    src = src.replace(OLD_RETURN, NEW_RETURN)
    src = src.replace(OLD_DEF, HELPER.rstrip() + "\n\n" + OLD_DEF)

    open(m.__file__, "w").write(src)
    print("journal_failure_memory: applied")


if __name__ == "__main__":
    main()
