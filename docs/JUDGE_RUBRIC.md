# Judge Rubric — Flag → Action → Outcome Classification

**Status: FROZEN.** This rubric is defined in advance of any evaluation run
and must not be revised after seeing results. If a revision proves
unavoidable, it must be documented as a dated amendment below, with all
previously judged runs re-judged under the amended rubric.

## Purpose

Given (a) one `SpecificationFlag` produced by the spec pipeline for a
competition and (b) the preserved solution artifacts of an agent run
(submission code, trajectory logs), classify the flag into exactly one of
three classes.

## Classes

### `not_acted_on`
The solution contains no identifiable design choice, code construct, or
logged decision that addresses the mechanism named in the flag. The mere
presence of generic good practice (e.g. a train/validation split existing at
all) does NOT count as acting on a flag unless the flag's specific mechanism
is addressed by the specific construction.

### `acted_on_unclear`
The solution contains an identifiable design choice that addresses the
flag's mechanism (examples: grouped/temporal CV where the flag names a
leakage path along that grouping; class rebalancing where the flag names
exposure bias; explicit resource budgeting where the flag names a resource
constraint), BUT the artifacts do not show a discernible connection between
that choice and the run's outcome — or the connection is mixed/ambiguous.
**This is the default class whenever action is present:** claiming positive
contribution requires affirmative evidence, not absence of contrary
evidence.

### `acted_on_positive`
The action criterion above is met, AND the artifacts affirmatively show the
choice contributed to the outcome. Acceptable evidence: trajectory logs
showing the addressed failure occurring before the mitigation and resolved
after; ablation-like comparisons within the run; validation-vs-leaderboard
gap behavior directly attributable to the choice. Weak/inadmissible
evidence: the run merely scoring well; the agent asserting the choice helped
without observable support.

## Procedure constraints

- The judge sees: the flag (category, explanation), the solution code, and
  the trajectory logs. The judge does NOT see the run's medal outcome or
  score, to prevent outcome-contamination of the action judgment.
- Each flag is judged independently.
- The judge must quote the specific artifact evidence (code line, log line)
  supporting any `acted_on_*` classification; a classification without a
  quote is invalid and defaults to `not_acted_on`.

## Amendments

(none)
