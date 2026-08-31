# Judge Rubric — Flag → Action Classification

**Status: FROZEN.** This rubric is defined in advance of any evaluation run
and must not be revised after seeing results. If a revision proves
unavoidable, it must be documented as a dated amendment below, with all
previously judged runs re-judged under the amended rubric.

*Last revised 2026-08-24, before any evaluation run and therefore not an
amendment. The revision replaced a three-class scheme
(`not_acted_on` / `acted_on_unclear` / `acted_on_positive`) with the two
classes below, and is argued in docs/DECISIONS.md. Only the dev smoke had
been judged; it was re-judged under this text.*

## Purpose

Given (a) one `SpecificationFlag` produced by the spec pipeline for a
competition and (b) the preserved solution artifacts of an agent run
(submission code, trajectory excerpt), classify the flag into exactly one of
two classes.

**Reusing a flag set as a probe.** Flags belong to the C2 run that produced
them. For the base-rate comparison they are judged a second time against the
B2 (and A) solutions for the *same competition* — the flag is unchanged, only
the solution differs. The question "does this solution address this mechanism"
remains well posed for a solution whose agent never received the flag, and the
rate at which those unconditioned solutions address the same mechanisms is the
baseline that C2's action rate is read against. Which runs are paired is an
analysis-plan matter (docs/RESEARCH_DESIGN.md), not a rubric one.

## Classes

### `not_acted_on`
The solution contains no identifiable design choice, code construct, or
logged decision that addresses the mechanism named in the flag. The mere
presence of generic good practice (e.g. a train/validation split existing at
all) does NOT count as acting on a flag unless the flag's specific mechanism
is addressed by the specific construction.

### `acted_on`
The solution contains an identifiable design choice that addresses the
flag's mechanism. Examples: grouped/temporal CV where the flag names a
leakage path along that grouping; class rebalancing where the flag names
exposure bias; explicit resource budgeting where the flag names a resource
constraint.

This class asserts only that the mechanism was addressed. It asserts nothing
about whether the choice helped, and nothing about whether the flag *caused*
the choice — a solution may address a mechanism for its own reasons. Both are
recovered by comparison outside this rubric: contribution by relating action to
outcome post hoc, attribution by comparing this rate against the base rate in
conditions that never received the flag.

## Procedure constraints

- The judge sees: the flag (category, explanation), the solution code, and a
  bounded trajectory excerpt. The judge does NOT see the run's medal outcome
  or score, to prevent outcome-contamination of the action judgment.
- The judge does NOT see which condition produced the solution, and no
  condition or run identifier appears in the prompt. The same flags are judged
  against conditioned and unconditioned solutions, so a visible condition label
  would invite expectancy bias in exactly the comparison that carries the
  attribution claim.
- The trajectory excerpt is the same shape for every item — the best node's own
  analysis and output — so that no flag is judged on more evidence than
  another. Evidence volume must not vary with a run's search-tree shape, or the
  action rate would partly measure tree depth rather than agent behavior.
- Each flag is judged independently.
- The judge must quote the specific artifact evidence (code line, log line)
  supporting an `acted_on` classification; a classification without a
  quote is invalid and defaults to `not_acted_on`.

## Amendments

(none)
