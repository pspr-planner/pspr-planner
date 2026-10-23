# Refinement diagnostics and miss analysis

This note records the instrumentation added to hierarchical refinement, the
results it produced on the IPC misses, and the fixes that followed.

## 1. Counters

`evaluation_tree::RefinementDiagnostics` accumulates, per refinement run:

| Field | Meaning |
| --- | --- |
| `budget_time_exhausted` | branch pruned because the time budget ran out |
| `budget_expansions_exhausted` | branch pruned because the expansion budget ran out |
| `depth_exhausted` | `refine_task` called with `depth == 0` |
| `cycle_pruned` | `(task, state)` already on the current path |
| `no_applicable_method` | grounded instances exist but no precondition holds |
| `no_applicable_tasks` | the same, per grounded task |
| `no_instance_tasks` | no grounded method instance at all for the task |
| `dead_end_samples` | bounded samples with the unmet precondition literals |
| `network_constraint_rejected` | method `:constraints` rejected the grounding |
| `partial_ready_empty` | partial-order frontier with no ready subtask |
| `abstract_step_failed` | abstract step that produced no candidate |

`DecompositionSession::diagnostics` collects them, and
`examples/diagnose.rs` / `examples/diagnose_refine_modes.rs` print them. The
samples name the failing action or method grounding together with the atoms that
do not hold, which is what turned the miss analysis from guesswork into two
concrete bugs.

## 2. What the counters showed

Six IPC instances failed before this work: Barman p01/p02, PCP, Freecell,
Minecraft, Monroe. All of them reported `no_applicable*` dead ends and zero
budget, depth or cycle exits, i.e. neither timeouts nor over-eager pruning:

| instance | dead-end task | reading |
| --- | --- | --- |
| Barman p01 | `pour-shot-to-clean-shaker` (16 per attempt) | every grounding of the action failed |
| PCP | `t1G1`, `t2G1`, `t3G1` (`turnA` missing) | cross-task interleaving required |
| Freecell | `ACHIEVE-HOME0(S2)`, `ACHIEVE-HOME3(S2)` | no grounded method precondition holds |
| Monroe | not reached | refinement ran past the diagnostic budget |

## 3. Fix 1: delete-before-add effect semantics

`verify::apply_action` rejected an action whose effect added and deleted the
same atom. Barman needs the standard reading: the delete list is applied first
and the add list second, so such an atom stays true. `pour-shot-to-clean-shaker`
adds and deletes `shakerLevel(s, l)` with the same `l` whenever `next(l, l)`
holds, which the instance has:

```
(SUCCESSOR level1 level1) (SUCCESSOR level2 level2)
```

Every grounding therefore aborted, which is exactly the 16 dead ends per
attempt. With the standard semantics both Barman instances refine: p01 in
18 actions, p02 in 46 actions.

## 4. Fix 2: interleaved frontier search (prototype)

`refine_plan_steps` refines one abstract step to completion before the next.
That is complete for a totally ordered top-level network but rejects plans that
interleave subtasks of different top-level tasks. PCP needs this: `t1G1` leaves
`turnB`, so the recursive `SG1` subtask can only continue after an `SG2` action
switched the turn back.

`refine_plan_interleaved` keeps every open task in one frontier and expands
whichever ready task the state allows. It runs as a fallback after the
sequential attempts fail, with a node bound (`PSPR_INTERLEAVED_NODES`, default
20 000 in release and 2 000 in debug builds) and a wall-clock bound
(`PSPR_INTERLEAVED_MS`, default 3 000 ms), and a single attempt rather than the
escalating retries of the sequential path. It prefers ready primitive actions
over further decomposition, and prefers decompositions whose first subtask
executes immediately.

The prototype does not yet solve PCP: the instance needs roughly 26 interleaved
actions and the blind depth-first search exhausts its bound first. What it does
provide is the mechanism and the measurement path; the next step is a
relaxed-plan or goal-count guide for the frontier.

## 5. Fix 3: cycle-key strategy

`RefinementOptions::cycle_pruning` selects how recursive branches are pruned:
`TaskState` (default, prune a task that recurs in the same state on the current
path) or `Off`. `PSPR_CYCLE_KEY=off` selects the second. Measured on the three
remaining misses, `Off` changes nothing, which matches `cycle_pruned == 0` from
the counters. The option stays as a diagnostic switch for domains whose
recursion legitimately revisits a state with a different remainder.

## 6. Remaining miss: Freecell

`ACHIEVE-HOME0(S2)` has grounded method instances (`no_instance_tasks` is
empty) but no precondition holds. The methods are the `DO-*` family; for
example `M108-ACHIEVE-HOME0` needs

```
HOME ?CARD0, BOTTOMCOL ?CARD0, SUIT ?CARD5 ?SUIT3, VALUE ?CARD5 ?NUM6,
SUCCESSOR ?NUM4 ?NUM6, SUCCESSOR ?NUM8 ?NUM7, SUIT ?CARD0 ?SUIT3,
VALUE ?CARD0 ?NUM4
```

`M108` has six free parameters. Grounding enumerates the free parameters
lexicographically over the typed universe and stops at the budget
(`max_groundings * 4` = 16 384 instances), so the assignments that satisfy these
atoms are cut off before they are ever tested. The method looks inapplicable
although the instance is solvable, which is a *grounding* gap rather than a
search gap.

The fix is demand-driven grounding: unify the positive precondition atoms
against the current state, atom by atom, instead of enumerating the cross
product of every free parameter. That is what grounders such as
`pandaPIgrounder` do, and it would also remove the lexicographic bias for the
other learned domains.
