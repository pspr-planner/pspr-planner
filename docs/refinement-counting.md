# Refinement: producer/consumer counting and the guided lineariser

This document describes the search that replaces the plain interleaved frontier
search of [§8](refinement-diagnostics.md#8-interleaved-search-correctness-fixes-and-pcp),
what it fixes, what it measures, and what it still does not solve.

## 1. The rule

Every pending primitive action either supplies or consumes each ground atom.
A plan runs every deletion at a moment where the atom is true, so for a node
whose remaining work is fixed:

```
deletions(a) <= additions(a) + [[a in state]]
```

cannot be repaired by any ordering. Goal atoms are counted more strictly,
because the goal has to hold *after* the last action:

| atom | bound |
| --- | --- |
| goal atom, true in the current state | `deletions <= additions` |
| goal atom, false in the current state | `additions >= 1` and `deletions <= additions - 1` |
| any other atom | `deletions <= additions + [[a in state]]` |

Three details keep the rule sound:

* **Delete-before-add.** An action that both adds and deletes the same atom
  leaves it true (`verify::apply_action` applies deletes first), so that deletion
  is dropped from the count. Counting it would reject a feasible plan.
* **Conditional effects.** `imply`-guarded effects depend on the state the
  action runs in, so an action with a conditional effect is skipped entirely
  instead of being counted with the wrong literals.
* **Uncommitted compounds.** A pending compound task has no fixed effect set
  yet. A deficit is only pruned when no pending compound task can add the
  atom's *predicate* — the predicate-level abilities come from a fixpoint over
  the method graph (`TaskAbilities`). Grounding every method instance to get
  atom-level abilities would cost more than the prune saves.

The rule is checked at **every** search node, not only at complete
decompositions. That matters: a deficit that no pending compound can repair
makes the node infeasible immediately, which is what stops the search from
enumerating the whole `(method choice)^depth` space.

## 2. Validation against the reference plan

`pandaPI` solves `PCP_1/p-pcp01.hddl` in 27 ms. Its 26-action plan is a perfect
alternation:

```
t1G1 t1G2 t3G1 t3G2 t2G1 t2G2 t3G1 t3G2
yG1 yG2 yG1 yG2 xG1 xG2 xG1 xG2 yG1 yG2
yG1 yG2 yG1 yG2 xG1 xG2 yG1 yG2
```

`G1` actions delete `turnA` and add `turnB`; `G2` actions do the reverse. The
count above derives `#G1 == #G2` exactly, and the reference plan has 13 of each.
The rule therefore agrees with the reference solution rather than excluding it.

## 3. The guided lineariser

Once every pending task is primitive, the action set is committed and only its
order is open. That is a classical planning problem over the same operators, so
the ordering is delegated to `fast_downward`:

* each ground atom becomes a two-valued variable,
* each action becomes an operator (precondition, add, delete),
* the classical goal is the problem goal,
* the executed-prefix bookkeeping that classical planning does not have — each
  action runs at most once — stays in the search state as a bitset,
* `best_first_search` runs with `ff_heuristic`.

A depth-first linearisation commits to whichever action is applicable, and in
alternating domains that is the wrong one: after `t1G1` consumes `turnA` the next
`G1` action is blocked and only a `G2` action can hand the turn back. FF prefers
exactly the action that restores `turnA`. Measured with the counting prune
disabled, 3 776 of 7 551 committed action sets were ordered successfully; with
the depth-first fallback none were.

## 4. Progression order

The search executes first and decomposes on demand:

1. Execute any ready and applicable action (branching over the alternatives).
2. Only then decompose ready compound tasks.

Decomposing a compound task that is *not* ready cannot expose an applicable
action, so it is deferred; losing nothing keeps the branching small. Executing
first means a wrong method choice is punished by the state within a few steps
instead of only once the whole network is decomposed.

Two supporting changes:

* **Edge splicing.** When a task is decomposed, its predecessors and successors
  are copied onto every child; otherwise the constraints `(pred < slot)` and
  `(slot < succ)` would vanish with the slot and the linearisation phase would be
  free to schedule the method's actions outside the window its parent allowed.
* **Edge collection.** Edges whose endpoints are no longer pending carry no
  information (readiness only looks at pending predecessors) and used to grow
  quadratically over a plan.

Method candidates are ordered by "first subtask executable now", then by
*descending* subtask count while the depth budget allows recursion, then by
index. Preferring the base case first made the search enumerate shallow dead
ends before it ever considered the depth the goal needs.

## 5. Measured outcome

| instance | before | after |
| --- | --- | --- |
| PCP_1/p-pcp01 | fail, 77 ms | fail, 3 018 ms |
| corpus (27 instances) | 24 solved, median 18 ms | 24 solved, median 16 ms |

No regressions: Blocksworld-GTOHP p05 still solves in 162 ms, Barman-BDI still
solves both files, Freecell and Monroe still fail exactly as before.

`PCP_1` is still unsolved. The counting prune fires — 12 704 times in the default
budget, 160 935 times with a 60 s budget — but the reference decomposition is
never reached:

* the plan needs both `SG1` and `SG2` to nest seven levels deep,
* at each level the method choice in one task must match the marker production of
  the other (`lt1` is produced by `t1G1` and consumed by `t1G2`, and so on),
* the marker constraints only become checkable once the producing compound is
  decomposed, which in a progression search happens late,
* a 60 s, 376 586-node run still ends at dead ends of the form `t1G2: lt1`.

Closing the instance needs a heuristic over the *task network* — pandaPI's RC
heuristic, which counts how many of the remaining tasks can reach the remaining
goal atoms. That is a larger piece of work than the counting rule and is not
implemented here.

## 6. Measurement knobs

| variable | effect |
| --- | --- |
| `PSPR_INTERLEAVED_NODES` | node limit of the interleaved search (default 20 000) |
| `PSPR_INTERLEAVED_MS` | wall-clock deadline of the interleaved search |
| `PSPR_REFINE_BUDGET_MS` | overrides the refinement time budget |
| `PSPR_NO_COUNT=1` | disables the counting prune |
| `PSPR_NO_MEMO=1` | disables the failed-frontier memo |

The last three exist for measurement and are not intended as user-facing
settings.

## 7. Diagnostics

`RefinementDiagnostics` gained the counters below; `examples/diagnose_refine_modes`
prints them on the `interleaved stats:` line.

| counter | meaning |
| --- | --- |
| `counting_pruned` | nodes rejected by the producer/consumer count |
| `complete_frontiers` | frontiers whose action set was fully committed |
| `linearisation_failed` | committed action sets with no applicable ready action |
| `nodes_visited` | frontier nodes visited by the interleaved search |
