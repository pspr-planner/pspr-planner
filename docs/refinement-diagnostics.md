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
| `primitive_prefix_pruned` | grounded method whose leading primitive chain cannot execute |
| `expansions` | task expansions actually performed |
| `peak_candidates`, `expanded_candidates` | widest step frontier and prefixes expanded |
| `frontier_truncated` | step frontiers cut to `max_frontier` |

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
escalating retries of the sequential path.

Section 8 records the correctness fixes this search needed before it could
produce any interleaved plan at all, and why PCP still exceeds its budget.

## 5. Fix 3: cycle-key strategy

`RefinementOptions::cycle_pruning` selects how recursive branches are pruned:
`TaskState` (default, prune a task that recurs in the same state on the current
path) or `Off`. `PSPR_CYCLE_KEY=off` selects the second. Measured on the three
remaining misses, `Off` changes nothing, which matches `cycle_pruned == 0` from
the counters. The option stays as a diagnostic switch for domains whose
recursion legitimately revisits a state with a different remainder.

## 6. Blocksworld-GTOHP p05: why it was slow

`p05` (BW-rand-13, twelve ordered `do_put_on` tasks over thirteen blocks) was
the slowest and least robust instance of the corpus: about 380 ms, and at the
default grounding budget it did not even finish with a goal-satisfying plan.

Instrumenting the counters gave the cause in two steps.

**Dead groundings.** The domain binds free method parameters that only appear
in a subtask: `m2_do_on_table(?x, ?y)` uses `?y` only in `(unstack ?x ?y)`, and
`m5_do_move(?x, ?y, ?z)` uses `?z` only in `(unstack ?x ?z)`. Every one of the
thirteen candidate supports was enumerated, recursed into, and failed at the
`unstack` action: 25 368 dead ends, and `no_applicable` was the only non-zero
counter.

The fix is sound for totally ordered networks: the first subtask executes
immediately, so a primitive first subtask must be applicable in the current
state. Pruning those instances (`primitive_prefix_pruned`) removed all 25 368
dead ends at the cost of one precondition check per grounding.

**Frontier growth.** After that fix the run still needed 31 888 task expansions
for a twelve-step plan. The frontier counters showed why:

| instance | peak candidates | expanded prefixes | expansions | time |
| --- | --- | --- | --- | --- |
| p04 (14 tasks) | 128 | 366 | 3 809 | 29 ms |
| p05 (12 tasks) | 2 048 | 4 158 | 31 888 | 351 ms |

`do_clear` and `do_on_table` can relocate the same towers in different orders,
so each abstract step produces thousands of *distinct* states that all remain
valid prefixes. Every one of them is expanded at the next step, which is where
the time goes. The cap was not the limit: lowering `MAX_REFINEMENTS` from
16 384 to 2 048 changed the runtime by less than 3 %, because the frontier
saturates well below the cap.

The fix is a goal-guided beam over the step frontier
(`RefinementOptions::max_frontier`, `PSPR_MAX_FRONTIER`). When the frontier
exceeds the bound, the prefixes that satisfy the most top-level goal conjuncts
are kept, ties resolved by insertion order so the result stays deterministic.
Problems without a classical goal fall back to truncation in insertion order.

Measured on p05 at `max_groundings = 4096` (the produced plan is identical at
every width: 72 actions, 65 methods):

| `max_frontier` | time |
| --- | --- |
| 0 (unbounded) | 364 ms |
| 256 | 179 ms |
| 128 (default) | 110 ms |
| 64 | 63 ms |
| 32 | 35 ms |

128 is the default because it is above every frontier observed elsewhere in the
corpus (p04 peaks at exactly 128 and is therefore never truncated). A sweep of
all 27 IPC cases at widths 32, 64, 128 and 256 kept 24/27 solved in every case,
with the same three pre-existing failures (Freecell, Monroe, PCP).

The beam also repairs the failure at the default grounding budget: with
`max_groundings = 256` the unbounded search spent its expansion budget on
unproductive prefixes and reported "the refinement does not reach the problem
goal", while the bounded frontier solves the instance in 110 ms.

## 7. Freecell: demand-driven grounding

`ACHIEVE-HOME0(S2)` had grounded method instances (`no_instance_tasks` was
empty) but no precondition held. The methods are the `DO-*` family; for example
`M108-ACHIEVE-HOME0` needs

```
HOME ?CARD0, BOTTOMCOL ?CARD0, SUIT ?CARD5 ?SUIT3, VALUE ?CARD5 ?NUM6,
SUCCESSOR ?NUM4 ?NUM6, SUCCESSOR ?NUM8 ?NUM7, SUIT ?CARD0 ?SUIT3,
VALUE ?CARD0 ?NUM4
```

`M108` has six free parameters. Grounding enumerated them lexicographically over
the typed universe and stopped at the budget (`max_groundings * 4` = 16 384
instances), so the assignments satisfying these atoms were cut off before they
were ever tested: the method looked inapplicable although the instance is
solvable. That was a *grounding* gap, not a search gap.

`graph::ground_methods_in_state` closes it. Instead of enumerating the cross
product, it takes the positive atoms of the method precondition, joins them
against the state atom by atom, and lets the state supply the bindings:

* only `And`/`Atom` nodes participate; parameters constrained only by `Not`,
  `Or`, `Imply`, quantifiers or `Equals` stay unbound and are enumerated over
  the typed universe as before;
* the caller still checks the full precondition, so the join only has to be
  complete, never exact;
* atoms are joined most-selective-first (those the task terms already bind), so
  `M108` collapses from a cross product to thirteen candidates.

`MethodApplicabilityCache` uses it only for the tasks whose enumeration was
actually truncated; every other task keeps the cached enumeration. For a
truncated task the demanded instances replace the arbitrary budget prefix, which
is sound because a prefix instance that the join cannot reproduce would fail the
same conjuncts.

The change moves Freecell from a *grounding* dead end to a *search* problem:
refinement now discovers `M108` and reaches `MOVE-B`, `SENDTOFREE` and
`SENDTOFREE-B`, and it exhausts its expansion budget inside the first abstract
step (`ACHIEVE-HOME(S2)`) instead of failing immediately. The instance is still
not solved: its methods encode the whole Freecell game, so the first top-level
task alone needs a state-space search, and pandaPI also times out on this
instance (30 s in the comparison harness). Closing it needs a heuristic for
goal-less HTN problems, not better grounding.

Because the demanded grounding turns a fast dead end into a long search,
`PSPR_DEMAND_GROUNDING` controls it: enabled in release builds, disabled in
debug builds (whose refinement budget is unbounded, which would make the
Freecell test run for minutes), and overridable in either profile. Freecell at
`max_groundings = 4096`: 1.5 s with the switch off, 16.7 s with it on, both
without a plan.


## 8. Interleaved search: correctness fixes and PCP

### The instance

`PCP_1/p-pcp01` has two unordered top-level tasks, `SG1` and `SG2`. Every `G1`
action requires `turnA` and produces `turnB` plus a marker (`lt1`, `lt2`, `lt3`,
`lx`, `ly`); every `G2` action consumes `turnB` plus the matching marker and
restores `turnA`. The methods of both tasks are recursive, and a marker produced
by a `G1` action must be consumed by the `G2` action of the same name.

pandaPI's plan alternates strictly and nests four method levels per task:

```
t1G1 t1G2 t3G1 t3G2 t2G1 t2G2 t3G1 t3G2 ...
SG1 -> G1_S2t1_S_y ( G1_S2t3_S_y_y_x ( G1_S2t2_S_x_y ( G1_S2t3_y_y_x ) ) )
SG2 -> G2_S2t1_S_y_x_y ( G2_S2t3_S_y_y ( G2_S2t2_S_x_x ( G2_S2t3_y_y ) ) )
```

No linearisation of a single top-level task can produce it: within one task the
actions are executed consecutively by the sequential step loop, and every `G1`
pair needs a `G2` action in between. Only the global frontier can interleave
them.

### Five defects that had to be fixed first

1. **The search committed to executable actions.** After trying the executable
   actions of a node it returned without also trying decompositions, so the
   branch that *decomposes* `SG1` while an outer `yG1` is pending and executable
   was never explored — and that is the shape of PCP's solution. Both passes now
   run, with actions tried first.
2. **A complete frontier was accepted even when it violated the goal.** PCP's
   shortest decomposition (7 actions, 2 methods) ends in `turnB`, so the search
   stopped on a plan that does not reach `turnA`. A complete frontier is now
   only accepted when the classical goal holds, or when the problem has none.
3. **The depth budget was charged to primitive actions.** A slot at depth 0
   could neither decompose nor execute, so at every depth limit the deepest
   actions were dead ends; at depth 3 that accounted for the bulk of the nodes.
   Depth now bounds decomposition nesting only.
4. **The expansion budget was shared with the sequential step loop.** The
   interleaved search probes every ready slot of every node, and the probes
   consumed the budget the sequential path had sized, after which every
   expansion reported "empty" and the search saw dead ends that were not dead.
   The interleaved search now has its own expansion budget and is bounded by its
   node limit and deadline.
5. **No protection against deep dives.** Recursive methods allow 32 nesting
   levels, so a plain depth-first search spent its budget on arbitrarily deep
   chains. The search now deepens iteratively (nesting bound 1, 2, 3, ...) and
   keeps a transposition table of failed frontiers keyed by the state plus the
   pending tasks, their depths and their ordering, so equivalent configurations
   are explored once.

Together these turn a search that produced no interleaved plan at all into one
that finds complete plans (the 7-action one is found in ~1 ms).

### Remaining blocker

The frontier branching is still too wide. Measured with a 20 000 000 node bound
and a 20 s deadline:

| nesting bound | nodes explored | cumulative depth-limit exits |
| --- | --- | --- |
| 1 | 324 | 324 |
| 2 | 30 000 | 30 852 |
| 3 | 1 200 000 | 1 183 000 |
| 4 | ~1 000 000 | 1 576 626 |
| 7 | ~3 800 000 | 3 062 166 |

PCP's plan needs nesting depth 4, and even 3.8 M nodes took 20 s without
reaching a goal-satisfying solution. At each `turnA` state the search may branch
over every method decomposition of both tasks and over every ready action, and
almost all those combinations are pruned only much later, when a marker cannot
be matched.

Closing the instance therefore needs a heuristic for the frontier, not more
budget: a reachability estimate over the pending task network (do the remaining
actions still produce the atoms that the remaining preconditions need, `h_max`
style) or a count of how many pending actions can currently execute. pandaPI
solves the instance in about 30 ms with its heuristic search, so the gap is
search guidance rather than modelling.
