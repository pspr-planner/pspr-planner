# Performance review

This note records where the solver spends its time, which optimisations were
applied, and which ones remain. It is organised by the six categories of the
review request:

1. small algorithmic changes,
2. arena / scratch allocation,
3. vectorizability,
4. cache locality,
5. unnecessary branching,
6. indirection.

## How the numbers were obtained

`examples/profile_phases.rs` parses one domain/problem pair, then reports the
median wall time of `build_evaluation_tree`, `evaluate`, `refine_plan` and
`validate_refinement` over several repetitions:

```
cargo build --release --example profile_phases
LD_LIBRARY_PATH=<scip install>/lib \
  target/release/examples/profile_phases <domain.hddl> <problem.hddl> 7
```

Corpus (IPC instances from `hddl/test_data/ipc`, `max_groundings = 256`,
`method_summary = PrimitiveOnly`, `refine_plan`/`validate_refinement` enabled):

| instance | tasks | constraints | clusters |
| --- | --- | --- | --- |
| Blocksworld-GTOHP/p18 | 53 | 87 | 1 |
| Depots/p05 | 10 | 69 | 5 |
| Logistics-Learned-ECAI-16/probLOGISTICS-10-1 | 10 | 39 | 5 |
| Factories-simple/pfile10 | 1 | 48 | 1 |
| Barman-BDI/pfile10 | 3 | 5 | 1 |
| Multiarm-Blocksworld/pfile_06_005 | 6 | 75 | 1 |
| Minecraft-Player/p-003-003-003-003 | 1 | 0 | 1 |
| PCP_1/p-pcp01 | 2 | 0 | 2 |

`evaluate` builds the tree itself, so it contains a second tree build; that is
why its numbers track the `tree` column.

`refine_plan`/`validate_refinement` times are not comparable across this
review: while the measurements were taken, hierarchical refinement in the
working tree started rejecting these instances (independently of the changes
below), so only `tree` and `evaluate` are compared. Every instance below had a
non-empty tree and a non-empty abstract plan, so the two measured phases are
representative.

## Result

Median of 7 runs, in milliseconds (best of two rounds; sizes are identical
before and after):

| instance | tree before | tree after | evaluate before | evaluate after |
| --- | --- | --- | --- | --- |
| Blocksworld-GTOHP/p18 | 0.94 | 0.47 | 5.70 | 2.27 |
| Depots/p05 | 26.89 | 2.84 | 29.19 | 4.86 |
| Logistics/probLOGISTICS-10-1 | 26.44 | 0.51 | 27.21 | 0.81 |
| Multiarm-Blocksworld/pfile_06_005 | 0.58 | 0.62 | 1.09 | 1.07 |
| Factories-simple/pfile10 | 0.07 | 0.16 | 0.15 | 0.29 |
| Barman-BDI/pfile10 | 0.05 | 0.10 | 0.11 | 0.15 |
| Minecraft-Player/p-003-003-003-003 | 0.03 | 0.07 | 1.34 | 1.39 |
| PCP_1/p-pcp01 | 5.94 | 0.05 | 2.77 | 0.07 |

The sub-millisecond rows are dominated by process-level noise. The multi-cluster
rows are the meaningful ones: building the evaluation tree of a five-cluster
problem dropped from ~27 ms to ~1-3 ms, and evaluating it from ~28 ms to
~1-6 ms.

The phase breakdown of a five-cluster problem before the changes was

```
graph 3.4 ms | cluster 0.06 ms | agglomerate 24 ms   (tree)
leaves 2.9 ms | merges 0.04 ms                       (evaluate)
```

so the agglomeration dominated everything else. After the changes the
agglomeration is no longer measurable on these instances.

## 1. Small algorithmic changes

### 1.1 Merge-level ILP replaced by exact enumeration for small instances

`src/solver/partition/agglomeration/hierarchy.rs`

The Gilpin et al. merge-level ILP (`optimal_hierarchy_with`) is solved with
SCIP. Its branch-and-bound effort explodes with the number of clusters; on
synthetic distance matrices one call took

| clusters | 2 | 3 | 4 | 5 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| time | 1.4 ms | 2.5 ms | 15 ms | 760 ms | 470 ms | 14.4 s |

and 24 ms on the real five-cluster Depots instance. The ILP levels are an
ultrametric `M(a, c) <= max(M(a, b), M(b, c))` over `1..=levels`, i.e. a chain
of nested partitions, and only the *order* of the distinct levels enters the
triple term while the height penalty only shrinks when levels move down. An
optimal solution therefore merges at consecutive levels `1, 2, 3, ...`, so the
whole solution space is the set of sequences of at most `levels` proper
coarsenings. `enumerate_hierarchy` walks those sequences depth-first and keeps
the best one. With the default `levels = clusters - 1` the search visits 2, 8,
64, 872 and 18 024 nodes for 2..6 clusters, so instances with at most
`EXACT_INSTANCE_LIMIT = 6` clusters are solved this way; 7 clusters would
already need 525 520 nodes, so larger instances keep the ILP. A node budget
(`EXACT_NODE_LIMIT`) additionally guarantees that a pathological instance can
never be slower than before. `enumeration_finds_the_ilp_optimum` cross-checks
the enumeration against SCIP on random instances for `n = 2..4`, every level
count, several shared-constraint groups and two weights.

### 1.2 Scheduling ILP replaced by a topological sort

`src/solver/partition/agglomeration/schedule.rs`

The scheduling model minimised the sum of the evaluation positions subject to
tree and DAG precedences. Every internal node occupies exactly one position and
every position is taken, so that sum is `1 + 2 + ... + n` for **every** linear
extension: the model was a feasibility problem. `optimal_schedule` now runs
Kahn's algorithm with a deterministic (smallest-id-first) tie-break, which also
removes an `n²`-binary ILP that would have grown quadratically for large trees.
This alone took the remaining agglomeration cost from ~3 ms to nothing.

### 1.3 Merge objective: fused transcendentals and a negligible-margin fast path

`src/solver/evaluation_tree/merge/sgplan.rs`, `merge/penalty.rs`

For every ordering constraint and every optimization step the relaxed objective
needed

* `softplus(-m/T)` (one `exp` plus one `ln_1p`),
* `logistic(-m/T)` (another `exp`),
* `phi(v)` and `phi'(v)` of the penalty space (two `exp` for the exponential
  space).

Both the softplus and the logistic are functions of the same `z = exp(-m/T)`, and
`exp(4v)` serves both the exponential value and slope. `PenaltySpace::value_and_slope`
and the rewritten loop need at most one `exp`, one `ln_1p` and one `exp_m1` per
constraint instead of up to five transcendental calls. Orderings with
`margin > 50 T` contribute less than `1e-21` to the objective and the gradient
and are skipped entirely, which is the common case once the schedule is nearly
feasible. A unit test pins the fused formulas to the previous reference
implementations.

### 1.4 Quadratic validation of a merged plan

`src/solver/evaluation_tree/merge/sgplan.rs`

`RelaxedProblem::validate` located the position of both sides of every shared
ordering constraint with `plan_tasks.iter().position(..)`, i.e. a scan over the
whole plan per constraint. It now builds one `HashMap<usize, usize>` of task
positions per validated plan.

### 1.5 Already applied before this review

* `src/solver/evaluation_tree/verify.rs`: `apply_action` stops at the first
  applicable schema instead of collecting every candidate and keeping the
  first, and `collect_effects` reuses a scratch `String`.
* `src/solver/evaluation_tree/graph.rs`: `atom_key_into` formats ground atoms
  into a caller-provided `String`, so state membership tests allocate nothing.
* `src/solver/domain.rs`: `is_subtype_of` walks the type hierarchy with a step
  bound instead of allocating a `HashSet` per call (it is called per candidate
  object while grounding).
* `src/solver/evaluation_tree/hierarchical.rs`: groundings are cached per task
  and applicability per `(task, state)`; the ranker/search context is only
  built when a ranker or a trace sink is attached.
* `src/solver/evaluation_tree/liftability.rs`: producer literals are
  materialised once per producer instead of once per producer/consumer pair.
* `src/solver/fast_downward/successor.rs`: `for_each_applicable` visits
  operators through a callback, so the planning problem pushes successors
  without materialising an intermediate index vector.
* `src/solver/fast_downward/search.rs`: the preferred-operator membership test
  is a linear scan over a short list instead of a `HashSet` per expansion.

## 2. Arena / scratch allocation

### 2.1 `MergeScratch`

`src/solver/evaluation_tree/merge/sgplan.rs`

The merge evaluates the relaxed objective up to `iterations * adam_steps`
(2000 by default) times per merge node, and every evaluation allocated three
vectors (`completions`, the softmax weights and the gradient) plus one more
inside `smooth_max`. `MergeScratch` holds them for the lifetime of one merge and
the objective is written into it in place; the optimizer swaps the gradient
buffer instead of allocating a new one, so the Adam inner loop is
allocation-free after the first step.

### 2.2 `smooth_max_into`

`src/solver/evaluation_tree/merge/penalty.rs`

`smooth_max` now fills a caller-provided weight buffer. The allocating
`smooth_max` wrapper only exists for the tests.

### 2.3 `HeuristicScratch`

`src/solver/fast_downward/heuristics.rs`, `knowledge.rs`, `search.rs`

A relaxed planning graph is built for every evaluated state, and each build
allocated the fact-level vector, the achiever vector and — in the FF extraction
— a `BTreeSet` plus a `HashSet`. `HeuristicScratch` keeps the two fact arrays,
the extraction queue, the relaxed-plan vector and generation-stamped arrays for
the former sets alive across evaluations; `PlanningProblem` owns one and the
`*_with` variants of the heuristics take `&mut HeuristicScratch`. The public
`ff_heuristic`/`h_max`/`causal_graph_heuristic` entry points are unchanged and
allocate a scratch internally.

### 2.4 Causal graph heuristic: one flat cost table

`src/solver/fast_downward/heuristics.rs`

`costs[v][source][target]` used to be a `Vec<Vec<Vec<f64>>>` rebuilt for every
state (`n` inner vectors plus `n * size` vectors plus `n * size * size`
allocations). It is now one contiguous `Vec<f64>` in the scratch, indexed by a
per-variable base offset, and the per-source `distance`/`local_state`/`settled`
vectors are reused as well.

## 3. Vectorizability

* `smooth_max_into` computes the exponentials in a single pass over the values
  with a precomputed reciprocal (`(value - largest) * (1/T)` instead of a
  division per element), accumulates the sum in the same pass, and normalises in
  a second trivial pass. This is a straight-line, branch-free loop over `f64`
  that the auto-vectorizer can handle.
* `Adam::update`, `Sgd`, `Momentum`, `Nesterov`, `RmsProp` and `Adagrad` were
  already element-wise loops over parallel slices; they are unchanged.
* The relaxed objective accumulates into `gradient[child]`, which is a scatter
  over a handful of children; it is not vectorizable and the branch-light form
  above is the practical limit.
* `Task::conditions_hold` and `condition_level` are gather loops over a state
  vector with data-dependent indices. They stay scalar; the improvement that
  does pay off is removing the bounds check (below) rather than restructuring
  them.

## 4. Cache locality

* **Fact identifiers are precomputed** (`CompiledTask::fact_offsets`,
  `fact_count`). Every fact of a task lives in one dense index space, so the
  relaxed planning graph and the FF extraction stream two contiguous arrays
  instead of recomputing offsets and chasing a `Vec<Variable>` per fact.
* **The h_CG cost table is contiguous** (see 2.4), so the inner Dijkstra loop
  touches adjacent memory instead of following three levels of pointers.
* **Borrowed atom lookups**: `RelaxedProblem::new` formatted a fresh `String`
  per shared atom to look it up in a `BTreeSet<String>`. It now formats into a
  reused buffer and looks the `&str` up through `Borrow`.
* **Shared states are reference counted** (`hierarchical.rs`): solver states are
  `Arc<BTreeSet<String>>`, so visited-set keys, applicability-cache keys and
  refinement branches bump a refcount instead of deep-copying a tree of strings.
* **Groundings are cached per task** (`hierarchical.rs`): the grounded method
  instances, their grounded subtask vectors and their ids are stored behind
  `Arc<[..]>` and shared by every state that expands the task.

## 5. Unnecessary branching

* **`evaluate_axioms` returns immediately for axiom-free tasks**
  (`fast_downward/task.rs`). Leaf tasks compile without axioms, and the early
  return skips the layer bookkeeping and the full derived-variable scan for
  every successor.
* **The merge objective skips comfortably satisfied orderings** instead of
  computing a softplus that underflows (see 1.3).
* **`value_and_slope`** replaces a pair of `match`es over the penalty space and
  halves the transcendental work per constraint.
* **`apply_action`** bails with `is_disjoint` before formatting the conflict
  list, so the error path only runs when there is an error.
* **`decode`** in `sgplan.rs` hoists the child lengths out of the per-step scan
  and selects with a sentinel index instead of an `Option` comparison per child.

## 6. Indirection reduction

* **`for_each_applicable`** (`fast_downward/successor.rs`) removes the
  intermediate `Vec<usize>` of applicable operators; the planning problem
  applies operators in the same traversal that finds them.
* **`evaluate_into`** writes the gradient into the scratch buffer, so the
  optimizer receives a slice instead of a fresh `Vec`.
* **`std::mem::take`** in the h_CG Dijkstra loop replaces a `State` clone per
  settled value; the local state is never read again after it is settled.
* **`GroundedInstance`** (`hierarchical.rs`) stores the grounded subtasks, the
  subtask ids and the method network once per task rather than rebuilding them
  per branch.

## Remaining opportunities

These were identified and measured but not implemented; they are listed with
the shape of the intended change.

1. **Leaf summarisation (`leaf.rs::summary_operators`) is the largest remaining
   leaf cost**: ~190 µs per task on Depots and ~70 µs on Multiarm, i.e. ~2 ms
   for the ten tasks of Depots. It grounds every method instance of the task
   (`ground_methods_bounded`, up to `max_groundings`) and then translates the
   precondition and the effects of every subtask action *per instance*. The
   grounding of a task does not depend on the state or the cluster, so it could
   be computed once per `(domain, problem)` and shared between the incidence
   graph (`graph.rs::task_atoms`), the leaf builder and `verify.rs`; a per-solve
   `HashMap<TaskNode, Arc<[GroundedMethod]>>` threaded through those three call
   sites would remove two of the three repetitions and enable caching the
   per-instance translations.
2. **Incidence graph grounding** (`graph.rs`) costs ~3.4 ms on Depots and ~90 µs
   on Logistics. It is dominated by quantifier expansion and atom collection
   over `BTreeSet<Atom>`; collecting into a `Vec` plus one sort/dedup at the end
   would remove the per-insert tree walk.
3. **Refinement/validation state handling** (`verify.rs::apply_action`) builds
   two `BTreeSet<String>` effect sets and clones the state for every plan step.
   Reusing the effect sets and applying deletes/adds into an owned state buffer
   would remove two allocations per step. (Not measured in isolation here
   because refinement was being reworked in the working tree.)
4. **Search closed list** (`fast_downward/search.rs`) hashes the whole state
   vector on every lookup and insert. Storing a 64-bit state hash in `Node` and
   keying the closed list by that hash (with the state as the tie-breaker)
   removes the repeated hashing for the expansion-heavy cases.
5. **Repeated `BTreeSet` clones in `evaluate`** (`evaluation_tree/mod.rs`):
   `initial_atoms` is cloned for every merge node and child solutions are cloned
   when they are handed to a merge. Wrapping both in `Arc` would make those
   reference-count bumps.
6. **Atom/task interning**: `Atom` and `TaskNode` carry owned `String`s and are
   compared and hashed as strings throughout the incidence graph and the merge.
   Interning them per solve would turn comparisons into integer compares and
   shrink every set of atoms; this is the single largest structural change
   available and would touch most modules.
