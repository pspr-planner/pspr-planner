# Compound tasks in PSPR

Status: implementation note, 2026-10-22. Stages C and G are implemented
together as the hierarchical path. Approach D was implemented, measured against
C+G, and removed; A, B, E and F remain design options.

By default the evaluation tree solves a projection of the HTN problem in which
every compound task is opaque. `graph.rs` collects only the atoms of the
top-level task network and of the method preconditions; `leaf.rs` turns a
compound task into one operator per method whose only effect is an `executed`
marker; and `verify.rs` checks the method precondition but does not execute the
method's subtasks. This is the reason the comparison in the README reports
`goal=violated` for Blocksworld-GTOHP, Depots, Satellite-GTOHP,
AssemblyHierarchical and Monroe_PO_1, and it is the main semantic gap between
PSPR and a complete HTN planner such as PANDA.

This note records several ways to close that gap. Each approach is stated with
its formal model, pseudo-code, the points in PSPR where it would attach, and
its correctness and cost conditions. The approaches are deliberately
independent: they can be combined, and the recommended roadmap at the end
orders them by implementation cost and expected value. In short, run the
two-phase refinement of Approach C together with primitive-only macro
summaries (Approach G); the benchmark showed C+G is faster in the median than
the adaptive C-first path and at least as complete as C alone. Progression
(Approach B) and exact encodings (Approach F) are the completeness fallbacks,
while eager expansion (Approach A) and task-atom compilation (Approach E) are
baseline implementations. Approach D was tried and removed because it matched
C+G coverage while adding overhead.

## Implementation status

- **Approach C, two-phase refinement.** Implemented in
  `src/solver/evaluation_tree/hierarchical.rs`. `solve_hierarchical` builds
  the abstract plan with the existing evaluation tree, then recursively
  grounds methods and actions. It uses the method recorded in the abstract
  plan first and then tries the other applicable methods. The refinement is a
  bounded breadth-first enumeration of alternatives (up to 2,048 partial
  refinements per level), so it can backtrack over method choices and reject a
  decomposition that leaves the state goal false. A path-local visited set of
  `(task, state)` pairs prunes recursive methods that rediscover the same
  subproblem (for example `get_to` through an intermediate location). The
  search is sound and bounded, but not exhaustive.
- **Approach G, primitive-only summaries.** Implemented as
  `EvaluationOptions::method_summary = MethodSummary::PrimitiveOnly` and
  `leaf.rs::summary_operators`. Only methods with primitive, totally ordered,
  deterministic subtasks are summarised; all other methods remain opaque, and
  the summary is skipped when a precondition clause or effect alternative
  would require a conditional or disjunctive encoding. `--summary` exposes
  this mode in the CLI.
- **C+G hierarchical path.** `solve_hierarchical` enables primitive method
  summaries unconditionally; the benchmark showed this is faster in the median
  than the adaptive C-first path and at least as complete as C alone.
  `--hierarchical` selects this path.
- **Approach D, recursive evaluation.** Removed after the comparison: it
  matched C+G coverage on the 27-instance suite but added overhead. Section 5
  is retained as a design record, not as implemented code.
- **Validation.** `validate_refinement` recursively re-executes method
  preconditions, subtask order and primitive effects, and checks the final
  state and goal. `ValidationOptions` and `validate_plan` retain their
  network-level semantics for the default abstract planner.
- **CLI.** `--hierarchical` selects C+G; `--summary` without refinement
  summarises the abstract solver; without either, PSPR keeps its historical
  abstract behaviour. `--max-groundings` raises the task/method grounding
  budget.

The implemented refinement is deliberately bounded and backtracking over a
finite set of alternatives. It handles the Blocksworld, Depots, Satellite,
AssemblyHierarchical, Transport and Logistics instances exercised in the
README comparison, but recursive or tightly state-dependent methods (for
example the PCP_1 task network and Barman's level-dependent pouring methods)
can still require the exhaustive search described in Approaches B or F. The
learned Freecell model also exceeds the 4,096-method grounding limit on one
task, and the Minecraft and Monroe instances exceed the benchmark's
30-second refinement budget. Those remain future work.

## 1. HTN semantics and the current abstraction

### 1.1 HTN planning

A grounded HTN problem is a tuple

$$
P = \langle F, A, C, M, s_0, TN_0, g \rangle,
$$

where:

- $F$ is a finite set of ground atoms;
- $A$ is a set of primitive action schemas;
- $C$ is a set of compound task schemas;
- $M$ is a set of methods;
- $s_0 \subseteq F$ is the initial state;
- $TN_0 = \langle T_0, \prec_0, \approx_0 \rangle$ is the initial task network;
- $g$ is a state-goal formula, possibly absent.

For a primitive action $a$, write $\mathrm{pre}_a$, $\mathrm{add}_a$ and
$\mathrm{del}_a$ for its precondition, add effects and delete effects. Its
state transition is

$$
\tau_a(s) =
\begin{cases}
(s \setminus \mathrm{del}_a) \cup \mathrm{add}_a & \text{if } s \models
\mathrm{pre}_a,\\
\text{undefined} & \text{otherwise}.
\end{cases}
$$

A task network is $TN = \langle T, \prec, \approx \rangle$, where $T$ is a
finite multiset of task occurrences, $\prec \subseteq T \times T$ is an acyclic
ordering relation, and $\approx$ is a set of equality and inequality
constraints. Throughout this note, $t$ denotes a **task occurrence**, not a
schema: two occurrences of the same ground task in different positions are
distinct, and they receive different markers, task atoms and method choices.
This distinction matters for duplicate tasks and for every compilation below.
A method is

$$
m = \langle c(\vec{x}), \phi_m, TN_m \rangle,
\qquad
TN_m = \langle T_m, \prec_m, \approx_m \rangle,
$$

and decomposes the compound task $c(\vec{x})$ into the subtask network
$TN_m$. A grounding $\sigma$ of the method's parameters produces the method
instance $m\sigma$ with subtasks $T_m\sigma$.

A single decomposition step replaces a compound task $t = c(\vec{t})$ in a
task network by an instance of one of its methods:

$$
TN \xrightarrow{t,m,\sigma} TN' =
\left\langle
(T \setminus \{t\}) \cup T_m\sigma,\;
\prec',\;
\approx \cup \approx_m\sigma
\right\rangle,
$$

with

$$
\prec' =
\prec
\;\cup\; \prec_m\sigma
\;\cup\; \{(u,v) \mid u \prec t,\ v \in T_m\sigma\}
\;\cup\; \{(v,w) \mid v \in T_m\sigma,\ t \prec w\}.
$$

The result must remain acyclic. A decomposition tree $D$ is a finite sequence
of such steps. Its frontier $L(D)$ is the multiset of tasks that were never
decomposed. $D$ is primitive when every task in $L(D)$ is a primitive action,
and a plan $\pi = \langle a_1,\dots,a_k\rangle$ is a linearisation of $L(D)$
that respects $\prec_D$ and the equality constraints. The plan is executable
from $s_0$ when each $a_i$ is applicable in the state produced by its
predecessors; it is a solution when, in addition, the final state satisfies
$g$ (if $g$ is present).

This is the standard task-network semantics used by UMCP
(Erol, Hendler, and Nau 1994, `erol1994umcp`) and SHOP2
(Nau et al. 2003, `nau2003shop2`); see also Ghallab, Nau, and Traverso (2004,
`ghallab2004automated`) for the textbook presentation. The complexity results
of Erol, Hendler, and Nau (1996, `erol1996complexity`) explain why all complete
versions of the approaches below need a depth bound, a finite grounding, or
both.

### 1.2 What the evaluation tree solves

Let $\alpha(P)$ be the following projection of $P$. Keep $F$, $A$, $s_0$,
$TN_0$ and $g$. For every ground task $t$ in $TN_0$, introduce a marker atom
$\mathrm{exec}_t$. A primitive task $a$ is compiled as usual, with the
additional effect $\mathrm{exec}_a$. A compound task $t$ with methods $M(t)$
gets one abstract operator per method instance $m\sigma$:

$$
\mathrm{pre}(o_{t,m}) =
\phi_m\sigma \wedge
\bigwedge_{u \prec t} \mathrm{exec}_u,
\qquad
\mathrm{add}(o_{t,m}) = \{\mathrm{exec}_t\},
\qquad
\mathrm{del}(o_{t,m}) = \varnothing.
$$

The leaf encoding in `leaf.rs` uses exactly this operator: the method name and
the precondition atoms come from the method schema, and the only effect is the
marker for the task at that position. The incidence graph in `graph.rs`
includes the method-precondition atoms (through `task_atoms`) but not the
atoms of the method's subtasks. The merge then orders the abstract operators
and checks their shared local constraints.

Thus PSPR currently computes plans of the task-network projection
$\alpha(P)$: sequences of top-level tasks that respect the top-level
ordering, the method preconditions, and the shared atoms of the projection.
It does not compute a decomposition tree. The classical goal is not part of
the abstraction at all; it is only checked afterwards by `validate_plan`.

### 1.3 The liftability gap

Let $\mathrm{Sol}_{\mathrm{HTN}}(P)$ be the set of HTN solutions and let
$\mathrm{proj}$ map a decomposition plan to its **top-level plan**: the
sequence of top-level tasks together with the method chosen for each top-level
compound occurrence. Let $\mathrm{TopPlans}(P)$ be the set of all top-level
plans consistent with the orderings of $TN_0$, and let

$$
\mathrm{Top}(P) =
\{\mathrm{proj}(D,\pi) \mid (D,\pi) \in \mathrm{Sol}_{\mathrm{HTN}}(P)\}
\;\subseteq\; \mathrm{TopPlans}(P)
$$

be the top-level plans induced by HTN solutions.

The abstraction $\alpha(P)$ is **not** a sound relaxation of $P$ in either
direction:

- **It can reject a valid HTN plan.** Suppose a compound task $t_1$ has a
  method whose subtasks add atom $p$, and a later task $t_2$ has a method
  whose precondition contains $p$. In a full HTN plan, the subtask of $t_1$
  establishes $p$ before $t_2$ is decomposed. In $\alpha(P)$, $t_1$ only adds
  $\mathrm{exec}_{t_1}$ and never writes $p$, so the abstract operator for
  $t_2$'s method cannot be applied even though the projection of the HTN plan
  is a perfectly valid top-level sequence.
- **It can accept a top-level plan that no decomposition supports.** Suppose
  a method for $t$ has precondition $\top$ but its first subtask needs an atom
  that is false in $s_0$. In $\alpha(P)$ the method operator is applicable;
  in the full problem, every decomposition using that method fails when the
  subtask is executed. More generally, the abstract plan may choose a method
  whose subtask effects do not establish $g$.

The reason is that the abstraction keeps method preconditions but drops
subtask effects. It is a **top-level projection**, not a relaxation, and no
general inclusion holds between $\mathrm{Top}(P)$ and the plans of
$\alpha(P)$. The projection is useful because it is fast and often predictive,
but its results must be refined or verified: `validate_plan` detects the
goal-level failures, while the strict recursive validation described in
Approach C detects the decomposition-level failures.

The first direction is detectable statically for a fixed grounding.
`evaluation_tree::liftability` computes the may-write literals of every
reachable ground task node as a least fixpoint over the method decomposition,
intersects them with the DNF literals of every later method or action
precondition in the transitive closure of the top-level ordering, and reports
each match as a *hidden satisfier* (same sign: the projection cannot see the
write) or a *hidden falsifier* (opposite sign: the projection keeps believing
the literal). The analysis over-approximates on purpose -- it ignores
preconditions, merges disjunctive effects, and expands every method grounding
-- so a reported pair is a candidate rather than a proof, and a clean report
is not a proof of liftability. `LiftabilityReport::truncated` records whether
a grounding or node bound was hit, and the `--liftability` flag of the CLI
prints the report for a problem.

Call a top-level plan $\pi \in \mathrm{TopPlans}(P)$ **liftable** when there
is a primitive decomposition $D$ and a linearisation $\pi'$ of $L(D)$ such
that

1. $\pi'$ is executable from $s_0$,
2. $\pi'$ satisfies $g$ (if present), and
3. $\pi$ is the top-level plan of $D$, up to the ordering of $TN_0$.

The central design question is how to make PSPR produce plans that are
liftable, or how to repair an abstract plan that is not. The approaches below
answer that question at different points in the pipeline; Table 1 summarises
the trade-offs.

| Approach | Where decomposition happens | Complete? | Reuses PSPR partitions? | Typical cost |
| --- | --- | --- | --- | --- |
| A. Eager expansion | before clustering | for a depth bound | after flattening | very high |
| B. Lazy progression | during leaf/global search | for a depth bound | partly | high |
| C. Two-phase refinement | after the abstract plan | no (sound only) | fully | low |
| D. Recursive trees | inside the evaluation tree | relative to the merger | fully | medium--high |
| E. Task-atom compilation | before classical search | for a finite closure | after compilation | high |
| F. SAT/ILP encoding | exact subproblem | for a horizon/depth bound | per cluster | very high |
| G. Macro summaries | inside leaf operators | only under restrictions | fully | low |

Table 1. The main design options. "Complete" always means complete for the
stated bound; full HTN planning is undecidable in the general recursive case
and intractable even for many finite fragments (`erol1996complexity`).

## 2. Approach A: eager expansion with guarded decomposition

### 2.1 Idea

Expand every compound task before clustering, recursively replacing it by the
subtask networks of its methods. To preserve the state-dependent meaning of
method preconditions without interleaving search, insert a **guard action** for
each decomposition. A guard has the method precondition as its precondition
and no effects. It marks the exact point at which the method is applied.

### 2.2 Formalisation

For a ground method instance $m\sigma$ that decomposes $t$, define the guard

$$
\gamma(m,\sigma) =
\left\langle
\phi_m\sigma,\;
\varnothing,\;
\varnothing
\right\rangle.
$$

The expansion replaces $t$ by $T_m\sigma \cup \{\gamma(m,\sigma)\}$ and adds
the orderings

$$
u \prec \gamma(m,\sigma) \quad \text{for every } u \prec t,
\qquad
\gamma(m,\sigma) \prec v \quad \text{for every } v \in T_m\sigma,
$$

and, for every $w$ with $t \prec w$,

$$
v \prec w \quad \text{for every } v \in T_m\sigma \cup \{\gamma(m,\sigma)\}.
$$

The guard is a zero-effect action, so it only restricts the linearisation; it
does not change the state. A guard can be shared by all subtasks because the
method precondition is evaluated once, at decomposition time.

**Proposition A.1.** For a fixed expansion derivation $D$, an executable
linearisation of the guarded primitive network projects to a valid HTN plan for
$D$ by contracting each guard into its method instance. Conversely, every HTN
plan for $D$ extends to a linearisation of the guarded network by inserting
the guard before every subtask of its method instance (hence before the first
subtask in any linearisation).

The proof is a straightforward induction on the decomposition tree: the guard
has no effects, its ordering constraints place it exactly at the decomposition
point, and every primitive action retains its original preconditions and
effects. The only subtlety is a method with an empty subtask network: its
guard has no successor inside the method, so it is placed before the method's
successor $w$ and after all predecessors.

If expansion enumerates all methods without considering the state, it can
produce many guarded networks whose guards are unreachable. This is not
incorrect, only wasteful; a partial-order planner or the leaf search prunes
them when it discovers that a guard's precondition is false. The alternative
is to interleave expansion and search (Approach B).

### 2.3 Pseudo-code

```text
function EXPAND(TN, depth):
    if depth = 0 or TN has no compound task:
        return {TN}

    t ← CHOOSE_COMPOUND(TN)
    out ← ∅
    for (m, σ) in GROUND_METHODS(t):
        TN' ← REPLACE_WITH_GUARD(TN, t, m, σ)
        out ← out ∪ EXPAND(TN', depth − 1)
    return out

function SOLVE_EAGER(P, depth):
    for TN in EXPAND(P.TN0, depth):
        π ← PLAN_CLASSICAL_OR_PARTIAL_ORDER(TN, P.s0, P.g)
        if π ≠ fail:
            return π
    return fail
```

`GROUND_METHODS(t)` must enumerate every substitution that is compatible with
the method's task terms and the problem's typed objects. In PSPR this is the
same enumeration as `Universe::assignments` in `graph.rs`, charged against
`EvaluationOptions::max_groundings`.

A more efficient variant does not materialise the set of all expansions. It
uses a partial-order planner as an oracle and expands a compound task when the
planner selects it:

```text
function EXPAND_AND_SOLVE(TN, s, depth):
    if TN has no compound task:
        return PLAN(TN, s, P.g)
    if depth = 0:
        return fail

    t ← CHOOSE_COMPOUND(TN)
    for (m, σ) in GROUND_METHODS(t):
        if s ⊭ φ_mσ:
            continue
        TN' ← REPLACE_WITH_GUARD(TN, t, m, σ)
        π ← EXPAND_AND_SOLVE(TN', s, depth − 1)
        if π ≠ fail:
            return π
    return fail
```

This variant is sound but is not complete for partial-order networks unless
`CHOOSE_COMPOUND` branches over a sufficient set of tasks; it is best used as
a bounded baseline rather than the default.

### 2.4 Integration in PSPR

- **New option.** Add `max_decomposition_depth: usize` and
  `max_decomposition_branches: usize` to `EvaluationOptions`. The first bounds
  recursive methods; the second bounds the product of method choices.
- **Graph construction.** Add a pre-pass `expand_task_network(domain,
  problem, options)` in `graph.rs` that returns a `TaskConstraintGraph` whose
  tasks are all primitive and whose constraints include the guards. The
  existing `ground_task_network` already knows how to ground top-level
  parameters; the new pass recursively calls `domain.methods_for` and
  `bindings_from` for each method.
- **New task metadata.** `TaskNode` needs an origin record: the decomposition
  node and method that produced it. A side table
  `ExpansionRecord { task: usize, parent: Option<usize>, method: Option<String> }`
  is enough and avoids changing the hash/ordering semantics of `TaskNode`.
- **Leaf construction.** `leaf.rs` no longer needs its compound branch; every
  leaf task is primitive. The `executed` markers can be retained for
  bookkeeping, but the guards are real operators with preconditions and no
  effects. This also removes the method-name parsing from the leaf plan.
- **Validation.** `verify.rs` can validate the expanded plan directly. Guards
  are no-ops; if the guard-to-method map is retained, the validator can report
  which method each guard licenses. Add a strict recursive mode only if the
  plan is kept in method-step form instead of being expanded.

### 2.5 Correctness and cost

- **Sound and complete for a fixed expansion.** With a complete classical or
  partial-order planner and all guards present, Proposition A.1 gives an
  equivalence between plans of the expanded network and HTN plans of that
  derivation.
- **Complete overall** only if the depth bound is at least the maximum
  decomposition depth of a solution and all method instances are enumerated.
  With recursive methods there is no finite bound in general; the planner must
  either detect cycles in the method-call graph or impose a bound.
- **Cost.** The number of guarded networks is at most
  $\prod_{c} |M(c)|^{d}$ in the worst case, where $d$ is the depth bound.
  Grounding adds a factor for each method parameter. This is the most
  expensive approach in the list and is mainly useful as an oracle or as a
  small-instance baseline.

## 3. Approach B: lazy progression search

### 3.1 Idea

Do not expand the whole hierarchy. Keep a residual task network and a state,
and at each step choose a ready task. If it is primitive, execute it; if it is
compound, replace it by the subtask network of an applicable method. This is
the agenda-based progression loop of SHOP2 (`nau2003shop2`) and the operational
core of HTN planners such as PANDA (`holler2021panda`).

### 3.2 Formalisation

A search state is a pair $\langle s, TN\rangle$, where
$TN = \langle T, \prec, \approx\rangle$ is the residual network and $s$ is
the current state. The ready tasks are

$$
R(TN) = \{ t \in T \mid \nexists u \in T : u \prec t \}.
$$

The two transition rules are

$$
\frac{
t \in R(TN), \quad t \in A, \quad s \models \mathrm{pre}_t
}{
\langle s, TN\rangle \to
\langle \tau_t(s), TN \setminus \{t\}\rangle
}
$$

and, for a ground method instance $m\sigma$ that decomposes $t$,

$$
\frac{
t \in R(TN), \quad m\sigma \in \mathrm{inst}(t), \quad s \models \phi_m\sigma
}{
\langle s, TN\rangle \to
\langle s, \mathrm{refine}(TN,t,m\sigma)\rangle
}.
$$

The function $\mathrm{refine}$ is exactly the replacement from Section 1.1.
A goal state has $T = \varnothing$ and $s \models g$ (or only
$T = \varnothing$ when $g$ is absent).

The rule for compound tasks does not change $s$; method preconditions are
therefore checked in the state at the moment of decomposition, which is the
HTN convention. The residual network encodes all outstanding orderings and
equalities, so interleaving is handled by choosing different ready tasks at
different search nodes.

**Proposition B.1 (completeness).** If the method-call depth is bounded by
$d$, the set of ground method instances is finite, and the search exhaustively
branches over every $t \in R(TN)$ and every applicable method instance, then
progression finds a solution whenever one exists. With a fixed deterministic
selection function $t = \mathrm{select}(R(TN))$, it is complete for totally
ordered networks but not for partial-order networks: a solution may require
expanding a compound task while another ready task is executed first.

The proof is the standard induction on the length of the derivation, using the
fact that every solution has a first decomposition or primitive step whose
precondition holds in $s_0$. The partial-order caveat is the usual commutation
argument: a canonical selection can commit to an order that prevents a later
method from being applicable.

### 3.3 Pseudo-code

```text
function PROGRESS(s, TN, steps, depth, memo):
    if T(TN) = ∅:
        return [] if s ⊨ g else fail
    if steps = 0 or depth = 0:
        return fail
    key ← (s, TN, steps, depth)
    if key ∈ memo:
        return memo[key]

    for t in READY(TN):                    # branch for partial order
        if PRIMITIVE(t):
            if s ⊨ pre(t):
                π ← PROGRESS(τ_t(s), REMOVE(TN, t), steps − 1, depth, memo)
                if π ≠ fail:
                    return [t] + π
        else:
            for (m, σ) in METHODS(t):
                if s ⊨ φ_mσ:
                    TN' ← REFINE(TN, t, m, σ)
                    π ← PROGRESS(s, TN', steps, depth − 1, memo)
                    if π ≠ fail:
                        return [METHOD_STEP(m, t)] + π

    memo[key] ← fail
    return fail

function PROGRESS_BEST_FIRST(P, h, stepsmax, depthmax):
    OPEN ← {(h(s0, TN0), 0, s0, TN0, stepsmax, depthmax, [])}
    while OPEN ≠ ∅:
        (f, g, s, TN, steps, depth, π) ← POP_MIN(OPEN)
        if T(TN) = ∅ and s ⊨ P.g:
            return π
        for (s', TN', step, cost, steps', depth') in
                SUCCESSORS(s, TN, steps, depth):
            if not VISITED(s', TN', depth'):
                PUSH(OPEN, (g + cost + h(s', TN'), g + cost, s', TN',
                            steps', depth', π + [step]))
    return fail
```

For a total-order network, `READY(TN)` is a singleton and the first loop is
the usual SHOP2 loop. For a partial-order network, branching over all ready
tasks is exact but can explore many equivalent interleavings; sleep sets,
symmetry reduction, or a canonical linearisation with backtracking can reduce
this.

### 3.4 Heuristics

The state includes both facts and the residual task network. Useful heuristics
are:

- the number of remaining tasks, or the number of primitive tasks remaining;
- a relaxed planning graph over the task atoms `pending(t)` and `done(t)`
  (Approach E), using the FF or additive heuristic
  (`hoffmann2001ff`, `helmert2006fastdownward`);
- a method-count estimate, e.g. the minimum number of method applications
  needed to reach a primitive frontier;
- the current merge penalty of the cluster that owns the selected task.

The first is admissible only in a very weak sense; the relaxed planning graph
heuristics are more informative but require compiling the agenda into the
state (Section 7).

### 3.5 Integration in PSPR

- **New module.** Add `src/solver/htn/progression.rs` with a residual
  `TaskNetwork` type (tasks, ordering, equalities, ready set) and the
  transition rules above. Reuse `Universe` for method grounding and
  `Domain::methods_for` for method lookup.
- **Leaf scope.** Run progression inside a leaf whose clusters are small and
  whose tasks are all in the same connected component; cross-cluster ordering
  is still handled by `MergeSolver`. If a method's subtasks leave the cluster,
  either add them to the cluster (extending `Cluster::n1s`) or expose them to
  the merge as interface tasks.
- **Fallback.** Use progression when a merge fails, or when the recursive
  evaluation tree (Approach D) cannot solve a method subtree. This gives a
  complete local oracle without changing the top-level partition.
- **Termination.** Memoise on `(s, TN)`, bound the method depth, and reject
  cyclic method-call graphs unless recursion is explicitly desired.

### 3.6 Correctness and cost

Progression is sound for any node expansion that applies the rules above. It
is complete under the bounded-depth and exhaustive-branching conditions of
Proposition B.1. Its cost is exponential in the plan length and in the number
of method choices; the branching factor at a compound task is the number of
applicable methods, and at a partial-order network it is the number of ready
tasks. It is the natural complete baseline for small clusters, but it loses
PSPR's local-grounding advantage if used on the whole problem.

## 4. Approach C: two-phase network planning and hierarchical refinement

### 4.1 Idea

Keep PSPR's fast abstract plan, then refine it into a primitive decomposition.
Phase 1 computes a top-level plan $\pi \in \mathrm{TopPlans}(P)$ from PSPR's
projection. Phase 2 recursively
decomposes each compound task in $\pi$, starting from the method choices that
the abstract plan already records. If refinement fails, try a different method
or a different abstract plan. This is the smallest change that can turn the
current pipeline into a full HTN planner.

### 4.2 Formalisation

Let $\pi = \langle t_1,\dots,t_n\rangle$ be an abstract plan (a linearisation
of the merged top-level plan). Define the partial function

$$
\mathrm{Refine}: \mathrm{Task} \times \mathrm{State} \rightharpoonup
(\mathrm{Plan} \times \mathrm{State})
$$

by

$$
\mathrm{Refine}(t,s) =
\begin{cases}
(\langle t\rangle, \tau_t(s)) & \text{if } t \in A \text{ and }
s \models \mathrm{pre}_t,\\
(\mathrm{RefineNetwork}(TN_m\sigma,s)) & \text{if } t \in C,\ m\sigma \in
\mathrm{inst}(t),\ s \models \phi_m\sigma,\\
\bot & \text{otherwise},
\end{cases}
$$

where $\mathrm{RefineNetwork}(TN,s)$ executes a task network:

$$
\mathrm{RefineNetwork}(\langle \varnothing,\prec,\approx\rangle,s) =
(\langle\rangle,s),
$$

and for a non-empty network it selects a ready task $t$, computes
$\mathrm{Refine}(t,s) = (\pi_t,s')$, then recursively refines the residual
network from $s'$. For a total-order network the first task is fixed. For a
partial-order network the selection must backtrack over ready tasks, and even
then the sequential recursion is not complete: it finishes one ready task
before starting another, whereas a solution may need to interleave primitive
steps from two ready tasks. The complete version of this phase is therefore
the progression loop of Section 3 restricted to the method choices of $\pi$;
the simpler recursion below is a sound first implementation.

**Proposition C.1 (soundness).** If $\mathrm{Refine}$ returns
$(\pi',s')$ for an abstract plan $\pi$, then $\pi'$ is an executable
decomposition of the tasks in $\pi$, and if $s' \models g$ then $\pi'$ is an
HTN solution. The statement is sound by construction: every method
precondition is checked, every primitive action is applied with its effects,
and the network orderings are respected.

**Proposition C.2 (incompleteness).** Refinement can fail even when $P$ has a
solution, because the abstract plan $\pi$ may not be liftable (Section 1.3).
Completeness requires searching over all top-level plans, or at least over a
repair neighbourhood of $\pi$.

The repair problem can be written as

$$
\pi^\ast =
\arg\min_{\pi' \in \mathrm{TopPlans}(P)}
\left\{ d(\pi,\pi') \;\middle|\; \mathrm{Refine}(\pi') \neq \bot
\right\},
$$

where $d$ is, for example, the number of changed top-level tasks or the number
of changed ordering edges. A practical search explores the neighbourhood
generated by (i) trying another method for a failing compound task, (ii)
reordering two adjacent independent tasks, and (iii) replacing one leaf
solution by another solution of the same cluster.

### 4.3 Pseudo-code

```text
function REFINE_PLAN(π, s):
    if π = []:
        return ([], s) if s ⊨ g else fail
    t ← HEAD(π)
    (πt, s') ← REFINE_TASK(t, s)
    if πt = fail:
        return fail
    (πrest, s'') ← REFINE_PLAN(TAIL(π), s')
    return (πt + πrest, s'') if πrest ≠ fail else fail

function REFINE_TASK(t, s):
    if PRIMITIVE(t):
        if s ⊨ pre(t):
            return ([t], τ_t(s))
        return fail

    for (m, σ) in METHODS(t):
        if s ⊭ φ_mσ:
            continue
        (π, s') ← REFINE_NETWORK(TN_mσ, s)
        if π ≠ fail:
            return ([METHOD_STEP(m, t)] + π, s')
    return fail

function REFINE_NETWORK(TN, s):
    if T(TN) = ∅:
        return ([], s)

    for t in READY(TN):                  # backtrack for partial order
        (πt, s') ← REFINE_TASK(t, s)
        if πt = fail:
            continue
        (πrest, s'') ← REFINE_NETWORK(REMOVE(TN, t), s')
        if πrest ≠ fail:
            return (πt + πrest, s'')
    return fail

function SOLVE_WITH_REFINEMENT(P):
    π ← PSPR(α(P))
    if π = fail:
        return fail
    (πfull, s) ← REFINE_PLAN(π, P.s0)
    if πfull ≠ fail and s ⊨ P.g:
        return πfull
    return REPAIR(P, π)
```

`REPAIR` can be a best-first search over the neighbourhood above, or simply a
loop that re-runs `PSPR` with an additional ordering constraint and a different
grouping. It must have its own node or time budget; otherwise the approach
silently becomes a complete search.

### 4.4 Integration in PSPR

- **Plan representation.** The current leaf operator for a compound task is
  named `method_name[task]` and `verify.rs` already parses it as
  `Step::Method { name, task }`. The abstract plan therefore already records a
  method choice. The first refinement attempt should use that choice before
  trying the other methods.
- **Where to run.** Add a `refine` stage after `evaluation_tree::evaluate` in
  `src/solver/solve.rs`. The stage consumes `Evaluation.plan` and
  `PartialSolution::plan_tasks`; it returns a primitive plan plus a final
  state. `SolveResult` can gain a `decomposition: Vec<PlanStep>` field while
  keeping `plan` as the primitive plan.
- **Strict validator.** `verify.rs` currently checks a method step by checking
  the method precondition (`check_method`) and a bare compound step by
  rejecting it if methods exist (`check_compound`). Add
  `ValidationOptions::expand_methods: bool` (default `false` to preserve the
  existing network-level tests). When enabled, `validate_plan` must recursively
  execute the chosen method's subtasks and check task coverage, ordering and
  equalities inside the method network. This is a necessary precondition for
  trusting any refinement result.
- **Repair hooks.** Re-running PSPR with a different `EvaluationOptions` can
  change the clustering and the abstract order. For finer repair, expose the
  leaf alternatives in `PartialSolution` and let the refiner try another leaf
  plan before replanning the whole tree.
- **Method alternatives.** `RefineNetwork` needs method instances, not just
  method schemas. Use `Universe::assignments` or a precomputed
  `MethodGrounding` table so that a failing method does not trigger a fresh
  grounding pass on every backtrack.

### 4.5 Correctness and cost

Refinement is sound but incomplete; it is the only approach whose correctness
does not depend on a new global search. Its cost is essentially the cost of
the abstract PSPR run plus the cost of decomposing the chosen plan. The
decomposition phase is local and can reuse the same clusters: the subtasks of
a method usually touch the same atoms as the parent task, so they belong to
the same or to a neighbouring cluster. The main risk is a high failure rate on
the current benchmark: 12 of the 26 solved task networks report
`goal=violated`, so the first implementation should measure liftability and
the repair budget before it is treated as a complete solution.

This approach is also the easiest to validate against PANDA: the primitive
plan can be checked with the same strict validator, and the abstract plan can
be kept as a certificate of the method choices.

## 5. Approach D: recursive evaluation trees (removed)

This approach was implemented, measured, and removed: on the 27-instance IPC
suite it produced the same complete plans as C+G but added overhead. The
design is retained below for reference.

### 5.1 Idea

Make the evaluation tree recursive. A primitive task is a leaf. A compound
task is an internal **method node**: its children are the subtasks of one
method, its guard is the method precondition, and its merge enforces the
method's ordering and equality constraints. The existing partition, ILP
scheduling and Adam merge are reused at every level. This is the approach most
faithful to PSPR's architecture, because it treats a method as just another
subproblem decomposition.

### 5.2 Formalisation

Let $U$ be the set of task occurrences reachable from $TN_0$ by method
application up to a depth bound. A **decomposition DAG** is a pair
$(U, E)$ where each compound occurrence $t$ has a hyperedge
$e_{t,m\sigma} = (t, T_m\sigma)$ labelled with $\phi_m\sigma$ and
$\prec_m\sigma$. A solution selects one hyperedge for every compound
occurrence and orders the resulting primitive frontier.

For each task $t$, let $\mathcal{P}(t)$ be the set of partial plans that solve
$t$ from a given interface state. For a primitive $t$,

$$
\mathcal{P}(t) = \{\langle t\rangle\}
\quad\text{(subject to the local leaf constraints)}.
$$

For a compound $t$,

$$
\mathcal{P}(t) =
\bigcup_{m \in M(t)}
\left\{
\mathrm{Merge}\left(\phi_m\sigma,\; \prec_m\sigma,\; \{\pi_j\}_{j=1}^{k}\right)
\;\middle|\;
\pi_j \in \mathcal{P}(s_j)
\right\},
$$

where $T_m\sigma = \{s_1,\dots,s_k\}$ and $\mathrm{Merge}$ receives the child
plans,
the method guard and ordering, and the interface atoms. It returns the
interleavings of the child plans that satisfy the guard, the orderings and the
shared atoms, or $\varnothing$ if there is none. In PSPR, the operation inside
the braces is exactly what `MergeSolver::merge` does for fixed child plans: it
receives the child solutions, the shared constraints and the initial atoms,
and returns an interleaved plan or an error.

The interface of a subtree $N$ is the set of atoms it shares with the rest of
the problem:

$$
\partial N = \mathrm{atoms}(N) \cap \mathrm{atoms}(U \setminus N).
$$

Only $\partial N$ is visible to the parent. The local atoms of $N$ are hidden
inside the subtree, which is what makes the recursive partition useful.

The cost of a method node can be estimated by

$$
\mathrm{cost}(t) = \min_{m \in M(t)}
\left[
\mathrm{cost}_{\mathrm{merge}}(m) +
\sum_{s \in T_m\sigma} \mathrm{cost}(s)
\right],
$$

where $\mathrm{cost}_{\mathrm{merge}}$ includes the guard and the number of
shared interface atoms. This is a heuristic for choosing the next method to
evaluate, not an optimality criterion.

### 5.3 Pseudo-code

```text
function SOLVE_TASK(t, interface, depth):
    if depth = 0:
        return ⊥
    if PRIMITIVE(t):
        return LEAF_SOLVE(t, interface)

    best ← ⊥
    for (m, σ) in METHODS(t):
        if not FEASIBLE_GUARD(m, interface):
            continue
        children ← []
        for s in T_mσ:
            child ← SOLVE_TASK(s, interface ∪ SHARED(s, t), depth − 1)
            if child = ⊥:
                break
            children.append((s, child))
        if |children| ≠ |T_mσ|:
            continue
        merged ← MERGE(children, φ_mσ, ≺_mσ, interface)
        if merged ≠ ⊥ and COST(merged) < COST(best):
            best ← merged
    return best

function BUILD_TREE(t, depth):
    if depth = 0 or PRIMITIVE(t):
        return LEAF(t)
    alternatives ← []
    for (m, σ) in METHODS(t):
        children ← [BUILD_TREE(s, depth − 1) for s in T_mσ]
        alternatives.append(METHOD_NODE(t, m, σ, children))
    return CHOICE_NODE(t, alternatives)
```

`FEASIBLE_GUARD` is a cheap consistency check: the guard's atoms must not
contradict the interface. The full check happens in `MERGE`. `CHOICE_NODE`
keeps the alternatives until `SOLVE_TASK` selects the best one; a
depth-first implementation can evaluate each alternative immediately and keep
only the best plan.

### 5.4 Integration in PSPR

- **Task graph.** Extend the grounded universe in `graph.rs` from the
  top-level network to the method closure. A `TaskNode` should carry a stable
  identity and a parent/method link; `ConstraintNode` gains a
  `MethodGuard { task: usize, method: String }` variant.
- **Recursive tree.** `EvaluationTree` in `clustering.rs` becomes a node type:
  `Leaf { cluster }`, `Merge { children, shared }`, or
  `Method { guard, method, children }`. `build_evaluation_tree` dispatches on
  the task kind and calls `cluster_tasks` on each method's local subgraph.
- **Shared constraints.** `attach_shared_constraints` already attaches a
  constraint to the lowest common ancestor of the clusters that use it. Run it
  independently inside each method subtree; the parent's `Method` node owns
  the guard and the interface constraints.
- **Merging.** `MergeProblem` gains an optional `guard: Option<ConstraintNode>`
  and the method's ordering constraints. `SgplanMerger` can stay unchanged if
  the guard is represented as a shared constraint that the merge must satisfy;
  a zero-effect guard is naturally checked by the existing exact validator.
- **Solver.** `solve.rs` becomes a memoised dynamic program over
  `(task, interface)`. The memo key should include the interface atoms, not the
  whole state, so that independent subtrees share work.
- **Cutoffs.** Add `max_depth`, `max_interface_atoms`, and a per-subtree leaf
  budget. If a subtree fails, fall back to progression (Approach B) or an exact
  encoding (Approach F) for that subtree.

### 5.5 Correctness and cost

- **Soundness.** If every `MERGE` returns only plans that satisfy the guard,
  the method orderings and the shared atoms, then a returned plan is a valid
  decomposition. The existing `SgplanMerger` already validates its decoded
  merge exactly, so the recursive version inherits this guarantee.
- **Completeness.** The recursion is complete relative to the child plan sets
  and the merge operator: it explores every method branch and every child
  solution that the leaves and merges produce. It is not complete for the HTN
  problem if a merge cannot repair a shared atom by replanning a child; this is
  the same limitation as the current tree. A fallback exact solver closes the
  gap at extra cost.
- **Cost.** The dominant new cost is the interface state: the number of
  distinct assignments to $\partial N$ explored at a node. The method
  branching factor multiplies the number of subtrees, but memoisation and
  branch-and-bound on $\mathrm{cost}(t)$ keep it manageable on the IPC
  instances where methods are shallow. Recursive methods require a depth bound
  or a cycle check.

## 6. Approach E: classical compilation with task atoms

### 6.1 Idea

Compile the HTN problem into a classical planning problem whose state contains
task-lifecycle atoms. Then any classical planner, including PSPR's Fast
Downward leaf solver, can solve it without knowing about methods. This is the
task-atom style of translation; Alford, Kuter, and Nau (2009,
`alford2009translating`) give a correctness proof for a restricted totally
ordered fragment, and the encoding below extends it with explicit ordering
guards for partial-order networks.

### 6.2 Formalisation

Let $U$ be a finite closure of task occurrences of $TN_0$ under method
application up to depth $d$. For every $t \in U$ introduce atoms

$$
\mathrm{pending}(t), \quad \mathrm{started}(t), \quad \mathrm{done}(t),
$$

and for every method instance $m\sigma$ introduce $\mathrm{chosen}(m\sigma)$.
The initial state contains $\mathrm{pending}(t)$ for every root task
$t \in T_0$ and none of the other task atoms.

**Primitive action.** For $a \in A$:

$$
\begin{aligned}
\mathrm{pre}(a') &=
\mathrm{pre}_a \wedge \mathrm{pending}(a)
\wedge \bigwedge_{u \prec a} \mathrm{done}(u)
\wedge \neg \mathrm{started}(a),\\
\mathrm{add}(a') &=
\mathrm{add}_a \cup \{\mathrm{started}(a), \mathrm{done}(a)\},\\
\mathrm{del}(a') &=
\mathrm{del}_a \cup \{\mathrm{pending}(a)\}.
\end{aligned}
$$

**Method expansion.** For $m\sigma$ decomposing $c$ with subtasks $T_m\sigma$:

$$
\begin{aligned}
\mathrm{pre}(m') &=
\phi_m\sigma \wedge \mathrm{pending}(c)
\wedge \bigwedge_{u \prec c} \mathrm{done}(u)
\wedge \neg \mathrm{started}(c),\\
\mathrm{add}(m') &=
\{\mathrm{started}(c), \mathrm{chosen}(m\sigma)\}
\cup \{\mathrm{pending}(s) \mid s \in T_m\sigma\},\\
\mathrm{del}(m') &= \{\mathrm{pending}(c)\}.
\end{aligned}
$$

**Method completion.** For each method instance $m\sigma$ of $c$:

$$
\begin{aligned}
\mathrm{pre}(\mathrm{finish}_{m\sigma}) &=
\mathrm{chosen}(m\sigma)
\wedge \bigwedge_{s \in T_m\sigma} \mathrm{done}(s)
\wedge \neg \mathrm{done}(c),\\
\mathrm{add}(\mathrm{finish}_{m\sigma}) &= \{\mathrm{done}(c)\},\\
\mathrm{del}(\mathrm{finish}_{m\sigma}) &= \varnothing.
\end{aligned}
$$

The `finish` action is what lets a parent ordering refer to the completion of
a compound task; without it, `done(c)` would never become true. The `chosen`
atom ensures that only the selected method's children are required to finish.
All task atoms are indexed by task occurrence: `pending(t)`, `started(t)` and
`done(t)` belong to the occurrence $t$, `chosen(m\sigma)` is shorthand for
`chosen(t,m\sigma)`, and `finish_{m\sigma}` is shorthand for
`finish_{t,m\sigma}` for the compound occurrence that selects the method. Two
occurrences of the same ground task therefore have independent lifecycles.

**Ordering inside a method.** For $u \prec_m v$, add $\mathrm{done}(u)$ to the
start precondition of $v$'s primitive operator or method operator. For a
compound $v$, the method operator's precondition already contains
$\bigwedge_{u \prec v}\mathrm{done}(u)$, so the same guard works. The
transitive closure of the method orderings is computed once when $U$ is built.

**Goal.** The classical goal is

$$
g \wedge \bigwedge_{t \in T_0} \mathrm{done}(t).
$$

If $P$ has no state goal, drop $g$.

**Proposition E.1.** Suppose the ground closure $U$ is finite and methods are
acyclic up to the depth bound. Then there is a bijection between HTN
decomposition plans of $P$ (up to the order of independent tasks) and
classical plans of the compiled problem. In particular, the compiled problem
is solvable iff $P$ is solvable within the closure.

The proof maps a decomposition step to the corresponding method operator, a
primitive action to its compiled action, and a compound completion to
`finish`; conversely, the `chosen` and `pending`/`done` atoms reconstruct the
method tree. The ordering guards are exactly the constraints that prevent a
task from starting before its predecessors finish. The finite-closure and
acyclicity assumptions are needed because otherwise the compiled task atoms
are infinite and the bijection is only defined for a bounded prefix.

Alford et al. (2009) use a different bookkeeping (`do_t`, `do_m`, and a
decomposition-level counter) for totally ordered task networks and prove a
one-to-one mapping under a bound on non-tail decomposition height. The
encoding above is more direct for HDDL's partial orders but uses more task
atoms; it should be presented as a task-lifecycle compilation in that
tradition, not as an exact reproduction of their translation.

### 6.3 Pseudo-code

```text
function COMPILE(P, depth):
    U ← GROUND_CLOSURE(P.TN0, P.M, depth)
    atoms ← P.F
    atoms ← atoms ∪ {pending(t), started(t), done(t) | t ∈ U}
    atoms ← atoms ∪ {chosen(t, mσ) | t ∈ U, mσ ∈ INSTANCES(t)}

    ops ← []
    for a in P.A:
        ops.append(PRIMITIVE_OPERATOR(a, U))
    for t in U where COMPOUND(t):
        for mσ in INSTANCES(t):
            ops.append(METHOD_OPERATOR(t, mσ, U))
            ops.append(COMPLETION_OPERATOR(t, mσ, U))

    init ← P.s0 ∪ {pending(t) | t ∈ P.TN0}
    goal ← CONJ(P.g, {done(t) | t ∈ P.TN0})
    return ⟨atoms, ops, init, goal⟩

function GROUND_CLOSURE(TN, M, depth):
    if depth = 0:
        return TASKS(TN)
    U ← TASKS(TN)
    for t in U where COMPOUND(t):
        for mσ in INSTANCES(t):
            U ← U ∪ GROUND_CLOSURE(TN_mσ, M, depth − 1)
    return U
```

`GROUND_CLOSURE` must key nodes by **task occurrence**, not by ground task
name. Two occurrences of the same ground task in different positions of the
network are distinct and need independent `pending`/`started`/`done` atoms;
collapsing them by name would lose task coverage and make the method-choice
bookkeeping unsound.

### 6.4 Integration in PSPR

- **New compilation module.** Add `src/solver/compilation/htn_to_classical.rs`.
  It can reuse `Universe`, `bindings_from` and `Domain::methods_for`; the
  output is a `fast_downward::Task` with variables for task atoms.
- **Search.** Call `fast_downward::best_first_search` directly, or feed the
  compiled task into `leaf.rs` as a leaf planning task. The latter reuses the
  FF/causal-graph heuristics and preferred operators.
- **Partitioning.** The compiled task can still be clustered: put all atoms of
  one task occurrence in the same incidence group and let `bisbm::cluster`
  partition the resulting graph. The groups must keep `pending`, `started`,
  `done` and `chosen` together, otherwise a leaf may see an inconsistent
  lifecycle.
- **Validation.** The compiled plan needs a projection back to primitive
  actions: `started`/`done` atoms are bookkeeping and must not appear in the
  final plan. The method operators and `finish` actions are compiled away.
- **Numeric fluents.** HDDL numeric tasks need a multi-valued extension of the
  task atoms; this is orthogonal to the hierarchy and should be treated
  separately.

### 6.5 Correctness and cost

The compilation is sound and complete for a finite closure. Its main cost is
the size of the closure: each method instance contributes task atoms, and each
method parameter adds a grounding dimension. The classical state space is at
worst $2^{|F| + 3|U| + |\mathrm{chosen}|}$, which is usually much larger than
the original state space. Heuristics can recover some of the structure through
`pending`/`done` atoms, but the method hierarchy is no longer explicit to the
planner. This approach is a useful reference implementation and a way to reuse
Fast Downward, not the best default for PSPR's partition-based architecture.

## 7. Approach F: exact method-choice and ordering encodings

### 7.1 Idea

Encode the existence of a decomposition of bounded depth and horizon as a
constraint problem. Method choices, task orderings and state transitions become
Boolean or integer variables; a SAT or ILP solver returns a model from which a
plan is extracted. Behnke, Höller, and Biundo (2019, `behnke2019bringing`)
give a compact SAT encoding for partial-order HTN planning; the same idea can
be expressed in the ILP infrastructure PSPR already uses for agglomeration
(`gilpin2013hierarchical`, `achterberg2009scip`, `goodlp`).

### 7.2 Formalisation

Fix a horizon $H$ and a decomposition depth $d$. Ground the task closure $U$
to depth $d$. For each compound occurrence $t \in U_C$ and method instance
$m\sigma \in \mathrm{inst}(t)$, introduce

$$
x_{t,m\sigma} \in \{0,1\}.
$$

For each task $t \in U$ and step $h \in \{0,\dots,H\}$, introduce a start
variable $y_{t,h} \in \{0,1\}$; for each fact $p \in F$ and step $h$,
introduce $f_{p,h} \in \{0,1\}$.

The core constraints are:

$$
\sum_{m\sigma \in \mathrm{inst}(t)} x_{t,m\sigma} = 1
\quad \text{for every } t \in U_C,
$$

$$
x_{t,m\sigma} \rightarrow
\bigvee_{h} y_{s,h}
\quad \text{for every } s \in T_m\sigma,
$$

$$
\sum_{h} y_{t,h} \le 1
\quad \text{for every } t \in U,
$$

and, for every ordering $u \prec v$,

$$
y_{v,h} \rightarrow
\bigvee_{h' < h} \mathrm{complete}(u,h').
$$

Here $\mathrm{complete}(u,h')$ means that $u$'s primitive action or its
method-completion step occurs at $h'$ and sets $\mathrm{done}(u)$. A method
expansion at step $h$ also requires its guard $\phi_m\sigma$ to hold in the
state at $h$; this is part of the transition constraints below. The sketch
omits the reverse implications that prevent a subtask from starting unless its
parent method is selected, and the reachability constraints that prevent an
unreachable occurrence from being chosen; a full encoding must include both.

The state constraints are the usual bounded-step transition constraints: if a
primitive task starts at $h$, its precondition holds at $h$, its add and delete
effects determine the facts at $h+1$, and all other facts are framed. The goal
requires $f_{p,H}=1$ for every $p \in g$ and completion of every root task.

If a numeric objective is wanted, replace the fixed $H$ by a makespan variable
$M$ and minimise $M$ subject to $y_{t,h} \rightarrow h < M$. For a
cost-sensitive variant, minimise

$$
\sum_{t \in U} \sum_{h} \mathrm{cost}(t)\, y_{t,h}
+ \lambda \sum_{\text{shared constraints}} \mathrm{violation},
$$

which is the natural bridge to PSPR's merge penalty. The second term is only
meaningful when the encoding is solved per cluster and the merge constraints
are represented explicitly.

**Proposition F.1.** If the encoding includes the complete transition relation,
the method-choice constraints and the ordering constraints, then a satisfying
assignment corresponds to an HTN plan of length at most $H$ and depth at most
$d$, and vice versa. The encoding is therefore complete for that bounded
fragment. Erol, Hendler, and Nau (1996) show why no polynomial bound is
expected in general.

### 7.3 Pseudo-code

```text
function SOLVE_EXACT(P, Hmax, depth):
    for H in 1..Hmax:
        φ ← ENCODE(P, H, depth)
        model ← SAT_SOLVE(φ)          # or ILP_SOLVE(φ)
        if model ≠ ⊥:
            return EXTRACT_PLAN(model)
    return fail

function ENCODE(P, H, depth):
    U ← GROUND_CLOSURE(P.TN0, P.M, depth)
    φ ← METHOD_CHOICE_CONSTRAINTS(U)
    φ ← φ ∧ ORDERING_CONSTRAINTS(U)
    φ ← φ ∧ TRANSITION_CONSTRAINTS(P, U, H)
    φ ← φ ∧ GOAL_CONSTRAINTS(P, U, H)
    return φ
```

The ILP variant replaces `SAT_SOLVE` by `good_lp`/SCIP; the SAT variant can
borrow the compact partial-order representation of `behnke2019bringing` to
avoid a quadratic number of ordering variables.

### 7.4 Integration in PSPR

- **New exact module.** Add `src/solver/exact/` with `encode.rs`, `sat.rs` and
  `ilp.rs`. Keep the encoding independent of the evaluation tree so it can be
  tested on small hand-written instances.
- **Per-cluster oracle.** Run the exact solver on a leaf cluster, with the
  interface atoms fixed to the values produced by the merge. This gives a
  complete local check and a way to measure how much the merge or the macro
  summary loses.
- **Merge solver.** A small exact solver can implement `MergeSolver` for
  clusters with few shared constraints; the current `SgplanMerger` remains the
  fast default.
- **Method-choice variables in the hierarchy ILP.** Add the $x_{t,m\sigma}$
  variables to `agglomeration::hierarchy` as an outer choice layer: first
  choose methods, then solve the scheduling ILP for the resulting subtask
  graph. This is a natural extension but requires a joint objective and a
  cycle check over method choices.

### 7.5 Correctness and cost

The exact encoding is sound and complete for the chosen horizon and depth. Its
cost is the size of the grounding closure times the horizon, plus the cost of
the SAT/ILP search. It scales poorly on the full IPC instances but is excellent
for small clusters, for validating approximations, and for producing
certificates on instances where the heuristic approaches fail. It is also the
cleanest way to test whether a failed merge is a limitation of the merge
operator or of the child plans.

## 8. Approach G: summarised effects and macro-operators

### 8.1 Idea

Replace a compound task by a **macro-operator** that summarises the effects of
one method's subtasks. If the method's subtasks are primitive, deterministic
and totally ordered, and no other action interleaves with them, the summary can
be exact. Otherwise it is an approximation, but a cheap one that can be used
inside the existing leaf encoding.

This is the macro-operator idea of Korf (1985, `korf1985macro`), applied to HTN
methods rather than to learned action sequences.

### 8.2 Formalisation

For a primitive action $a$, write the state transformer $\tau_a$. For a
formula $\psi$, the weakest precondition of $a$ with respect to $\psi$ is

$$
\mathrm{wp}(a,\psi) =
\mathrm{pre}_a \wedge
\psi[\mathrm{add}_a \leftarrow \top,\;
     \mathrm{del}_a \leftarrow \bot],
$$

where the substitution is applied to the literals of $\psi$; for a general
formula it is the usual precondition regression. For a sequence, define

$$
\mathrm{wp}(s_1;\dots;s_j,\psi) =
\mathrm{wp}(s_1,\mathrm{wp}(s_2,\dots,\mathrm{wp}(s_j,\psi))).
$$

For a method instance $m\sigma$ with a chosen linearisation $s_1,\dots,s_k$
of $T_m\sigma$, define

$$
\mathrm{pre}^{\mathrm{seq}}_{m\sigma} =
\phi_m\sigma \wedge \mathrm{wp}(s_1;\dots;s_k,\top),
$$

and the summary effects

$$
\mathrm{add}^{\mathrm{seq}}_{m\sigma} =
\bigcup_{i=1}^{k}
\left(
\mathrm{add}_{s_i}
\setminus
\bigcup_{j>i} \mathrm{del}_{s_j}
\right),
\qquad
\mathrm{del}^{\mathrm{seq}}_{m\sigma} =
\bigcup_{i=1}^{k}
\left(
\mathrm{del}_{s_i}
\setminus
\bigcup_{j>i} \mathrm{add}_{s_j}
\right).
$$

The second subtraction is necessary: a fact deleted by an early subtask and
re-added by a later one is true in the final state and must not appear in the
macro's delete set.

If a later subtask has a conditional effect or a precondition that depends on
an earlier effect, the summary must keep that condition as a conditional
effect or as an extra precondition; otherwise the macro is unsound.

**Proposition G.1 (exactness).** If all subtasks are primitive, deterministic,
totally ordered, and the macro is executed atomically (no external action
interleaves with its subtasks), then the macro

$$
\left\langle
\mathrm{pre}^{\mathrm{seq}}_{m\sigma},
\mathrm{add}^{\mathrm{seq}}_{m\sigma},
\mathrm{del}^{\mathrm{seq}}_{m\sigma}
\right\rangle
$$

is exactly equivalent to the sequence $s_1;\dots;s_k$.

The proof is induction on $k$: the weakest-precondition composition is the
standard regression rule, and the add/delete summary is the standard
frame-based composition of deterministic updates. Outside the stated
conditions the summary may be too weak (unsound), too strong (misses plans),
or simply a different linearisation of a partially ordered method. In those
cases it should be treated as a heuristic approximation, not as a replacement
for refinement.

For a partially ordered method, one can enumerate the linearisations that are
consistent with $\prec_m$ and take the disjunction of their summaries. The
number of linearisations can be exponential, so cap it and fall back to the
original method when the cap is exceeded. For recursive methods, summarise
bottom-up with memoisation and a depth bound.

### 8.3 Pseudo-code

```text
function SUMMARY(t, memo, depth):
    # One branch is shown; iterate over all methods and linearisations by
    # calling SUMMARY for each alternative and taking the union.
    if depth = 0:
        return ⊥
    if PRIMITIVE(t):
        return ⟨pre(t), add(t), del(t)⟩
    if t ∈ memo:
        return memo[t]

    alternatives ← ∅
    for (m, σ) in METHODS(t):
        for seq in LINEARISATIONS(T_mσ, CAP):
            pre ← φ_mσ
            add ← ∅
            del ← ∅
            for s in REVERSE(seq):
                summary ← SUMMARY(s, memo, depth − 1)
                if summary = ⊥:
                    break
                pre ← pre ∧ REGRESS(summary.pre, add, del)
                add ← (summary.add \ del) ∪ add
                del ← (summary.del \ add) ∪ del
            else:
                alternatives ← alternatives ∪ {⟨pre, add, del⟩}
    memo[t] ← alternatives
    return alternatives

function REGRESS(ψ, add, del):
    return ψ with every atom in add replaced by true
             and every atom in del replaced by false
```

The `for ... else` in the pseudo-code means that the alternative is kept only
when every subtask was summarised successfully.

### 8.4 Integration in PSPR

- **Leaf encoding.** In `leaf.rs`, replace the compound branch that emits an
  `executed`-only operator by one operator per summary alternative. The
  operator gets the summary's precondition and add/delete effects, plus the
  usual `executed` marker. The method name is still stored so that the plan can
  be refined later.
- **Incidence graph.** `graph.rs::task_atoms` must include the atoms of the
  subtask summaries; otherwise the leaf variables do not contain the atoms the
  macro writes and the merge cannot see the effects. This is the same change
  that the recursive tree and progression approaches need.
- **Option.** Add `EvaluationOptions::method_summary: SummaryMode` with
  variants `Off`, `PrimitiveOnly`, and `Recursive { depth }`. The default
  should remain `Off` until the exactness conditions are tested on the IPC
  domains; `PrimitiveOnly` is the safe first step.
- **Validation.** A macro plan must be expanded before `validate_plan`: the
  summary is not a method step. Keep the chosen method and its linearisation
  in the plan metadata so that the validator can re-execute the original
  subtasks.
- **Interaction with refinement.** A summary can be used as a fast
  approximation to choose methods and to get a state-goal estimate; the final
  plan is still produced by Approach C or B.

### 8.5 Correctness and cost

The summary is exact only under the conditions of Proposition G.1. It is cheap
to compute for shallow, primitive methods and can dramatically improve the
state-goal behaviour of the current leaves, because the macro writes the atoms
that the subtasks would write. It cannot represent arbitrary interleavings,
conditional effects with dependencies, or recursive methods exactly; those
cases need the original method and a refinement or exact solver. The cost is
the number of linearisations considered and the formula regression; both
should be capped.

## 9. Comparison and recommended roadmap

### 9.1 What each approach buys

- **A (eager expansion).** The cleanest semantics and a useful oracle, but the
  worst grounding cost. Keep it for testing on small instances and for
  generating all decompositions of a method-local cluster.
- **B (progression).** Complete for a bounded depth and the natural fallback
  when refinement or merging fails. It is a good per-cluster solver, not a
  replacement for the partitioner.
- **C (two-phase refinement).** The lowest implementation cost and the only
  approach that leaves the current abstract planner untouched. It is the
  right first step, but it should be measured as a liftability filter rather
  than assumed to be complete.
- **D (recursive trees).** The approach that best matches PSPR's architecture,
  but the benchmark showed no coverage gain over C+G and extra overhead; it was
  removed from the implementation. The design remains in section 5 as a record.
- **E (task-atom compilation).** A sound and complete reference for finite
  closures that reuses Fast Downward, but it discards most of PSPR's
  structural advantage. Use it as a baseline and for small clusters.
- **F (SAT/ILP).** The strongest completeness guarantee for a bounded
  horizon/depth and the best oracle for measuring approximation error. Use it
  on clusters, not on the full IPC suite.
- **G (macro summaries).** A cheap enhancement that can make the current
  leaves state-goal-aware for simple methods. It is exact only under strong
  conditions and should be paired with a refinement step.

### 9.2 Suggested implementation order

1. **Strict validation and method metadata (prerequisite).** Add a strict
   mode to `verify.rs` that recursively executes method subtasks, and expose
   the chosen method and origin of every plan step. Without this, no approach
   can be measured reliably.
2. **Approach C, phase 2.** Implement `REFINE_TASK`/`REFINE_NETWORK` after
   `evaluate` and use the method names already present in the abstract plan.
   Add a repair budget and report the lift rate on the 27-instance benchmark.
3. **Approach G, primitive-only summaries.** Enable summaries for methods
   whose subtasks are all primitive and totally ordered; keep the original
   method as a fallback. Measure how many of the 12 `violated` cases become
   `satisfied`.
4. **Approach B or F as a fallback.** Use progression or a small exact
   encoding for method subtrees where the bounded refinement fails. This closes
   the completeness gap without making the default path expensive.
5. **Approaches A and E as baselines.** Keep them behind feature flags and use
   them to validate the refinement results on small instances.

### 9.3 Open issues

- **Guards and recursion.** A guard is a zero-effect action, but recursive
  methods can create infinite guard chains. The implementation needs a
  method-call cycle check or a depth bound; HDDL itself does not require
  acyclicity.
- **Interface state.** Approach D's memo key must include the interface atoms,
  not the whole state, or it degenerates to global search. Choosing the
  interface is a tree-decomposition problem; the current incidence graph gives
  a good starting point but not an optimal one.
- **Merge incompleteness.** The current merger can only interleave fixed child
  plans; it cannot replan a child to establish a shared atom. Any recursive
  tree inherits this limitation. The exact fallback in Approach F is the
  cleanest way to detect it.
- **Numeric fluents.** HDDL numeric tasks and functions need a multi-valued
  extension of task atoms and summaries. The formula translation in
  `formula.rs` already handles the propositional fragment; the numeric
  semantics should be specified separately.
- **Plan quality.** The approaches above mostly target feasibility. HTN
  planning often has cost or makespan objectives; the SAT/ILP encoding can
  express them, while the refinement and macro approaches need a cost-aware
  merge.
- **Validation of partial orders.** A primitive plan is a linearisation, but a
  decomposition tree also has a partial-order structure. The validator should
  check both the linearisation and the tree, not only the final sequence.

## 10. Citation map

All keys below are defined in [`references.bib`](references.bib). The `annote`
fields there record where each work is used in the repository.

| Key | Work | Used for |
| --- | --- | --- |
| `erol1994umcp` | Erol, Hendler, and Nau (1994), *UMCP: A Sound and Complete Procedure for Hierarchical Task Network Planning* | Task-network semantics, decomposition relation, soundness |
| `erol1996complexity` | Erol, Hendler, and Nau (1996), *Complexity Results for HTN Planning* | Why depth/horizon bounds are necessary |
| `ghallab2004automated` | Ghallab, Nau, and Traverso (2004), *Automated Planning: Theory and Practice* | Textbook state-transition and HTN definitions |
| `nau2003shop2` | Nau et al. (2003), *SHOP2: An HTN Planning System* | Agenda-based progression, ordered decomposition |
| `alford2009translating` | Alford, Kuter, and Nau (2009), *Translating HTNs to PDDL* | Task-atom compilation, restricted totally ordered fragment |
| `behnke2019bringing` | Behnke, Höller, and Biundo (2019), *Bringing Order to Chaos* | Compact SAT encoding of partial-order HTN planning |
| `korf1985macro` | Korf (1985), *Macro-Operators: A Weak Method for Learning* | Summarised effects and macro-operators |
| `holler2020hddl` | Höller et al. (2020), *HDDL* | Input language and task-network semantics |
| `holler2021panda` | Höller et al. (2021), *The PANDA Framework for Hierarchical Planning* | Complete HTN baseline |
| `behnke2020grounding` | Behnke et al. (2020), *On Succinct Groundings of HTN Planning Problems* | Grounding and instance sizes |
| `hsu2006sgplan` | Hsu et al. (2006), *Constraint Partitioning in Penalty Formulations* | Evaluation-tree partitioning and merge |
| `gilpin2013hierarchical` | Gilpin, Nijssen, and Davidson (2013), *Formalizing Hierarchical Clustering as ILP* | Merge-level ILP |
| `braun2023bisbm` | Braun et al. (2023), *BiSBM* | Spectral clustering of the task/constraint graph |
| `helmert2006fastdownward` | Helmert (2006), *The Fast Downward Planning System* | Leaf planning tasks and heuristics |
| `hoffmann2001ff` | Hoffmann and Nebel (2001), *The FF Planning System* | Relaxed planning graph heuristics |
| `achterberg2009scip` | Achterberg (2009), *SCIP* | ILP backend for exact encodings |
| `goodlp` | `good_lp` crate | Rust ILP modelling layer |
