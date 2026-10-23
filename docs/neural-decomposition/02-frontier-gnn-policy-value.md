# Design 2: Frontier GNN policy and value for `search_partial`

Status: implemented in `src/solver/evaluation_tree/decomposition/` at the
initial frontier-GNN scope. This is the selected search-policy design from
[the longlist](README.md). It learns which ready task to expand next in a
partial-order method network. The symbolic recursion, state transition,
backtracking, cycle check, and result cap remain unchanged.

## 1. Summary

Partial-order refinement in `search_partial_inner` currently iterates ready
tasks in index order:

```text
for index in 0..tasks.len():
    if not ready(index): continue
    refine_task(tasks[index])
```

The index order is arbitrary. The same decomposition can be found after very
different amounts of search depending on which ready task is expanded first,
especially when several methods fail late.

This design adds a **frontier graph neural policy** with a **value head** over
the residual task network. At every recursion node it scores the ready tasks:

\[
\pi_\theta(i\mid F), \qquad i\in \mathrm{Ready}(F),
\]

and estimates the remaining cost:

\[
V_\theta(F)\approx \text{cost-to-go}(F).
\]

The first implementation uses the policy only to order ready tasks. The value
is used for ordering and diagnostics, not for hard pruning. This preserves
the current enumeration semantics up to `MAX_REFINEMENTS`.

## 2. Why this is the selected approach

The longlist compared:

1. a handcrafted linear heuristic;
2. an MLP over a flattened frontier;
3. this frontier GNN policy with a value head;
4. a transformer over a sequence of remaining tasks;
5. an LLM policy with a symbolic verifier.

The frontier is naturally a graph:

- precedence edges say which tasks are ready;
- shared atoms connect tasks that can interfere;
- method-origin edges group subtasks from the same decomposition;
- the current state determines whether a task’s methods are applicable.

A sequence representation (approach 4) reintroduces an arbitrary order and
makes precedence constraints harder to express. An LLM (approach 5) adds
latency and nondeterminism without addressing the combinatorial structure.
The GNN policy shares node/edge features with the method ranker and scales to
variable ready sets.

## 3. Formal setting

At a partial-order refinement node, let

\[
F=(T_{\text{rem}},\prec,s,d)
\]

be the frontier, where:

- \(T_{\text{rem}}\) is the set of unexpanded subtasks;
- \(\prec\) is the precedence relation;
- \(s\) is the current state;
- \(d\) is the recursion depth.

The ready set is

\[
\mathrm{Ready}(F)=\{t\in T_{\text{rem}}\mid
\nexists u\in T_{\text{rem}} : u\prec t\}.
\]

An action chooses one ready task \(t_i\). The transition is the symbolic
`refine_task` call, which returns zero or more refined branches with new
states. The policy does not choose a method; that is the job of the method
ranker from [Design 1](01-state-conditioned-relational-method-ranker.md).

### 3.1 What the policy does not do

- It does not remove ready tasks from the search. Every ready task remains
  available.
- It does not change `refine_task`, state transitions, or the visited set.
- It does not validate the final plan.
- It does not prune branches with the value head in the first implementation.

The first version is therefore an **ordering heuristic**, not a new planner.

## 4. Frontier graph

### 4.1 Node types

| Type | Count | Meaning |
| --- | ---: | --- |
| `remaining` | \(|T_{\text{rem}}|\) | A subtask that is not yet expanded |
| `atom` | \(A\) | Atom touched by a remaining task or its methods |
| `global` | 1 | Frontier summary and search context |

Optional `done` nodes or completed-task context can be included if they help
the value head, but they are not needed for the first version.

### 4.2 Edge types

| Edge | Direction | Meaning |
| --- | --- | --- |
| `precedes` | directed | \(u\) must execute before \(v\) |
| `shares_atom` | bidir. | Two tasks touch the same atom |
| `same_method` | bidir. | Tasks originate from the same method network |
| `task -- atom` | bidir. | Task precondition/effect incidence |
| `atom -- state` | bidir. | Atom truth in the current state |
| `global -- *` | bidir. | Global readout |

The ready mask is a node feature, not an edge. This lets the same encoder
represent a task that is ready now and blocked later.

### 4.3 Node features

For each remaining task:

- ready flag;
- number of remaining predecessors and successors;
- total remaining task count and normalized position in the network;
- primitive/compound flag;
- arity and task-name hash;
- number of method groundings and number of applicable methods;
- fraction of the task’s precondition atoms true in the current state;
- number of shared atoms with other remaining tasks;
- whether the task is the originally preferred/current task;
- recursion depth and remaining refinement budget;
- visit count / number of times this task was tried at this frontier
  (optional search feature, only if available from the trace).

Do **not** use the raw task index as a primary feature. It is an arbitrary
order and would teach the policy to reproduce the current baseline.

### 4.4 Global features

- number of ready tasks;
- remaining task count;
- depth and remaining budget;
- state size;
- number of unsatisfied goal atoms, if a goal exists;
- number of precedence edges and connected components in the frontier graph;
- current merge penalty or cluster cost, if available;
- search statistics: nodes expanded so far and result count.

## 5. Architecture

Use a shared relational GAT encoder:

1. **Input encoders.** Separate MLPs for task, atom, and global nodes, plus a
   one-hot node type.
2. **Message passing.** Three layers, four heads, hidden width 96–128.
   Precedence and shared-atom edges use separate relation projections or
   biases.
3. **Residual normalization.** `LayerNorm(h + MLP(aggregate(h)))` with
   LeakyReLU and dropout during training.
4. **Policy head.** A per-node scalar score for remaining-task nodes:

\[
u_i = \operatorname{MLP}_\pi(h_i),
\qquad
\pi_\theta(i\mid F)=
\frac{\exp u_i}{\sum_{j\in \mathrm{Ready}(F)}\exp u_j}.
\]

Mask all non-ready and non-remaining nodes before the softmax.
5. **Value head.** Global mean/max pooling concatenated with the global
   feature vector, then an MLP:

\[
\hat V_\theta(F)=\operatorname{MLP}_V\left(
[\operatorname{mean}(h_i) \| \max(h_i) \| h_g]
\right).
\]

The value target can be the remaining number of primitive steps, remaining
`refine_task` calls, or a normalized search cost. Use one target consistently
across experiments.

### 5.1 Optional action-value head

A Q-head over ready tasks can support PUCT:

\[
a^\star = \arg\max_{a\in\mathrm{Ready}}
\left(Q(a)+c\,P(a)\frac{\sqrt{N}}{1+N(a)}\right).
\]

This is a later extension. The first implementation uses policy-ordered DFS
so the existing enumeration and result cap remain easy to reason about.

## 6. Training

### 6.1 Trace collection

Instrument `search_partial_inner` behind an offline tracing flag. For each
recursion node with more than one ready task, record:

- the task list, remaining mask, precedence relation, and state hash;
- the ready set;
- the task expanded by the successful C+G refinement;
- whether each ready choice leads to a complete refinement from this node;
- the number of recursive calls below each choice;
- the remaining primitive steps and final plan length;
- whether the whole refinement succeeded.

If exhaustive enumeration is affordable on a small network, compute the
successful choices directly. Otherwise, use the successful trace as a
positive and the failed alternatives as negatives.

### 6.2 Labels and symmetry

Partial-order plans have many equivalent linearisations. Do not assume the
single task chosen by one successful trace is the only correct action.

Use one of:

- a set-valued target over all choices that can complete the refinement;
- a soft target proportional to the number of solutions below each ready
  task;
- a pairwise preference loss from successful versus failing choices.

If two choices are symmetric and both succeed, train with a uniform target
over them rather than tie-breaking by task index.

### 6.3 Stage A: policy imitation

Train the policy with listwise cross-entropy:

\[
\mathcal L_\pi=
-\sum_{i\in \mathrm{Ready}(F)} y_i \log \pi_\theta(i\mid F),
\]

where \(y\) is the normalized target over successful choices.

Train the value head simultaneously with a regression loss:

\[
\mathcal L_V=
\left(\hat V_\theta(F)-V^\star(F)\right)^2.
\]

Use a curriculum:

1. small frontiers with at most 4 ready tasks;
2. frontiers with at most 12 ready tasks;
3. full partial-order networks, including hard negatives.

Split by domain. Do not split by instance.

### 6.4 Stage B: search bootstrapping

Once the policy is reasonable, run a bounded policy-guided search on the
training domains and record:

- the visit distribution over ready tasks;
- whether the node led to a solution;
- the remaining cost-to-go;
- the best child value.

Train the policy toward the visit distribution and the value toward the
realized outcome, in the style of AlphaZero-style policy improvement. Because
the symbolic environment is deterministic and the networks are small, this is
feasible without a simulator.

A simpler alternative is REINFORCE with a terminal reward:

\[
R =
\begin{cases}
+\text{plan quality} & \text{refinement succeeds},\\
-\text{penalty} & \text{refinement fails},
\end{cases}
\]

optionally penalizing the number of expanded nodes or wall-clock cost. This
has higher variance but requires less search instrumentation.

### 6.5 Value targets

Use a single target per experiment:

- remaining primitive action count;
- remaining method applications;
- remaining `refine_task` calls;
- \(\log(1+\text{remaining nodes})\);
- a weighted combination of plan length and search cost.

Normalize per frontier or per domain. Report value calibration and rank
correlation, not only MSE.

## 7. Integration in `search_partial`

The change should be narrow:

```text
fn search_partial(...):
    remaining = all true
    results = []
    order = policy.order_ready(frontier) or ready_indices_ascending
    search_partial_inner(..., order, ...)
```

Inside `search_partial_inner`, replace the fixed loop:

```text
for index in 0..tasks.len():
```

with:

```text
let order = if let Some(policy) = policy {
    policy.order_ready(&context).unwrap_or_else(ready_ascending)
} else {
    ready_ascending
};
for index in order:
    if not remaining[index]: continue
    if not ready(index): continue
    ...
```

Validation rules:

- `order_ready` must return a permutation of the ready set; if it does not,
  fall back to ascending order.
- Keep every ready branch; do not filter by the policy.
- Keep `visited`, depth, and `MAX_REFINEMENTS`.
- Keep the symbolic `refine_task` call and its method/precondition logic.
- The value head may be used to sort branches by predicted cost, but not to
  discard them until an admissible bound or validated lower bound exists.

### 7.1 Composition with the method ranker

The two models are orthogonal:

- **Frontier policy:** which ready task to expand next.
- **Method ranker:** which grounded method to try for that task.

Evaluate four configurations:

1. baseline: index order + preferred/subtask count;
2. method ranker only;
3. frontier policy only;
4. both.

The combined system should be strictly optional. If either model is absent or
invalid, the corresponding deterministic baseline is used.

### 7.2 Module layout

```text
src/solver/evaluation_tree/refine/
    mod.rs
    method_features.rs
    method_ranker.rs
    search_features.rs
    search_policy.rs
```

`hierarchical.rs` should depend only on the `SearchPolicy` trait, not on Burn.
The model and weight loading live behind the optional `neural` feature. Use an
environment variable such as `PSPR_SEARCH_POLICY`, matching the existing
`PSPR_MERGE_MODEL` pattern.

`RefinementOptions` is `Copy`; pass the optional policy separately or use a
`RefinementSession` that owns the model handles and borrows the options.

## 8. Cost

Let \(N\) be the number of remaining tasks plus atom nodes. Three GAT layers
cost \(O(LN d^2)\) per frontier. For small method networks this is well below
a millisecond, but `search_partial_inner` can be called thousands of times.

Mitigations:

- skip the policy when the ready set has size 0 or 1;
- skip the value head when there is only one branch;
- cache the schema/task features by task identity;
- compute the frontier graph incrementally after each task removal;
- cap the number of tasks in a network that uses the policy;
- run the policy only under a `--neural-search` flag while collecting
  baseline timings.

The policy should not be invoked for every recursive call if the graph
construction dominates. Measure feature extraction and inference separately.

## 9. Evaluation

### 9.1 Offline metrics

- top-1 agreement with successful choices;
- top-k recall of successful choices;
- mean reciprocal rank;
- value MSE, Spearman/Pearson correlation with realized cost-to-go;
- calibration by frontier size.

### 9.2 End-to-end metrics

Run C+G on the comparison suite and report:

- solved instances and goal satisfaction;
- primitive plan length and method count;
- number of `refine_task` calls and `search_partial_inner` nodes;
- wall-clock median/mean and timeout count;
- performance on partial-order stress cases: `Transport`, `PO_Transport`,
  `Satellite-GTOHP`, `PO_Satellite`, `Blocksworld-GTOHP`, and
  `Logistics-Learned-ECAI-16`.

Compare with:

- ascending index order;
- random ready-task order with the same seed and budget;
- a handcrafted heuristic (ready count, precedence degree, goal relevance);
- policy only, value only, and both;
- method ranker only and method ranker plus policy;
- an oracle that tries the successful trace’s next ready task first.

### 9.3 Ablations

- remove precedence edges;
- remove shared-atom edges;
- remove state features;
- remove the value head;
- replace the GNN with a sequence transformer;
- remove policy and use only value-based branch ordering;
- test leave-one-domain-out and cross-domain transfer.

## 10. Risks and mitigations

- **Partial-order symmetry.** Many linearisations are equivalent. Use
  set-valued or soft labels; do not penalize a different but successful order.
- **Policy collapse.** The model may memorize a domain-specific task order.
  Exclude raw indices, split by domain, and test cross-domain transfer.
- **Value overestimation.** A learned value is not an admissible heuristic.
  Do not prune with it. If pruning is added later, require a proven bound or a
  conservative cutoff with fallback.
- **Compounding recursion.** A locally promising ready task can fail deep.
  Train on complete-refinement outcomes and include hard negatives.
- **Latency.** Gate the policy by ready-set size and cache features. Measure
  the end-to-end effect, not only node counts.
- **Safety.** The exact validator remains authoritative. A policy that changes
  the search order must not be able to change the set of accepted plans.

## 11. Pseudo-code

```text
function SEARCH_PARTIAL(tasks, predecessors, state):
    remaining = [true; len(tasks)]
    results = []

    function VISIT(state, steps):
        if results.len() >= MAX_REFINEMENTS:
            return
        if no remaining task:
            results.append(steps, state)
            return

        ready = [i for i in tasks if remaining[i]
                 and no remaining predecessor of i]

        if policy is None or len(ready) <= 1:
            order = ready in ascending index order
        else:
            context = build_frontier_graph(tasks, remaining, predecessors, state)
            order = policy.order_ready(context)
            if order is not a permutation of ready:
                order = ready in ascending index order

        for index in order:
            for refined in REFINE_TASK(tasks[index], state):
                remaining[index] = false
                VISIT(refined.state, steps + refined.steps)
                remaining[index] = true
                if results.len() >= MAX_REFINEMENTS:
                    return

    VISIT(state, [])
    return results
```

The only learned decision is the `order`. All symbolic transitions,
backtracking, and caps remain.

## 12. References

`silver2018alphazero` for policy/value search; `sutton1999options` and
`dietterich2000maxq` for hierarchical temporal abstraction;
`velickovic2018gat` for graph attention and `schlichtkrull2018rgcn` for typed
relational message passing. `erol1994umcp`, `nau2003shop2`, and
`holler2020hddl` provide the HTN/HDDL background. `kambhampati2024llmmodulo`,
`liu2023llmp`, and `ahn2022saycan` are the proposal-plus-verifier alternatives
that this design deliberately does not use as the default.
