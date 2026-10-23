# Design 1: State-conditioned relational method ranker

Status: proposed. This is the selected method-ranking design from
[the longlist](README.md). It is a narrow neural component for
`refine_task`; it never replaces symbolic applicability checking, state
transition, or validation.

## 1. Summary

Train a heterogeneous relational graph attention network to score the grounded
method instances that can decompose a compound task in the current state. The
ranker orders candidates before `refine_task` recurses into their subtasks.
The existing preferred-method priority, precondition check, depth limit,
backtracking, `MAX_REFINEMENTS`, and strict validator remain unchanged.

The model receives:

- the query task \(t\);
- the current state \(s\);
- the grounded candidates \(\mathcal C(t,s)=\{(m,\sigma)\}\), where \(m\) is
  a method schema and \(\sigma\) a binding;
- the method/subtask/atom schema graph and the state atoms.

It predicts a score

\[
s_\theta(c \mid t,s), \qquad c=(m,\sigma)\in\mathcal C(t,s),
\]

and returns the candidates in descending score order. A listwise softmax is
used during training, but inference only needs a permutation.

## 2. Why this is the selected approach

The longlist compared:

1. a homogeneous task–method GAT;
2. this state-conditioned heterogeneous relational GAT;
3. a set transformer over grounded candidates;
4. a graph transformer with per-candidate cross-attention;
5. an RGAT plus a text encoder over HDDL names.

The selected model keeps the structural inductive bias that the other cheap
baselines lack, without introducing a language model:

- `task --method` means “this schema can decompose the task”;
- `candidate --method` means “this grounding instantiates the schema”;
- `method --subtask` encodes the decomposition shape;
- `method --atom` and `subtask --atom` encode preconditions and effects;
- `atom --state` encodes current truth.

A flat candidate set (approach 3) loses the relations between subtasks and
atoms. A text hybrid (approach 5) adds latency and tokenization without a
clear benefit on the formal HDDL benchmark. The selected approach also shares
its node/feature vocabulary with the search-policy design, so the two models
can later share an encoder.

## 3. Formal setting

Let a compound task be \(t=c(\vec{x})\), a method schema be
\(m=\langle \phi_m, TN_m\rangle\), and a substitution be \(\sigma\). The
grounded candidate is applicable when

\[
s \models \phi_m\sigma.
\]

`refine_task` currently:

1. tries a primitive action with the same name first;
2. grounds all method instances with `ground_methods`;
3. sorts by `(not preferred, number of subtasks)`;
4. skips candidates whose precondition does not hold;
5. recursively refines the chosen method network;
6. keeps alternatives until the result cap.

The ranker replaces only step 3 for the applicable candidates. It may also
use the preferred flag as a feature, so the current behaviour remains a
special case when the model is absent.

### 3.1 What the ranker does not do

- It does not choose raw object bindings without grounding; binding
  enumeration stays in `Universe::assignments`.
- It does not skip the precondition check. Inapplicable candidates are
  filtered symbolically before ranking.
- It does not decide when to stop recursion or how many alternatives to keep.
- It does not validate the final plan.
- It does not change the plan representation or `RefinementStep`.

This keeps soundness and completeness arguments local: a ranking can change
which solution is found first, but it cannot make an invalid decomposition
valid.

## 4. Graph construction

The graph is built once per `(task, state, candidate set)` call.

### 4.1 Node types

| Type | Count | Meaning |
| --- | ---: | --- |
| `task` | 1 | The query compound task |
| `global` | 1 | Merge/refinement context |
| `method` | \(M\) | Method schemas that can decompose the task |
| `candidate` | \(C\) | Grounded method instances \((m,\sigma)\) |
| `subtask` | \(S\) | Subtasks of the method schemas |
| `atom` | \(A\) | Precondition/effect atoms mentioned by the candidates |
| `state` | \(K\) | Optional state-atom nodes, or atom truth features |

The preferred implementation uses shared `method` schema nodes and separate
`candidate` nodes. That avoids duplicating the method structure once per
grounding while still allowing binding-specific features.

### 4.2 Edge types

| Edge | Direction | Meaning |
| --- | --- | --- |
| `task -- method` | bidir. | Method schema can decompose the task |
| `candidate -- method` | bidir. | Grounding instantiates the schema |
| `task -- candidate` | bidir. | Candidate is being considered for the task |
| `method -- subtask` | bidir. | Decomposition member |
| `subtask -- atom` | bidir. | Precondition/effect atom |
| `method -- atom` | bidir. | Method-level precondition/effect atom |
| `atom -- state` | bidir. | Atom is true/false in the current state |
| `subtask -- subtask` | directed | Ordering inside the method network |
| `method -- method` | directed | Recursive call relation |
| `global -- *` | bidir. | Global context readout |

The adjacency is dense and typed. For \(N<200\) nodes this is cheap; the
implementation should still cap candidates and atoms to keep \(N\) bounded.

### 4.3 Candidate features

Each candidate node is described by an owned snapshot:

- method name hash and schema index;
- preferred-method flag;
- number of bound parameters and number of newly bound variables;
- type-compatibility score between bound objects and parameter types;
- precondition satisfaction fraction;
- number of satisfied and unsatisfied precondition atoms;
- equality/inequality constraints and whether they hold;
- number of subtasks, number of primitives, and number of compound subtasks;
- number of subtasks that already appear in the current plan;
- grounding ordinal within `ground_methods` (weak tie-breaker only);
- if available, a symbolic prior such as grounding cost or method usage count.

Do not use the candidate index as a primary feature. It is an arbitrary
grounding order and would make the model sensitive to `Universe` enumeration.

### 4.4 Task and method features

Task:

- arity, predicate hash, and name hash;
- number of method schemas and grounded candidates;
- recursive-task flag;
- current depth and remaining refinement budget;
- number of satisfied atoms in the task’s incidence set.

Method schema:

- parameter count and task-term count;
- number of subtasks, split into primitive and compound;
- number of precondition/effect atoms;
- ordering type (total vs. partial) and edge count;
- recursion depth and whether the method calls its own task;
- historical success/usage rate if the trace set is available;
- preferred-method flag.

### 4.5 Atom and state features

Atom nodes carry predicate hash, arity, polarity, and whether the atom is a
precondition or an effect. The global/state representation adds:

- state size;
- fraction of candidate precondition atoms true;
- number of unsatisfied goal atoms if a goal is available;
- number of remaining tasks and current depth;
- candidate count and cap pressure.

## 5. Architecture

Use a small relational GAT:

1. **Input encoders.** Separate MLPs for task, method, candidate, subtask,
   atom, and global nodes. Concatenate a one-hot node type with the features.
2. **Relational message passing.** Three layers. Each edge type has its own
   query/key/value projections (or at least its own relation bias). Four
   attention heads and hidden width 96–128 are sufficient.
3. **Residual normalization.** `h' = LayerNorm(h + MLP(aggregate(h)))`, with
   LeakyReLU and dropout during training.
4. **Global readout.** Mean/max pooling over all nodes, concatenated with the
   task and state summary.
5. **Candidate head.** For each candidate \(c\),

\[
z_c = \left[h_c \,\|\, h_t \,\|\, h_g \,\|\, \phi_c\right],
\qquad
s_c = \operatorname{MLP}_\theta(z_c).
\]

6. **Listwise output.** Training uses

\[
p(c\mid t,s)=\operatorname{softmax}_{c\in\mathcal C_{\text{app}}} s_c.
\]

At inference, return the candidates sorted by \(s_c\). The preferred method,
subtask count, and original index are deterministic tie-breakers.

### 5.1 Optional candidate pairwise head

A second head can score pairs \((c_i,c_j)\) for a RankNet-style loss. It is
useful when several candidates lead to a solution and pointwise labels are
ambiguous. It is disabled in the first experiment.

## 6. Training

### 6.1 Trace collection

Instrument `refine_task` behind an offline tracing flag. For every compound
task call with more than one applicable candidate, record:

- the task and state hash;
- the candidate snapshots and whether each precondition held;
- the method/binding that the successful C+G refinement chose;
- whether each candidate led to a complete refinement from this state;
- the number of recursive calls and primitive steps below each candidate;
- the final plan length and whether the overall refinement succeeded.

A candidate is a **positive** if it participates in at least one complete
refinement from the recorded state. If several candidates are positive, use a
soft target proportional to their success counts or inverse cost. A candidate
that passes the precondition but cannot complete is a **hard negative**.

### 6.2 Stage A: imitation

Train with listwise cross-entropy:

\[
\mathcal L_{\text{imitate}}
  = -\sum_{i} y_i \log p_\theta(c_i\mid t,s),
\]

where \(y\) is the normalized positive mass. Include the current baseline
order as a weak prior only as a feature, not as a hard label.

Use a curriculum:

1. tasks with at most 8 candidates;
2. tasks with at most 32 candidates;
3. all tasks, with hard-negative mining.

Split by domain. Do not split by task instance: methods and object names leak
across instances of the same domain.

### 6.3 Stage B: decision-focused fine-tuning

After imitation, freeze the symbolic refiner and rank candidates with the
model. For each training task, run `refine_task` with a fixed budget and
record:

- whether refinement succeeded;
- the number of recursive calls until the first success;
- the final plan length;
- wall-clock time.

Use a ranking surrogate or REINFORCE to increase the probability of
candidates that reduce those costs. A simple and robust alternative is
pairwise preference fine-tuning: for pairs where candidate A succeeded and B
failed, add a margin loss

\[
\mathcal L_{\text{pair}}
  = \max(0, m - s_A + s_B).
\]

This avoids differentiating through the recursive refiner while still using
post-search outcomes.

### 6.4 Negative mining

Hard negatives are essential because most grounded candidates are obviously
inapplicable or obviously worse:

- same method schema, different binding;
- applicable method that fails deeper in the network;
- preferred method that fails while a lower-priority method succeeds;
- methods with many subtasks that look cheap by the current sort but are not.

Balance the candidate set per task; otherwise the model learns “the preferred
method is usually right” and ignores the graph.

## 7. Integration in `refine_task`

The intended call sequence is:

```text
methods = ground_methods(...)
methods.retain(precondition holds)
order = ranker.rank(task, state, methods)
for instance in order:
    refine_network(...)
```

Keep the existing primitive-action branch before method ranking. Keep the
`MAX_REFINEMENTS` cap and the `visited` cycle check. If the ranker returns
`None`, a vector of the wrong length, or a non-finite score, use the current
sort exactly.

`RefinementOptions` is `Copy`; do not add `Arc<dyn MethodRanker>` to it.
Pass the optional ranker as a new argument, or introduce a
`RefinementSession`:

```rust
pub struct RefinementSession<'a> {
    pub options: &'a RefinementOptions,
    pub method_ranker: Option<&'a dyn MethodRanker>,
    pub search_policy: Option<&'a dyn SearchPolicy>,
}
```

The model should live behind the optional `neural` feature. A model path can
be supplied through an environment variable such as
`PSPR_METHOD_RANKER`, matching the existing `PSPR_MERGE_MODEL` pattern. The
default build and the default `pspr` invocation must not require Burn.

### 7.1 Feature extraction location

Put graph extraction and the Burn model in a new module, for example:

```text
src/solver/evaluation_tree/refine/
    mod.rs
    method_features.rs
    method_ranker.rs
    search_features.rs
    search_policy.rs
```

The symbolic refiner should depend on the trait, not on Burn types. This keeps
the model optional and makes the set-transformer ablation possible without
touching `hierarchical.rs`.

## 8. Cost

Let \(N=M+C+S+A+2\) be the number of graph nodes and \(d\) the hidden width.
Three relational GAT layers cost \(O(LN d^2)\), which is well below a
millisecond for \(N<200\) on a CPU. Candidate grounding and precondition
checking dominate most calls.

Mitigations:

- skip the ranker when there is zero or one applicable candidate;
- cache schema features by method schema;
- cap candidates (for example 256) and keep the top candidates by the current
  deterministic order before running the model;
- run the model only for compound tasks whose candidate count exceeds a
  threshold;
- batch multiple `refine_task` calls only if the recursion structure allows
  it; otherwise measure the per-call overhead directly.

## 9. Evaluation

### 9.1 Offline ranking metrics

- top-1 accuracy over applicable candidates;
- top-5 accuracy and mean reciprocal rank of the first successful candidate;
- oracle-normalized rank;
- calibration of the preferred-method baseline versus the model.

### 9.2 End-to-end metrics

Run the full C+G pipeline on the comparison suite and report:

- solved instances and goal satisfaction;
- primitive plan length and method count;
- number of `refine_task` calls and recursive calls;
- wall-clock time, median and mean, with timeouts separate;
- improvement on the known failure cases: `Barman-BDI`, `PCP_1`,
  `Freecell`, `Minecraft-Player`, and `Monroe_PO_1`.

Compare with:

- the current preferred/subtask-count order;
- random candidate order with the same budget;
- the set-transformer ablation;
- the ranker with the preferred feature removed;
- a ranker trained only on successful traces versus one trained with hard
  negatives.

### 9.3 Ablations

- remove state-conditioned atom features;
- remove typed edges and use a homogeneous GAT;
- remove candidate binding features;
- keep only the schema graph;
- replace listwise loss with pointwise regression;
- test leave-one-domain-out and per-domain transfer.

## 10. Risks and mitigations

- **Label ambiguity.** Different method sequences can yield the same plan.
  Use set-valued positives and pairwise losses; do not force a single label.
- **Distribution shift.** Candidate counts and method schemas vary by domain.
  Split by domain and report leave-one-domain-out results.
- **Shortcut learning.** The preferred-method flag and subtask count are
  strong baselines. Ablate them to confirm the graph is used.
- **Compounding recursion.** A locally good method can fail deeper. Train on
  complete-refinement outcomes, not only immediate precondition success.
- **Model latency.** Gate the ranker by candidate count and cache schema
  features.
- **Safety.** Keep the exact validator and fallback. A learned ranker must
  never be able to accept an invalid decomposition.

## 11. Pseudo-code

```text
function REFINE_TASK(task, state, depth):
    if primitive action applies:
        return primitive refinement

    candidates = GROUND_METHODS(task)
    applicable = [c for c in candidates if HOLDS(c.precondition, state)]

    if ranker is None or len(applicable) <= 1:
        order = CURRENT_ORDER(applicable)
    else:
        scores = ranker(task, state, applicable)
        if scores is invalid:
            order = CURRENT_ORDER(applicable)
        else:
            order = SORT_DESC(scores, tie_break=(preferred, subtasks, index))

    results = []
    for c in order:
        for refined in REFINE_NETWORK(c.method.network, c.bindings, state):
            results.append(make_method_step(c, refined))
            if len(results) >= MAX_REFINEMENTS:
                return results
    return results
```

The only difference from the current implementation is the `order`
computation. All symbolic rules remain in place.

## 12. References

`velickovic2018gat` for multi-head graph attention; `schlichtkrull2018rgcn`
for typed relational message passing; `xu2019gin` for expressiveness
considerations; `kipf2017gcn` for the GCN baseline. `erol1994umcp`,
`nau2003shop2`, and `holler2020hddl` cover HTN and HDDL semantics.
`elmachtoub2022spoplus` and `donti2017dc3` motivate decision-focused
fine-tuning. `silver2018alphazero` is background for the search-policy design,
not for this ranker.
