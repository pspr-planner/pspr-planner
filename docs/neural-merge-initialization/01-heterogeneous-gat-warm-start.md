# Design 1: Heterogeneous GAT warm start for the merge solver

Status: implemented as a first pass behind the `neural` feature. The
implementation covers the typed node features, bidirectional incidence edges,
three normalized attention layers, bounded position and positive weight heads,
teacher collection, and Burn model persistence. Multi-head attention and the
two-stage unrolled-Adam loss remain follow-up work.

## 1. Summary

Train a heterogeneous graph attention network (GAT) on the child/constraint
graph of a merge node. The network predicts one start position
\(\hat q_c\in[0,1]\) per child and one positive penalty weight \(\hat w_k\)
per shared constraint. Those values initialise the existing Adam refinement in
`SgplanMerger::optimize`; the decoder and exact validator are unchanged.

The model is deliberately small. A merge node typically has tens of children
and constraints, not millions of nodes. The value is not raw capacity but a
graph-aware inductive bias: it sees which child owns the steps that participate
in each shared ordering, how long each child is, and which constraints are
already structurally satisfied.

## 2. Why this graph

The current initialization \(q_c^0=\sum_{d<c}l_d/L\) treats children as an
arbitrary sequence. It cannot know that child `b` should start before child
`a`, that child `c` has no shared constraints, or that a long child should be
placed early to reduce makespan.

The natural graph is bipartite and typed:

- a **child node** \(c\) for every `PartialSolution`;
- a **constraint node** \(k\) for every `ConstraintNode`;
- a **global node** \(g\) for merge-level context;
- edges \(c\to k\) and \(k\to c\) for every incidence;
- typed edges \(k\to c_{\text{before}}\) and \(k\to c_{\text{after}}\) for
  ordering constraints;
- optional child-to-child edges when two children share an ordering
  constraint.

The graph is small, permutation-equivariant, and directly mirrors the data
structures already present in `merge/mod.rs`.

## 3. Node and edge features

All continuous features are normalized by the merge node, not by the whole
planning problem, so the model transfers across domains.

### Child node \(c\)

| Feature | Definition |
| --- | --- |
| length fraction | \(l_c/L\) |
| log length | \(\log(1+l_c)\) |
| solved | 0/1 from `PartialSolution::solved` |
| incoming orderings | number of shared orderings where this child owns the `after` task |
| outgoing orderings | number where this child owns the `before` task |
| shared atom count | number of atom/equality constraints incident to the child |
| final-atom overlap | bag-of-atoms intersection with shared atoms, hashed into 16 buckets |
| achieved-atom overlap | same for `achieved` |
| plan-task span | \((\min \text{task}, \max \text{task})/N_{\text{tasks}}\) |
| child degree | typed degree, normalized by \(C+K\) |

Do **not** feed the child’s index in the current sequence as a primary
feature. It is an arbitrary order and would make the model non-equivariant.
An optional “current position fraction” can be supplied as a weak feature for
tie-breaking, but it should be ablated.

### Constraint node \(k\)

| Feature | Definition |
| --- | --- |
| type | one-hot: ordering, atom, equality |
| ordering gap | \((\tau_{\text{after}}-\tau_{\text{before}})\), clipped |
| unresolved | 0/1 for an ordering whose task was not located |
| polarity | equality `equal`/`not equal` as \(\pm1\) |
| atom hash | learned 32-bucket embedding of the predicate/atom key |
| initial weight | current \(w_k^0\) (normally 1) |
| degree | number of incident children |
| child containment | whether before/after occur in the same child |

### Edge features

- relation type: `constraint_of`, `before_of`, `after_of`, `same_child`;
- normalized offset difference;
- same-child indicator;
- ordering polarity or constraint type.

## 4. Architecture

Let \(d=96\), \(H=4\) heads, and use three message-passing layers.

### 4.1 Input encoders

For each node type \(t\in\{\text{child},\text{constraint},\text{global}\}\),

\[
h_i^{(0)} = \operatorname{LayerNorm}
\left(\operatorname{MLP}_t(x_i)\right),
\]

where the child and constraint MLPs are separate. The global node is the
mean-pooled concatenation of all child and constraint features plus scalar
merge statistics \((C,K,L,\sum l_c^2/L^2)\).

### 4.2 Typed attention layer

For relation type \(r\) and head \(h\),

\[
e_{ij}^{(r,h)}
  = \operatorname{LeakyReLU}\!\left(
      a_{r,h}^{\top}
      \left[W_{r,h}h_i \,\|\, W_{r,h}h_j \,\|\, U_{r,h}u_{ij}\right]
    \right),
\]

\[
\alpha_{ij}^{(r,h)}
  = \operatorname{softmax}_{j\in\mathcal N_r(i)}
      e_{ij}^{(r,h)},
\qquad
h_i' = \operatorname{LayerNorm}\!\left(
  h_i + \operatorname{MLP}\!\left(
    \big\|_{h=1}^{H}\sum_{j\in\mathcal N_r(i)}
      \alpha_{ij}^{(r,h)} W_{r,h}h_j
  \right)\right).
\]

Use separate parameters per relation type, not one shared attention matrix.
This is the heterogeneous/relational GAT variant; it avoids forcing
“constraint-of” and “before-of” edges to share semantics. Residual connections
and LayerNorm keep the small graph stable. Dropout on edges (5–10%) is used
only during training.

### 4.3 Output heads

For a child \(c\),

\[
\hat q_c
  = \left(1-\frac{l_c}{L}\right)\sigma\!\left(
      \operatorname{MLP}_q(h_c \| h_g)
    \right).
\]

The factor \(1-l_c/L\) makes the child’s interval fit inside \([0,1]\) without
a projection. The model can still represent the sequential baseline, because
\(q_c^0\) is in that range.

For a constraint \(k\),

\[
\hat w_k
  = w_{\min} + \operatorname{softplus}\!\left(
      \operatorname{MLP}_w(h_k \| h_g)
    \right),
\]

clipped to \([w_{\min},w_{\max}]\). The softplus avoids the zero-gradient region
of a hard ReLU and keeps weights positive.

An optional head predicts a distribution over the three penalty spaces and a
bounded learning-rate multiplier. This is disabled in the first experiment; it
is useful only after \(q,w\) prediction is shown to help.

## 5. Training

The model is trained in two stages. Both use the same graph construction and
the same train/validation split.

### 5.1 Stage A: teacher imitation

Collect teacher initializations as described in the directory README. For each
merge node, keep the best \((q^\star,w^\star)\) by exact violation, then
objective. Train

\[
\mathcal L_{\text{imitate}}
  = \frac{1}{C}\sum_c \left(\hat q_c-q_c^\star\right)^2
  + \lambda_w\frac{1}{K}\sum_k
      \left(\log \hat w_k-\log w_k^\star\right)^2.
\]

Because the optimum is not unique, also keep up to \(S=4\) near-optimal teacher
points and use the minimum loss over them (a small Chamfer-style set loss).
This prevents the model from being penalized for choosing a different but
equally good ordering.

### 5.2 Stage B: truncated unrolled Adam

The real objective is not “match a teacher”; it is “after Adam refinement,
have low violation and low makespan.” Run \(T=8\) differentiable Adam steps
from \(\hat q,\hat w\) on the smoothed objective:

\[
q_{t+1} = \Pi_{[0,1]}
\left(q_t-\eta\,
  \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right),
\qquad
w_{t+1} = \operatorname{clip}_{+}
\left(w_t-\eta_w\,
  \frac{\hat m_t^w}{\sqrt{\hat v_t^w}+\epsilon}\right).
\]

Use the smooth hinge/makespan surrogate from `RelaxedProblem::evaluate`, not
the non-differentiable exact validator. The clamped projection is handled with
a straight-through estimator at the boundary. Adam moments are differentiated
for the first \(T\) steps; beyond that, detach the moments to bound memory.

The training loss is

\[
\mathcal L
  = \lambda_v \sum_k \max(0,-m_k(q_T))_{\text{smooth}}
  + \lambda_o J(q_T,w_T)
  + \lambda_r \,\text{rounds}(q_T,w_T)
  + \lambda_q \|\hat q-q^\star\|^2
  + \lambda_w \|\log\hat w-\log w^\star\|^2.
\]

Use a curriculum: first train on merge nodes with \(C\le 8\), then \(C\le 16\),
then all nodes. The exact objective is evaluated only for validation and
early stopping.

### 5.3 Why backprop through Adam

The merge solver is the differentiable part of the pipeline. Its Adam
recurrence is cheap for \(T=8\) steps and has a small parameter dimension
\((C+K)\). Back-propagating through it tells the GAT whether a start position
that looks good under the imitation loss actually reduces the *post-Adam*
violation. This is the predict-then-optimize training signal; it is more
relevant than fitting \(q^\star\) alone.

For long runs, an implicit differentiation variant can be added later: solve
the KKT conditions of the relaxed problem and differentiate the fixed point.
The truncated variant is the lower-risk first implementation.

## 6. Integration

Add the following in `src/solver/evaluation_tree/merge/`:

- `features.rs`: `MergeGraph`, typed node/edge features, and a deterministic
  extractor from `MergeProblem` and `RelaxedProblem`.
- `initializer.rs`: the `MergeInitializer` trait and
  `MergeInitialization { positions, weights, space }`.
- `neural/mod.rs`: Burn model definition, feature schema version, and
  serialization; enabled by a `neural` Cargo feature.
- `SgplanMerger::with_initializer(...)` or an `initializer` field on the
  struct. The default constructor keeps the sequential initialization.

The merge path becomes:

```text
extract MergeProblem
  -> RelaxedProblem::new
  -> if initializer: f_theta(graph)
  -> if no prediction or invalid: sequential initial_parameters()
  -> SgplanMerger::optimize(q0, w0)
  -> decode
  -> exact validation
```

Feature layout and model version must be recorded in `PenaltyReport` (or a
small side-channel log). A model trained on a different feature layout must be
rejected rather than silently produce garbage.

## 7. Cost

For \(N=C+K\), three layers of width \(d=96\) and four heads cost
\(O(LN d^2)\); with \(N<100\) this is well below a millisecond on a CPU. The
dominant cost remains Adam refinement. Training is offline and can use CPU for
the small graphs; GPU is useful only when the training set grows to many
thousands of merge nodes.

## 8. Evaluation and ablations

Compare:

1. sequential initialization (current);
2. GAT \(q\) only;
3. GAT \(q+w\);
4. GAT \(q+w+\) penalty-space head;
5. random best-of-\(N\) with the same inference budget;
6. GAT initialization followed by the same Adam budget.

Report exact violation at \(q_0\), final violation, rounds to tolerance,
decoded makespan, and end-to-end planner time. Report leave-one-domain-out
results and separate timing for feature extraction, inference, and Adam.

## 9. Risks

- **Label ambiguity.** Different \(q\) values can decode to the same plan. Use
  the set loss and post-Adam loss, not only pointwise regression.
- **Distribution shift.** Child counts and constraint types vary by domain.
  Leave-one-domain-out testing is mandatory.
- **Overfitting to length.** Length is a strong feature; ablate it to confirm
  the model uses constraint structure too.
- **Non-differentiable decode.** Do not back-propagate through the discrete
  decoder. Train on the smooth relaxed objective and validate with the exact
  decoder.
- **Model absence.** The sequential path must remain the default and must
  never depend on a model file.

## 10. Pseudo-code

```text
function INITIALIZE(merge):
    graph = extract_graph(merge)
    if model is None or graph is invalid:
        return SEQUENTIAL(merge)
    q_hat, w_hat = model(graph)
    if not finite(q_hat, w_hat) or out_of_bounds(q_hat, w_hat):
        return SEQUENTIAL(merge)
    return MergeInitialization(q_hat, clamp(w_hat, w_min, w_max))

function MERGE(merge):
    relaxed = RelaxedProblem(merge)
    init = INITIALIZE(relaxed)
    q, w, report = ADAM_REFINE(relaxed, init.q, init.w)
    plan = DECODE(relaxed, q)
    VALIDATE_EXACT(merge, plan)
    return plan
```

## 11. References

`velickovic2018gat` for graph attention; `schlichtkrull2018rgcn` for typed
relational message passing; `kipf2017gcn` for the GCN baseline;
`hsu2006sgplan` for the merge objective; `kingma2015adam` for Adam;
`elmachtoub2022spoplus` and `donti2017dc3` for predict-then-optimize training.

## 12. Implementation in this repository

- Graph extraction and the Burn model: `src/solver/evaluation_tree/merge/neural.rs`.
- Teacher collection, Adam training, and weight persistence:
  `src/bin/train_merge_initializer.rs` (requires `--features neural`).
- The merger API and sequential fallback:
  `src/solver/evaluation_tree/merge/mod.rs` and `merge/sgplan.rs`.
- Environment hook: `PSPR_MERGE_MODEL` is read by `SgplanMerger::default()`;
  an absent or invalid model leaves the sequential path in place.

```bash
cargo test --features neural
cargo run --features neural --bin train_merge_initializer -- \
  --domain ../hddl/test_data/ipc/Satellite-GTOHP/domain.hddl \
  --problem ../hddl/test_data/ipc/Satellite-GTOHP/p03.hddl \
  --epochs 40 --output target/merge-initializer.burnpack

PSPR_MERGE_MODEL=target/merge-initializer.burnpack \
  cargo run --features neural -- --hierarchical \
  ../hddl/test_data/ipc/Satellite-GTOHP/domain.hddl \
  ../hddl/test_data/ipc/Satellite-GTOHP/p03.hddl
```

The trainer writes `<output>.burnpack` and a sibling `<output>.json` with the
hidden width and sample count. `NeuralMergeInitializer::from_path` reads the
sidecar so the model can be loaded without repeating `--hidden`.

The first pass uses four attention heads with three normalized residual
layers; relation-specific parameter groups, edge dropout, the set-valued
teacher loss, and the truncated unrolled-Adam stage are left for the next
iteration.
