# Design 2: Constraint GCN with pairwise precedence and a differentiable layout

Status: proposed. This is the structured alternative to direct position
regression in Design 1.

## 1. Summary

Instead of predicting absolute start positions \(q_c\) directly, a relational
graph convolutional network (R-GCN/GCN) predicts a preference for every
ordering relation: “should child \(c\) run before child \(d\)?” The preferences
are aggregated into a child-by-child precedence matrix, and a differentiable
soft-sort/layout layer converts that matrix into start positions \(q\).
Penalty weights are predicted by a second head.

The existing Adam refinement still runs afterwards. The layout is only an
initializer; the exact decoder and validator are unchanged.

This design is motivated by the structure of the objective. The terms that
Adam can repair are ordering constraints; atom and equality constraints are
fixed by the child plans. Predicting an ordering is therefore a more natural
inductive bias than regressing arbitrary real numbers in \([0,1]\).

## 2. Graph

Use the same child/constraint incidence graph as Design 1, but make the
constraint nodes the centre of the computation:

- **child nodes** \(c\): length fraction, incoming/outgoing ordering counts,
  solved flag, atom-overlap features, degree;
- **constraint nodes** \(k\): type, ordering gap, unresolved flag, atom hash,
  equality polarity, before/after child identities as edge endpoints;
- **global node** \(g\): merge size and aggregate statistics;
- **typed edges**:
  - `before(k) -> c` for the child containing the `before` task,
  - `after(k) -> c` for the child containing the `after` task,
  - `incident(c,k)` for every shared constraint,
  - optional `same_child(c,k)` and `same_atom(c,k)` relations.

The graph is small enough that a full relational adjacency is practical. Use
edge types rather than one homogeneous adjacency, because an ordering edge and
an atom incidences have different semantics.

## 3. GCN backbone

For each relation type \(r\), use a relational GCN layer

\[
h_i' = \sigma\!\left(
  \sum_{r}\sum_{j\in\mathcal N_r(i)}
  \frac{1}{c_{i,r}}\,W_r h_j
  + W_0 h_i
\right),
\]

with degree normalization \(c_{i,r}=|\mathcal N_r(i)|\). Use two or three
layers, width \(d=64\), ReLU/GELU, residual connections, and LayerNorm. A
two-layer GCN is deliberately chosen over a deep GAT: the graph is small and
the pairwise head needs stable, permutation-invariant embeddings rather than
long-range attention.

A GAT variant is a reasonable fallback if edge types are too many; the
important part of this design is the output parameterization and the layout
layer, not the exact aggregation rule.

## 4. Outputs

### 4.1 Pairwise precedence logits

For an ordering constraint \(k\) with `before` in child \(b(k)\) and `after`
in child \(a(k)\), predict

\[
s_k = \operatorname{MLP}_s(h_k \| h_{b(k)} \| h_{a(k)} \| h_g).
\]

Use a symmetric parameterization for child pairs:

\[
S_{cd} = \sum_{k:\,b(k)=c,\,a(k)=d} \alpha_k\,\sigma(s_k)
       - \sum_{k:\,b(k)=d,\,a(k)=c} \alpha_k\,\sigma(s_k)
       + \lambda_{\text{prior}}\,\pi_{cd},
\]

where \(\pi_{cd}\) is an optional cheap prior (for example a spectral
seriation score or the child-length order), and \(\alpha_k\) can be a learned
constraint-type weight. The antisymmetric matrix \(S\) is the input to the
layout.

Atom and equality constraints do not produce a precedence. They contribute a
feasibility logit

\[
z_k = \operatorname{MLP}_f(h_k\|h_g),
\]

which is used as a training auxiliary signal and as a gate on the child
embedding. A child whose atom constraint is predicted infeasible should not be
moved confidently; the exact validator will reject the merge if the child
plans really do not establish the atom.

### 4.2 Penalty weights

Predict positive weights exactly as in Design 1:

\[
\hat w_k = w_{\min} + \operatorname{softplus}(\operatorname{MLP}_w(h_k\|h_g)).
\]

The layout determines \(q\); the weights determine how strongly Adam enforces
the constraints. Both are passed to `SgplanMerger::optimize`.

## 5. Differentiable layout

Given pairwise scores \(S\in\mathbb R^{C\times C}\), produce a start position
for each child.

### 5.1 Soft permutation

Use a Gumbel–Sinkhorn layer. Let \(R\in\mathbb R^{C\times C}\) be a score
matrix whose entry \(R_{r,c}\) scores assigning child \(c\) to rank \(r\).
Initialize \(R_{r,c}\) from a learned child rank head and add the pairwise
term

\[
R_{r,c} \mathrel{+}= \beta \sum_{d}
  \mathbf 1[\operatorname{rank}(d)<r]\,S_{d,c}.
\]

The exact indicator is replaced by a soft cumulative rank during training.
Sinkhorn normalization with temperature \(\tau\) yields a doubly stochastic
matrix \(P\),

\[
P = \operatorname{Sinkhorn}\!\left(\exp(R/\tau)\right),
\]

where \(P_{r,c}\) is the probability that child \(c\) receives rank \(r\).
Anneal \(\tau\) from 1.0 to 0.1. At inference, take the hard assignment
\(P^\star_{r,c}=1\) for the arg-max rank and use it in a straight-through
estimator during training.

### 5.2 From a permutation to positions

For a hard permutation \(\rho\), let

\[
q_c = \frac{1}{L}\sum_{d:\,\rho(d)<\rho(c)} l_d .
\]

This is exactly the sequential initialization in the chosen order, so the
layout can reproduce the current baseline and every permutation baseline. It
also guarantees \(q_c+l_c/L\in[0,1]\). During training, replace the hard
prefix sum with the expected prefix under \(P\); the straight-through
permutation supplies gradients to \(S\).

If the model predicts a low-confidence layout, blend with the current order:

\[
q_c = (1-\gamma)\,q_c^{\text{seq}} + \gamma\,q_c^{\text{layout}},
\]

where \(\gamma=\sigma(\operatorname{MLP}_\gamma(h_g))\). This gives a learned
fallback rather than a hard switch.

## 6. Training

### 6.1 Pairwise teacher

For each merge node, run the current merger from several candidate orders and
keep the best one by exact violation, then objective. For every pair of
children whose relative order is stable across all best candidates, create a
binary label \(y_{cd}=1\) if \(c\) precedes \(d\). If no stable order exists,
mask the pair rather than invent a label.

The pairwise loss is

\[
\mathcal L_{\text{pair}}
  = \frac{1}{|\mathcal P|}\sum_{(c,d)\in\mathcal P}
    \operatorname{BCE}\!\left(\sigma(S_{cd}), y_{cd}\right)
  + \lambda_{\text{cycle}}\,\mathcal L_{\text{cycle}},
\]

where \(\mathcal L_{\text{cycle}}\) penalizes strong three-cycles
\(S_{cd}+S_{de}+S_{ec}\). This is important because a tournament of pairwise
preferences need not be acyclic.

### 6.2 Layout loss

After Sinkhorn, decode \(q\) and run \(T=8\) differentiable Adam steps on the
relaxed objective. The loss is

\[
\mathcal L
 = \lambda_v \sum_k \max(0,-m_k(q_T))_{\text{smooth}}
 + \lambda_o J(q_T,w_T)
 + \lambda_p \mathcal L_{\text{pair}}
 + \lambda_z \operatorname{BCE}(\sigma(z_k), \text{feasible}_k)
 + \lambda_r \|\hat w-w^\star\|^2.
\]

The exact violations are used only for validation; the training signal is the
smooth hinge and the unrolled Adam state. The Sinkhorn temperature and the
number of unrolled steps are curriculum parameters: start with a soft layout
and few steps, then sharpen and lengthen.

### 6.3 Why not regress \(q\) directly

Direct regression has to learn the combinatorial ordering and the metric
positions simultaneously. The pairwise/layout parameterization factors the
problem into (i) which child should precede which, and (ii) how long the
resulting intervals are. The second part is a deterministic prefix sum of the
predicted order, so the network only has to learn the combinatorial part. This
is the main inductive-bias advantage over Design 1.

## 7. Integration

The public seam is the same `MergeInitializer` as Design 1. Internally:

- `merge/features.rs` exposes child/constraint features and the typed
  adjacency;
- `merge/layout.rs` contains the differentiable Sinkhorn/prefix-sum layer;
- `merge/initializer.rs` returns `MergeInitialization { positions, weights,
  space }`;
- the model is behind a `neural` feature and selected by
  `PSPR_MERGE_MODEL`;
- a confidence gate falls back to the sequential order when Sinkhorn is
  near-uniform or the predicted cycle penalty is high.

## 8. Cost and risks

The GCN is \(O((C+K)d^2)\) per layer. Sinkhorn is \(O(C^2)\) per iteration and
usually runs for 5–20 iterations; for a merge node with \(C<50\) this remains
small compared with Adam. The main training cost is the unrolled Adam and the
Sinkhorn backpropagation.

Risks:

- **Cycle formation.** Pairwise preferences can be cyclic. The cycle penalty
  and a confidence fallback are required.
- **Sinkhorn instability.** Very low temperatures produce saturated
  assignments and vanishing gradients. Anneal the temperature and clip logits.
- **Label sparsity.** Many pairs have no stable teacher order. Mask them; do
  not treat “unknown” as “after”.
- **Constraint types.** Atom/equality constraints cannot be repaired by
  ordering. They must remain auxiliary feasibility signals, not layout edges.
- **Extra layer.** The layout adds a non-Rust component to the training path;
  inference is still deterministic and small.

## 9. Evaluation

Use the same protocol as Design 1, with additional baselines:

- a random permutation layout with the same Adam budget;
- a spectral seriation layout from the constraint graph;
- the GCN layout without the pairwise head (direct child rank);
- the GCN layout with and without the cycle penalty;
- leave-one-domain-out evaluation to test whether pairwise structure
  transfers better than direct \(q\) regression.

The key hypothesis is that the constraint GCN has a smaller generalization gap
than the direct GAT when the test domain has a different constraint topology.

## 10. Pseudo-code

```text
function INITIALIZE(merge):
    graph = typed_merge_graph(merge)
    h = GCN(graph)
    S = pairwise_precedence(h, graph)
    P = sinkhorn(rank_scores(h) + pairwise_rank_term(S), temperature)
    q = prefix_sum_layout(P, lengths)
    w = softplus(MLP_w(h_constraints, h_global))
    if confidence(P) < threshold:
        q = sequential_positions(merge)
    return (q, w)

function MERGE(merge):
    (q, w) = INITIALIZE(merge)
    (q, w) = ADAM_REFINE(merge, q, w)
    return VALIDATE(merge, DECODE(merge, q))
```

## 11. References

`kipf2017gcn` for graph convolution; `schlichtkrull2018rgcn` for typed
relational aggregation; `velickovic2018gat` for the attention fallback;
`cuturi2013sinkhorn` for Sinkhorn normalization; `hsu2006sgplan` for the
merge objective; `kingma2015adam` for Adam; `elmachtoub2022spoplus` for
predict-then-optimize.
