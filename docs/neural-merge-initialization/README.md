# Neural initialization of the SGPlan merge solver

Status: design note, 2026-10-22. Design 1 is now implemented behind the
optional `neural` feature; the default build remains free of Burn and uses the
sequential merge initialization. The other designs remain proposals.

## 1. What is being predicted

The merge solver in `src/solver/evaluation_tree/merge/sgplan.rs` relaxes a
merge node with `C` child plans to one continuous start position
\(q_c \in [0,1]\) per child. A child plan \(\pi_c\) has \(l_c\) steps and
\(L=\sum_c l_c\), so its normalized extent is
\([q_c, q_c+l_c/L]\). The relaxed objective is

\[
J(q,w;\phi)
  = \lambda_{\text{cost}}\operatorname{smoothmax}_c
      \left(q_c+\frac{l_c}{L}\right)
  + \sum_{k=1}^{K} w_k\,\phi\!\left(h_k(q)\right),
\]

where \(h_k\) is the smoothed violation of shared constraint \(k\), \(w_k\)
is its penalty weight, and \(\phi\) is one of the continuation penalty
spaces. Adam minimizes this objective. The resulting \(q\) is decoded into a
concrete interleaving and checked against the exact shared constraints.

The current initialization is deliberately naive:

\[
q_c^{0} = \frac{1}{L}\sum_{d<c} l_d,
\qquad
w_k^{0} = 1.
\]

Every child is concatenated in its current order, and every shared constraint
starts with the same penalty. Adam must therefore move start positions over
potentially large distances and discover which constraints deserve pressure.
The graph structure of the merge problem—which child owns which step, which
constraints connect which children, which constraints are ordering versus
atom/equality constraints, and how large each child is—is not used.

This design study asks whether a graph neural network can predict a better
initial point

\[
(\hat q,\hat w,\hat s)=f_\theta(G_m)
\]

from the merge graph \(G_m\), where \(\hat q\in[0,1]^C\) is a warm start for
the child positions, \(\hat w>0\) is an initial penalty weight per shared
constraint, and \(\hat s\) is an optional penalty-space or Adam schedule. The
existing Adam iteration then refines the point. The exact decoder and
validator remain unchanged.

There are two distinct uses of Adam:

1. **Inference refinement.** Adam runs on \(J\) as it does today, starting
   from \(f_\theta(G_m)\) instead of the sequential initialization.
2. **Training signal.** The predictor is trained through a truncated unrolled
   Adam recurrence. A forward pass runs \(T\) differentiable Adam steps, and
   the loss on the resulting violation/objective is back-propagated through
   those steps to \(\theta\). At inference, Adam is ordinary non-differentiable
   optimization.

This is a *predict-then-optimize* architecture: the network does not replace
the solver, and it cannot make an invalid merge valid. It can only make the
optimizer start closer to a low-violation, low-makespan region. The exact
validator is still the final authority.

## 2. The longlist

The longlist deliberately includes direct regression, structured
constraint-centric models, step-level models, and meta-learning variants.

| # | Approach | GNN family | Main outputs | Expected gain | Label cost | Integration ease | Robustness | Total |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Heterogeneous GAT over children and constraints | GAT | \(q\), \(w\) | 5 | 4 | 4 | 4 | **21** |
| 2 | Constraint GCN with pairwise precedence and a differentiable layout | GCN/R-GCN | pairwise scores, \(q\) | 4 | 4 | 4 | 5 | **21** |
| 3 | Hierarchical plan-step GAT with unrolled Adam | GAT | step times, \(q\), \(w\) | 5 | 3 | 2 | 4 | **19** |
| 4 | Relational GCN with child and constraint nodes | R-GCN | \(q\), \(w\) | 4 | 3 | 4 | 3 | 18 |
| 5 | GraphSAGE child encoder | GraphSAGE | \(q\) | 3 | 5 | 5 | 3 | 18 |
| 6 | Graph Transformer over the constraint graph | Graph Transformer | \(q\), \(w\) | 4 | 3 | 2 | 3 | 15 |
| 7 | GAT predicts penalty weights only; positions from spectral/CPM layout | GAT | \(w\) | 3 | 4 | 5 | 4 | 17 |
| 8 | GAT hypernetwork for Adam learning rates and penalty schedule | GAT | \(\hat s\), \(q\) prior | 3 | 3 | 3 | 3 | 15 |
| 9 | GAT ranks candidate initializations from a small generator | GAT | ranking | 3 | 4 | 2 | 4 | 15 |
| 10 | GAT encoder plus an end-to-end differentiable Adam solver | GAT + unrolled optimizer | all parameters | 5 | 2 | 2 | 4 | 15 |

The scores are 1–5; “integration ease” is inverted so that 5 means the
smallest change to the current Rust code. The total is a prioritization aid,
not a claim of measured quality.

### 2.1 Selected top three

1. **Heterogeneous GAT warm start** (Approach 1). It has the most direct
   payoff, needs only the existing child/constraint graph, and produces both
   \(q\) and \(w\). It is the recommended first implementation.
2. **Constraint-centric GCN with a differentiable layout** (Approach 2). It
   replaces unconstrained regression with pairwise precedence decisions, so
   its inductive bias matches the shared ordering constraints. It is the
   strongest option for unseen graph topologies and is the natural second
   model if direct regression overfits.
3. **Hierarchical plan-step GAT with truncated unrolled Adam** (Approach 3).
   It has the highest ceiling because it can reason about which steps inside a
   child need to move, but it is also the largest change. It should be
   prototyped only after the child-level graph representation is stable.

The remaining approaches are not rejected as useless. R-GCN (4) is close to
the chosen GAT in expressiveness; GraphSAGE (5) is a good cheap baseline;
weight-only prediction (7) is a useful ablation; and the meta-learning,
ranking, and end-to-end variants (8–10) are research directions with higher
training and integration cost.

## 3. Shared experimental protocol

All three designs should use the same data and reporting so their results are
comparable.

### 3.1 Data collection

Record one example per merge node while running the existing evaluation tree
on the IPC suite. For each example store:

- the child plans, their `plan_tasks`, lengths, final atoms, and achieved
  atoms;
- the shared constraints and their exact types;
- the sequential baseline’s final \(q\), \(w\), rounds, violations, and
  decoded objective;
- a teacher initialization obtained by running the current merger from several
  deterministic starting points (sequential, reversed, random permutations,
  and a small grid over the two largest children) and keeping the best exact
  violation, then objective;
- for small nodes, an exact or MILP teacher ordering when available.

Split by **domain**, not by merge node, and report leave-one-domain-out
results. Otherwise the model can memorize domain-specific child counts and
inflate the apparent generalization.

### 3.2 Metrics

At the merge level:

- exact violation \(\sum_k \max(0,-m_k(q))\) at the initial point;
- exact violation and smoothed objective after the same Adam budget;
- feasibility of the decoded plan and number of Adam rounds to tolerance;
- wall-clock time for feature extraction, inference, and Adam refinement.

At the planner level:

- solved instances and plan length on the IPC comparison suite;
- end-to-end median/mean/wall-clock with timeouts reported separately.

Every neural run must be compared with:

- the current sequential initialization;
- a random-init best-of-\(N\) baseline using the same inference budget;
- the same model with \(q\) only and with \(q+w\);
- an Adam-only run given extra rounds, to separate “better start” from “more
  compute”.

### 3.3 Integration contract

The production code should add a narrow seam rather than embedding a model in
`sgplan.rs`:

```rust
pub struct MergeInitialization {
    pub positions: Vec<f64>,   // C values in [0, 1]
    pub weights: Vec<f64>,     // K positive values
    pub space: Option<PenaltySpace>,
}

pub trait MergeInitializer {
    fn initialize(&self, graph: &MergeGraph) -> Option<MergeInitialization>;
}
```

`SgplanMerger` gets an optional initializer. If it returns `None`, the current
sequential start is used. Feature extraction belongs in a new
`merge/features.rs`; models and Burn code live behind a non-default `neural`
feature or a separate training example, so normal planner builds do not pull
in a tensor runtime. The model file is selected explicitly (for example
`PSPR_MERGE_MODEL`) and inference is deterministic on CPU.

### 3.4 Safety and failure behaviour

- Validate the predicted shape, finiteness, and bounds before using it.
- Fall back to the sequential start when inference fails or the model is
  absent.
- Keep the exact validator; never accept a merge solely because the model
  scored it highly.
- Log the predictor version, feature schema, and initialization delta in
  `PenaltyReport` so a regression can be attributed to the model rather than
  to Adam.

## 4. Design documents

- [`01-heterogeneous-gat-warm-start.md`](01-heterogeneous-gat-warm-start.md):
  heterogeneous GAT over child and constraint nodes, direct \(q,w\) heads.
- [`02-constraint-gcn-layout.md`](02-constraint-gcn-layout.md): constraint GCN
  with pairwise precedence scores and a differentiable layout.
- [`03-plan-step-gat-unrolled-adam.md`](03-plan-step-gat-unrolled-adam.md):
  hierarchical step-level GAT trained through truncated Adam.

## 5. References

The graph-network background is `kipf2017gcn`, `velickovic2018gat`,
`hamilton2017graphsage`, `schlichtkrull2018rgcn`, and `xu2019gin`.
Predict-then-optimize and differentiable optimization are covered by
`elmachtoub2022spoplus`, `donti2017dc3`, and `agrawal2019differentiable`.
The merge objective and optimizer are `hsu2006sgplan` and `kingma2015adam`.
