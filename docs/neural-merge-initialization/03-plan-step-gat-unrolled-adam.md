# Design 3: Hierarchical plan-step GAT trained through truncated Adam

Status: research prototype. This is the highest-ceiling and highest-cost design.

## 1. Summary

Designs 1 and 2 operate on one node per child plan. That is a good match for
the current merge solver, which has one start position \(q_c\) per child, but
it throws away the internal structure of the child plans: which steps are
before or after the shared task, how many steps separate two shared
occurrences, and which child steps establish the shared atoms.

This design builds a **step-level heterogeneous graph** and applies a
hierarchical GAT:

1. a step-level GAT encodes every plan step and its local task context;
2. attention pooling from steps to children produces child embeddings;
3. a merge-level GAT attends over child and constraint nodes;
4. output heads predict \(\hat q_c\), \(\hat w_k\), and an optional
   per-step residual field.

The model is trained with a truncated unrolled Adam recurrence, so the
gradient tells the network whether its initialization actually helps the
optimizer that will run at inference.

The first implementation projects step-level information back to the current
one-position-per-child solver. A second phase can extend the solver to
per-step positions if the projected model shows a clear gain.

## 2. Why step-level structure matters

Two child plans can have the same length and the same number of shared
constraints but very different internal structure. In one, the shared task is
the first step; in the other, it is behind ten local steps. The child-level
features in Design 1 can see the task index, but they cannot see which local
preconditions/effects are adjacent to the shared occurrence or how much
slack exists before it.

The current relaxation already computes a normalized occurrence time for each
task via `locate`: `time = position / total`. Those occurrences are the natural
step nodes. A step-level model can learn that a shared `before` task occurs
late inside its child and therefore that child should start early, rather than
inferring this only from a coarse “task span” feature.

## 3. Graph schema

### 3.1 Nodes

| Node type | Count | Features |
| --- | ---: | --- |
| plan step \(s=(c,i)\) | \(\sum_c l_c\) | normalized position \(i/l_c\), normalized global position \(i/L\), task index bucket, child length fraction, solved flag, local in/out degree, shared-constraint count, atom/predicate hash buckets, achieved/final overlap |
| child \(c\) | \(C\) | pooled step features, length fraction, number of shared occurrences, degree |
| constraint \(k\) | \(K\) | type, ordering gap, unresolved flag, atom/equality hash, before/after child, initial weight |
| global \(g\) | 1 | \(C\), \(K\), \(L\), length entropy, constraint-type histogram |

The task index is a grounded integer, not a semantic embedding. Use a learned
embedding table keyed by `(domain, task_id)` when training per domain, or a
hash bucket when evaluating cross-domain transfer. A predicate/atom hash is
similarly bucketed; do not assume a fixed vocabulary across domains.

### 3.2 Edges

- `next(s_{c,i}, s_{c,i+1})` for consecutive steps inside a child;
- `contains(c, s_{c,i})` and its reverse;
- `before(k, s_{c,i})` and `after(k, s_{c,j})` for a located shared ordering;
- `same_child(s_{c,i}, s_{c,j})` for steps in the same child (optional,
  restricted to a local window);
- `incident(k, c)` for every constraint/child incidence;
- `global -> all` and `all -> global`.

Use typed attention or typed convolution; a homogeneous GCN loses the
difference between “next step” and “shared ordering”.

## 4. Architecture

### 4.1 Step encoder

For each step node \(s\),

\[
h_s^{(0)} = \operatorname{MLP}_{\text{step}}(x_s),
\]

then run \(L_s=2\) GAT layers on the step graph with relation-specific
attention. The intra-child `next` edges are the strongest locality prior; the
constraint edges connect the steps that actually participate in a shared
ordering.

### 4.2 Child pooling

For a child \(c\), pool its step embeddings with attention:

\[
\alpha_s = \operatorname{softmax}_{s\in c}
  \left(u_c^\top \tanh(W_c h_s)\right),
\qquad
h_c = \sum_{s\in c}\alpha_s h_s.
\]

This is preferable to mean pooling because the relevant step is often the
shared occurrence, not the average step. Concatenate the pooled vector with
the child-level scalar features.

### 4.3 Merge-level GAT

Run two GAT layers over the child/constraint graph, initialised with the
pooled child embeddings. This is the same message passing as Design 1 but with
richer child states. A global node is updated by mean pooling and supplies
context to every output head.

### 4.4 Outputs

For compatibility with the current solver, predict child positions:

\[
\hat q_c
 = \left(1-\frac{l_c}{L}\right)\sigma\!\left(
     \operatorname{MLP}_q(h_c\|h_g)
   \right),
\]

and constraint weights:

\[
\hat w_k = w_{\min}+\operatorname{softplus}
  \left(\operatorname{MLP}_w(h_k\|h_g)\right).
\]

Optionally predict step residuals

\[
\hat r_{c,i} = \operatorname{MLP}_r(h_{s_{c,i}}\|h_c)
\]

with the constraint \(\sum_i \hat r_{c,i}=0\), and define the projected child
position

\[
\tilde q_c = \frac{1}{l_c}\sum_i
  \left(\hat q_c+\frac{i}{L}+\hat r_{c,i}\right).
\]

In the first phase, \(\tilde q_c\) is passed to the existing one-position
solver. In a later phase, the residuals can be exposed to a step-level
extension of `RelaxedProblem`, where Adam optimizes one scalar per step rather
than one per child.

## 5. Training with unrolled Adam

### 5.1 Teacher data

Use the same teacher protocol as Design 1. For step-level supervision, only
the aggregated \(q^\star\) is required; per-step labels are usually
unidentifiable because only relative order and shared occurrence times affect
the decoded plan. If an exact small-instance solver is available, its step
times can be used as an auxiliary label, but they should be down-weighted.

### 5.2 Differentiable refinement

Run \(T\) Adam steps on the relaxed objective starting from
\(\hat q,\hat w\). For the projected model, the state is still \(C+K\)
dimensional, so the cost is the same as Design 1:

\[
q_{t+1}=\Pi\!\left(q_t-\eta\,
  \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\right).
\]

Use the smooth hinge and smooth maximum as the differentiable surrogate. The
exact decoder is applied only on the forward validation path. Truncate
back-propagation after \(T=8\) steps and detach the Adam moments after the
first few steps to keep memory bounded.

The loss is

\[
\mathcal L
 = \lambda_v \sum_k \max(0,-m_k(q_T))_{\text{smooth}}
 + \lambda_o J(q_T,w_T)
 + \lambda_q \|\hat q-q^\star\|^2
 + \lambda_r \|\hat r\|^2
 + \lambda_w \|\log\hat w-\log w^\star\|^2.
\]

The step-residual regularizer is important: without it, the model can
overfit local step noise while the projected \(q\) is already correct.

### 5.3 Curriculum

Train first on graphs with at most 32 step nodes, then 64, then 128. At each
stage, mix in the previous stage’s graphs to avoid catastrophic forgetting.
For large merge nodes, sample a fixed number of step nodes per child while
keeping all shared occurrences; this is a training-time approximation only.

## 6. Integration

The seam remains `MergeInitializer`, but the graph extractor must expose step
nodes:

- `merge/features.rs`: add a `StepNode` variant and a `MergeGraph::steps`
  vector; keep the child/constraint edge lists compatible with Designs 1 and
  2.
- `merge/neural/plan_gat.rs`: Burn model and pooling.
- `merge/initializer.rs`: return the predicted child positions and weights.
- Optional `merge/step_solver.rs`: a future extension of `RelaxedProblem` with
  one parameter per step. Do not implement it in the first prototype.

The current one-position solver remains the default. If the step model cannot
be loaded or the graph exceeds a configured cap, the system falls back to the
child-level model or to the sequential initialization.

## 7. Cost

Inference is \(O((N_{\text{step}}+C+K)d^2)\) per layer. A merge node with a few
hundred steps and \(d=64\) is still cheap on a CPU, but training is heavier
because the graph is larger and the unrolled Adam path stores activations for
each layer. Use mini-batching by merge node, not by step, and cache the static
graph tensors.

The projected model does not increase the dimension of Adam’s state, so
inference cost remains close to Design 1. A future per-step solver would
increase Adam’s dimension from \(C+K\) to \(N_{\text{step}}+K\), which is a
separate performance decision.

## 8. Evaluation

Compare:

- sequential initialization;
- Design 1 child-level GAT;
- step GAT without step-residual head (pooled only);
- step GAT with projected \(\tilde q\);
- step GAT with a step-level Adam extension, if implemented.

Report graph construction time separately from inference and Adam. Use
leave-one-domain-out splits and ablate the step features one family at a time
(task index, atom overlap, local degree, internal position). The key question
is whether internal plan structure predicts a better warm start than the
child-level features; if not, Design 3 is not worth its complexity.

## 9. Risks

- **Graph size.** Step-level graphs can be much larger than child-level
  graphs; cap or sample nodes during training and fall back at inference.
- **Overfitting to local step tokens.** Task indices and atom hashes are
  domain-specific. Use hashing and leave-one-domain-out testing.
- **Label ambiguity.** The same child-level plan can result from many
  step-level time assignments. Use projected labels and strong residual
  regularization.
- **Solver mismatch.** The existing Adam solver has one parameter per child;
  a step-level model must be projected until the solver is extended.
- **Training cost.** Back-propagating through both the step GAT and unrolled
  Adam is expensive; start with a frozen encoder and only train the output
  heads, then unfreeze.

## 10. Pseudo-code

```text
function INITIALIZE_STEP(merge):
    steps = flatten_steps(merge.children)
    graph = build_step_graph(steps, merge.constraints)
    h_step = step_gat(graph)
    h_child = attention_pool(h_step, children)
    h = merge_gat(h_child, constraint_nodes)
    q_hat = bounded_mlp_q(h_child, h_global)
    w_hat = positive_mlp_w(h_constraint, h_global)
    r_hat = residual_mlp(h_step)       # optional
    q_hat = project(q_hat, r_hat)
    return (q_hat, w_hat)

function TRAIN_STEP(model, examples):
    for (graph, teacher) in examples:
        q0, w0 = model(graph)
        qT, wT = unrolled_adam(graph, q0, w0, steps=T)
        loss = smooth_violation(qT) + objective(qT, wT)
               + imitation(q0, teacher) + residual_penalty
        loss.backward(truncate_after=T)
        adam_trainer.step()
```

## 11. References

`velickovic2018gat` for step-level attention; `hamilton2017graphsage` for
inductive pooling and neighbor sampling; `xu2019gin` for expressiveness
considerations; `hsu2006sgplan` for the relaxed merge objective;
`kingma2015adam` for both the solver and the differentiable meta-training
path; `donti2017dc3` and `agrawal2019differentiable` for differentiating
through optimization.
