# Design 3: Abstract-plan permutation controller for refinement recovery

Status: proposed, 2026-10-23. This note proposes a new neural seam above the
current C+G decomposition guidance: permuting the abstract top-level plan when
refinement fails or appears low quality. Soundness and exact validation stay in
the symbolic pipeline.

## 1. Problem statement

PSPR currently does:

1. Build one abstract top-level plan with `evaluate(...)`.
2. Refine that fixed order with `refine_plan_with_session(...)`.
3. Retry refinement with larger budgets (`refined_retry_options(...)`).

The existing neural components reorder decisions inside refinement
(`refine_task` method ordering and `search_partial_inner` ready-task ordering),
but they do not change the top-level abstract step order itself.

On some instances, refinement fails because the chosen top-level order pushes
hard subproblems too early, commits to difficult method branches first, or
creates state/context drift that blocks later decompositions. Even when
refinement succeeds, another abstract order can produce shorter or faster plans.

Goal: add an optional AI component that proposes one or more permutations of
the abstract plan before or between refinement retries.

## 2. Design constraints

- The symbolic refiner remains authoritative.
- Every candidate permutation is evaluated by normal refinement and exact
  validation.
- If the model is missing, malformed, or uncertain, fallback is the current
  deterministic order.
- Hard ordering constraints must be preserved.
- Runtime overhead must be bounded and small compared with refinement.
- The interface must fit the existing `DecompositionSession` pattern and keep
  `RefinementOptions` unchanged.

## 3. Three candidate architectures

All approaches below produce a ranked list of valid permutations for one
abstract plan.

### 3.1 Architecture A: failure-conditioned sequence editor

Model the abstract plan as a token sequence and use an encoder-decoder
transformer to emit an edited sequence (or edit script).

Inputs:

- abstract steps as structured tokens (`method[task(args)]`, `action(args)`),
- attempt metadata (budget, depth),
- failure summary from the last refinement attempt.

Outputs:

- top-k edited permutations (plus identity).

Pros:

- very expressive, can learn nonlocal reorder patterns.
- naturally consumes textual/serialized traces.

Cons:

- weak structural bias for precedence constraints and HTN semantics.
- highest integration and debugging risk.
- harder to guarantee valid permutations without heavy constrained decoding.

### 3.2 Architecture B: constrained plan-graph pointer policy with value head (selected)

Represent the abstract plan as a typed graph, encode with relational GNN
(compatible with existing decomposition graphs), and decode a permutation via a
masked pointer policy that only emits valid next steps.

Inputs:

- step nodes with symbolic/task features,
- precedence and dependency edges,
- optional failure features from prior failed attempt.

Outputs:

- top-k valid permutations from beam search,
- value estimate of refinement success/cost for reranking.

Pros:

- strongest structural alignment with HTN/precedence data.
- straightforward validity masking.
- best fit with current Burn GAT code and trace pipeline.

Cons:

- more complex than local heuristics.
- requires graph feature engineering and permutation labels.

### 3.3 Architecture C: contextual bandit over local edit operators

Avoid full-sequence decoding. Predict scores over local edit operators applied
to the current plan: adjacent swap, block move, window rotation, and segment
reverse (all constraint-checked).

Inputs:

- aggregate plan features,
- per-window features,
- failure context.

Outputs:

- top-k edited plans produced by best-scoring operators.

Pros:

- easiest to implement and stabilize.
- very low latency and easy safety checks.

Cons:

- limited search radius; can miss nonlocal reorderings.
- quality depends heavily on handcrafted operator set.

## 4. Architecture comparison

Scores are 1-5, higher is better.

| # | Architecture | Expected gain | Integration cost | Robustness | Latency | Total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | Failure-conditioned sequence editor | 5 | 2 | 3 | 2 | 12 |
| B | Constrained plan-graph pointer + value | 5 | 4 | 5 | 4 | 18 |
| C | Contextual bandit over local edits | 3 | 5 | 4 | 5 | 17 |

Selected architecture: **B**.

Rationale:

- same symbolic compatibility pattern as the current method/policy guidance,
- best trade-off between global reordering power and reliability,
- can reuse typed-graph modeling and training infrastructure already in
  `src/solver/evaluation_tree/decomposition/`.

## 5. Detailed design for architecture B

### 5.1 Interface and integration seam

Add a new optional trait in `evaluation_tree/decomposition/mod.rs`:

```rust
pub struct AbstractPlanContext {
    pub steps: Vec<String>,
    pub parsed: Vec<Step>,
    pub depth_budget: usize,
    pub grounding_budget: usize,
    pub attempt: usize,
    pub previous_failure: Option<PermutationFailure>,
}

pub struct AbstractPermutation {
    pub order: Vec<usize>,
    pub score: f64,
}

pub trait AbstractPlanPermuter: std::fmt::Debug + Send + Sync {
    fn propose(&self, context: &AbstractPlanContext) -> Option<Vec<AbstractPermutation>>;
}
```

Extend `DecompositionSession`:

```rust
pub struct DecompositionSession<'a> {
    pub method_ranker: Option<&'a dyn MethodRanker>,
    pub search_policy: Option<&'a dyn SearchPolicy>,
    pub plan_permuter: Option<&'a dyn AbstractPlanPermuter>,
    pub trace: Option<&'a mut dyn DecompositionTraceSink>,
}
```

Wire this in `solve_hierarchical_with_session(...)` in
`src/solver/evaluation_tree/hierarchical.rs`:

1. Try identity order first (current behavior).
2. If refinement fails, ask `plan_permuter` for alternatives.
3. Validate permutation shape and hard-order feasibility.
4. Retry refinement on candidates in score order, within the current retry
   budget.
5. Keep existing fallback and error behavior.

This preserves the existing public API and keeps permutation optional.

### 5.2 Hard-constraint preserving permutations

Let original abstract plan indices be `[0..n)`. A permutation `pi` is valid if:

1. it is a bijection over `[0..n)`, and
2. for every hard precedence edge `(i -> j)`, position(`i`) < position(`j`).

Precedence edges come from the grounded top-level network (or existing derived
repair orderings when enabled).

If constraints cannot be recovered for a case, only identity is accepted.

### 5.3 Graph representation

Build one graph per permutation decision:

- step nodes: one per abstract step instance,
- global node: plan-level context.

Node features (step):

- action/method/compound indicator,
- task-name and method-name hashes,
- arity, normalized original position,
- local branching proxy (number of applicable methods seen in prior attempts),
- failure-locality flags (step on first failing frontier, dead-end frequency),
- depth/budget features.

Node features (global):

- plan length,
- attempt id,
- previous failure code (timeout/dead-end/budget exhaustion),
- recent success rate for this domain (optional offline stat).

Edge types:

- original sequence adjacency,
- hard precedence edges,
- soft producer-consumer links from atom overlap,
- conflict links (negative interaction estimate),
- same-task-schema links.

### 5.4 Model architecture

Encoder:

- 3-layer relational GAT, hidden width 64-128,
- residual + layer norm, 4 attention heads.

Decoder:

- masked pointer policy over remaining steps,
- hard mask enforces precedence and uniqueness,
- beam search (width B, e.g. 8) returns top-k permutations.

Value heads:

- `p_success`: probability refinement succeeds under this order,
- `cost_to_first_solution`: predicted recursive calls/time proxy.

Candidate score:

$$
S(pi) = \alpha \log p_\theta(\text{success}\mid pi)
      - \beta \hat{c}_\theta(pi)
      - \gamma d_{KT}(pi, \text{identity}),
$$

where $d_{KT}$ is Kendall-tau distance used as a regularizer to avoid
unnecessarily aggressive reorderings.

### 5.5 Runtime selection policy

For each retry attempt:

1. Candidate set starts with identity.
2. Add up to `k` model proposals (deduplicated).
3. Evaluate candidates with symbolic refinement in descending `S(pi)`.
4. Stop on first successful refinement, else continue existing retry logic.

Safety gates:

- reject non-permutations,
- reject precedence-violating outputs,
- reject non-finite scores,
- if all rejected: identity only.

### 5.6 Failure signal structure

Add lightweight failure diagnostics from refinement:

- first abstract step index with zero alternatives,
- whether budget/deadline hit,
- total branch expansions consumed,
- counts of method dead-ends and partial-order dead-ends.

These features condition the next permutation proposal.

## 6. Training data and objectives

### 6.1 Data generation

Use existing trace collection process and extend it:

1. For each `(domain, problem)` obtain baseline abstract plan `P`.
2. Evaluate identity refinement outcome.
3. Generate neighborhood permutations (constraint-safe local edits + sampled
   beam proposals from a bootstrap heuristic).
4. Run bounded refinement for each candidate order.
5. Record success, time/expansions, and plan quality.

This yields tuples `(context, permutation, outcome)` without external labels.

### 6.2 Supervision targets

Policy target:

- probability mass over top-performing permutations (soft target from
  success/cost).

Value targets:

- binary success,
- normalized refinement cost.

Loss:

$$
L = L_{policy} + \lambda_1 L_{success} + \lambda_2 L_{cost}.
$$

Optional pairwise preference term for robust ranking:

$$
L_{pair} = \max(0, m - s_{good} + s_{bad}).
$$

### 6.3 Curriculum

1. Small plans (length <= 12) with dense local edits.
2. Medium plans (<= 32) with beam-generated candidates.
3. Full IPC traces with hard negatives from failures.

Split by domain, not by problem instance.

## 7. Evaluation protocol

Primary metrics:

- refinement success rate on instances where baseline fails,
- solved count under fixed wall-clock budget,
- time to first valid refinement,
- recursive calls / expansions to first success.

Quality metrics:

- primitive plan length,
- method count,
- goal status and exact-validator pass rate (must match baseline guarantees).

Ablations:

- identity only,
- architecture C (bandit local edits),
- architecture B without value head,
- architecture B without failure conditioning,
- architecture B with/without Kendall-tau regularization.

## 8. Rollout plan

Phase 0 (instrumentation):

- add failure diagnostics and permutation evaluation harness,
- no model yet, only identity + heuristic local edits.

Phase 1 (baseline learner):

- train architecture C (bandit over local edits) as a low-risk baseline.

Phase 2 (selected model):

- implement architecture B policy/value model,
- connect via `DecompositionSession.plan_permuter`.

Phase 3 (joint optimization):

- optionally co-train with existing method and frontier heads so all three
  guidance components share one encoder.

## 9. Risks and mitigations

Risk: model overfits domain-specific ordering patterns.

- Mitigation: domain-level splits, conservative regularization, identity in
  candidate set.

Risk: permutation overhead consumes refinement budget.

- Mitigation: strict `k` cap, beam width cap, early stop on first success,
  lightweight encoder.

Risk: invalid permutations corrupt behavior.

- Mitigation: hard validation before refinement, deterministic fallback.

Risk: improvements come from extra retries, not model quality.

- Mitigation: compare against compute-matched random and heuristic baselines.

## 10. Why this is the best fit now

Architecture B is the best near-term path for PSPR because it adds a missing
decision layer (top-level order) while preserving the existing symbolic
correctness boundary. It reuses the current graph-neural tooling, keeps runtime
bounded, and directly targets the observed refine-failed cases where internal
method/frontier ordering alone is insufficient.