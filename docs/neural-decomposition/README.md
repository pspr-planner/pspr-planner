# Neural guidance for compound-task decomposition

Status: implementation note, 2026-10-22. This directory compares neural
components for the C+G refinement path and documents the selected
implementation. It does not add a model to the default build; the symbolic
refiner remains the correctness boundary.

## 1. Scope

PSPR's hierarchical solver first computes an abstract task-network plan and
then refines it in `src/solver/evaluation_tree/hierarchical.rs`. Two decisions
dominate refinement cost:

1. **Which grounded method should decompose a compound task?**
   `refine_task` currently grounds every method instance, checks its
   precondition, and tries the preferred method first, then the remaining
   methods sorted by subtask count.
2. **Which ready task should be expanded next in a partial-order network?**
   `search_partial_inner` currently scans ready tasks in index order and
   backtracks over all alternatives up to `MAX_REFINEMENTS`.

Both decisions are good candidates for a narrow neural seam: the network
proposes or ranks, while the existing symbolic refiner applies methods,
transitions the state, backtracks, and validates the final decomposition.

This note compares five designs for each decision, selects one of each, and
links the two selected design docs.

## 2. The two selected designs

### 2.1 Method ranking

**Selected: state-conditioned heterogeneous relational GAT with a
candidate-centric scoring head.**

The model scores grounded method instances using a typed graph over the
query task, method schemas, subtasks, atoms, and the current state. It only
reorders the candidates that already passed the symbolic applicability check.
It does not invent methods, bindings, or subtasks.

See [Design 1](01-state-conditioned-relational-method-ranker.md).

### 2.2 Search policy and value

**Selected: frontier GNN policy with a value head over `search_partial`.**

The policy scores the currently ready tasks; the value head estimates the
remaining refinement cost. The symbolic DFS/PUCT controller still performs
state transitions and exact backtracking. The value is initially used only to
order branches, not to prune them unsoundly.

See [Design 2](02-frontier-gnn-policy-value.md).

## 3. Method-ranking longlist

All scores are 1–5. “Integration” is inverted so 5 means the smallest change
to the current Rust code. The totals are prioritization aids, not measured
accuracy.

| # | Approach | Representation | Main output | Gain | Label cost | Integration | Robustness | Latency | Total |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Homogeneous task–method GAT | One node/edge type; method nodes and task nodes | Method score | 3 | 4 | 5 | 3 | 5 | **20** |
| 2 | State-conditioned heterogeneous relational GAT | Task, method schema, grounded candidate, subtask, atom, and global nodes with typed edges | Candidate score | 5 | 4 | 4 | 5 | 4 | **22** |
| 3 | Set transformer over grounded candidates | Flat candidate set plus a state token; no task–subtask graph | Candidate score | 4 | 4 | 5 | 4 | 4 | **21** |
| 4 | Graph transformer with candidate cross-attention | Global graph plus a separate query per candidate | Candidate score | 5 | 3 | 3 | 4 | 3 | **18** |
| 5 | RGAT plus HDDL text encoder | Typed graph plus sentence embeddings of method and atom names | Candidate score | 4 | 3 | 3 | 3 | 3 | **16** |

### 3.1 Why approach 2 wins

- **Typed relations matter.** `task --method`, `method --subtask`,
  `method --atom`, `candidate --method`, and `atom --state` have different
  semantics. Sharing one attention matrix loses that distinction.
- **State conditioning is essential.** The same method can be applicable in
  one state and impossible in another. The selected model has a global state
  node and per-atom truth features.
- **Candidate-centric scoring handles arbitrary arity.** Method parameters
  vary widely across domains; a separate candidate node with binding features
  is more stable than a fixed-size method classifier.
- **No text dependency.** HDDL provides names, schemas, and preconditions.
  The text hybrid is useful for cross-domain transfer but adds an encoder and
  fragile tokenization for little expected gain on this benchmark.
- **Safe integration.** The ranker only changes method order. The preferred
  method, precondition check, depth limit, and `MAX_REFINEMENTS` stay intact.

Approach 3 is the strongest cheap baseline: it uses the same candidate
features but removes the structural graph. It should be implemented as an
ablation, not as the main design.

## 4. Search-policy/value longlist

| # | Approach | Representation | Main output | Gain | Label cost | Integration | Robustness | Latency | Total |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Handcrafted linear heuristic | Ready-set counts, depth, precedence degree | Task order | 2 | 5 | 5 | 4 | 5 | **21** |
| 2 | MLP over flattened frontier | Fixed-size vector of ready-task and state features | Task order / value | 3 | 4 | 5 | 3 | 5 | **20** |
| 3 | Frontier GNN policy + value | Typed graph over remaining tasks, precedence, atoms, and state | Ready-task distribution and cost-to-go | 5 | 4 | 4 | 5 | 4 | **22** |
| 4 | Residual-task-sequence transformer | Sequence of remaining tasks plus a state token | Next-task distribution | 4 | 3 | 4 | 4 | 3 | **18** |
| 5 | LLM policy with symbolic verifier | Prompt containing the residual network and state | Task/method proposal | 3 | 5 | 2 | 2 | 1 | **13** |

### 4.1 Why approach 3 wins

- **The frontier is a graph, not a sequence.** Precedence, shared atoms, and
  method-origin relations are the useful signal; a sequence order would
  reintroduce the arbitrary index bias the policy is meant to remove.
- **Variable ready sets are natural.** A pointer/softmax over ready nodes
  handles one, three, or twenty ready tasks without padding tricks.
- **One encoder serves policy and value.** The shared graph representation
  can produce both the action distribution and a cost-to-go estimate.
- **The symbolic controller remains complete.** The policy only reorders
  ready tasks; every branch is still explored until the existing budget is
  exhausted.
- **The LLM option is a poor default here.** HDDL already provides a formal
  domain model, and the LLM adds latency, nondeterminism, and an external
  dependency without solving the partial-order search problem directly.

Approach 1 is a useful baseline and may be enough on small networks. Approach
3 is the design worth implementing first because it can share node/feature
code with the method ranker.

## 5. Shared data and evaluation protocol

### 5.1 Trace collection

Instrument `refine_task` and `search_partial_inner` behind an offline tracing
feature. For each compound-task call, record:

- the query `TaskNode` and the current state;
- every grounded method candidate, its bindings, and whether its
  precondition held;
- the method and bindings chosen in the successful C+G refinement;
- for each candidate, whether it led to a complete refinement, the number of
  recursive calls, and the resulting primitive-plan length;
- for partial-order search, the ready set at every recursion node, the chosen
  task, and the remaining cost.

Use successful C+G runs as the primary labels and PANDA decompositions as an
independent oracle where the plans can be aligned. Keep failed branches: they
are hard negatives, not noise.

### 5.2 Splits and baselines

Split by **domain**, not by problem instance, and report leave-one-domain-out
results. Compare against:

- the current preferred-method/subtask-count order;
- ascending task index in `search_partial`;
- random ordering with the same budget;
- a handcrafted heuristic baseline;
- the learned model with and without the preferred-method feature;
- method ranking only, policy only, and both.

### 5.3 Metrics

Method ranking:

- top-1 and top-5 accuracy over the candidate set;
- mean reciprocal rank of the first successful candidate;
- oracle-normalized rank;
- refinements, recursive calls, wall-clock time, solved instances, and plan
  length after exact validation.

Search policy/value:

- nodes expanded and recursive calls;
- solved instances and goal satisfaction;
- plan length and method count;
- wall-clock time and timeout rate;
- Spearman correlation between predicted value and realized cost-to-go;
- policy agreement with the successful trace.

Every neural result must report the same exact-validator outcome as the
symbolic baseline. A learned component that changes semantics is a bug.

## 6. Integration seams

Both designs should use narrow, optional traits rather than embedding Burn
types in the refiner. `RefinementOptions` is currently `Copy`; adding an
`Arc<dyn ...>` to it would break that property. Pass the optional component as
a separate argument or introduce a `RefinementSession` that borrows the
options and owns the model handles.

```rust
pub struct MethodCandidate {
    pub method: String,
    pub bindings: BTreeMap<String, String>,
    pub applicable: bool,
}

pub struct MethodRankingContext<'a> {
    pub task: &'a TaskNode,
    pub state: &'a BTreeSet<String>,
    pub remaining_budget: usize,
}

pub trait MethodRanker: Debug + Send + Sync {
    fn rank(
        &self,
        context: &MethodRankingContext<'_>,
        candidates: &[MethodCandidate],
    ) -> Option<Vec<f64>>;
}

pub struct SearchContext<'a> {
    pub tasks: &'a [TaskNode],
    pub remaining: &'a [bool],
    pub predecessors: &'a [HashSet<usize>],
    pub state: &'a BTreeSet<String>,
    pub depth: usize,
}

pub trait SearchPolicy: Debug + Send + Sync {
    fn order_ready(&self, context: &SearchContext<'_>) -> Option<Vec<usize>>;
    fn value(&self, context: &SearchContext<'_>) -> Option<f64> {
        None
    }
}
```

Both traits return `None` on missing models, shape errors, or non-finite
outputs. The caller then uses the current deterministic order. Feature
extraction, Burn code, and weight loading live behind the optional `neural`
feature; the default build keeps the existing symbolic path.

## 7. Recommended implementation order

1. **Method ranker** with exact symbolic fallback. It is local, easy to label,
   and has the smallest semantic surface.
2. **Search policy** over the ready set, initially using it only to order
   branches.
3. **Value head** for branch ordering and diagnostics; do not hard-prune until
   an admissible bound or a validated underestimate is available.
4. **Joint model** sharing the encoder between method ranking and frontier
   scoring.
5. **LLM proposals** only if a future setting lacks a formal domain model or
   needs cross-domain priors; keep the symbolic verifier mandatory.

## 8. Current implementation

The first implementation lives in
`src/solver/evaluation_tree/decomposition/` and is enabled by the
`neural-base` feature:

```bash
cargo build --features neural-base --bin pspr
```

The default build has no Burn dependency. The symbolic refiner owns all
applicability checks, state transitions, backtracking, and validation; the
model only reorders candidate methods and ready tasks.

### 8.1 Unified training

`train_decomposition_model` collects traces from the symbolic C+G path and
trains the shared graph encoder with listwise cross-entropy:

- `--task method`: method-ranking loss only;
- `--task policy`: frontier-policy plus value-regression loss only;
- `--task both`: both losses interleaved in one Adam step stream.

```bash
cargo run --features neural-base --bin train_decomposition_model -- \
  --task both \
  --domain ../hddl/test_data/ipc/Blocksworld-GTOHP/domain.hddl \
  --problem ../hddl/test_data/ipc/Blocksworld-GTOHP/p01.hddl \
  --problem ../hddl/test_data/ipc/Blocksworld-GTOHP/p02.hddl \
  --epochs 20 \
  --max-samples 4096 \
  --output target/decomposition-model.burnpack
```

The trainer writes the Burn checkpoint and a `.json` sidecar recording the
trained head(s), hidden width, and feature width. The sidecar is read at
inference time, so a method-only checkpoint never enables the untrained
policy head (and vice versa).

Policy examples require a genuinely unordered method network. Many IPC
domains encode partial orderings as chains, so `--task policy` may report no
usable decision. `tests/fixtures/partial/` is a small two-subtask network that
exercises both heads:

```bash
cargo run --features neural-base --bin train_decomposition_model -- \
  --domain tests/fixtures/partial/domain.hddl \
  --problem tests/fixtures/partial/p01.hddl \
  --task both --epochs 8 --max-samples 64 \
  --output target/decomposition-partial.burnpack
```

### 8.2 Inference

```bash
PSPR_METHOD_RANKER=target/decomposition-method.burnpack \
PSPR_SEARCH_POLICY=target/decomposition-policy.burnpack \
target/debug/pspr --hierarchical --max-groundings 4096 \
  ../hddl/test_data/ipc/Blocksworld-GTOHP/domain.hddl \
  ../hddl/test_data/ipc/Blocksworld-GTOHP/p01.hddl
```

`PSPR_DECOMPOSITION_MODEL` remains a convenience path for a joint checkpoint;
its sidecar selects the enabled heads. Missing or malformed checkpoints are
reported on stderr and ignored, leaving the symbolic path available. A
returned score vector of the wrong shape, non-finite values, or an invalid
permutation also falls back to the deterministic order.

The current model is deliberately compact: three shared GAT layers, hidden
width 64, a candidate-centric method head, a per-ready-task policy head, and a
global value head. Atom nodes, typed relation-specific projections, and
multi-head attention from the long-term designs are not yet implemented; the
graph currently uses task, candidate, frontier, and global nodes.

## 9. References

`velickovic2018gat` for graph attention, `schlichtkrull2018rgcn` for typed
relational message passing, `xu2019gin` for expressiveness, and
`kipf2017gcn` for the GCN baseline. `silver2018alphazero` motivates
policy/value search, `sutton1999options` and `dietterich2000maxq` cover
hierarchical temporal abstraction, and `kambhampati2024llmmodulo`,
`liu2023llmp`, and `ahn2022saycan` cover the proposal-plus-verifier pattern.
`erol1994umcp`, `nau2003shop2`, and `holler2020hddl` provide the HTN and HDDL
background; `elmachtoub2022spoplus` and `donti2017dc3` cover decision-focused
training.
