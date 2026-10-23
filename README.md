# pspr

This repository contains the code for the HTN planner, PSPR.
The project is largely based off of Hsu et al.'s SGPlan, adapted to make use of
block preconditioning and learned heuristics for penalty initialization.

Thanks to Yousefi et al. for [the HDDL parser I am using](https://github.com/koala-planner/HDDL-Parser). Currently I am calling the cli for licensing reasons,
I will be rewriting the parser for optimization and to deal with these issues later.

This program is licensed under the AGPLv3.

I will additionally note that, as a result of the licensing of the HDDL-Parser, this
program can only be used for non-commercial purposes unless modified to make use of
a different parser.

## Requirements

- A Rust toolchain (edition 2024, tested with `rustc` 1.100).
- The `hddl_analyzer` executable from HDDL-Parser.

Build the parser once:

```bash
git clone https://github.com/koala-planner/HDDL-Parser
cd HDDL-Parser
cargo build --release --bin hddl_analyzer
export HDDL_PARSER="$PWD/target/release/hddl_analyzer"
```

## Build and test

```bash
cargo build
cargo test
```

`cargo test` also runs `tests/ipc.rs` against the IPC HTN benchmarks in
`../hddl/test_data/ipc`. The tests are skipped when that directory or the
`hddl_analyzer` executable is missing. Set `PSPR_IPC_DIR` to point at another
benchmark checkout and `HDDL_PARSER` to the parser executable.

Every IPC plan is re-executed with
`pspr::solver::evaluation_tree::validate_plan`, which checks action and method
preconditions, full effect semantics, grounding coverage, the ordering and
equality constraints of the task network, and the problem goal (reported as a
`GoalStatus`).

## Usage

```bash
# Domain + explicit problem files
pspr path/to/domain.hddl path/to/p01.hddl path/to/p02.hddl

# Domain + every sibling problem file (all *.hddl except domain.hddl)
pspr path/to/domain.hddl

# Domain + a directory of problems
pspr path/to/domain.hddl path/to/problems/

# A whole-program JSON export (domain + at most one problem)
pspr path/to/program.json

# Refine with stage C and primitive summaries (C+G)
pspr path/to/domain.hddl path/to/p01.hddl --hierarchical
```

The parser executable is found from the first available source:

1. `--parser PATH`
2. `$HDDL_PARSER`
3. `hddl_analyzer` on `PATH`

Example:

```bash
cargo run -- \
  ../hddl/test_data/ipc/Blocksworld-GTOHP/domain.hddl \
  ../hddl/test_data/ipc/Blocksworld-GTOHP/p01.hddl \
  --parser /tmp/HDDL-Parser/target/release/hddl_analyzer
```

Output:

```
Domain BLOCKS
  1 types, 0 constants, 5 predicates
  5 actions, 4 compound tasks, 8 methods
Loading 1 problem file(s).
  problem BW-rand-5 (5 objects)
    solve BW-rand-5: plan length 3
```

With `--hierarchical`, the same problem prints the complete primitive plan:

```
    solve BW-rand-5: plan length 21 (18 methods, goal Satisfied)
```

`--hierarchical` refines the abstract plan with stage C and primitive method
summaries enabled (C+G). `--summary` without `--hierarchical` enables
primitive method-effect summaries for the abstract solver.

### Neural merge warm start (optional)

The `neural` feature builds the Design 1 heterogeneous GAT that predicts the
start positions and penalty weights of the SGPlan merger. It is off by default,
so the sequential initialization remains the default and has no Burn
dependency. The `neural-gpu` feature adds Burn's `wgpu` backend and makes the
training binary run on the GPU:

```bash
# Collect teacher labels, train on the GPU, and write the model plus metadata.
cargo run --features neural-gpu --bin train_merge_initializer -- \
  --ipc-dir ../hddl/test_data/ipc \
  --epochs 40 --output target/merge-initializer.burnpack

# Use the trained model during planning (also on the GPU).
PSPR_MERGE_MODEL=target/merge-initializer.burnpack \
  cargo run --features neural-gpu -- --hierarchical domain.hddl p01.hddl
```

Use `--features neural` instead for the CPU-only training path. The training
binary prints the selected device before starting.

`train_merge_initializer` supports `--domain`/`--problem` for a single pair,
`--max-samples`, `--max-groundings`, `--batch-size`, `--hidden`, and
`--learning-rate`. The CPU path uses Rayon to evaluate each mini-batch in
parallel and averages the per-sample gradients before one optimizer step;
`--batch-size` controls the mini-batch size and `RAYON_NUM_THREADS` controls
the worker count. The GPU path ignores `--batch-size` and keeps one optimizer
step per sample.
Its `--max-groundings` default is `32768`, rather than the planner's smaller
default, because the larger IPC Assembly instances need roughly 28k method
groundings for one compound task; pass a smaller value for faster collection.
Training prints one progress line per step with overall progress, current
epoch and sample, current loss, elapsed time, and an ETA based on the average
observed step time. Epoch summaries and a final best-loss summary follow.
Predictions are validated for shape and finiteness; the merger falls back to
the sequential start and still runs the exact validator.

### Neural decomposition guidance (optional)

The `neural-base` feature builds the unified method-ranker and frontier
policy/value GAT used by the C+G refiner. It is off by default, so the
symbolic C+G path has no Burn dependency. `neural-gpu` selects Burn's `wgpu`
backend for training.

```bash
# Collect method and frontier traces and train both heads together.
cargo run --features neural-base --bin train_decomposition_model -- \
  --task both --ipc-dir ../hddl/test_data/ipc --epochs 20 \
  --output target/decomposition-model.burnpack

# Use the joint checkpoint during refinement. Separate checkpoints can be
# enabled with PSPR_METHOD_RANKER and PSPR_SEARCH_POLICY.
PSPR_DECOMPOSITION_MODEL=target/decomposition-model.burnpack \
  cargo run --features neural-base -- --hierarchical domain.hddl p01.hddl
```

`--task method`, `--task policy`, and `--task both` choose the training
objective; `both` interleaves both listwise losses in one optimizer stream.
The trainer writes a `.json` sidecar recording the heads that actually had
training examples, and inference uses it so an untrained head is never
activated. Method ranking only reorders applicable methods; the policy only
reorders ready tasks; wrong-length, non-finite, or non-permutation outputs
fall back to the symbolic order. `PSPR_DECOMPOSITION_MODEL` remains available
for a single joint checkpoint. See
[`docs/neural-decomposition/README.md`](docs/neural-decomposition/README.md)
for details and the unordered test fixture used to exercise policy training.

## Layout

| Module | Role |
| --- | --- |
| `src/parsing/` | Input discovery, parser dispatch, and JSON decoding |
| `src/parsing/json.rs` | Mirror of the HDDL-Parser JSON AST with `serde` decoding |
| `src/parsing/parser.rs` | Locating and invoking `hddl_analyzer convert --to json` |
| `src/parsing/discovery.rs` | Finding the problem files that belong to a domain |
| `src/solver/` | Solver-facing representations and entry point |
| `src/solver/domain.rs` | `process_domain`: parser AST to solver-facing domain |
| `src/solver/problem.rs` | `load_problem`: parser AST to solver-facing problem |
| `src/solver/solve.rs` | `solve`: evaluation-tree entry point |
| `src/solver/evaluation_tree/` | Incidence graph, clustering/tree, leaf solving, and merging |
| `src/solver/evaluation_tree/hierarchical.rs` | Two-phase refinement with summaries, and strict refinement validation |
| `src/solver/evaluation_tree/leaf.rs` | Leaf encoding and optional primitive method summaries |
| `src/solver/fast_downward/` | Multi-valued planning tasks, heuristics, and best-first search |
| `src/solver/partition/` | Graph partitioning passes |
| `src/solver/partition/agglomeration/` | Gilpin et al. merge-level ILP and evaluation scheduling |
| `src/solver/partition/bisbm/` | Spectral clustering of bipartite task/constraint graphs |
| `src/main.rs` | CLI pipeline |
| `src/bin/train_merge_initializer.rs` | Offline Burn training for the neural merge warm start |
| `src/solver/evaluation_tree/merge/neural.rs` | Design 1 graph features, GAT, and inference initializer |
| `src/bin/train_decomposition_model.rs` | Unified method-ranker and frontier-policy/value trainer |
| `src/solver/evaluation_tree/decomposition/` | Decomposition graph features, shared GAT, traits, and inference wrappers |
| `examples/bench_modes.rs` | Machine-readable benchmark for the three compound-task modes |
| `examples/compare_merge_optimizers.rs` | Adam versus SGD, momentum, adaptive, CG, and quasi-Newton merge optimizers |
| `scripts/compare_decomposition.sh` | PSPR-versus-pandaPI benchmark harness |

## Formula support

The translation of the parser's formula language into leaf planning tasks is
documented in [`docs/unsupported-atoms.md`](docs/unsupported-atoms.md): the
DNF and conditional-effect encoding, the supported constructors, and the
remaining gaps (probabilistic formulas and the unused problem goal).

Compound tasks are modelled as opaque operators by default. The implemented
stages go beyond that in two ways: `--hierarchical` refines the abstract plan
into a complete HTN plan with primitive summaries (C+G), and `--summary`
summarises primitive, totally ordered methods into macro operators.
The remaining approaches, with formal models, pseudo-code and citations, are
surveyed in [`docs/compound-tasks.md`](docs/compound-tasks.md).

## Validation

The decoder has been exercised against all 43 IPC domains and all 767 problem
files used to validate the HDDL-Parser it links to.

## Empirical comparison with PANDA

PSPR can now solve the initial task network in three modes. They differ in how
much of the hierarchy is executed, and only the refinement mode guarantees a
primitive decomposition when the network contains compound tasks:

| Mode | Technique | What it produces |
| --- | --- | --- |
| `abstract` | opaque compound tasks | a plan over the initial task network |
| `summary` | primitive method summaries (stage G) | the same network plan, with macro operators |
| `hierarchical` | two-phase refinement with summaries (C+G) | a complete primitive decomposition |

`abstract` and `summary` measure PSPR's top-level projection: they execute the
network but do not apply the effects of the methods that would decompose its
compound tasks, so they are not complete HTN planners. `hierarchical` is the
fair comparison to PANDA, because it refines the abstract plan into primitive
actions, with primitive method summaries enabled, and checks the state goal.

The baseline is the PANDA framework (`holler2021panda`) as implemented by the
maintained `pandaPI` planner, using its grounding algorithm
(`behnke2020grounding`); both entries are annotated in
[`docs/references.bib`](docs/references.bib). The older 2019 PANDA3 jar was
not used: its HDDL front end rejects most of the IPC domains with
`Error: could not find task name`.

### Method

- **Instances.** The 26 successful cases in `tests/ipc.rs` plus
  `Satellite-GTOHP/p03.hddl`, the known partial-plan case, for 27 instances in
  total. They cover totally ordered and partially ordered networks, state
  goals, numeric fluents, learned HTN models, and domains whose initial
  networks contain compound tasks.
- **PSPR.** `target/release/examples/bench_modes` parses the HDDL files, runs
  the selected mode, and re-executes the result with the independent
  validator (`validate_plan` for `abstract`/`summary`, `validate_refinement`
  for `hierarchical`). The hierarchical mode enables
  `method_summary=PrimitiveOnly` unconditionally (C+G). The grounding budget is
  `PSPR_MAX_GROUNDINGS=4096` for the benchmark; the CLI default remains 256.
  Each successful PSPR run is repeated three times and the median end-to-end
  time is reported. A run that exceeds `PSPR_TIMEOUT=30` seconds is reported as
  a timeout and is not repeated. All times include parsing, solving, and
  validation.
- **PANDA.** `pandaPIparser` (default options), `pandaPIgrounder -q`, and
  `pandaPIengine -t 60` (default progression search: A* with the RC2/FF
  heuristic). The reported total is one wall-clock run including parsing,
  grounding, instance loading, and search. `PANDA actions` is the engine's
  `Total costs of actions`, which counts primitive domain actions and is
  comparable to PSPR's plan length. The engine's `Found solution of length`
  also counts method and precondition operations, so it is deliberately not
  used in the table.
- **Limits.** 30 seconds of PSPR solving per instance and 60 seconds of PANDA
  search per instance. Both planners ran single-threaded.
- **Platform.** AMD Ryzen AI 7 350 (16 logical CPUs), 22 GiB RAM, Arch Linux
  7.2.11, `rustc` 1.100.0-nightly, GCC 16.2.1. pandaPI commits: parser
  `88c0995` (2024-06-10), grounder `b04ff7b` (2025-01-20), engine `810f043`
  (2024-05-14). The grounder was built with `USE_OPENCL=no` and one
  `#include <algorithm>` line added to `src/parser.cpp` for GCC 16.

### Results

In the per-instance table, `S(n)` means solved with `n` primitive actions (or
network steps for `Abstract` and `G`) and the state goal satisfied or absent;
`V(n)` means the task network was solved but the state goal is violated
(because compound tasks are opaque); `F` means the mode failed; `T` means the
mode hit its time limit. For `Abstract` and `G`, `V` is expected and is not a
complete HTN solution. PANDA's count is its primitive action cost.

| Domain | Problem | Abstract | G | C+G | PANDA |
| --- | --- | --- | --- | --- | --- |
| Blocksworld-GTOHP | p01.hddl | V(3) | V(3) | S(21) | S(21) |
| Blocksworld-GTOHP | p02.hddl | V(6) | V(6) | S(35) | S(35) |
| Blocksworld-GTOHP | p03.hddl | V(5) | V(5) | S(39) | S(39) |
| Blocksworld-GTOHP | p04.hddl | V(14) | V(14) | S(98) | S(62) |
| Blocksworld-GTOHP | p05.hddl | V(12) | V(12) | S(71) | S(71) |
| Depots | p01.hddl | V(2) | V(2) | S(15) | S(15) |
| Depots | p02.hddl | V(4) | V(4) | S(24) | S(24) |
| Depots | p03.hddl | V(6) | V(6) | S(44) | S(44) |
| Satellite-GTOHP | p01.hddl | V(3) | V(3) | S(12) | S(12) |
| Satellite-GTOHP | p02.hddl | V(5) | V(5) | S(26) | S(18) |
| Satellite-GTOHP | p03.hddl | F | V(5) | S(16) | S(16) |
| Transport | pfile01.hddl | S(2) | S(2) | S(8) | S(8) |
| Transport | pfile02.hddl | S(3) | S(3) | S(19) | S(19) |
| Transport | pfile03.hddl | S(3) | S(3) | S(15) | S(15) |
| PO_Transport | pfile01.hddl | S(2) | S(2) | S(8) | S(8) |
| PO_Satellite | 1obs-1sat-1mod.hddl | S(1) | S(1) | S(5) | S(5) |
| Freecell-Learned-ECAI-16 | probfreecell-02-1.hddl | S(4) | S(4) | F | T |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-0.hddl | S(4) | S(4) | S(63) | S(63) |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-1.hddl | S(4) | S(4) | S(58) | S(58) |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-2.hddl | S(4) | S(4) | S(40) | S(40) |
| AssemblyHierarchical | genericLinearProblem_depth01.hddl | V(1) | V(1) | S(6) | S(4) |
| Barman-BDI | pfile01.hddl | S(1) | S(1) | F | S(10) |
| Barman-BDI | pfile02.hddl | S(2) | S(2) | F | S(25) |
| Robot | pfile_01_001.hddl | S(1) | S(1) | S(0) | S(0) |
| Minecraft-Player | p-003-003-003-003.hddl | S(1) | S(1) | F | S(35) |
| Monroe_PO_1 | pfile01-p-0014-fix-power-line-4.hddl | V(1) | V(1) | F | S(11) |
| PCP_1 | p-pcp01.hddl | S(2) | S(2) | F | S(26) |

| Technique | Solved | State goal satisfied/absent | Complete HTN plans | Median ms | Mean ms | Total ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abstract | 26/27 | 14/27 | 0/27 | 11 | 15 | 393 | 25 |
| G | 27/27 | 14/27 | 0/27 | 12 | 15 | 394 | 24 |
| C+G | 21/27 | 21/27 | 21/27 | 26 | 2366 | 63882 | 30073 |
| PANDA | 26/27 | 26/27 | 26/27 | 31 | 530 | 13772 | 6851 |

`PANDA` times exclude the Freecell timeout (60 s) and its `Max ms` entry is
its slowest solved instance. The PSPR means and totals are dominated by the
two 30-second Minecraft and Monroe runs; the medians are not.

### Interpretation

- **Projection versus completion.** The abstract projection still solves 26 of
  27 task networks with a median of 12 ms, and method summaries solve all 27
  task networks. But only 14 of the 27 instances reach a state goal or have no
  state goal: the other 12 execute the network with opaque compound tasks and
  leave the goal false. Those 12 are exactly the domains with compound tasks
  in the initial network (five Blocksworld, three Depots, two Satellite,
  Assembly, and Monroe), so the abstract and summary rows are measures of
  network-level work, not complete HTN solutions.
- **Coverage of complete plans.** C+G solves 21 of 27 HTN problems. PANDA
  solves 26 of 27, its only miss being the Freecell search timeout. Recursive
  evaluation was removed after the previous benchmark showed that it matched
  C+G coverage while adding overhead.
- **Summary operators.** G alone changes one task network from unsolved to
  solved (`Satellite-GTOHP/p03`), but that plan still violates the state goal.
  Combined with refinement, C+G solves `p03` completely. G has no effect on
  most other domains because only methods with primitive, totally ordered,
  unconditional subtasks are summarised. C+G was faster in the median than
  the adaptive C-first path and at least as complete as C alone, which is why
  hierarchical solving enables summaries directly.
- **Plan quality.** On the 21 instances solved completely by both C+G and
  PANDA, PSPR's primitive plan length equals PANDA's `Total costs of actions`
  in 18 cases. The exceptions are `Blocksworld-GTOHP/p04` (98 vs. 62 actions),
  `Satellite-GTOHP/p02` (26 vs. 18), and `AssemblyHierarchical/depth01`
  (6 vs. 4). PSPR returns the first goal-reaching decomposition found by its
  bounded search, not a cost-optimal one; PANDA's progression search optimises
  its path cost. The 18 exact matches show that the refinement is producing
  the expected decompositions, not merely any valid plan.
- **Timing.** On the 21 instances solved completely by both planners, C+G took
  3.72 s in total and PANDA 0.65 s. PANDA is therefore about 5.7 times faster
  in aggregate on this common set. The per-instance median is closer: C+G is
  26 ms and PANDA is 31 ms. PSPR pays a visible backtracking cost on a few
  instances (`Blocksworld/p04` 0.93 s, `Blocksworld/p05` 0.82 s,
  `Depots/p03` 0.55 s), while PANDA's aggregate is dominated by grounding and
  instance loading (`Minecraft-Player` 6.3 s, `Monroe_PO_1` about 6 s).
  PANDA's search alone has a median of 11 ms. The old abstract-only figure of
  37.8x against PANDA measured the projection, not a complete HTN solution,
  and is not comparable to the C+G row.
- **Where refinement fails.** `Satellite-GTOHP/p03` is unsolved by the abstract
  tree without summaries; C+G builds a different tree and solves it. The
  learned Freecell model exceeds the 4,096-method grounding budget on
  `ACHIEVE-HOME0(S2)`, and `Minecraft-Player` and `Monroe_PO_1` hit the
  30-second refinement limit. Barman's level-dependent pouring methods and
  PCP's recursive task network do not refine at all: their method choices
  depend on primitive preconditions and level indices, which the current
  bounded network search does not explore. These are the concrete cases that
  motivate the progression and exact-encoding fallbacks in
  [`docs/compound-tasks.md`](docs/compound-tasks.md).
- **Implementation note.** Refinement now enumerates up to 2,048 alternative
  partial refinements per level and backtracks when a decomposition leaves the
  state goal false. This is what solves `Blocksworld-GTOHP/p02` and `p05`
  (versus the earlier first-success search) and what keeps the
  `Transport/pfile02` `get_to` methods from cycling. The search is bounded,
  not exhaustive: the caps are the reason some hard instances fail or time out
  rather than run forever.
- **Scope.** These are 27 instances from one benchmark suite, run with one
  PANDA configuration and one random seed. The comparison measures
  satisfiability, primitive plan length, and wall-clock time, not decomposition
  quality or plan cost optimality; a full evaluation would sweep the PANDA
  timeout, heuristic, grounding options, and PSPR's grounding and refinement
  budgets as well. The 60-second PANDA timeout for Freecell says nothing about
  solvability, and the 30-second PSPR limit is a benchmark choice, not a
  property of the refinement algorithm.

### Reproducing the benchmark

```bash
cargo build --release --example bench_modes
scripts/compare_decomposition.sh
```

The script prints the per-instance and per-technique tables and writes
per-instance logs plus `results.csv` to `target/decomposition-comparison/`.
It reads `IPC_DIR`, `PSPR_BIN`, `HDDL_PARSER`, `PANDA_PARSER`,
`PANDA_GROUNDER`, `PANDA_ENGINE`, `TIME_LIMIT` (PANDA search seconds),
`PSPR_TIMEOUT` (PSPR solving seconds), `PSPR_REPEATS`, `PSPR_MAX_GROUNDINGS`,
`PSPR_MODES`, and `OUT_DIR` from the environment. The script sets
`PSPR_MAX_GROUNDINGS=4096` by default for the benchmark, discovers the
bundled SCIP shared library, and refuses to start if another run holds the
output lock. Set `PSPR_MODES="hierarchical"` to run only one technique.

Build pandaPI from the `panda-planner-dev` `pandaPIparser`, `pandaPIgrounder`,
and `pandaPIengine` repositories; the grounder needs its `cpddl` submodule and
`gengetopt`. Timing numbers vary between runs, especially for pandaPI's
grounding and instance loading; the table records one representative run, and
the script's median-of-three PSPR timings are stable to a few milliseconds.
The two 30-second Minecraft and Monroe timeouts per refinement mode dominate
the means and totals, so use the medians when comparing typical behaviour.

## Design notes

- [Neural initialization of the SGPlan merge solver](docs/neural-merge-initialization/README.md):
  a longlist of ten GAT/GCN approaches, a scored selection of the top three,
  and three design docs for learned warm starts of the merge solver followed by
  Adam refinement.
- [Neural guidance for compound-task decomposition](docs/neural-decomposition/README.md):
  five method-ranking GAT designs and five search-policy/value designs, with
  selected design docs for a state-conditioned relational method ranker and a
  frontier GNN policy/value for partial-order refinement.
- [Merge-solver optimizer comparison](docs/merge-optimizer-comparison.md):
  Adam versus projected SGD, momentum, Nesterov, RMSProp, Adagrad, BFGS,
  L-BFGS, and nonlinear CG on real and synthetic merge problems.

## References

The bibliography for the algorithms, languages, tools and benchmark data used
by the planner is in [`docs/references.bib`](docs/references.bib). Each entry
has an `annote` field with the places where it should be cited and any caveats
(for example, the SGPlan paper covers the partitioning and penalty
formulation, while the Adam-based relaxed merger is an extension of it). The
neural merge-initialization design note adds `kipf2017gcn`,
`velickovic2018gat`, `hamilton2017graphsage`, `schlichtkrull2018rgcn`,
`xu2019gin`, `cuturi2013sinkhorn`, `elmachtoub2022spoplus`, `donti2017dc3`, and
`agrawal2019differentiable`. The neural-decomposition design note adds
`silver2018alphazero`, `sutton1999options`, `dietterich2000maxq`,
`kambhampati2024llmmodulo`, `liu2023llmp`, `ahn2022saycan`,
`manhaeve2018deepproblog`, and `ellis2021dreamcoder`. The
PANDA comparison adds `holler2021panda` for the baseline planner and
`behnke2020grounding` for its grounder. The compound-task design note adds
`nau2003shop2`, `alford2009translating`, `behnke2019bringing`,
`korf1985macro`, `erol1996complexity`, and `ghallab2004automated`.
