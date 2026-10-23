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
  solve BW-rand-5: plan length 21 (18 methods, goal Satisfied)
```

PSPR always runs the hierarchical refinement pipeline (C+G): it builds the
top-level task-network plan, refines it into a primitive decomposition, and
reports the final goal status.

Two flags relate to the abstraction gap. `--liftability` prints the top-level
task pairs whose subtask effects establish or destroy a literal that a later
method or action precondition reads. `--repair-orderings` inserts the implied
orderings into the task network before planning, so the abstract plan runs the
producer before the consumer and refinement can thread the atom through. The
hierarchical path also retries with the repair automatically when the first
attempt fails and the analysis finds such an edge.

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
  cargo run --features neural-gpu -- domain.hddl p01.hddl
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
backend for training. The `neural` and `neural-gpu` builds use Rayon to
parse/load, evaluate, and refine independent problems in parallel. Each
problem creates its own SCIP model; use `--jobs 1` if a downstream SCIP build
is not thread-safe. `--jobs N` sets the worker count, and
`RAYON_NUM_THREADS` is the fallback.

```bash
# Collect method, frontier, and permutation traces and train all heads together.
cargo run --features neural-gpu --bin train_decomposition_model -- \
  --task all --jobs 16 --ipc-dir ../hddl/test_data/ipc --epochs 20 \
  --output target/decomposition-model.burnpack

# Use the joint checkpoint during refinement. Separate checkpoints can be
# enabled with PSPR_METHOD_RANKER and PSPR_SEARCH_POLICY.
PSPR_DECOMPOSITION_MODEL=target/decomposition-model.burnpack \
  cargo run --features neural-base -- domain.hddl p01.hddl
```

`--task method`, `--task policy`, `--task permutation`, `--task both`, and
`--task all` choose the training objective. `both` interleaves method and
frontier-policy losses, while `all` interleaves method, frontier-policy, and
permutation losses in one optimizer stream.
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
| `examples/bench_modes.rs` | Machine-readable benchmark for hierarchical refinement |
| `examples/compare_merge_optimizers.rs` | Adam versus SGD, momentum, adaptive, CG, and quasi-Newton merge optimizers |
| `scripts/compare_decomposition.sh` | PSPR-versus-pandaPI benchmark harness |
| `scripts/install_panda.sh` | Builds the pandaPI parser, grounder, and engine under `PANDA_ROOT` |
| `scripts/run_ipc_vs_panda.sh` | Timeout-safe per-instance IPC runner that reports PSPR and PANDA side-by-side |

## Comparing against pandaPI

The comparison harness needs the `pandaPI` toolchain (parser, grounder, and
engine). `scripts/install_panda.sh` builds all three plus the `gengetopt` they
require into a self-contained prefix, so nothing is installed system-wide:

```bash
scripts/install_panda.sh                     # default PANDA_ROOT=/tmp/pandaPI
PANDA_ROOT=/opt/panda scripts/install_panda.sh  # or any other prefix
```

The installer is idempotent, disables the grounder's OpenCL path (the runtime
library is often present without the headers), and patches the grounder for
GCC 13+ where `<algorithm>` is no longer included transitively.

`scripts/run_ipc_vs_panda.sh` then runs the IPC case list one instance at a
time, with per-stage timeouts so a single hard instance cannot stall the whole
run. It reports a markdown table plus `target/panda-comparison/results.csv`:

```bash
scripts/run_ipc_vs_panda.sh
CASE_FILTER='Blocksworld|Depots' scripts/run_ipc_vs_panda.sh
```

`scripts/compare_decomposition.sh` is the older full sweep that also reports
PANDA parse/ground/search splits for every mode of the planner.

## Formula support

The translation of the parser's formula language into leaf planning tasks is
documented in [`docs/unsupported-atoms.md`](docs/unsupported-atoms.md): the
DNF and conditional-effect encoding, the supported constructors, and the
remaining gaps (probabilistic formulas and the unused problem goal).

Compound tasks are solved through hierarchical refinement with primitive method
summaries (C+G).
The remaining approaches, with formal models, pseudo-code and citations, are
surveyed in [`docs/compound-tasks.md`](docs/compound-tasks.md).

## Validation

The decoder has been exercised against all 43 IPC domains and all 767 problem
files used to validate the HDDL-Parser it links to.

## Empirical comparison with PANDA

PSPR now exposes a single planning path: hierarchical refinement with
primitive method summaries (C+G). The benchmark therefore compares this single
PSPR path against PANDA.

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
  hierarchical refinement, and re-executes the result with
  `validate_refinement`. The grounding budget is
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

### Results and interpretation

The benchmark script now reports only hierarchical refinement results for PSPR
plus PANDA as reference. Typical outcomes are:

- PSPR hierarchical refinement solves many IPC instances quickly when method
  branching stays moderate.
- Difficult learned and recursive domains (for example Freecell/Barman/PCP and
  some Minecraft/Monroe instances) remain the primary failure or timeout cases.
- PANDA remains a useful coverage baseline for domains where PSPR's bounded
  refinement search still fails.

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
and `OUT_DIR` from the environment. The script sets
`PSPR_MAX_GROUNDINGS=4096` by default for the benchmark, discovers the
bundled SCIP shared library, and refuses to start if another run holds the
output lock.

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
