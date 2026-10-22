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

# Refine the abstract task-network plan into a complete HTN plan
pspr path/to/domain.hddl path/to/p01.hddl --hierarchical

# Recursive method-subproblem evaluation and primitive method summaries
pspr path/to/domain.hddl path/to/p01.hddl --recursive --summary
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

`--recursive` uses the recursive method-subproblem path (falling back to the
linear refinement when a subproblem needs compound effects). `--summary`
enables primitive method-effect summaries; it is independent of the refinement
mode and can also be used with the default abstract solver.

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
| `src/solver/evaluation_tree/hierarchical.rs` | Two-phase refinement, recursive evaluation, and strict refinement validation |
| `src/solver/evaluation_tree/leaf.rs` | Leaf encoding and optional primitive method summaries |
| `src/solver/fast_downward/` | Multi-valued planning tasks, heuristics, and best-first search |
| `src/solver/partition/` | Graph partitioning passes |
| `src/solver/partition/agglomeration/` | Gilpin et al. merge-level ILP and evaluation scheduling |
| `src/solver/partition/bisbm/` | Spectral clustering of bipartite task/constraint graphs |
| `src/main.rs` | CLI pipeline |

## Formula support

The translation of the parser's formula language into leaf planning tasks is
documented in [`docs/unsupported-atoms.md`](docs/unsupported-atoms.md): the
DNF and conditional-effect encoding, the supported constructors, and the
remaining gaps (probabilistic formulas and the unused problem goal).

Compound tasks are modelled as opaque operators by default. The implemented
stages add two ways to go beyond that: `--hierarchical` refines the abstract
plan into a complete HTN plan, and `--summary` summarises primitive, totally
ordered methods into macro operators. The remaining approaches, with formal
models, pseudo-code and citations, are surveyed in
[`docs/compound-tasks.md`](docs/compound-tasks.md).

## Validation

The decoder has been exercised against all 43 IPC domains and all 767 problem
files used to validate the HDDL-Parser it links to.

## Empirical comparison with PANDA

PSPR and PANDA do not solve the same problem. PSPR builds the evaluation tree
for the *initial task network* and plans it with compound tasks as opaque
operators; it does not decompose a compound task into primitive actions and it
does not plan for the classical goal. PANDA grounds the complete HTN problem
and searches for a primitive decomposition. The comparison below therefore
measures PSPR's top-level task-network projection against a complete HTN
planner. It is useful for locating where PSPR's partitioning saves work and
for measuring how much of the HTN problem is left to a downstream
decomposition step, but the plan lengths and the speed ratios are not
like-for-like.

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
- **PSPR.** `target/release/examples/bench` parses the HDDL files, runs the
  evaluation-tree solver, and re-executes the result with the independent
  validator. The reported time is the median of five end-to-end runs and
  includes parsing, solving, and validation.
- **PANDA.** `pandaPIparser` (default options), `pandaPIgrounder -q`, and
  `pandaPIengine -t 60` (default progression search: A* with the RC2/FF
  heuristic). The reported total is one wall-clock run including parsing,
  grounding, instance loading, and search. `PANDA search ms` is the engine's
  own search counter and excludes parsing/grounding.
- **Limits.** 60 seconds of PANDA search per instance; no PSPR timeout was
  reached. Both planners ran single-threaded.
- **Platform.** AMD Ryzen AI 7 350 (16 logical CPUs), 22 GiB RAM, Arch Linux
  7.2.11, `rustc` 1.100.0-nightly, GCC 16.2.1. pandaPI commits: parser
  `88c0995` (2024-06-10), grounder `b04ff7b` (2025-01-20), engine `810f043`
  (2024-05-14). The grounder was built with `USE_OPENCL=no` and one
  `#include <algorithm>` line added to `src/parser.cpp` for GCC 16.

### Results

`PSPR goal` is the status produced by `validate_plan`: `absent` means the
problem has no state goal, `satisfied` means the executed plan reaches it,
`violated` means the task-network plan executes but leaves the goal false
(because compound tasks are opaque), and `invalid` means the partial plan does
not execute all grounded tasks. `PSPR steps` counts actions in the task-network
plan; `PANDA actions` counts primitive actions after full decomposition.

| Domain | Problem | PSPR solved | PSPR goal | PSPR steps | PSPR ms | PANDA status | PANDA actions | PANDA search ms | PANDA total ms |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Blocksworld-GTOHP | p01.hddl | true | violated | 3 | 9 | solved | 59 | 6 | 25 |
| Blocksworld-GTOHP | p02.hddl | true | violated | 6 | 12 | solved | 101 | 7 | 29 |
| Blocksworld-GTOHP | p03.hddl | true | violated | 5 | 11 | solved | 102 | 6 | 28 |
| Blocksworld-GTOHP | p04.hddl | true | violated | 14 | 10 | solved | 170 | 14 | 37 |
| Blocksworld-GTOHP | p05.hddl | true | violated | 12 | 12 | solved | 203 | 14 | 39 |
| Depots | p01.hddl | true | violated | 2 | 11 | solved | 36 | 7 | 28 |
| Depots | p02.hddl | true | violated | 4 | 20 | solved | 59 | 9 | 37 |
| Depots | p03.hddl | true | violated | 6 | 23 | solved | 114 | 26 | 60 |
| Satellite-GTOHP | p01.hddl | true | violated | 3 | 11 | solved | 32 | 3 | 22 |
| Satellite-GTOHP | p02.hddl | true | violated | 5 | 22 | solved | 50 | 8 | 31 |
| Satellite-GTOHP | p03.hddl | false | invalid | 3 | 23 | solved | 46 | 8 | 33 |
| Transport | pfile01.hddl | true | absent | 2 | 11 | solved | 15 | 2 | 20 |
| Transport | pfile02.hddl | true | absent | 3 | 11 | solved | 42 | 13 | 32 |
| Transport | pfile03.hddl | true | absent | 3 | 11 | solved | 34 | 11 | 29 |
| PO_Transport | pfile01.hddl | true | absent | 2 | 20 | solved | 15 | 5 | 22 |
| PO_Satellite | 1obs-1sat-1mod.hddl | true | absent | 1 | 11 | solved | 8 | 1 | 18 |
| Freecell-Learned-ECAI-16 | probfreecell-02-1.hddl | true | absent | 4 | 27 | timeout | - | 61780 | 63130 |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-0.hddl | true | absent | 4 | 17 | solved | 280 | 16 | 45 |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-1.hddl | true | absent | 4 | 17 | solved | 264 | 14 | 45 |
| Logistics-Learned-ECAI-16 | probLOGISTICS-04-2.hddl | true | absent | 4 | 16 | solved | 172 | 12 | 43 |
| AssemblyHierarchical | genericLinearProblem_depth01.hddl | true | violated | 1 | 10 | solved | 13 | 3 | 23 |
| Barman-BDI | pfile01.hddl | true | absent | 1 | 11 | solved | 73 | 8 | 35 |
| Barman-BDI | pfile02.hddl | true | absent | 2 | 11 | solved | 162 | 63 | 92 |
| Robot | pfile_01_001.hddl | true | satisfied | 1 | 11 | solved | 3 | 0 | 17 |
| Minecraft-Player | p-003-003-003-003.hddl | true | absent | 1 | 24 | solved | 230 | 660 | 6838 |
| Monroe_PO_1 | pfile01-p-0014-fix-power-line-4.hddl | true | violated | 1 | 11 | solved | 22 | 554 | 5559 |
| PCP_1 | p-pcp01.hddl | true | satisfied | 2 | 16 | solved | 35 | 12 | 30 |

### Interpretation

- **Coverage.** PSPR solved 26 of 27 task networks; the exception is
  `Satellite-GTOHP/p03`, where two leaf tasks remain unsolved and the validator
  rejects the partial plan. PANDA solved 26 of 27 complete HTN problems; its
  exception is `Freecell`, where the 60-second search limit expired after
  251,321 nodes. The timeout says nothing about solvability.
- **What PSPR's solutions mean.** Of the 26 solved networks, 12 have no state
  goal (`absent`), 2 satisfy a state goal (`Robot`, `PCP_1`), and 12 report
  `violated` (the five Blocksworld, three Depots, two Satellite, Assembly, and
  Monroe instances). Those 12 are exactly the domains with compound tasks in
  the initial network: PSPR's plan executes the network but never applies the
  effects of the methods that would decompose the compounds, so the state goal
  stays false. The `absent` rows are no stronger evidence of a complete HTN
  solution; they are simply unconstrained by a state goal. PANDA produces a
  primitive decomposition in every solved case.
- **Plan lengths.** PSPR's plans contain 1--14 network actions; PANDA's contain
  3--280 primitive actions. The comparison shows how much decomposition work
  PSPR leaves out, not that its plans are shorter in the same sense. In the
  two goal-satisfying cases the gap is 1 vs. 3 actions (`Robot`) and 2 vs. 35
  actions (`PCP_1`).
- **Timing.** PSPR stayed in a narrow 9--27 ms band (median 11 ms) across all
  27 instances because its work is local to the clusters. pandaPI ranged from
  17 ms to 6.8 s on solved instances (median 32 ms), plus the 63 s Freecell
  timeout. On the 25 instances solved by both, PSPR totalled 349 ms and
  pandaPI 13.2 s (37.8x aggregate); the median per-instance ratio was 2.6x.
  The aggregate is dominated by grounding and instance loading rather than
  search: `Minecraft-Player` spent 5.9 s grounding a 230-action problem, and
  `Monroe_PO_1` spent 1.3 s grounding and several seconds loading a 10 MB
  SAS+ instance before its 0.55 s search. On the same instances PSPR took
  24 ms and 11 ms. Conversely, pandaPI's search alone (median 9 ms) is often
  as fast as PSPR's whole pipeline; the difference is that PSPR never grounds
  the full problem.
- **Reading the result.** The experiment supports the intended role of the
  evaluation tree: it is a fast top-level projection that partitions and
  orders the task network without committing to a full grounding, and it is
  not a replacement for a complete HTN planner. On these benchmarks it
  identifies network-level plans quickly, but for complete HTN solutions
  PANDA remains the stronger system. The table is a baseline for future work
  on feeding PSPR's partitions into a decomposition step, not a claim of
  superiority.
- **Scope.** These are 27 instances from one benchmark suite, run with one
  PANDA configuration and one random seed. The comparison measures
  satisfiability and wall-clock time, not plan cost or decomposition quality;
  a full evaluation would sweep the PANDA timeout, heuristic, and grounding
  options as well. The table measures the default abstract projection; the
  `--hierarchical`, `--recursive`, and `--summary` modes are exercised
  separately and are not part of these timings.

### Reproducing the benchmark

```bash
cargo build --release --example bench
scripts/compare_panda.sh
```

The script prints the table above and writes per-instance logs plus
`results.csv` to `target/panda-comparison/`. It reads `IPC_DIR`,
`PSPR_BIN`, `HDDL_PARSER`, `PANDA_PARSER`, `PANDA_GROUNDER`, `PANDA_ENGINE`,
`TIME_LIMIT`, and `PSPR_REPEATS` from the environment. PSPR finds the bundled
SCIP shared library automatically. Build pandaPI from the `panda-planner-dev`
`pandaPIparser`, `pandaPIgrounder`, and `pandaPIengine` repositories; the
grounder needs its `cpddl` submodule and `gengetopt`.

Timing numbers vary between runs, especially for pandaPI's grounding and
instance loading; the table records one representative run, and the script's
median-of-five PSPR timings are stable to a few milliseconds.

## References

The bibliography for the algorithms, languages, tools and benchmark data used
by the planner is in [`docs/references.bib`](docs/references.bib). Each entry
has an `annote` field with the places where it should be cited and any caveats
(for example, the SGPlan paper covers the partitioning and penalty
formulation, while the Adam-based relaxed merger is an extension of it). The
PANDA comparison adds `holler2021panda` for the baseline planner and
`behnke2020grounding` for its grounder. The compound-task design note adds
`nau2003shop2`, `alford2009translating`, `behnke2019bringing`,
`korf1985macro`, `erol1996complexity`, and `ghallab2004automated`.
