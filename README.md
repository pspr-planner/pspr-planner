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
    solve BW-rand-5: stub, plan length 0
```

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
| `src/solver/fast_downward/` | Multi-valued planning tasks, heuristics, and best-first search |
| `src/solver/partition/` | Graph partitioning passes |
| `src/solver/partition/agglomeration/` | Gilpin et al. merge-level ILP and evaluation scheduling |
| `src/solver/partition/bisbm/` | Spectral clustering of bipartite task/constraint graphs |
| `src/main.rs` | CLI pipeline |

## Validation

The decoder has been exercised against all 43 IPC domains and all 767 problem
files used to validate the HDDL-Parser it links to.
