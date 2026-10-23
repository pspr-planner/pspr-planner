# Merge-solver optimizer comparison

Status: empirical result, 2026-10-22. Adam remains the default; the comparison
backend is selectable through `SgplanMerger::with_optimizer`.

## 1. What was compared

The SGPlan merge solver minimizes the same relaxed objective for every
optimizer:

\[
J(q)=\lambda_{\text{cost}}\operatorname{smoothmax}_c(q_c+l_c/L)
+\sum_k w_k\,\phi(h_k(q)).
\]

The comparison used the existing multi-space continuation and adaptive penalty
weights. Every backend received the same number of outer rounds and inner
steps (`iterations = 10`, `adam_steps = 200`), the same bounds `q in [0,1]`,
and the same starting point. The only difference was the optimizer used inside
each round.

Backends:

| Name | Family |
| --- | --- |
| `adam` | Adaptive first-order baseline |
| `sgd` | Projected gradient descent |
| `momentum` | Heavy-ball momentum |
| `nesterov` | Nesterov accelerated gradient |
| `rmsprop` | Adaptive squared-gradient scaling |
| `adagrad` | Accumulated squared-gradient scaling |
| `lbfgs` | Limited-memory BFGS with projected backtracking line search |
| `bfgs` | Full BFGS with projected backtracking line search |
| `cg` | Polak-Ribiere nonlinear conjugate gradient |

The quasi-Newton methods use a backtracking Armijo line search. All updates
project onto the same box. Continuation resets the optimizer state when the
penalty space changes.

## 2. Real IPC merge nodes

The C+G pipeline produced 11 merge problems from the built-in IPC case list.
Most of these are already feasible from the sequential start, so the optimizer
rarely has room to change the outcome.

Command:

```bash
cargo build --release --example compare_merge_optimizers
LD_LIBRARY_PATH=... target/release/examples/compare_merge_optimizers --repeats 3
```

| Optimizer | Runs | Feasible | Mean violation | Mean relaxed objective | Mean plan | Mean rounds | Median ms | Mean ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adam | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.002 | 0.018 | 0.077 |
| sgd | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.012 | 0.044 |
| momentum | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.013 | 0.046 |
| nesterov | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.012 | 0.045 |
| rmsprop | 33 | 33/33 | 0.000e0 | 1.056e0 | 4.00 | 0.5 | 0.001 | 0.013 | 0.047 |
| adagrad | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.013 | 0.046 |
| lbfgs | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.237 | 0.897 |
| bfgs | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.240 | 0.909 |
| cg | 33 | 33/33 | 0.000e0 | 1.053e0 | 4.00 | 0.5 | 0.001 | 0.164 | 0.607 |

There is no meaningful difference on these cases: the sequential start is
already feasible, so most runs stop after zero or one optimizer round. The
quasi-Newton methods are slower because they are not called often enough for
their better local convergence to pay off.

## 3. Synthetic reverse-order stress suite

The real merge nodes do not exercise the optimizers enough, so the example also
contains a synthetic suite of partially ordered networks whose constraints
require reversing the sequential child order. The suite varies the number of
children (4–24) and child lengths. It is deliberately adversarial for the
initialization, not a representative IPC distribution.

Command:

```bash
target/release/examples/compare_merge_optimizers --synthetic --repeats 3
```

| Optimizer | Runs | Feasible | Mean violation | Mean relaxed objective | Mean plan | Mean rounds | Median ms | Mean ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adam | 27 | 21/27 | 2.222e-1 | 1.444e0 | 27.14 | 4.9 | 0.255 | 0.652 | 1.804 |
| sgd | 27 | 21/27 | 1.444e0 | 2.215e4 | 35.71 | 7.3 | 0.969 | 0.951 | 1.959 |
| momentum | 27 | 21/27 | 2.222e-1 | 1.280e1 | 27.14 | 4.9 | 0.207 | 0.609 | 1.790 |
| nesterov | 27 | 3/27 | 8.111e0 | 8.402e-1 | 10.00 | 9.1 | 1.005 | 1.107 | 2.024 |
| rmsprop | 27 | 27/27 | 0.000e0 | 2.094e0 | 34.44 | 6.9 | 0.722 | 0.695 | 1.612 |
| adagrad | 27 | 27/27 | 0.000e0 | 1.366e0 | 34.44 | 5.0 | 0.219 | 0.640 | 1.816 |
| lbfgs | 27 | 21/27 | 2.222e-1 | 1.448e0 | 27.14 | 5.2 | 2.463 | 6.239 | 15.739 |
| bfgs | 27 | 21/27 | 2.222e-1 | 1.442e0 | 27.14 | 4.9 | 3.204 | 8.364 | 23.924 |
| cg | 27 | 21/27 | 2.222e-1 | 1.436e0 | 27.14 | 4.9 | 1.487 | 3.770 | 10.611 |

### Interpretation

- **RMSProp and Adagrad were the most robust** on the reverse-order suite:
  both solved all 27 runs, while Adam solved 21/27.
- **Adam was faster than RMSProp/Adagrad** in median time, and matched their
  success rate on the real IPC merge nodes.
- **Momentum behaved like Adam on the easy cases** but was less stable on the
  24-child cases.
- **Nesterov was unstable with the shared learning rate**. It solved only the
  smallest cases. This is a hyperparameter-sensitivity result, not evidence
  that Nesterov is inherently unsuitable.
- **SGD often reached feasibility but produced much larger makespan
  objectives**, showing why an adaptive method is preferable with penalty
  weights spanning several orders of magnitude.
- **BFGS, L-BFGS, and CG did not improve success** over Adam under this
  continuation. The box projection and the changing penalty weights break the
  smooth-curvature assumptions behind quasi-Newton updates, and line search
  overhead makes them 3–10× slower in mean time.
- The **mean relaxed objective is not directly comparable across optimizers**
  because it is evaluated at different relaxed points and adapted weights. Use
  feasibility, mean plan length, rounds, and wall-clock time as the primary
  comparison. `Mean plan` is over feasible runs only, so an optimizer that
  fails the hardest cases can show a smaller mean plan.

## 4. Recommendation

Keep **Adam as the default**. It is the fastest robust option on the real merge
nodes and remains competitive under the continuation.

For a second empirical pass, the most promising alternative is **RMSProp or
Adagrad** on larger merge nodes where Adam gets stuck. The next experiment
should:

1. tune the learning rate per optimizer instead of sharing Adam’s `0.05`;
2. evaluate per-domain, not only aggregate;
3. report the exact decoded plan and makespan for every instance;
4. add a projected L-BFGS-B implementation rather than the current projected
   line-search approximation;
5. test whether the hard 24-child failures are caused by bounds, the
   continuation schedule, or the penalty-space reset.

## 5. Code

- `src/solver/evaluation_tree/merge/optimizer.rs`: optimizer implementations.
- `src/solver/evaluation_tree/merge/sgplan.rs`: `OptimizerKind` field,
  `with_optimizer`, and the continuation dispatch.
- `examples/compare_merge_optimizers.rs`: collection and comparison harness.

Default behavior is unchanged:

```rust
let merger = SgplanMerger::default(); // Adam
let merger = SgplanMerger::default().with_optimizer(OptimizerKind::RmsProp);
```

The harness supports `--synthetic`, `--repeats`, `--optimizer`, `--domain`,
and `--problem`.
