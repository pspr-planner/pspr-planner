//! Shared-constraint merging across the evaluation tree.
//!
//! # The SGPlan penalty formulation
//!
//! SGPlan (Hsu et al., *Constraint partitioning in penalty formulations for
//! solving temporal planning problems*, Artificial Intelligence 170(3), 2006)
//! solves a planning problem by partitioning its constraints into subproblems
//! `P_1, ..., P_k`. Subproblem `i` is solved under the weighted objective
//!
//! ```text
//! minimize  cost_i(pi_i) + sum_{c in shared(i)} w_c * v_c(pi)
//! ```
//!
//! where `v_c` is the violation of a shared constraint and `w_c` its penalty
//! weight. The subplans are sequences of discrete operators, the schedule
//! positions are continuous, and the makespan that `cost_i` measures is a
//! maximum over positions. The combined problem
//!
//! ```text
//! minimize  sum_i cost_i(pi_i) + sum_c w_c * v_c(pi)
//! subject to v_c(pi) = 0          for every shared constraint c
//!            pi_i in Plans(P_i)
//!            w_c >= 0
//! ```
//!
//! is therefore a mixed-integer nonlinear program (MINLP).
//!
//! # Continuous relaxation
//!
//! [`SgplanMerger`] relaxes the MINLP to a smooth problem over one start
//! position `q_c in [0, 1]` per child plan. A child with `l_c` steps started at
//! `q_c` occupies the normalized interval `[q_c, q_c + l_c / L]`, where `L` is
//! the total number of plan steps of the merge node, so the discrete decision
//! "child `c` runs before child `d`" becomes the continuous comparison
//! `q_c < q_d`. The remaining nonlinearities are smoothed:
//!
//! * the makespan `max_c (q_c + l_c / L)` becomes a log-sum-exp smooth
//!   maximum, whose gradient is the softmax over the children;
//! * an ordering constraint contributes the one-sided smooth hinge
//!   `h = T * softplus(-g / T)` of its signed margin
//!   `g = t_after - t_before - 1/L` (satisfied when `g >= 0`), so `dh/dg` is
//!   minus the logistic function and the penalty stops pushing once the
//!   constraint holds;
//! * atom and equality constraints do not depend on the schedule. They are
//!   fixed by what the children establish and enter the objective as
//!   constants; they still drive feasibility and the weight adaptation.
//!
//! The relaxed objective
//!
//! ```text
//! J(q) = cost_weight * smoothmax_c(q_c + l_c / L)
//!      + sum_c w_c * phi(h_c(q))
//! ```
//!
//! is minimized with Adam ([`AdamOptions`]), which normalizes the step size per
//! coordinate and therefore tolerates penalty weights that span several orders
//! of magnitude. The final `q` is decoded into a concrete interleaving: the
//! ready plan step with the smallest relaxed time is scheduled first, which
//! keeps the order inside every child, and the resulting plan is validated
//! against the exact constraints.
//!
//! # Multiple penalty spaces and adaptive weights
//!
//! One penalty function is always a compromise: a linear penalty leaves a
//! constant force at the boundary, a quadratic one vanishes exactly where the
//! pressure is needed and becomes ill-conditioned for large weights, and an
//! exponential one is numerically hostile early on. The merger therefore runs
//! a continuation over [`PenaltySpace`]s (`Linear -> Quadratic -> Exponential`
//! by default):
//!
//! 1. every outer round runs [`SgplanMerger::adam_steps`] Adam steps in the
//!    current space;
//! 2. the *exact* violations are measured afterwards — the smooth hinge is
//!    positive at the boundary, so only the exact margin decides feasibility;
//! 3. weights of violated constraints are multiplied by
//!    [`SgplanMerger::penalty_growth`], weights of satisfied constraints are
//!    decayed towards [`SgplanMerger::min_penalty`], which keeps the active set
//!    well conditioned;
//! 4. a round that fails to improve the violation by
//!    [`SgplanMerger::min_progress`] escalates to the next penalty space and
//!    resets Adam's moments so the new curvature is measured from scratch;
//!    stalled rounds therefore raise the pressure instead of oscillating;
//! 5. the best iterate of the run (lowest exact violation, then lowest
//!    objective) is kept, so a late divergent round cannot degrade the result.
//!
//! [`SgplanMerger::report`] exposes the rounds, the final penalty space, the
//! adapted weights, and the residuals of the last merge.
//!
//! # Modelling limitations
//!
//! Ordering constraints are optimized; atom and equality constraints cannot be
//! repaired by reordering the children, so a merge whose child plans do not
//! establish one of them is reported as an error. Ordering constraints whose
//! tasks cannot be located in the child plans are counted in
//! [`PenaltyReport::unresolved`] and treated as satisfied.

use std::collections::BTreeSet;

use super::graph::ConstraintNode;

mod adam;
#[cfg(feature = "neural-base")]
mod neural;
mod optimizer;
mod penalty;
mod sgplan;

pub use adam::AdamOptions;
#[cfg(feature = "neural-base")]
pub use neural::{
    DEFAULT_HEADS, DEFAULT_HIDDEN, FEATURE_DIM, GatModel, MergeGraph, NeuralMergeInitializer,
    TrainingSample,
};
pub use optimizer::OptimizerKind;
pub use penalty::PenaltySpace;
pub use sgplan::SgplanMerger;

/// A learned or heuristic warm start for the relaxed merge problem.
#[derive(Debug, Clone, Default)]
pub struct MergeInitialization {
    /// One normalized start position per child, in `[0, 1]`.
    pub positions: Vec<f64>,
    /// One positive penalty weight per shared constraint.
    pub weights: Vec<f64>,
    /// Optional first penalty space to try.
    pub space: Option<PenaltySpace>,
}

/// A predictor of merge-solver parameters.
pub trait MergeInitializer: std::fmt::Debug + Send + Sync {
    /// Return a warm start for `problem`, or `None` to use the sequential
    /// initialization.
    fn initialize(&self, problem: &MergeProblem) -> Option<MergeInitialization>;
}

/// A solution of a leaf or of a merge node.
#[derive(Debug, Clone)]
pub struct PartialSolution {
    pub label: String,
    pub plan: Vec<String>,
    /// Grounded-task index of every plan step.
    ///
    /// Step `i` executes the task `plan_tasks[i]` of the evaluation tree's
    /// grounded task list, which uses the same indices as
    /// [`ConstraintNode::Ordering`]. The vector is parallel to
    /// [`PartialSolution::plan`].
    pub plan_tasks: Vec<usize>,
    /// Local atoms true at the end of the plan.
    pub final_atoms: BTreeSet<String>,
    /// Ground effect atoms of the plan's operators (used to provision shared
    /// constraints for the merge check).
    pub achieved: BTreeSet<String>,
    pub solved: bool,
}

/// The input of a [`MergeSolver`].
#[derive(Debug, Clone)]
pub struct MergeProblem {
    /// Evaluation id of the merge node.
    pub node: usize,
    /// Constraints shared by the children of this node.
    pub shared_constraints: Vec<ConstraintNode>,
    pub children: Vec<PartialSolution>,
    /// Atoms true in the initial state of the whole problem.
    pub initial_atoms: BTreeSet<String>,
}

/// The result of merging the children of a node.
#[derive(Debug, Clone)]
pub struct MergeOutcome {
    pub plan: Vec<String>,
    /// Grounded-task index of every plan step; parallel to
    /// [`MergeOutcome::plan`].
    pub plan_tasks: Vec<usize>,
    pub final_atoms: BTreeSet<String>,
    pub solved: bool,
}

/// A merge node that was evaluated.
#[derive(Debug, Clone)]
pub struct MergeRecord {
    pub node: usize,
    pub shared: Vec<usize>,
    pub plan: Vec<String>,
    pub solved: bool,
}

/// A solver for the shared constraints of a merge node.
pub trait MergeSolver {
    /// Merge the child plans of `problem` into one plan.
    ///
    /// Returns an error when the shared constraints cannot be satisfied by the
    /// merge; the error describes the remaining residuals.
    fn merge(&mut self, problem: &MergeProblem) -> Result<MergeOutcome, anyhow::Error>;
}

/// Diagnostics of one penalty optimization run.
#[derive(Debug, Clone, PartialEq)]
pub struct PenaltyReport {
    /// Number of outer rounds (Adam blocks) that were executed.
    pub rounds: usize,
    /// Penalty space of the last round.
    pub space: PenaltySpace,
    /// Final weight of every shared constraint, in problem order.
    pub weights: Vec<f64>,
    /// Exact violation of every shared constraint at the decoded solution.
    pub violations: Vec<f64>,
    /// Smoothed objective at the decoded solution.
    pub objective: f64,
    /// Whether the decoded merge satisfies every shared constraint.
    pub feasible: bool,
    /// Ordering constraints whose tasks could not be located in the children.
    pub unresolved: usize,
}

#[cfg(test)]
mod tests {
    use super::super::graph::Atom;
    use super::*;

    fn child(label: &str, steps: &[&str], tasks: &[usize]) -> PartialSolution {
        PartialSolution {
            label: label.to_string(),
            plan: steps.iter().map(|step| step.to_string()).collect(),
            plan_tasks: tasks.to_vec(),
            final_atoms: BTreeSet::new(),
            achieved: BTreeSet::new(),
            solved: true,
        }
    }

    fn ordering(before: usize, after: usize) -> ConstraintNode {
        ConstraintNode::Ordering { before, after }
    }

    fn problem(shared: Vec<ConstraintNode>, children: Vec<PartialSolution>) -> MergeProblem {
        MergeProblem {
            node: 3,
            shared_constraints: shared,
            children,
            initial_atoms: BTreeSet::new(),
        }
    }

    #[test]
    fn merger_reports_atoms_no_child_establishes() {
        let mut merger = SgplanMerger::default();
        let problem = problem(
            vec![ConstraintNode::Atom(Atom::new("at", vec!["z".to_string()]))],
            Vec::new(),
        );
        let error = merger.merge(&problem).expect_err("no child achieves at(z)");
        assert!(format!("{error}").contains("not established"));
        let report = merger.report.expect("a report is recorded");
        assert!(!report.feasible);
        assert_eq!(report.rounds, 0);
    }

    #[test]
    fn merger_reorders_children_to_satisfy_a_shared_ordering() {
        let mut merger = SgplanMerger::default();
        let problem = problem(
            vec![ordering(0, 1)],
            vec![
                child("later", &["drive(b, c)"], &[1]),
                child("earlier", &["drive(a, b)"], &[0]),
            ],
        );
        let outcome = merger
            .merge(&problem)
            .expect("reordering the children repairs the ordering");
        assert_eq!(
            outcome.plan,
            vec!["drive(a, b)".to_string(), "drive(b, c)".to_string()]
        );
        assert_eq!(outcome.plan_tasks, vec![0, 1]);
        assert!(outcome.solved);
        let report = merger.report.expect("a report is recorded");
        assert!(report.feasible);
        assert!(report.rounds >= 1);
    }

    #[test]
    fn merger_keeps_a_satisfied_order_without_optimizing() {
        let mut merger = SgplanMerger::default();
        let problem = problem(
            vec![ordering(0, 1)],
            vec![child("first", &["a"], &[0]), child("second", &["b"], &[1])],
        );
        let outcome = merger.merge(&problem).expect("the order is already valid");
        assert_eq!(outcome.plan, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            merger.report.expect("a report is recorded").rounds,
            0,
            "a feasible start must not run Adam"
        );
    }

    #[test]
    fn weights_grow_and_spaces_escalate_when_a_round_stalls() {
        let mut merger = SgplanMerger::default().without_initializer();
        merger.adam_steps = 1;
        merger.iterations = 3;
        merger.adam.learning_rate = 1e-12;
        let problem = problem(
            vec![ordering(0, 1)],
            vec![child("later", &["b"], &[1]), child("earlier", &["a"], &[0])],
        );
        assert!(
            merger.merge(&problem).is_err(),
            "a frozen optimizer cannot repair the order"
        );
        let report = merger.report.expect("a report is recorded");
        assert_eq!(report.rounds, 3);
        assert_eq!(report.space, PenaltySpace::Exponential);
        assert!((report.weights[0] - 64.0).abs() < 1e-9);
        assert!(!report.feasible);
    }

    #[test]
    fn interleaving_preserves_the_order_inside_each_child() {
        let mut merger = SgplanMerger::default();
        let problem = problem(
            Vec::new(),
            vec![
                child("left", &["l0", "l1"], &[0, 1]),
                child("right", &["r0", "r1"], &[2, 3]),
            ],
        );
        let outcome = merger.merge(&problem).expect("there is nothing to violate");
        assert_eq!(
            outcome.plan,
            vec![
                "l0".to_string(),
                "l1".to_string(),
                "r0".to_string(),
                "r1".to_string()
            ]
        );
        assert_eq!(outcome.plan_tasks, vec![0, 1, 2, 3]);
        assert!(merger.report.expect("a report is recorded").feasible);
    }

    #[test]
    fn equality_constraints_are_validated() {
        let mut merger = SgplanMerger::default();
        let problem = problem(
            vec![ConstraintNode::Equality {
                left: "a".to_string(),
                right: "b".to_string(),
                equal: true,
            }],
            Vec::new(),
        );
        assert!(merger.merge(&problem).is_err());
    }

    #[test]
    fn merge_rejects_plans_without_task_indices() {
        let mut merger = SgplanMerger::default();
        let mut broken = child("broken", &["a"], &[]);
        broken.plan_tasks = Vec::new();
        let problem = problem(Vec::new(), vec![broken]);
        assert!(merger.merge(&problem).is_err());
    }
}
