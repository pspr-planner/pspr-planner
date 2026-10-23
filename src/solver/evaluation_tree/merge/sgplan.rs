//! The SGPlan penalty merger: a relaxed MINLP optimized with Adam.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::{Result, bail, ensure};

use super::super::graph::ConstraintNode;
use super::adam::AdamOptions;
use super::optimizer::{OptimizerKind, OptimizerState};
use super::penalty::{PenaltySpace, logistic, smooth_max, softplus};
use super::{
    MergeInitialization, MergeInitializer, MergeOutcome, MergeProblem, MergeSolver,
    PartialSolution, PenaltyReport,
};

/// The constraint-partitioning merger of SGPlan (Hsu et al., *Constraint
/// partitioning in penalty formulations for solving temporal planning
/// problems*, Artificial Intelligence 170(3), 2006), solved as a continuously
/// relaxed MINLP.
///
/// The merger decides how the child plans of a merge node are interleaved. It
/// relaxes the child order to continuous start positions and minimizes
///
/// ```text
/// J(q) = cost_weight * smoothmax_c(q_c + l_c / L)
///      + sum_c w_c * phi(h_c(q))
/// ```
///
/// with Adam, where `h_c` are the smoothed violations of the shared ordering
/// constraints and `w_c` are adapted penalty weights. See the module
/// documentation for the derivation and for the multi-space continuation.
///
/// The decoded plan is validated exactly; atom and equality constraints cannot
/// be repaired by reordering and are reported as errors when the child plans do
/// not establish them.
#[derive(Debug, Clone)]
pub struct SgplanMerger {
    /// Initial penalty weight of every shared constraint.
    pub penalty_weight: f64,
    /// Lower bound of an adapted penalty weight.
    pub min_penalty: f64,
    /// Upper bound of an adapted penalty weight.
    pub max_penalty: f64,
    /// Multiplicative increase for a constraint that is still violated after a
    /// round.
    pub penalty_growth: f64,
    /// Multiplicative decrease for a constraint that is satisfied after a
    /// round.
    pub penalty_decay: f64,
    /// Maximum number of outer rounds (Adam blocks).
    pub iterations: usize,
    /// Adam steps per outer round.
    pub adam_steps: usize,
    /// Adam hyper-parameters.
    pub adam: AdamOptions,
    /// Optimization backend used for the relaxed objective.
    pub optimizer: OptimizerKind,
    /// Weight of the makespan term of the relaxed objective.
    pub cost_weight: f64,
    /// Temperature of the smooth maximum and of the smooth hinge.
    pub temperature: f64,
    /// Residual below which an exact violation counts as satisfied.
    pub tolerance: f64,
    /// Relative improvement a round must reach to stay in its penalty space.
    pub min_progress: f64,
    /// Penalty spaces, in continuation order.
    pub spaces: Vec<PenaltySpace>,
    /// Diagnostics of the most recent merge.
    pub report: Option<PenaltyReport>,
    /// Optional learned warm start.
    initializer: Option<Arc<dyn MergeInitializer>>,
}

impl Default for SgplanMerger {
    fn default() -> Self {
        #[cfg_attr(not(feature = "neural-base"), allow(unused_mut))]
        let mut merger = SgplanMerger {
            penalty_weight: 1.0,
            min_penalty: 1e-3,
            max_penalty: 1e9,
            penalty_growth: 4.0,
            penalty_decay: 0.5,
            iterations: 10,
            adam_steps: 200,
            adam: AdamOptions::default(),
            optimizer: OptimizerKind::Adam,
            cost_weight: 1.0,
            temperature: 0.05,
            tolerance: 1e-3,
            min_progress: 0.05,
            spaces: vec![
                PenaltySpace::Linear,
                PenaltySpace::Quadratic,
                PenaltySpace::Exponential,
            ],
            report: None,
            initializer: None,
        };
        #[cfg(feature = "neural-base")]
        if let Some(path) = std::env::var_os("PSPR_MERGE_MODEL") {
            let path = std::path::PathBuf::from(path);
            match super::neural::NeuralMergeInitializer::from_path(&path) {
                Ok(initializer) => {
                    merger.initializer = Some(Arc::new(initializer));
                }
                Err(error) => {
                    eprintln!("ignoring PSPR_MERGE_MODEL={}: {error:#}", path.display());
                }
            }
        }
        merger
    }
}

impl SgplanMerger {
    /// Attach a learned initializer. The sequential start remains the fallback
    /// whenever the initializer declines or returns an invalid vector.
    pub fn with_initializer(mut self, initializer: Arc<dyn MergeInitializer>) -> Self {
        self.initializer = Some(initializer);
        self
    }

    /// Disable any learned initializer, for example for the teacher run that
    /// produces training targets.
    pub fn without_initializer(mut self) -> Self {
        self.initializer = None;
        self
    }

    /// Select the optimizer used for the relaxed objective.
    pub fn with_optimizer(mut self, optimizer: OptimizerKind) -> Self {
        self.optimizer = optimizer;
        self
    }
}

/// A start position inside a child plan.
#[derive(Debug, Clone, Copy)]
struct Occurrence {
    child: usize,
    /// Normalized start offset of the step inside the child.
    time: f64,
}

/// A shared constraint of the relaxed problem.
#[derive(Debug, Clone)]
enum RelaxedConstraint {
    /// `before` must be executed before `after`.
    Ordering {
        before: Occurrence,
        after: Occurrence,
    },
    /// The violation does not depend on the schedule.
    Fixed { violated: bool },
}

/// The continuous relaxation of one merge node.
struct RelaxedProblem<'a> {
    children: &'a [PartialSolution],
    /// Normalized duration of every child plan.
    lengths: Vec<f64>,
    constraints: Vec<RelaxedConstraint>,
    /// Total number of plan steps (normalizer), at least one.
    total: f64,
    /// Ordering constraints whose tasks could not be located in the children.
    unresolved: usize,
}

impl<'a> RelaxedProblem<'a> {
    fn new(problem: &'a MergeProblem, atoms: &BTreeSet<String>) -> Self {
        let steps = problem
            .children
            .iter()
            .map(|child| child.plan.len())
            .sum::<usize>();
        let total = steps.max(1) as f64;
        let lengths = problem
            .children
            .iter()
            .map(|child| child.plan.len() as f64 / total)
            .collect();
        let mut unresolved = 0;
        let mut constraints = Vec::with_capacity(problem.shared_constraints.len());
        for constraint in &problem.shared_constraints {
            match constraint {
                ConstraintNode::Ordering { before, after } => {
                    match (
                        locate(&problem.children, *before, total),
                        locate(&problem.children, *after, total),
                    ) {
                        (Some(before), Some(after)) => {
                            constraints.push(RelaxedConstraint::Ordering { before, after });
                        }
                        _ => {
                            unresolved += 1;
                            constraints.push(RelaxedConstraint::Fixed { violated: false });
                        }
                    }
                }
                ConstraintNode::Atom(atom) => {
                    constraints.push(RelaxedConstraint::Fixed {
                        violated: !atoms.contains(&atom.to_string()),
                    });
                }
                ConstraintNode::Equality { left, right, equal } => {
                    constraints.push(RelaxedConstraint::Fixed {
                        violated: (left == right) != *equal,
                    });
                }
            }
        }
        RelaxedProblem {
            children: &problem.children,
            lengths,
            constraints,
            total,
            unresolved,
        }
    }

    fn has_orderings(&self) -> bool {
        self.constraints
            .iter()
            .any(|constraint| matches!(constraint, RelaxedConstraint::Ordering { .. }))
    }

    /// The relaxed objective and its gradient with respect to the positions.
    fn evaluate(
        &self,
        parameters: &[f64],
        weights: &[f64],
        space: PenaltySpace,
        cost_weight: f64,
        temperature: f64,
    ) -> (f64, Vec<f64>) {
        let temperature = temperature.max(1e-9);
        let completions: Vec<f64> = parameters
            .iter()
            .zip(&self.lengths)
            .map(|(position, length)| position + length)
            .collect();
        let (makespan, softmax) = smooth_max(&completions, temperature);
        let mut objective = cost_weight * makespan;
        let mut gradient: Vec<f64> = softmax.iter().map(|weight| cost_weight * weight).collect();
        for (constraint, &weight) in self.constraints.iter().zip(weights) {
            match constraint {
                RelaxedConstraint::Ordering { before, after } => {
                    let margin = self.margin(parameters, before, after);
                    let violation = temperature * softplus(-margin / temperature);
                    objective += weight * space.value(violation);
                    let slope = weight * space.slope(violation) * -logistic(-margin / temperature);
                    gradient[after.child] += slope;
                    gradient[before.child] -= slope;
                }
                RelaxedConstraint::Fixed { violated } => {
                    let violation = if *violated { 1.0 } else { 0.0 };
                    objective += weight * space.value(violation);
                }
            }
        }
        (objective, gradient)
    }

    /// Exact violations at a relaxed point; positive means violated.
    fn exact_violations(&self, parameters: &[f64]) -> Vec<f64> {
        self.constraints
            .iter()
            .map(|constraint| match constraint {
                RelaxedConstraint::Ordering { before, after } => {
                    (-self.margin(parameters, before, after)).max(0.0)
                }
                RelaxedConstraint::Fixed { violated } => {
                    if *violated {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect()
    }

    /// Exact violations of a concrete plan, in shared-constraint order.
    fn validate(&self, problem: &MergeProblem, plan_tasks: &[usize]) -> Vec<f64> {
        problem
            .shared_constraints
            .iter()
            .zip(&self.constraints)
            .map(|(constraint, relaxed)| match (constraint, relaxed) {
                (_, RelaxedConstraint::Fixed { violated }) => {
                    if *violated {
                        1.0
                    } else {
                        0.0
                    }
                }
                (
                    ConstraintNode::Ordering { before, after },
                    RelaxedConstraint::Ordering { .. },
                ) => {
                    let before = plan_tasks.iter().position(|task| task == before);
                    let after = plan_tasks.iter().position(|task| task == after);
                    match (before, after) {
                        (Some(before), Some(after)) if before < after => 0.0,
                        (Some(_), Some(_)) => 1.0,
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            })
            .collect()
    }

    fn initial_parameters(&self) -> Vec<f64> {
        let mut start = 0.0;
        self.lengths
            .iter()
            .map(|&length| {
                let position = start;
                start += length;
                position
            })
            .collect()
    }

    /// Signed margin of an ordering constraint: satisfied when non-negative.
    fn margin(&self, parameters: &[f64], before: &Occurrence, after: &Occurrence) -> f64 {
        parameters[after.child] + after.time
            - parameters[before.child]
            - before.time
            - 1.0 / self.total
    }

    /// Decode the relaxed positions into a concrete interleaving that keeps the
    /// order inside every child.
    fn decode(&self, parameters: &[f64]) -> (Vec<String>, Vec<usize>) {
        let mut next = vec![0usize; self.children.len()];
        let steps: usize = self.children.iter().map(|child| child.plan.len()).sum();
        let mut plan = Vec::with_capacity(steps);
        let mut tasks = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut selected: Option<(usize, f64)> = None;
            for (index, child) in self.children.iter().enumerate() {
                if next[index] >= child.plan.len() {
                    continue;
                }
                let time = parameters[index] + next[index] as f64 / self.total;
                if selected.is_none_or(|(_, best)| time < best) {
                    selected = Some((index, time));
                }
            }
            match selected {
                Some((index, _)) => {
                    let child = &self.children[index];
                    plan.push(child.plan[next[index]].clone());
                    tasks.push(child.plan_tasks[next[index]]);
                    next[index] += 1;
                }
                None => break,
            }
        }
        (plan, tasks)
    }
}

/// Find the child and the normalized offset of a grounded task.
fn locate(children: &[PartialSolution], task: usize, total: f64) -> Option<Occurrence> {
    children.iter().enumerate().find_map(|(child, solution)| {
        solution
            .plan_tasks
            .iter()
            .position(|&candidate| candidate == task)
            .map(|position| Occurrence {
                child,
                time: position as f64 / total,
            })
    })
}

/// The result of the penalty iteration.
struct Optimization {
    parameters: Vec<f64>,
    rounds: usize,
    space: PenaltySpace,
    weights: Vec<f64>,
}

impl SgplanMerger {
    /// Run the multi-space, adaptively weighted Adam iteration.
    fn optimize(
        &self,
        problem: &RelaxedProblem<'_>,
        initialization: Option<MergeInitialization>,
    ) -> Optimization {
        let spaces: Vec<PenaltySpace> = if self.spaces.is_empty() {
            vec![PenaltySpace::Linear]
        } else {
            self.spaces.clone()
        };
        let mut parameters = problem.initial_parameters();
        let mut weights = vec![self.penalty_weight; problem.constraints.len()];
        let mut preferred_space = None;
        if let Some(initialization) = initialization
            && initialization.positions.len() == parameters.len()
            && initialization.weights.len() == weights.len()
            && initialization
                .positions
                .iter()
                .all(|value| value.is_finite())
            && initialization
                .weights
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        {
            parameters = initialization.positions;
            weights = initialization.weights;
            preferred_space = initialization.space;
        }
        let initial: f64 = problem.exact_violations(&parameters).iter().sum();
        if !problem.has_orderings() || self.iterations == 0 || initial <= self.tolerance {
            return Optimization {
                parameters,
                rounds: 0,
                space: preferred_space.unwrap_or(spaces[0]),
                weights,
            };
        }

        let mut optimizer = OptimizerState::new(self.optimizer, problem.children.len());
        let mut space_index = preferred_space
            .and_then(|space| spaces.iter().position(|candidate| *candidate == space))
            .unwrap_or(0);
        let mut previous: Option<f64> = None;
        let mut best: Option<(f64, f64, Vec<f64>)> = None;
        let mut rounds = 0;
        for _ in 0..self.iterations {
            rounds += 1;
            let space = spaces[space_index];
            for _ in 0..self.adam_steps.max(1) {
                let (objective, gradient) = problem.evaluate(
                    &parameters,
                    &weights,
                    space,
                    self.cost_weight,
                    self.temperature,
                );
                let mut evaluate = |candidate: &[f64]| {
                    problem.evaluate(
                        candidate,
                        &weights,
                        space,
                        self.cost_weight,
                        self.temperature,
                    )
                };
                optimizer.update(
                    &mut parameters,
                    &gradient,
                    objective,
                    &mut evaluate,
                    &self.adam,
                    0.0,
                    1.0,
                );
            }
            let violations = problem.exact_violations(&parameters);
            let total: f64 = violations.iter().sum();
            let (objective, _) = problem.evaluate(
                &parameters,
                &weights,
                space,
                self.cost_weight,
                self.temperature,
            );
            let better = match &best {
                None => true,
                Some((best_violation, best_objective, _)) => {
                    total < *best_violation
                        || (total == *best_violation && objective < *best_objective)
                }
            };
            if better {
                best = Some((total, objective, parameters.clone()));
            }
            if total <= self.tolerance {
                break;
            }
            for (weight, violation) in weights.iter_mut().zip(&violations) {
                if *violation > self.tolerance {
                    *weight =
                        (*weight * self.penalty_growth).clamp(self.min_penalty, self.max_penalty);
                } else {
                    *weight =
                        (*weight * self.penalty_decay).clamp(self.min_penalty, self.max_penalty);
                }
            }
            let stalled =
                previous.is_some_and(|previous| total > previous * (1.0 - self.min_progress));
            if stalled && space_index + 1 < spaces.len() {
                space_index += 1;
                optimizer.reset();
            }
            previous = Some(total);
        }
        if let Some((best_violation, _, best_parameters)) = best {
            let current: f64 = problem.exact_violations(&parameters).iter().sum();
            if best_violation < current {
                parameters = best_parameters;
            }
        }
        Optimization {
            parameters,
            rounds,
            space: spaces[space_index],
            weights,
        }
    }
}

impl SgplanMerger {
    /// Return the parameters reached by the current optimizer from the
    /// sequential start. Used as the teacher target for training a learned
    /// initializer; it does not touch the configured initializer.
    pub fn teacher_initialization(&self, problem: &MergeProblem) -> MergeInitialization {
        let mut atoms = problem.initial_atoms.clone();
        for child in &problem.children {
            atoms.extend(child.final_atoms.iter().cloned());
            atoms.extend(child.achieved.iter().cloned());
        }
        let relaxed = RelaxedProblem::new(problem, &atoms);
        let optimization = self.clone().without_initializer().optimize(&relaxed, None);
        MergeInitialization {
            positions: optimization.parameters,
            weights: optimization.weights,
            space: Some(optimization.space),
        }
    }

    fn merge_inner(&mut self, problem: &MergeProblem) -> Result<MergeOutcome> {
        for child in &problem.children {
            ensure!(
                child.plan.len() == child.plan_tasks.len(),
                "child `{}` has {} plan steps but {} task indices",
                child.label,
                child.plan.len(),
                child.plan_tasks.len()
            );
        }
        let mut atoms = problem.initial_atoms.clone();
        for child in &problem.children {
            atoms.extend(child.final_atoms.iter().cloned());
            atoms.extend(child.achieved.iter().cloned());
        }
        let relaxed = RelaxedProblem::new(problem, &atoms);
        let initialization = self
            .initializer
            .as_ref()
            .and_then(|initializer| initializer.initialize(problem));
        let optimization = self.optimize(&relaxed, initialization);
        let (plan, plan_tasks) = relaxed.decode(&optimization.parameters);
        let violations = relaxed.validate(problem, &plan_tasks);
        let feasible = violations
            .iter()
            .all(|violation| *violation <= self.tolerance);
        let objective = relaxed
            .evaluate(
                &optimization.parameters,
                &optimization.weights,
                optimization.space,
                self.cost_weight,
                self.temperature,
            )
            .0;
        self.report = Some(PenaltyReport {
            rounds: optimization.rounds,
            space: optimization.space,
            weights: optimization.weights.clone(),
            violations: violations.clone(),
            objective,
            feasible,
            unresolved: relaxed.unresolved,
        });
        if !feasible {
            let mut detail = String::new();
            for (index, violation) in violations.iter().enumerate() {
                if *violation <= self.tolerance {
                    continue;
                }
                match &problem.shared_constraints[index] {
                    ConstraintNode::Atom(atom) => {
                        let _ = write!(
                            detail,
                            "\n  `{atom}` is not established by the child solutions"
                        );
                    }
                    ConstraintNode::Equality { .. } => {
                        let _ = write!(
                            detail,
                            "\n  `{}` is violated by the grounding",
                            problem.shared_constraints[index]
                        );
                    }
                    ConstraintNode::Ordering { .. } => {
                        let weight = optimization
                            .weights
                            .get(index)
                            .copied()
                            .unwrap_or(self.penalty_weight);
                        let _ = write!(
                            detail,
                            "\n  `{}` is still violated (residual {violation:.3e}, weight {weight:.3e})",
                            problem.shared_constraints[index]
                        );
                    }
                }
            }
            bail!(
                "the SGPlan penalty merger could not satisfy every shared constraint of merge node {} \
                 after {} Adam round(s) in the `{}` penalty space:{detail}",
                problem.node,
                optimization.rounds,
                optimization.space.name()
            );
        }
        Ok(MergeOutcome {
            plan,
            plan_tasks,
            final_atoms: atoms,
            solved: problem.children.iter().all(|child| child.solved),
        })
    }
}

impl MergeSolver for SgplanMerger {
    fn merge(&mut self, problem: &MergeProblem) -> Result<MergeOutcome> {
        self.merge_inner(problem)
    }
}
