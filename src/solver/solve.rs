use crate::solver::domain::Domain;
use crate::solver::evaluation_tree::{self, EvaluationOptions, SgplanMerger};
use crate::solver::problem::Problem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveResult {
    pub problem: String,
    pub plan: Vec<String>,
    /// Whether every leaf of the evaluation tree was solved and every merge
    /// satisfied its shared constraints.
    pub solved: bool,
}

/// Build the constraint-partitioned evaluation tree and evaluate it with the
/// SGPlan-style merger.
///
/// The returned plan is empty when no leaf of the tree can be solved (or when
/// the problem has no initial task network). A partially solved tree returns
/// the steps that were found and sets [`SolveResult::solved`] to `false`.
pub fn solve(domain: &Domain, problem: &Problem) -> Result<SolveResult, String> {
    let mut merger = SgplanMerger::default();
    let evaluation =
        evaluation_tree::evaluate(domain, problem, &mut merger, &EvaluationOptions::default())
            .map_err(|error| format!("{error:#}"))?;
    Ok(SolveResult {
        problem: problem.name.clone(),
        plan: evaluation.plan,
        solved: evaluation.solved,
    })
}
