use crate::solver::domain::Domain;
use crate::solver::evaluation_tree::{
    self, EvaluationOptions, MethodSummary, Refinement, RefinementOptions, SgplanMerger,
};
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
    solve_with_options(domain, problem, &EvaluationOptions::default())
}

/// Solve with explicit evaluation options (for example enabling primitive
/// method summaries).
pub fn solve_with_options(
    domain: &Domain,
    problem: &Problem,
    options: &EvaluationOptions,
) -> Result<SolveResult, String> {
    let mut merger = SgplanMerger::default();
    let evaluation = evaluation_tree::evaluate(domain, problem, &mut merger, options)
        .map_err(|error| format!("{error:#}"))?;
    Ok(SolveResult {
        problem: problem.name.clone(),
        plan: evaluation.plan,
        solved: evaluation.solved,
    })
}

/// Solve the initial task network and refine its abstract plan into a complete
/// HTN plan with primitive method summaries (C+G).
pub fn solve_hierarchical(domain: &Domain, problem: &Problem) -> Result<Refinement, String> {
    let evaluation = EvaluationOptions {
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };
    solve_hierarchical_with_options(domain, problem, &RefinementOptions::default(), &evaluation)
}

/// Hierarchical refinement with explicit options.
pub fn solve_hierarchical_with_options(
    domain: &Domain,
    problem: &Problem,
    refinement: &RefinementOptions,
    evaluation: &EvaluationOptions,
) -> Result<Refinement, String> {
    evaluation_tree::solve_hierarchical(domain, problem, refinement, evaluation)
        .map_err(|error| format!("{error:#}"))
}
