use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveResult {
    pub problem: String,
    pub plan: Vec<String>,
}

/// Stub: report the problem as unsolved (an empty plan).
pub fn solve(_domain: &Domain, problem: &Problem) -> Result<SolveResult, String> {
    Ok(SolveResult {
        problem: problem.name.clone(),
        plan: Vec::new(),
    })
}
