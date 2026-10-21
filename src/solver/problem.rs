use crate::parsing::json::{Formula, InitialTaskNetwork, Predicate, ProblemAst, Symbol};
use crate::solver::domain::Domain;

#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub name: String,
    pub domain_name: String,
    pub objects: Vec<Symbol>,
    pub init_state: Vec<Predicate>,
    pub goal: Option<Formula>,
    pub init_task_network: Option<InitialTaskNetwork>,
}

/// Stub: turn the parser's problem AST into a problem that can be handed to
/// [`crate::solver::solve::solve`]. Later this is where objects are grounded against
/// the domain's type hierarchy and the initial task network is elaborated.
pub fn load_problem(_domain: &Domain, ast: &ProblemAst) -> Result<Problem, String> {
    Ok(Problem {
        name: ast.name.clone(),
        domain_name: ast.domain_name.clone(),
        objects: ast.objects.clone(),
        init_state: ast.init_state.clone(),
        goal: ast.goal.clone(),
        init_task_network: ast.init_tn.clone(),
    })
}
