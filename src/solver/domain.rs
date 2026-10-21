use crate::parsing::json::{
    Action, DomainAst, Function, Method, Predicate, RequirementType, Symbol, Task,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub name: String,
    pub requirements: Vec<RequirementType>,
    pub types: Vec<Symbol>,
    pub constants: Vec<Symbol>,
    pub predicates: Vec<Predicate>,
    pub compound_tasks: Vec<Task>,
    pub methods: Vec<Method>,
    pub actions: Vec<Action>,
    pub functions: Vec<Function>,
}

/// Stub: copy the parser AST into the solver-facing [`Domain`].
pub fn process_domain(ast: &DomainAst) -> Result<Domain, String> {
    Ok(Domain {
        name: ast.name.clone(),
        requirements: ast.requirements.clone(),
        types: ast.types.clone().unwrap_or_default(),
        constants: ast.constants.clone().unwrap_or_default(),
        predicates: ast.predicates.clone(),
        compound_tasks: ast.compound_tasks.clone(),
        methods: ast.methods.clone(),
        actions: ast.actions.clone(),
        functions: ast.functions.clone(),
    })
}
