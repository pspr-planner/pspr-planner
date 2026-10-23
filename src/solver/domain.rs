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

impl Domain {
    /// Actions whose name matches the primitive task `task`.
    pub fn actions_for(&self, task: &str) -> impl Iterator<Item = &Action> {
        self.actions
            .iter()
            .filter(move |action| action.name == task)
    }

    /// Methods that decompose the compound task `task`.
    pub fn methods_for(&self, task: &str) -> impl Iterator<Item = &Method> {
        self.methods
            .iter()
            .filter(move |method| method.task.name == task)
    }

    /// Whether `type_name` is `ancestor` or a (transitive) subtype of it.
    ///
    /// Unknown types are only considered equal to themselves.
    pub fn is_subtype_of(&self, type_name: &str, ancestor: &str) -> bool {
        if type_name == ancestor {
            return true;
        }
        // The type hierarchy is tiny and acyclic in practice. The bound on the
        // number of steps replaces the per-call `HashSet` that used to detect
        // cycles, which matters because grounding calls this per candidate
        // object.
        let mut current = type_name;
        for _ in 0..=self.types.len() {
            let Some(symbol) = self.types.iter().find(|symbol| symbol.name == current) else {
                return false;
            };
            match symbol.symbol_type.as_deref() {
                Some(parent) if parent == ancestor => return true,
                Some(parent) => current = parent,
                None => return false,
            }
        }
        false
    }
}
