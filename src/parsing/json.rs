//! Mirror of the JSON schema produced by
//! [koala-planner/HDDL-Parser](https://github.com/koala-planner/HDDL-Parser).

use serde::Deserialize;

/// Top-level `HDDLProgram`. `problem` is omitted by the parser when it was
/// only given a domain.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Program {
    pub domain: DomainAst,
    #[serde(default)]
    pub problem: Option<ProblemAst>,
}

/// `DomainAST`. `types` and `constants` are `Option` in the parser and are
/// serialised as `null` when absent.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DomainAst {
    pub name: String,
    #[serde(default)]
    pub types: Option<Vec<Symbol>>,
    #[serde(default)]
    pub constants: Option<Vec<Symbol>>,
    pub requirements: Vec<RequirementType>,
    pub predicates: Vec<Predicate>,
    pub compound_tasks: Vec<Task>,
    pub methods: Vec<Method>,
    pub actions: Vec<Action>,
    pub functions: Vec<Function>,
}

/// `ProblemAST`. `name` and `domain_name` have serde defaults in the parser,
/// so they may be missing in hand-written JSON.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProblemAst {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub domain_name: String,
    pub requirements: Vec<RequirementType>,
    #[serde(default)]
    pub init_tn: Option<InitialTaskNetwork>,
    pub init_state: Vec<Predicate>,
    #[serde(default)]
    pub goal: Option<Formula>,
    pub objects: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Symbol {
    pub name: String,
    #[serde(default)]
    pub symbol_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Predicate {
    pub name: String,
    pub variables: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Task {
    pub name: String,
    pub parameters: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Function {
    pub name: String,
    pub variables: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Action {
    pub name: String,
    pub parameters: Vec<Symbol>,
    #[serde(default)]
    pub preconditions: Option<Formula>,
    #[serde(default)]
    pub effects: Option<Formula>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Method {
    pub name: Symbol,
    pub task: Symbol,
    pub task_terms: Vec<Symbol>,
    pub params: Vec<Symbol>,
    #[serde(default)]
    pub precondition: Option<Formula>,
    pub tn: Htn,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct InitialTaskNetwork {
    #[serde(default)]
    pub parameters: Option<Vec<Symbol>>,
    pub tn: Htn,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Htn {
    pub subtasks: Vec<Subtask>,
    pub orderings: TaskOrdering,
    #[serde(default)]
    pub constraints: Option<Vec<Constraint>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Subtask {
    #[serde(default)]
    pub id: Option<Symbol>,
    pub task: Symbol,
    pub terms: Vec<Symbol>,
}

/// serde: `"Total"` or `{"Partial": [["t1", "t2"], ...]}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum TaskOrdering {
    Total,
    Partial(Vec<(String, String)>),
}

/// serde: `{"Equal": ["a", "b"]}` / `{"NotEqual": ["a", "b"]}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum Constraint {
    Equal(String, String),
    NotEqual(String, String),
}

/// The parser's formula language. This mirrors the parser's `Formula` enum one
/// constructor at a time, so no information from the parser is lost.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum Formula {
    Empty,
    Atom(Predicate),
    Not(Box<Formula>),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Xor(Vec<Formula>),
    Imply(Box<Formula>, Box<Formula>),
    Exists(Vec<Symbol>, Box<Formula>),
    ForAll(Vec<Symbol>, Box<Formula>),
    Probabilistic(NumberType, Box<Formula>),
    Equals(String, String),
}

/// serde: `{"Real": 0.5}` or `{"Integer": 3}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub enum NumberType {
    Real(f64),
    Integer(i64),
}

/// Requirements are serialised as bare parser variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum RequirementType {
    MethodPreconditions,
    Hierarchy,
    TypedObjects,
    NegativePreconditions,
    UniversalPreconditions,
    ExistentialPreconditions,
    QuantifiedPreconditions,
    Equality,
    #[allow(non_camel_case_types)]
    STRIPS,
    ProbabilisticEffects,
    ConditionalEffects,
    DomainAxioms,
    ADL,
}

/// Decode one of the parser's JSON exports.
pub fn decode_program(bytes: &[u8]) -> Result<Program, String> {
    serde_json::from_slice(bytes).map_err(|err| err.to_string())
}
