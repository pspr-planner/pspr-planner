//! Grounding and the task/constraint incidence graph.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use anyhow::{Result, ensure};

use crate::parsing::json::{
    Constraint as NetworkConstraint, Formula, Method, Predicate, Symbol, TaskOrdering,
};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::formula::{self, Universe};

/// A ground atom `predicate(term, ...)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atom {
    pub predicate: String,
    pub terms: Vec<String>,
}

impl Atom {
    pub fn new(predicate: impl Into<String>, terms: Vec<String>) -> Self {
        Atom {
            predicate: predicate.into(),
            terms,
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            write!(formatter, "{}", self.predicate)
        } else {
            write!(formatter, "{}({})", self.predicate, self.terms.join(", "))
        }
    }
}

/// A task of the initial task network after grounding.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskNode {
    pub name: String,
    pub terms: Vec<String>,
}

impl fmt::Display for TaskNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            write!(formatter, "{}", self.name)
        } else {
            write!(formatter, "{}({})", self.name, self.terms.join(", "))
        }
    }
}

/// A constraint node of the incidence graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintNode {
    /// A ground atom occurring in an action or method schema of the task.
    Atom(Atom),
    /// `before` must be executed before `after` (task indices).
    Ordering { before: usize, after: usize },
    /// An `Equal`/`NotEqual` constraint between two terms of the network.
    Equality {
        left: String,
        right: String,
        equal: bool,
    },
}

impl fmt::Display for ConstraintNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintNode::Atom(atom) => write!(formatter, "{atom}"),
            ConstraintNode::Ordering { before, after } => {
                write!(formatter, "before({before}, {after})")
            }
            ConstraintNode::Equality { left, right, equal } => {
                write!(
                    formatter,
                    "{left} {} {right}",
                    if *equal { "=" } else { "!=" }
                )
            }
        }
    }
}

/// The task/constraint incidence graph.
#[derive(Debug, Clone)]
pub struct TaskConstraintGraph {
    pub tasks: Vec<TaskNode>,
    pub constraints: Vec<ConstraintNode>,
    /// `incidence[task][constraint]` is `true` iff the task touches the constraint.
    pub incidence: Vec<Vec<bool>>,
    /// Number of probabilistic formulas, the only constructor the leaf encoding
    /// cannot translate.
    pub unmodeled: usize,
}

// ---------------------------------------------------------------------------
// Grounding
// ---------------------------------------------------------------------------

struct Grounding {
    bindings: BTreeMap<String, String>,
    tasks: Vec<TaskNode>,
}

/// A method schema together with one grounding of its parameters.
pub(super) struct GroundedMethod<'a> {
    pub(super) method: &'a Method,
    pub(super) bindings: BTreeMap<String, String>,
}

/// A bounded set of grounded method instances.
pub(super) struct BoundedMethods<'a> {
    pub(super) instances: Vec<GroundedMethod<'a>>,
    pub(super) truncated: bool,
}

/// Ground every method instance that can decompose `task`.
///
/// The task terms bind `method.task_terms`; the remaining method parameters are
/// enumerated over the typed universe. The method precondition is **not**
/// checked here, because callers evaluate it in their own state.
pub(super) fn ground_methods<'a>(
    domain: &'a Domain,
    universe: &Universe<'_>,
    task: &TaskNode,
    budget: usize,
) -> Result<Vec<GroundedMethod<'a>>> {
    let mut grounded = Vec::new();
    for method in domain.methods_for(&task.name) {
        if method.task_terms.len() != task.terms.len() {
            continue;
        }
        if !method
            .task_terms
            .iter()
            .zip(&task.terms)
            .all(|(parameter, term)| universe.compatible(parameter, term))
        {
            continue;
        }

        let bindings = bindings_from(&method.task_terms, &task.terms);
        let task_names: BTreeSet<&str> = method
            .task_terms
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        let extra: Vec<Symbol> = method
            .params
            .iter()
            .filter(|parameter| !task_names.contains(parameter.name.as_str()))
            .cloned()
            .collect();

        for assignment in universe.assignments(&extra, budget)? {
            let mut instance = bindings.clone();
            instance.extend(assignment);
            grounded.push(GroundedMethod {
                method,
                bindings: instance,
            });
            ensure!(
                grounded.len() <= budget,
                "task `{task}` has more than {budget} method groundings; \
                 raise EvaluationOptions::max_groundings"
            );
        }
    }
    Ok(grounded)
}

/// Ground method instances up to `budget` without failing on overflow.
///
/// Returns whether additional instances were skipped due to the bound.
pub(super) fn ground_methods_bounded<'a>(
    domain: &'a Domain,
    universe: &Universe<'_>,
    task: &TaskNode,
    budget: usize,
) -> BoundedMethods<'a> {
    if budget == 0 {
        return BoundedMethods {
            instances: Vec::new(),
            truncated: true,
        };
    }
    let mut grounded = Vec::new();
    let mut truncated = false;
    for method in domain.methods_for(&task.name) {
        if method.task_terms.len() != task.terms.len() {
            continue;
        }
        if !method
            .task_terms
            .iter()
            .zip(&task.terms)
            .all(|(parameter, term)| universe.compatible(parameter, term))
        {
            continue;
        }

        let bindings = bindings_from(&method.task_terms, &task.terms);
        let task_names: BTreeSet<&str> = method
            .task_terms
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        let extra: Vec<Symbol> = method
            .params
            .iter()
            .filter(|parameter| !task_names.contains(parameter.name.as_str()))
            .cloned()
            .collect();

        for assignment in universe.assignments_capped(&extra, budget) {
            if grounded.len() >= budget {
                truncated = true;
                break;
            }
            let mut instance = bindings.clone();
            instance.extend(assignment);
            grounded.push(GroundedMethod {
                method,
                bindings: instance,
            });
        }
        if grounded.len() >= budget {
            truncated = true;
            break;
        }
    }

    BoundedMethods {
        instances: grounded,
        truncated,
    }
}

pub(super) fn bindings_from(parameters: &[Symbol], terms: &[String]) -> BTreeMap<String, String> {
    parameters
        .iter()
        .zip(terms)
        .map(|(parameter, term)| (parameter.name.clone(), term.clone()))
        .collect()
}

pub(super) fn ground_atom(predicate: &Predicate, bindings: &BTreeMap<String, String>) -> Atom {
    Atom::new(
        predicate.name.clone(),
        predicate
            .variables
            .iter()
            .map(|symbol| {
                bindings
                    .get(&symbol.name)
                    .cloned()
                    .unwrap_or_else(|| symbol.name.clone())
            })
            .collect(),
    )
}

/// Format the ground atom of `predicate` into `out`, reusing its capacity.
///
/// This is the allocation-free counterpart of `ground_atom(..).to_string()`,
/// which the state-membership and effect loops call once per literal.
pub(super) fn atom_key_into(
    predicate: &Predicate,
    bindings: &BTreeMap<String, String>,
    out: &mut String,
) {
    out.clear();
    out.push_str(&predicate.name);
    if predicate.variables.is_empty() {
        return;
    }
    out.push('(');
    for (index, symbol) in predicate.variables.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(
            bindings
                .get(&symbol.name)
                .map(String::as_str)
                .unwrap_or(&symbol.name),
        );
    }
    out.push(')');
}

/// Ground the free parameters of the initial task network over the constants
/// and objects of the problem.
fn ground_task_network(
    universe: &Universe<'_>,
    problem: &Problem,
    max_groundings: usize,
) -> Result<Vec<Grounding>> {
    let Some(network) = problem.task_network() else {
        return Ok(Vec::new());
    };
    let mut assignments: Vec<BTreeMap<String, String>> = vec![BTreeMap::new()];
    for parameter in problem.network_parameters() {
        let candidates = universe.candidates(parameter);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let mut extended = Vec::new();
        for assignment in &assignments {
            for candidate in &candidates {
                let mut copy = assignment.clone();
                copy.insert(parameter.name.clone(), candidate.clone());
                extended.push(copy);
                ensure!(
                    extended.len() <= max_groundings,
                    "the initial task network has more than {max_groundings} groundings; \
                     raise EvaluationOptions::max_groundings"
                );
            }
        }
        assignments = extended;
    }

    Ok(assignments
        .into_iter()
        .map(|bindings| Grounding {
            tasks: network
                .subtasks
                .iter()
                .map(|subtask| TaskNode {
                    name: subtask.task.name.clone(),
                    terms: subtask
                        .terms
                        .iter()
                        .map(|term| {
                            bindings
                                .get(&term.name)
                                .cloned()
                                .unwrap_or_else(|| term.name.clone())
                        })
                        .collect(),
                })
                .collect(),
            bindings,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Constraint extraction
// ---------------------------------------------------------------------------

/// Traverse a formula and collect every atom it mentions.
///
/// The traversal expands quantifiers over the typed universe, so the incidence
/// graph contains the same ground atoms the leaf encoding will use. Only
/// [`Formula::Probabilistic`] is counted in `unmodeled`; the other constructors
/// are translated by [`super::formula`].
fn collect_atoms(
    formula: &Formula,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    budget: usize,
    atoms: &mut BTreeSet<Atom>,
    unmodeled: &mut usize,
) -> Result<()> {
    match formula {
        Formula::Empty | Formula::Equals(_, _) => {}
        Formula::Atom(predicate) => {
            atoms.insert(ground_atom(predicate, bindings));
        }
        Formula::Not(inner) => {
            collect_atoms(inner, bindings, universe, budget, atoms, unmodeled)?;
        }
        Formula::And(items) | Formula::Or(items) | Formula::Xor(items) => {
            for item in items {
                collect_atoms(item, bindings, universe, budget, atoms, unmodeled)?;
            }
        }
        Formula::Imply(left, right) => {
            collect_atoms(left, bindings, universe, budget, atoms, unmodeled)?;
            collect_atoms(right, bindings, universe, budget, atoms, unmodeled)?;
        }
        Formula::Exists(variables, inner) | Formula::ForAll(variables, inner) => {
            let assignments = match universe.assignments(variables, budget) {
                Ok(assignments) => assignments,
                Err(error) if formula::is_quantifier_overflow(&error) => {
                    universe.assignments_capped(variables, budget)
                }
                Err(error) => return Err(error),
            };
            for assignment in assignments {
                let mut local = bindings.clone();
                local.extend(assignment);
                collect_atoms(inner, &local, universe, budget, atoms, unmodeled)?;
            }
        }
        Formula::Probabilistic(_, inner) => {
            *unmodeled += 1;
            collect_atoms(inner, bindings, universe, budget, atoms, unmodeled)?;
        }
    }
    Ok(())
}

fn task_atoms(
    domain: &Domain,
    universe: &Universe<'_>,
    task: &TaskNode,
    budget: usize,
    summaries: bool,
    unmodeled: &mut usize,
) -> Result<BTreeSet<Atom>> {
    let mut atoms = BTreeSet::new();
    for action in domain.actions_for(&task.name) {
        if action.parameters.len() != task.terms.len() {
            continue;
        }
        let bindings = bindings_from(&action.parameters, &task.terms);
        if let Some(precondition) = &action.preconditions {
            collect_atoms(
                precondition,
                &bindings,
                universe,
                budget,
                &mut atoms,
                unmodeled,
            )?;
        }
        if let Some(effects) = &action.effects {
            collect_atoms(effects, &bindings, universe, budget, &mut atoms, unmodeled)?;
        }
    }
    let bounded_methods = ground_methods_bounded(domain, universe, task, budget);
    for instance in &bounded_methods.instances {
        if let Some(precondition) = &instance.method.precondition {
            collect_atoms(
                precondition,
                &instance.bindings,
                universe,
                budget,
                &mut atoms,
                unmodeled,
            )?;
        }
    }

    if summaries {
        for instance in &bounded_methods.instances {
            if !matches!(instance.method.tn.orderings, TaskOrdering::Total)
                || instance.method.tn.constraints.is_some()
            {
                continue;
            }
            let subtasks: Vec<TaskNode> = instance
                .method
                .tn
                .subtasks
                .iter()
                .map(|subtask| TaskNode {
                    name: subtask.task.name.clone(),
                    terms: subtask
                        .terms
                        .iter()
                        .map(|term| {
                            instance
                                .bindings
                                .get(&term.name)
                                .cloned()
                                .unwrap_or_else(|| term.name.clone())
                        })
                        .collect(),
                })
                .collect();
            if subtasks.is_empty()
                || !subtasks.iter().all(|subtask| {
                    domain.actions_for(&subtask.name).next().is_some()
                        && domain.methods_for(&subtask.name).next().is_none()
                })
            {
                continue;
            }
            for subtask in &subtasks {
                for action in domain.actions_for(&subtask.name) {
                    if action.parameters.len() != subtask.terms.len() {
                        continue;
                    }
                    let action_bindings = bindings_from(&action.parameters, &subtask.terms);
                    if let Some(precondition) = &action.preconditions {
                        collect_atoms(
                            precondition,
                            &action_bindings,
                            universe,
                            budget,
                            &mut atoms,
                            unmodeled,
                        )?;
                    }
                    if let Some(effects) = &action.effects {
                        collect_atoms(
                            effects,
                            &action_bindings,
                            universe,
                            budget,
                            &mut atoms,
                            unmodeled,
                        )?;
                    }
                }
            }
        }
    }
    Ok(atoms)
}

#[derive(Default)]
struct ConstraintIndex {
    by_node: BTreeMap<ConstraintNode, usize>,
    nodes: Vec<ConstraintNode>,
}

impl ConstraintIndex {
    fn intern(&mut self, node: ConstraintNode) -> usize {
        if let Some(&index) = self.by_node.get(&node) {
            index
        } else {
            let index = self.nodes.len();
            self.by_node.insert(node.clone(), index);
            self.nodes.push(node);
            index
        }
    }
}

/// A grounded top-level task network.
pub(super) struct GroundedNetwork {
    pub(super) tasks: Vec<TaskNode>,
    /// Ordering edges as `(before, after)` task indices.
    pub(super) edges: Vec<(usize, usize)>,
    /// Task-index range of every grounding.
    pub(super) groundings: Vec<(usize, usize)>,
}

/// Ground the initial task network and return its tasks together with the
/// ordering edges of every grounding.
///
/// The indices are global: a grounding that starts at `offset` contributes
/// edges `offset + i -> offset + j`. This is the same indexing that
/// [`build_task_constraint_graph`] uses for its `Ordering` constraint nodes,
/// but without building the atom incidence.
pub(super) fn ground_task_network_edges(
    domain: &Domain,
    problem: &Problem,
    max_groundings: usize,
) -> Result<GroundedNetwork> {
    let universe = Universe::new(domain, problem);
    let groundings = ground_task_network(&universe, problem, max_groundings)?;
    let mut tasks = Vec::new();
    let mut edges = Vec::new();
    let mut ranges = Vec::new();
    let Some(network) = problem.task_network() else {
        return Ok(GroundedNetwork {
            tasks,
            edges,
            groundings: ranges,
        });
    };
    let id_of: HashMap<&str, usize> = network
        .subtasks
        .iter()
        .enumerate()
        .filter_map(|(position, subtask)| {
            subtask.id.as_ref().map(|id| (id.name.as_str(), position))
        })
        .collect();

    for grounding in &groundings {
        let offset = tasks.len();
        let count = grounding.tasks.len();
        tasks.extend(grounding.tasks.iter().cloned());
        ranges.push((offset, offset + count));
        match &network.orderings {
            TaskOrdering::Total => {
                for position in 0..count.saturating_sub(1) {
                    edges.push((offset + position, offset + position + 1));
                }
            }
            TaskOrdering::Partial(pairs) => {
                for (before, after) in pairs {
                    let before = id_of.get(before.as_str()).copied().ok_or_else(|| {
                        anyhow::anyhow!("unknown subtask id `{before}` in the task ordering")
                    })?;
                    let after = id_of.get(after.as_str()).copied().ok_or_else(|| {
                        anyhow::anyhow!("unknown subtask id `{after}` in the task ordering")
                    })?;
                    edges.push((offset + before, offset + after));
                }
            }
        }
    }
    Ok(GroundedNetwork {
        tasks,
        edges,
        groundings: ranges,
    })
}

/// Build the task/constraint incidence graph of the initial task network.
pub fn build_task_constraint_graph(
    domain: &Domain,
    problem: &Problem,
    max_groundings: usize,
) -> Result<TaskConstraintGraph> {
    build_task_constraint_graph_with_summary(domain, problem, max_groundings, false, false)
}

/// Like [`build_task_constraint_graph`], optionally including the atoms read
/// and written by eligible primitive method summaries, and optionally adding
/// the implicit orderings that repair the liftability gaps of the projection.
pub(super) fn build_task_constraint_graph_with_summary(
    domain: &Domain,
    problem: &Problem,
    max_groundings: usize,
    summaries: bool,
    implicit_orderings: bool,
) -> Result<TaskConstraintGraph> {
    let universe = Universe::new(domain, problem);
    let groundings = ground_task_network(&universe, problem, max_groundings)?;
    let mut tasks = Vec::new();
    let mut offsets = Vec::new();
    for grounding in &groundings {
        offsets.push((tasks.len(), grounding));
        tasks.extend(grounding.tasks.iter().cloned());
    }

    let mut unmodeled = 0;
    let mut index = ConstraintIndex::default();
    let mut unresolved_atoms = 0usize;
    let mut incidence: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); tasks.len()];
    for (task_index, task) in tasks.iter().enumerate() {
        for atom in task_atoms(
            domain,
            &universe,
            task,
            max_groundings,
            summaries,
            &mut unmodeled,
        )? {
            if atom.terms.iter().any(|term| term.starts_with('?')) {
                unresolved_atoms += 1;
            }
            let constraint = index.intern(ConstraintNode::Atom(atom));
            incidence[task_index].insert(constraint);
        }
    }
    debug_assert_eq!(
        unresolved_atoms, 0,
        "constraint extraction produced unresolved atom terms"
    );

    let mut orderings: Vec<(usize, usize)> = Vec::new();
    if let Some(network) = problem.task_network() {
        let id_of: HashMap<&str, usize> = network
            .subtasks
            .iter()
            .enumerate()
            .filter_map(|(position, subtask)| {
                subtask.id.as_ref().map(|id| (id.name.as_str(), position))
            })
            .collect();

        for &(offset, grounding) in &offsets {
            let count = grounding.tasks.len();
            let mut add_ordering = |before: usize, after: usize| {
                orderings.push((before, after));
                let constraint = index.intern(ConstraintNode::Ordering { before, after });
                incidence[before].insert(constraint);
                incidence[after].insert(constraint);
            };
            match &network.orderings {
                TaskOrdering::Total => {
                    for position in 0..count.saturating_sub(1) {
                        add_ordering(offset + position, offset + position + 1);
                    }
                }
                TaskOrdering::Partial(pairs) => {
                    for (before, after) in pairs {
                        let before = id_of.get(before.as_str()).copied().ok_or_else(|| {
                            anyhow::anyhow!("unknown subtask id `{before}` in the task ordering")
                        })?;
                        let after = id_of.get(after.as_str()).copied().ok_or_else(|| {
                            anyhow::anyhow!("unknown subtask id `{after}` in the task ordering")
                        })?;
                        add_ordering(offset + before, offset + after);
                    }
                }
            }

            if let Some(constraints) = &network.constraints {
                for constraint in constraints {
                    let (left, right, equal) = match constraint {
                        NetworkConstraint::Equal(left, right) => (left, right, true),
                        NetworkConstraint::NotEqual(left, right) => (left, right, false),
                    };
                    let left = grounding
                        .bindings
                        .get(left)
                        .cloned()
                        .unwrap_or_else(|| left.clone());
                    let right = grounding
                        .bindings
                        .get(right)
                        .cloned()
                        .unwrap_or_else(|| right.clone());
                    let node = ConstraintNode::Equality {
                        left: left.clone(),
                        right: right.clone(),
                        equal,
                    };
                    let id = index.intern(node);
                    let mut matched = false;
                    for position in 0..count {
                        let task = &grounding.tasks[position];
                        if task.terms.contains(&left) || task.terms.contains(&right) {
                            incidence[offset + position].insert(id);
                            matched = true;
                        }
                    }
                    if !matched {
                        // A network-level constraint that names no subtask term
                        // still partitions with every task of the grounding.
                        for position in 0..count {
                            incidence[offset + position].insert(id);
                        }
                    }
                }
            }
        }
    }

    // Implicit orderings that make the abstract plan refine into a complete
    // HTN plan: a producer task that establishes (or destroys) a literal a
    // later method needs must run before (after) that consumer. Once added,
    // the edges are ordinary ordering constraints, so the clusterer, the
    // leaves (transitive closure) and the merge nodes all enforce them.
    if implicit_orderings && !tasks.is_empty() {
        let liftability = super::liftability::LiftabilityOptions {
            max_groundings,
            ..super::liftability::LiftabilityOptions::default()
        };
        let groundings: Vec<(usize, usize)> = offsets
            .iter()
            .map(|&(offset, grounding)| (offset, offset + grounding.tasks.len()))
            .collect();
        for (before, after) in super::liftability::implicit_orderings(
            domain,
            problem,
            &tasks,
            &orderings,
            &groundings,
            &liftability,
        )? {
            let constraint = index.intern(ConstraintNode::Ordering { before, after });
            incidence[before].insert(constraint);
            incidence[after].insert(constraint);
        }
    }

    let constraints = index.nodes;
    let incidence = incidence
        .into_iter()
        .map(|touched| {
            (0..constraints.len())
                .map(|id| touched.contains(&id))
                .collect()
        })
        .collect();

    Ok(TaskConstraintGraph {
        tasks,
        constraints,
        incidence,
        unmodeled,
    })
}
