//! Leaf planning tasks: local constraints encoded for Fast Downward.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, ensure};

use crate::parsing::json::Formula;
use crate::solver::domain::Domain;
use crate::solver::fast_downward;
use crate::solver::problem::Problem;

use super::clustering::EvaluationTree;
use super::graph::{Atom, ConstraintNode, bindings_from, ground_atom};

pub(super) struct LeafPlanningTask {
    pub(super) task: fast_downward::Task,
    /// Local atoms by variable index.
    pub(super) local_atoms: Vec<Atom>,
    /// Ground atoms of every operator's effects (local and shared).
    pub(super) effect_atoms: Vec<BTreeSet<Atom>>,
    /// Grounded task index executed by every operator.
    pub(super) operator_tasks: Vec<usize>,
}

pub(super) fn initial_atoms(problem: &Problem) -> BTreeSet<String> {
    problem
        .init_state
        .iter()
        .map(|predicate| {
            Atom::new(
                predicate.name.clone(),
                predicate
                    .variables
                    .iter()
                    .map(|symbol| symbol.name.clone())
                    .collect(),
            )
            .to_string()
        })
        .collect()
}

/// Return the conjunctive literals of a formula, or `None` when the formula is
/// outside the supported fragment.
fn conjunctive_literals(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
) -> Option<Vec<(Atom, bool)>> {
    fn walk(
        formula: &Formula,
        bindings: &BTreeMap<String, String>,
        literals: &mut Vec<(Atom, bool)>,
    ) -> bool {
        match formula {
            Formula::Empty => true,
            Formula::Atom(predicate) => {
                literals.push((ground_atom(predicate, bindings), true));
                true
            }
            Formula::Not(inner) => match inner.as_ref() {
                Formula::Atom(predicate) => {
                    literals.push((ground_atom(predicate, bindings), false));
                    true
                }
                _ => false,
            },
            Formula::And(items) => items.iter().all(|item| walk(item, bindings, literals)),
            _ => false,
        }
    }

    let formula = formula?;
    let mut literals = Vec::new();
    if !walk(formula, bindings, &mut literals) {
        return None;
    }
    literals.sort_by(|left, right| left.0.cmp(&right.0));
    Some(literals)
}

/// Transitive closure of the ordering constraints inside a cluster.
fn precedence_closure(count: usize, edges: &[(usize, usize)]) -> Result<Vec<BTreeSet<usize>>> {
    let mut predecessors = vec![BTreeSet::new(); count];
    for &(before, after) in edges {
        predecessors[after].insert(before);
    }
    for middle in 0..count {
        for node in 0..count {
            if predecessors[node].contains(&middle) {
                let inherited = predecessors[middle].clone();
                predecessors[node].extend(inherited);
            }
        }
    }
    for (node, before) in predecessors.iter().enumerate() {
        ensure!(
            !before.contains(&node),
            "the ordering constraints inside a cluster are cyclic"
        );
    }
    Ok(predecessors)
}

fn grounded_name(name: &str, terms: &[String]) -> String {
    if terms.is_empty() {
        name.to_string()
    } else {
        format!("{name}({})", terms.join(", "))
    }
}

pub(super) fn build_leaf_task(
    domain: &Domain,
    tree: &EvaluationTree,
    cluster_index: usize,
    initial_atoms: &BTreeSet<String>,
) -> Result<LeafPlanningTask> {
    let cluster = &tree.clusters[cluster_index];
    let local_ids = &tree.local[cluster_index];

    let mut local_atoms: Vec<Atom> = local_ids
        .iter()
        .filter_map(|&constraint| match &tree.constraints[constraint] {
            ConstraintNode::Atom(atom) => Some(atom.clone()),
            _ => None,
        })
        .collect();
    local_atoms.sort();
    local_atoms.dedup();
    let local_index: BTreeMap<Atom, usize> = local_atoms
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, atom)| (atom, index))
        .collect();

    let position: HashMap<usize, usize> = cluster
        .n1s
        .iter()
        .enumerate()
        .map(|(position, &task)| (task, position))
        .collect();
    let mut edges = Vec::new();
    for &constraint in local_ids {
        if let ConstraintNode::Ordering { before, after } = &tree.constraints[constraint]
            && let (Some(&before), Some(&after)) = (position.get(before), position.get(after))
        {
            edges.push((before, after));
        }
    }
    let predecessors = precedence_closure(cluster.n1s.len(), &edges)?;

    let mut variables: Vec<(String, usize, bool)> = local_atoms
        .iter()
        .map(|atom| (atom.to_string(), 2, false))
        .collect();
    for &task in &cluster.n1s {
        variables.push((format!("executed:{}", tree.tasks[task]), 2, false));
    }
    let executed = |position: usize| local_atoms.len() + position;

    let mut initial = vec![0usize; variables.len()];
    for (index, atom) in local_atoms.iter().enumerate() {
        initial[index] = usize::from(initial_atoms.contains(&atom.to_string()));
    }
    let goal: Vec<(usize, usize)> = (0..cluster.n1s.len())
        .map(|position| (executed(position), 1))
        .collect();

    let mut operators: Vec<fast_downward::OperatorSpec> = Vec::new();
    let mut effect_atoms: Vec<BTreeSet<Atom>> = Vec::new();
    let mut operator_tasks: Vec<usize> = Vec::new();
    for (position, &task_index) in cluster.n1s.iter().enumerate() {
        let task = &tree.tasks[task_index];
        let base: Vec<(usize, usize)> = predecessors[position]
            .iter()
            .map(|&before| (executed(before), 1))
            .collect();

        let actions: Vec<_> = domain
            .actions_for(&task.name)
            .filter(|action| action.parameters.len() == task.terms.len())
            .collect();
        if !actions.is_empty() {
            for action in actions {
                let bindings = bindings_from(&action.parameters, &task.terms);
                let mut precondition = base.clone();
                if let Some(literals) =
                    conjunctive_literals(action.preconditions.as_ref(), &bindings)
                {
                    for (atom, positive) in &literals {
                        if let Some(&variable) = local_index.get(atom) {
                            precondition.push((variable, usize::from(*positive)));
                        }
                    }
                }
                let mut effects = Vec::new();
                let mut grounded_effects = BTreeSet::new();
                if let Some(literals) = conjunctive_literals(action.effects.as_ref(), &bindings) {
                    for (atom, positive) in literals {
                        grounded_effects.insert(atom.clone());
                        if let Some(&variable) = local_index.get(&atom) {
                            effects.push((Vec::new(), variable, usize::from(positive)));
                        }
                    }
                }
                effects.push((Vec::new(), executed(position), 1));
                precondition.sort_unstable();
                precondition.dedup();
                operators.push((
                    grounded_name(&action.name, &task.terms),
                    precondition,
                    effects,
                    1.0,
                ));
                effect_atoms.push(grounded_effects);
                operator_tasks.push(task_index);
            }
        } else {
            let methods: Vec<_> = domain
                .methods_for(&task.name)
                .filter(|method| method.task_terms.len() == task.terms.len())
                .collect();
            if methods.is_empty() {
                operators.push((
                    format!("task[{task}]"),
                    base.clone(),
                    vec![(Vec::new(), executed(position), 1)],
                    1.0,
                ));
                effect_atoms.push(BTreeSet::new());
                operator_tasks.push(task_index);
            } else {
                for method in methods {
                    let bindings = bindings_from(&method.task_terms, &task.terms);
                    let mut precondition = base.clone();
                    if let Some(literals) =
                        conjunctive_literals(method.precondition.as_ref(), &bindings)
                    {
                        for (atom, positive) in &literals {
                            if let Some(&variable) = local_index.get(atom) {
                                precondition.push((variable, usize::from(*positive)));
                            }
                        }
                    }
                    precondition.sort_unstable();
                    precondition.dedup();
                    operators.push((
                        format!("{}[{task}]", method.name.name),
                        precondition,
                        vec![(Vec::new(), executed(position), 1)],
                        1.0,
                    ));
                    effect_atoms.push(BTreeSet::new());
                    operator_tasks.push(task_index);
                }
            }
        }
    }

    let task = fast_downward::Task::new(variables, initial, goal, operators, Vec::new())?;
    Ok(LeafPlanningTask {
        task,
        local_atoms,
        effect_atoms,
        operator_tasks,
    })
}
