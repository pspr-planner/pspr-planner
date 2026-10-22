//! Leaf planning tasks: local constraints encoded for Fast Downward.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, ensure};

use crate::solver::domain::Domain;
use crate::solver::fast_downward;
use crate::solver::problem::Problem;

use super::clustering::EvaluationTree;
use super::formula::{self, Universe};
use super::graph::{Atom, ConstraintNode, bindings_from};

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
    problem: &Problem,
    tree: &EvaluationTree,
    cluster_index: usize,
    initial_atoms: &BTreeSet<String>,
    max_groundings: usize,
) -> Result<LeafPlanningTask> {
    let cluster = &tree.clusters[cluster_index];
    let local_ids = &tree.local[cluster_index];
    let universe = Universe::new(domain, problem);

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
                let clauses = formula::precondition_clauses(
                    action.preconditions.as_ref(),
                    &bindings,
                    &universe,
                    max_groundings,
                )?;
                if clauses.is_empty() {
                    // The precondition is unsatisfiable: the action can never run.
                    continue;
                }
                let alternatives = formula::effect_alternatives_of(
                    action.effects.as_ref(),
                    &bindings,
                    &universe,
                    max_groundings,
                )?;
                if alternatives.is_empty() {
                    // Every outcome of the effect formula is impossible.
                    continue;
                }
                for clause in &clauses {
                    for alternative in &alternatives {
                        let mut precondition = base.clone();
                        for (atom, positive) in clause {
                            if let Some(&variable) = local_index.get(atom) {
                                precondition.push((variable, usize::from(*positive)));
                            }
                        }
                        let mut effects = Vec::new();
                        let mut grounded_effects = BTreeSet::new();
                        for term in alternative {
                            if term.value {
                                grounded_effects.insert(term.atom.clone());
                            }
                            if let Some(&variable) = local_index.get(&term.atom) {
                                let condition: Vec<(usize, usize)> = term
                                    .condition
                                    .iter()
                                    .filter_map(|(atom, positive)| {
                                        local_index
                                            .get(atom)
                                            .map(|&variable| (variable, usize::from(*positive)))
                                    })
                                    .collect();
                                effects.push((condition, variable, usize::from(term.value)));
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
                }
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
                    let clauses = formula::precondition_clauses(
                        method.precondition.as_ref(),
                        &bindings,
                        &universe,
                        max_groundings,
                    )?;
                    if clauses.is_empty() {
                        continue;
                    }
                    for clause in &clauses {
                        let mut precondition = base.clone();
                        for (atom, positive) in clause {
                            if let Some(&variable) = local_index.get(atom) {
                                precondition.push((variable, usize::from(*positive)));
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
    }

    let task = fast_downward::Task::new(variables, initial, goal, operators, Vec::new())?;
    Ok(LeafPlanningTask {
        task,
        local_atoms,
        effect_atoms,
        operator_tasks,
    })
}
