//! Leaf planning tasks: local constraints encoded for Fast Downward.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, ensure};

use crate::parsing::json::TaskOrdering;
use crate::solver::domain::Domain;
use crate::solver::fast_downward;
use crate::solver::problem::Problem;

use super::clustering::EvaluationTree;
use super::formula::{self, Universe};
use super::graph::{Atom, ConstraintNode, TaskNode, bindings_from, ground_methods_bounded};

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

struct SummaryOperator {
    name: String,
    precondition: Vec<(Atom, bool)>,
    add: BTreeSet<Atom>,
    del: BTreeSet<Atom>,
}

fn bounded_precondition_clauses(
    formula: Option<&crate::parsing::json::Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    max_groundings: usize,
) -> Result<Vec<formula::Clause>> {
    match formula::precondition_clauses(formula, bindings, universe, max_groundings) {
        Ok(clauses) => Ok(clauses),
        Err(error) if formula::is_quantifier_overflow(&error) => Ok(Vec::new()),
        Err(error) => Err(error),
    }
}

fn bounded_effect_alternatives(
    formula: Option<&crate::parsing::json::Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    max_groundings: usize,
) -> Result<Vec<formula::EffectAlternative>> {
    match formula::effect_alternatives_of(formula, bindings, universe, max_groundings) {
        Ok(alternatives) => Ok(alternatives),
        Err(error) if formula::is_quantifier_overflow(&error) => Ok(Vec::new()),
        Err(error) => Err(error),
    }
}

/// Compile a method whose subtasks are primitive, totally ordered and free of
/// conditional effects into a macro operator. The summary is exact under the
/// restrictions stated in `docs/compound-tasks.md` (Approach G); any method
/// outside those restrictions is skipped and stays opaque.
fn summary_operators(
    domain: &Domain,
    universe: &Universe<'_>,
    task: &TaskNode,
    max_groundings: usize,
) -> Result<Vec<SummaryOperator>> {
    let mut summaries = Vec::new();
    for instance in ground_methods_bounded(domain, universe, task, max_groundings).instances {
        let method = instance.method;
        if !matches!(method.tn.orderings, TaskOrdering::Total) || method.tn.constraints.is_some() {
            continue;
        }

        let mut actions = Vec::new();
        let mut valid = true;
        for subtask in &method.tn.subtasks {
            let terms: Vec<String> = subtask
                .terms
                .iter()
                .map(|term| {
                    instance
                        .bindings
                        .get(&term.name)
                        .cloned()
                        .unwrap_or_else(|| term.name.clone())
                })
                .collect();
            if domain.methods_for(&subtask.task.name).next().is_some() {
                valid = false;
                break;
            }
            let Some(action) = domain
                .actions_for(&subtask.task.name)
                .find(|action| action.parameters.len() == terms.len())
            else {
                valid = false;
                break;
            };
            let action_bindings = bindings_from(&action.parameters, &terms);
            let clauses = bounded_precondition_clauses(
                action.preconditions.as_ref(),
                &action_bindings,
                universe,
                max_groundings,
            )?;
            let effects = bounded_effect_alternatives(
                action.effects.as_ref(),
                &action_bindings,
                universe,
                max_groundings,
            )?;
            if clauses.len() != 1 || effects.len() != 1 {
                valid = false;
                break;
            }
            let alternative = &effects[0];
            if alternative.iter().any(|term| !term.condition.is_empty()) {
                valid = false;
                break;
            }
            let add: BTreeSet<Atom> = alternative
                .iter()
                .filter(|term| term.value)
                .map(|term| term.atom.clone())
                .collect();
            let del: BTreeSet<Atom> = alternative
                .iter()
                .filter(|term| !term.value)
                .map(|term| term.atom.clone())
                .collect();
            if !add.is_disjoint(&del) {
                valid = false;
                break;
            }
            actions.push((clauses[0].clone(), add, del));
        }
        if !valid {
            continue;
        }

        let mut required: BTreeMap<Atom, bool> = BTreeMap::new();
        let mut add: BTreeSet<Atom> = BTreeSet::new();
        let mut del: BTreeSet<Atom> = BTreeSet::new();
        for (clause, action_add, action_del) in actions.iter().rev() {
            // `required` describes the formula that must hold after this
            // action. Regress it through the action's effects.
            let mut next = BTreeMap::new();
            for (atom, positive) in &required {
                if action_add.contains(atom) {
                    if !*positive {
                        valid = false;
                        break;
                    }
                } else if action_del.contains(atom) {
                    if *positive {
                        valid = false;
                        break;
                    }
                } else {
                    next.insert(atom.clone(), *positive);
                }
            }
            if !valid {
                break;
            }

            // The action's own precondition must hold before the action; it
            // cannot be satisfied by the action's or a later action's effects.
            for (atom, positive) in clause {
                match next.get(atom) {
                    Some(existing) if existing != positive => {
                        valid = false;
                        break;
                    }
                    Some(_) => {}
                    None => {
                        next.insert(atom.clone(), *positive);
                    }
                }
            }
            if !valid {
                break;
            }
            required = next;

            for atom in action_add {
                if !del.contains(atom) {
                    add.insert(atom.clone());
                }
            }
            for atom in action_del {
                if !add.contains(atom) {
                    del.insert(atom.clone());
                }
            }
        }
        if !valid {
            continue;
        }

        let method_clauses = bounded_precondition_clauses(
            method.precondition.as_ref(),
            &instance.bindings,
            universe,
            max_groundings,
        )?;
        for clause in method_clauses {
            let mut precondition = clause;
            let mut compatible = true;
            for (atom, &positive) in &required {
                match precondition.iter().find(|(candidate, _)| candidate == atom) {
                    Some((_, existing)) if *existing != positive => {
                        compatible = false;
                        break;
                    }
                    Some(_) => {}
                    None => precondition.push((atom.clone(), positive)),
                }
            }
            if !compatible {
                continue;
            }
            summaries.push(SummaryOperator {
                name: format!("{}[{task}]", method.name.name),
                precondition,
                add: add.clone(),
                del: del.clone(),
            });
        }
    }
    Ok(summaries)
}

pub(super) fn build_leaf_task(
    domain: &Domain,
    problem: &Problem,
    tree: &EvaluationTree,
    cluster_index: usize,
    initial_atoms: &BTreeSet<String>,
    max_groundings: usize,
    summaries: bool,
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
                let clauses = bounded_precondition_clauses(
                    action.preconditions.as_ref(),
                    &bindings,
                    &universe,
                    max_groundings,
                )?;
                if clauses.is_empty() {
                    // The precondition is unsatisfiable: the action can never run.
                    continue;
                }
                let alternatives = bounded_effect_alternatives(
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
            let has_methods = domain
                .methods_for(&task.name)
                .any(|method| method.task_terms.len() == task.terms.len());
            if !has_methods {
                operators.push((
                    format!("task[{task}]"),
                    base.clone(),
                    vec![(Vec::new(), executed(position), 1)],
                    1.0,
                ));
                effect_atoms.push(BTreeSet::new());
                operator_tasks.push(task_index);
            } else {
                let summaries = if summaries {
                    summary_operators(domain, &universe, task, max_groundings)?
                } else {
                    Vec::new()
                };
                let mut summarised_methods = BTreeSet::new();
                for summary in summaries {
                    summarised_methods.insert(
                        summary
                            .name
                            .split_once('[')
                            .map(|(name, _)| name.to_string())
                            .unwrap_or_else(|| summary.name.clone()),
                    );
                    let mut precondition = base.clone();
                    for (atom, positive) in &summary.precondition {
                        if let Some(&variable) = local_index.get(atom) {
                            precondition.push((variable, usize::from(*positive)));
                        }
                    }
                    let mut effects = Vec::new();
                    let mut grounded_effects = BTreeSet::new();
                    for atom in &summary.add {
                        grounded_effects.insert(atom.clone());
                        if let Some(&variable) = local_index.get(atom) {
                            effects.push((Vec::new(), variable, 1));
                        }
                    }
                    for atom in &summary.del {
                        if let Some(&variable) = local_index.get(atom) {
                            effects.push((Vec::new(), variable, 0));
                        }
                    }
                    effects.push((Vec::new(), executed(position), 1));
                    precondition.sort_unstable();
                    precondition.dedup();
                    operators.push((summary.name, precondition, effects, 1.0));
                    effect_atoms.push(grounded_effects);
                    operator_tasks.push(task_index);
                }

                // Methods that could not be summarised remain opaque.
                for method in domain
                    .methods_for(&task.name)
                    .filter(|method| method.task_terms.len() == task.terms.len())
                {
                    if summarised_methods.contains(&method.name.name) {
                        continue;
                    }
                    let bindings = bindings_from(&method.task_terms, &task.terms);
                    let clauses = bounded_precondition_clauses(
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
