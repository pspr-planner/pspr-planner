//! Heuristics over compiled planning tasks (Sections 6.1 and 6.2 of Helmert,
//! JAIR 2006).

use std::collections::{BTreeSet, HashSet};

use super::knowledge::CompiledTask;
use super::{Conditions, State, Task};

/// The estimate of a heuristic for a state plus the operators it prefers.
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicValue {
    pub estimate: f64,
    pub preferred: Vec<usize>,
}

/// Fact identifiers: `(variable, value)` is encoded as an index into a dense
/// array. `offsets[v]` is the identifier of `(v, 0)`.
fn fact_offsets(task: &Task) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(task.variables.len());
    let mut next = 0;
    for variable in &task.variables {
        offsets.push(next);
        next += variable.size;
    }
    offsets
}

fn fact_id(offsets: &[usize], variable: usize, value: usize) -> usize {
    offsets[variable] + value
}

fn condition_level(conditions: &Conditions, levels: &[f64], offsets: &[usize]) -> Option<f64> {
    let mut level: f64 = 0.0;
    for &(variable, value) in conditions {
        let fact_level = levels[fact_id(offsets, variable, value)];
        if fact_level.is_infinite() {
            return None;
        }
        level = level.max(fact_level);
    }
    Some(level)
}

#[derive(Debug, Clone, Copy)]
enum Achiever {
    Operator { index: usize, effect: usize },
    Axiom { index: usize },
}

struct Relaxation {
    levels: Vec<f64>,
    achievers: Vec<Option<Achiever>>,
    unreachable: bool,
}

/// Relaxed planning graph over facts (Section 6.2). Each variable may hold
/// several values at once; axioms are zero-cost rules that are applied to a
/// fixed point on every layer.
fn relaxed_planning_graph(compiled: &CompiledTask, state: &State) -> Relaxation {
    let task = compiled.task;
    let offsets = fact_offsets(task);
    let fact_count =
        offsets.last().copied().unwrap_or(0) + task.variables.last().map_or(0, |v| v.size);
    let mut levels = vec![f64::INFINITY; fact_count];
    let mut achievers = vec![None; fact_count];
    for (variable, &value) in state.iter().enumerate() {
        levels[fact_id(&offsets, variable, value)] = 0.0;
    }

    loop {
        let mut changed = false;
        for (index, axiom) in task.axioms.iter().enumerate() {
            let Some(before) = condition_level(&axiom.condition, &levels, &offsets) else {
                continue;
            };
            let fact = fact_id(&offsets, axiom.variable, axiom.value);
            if before < levels[fact] {
                levels[fact] = before;
                achievers[fact] = Some(Achiever::Axiom { index });
                changed = true;
            }
        }
        for (index, operator) in task.operators.iter().enumerate() {
            let Some(base) = condition_level(&operator.precondition, &levels, &offsets) else {
                continue;
            };
            for (effect_index, effect) in operator.effects.iter().enumerate() {
                let Some(condition) = condition_level(&effect.condition, &levels, &offsets) else {
                    continue;
                };
                let level = base.max(condition) + operator.cost;
                let fact = fact_id(&offsets, effect.variable, effect.value);
                if level < levels[fact] {
                    levels[fact] = level;
                    achievers[fact] = Some(Achiever::Operator {
                        index,
                        effect: effect_index,
                    });
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let unreachable = task
        .goal
        .iter()
        .any(|&(variable, value)| levels[fact_id(&offsets, variable, value)].is_infinite());
    Relaxation {
        levels,
        achievers,
        unreachable,
    }
}

/// The FF heuristic (`h_FF`) with helpful actions (Section 6.2).
pub fn ff_heuristic(compiled: &CompiledTask, state: &State) -> HeuristicValue {
    let task = compiled.task;
    let offsets = fact_offsets(task);
    let relaxation = relaxed_planning_graph(compiled, state);
    if relaxation.unreachable {
        return HeuristicValue {
            estimate: f64::INFINITY,
            preferred: Vec::new(),
        };
    }

    let mut relaxed_plan: BTreeSet<usize> = BTreeSet::new();
    let mut seen_facts: HashSet<usize> = HashSet::new();
    let mut queue: Vec<usize> = task
        .goal
        .iter()
        .map(|&(variable, value)| fact_id(&offsets, variable, value))
        .collect();
    while let Some(fact) = queue.pop() {
        if relaxation.levels[fact] == 0.0 || !seen_facts.insert(fact) {
            continue;
        }
        match relaxation.achievers[fact] {
            Some(Achiever::Axiom { index }) => {
                for &(variable, value) in &task.axioms[index].condition {
                    queue.push(fact_id(&offsets, variable, value));
                }
            }
            Some(Achiever::Operator { index, effect }) => {
                relaxed_plan.insert(index);
                let operator = &task.operators[index];
                for &(variable, value) in &operator.precondition {
                    queue.push(fact_id(&offsets, variable, value));
                }
                for &(variable, value) in &operator.effects[effect].condition {
                    queue.push(fact_id(&offsets, variable, value));
                }
            }
            None => {}
        }
    }

    let preferred = relaxed_plan
        .iter()
        .copied()
        .filter(|&operator| task.conditions_hold(&task.operators[operator].precondition, state))
        .collect();
    HeuristicValue {
        estimate: relaxed_plan.len() as f64,
        preferred,
    }
}

/// The `h_max` heuristic: the largest relaxed fact level among the goals.
pub fn h_max(compiled: &CompiledTask, state: &State) -> f64 {
    let task = compiled.task;
    let offsets = fact_offsets(task);
    let relaxation = relaxed_planning_graph(compiled, state);
    let mut estimate: f64 = 0.0;
    for &(variable, value) in &task.goal {
        let level = relaxation.levels[fact_id(&offsets, variable, value)];
        if level.is_infinite() {
            return f64::INFINITY;
        }
        estimate = estimate.max(level);
    }
    estimate
}

/// The causal graph heuristic `h_CG` (Section 6.1, Fig. 18).
///
/// Cost tables are computed bottom-up over the pruned causal graph, which
/// yields the same estimates as the top-down recursive formulation of
/// Fig. 18 (see the discussion in the paper following Fig. 17).
pub fn causal_graph_heuristic(compiled: &CompiledTask, state: &State) -> f64 {
    let task = compiled.task;
    let pruned = &compiled.pruned;
    let n = task.variables.len();

    // `costs[v][source][target]` is the cost of changing `v` from `source` to
    // `target` in the local subproblem rooted at `v`.
    let mut costs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
    for variable in &task.variables {
        costs.push(vec![vec![f64::INFINITY; variable.size]; variable.size]);
    }

    for &variable in &pruned.order {
        let size = task.variables[variable].size;
        let dtg = &compiled.dtgs[variable];
        for source in 0..size {
            let mut distance = vec![f64::INFINITY; size];
            let mut local_state: Vec<Option<State>> = vec![None; size];
            let mut settled = vec![false; size];
            distance[source] = 0.0;
            local_state[source] = Some(state.clone());

            loop {
                let mut current = usize::MAX;
                let mut current_distance = f64::INFINITY;
                for value in 0..size {
                    if !settled[value] && distance[value] < current_distance {
                        current_distance = distance[value];
                        current = value;
                    }
                }
                if current == usize::MAX {
                    break;
                }
                settled[current] = true;
                let base_state = local_state[current]
                    .clone()
                    .expect("settled values always have a local state");

                for transition in dtg.transitions.iter().filter(|t| t.from == current) {
                    if settled[transition.to] {
                        continue;
                    }
                    let mut cost = transition.cost;
                    let mut next_state = base_state.clone();
                    let mut feasible = true;
                    for &(other, value) in &transition.conditions {
                        if other == variable {
                            continue;
                        }
                        if pruned.is_predecessor(other, variable) {
                            let from = next_state[other];
                            let additional = costs[other][from][value];
                            if additional.is_infinite() {
                                feasible = false;
                                break;
                            }
                            cost += additional;
                            next_state[other] = value;
                        }
                        // Conditions on non-predecessors have been pruned;
                        // any that remain are treated as satisfied.
                    }
                    if !feasible {
                        continue;
                    }
                    let candidate = distance[current] + cost;
                    if candidate < distance[transition.to] {
                        distance[transition.to] = candidate;
                        local_state[transition.to] = Some(next_state);
                    }
                }
            }
            costs[variable][source] = distance;
        }
    }

    let mut estimate = 0.0;
    for &(variable, value) in &task.goal {
        let cost = costs[variable][state[variable]][value];
        if cost.is_infinite() {
            return f64::INFINITY;
        }
        estimate += cost;
    }
    estimate
}
