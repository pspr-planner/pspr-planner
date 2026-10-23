//! Heuristics over compiled planning tasks (Sections 6.1 and 6.2 of Helmert,
//! JAIR 2006).

use super::State;
use super::knowledge::CompiledTask;

/// The estimate of a heuristic for a state plus the operators it prefers.
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicValue {
    pub estimate: f64,
    pub preferred: Vec<usize>,
}

/// Buffers reused by every heuristic evaluation of one search.
///
/// The relaxed planning graph is built for every evaluated state and used to
/// allocate three vectors. The search evaluates up to millions of states, so
/// the buffers are kept alive here instead; generations replace the per-state
/// hash sets that the FF extraction used to build.
#[derive(Debug)]
pub struct HeuristicScratch {
    levels: Vec<f64>,
    achievers: Vec<Option<Achiever>>,
    /// Generation stamp of every fact for `seen_facts`.
    seen_facts: Vec<u32>,
    /// Fact queue of the relaxed-plan extraction.
    queue: Vec<usize>,
    /// Operators of the relaxed plan, deduplicated by `seen_operators`.
    relaxed_plan: Vec<usize>,
    seen_operators: Vec<u32>,
    /// Generation of the current evaluation; `0` means "unstamped".
    generation: u32,
    /// Flat `costs[v][source][target]` tables of the causal graph heuristic.
    costs: Vec<f64>,
    distance: Vec<f64>,
    settled: Vec<bool>,
    local_state: Vec<State>,
}

impl Default for HeuristicScratch {
    fn default() -> Self {
        HeuristicScratch {
            levels: Vec::new(),
            achievers: Vec::new(),
            seen_facts: Vec::new(),
            queue: Vec::new(),
            relaxed_plan: Vec::new(),
            seen_operators: Vec::new(),
            generation: 0,
            costs: Vec::new(),
            distance: Vec::new(),
            settled: Vec::new(),
            local_state: Vec::new(),
        }
    }
}

impl HeuristicScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new evaluation and return its stamp.
    fn next_generation(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1).max(1);
        if self.generation == 1 {
            // The counter wrapped: forget the stale stamps.
            self.seen_facts.fill(0);
            self.seen_operators.fill(0);
        }
        self.generation
    }
}

#[derive(Debug, Clone, Copy)]
enum Achiever {
    Operator { index: usize, effect: usize },
    Axiom { index: usize },
}

struct Relaxation {
    unreachable: bool,
}

/// Relaxed planning graph over facts (Section 6.2). Each variable may hold
/// several values at once; axioms are zero-cost rules that are applied to a
/// fixed point on every layer.
///
/// The fact levels and the achievers are written into `scratch`.
fn relaxed_planning_graph(
    compiled: &CompiledTask,
    state: &State,
    scratch: &mut HeuristicScratch,
) -> Relaxation {
    let task = compiled.task;
    let fact_count = compiled.fact_count;
    scratch.levels.clear();
    scratch.levels.resize(fact_count, f64::INFINITY);
    scratch.achievers.clear();
    scratch.achievers.resize(fact_count, None);
    for (variable, &value) in state.iter().enumerate() {
        scratch.levels[compiled.fact_id(variable, value)] = 0.0;
    }

    loop {
        let mut changed = false;
        for (index, axiom) in task.axioms.iter().enumerate() {
            let Some(before) =
                condition_level(axiom.condition.as_slice(), compiled, &scratch.levels)
            else {
                continue;
            };
            let fact = compiled.fact_id(axiom.variable, axiom.value);
            if before < scratch.levels[fact] {
                scratch.levels[fact] = before;
                scratch.achievers[fact] = Some(Achiever::Axiom { index });
                changed = true;
            }
        }
        for (index, operator) in task.operators.iter().enumerate() {
            let Some(base) =
                condition_level(operator.precondition.as_slice(), compiled, &scratch.levels)
            else {
                continue;
            };
            for (effect_index, effect) in operator.effects.iter().enumerate() {
                let Some(condition) =
                    condition_level(effect.condition.as_slice(), compiled, &scratch.levels)
                else {
                    continue;
                };
                let level = base.max(condition) + operator.cost;
                let fact = compiled.fact_id(effect.variable, effect.value);
                if level < scratch.levels[fact] {
                    scratch.levels[fact] = level;
                    scratch.achievers[fact] = Some(Achiever::Operator {
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
        .any(|&(variable, value)| scratch.levels[compiled.fact_id(variable, value)].is_infinite());
    Relaxation { unreachable }
}

fn condition_level(
    conditions: &[(usize, usize)],
    compiled: &CompiledTask,
    levels: &[f64],
) -> Option<f64> {
    let mut level: f64 = 0.0;
    for &(variable, value) in conditions {
        let fact_level = levels[compiled.fact_id(variable, value)];
        if fact_level.is_infinite() {
            return None;
        }
        level = level.max(fact_level);
    }
    Some(level)
}

/// The FF heuristic (`h_FF`) with helpful actions (Section 6.2).
pub fn ff_heuristic(compiled: &CompiledTask, state: &State) -> HeuristicValue {
    let mut scratch = HeuristicScratch::new();
    ff_heuristic_with(compiled, state, &mut scratch)
}

/// [`ff_heuristic`] reusing `scratch` across evaluations.
pub fn ff_heuristic_with(
    compiled: &CompiledTask,
    state: &State,
    scratch: &mut HeuristicScratch,
) -> HeuristicValue {
    let task = compiled.task;
    let relaxation = relaxed_planning_graph(compiled, state, scratch);
    if relaxation.unreachable {
        return HeuristicValue {
            estimate: f64::INFINITY,
            preferred: Vec::new(),
        };
    }

    let generation = scratch.next_generation();
    scratch.seen_facts.resize(compiled.fact_count, 0);
    scratch.seen_operators.resize(task.operators.len(), 0);
    scratch.relaxed_plan.clear();
    scratch.queue.clear();
    scratch.queue.extend(
        task.goal
            .iter()
            .map(|&(variable, value)| compiled.fact_id(variable, value)),
    );
    while let Some(fact) = scratch.queue.pop() {
        if scratch.levels[fact] == 0.0 || scratch.seen_facts[fact] == generation {
            continue;
        }
        scratch.seen_facts[fact] = generation;
        match scratch.achievers[fact] {
            Some(Achiever::Axiom { index }) => {
                for &(variable, value) in &task.axioms[index].condition {
                    scratch.queue.push(compiled.fact_id(variable, value));
                }
            }
            Some(Achiever::Operator { index, effect }) => {
                if scratch.seen_operators[index] != generation {
                    scratch.seen_operators[index] = generation;
                    scratch.relaxed_plan.push(index);
                }
                let operator = &task.operators[index];
                for &(variable, value) in &operator.precondition {
                    scratch.queue.push(compiled.fact_id(variable, value));
                }
                for &(variable, value) in &operator.effects[effect].condition {
                    scratch.queue.push(compiled.fact_id(variable, value));
                }
            }
            None => {}
        }
    }

    // The extraction used to collect the operators in a `BTreeSet`; sorting the
    // reusable vector keeps the helpful actions in the same order.
    scratch.relaxed_plan.sort_unstable();
    let preferred = scratch
        .relaxed_plan
        .iter()
        .copied()
        .filter(|&operator| task.conditions_hold(&task.operators[operator].precondition, state))
        .collect();
    HeuristicValue {
        estimate: scratch.relaxed_plan.len() as f64,
        preferred,
    }
}

/// The `h_max` heuristic: the largest relaxed fact level among the goals.
pub fn h_max(compiled: &CompiledTask, state: &State) -> f64 {
    let mut scratch = HeuristicScratch::new();
    h_max_with(compiled, state, &mut scratch)
}

/// [`h_max`] reusing `scratch` across evaluations.
pub fn h_max_with(compiled: &CompiledTask, state: &State, scratch: &mut HeuristicScratch) -> f64 {
    let _ = relaxed_planning_graph(compiled, state, scratch);
    let mut estimate: f64 = 0.0;
    for &(variable, value) in &compiled.task.goal {
        let level = scratch.levels[compiled.fact_id(variable, value)];
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
    let mut scratch = HeuristicScratch::new();
    causal_graph_heuristic_with(compiled, state, &mut scratch)
}

/// [`causal_graph_heuristic`] reusing `scratch` across evaluations.
pub fn causal_graph_heuristic_with(
    compiled: &CompiledTask,
    state: &State,
    scratch: &mut HeuristicScratch,
) -> f64 {
    let task = compiled.task;
    let pruned = &compiled.pruned;

    // `costs[v][source][target]` is the cost of changing `v` from `source` to
    // `target` in the local subproblem rooted at `v`. The tables are stored
    // flat in one reusable buffer: one allocation per evaluation instead of a
    // vector of vectors of vectors.
    let mut cost_base = vec![0usize; task.variables.len()];
    let mut total = 0usize;
    for (index, variable) in task.variables.iter().enumerate() {
        cost_base[index] = total;
        total += variable.size * variable.size;
    }
    scratch.costs.clear();
    scratch.costs.resize(total, f64::INFINITY);

    for &variable in &pruned.order {
        let size = task.variables[variable].size;
        let dtg = &compiled.dtgs[variable];
        for source in 0..size {
            scratch.distance.clear();
            scratch.distance.resize(size, f64::INFINITY);
            scratch.local_state.clear();
            scratch.local_state.resize(size, Vec::new());
            scratch.settled.clear();
            scratch.settled.resize(size, false);
            scratch.distance[source] = 0.0;
            scratch.local_state[source] = state.clone();

            loop {
                let mut current = usize::MAX;
                let mut current_distance = f64::INFINITY;
                for value in 0..size {
                    if !scratch.settled[value] && scratch.distance[value] < current_distance {
                        current_distance = scratch.distance[value];
                        current = value;
                    }
                }
                if current == usize::MAX {
                    break;
                }
                scratch.settled[current] = true;
                let base_state = std::mem::take(&mut scratch.local_state[current]);

                for transition in dtg.transitions.iter().filter(|t| t.from == current) {
                    if scratch.settled[transition.to] {
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
                            let additional = scratch.costs
                                [cost_base[other] + from * task.variables[other].size + value];
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
                    let candidate = scratch.distance[current] + cost;
                    if candidate < scratch.distance[transition.to] {
                        scratch.distance[transition.to] = candidate;
                        scratch.local_state[transition.to] = next_state;
                    }
                }
            }
            let base = cost_base[variable];
            scratch.costs[base + source * size..base + source * size + size]
                .copy_from_slice(&scratch.distance);
        }
    }

    let mut estimate = 0.0;
    for &(variable, value) in &task.goal {
        let size = task.variables[variable].size;
        let cost = scratch.costs[cost_base[variable] + state[variable] * size + value];
        if cost.is_infinite() {
            return f64::INFINITY;
        }
        estimate += cost;
    }
    estimate
}
