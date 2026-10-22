//! Domain transition graphs, causal graphs, and causal-graph pruning
//! (Sections 5.1 and 5.2 of Helmert, JAIR 2006).

use std::collections::{BTreeMap, BTreeSet};

use super::{Conditions, Task};

/// A labelled transition of a domain transition graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub from: usize,
    pub to: usize,
    /// Conditions on *other* variables that must hold for the transition.
    pub conditions: Conditions,
    pub cost: f64,
    /// The operator inducing the transition, if it is not induced by an axiom.
    pub operator: Option<usize>,
    /// The axiom inducing the transition, if it is not induced by an operator.
    pub axiom: Option<usize>,
}

/// The domain transition graph of one state variable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DomainTransitionGraph {
    pub transitions: Vec<Transition>,
}

fn merge_conditions(first: &Conditions, second: &Conditions) -> Option<Conditions> {
    let mut merged: BTreeMap<usize, usize> = first.iter().copied().collect();
    for &(variable, value) in second {
        match merged.get(&variable) {
            Some(&previous) if previous != value => return None,
            _ => {
                merged.insert(variable, value);
            }
        }
    }
    Some(merged.into_iter().collect())
}

pub(super) fn build_domain_transition_graphs(task: &Task) -> Vec<DomainTransitionGraph> {
    let mut dtgs = vec![DomainTransitionGraph::default(); task.variables.len()];
    for (index, operator) in task.operators.iter().enumerate() {
        for effect in &operator.effects {
            let Some(mut labels) = merge_conditions(&operator.precondition, &effect.condition)
            else {
                // The operator is trivially inapplicable; it induces no transition.
                continue;
            };
            let affected = effect.variable;
            if let Some(position) = labels
                .iter()
                .position(|&(variable, _)| variable == affected)
            {
                let (_, from) = labels.remove(position);
                dtgs[affected].transitions.push(Transition {
                    from,
                    to: effect.value,
                    conditions: labels,
                    cost: operator.cost,
                    operator: Some(index),
                    axiom: None,
                });
            } else {
                for from in 0..task.variables[affected].size {
                    if from != effect.value {
                        dtgs[affected].transitions.push(Transition {
                            from,
                            to: effect.value,
                            conditions: labels.clone(),
                            cost: operator.cost,
                            operator: Some(index),
                            axiom: None,
                        });
                    }
                }
            }
        }
    }
    for (index, axiom) in task.axioms.iter().enumerate() {
        let affected = axiom.variable;
        if let Some(position) = axiom
            .condition
            .iter()
            .position(|&(variable, _)| variable == affected)
        {
            let mut labels = axiom.condition.clone();
            let (_, from) = labels.remove(position);
            dtgs[affected].transitions.push(Transition {
                from,
                to: axiom.value,
                conditions: labels,
                cost: 0.0,
                operator: None,
                axiom: Some(index),
            });
        } else {
            for from in 0..task.variables[affected].size {
                if from != axiom.value {
                    dtgs[affected].transitions.push(Transition {
                        from,
                        to: axiom.value,
                        conditions: axiom.condition.clone(),
                        cost: 0.0,
                        operator: None,
                        axiom: Some(index),
                    });
                }
            }
        }
    }
    dtgs
}

// ---------------------------------------------------------------------------
// Causal graphs (Section 5.2)
// ---------------------------------------------------------------------------

/// The causal graph (Definition 6). `arcs[v]` maps each successor `v'` of `v`
/// to the number of operators or axioms inducing the arc `(v, v')`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CausalGraph {
    pub arcs: Vec<BTreeMap<usize, usize>>,
}

impl CausalGraph {
    pub(super) fn build(task: &Task, dtgs: &[DomainTransitionGraph]) -> Self {
        let mut graph = CausalGraph {
            arcs: vec![BTreeMap::new(); task.variables.len()],
        };
        // Arcs induced by transition conditions.
        for (target, dtg) in dtgs.iter().enumerate() {
            for transition in &dtg.transitions {
                for &(source, _) in &transition.conditions {
                    if source != target {
                        *graph.arcs[source].entry(target).or_insert(0) += 1;
                    }
                }
            }
        }
        // Arcs induced by co-occurring effects.
        for operator in &task.operators {
            let affected: BTreeSet<usize> = operator.effects.iter().map(|e| e.variable).collect();
            for &source in &affected {
                for &target in &affected {
                    if source != target {
                        *graph.arcs[source].entry(target).or_insert(0) += 1;
                    }
                }
            }
        }
        graph
    }

    /// All variables that can reach a goal variable (including the goal
    /// variables themselves).
    fn goal_ancestors(&self, goal: &Conditions) -> Vec<bool> {
        let mut relevant = vec![false; self.arcs.len()];
        let mut stack: Vec<usize> = goal.iter().map(|&(variable, _)| variable).collect();
        while let Some(variable) = stack.pop() {
            if relevant[variable] {
                continue;
            }
            relevant[variable] = true;
            for (predecessor, targets) in self.arcs.iter().enumerate() {
                if targets.contains_key(&variable) {
                    stack.push(predecessor);
                }
            }
        }
        relevant
    }

    /// Prune cycles as described in Section 5.2.2.
    pub fn prune(&self, task: &Task) -> PrunedCausalGraph {
        let relevant = self.goal_ancestors(&task.goal);
        let n = self.arcs.len();

        let components = strongly_connected_components(n, &self.arcs);
        let mut component_of = vec![0usize; n];
        for (index, component) in components.iter().enumerate() {
            for &variable in component {
                component_of[variable] = index;
            }
        }

        // Topological order of the condensation, restricted to relevant
        // components. Kahn's algorithm with a sorted queue is deterministic.
        let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); components.len()];
        let mut indegree = vec![0usize; components.len()];
        for (source, targets) in self.arcs.iter().enumerate() {
            if !relevant[source] {
                continue;
            }
            for &target in targets.keys() {
                if !relevant[target] {
                    continue;
                }
                let (from, to) = (component_of[source], component_of[target]);
                if from != to && successors[from].insert(to) {
                    indegree[to] += 1;
                }
            }
        }
        let mut queue: BTreeSet<usize> = (0..components.len())
            .filter(|&component| relevant[components[component][0]] && indegree[component] == 0)
            .collect();
        let mut component_order = Vec::new();
        while let Some(component) = queue.pop_first() {
            component_order.push(component);
            for &successor in &successors[component] {
                indegree[successor] -= 1;
                if indegree[successor] == 0 {
                    queue.insert(successor);
                }
            }
        }

        // Within each strongly connected component, greedily order the
        // variables by the total weight of their incoming arcs.
        let mut levels = vec![usize::MAX; n];
        let mut order = Vec::new();
        for component in component_order {
            let members = &components[component];
            let mut remaining: BTreeSet<usize> = members.iter().copied().collect();
            while !remaining.is_empty() {
                let mut best = usize::MAX;
                let mut best_weight = usize::MAX;
                for &candidate in &remaining {
                    let weight: usize = remaining
                        .iter()
                        .filter(|&&source| source != candidate)
                        .filter_map(|source| self.arcs[*source].get(&candidate).copied())
                        .sum();
                    if weight < best_weight {
                        best_weight = weight;
                        best = candidate;
                    }
                }
                levels[best] = order.len();
                order.push(best);
                remaining.remove(&best);
            }
        }

        // Keep only arcs that agree with the total order.
        let mut arcs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for (source, targets) in self.arcs.iter().enumerate() {
            if !relevant[source] {
                continue;
            }
            for &target in targets.keys() {
                if relevant[target] && levels[source] < levels[target] {
                    arcs[source].insert(target);
                }
            }
        }

        PrunedCausalGraph {
            arcs,
            levels,
            order,
            relevant,
        }
    }
}

/// A causal graph with cycles removed, as used by the causal graph heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunedCausalGraph {
    /// Pruned arcs; the graph is acyclic.
    pub arcs: Vec<BTreeSet<usize>>,
    /// Total order on the variables: `levels[v] < levels[v']` means that `v`
    /// has a lower level than `v'`.
    pub levels: Vec<usize>,
    /// Relevant variables in bottom-up (predecessor-first) order.
    pub order: Vec<usize>,
    /// Variables that are ancestors of the goal.
    pub relevant: Vec<bool>,
}

impl PrunedCausalGraph {
    /// Whether `source` is an immediate causal predecessor of `target`.
    pub fn is_predecessor(&self, source: usize, target: usize) -> bool {
        self.arcs[source].contains(&target)
    }
}

/// Tarjan's algorithm; returns the strongly connected components in reverse
/// topological order.
fn strongly_connected_components(n: usize, arcs: &[BTreeMap<usize, usize>]) -> Vec<Vec<usize>> {
    struct Tarjan<'a> {
        arcs: &'a [BTreeMap<usize, usize>],
        index: Vec<usize>,
        low: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        next_index: usize,
        components: Vec<Vec<usize>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, v: usize) {
            self.index[v] = self.next_index;
            self.low[v] = self.next_index;
            self.next_index += 1;
            self.stack.push(v);
            self.on_stack[v] = true;
            let targets: Vec<usize> = self.arcs[v].keys().copied().collect();
            for target in targets {
                if self.index[target] == usize::MAX {
                    self.visit(target);
                    self.low[v] = self.low[v].min(self.low[target]);
                } else if self.on_stack[target] {
                    self.low[v] = self.low[v].min(self.index[target]);
                }
            }
            if self.low[v] == self.index[v] {
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("Tarjan stack is non-empty");
                    self.on_stack[member] = false;
                    component.push(member);
                    if member == v {
                        break;
                    }
                }
                component.sort_unstable();
                self.components.push(component);
            }
        }
    }

    let mut tarjan = Tarjan {
        arcs,
        index: vec![usize::MAX; n],
        low: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        next_index: 0,
        components: Vec::new(),
    };
    for variable in 0..n {
        if tarjan.index[variable] == usize::MAX {
            tarjan.visit(variable);
        }
    }
    tarjan.components
}

pub(super) fn prune_and_simplify_dtgs(
    mut dtgs: Vec<DomainTransitionGraph>,
    pruned: &PrunedCausalGraph,
) -> Vec<DomainTransitionGraph> {
    for (variable, graph) in dtgs.iter_mut().enumerate() {
        // Ignore conditions on variables at a higher causal level.
        for transition in &mut graph.transitions {
            transition
                .conditions
                .retain(|&(other, _)| pruned.levels[other] <= pruned.levels[variable]);
        }
        graph.transitions = simplify_transitions(std::mem::take(&mut graph.transitions));
    }
    dtgs
}

/// Remove dominated transitions: if `t` and `t'` connect the same values and
/// the condition of `t` is a subset of the condition of `t'` while `t` is not
/// more expensive, then `t'` is never useful.
fn simplify_transitions(mut transitions: Vec<Transition>) -> Vec<Transition> {
    transitions.sort_by(|left, right| {
        (left.from, left.to)
            .cmp(&(right.from, right.to))
            .then_with(|| left.conditions.len().cmp(&right.conditions.len()))
            .then_with(|| left.cost.total_cmp(&right.cost))
    });
    let mut kept: Vec<Transition> = Vec::new();
    for transition in transitions {
        let dominated = kept.iter().any(|other| {
            other.from == transition.from
                && other.to == transition.to
                && other.cost <= transition.cost
                && other
                    .conditions
                    .iter()
                    .all(|condition| transition.conditions.contains(condition))
        });
        if !dominated {
            kept.push(transition);
        }
    }
    kept
}
