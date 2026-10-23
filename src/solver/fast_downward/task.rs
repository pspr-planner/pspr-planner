//! Multi-valued planning tasks (Section 4 of Helmert, JAIR 2006).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Result, bail, ensure};

use super::{Conditions, OperatorSpec, State, UNDEFINED};

/// A state variable with a finite domain `{0, ..., size - 1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub size: usize,
    /// Derived variables are computed by axioms and cannot be changed by
    /// operators. Their default value is [`UNDEFINED`].
    pub derived: bool,
}

/// A conditional effect `condition -> variable := value` on a fluent.
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub condition: Conditions,
    pub variable: usize,
    pub value: usize,
}

/// An operator `precondition -> effects` with an associated cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    pub name: String,
    pub precondition: Conditions,
    pub effects: Vec<Effect>,
    pub cost: f64,
}

/// A layered axiom `condition -> variable := value` defining a derived variable.
#[derive(Debug, Clone, PartialEq)]
pub struct Axiom {
    pub condition: Conditions,
    pub variable: usize,
    pub value: usize,
    /// Axioms are evaluated layer by layer in increasing layer order.
    pub layer: usize,
}

/// A multi-valued planning task (Definition 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub variables: Vec<Variable>,
    pub initial: State,
    pub goal: Conditions,
    pub operators: Vec<Operator>,
    pub axioms: Vec<Axiom>,
}

impl Task {
    /// Build a task from variables `(name, domain size, derived)` and operators
    /// given as `(name, precondition, effects, cost)` with effects encoded as
    /// `(condition, variable, value)`.
    pub fn new(
        variables: Vec<(String, usize, bool)>,
        initial: State,
        goal: Conditions,
        operators: Vec<OperatorSpec>,
        axioms: Vec<Axiom>,
    ) -> Result<Self> {
        let task = Task {
            variables: variables
                .into_iter()
                .map(|(name, size, derived)| Variable {
                    name,
                    size,
                    derived,
                })
                .collect(),
            initial,
            goal,
            operators: operators
                .into_iter()
                .map(|(name, precondition, effects, cost)| Operator {
                    name,
                    precondition,
                    effects: effects
                        .into_iter()
                        .map(|(condition, variable, value)| Effect {
                            condition,
                            variable,
                            value,
                        })
                        .collect(),
                    cost,
                })
                .collect(),
            axioms,
        };
        task.validate()?;
        Ok(task)
    }

    /// The number of variables of the task.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Whether `conditions` hold in the extended state `state`.
    pub fn conditions_hold(&self, conditions: &Conditions, state: &State) -> bool {
        conditions
            .iter()
            .all(|&(variable, value)| state[variable] == value)
    }

    /// Check that the task is well-formed. Called by [`Task::new`] and
    /// [`Task::compile`].
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.variables.is_empty(),
            "a task needs at least one state variable"
        );
        ensure!(
            self.initial.len() == self.variables.len(),
            "the initial state has {} values but the task has {} variables",
            self.initial.len(),
            self.variables.len()
        );
        for (index, variable) in self.variables.iter().enumerate() {
            ensure!(variable.size > 0, "variable {index} has an empty domain");
            ensure!(
                self.initial[index] < variable.size,
                "initial value {} of variable {index} is outside its domain of size {}",
                self.initial[index],
                variable.size
            );
        }
        self.check_conditions("goal", &self.goal)?;
        for (index, operator) in self.operators.iter().enumerate() {
            ensure!(operator.cost >= 0.0, "operator {index} has a negative cost");
            ensure!(
                !operator.effects.is_empty(),
                "operator {index} has no effects"
            );
            self.check_conditions(&operator.name, &operator.precondition)?;
            for effect in &operator.effects {
                ensure!(
                    !self.variables[effect.variable].derived,
                    "operator {} affects the derived variable {}",
                    operator.name,
                    effect.variable
                );
                ensure!(
                    effect.value < self.variables[effect.variable].size,
                    "operator {} sets variable {} to a value outside its domain",
                    operator.name,
                    effect.variable
                );
                self.check_conditions(&operator.name, &effect.condition)?;
            }
        }
        for (index, axiom) in self.axioms.iter().enumerate() {
            ensure!(
                self.variables[axiom.variable].derived,
                "axiom {index} defines the fluent variable {}",
                axiom.variable
            );
            ensure!(
                axiom.value < self.variables[axiom.variable].size,
                "axiom {index} derives a value outside the domain of variable {}",
                axiom.variable
            );
            self.check_conditions(&format!("axiom {index}"), &axiom.condition)?;
        }
        self.check_axiom_layering()?;
        Ok(())
    }

    fn check_conditions(&self, owner: &str, conditions: &Conditions) -> Result<()> {
        let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
        for &(variable, value) in conditions {
            ensure!(
                variable < self.variables.len(),
                "{owner} refers to the unknown variable {variable}"
            );
            ensure!(
                value < self.variables[variable].size,
                "{owner} assigns the value {value} outside the domain of variable {variable}"
            );
            if let Some(&previous) = seen.get(&variable) {
                ensure!(
                    previous == value,
                    "{owner} is inconsistent: variable {variable} is required to be both {previous} and {value}"
                );
            }
            seen.insert(variable, value);
        }
        Ok(())
    }

    /// The layering property from Definition 1: within one layer, a derived
    /// variable may only be associated with a single value in axiom heads and
    /// bodies. Bodies may only refer to derivations from the same or an
    /// earlier layer.
    fn check_axiom_layering(&self) -> Result<()> {
        let mut layers: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
        let mut latest_definition: BTreeMap<usize, usize> = BTreeMap::new();
        for axiom in &self.axioms {
            let heads = layers.entry(axiom.layer).or_default();
            if let Some(&value) = heads.get(&axiom.variable) {
                ensure!(
                    value == axiom.value,
                    "axiom layer {} derives two values for variable {}",
                    axiom.layer,
                    axiom.variable
                );
            }
            heads.insert(axiom.variable, axiom.value);
            let entry = latest_definition.entry(axiom.variable).or_insert(0);
            *entry = (*entry).max(axiom.layer);
        }
        for axiom in &self.axioms {
            let heads = &layers[&axiom.layer];
            for &(variable, value) in &axiom.condition {
                if let Some(&derived) = heads.get(&variable) {
                    ensure!(
                        derived == value,
                        "axiom layer {} uses variable {variable} both with value {derived} and value {value}",
                        axiom.layer
                    );
                }
                if self.variables[variable].derived {
                    ensure!(
                        latest_definition.get(&variable).copied().unwrap_or(0) <= axiom.layer,
                        "axiom in layer {} depends on a later derivation of variable {variable}",
                        axiom.layer
                    );
                }
            }
        }
        Ok(())
    }

    /// Evaluate all axioms in the extended state `state`, in place (Fig. 16).
    pub fn evaluate_axioms(&self, state: &mut State) {
        // Leaf tasks compile without axioms; this early return keeps the
        // per-successor call free of the layer bookkeeping and the full
        // derived-variable scan below.
        if self.axioms.is_empty() {
            return;
        }
        let mut layers: BTreeSet<usize> = BTreeSet::new();
        for axiom in &self.axioms {
            layers.insert(axiom.layer);
        }
        for (variable, definition) in self.variables.iter().enumerate() {
            if definition.derived {
                state[variable] = UNDEFINED;
            }
        }
        for layer in layers {
            let indices: Vec<usize> = (0..self.axioms.len())
                .filter(|&index| self.axioms[index].layer == layer)
                .collect();
            if indices.is_empty() {
                continue;
            }
            // Number of conditions of each axiom that are not yet satisfied.
            let mut remaining: Vec<usize> = indices
                .iter()
                .map(|&index| self.axioms[index].condition.len())
                .collect();
            let mut queue: VecDeque<usize> = VecDeque::new();
            for (position, &index) in indices.iter().enumerate() {
                let axiom = &self.axioms[index];
                let satisfied = axiom
                    .condition
                    .iter()
                    .filter(|&&(variable, value)| state[variable] == value)
                    .count();
                remaining[position] = axiom.condition.len() - satisfied;
                if remaining[position] == 0 {
                    queue.push_back(position);
                }
            }
            while let Some(position) = queue.pop_front() {
                let axiom = &self.axioms[indices[position]];
                if state[axiom.variable] == axiom.value {
                    continue;
                }
                state[axiom.variable] = axiom.value;
                for (other, &index) in indices.iter().enumerate() {
                    if remaining[other] == 0 {
                        continue;
                    }
                    let candidate = &self.axioms[index];
                    if candidate.condition.iter().any(|&(variable, value)| {
                        variable == axiom.variable && value == axiom.value
                    }) {
                        remaining[other] -= 1;
                        if remaining[other] == 0 {
                            queue.push_back(other);
                        }
                    }
                }
            }
        }
    }

    /// The initial state with derived variables evaluated.
    pub fn initial_state(&self) -> State {
        let mut state = self.initial.clone();
        self.evaluate_axioms(&mut state);
        state
    }

    /// Whether `state` satisfies the goal.
    pub fn is_goal(&self, state: &State) -> bool {
        self.conditions_hold(&self.goal, state)
    }

    /// Apply `operator` in `state` (Definition 3). Returns `None` if the
    /// precondition does not hold.
    pub fn apply(&self, operator: usize, state: &State) -> Option<State> {
        let operator = &self.operators[operator];
        if !self.conditions_hold(&operator.precondition, state) {
            return None;
        }
        let mut next = state.clone();
        for effect in &operator.effects {
            if self.conditions_hold(&effect.condition, state) {
                next[effect.variable] = effect.value;
            }
        }
        self.evaluate_axioms(&mut next);
        Some(next)
    }

    /// Execute a plan and return the final state, checking applicability.
    pub fn execute(&self, plan: &[usize]) -> Result<State> {
        let mut state = self.initial_state();
        for (step, &operator) in plan.iter().enumerate() {
            ensure!(
                operator < self.operators.len(),
                "step {step} uses unknown operator {operator}"
            );
            state = match self.apply(operator, &state) {
                Some(next) => next,
                None => bail!(
                    "step {step} ({}) is not applicable",
                    self.operators[operator].name
                ),
            };
        }
        Ok(state)
    }
}
