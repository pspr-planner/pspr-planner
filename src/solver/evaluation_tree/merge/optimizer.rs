//! Optimization backends for the relaxed merge objective.
//!
//! Adam is the default because it tolerates penalty weights that span several
//! orders of magnitude. The comparison harness can additionally select
//! projected first-order methods, nonlinear conjugate gradients, and
//! quasi-Newton methods. All methods operate on the same bounded vector of
//! child start positions and use the same continuation strategy.

use std::collections::VecDeque;
use std::fmt;

use super::adam::{Adam, AdamOptions};

/// A function returning the objective and gradient at a candidate point.
///
/// The merge only calls it from the line search of the quasi-Newton methods;
/// the Adam inner loop evaluates the objective directly into a reused buffer.
type Evaluate<'a> = dyn FnMut(&[f64]) -> (f64, Vec<f64>) + 'a;

/// A selectable optimization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizerKind {
    /// Adaptive moment estimation.
    #[default]
    Adam,
    /// Plain projected gradient descent.
    Sgd,
    /// Heavy-ball momentum.
    Momentum,
    /// Nesterov accelerated gradient.
    Nesterov,
    /// Root-mean-square propagation.
    RmsProp,
    /// Accumulated squared-gradient scaling.
    Adagrad,
    /// Limited-memory BFGS with projected backtracking line search.
    Lbfgs,
    /// Full BFGS with projected backtracking line search.
    Bfgs,
    /// Polak-Ribiere nonlinear conjugate gradient.
    ConjugateGradient,
}

impl OptimizerKind {
    /// Every backend exposed by the comparison harness.
    pub const ALL: [Self; 9] = [
        Self::Adam,
        Self::Sgd,
        Self::Momentum,
        Self::Nesterov,
        Self::RmsProp,
        Self::Adagrad,
        Self::Lbfgs,
        Self::Bfgs,
        Self::ConjugateGradient,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::Sgd => "sgd",
            Self::Momentum => "momentum",
            Self::Nesterov => "nesterov",
            Self::RmsProp => "rmsprop",
            Self::Adagrad => "adagrad",
            Self::Lbfgs => "lbfgs",
            Self::Bfgs => "bfgs",
            Self::ConjugateGradient => "cg",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "adam" => Some(Self::Adam),
            "sgd" | "gd" => Some(Self::Sgd),
            "momentum" => Some(Self::Momentum),
            "nesterov" | "nag" => Some(Self::Nesterov),
            "rmsprop" => Some(Self::RmsProp),
            "adagrad" => Some(Self::Adagrad),
            "lbfgs" | "l-bfgs" => Some(Self::Lbfgs),
            "bfgs" => Some(Self::Bfgs),
            "cg" | "conjugate-gradient" | "conjugate_gradient" => Some(Self::ConjugateGradient),
            _ => None,
        }
    }
}

impl fmt::Display for OptimizerKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// State of one selected optimizer.
#[derive(Debug, Clone)]
pub(super) struct OptimizerState {
    kind: OptimizerKind,
    dimension: usize,
    adam: Adam,
    velocity: Vec<f64>,
    second: Vec<f64>,
    /// Inverse Hessian approximation of full BFGS.
    hessian: Vec<f64>,
    previous_gradient: Option<Vec<f64>>,
    previous_direction: Option<Vec<f64>>,
    history: VecDeque<HistoryEntry>,
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    parameters: Vec<f64>,
    gradient: Vec<f64>,
    rho: f64,
}

impl OptimizerState {
    pub(super) fn new(kind: OptimizerKind, dimension: usize) -> Self {
        let mut state = Self {
            kind,
            dimension,
            adam: Adam::new(dimension),
            velocity: vec![0.0; dimension],
            second: vec![0.0; dimension],
            hessian: vec![0.0; dimension * dimension],
            previous_gradient: None,
            previous_direction: None,
            history: VecDeque::new(),
        };
        state.set_identity_hessian();
        state
    }

    pub(super) fn reset(&mut self) {
        self.adam.reset();
        self.velocity.fill(0.0);
        self.second.fill(0.0);
        self.set_identity_hessian();
        self.previous_gradient = None;
        self.previous_direction = None;
        self.history.clear();
    }

    fn set_identity_hessian(&mut self) {
        self.hessian.fill(0.0);
        for index in 0..self.dimension {
            self.hessian[index * self.dimension + index] = 1.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        parameters: &mut [f64],
        gradient: &[f64],
        objective: f64,
        evaluate: &mut Evaluate<'_>,
        options: &AdamOptions,
        lower: f64,
        upper: f64,
    ) {
        match self.kind {
            OptimizerKind::Adam => {
                self.adam
                    .update(parameters, gradient, options, lower, upper);
            }
            OptimizerKind::Sgd => {
                let rate = options.learning_rate;
                for (value, derivative) in parameters.iter_mut().zip(gradient) {
                    *value = (*value - rate * derivative).clamp(lower, upper);
                }
            }
            OptimizerKind::Momentum => {
                let rate = options.learning_rate;
                let beta = options.beta1;
                for (index, value) in parameters.iter_mut().enumerate() {
                    self.velocity[index] = beta * self.velocity[index] + gradient[index];
                    *value = (*value - rate * self.velocity[index]).clamp(lower, upper);
                }
            }
            OptimizerKind::Nesterov => {
                let rate = options.learning_rate;
                let beta = options.beta1;
                for (index, value) in parameters.iter_mut().enumerate() {
                    self.velocity[index] = beta * self.velocity[index] + gradient[index];
                    let step = beta * self.velocity[index] + gradient[index];
                    *value = (*value - rate * step).clamp(lower, upper);
                }
            }
            OptimizerKind::RmsProp => {
                let rate = options.learning_rate;
                let decay = options.beta2;
                for (index, value) in parameters.iter_mut().enumerate() {
                    self.second[index] = decay * self.second[index]
                        + (1.0 - decay) * gradient[index] * gradient[index];
                    let step =
                        rate * gradient[index] / (self.second[index].sqrt() + options.epsilon);
                    *value = (*value - step).clamp(lower, upper);
                }
            }
            OptimizerKind::Adagrad => {
                let rate = options.learning_rate;
                for (index, value) in parameters.iter_mut().enumerate() {
                    self.second[index] += gradient[index] * gradient[index];
                    let step =
                        rate * gradient[index] / (self.second[index].sqrt() + options.epsilon);
                    *value = (*value - step).clamp(lower, upper);
                }
            }
            OptimizerKind::Bfgs | OptimizerKind::Lbfgs | OptimizerKind::ConjugateGradient => {
                self.update_second_order(parameters, gradient, objective, evaluate, lower, upper);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_second_order(
        &mut self,
        parameters: &mut [f64],
        gradient: &[f64],
        objective: f64,
        evaluate: &mut Evaluate<'_>,
        lower: f64,
        upper: f64,
    ) {
        let old_parameters = parameters.to_vec();
        let old_gradient = gradient.to_vec();
        let mut direction = match self.kind {
            OptimizerKind::Bfgs => self.bfgs_direction(gradient),
            OptimizerKind::Lbfgs => self.lbfgs_direction(gradient),
            OptimizerKind::ConjugateGradient => self.conjugate_direction(gradient),
            _ => unreachable!(),
        };
        if direction.iter().any(|value| !value.is_finite()) || dot(&direction, gradient) >= -1e-12 {
            direction = gradient.iter().map(|value| -value).collect();
        }

        let accepted = line_search(
            parameters, gradient, objective, &direction, evaluate, lower, upper, 1.0,
        );
        parameters.copy_from_slice(&accepted.parameters);

        let step = subtract(&accepted.parameters, &old_parameters);
        let change = subtract(&accepted.gradient, &old_gradient);
        let curvature = dot(&step, &change);
        let step_norm = norm(&step);
        let change_norm = norm(&change);
        if curvature > 1e-12 * step_norm.max(1e-12) * change_norm.max(1e-12) {
            match self.kind {
                OptimizerKind::Bfgs => self.bfgs_update(&step, &change),
                OptimizerKind::Lbfgs => {
                    self.history.push_back(HistoryEntry {
                        parameters: step,
                        gradient: change,
                        rho: 1.0 / curvature,
                    });
                    if self.history.len() > 10 {
                        self.history.pop_front();
                    }
                }
                _ => {}
            }
        } else {
            self.history.clear();
            self.set_identity_hessian();
        }
        if self.kind == OptimizerKind::ConjugateGradient {
            self.previous_direction = Some(direction);
        }
        self.previous_gradient = Some(accepted.gradient);
    }

    fn bfgs_direction(&self, gradient: &[f64]) -> Vec<f64> {
        let n = self.dimension;
        let mut direction = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                direction[i] -= self.hessian[i * n + j] * gradient[j];
            }
        }
        direction
    }

    fn lbfgs_direction(&self, gradient: &[f64]) -> Vec<f64> {
        if self.history.is_empty() {
            return gradient.iter().map(|value| -value).collect();
        }
        let n = self.dimension;
        let mut q = gradient.to_vec();
        let mut alpha = vec![0.0; self.history.len()];
        for (index, entry) in self.history.iter().enumerate().rev() {
            alpha[index] = entry.rho * dot(&entry.parameters, &q);
            for i in 0..n {
                q[i] -= alpha[index] * entry.gradient[i];
            }
        }
        let last = self.history.back().expect("history is non-empty");
        let scale = dot(&last.parameters, &last.gradient)
            / last
                .gradient
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .max(1e-12);
        for value in &mut q {
            *value *= scale.max(1e-6);
        }
        for (index, entry) in self.history.iter().enumerate() {
            let beta = entry.rho * dot(&entry.gradient, &q);
            for i in 0..n {
                q[i] += entry.parameters[i] * (alpha[index] - beta);
            }
        }
        q.iter().map(|value| -value).collect()
    }

    fn conjugate_direction(&self, gradient: &[f64]) -> Vec<f64> {
        let Some(previous_gradient) = &self.previous_gradient else {
            return gradient.iter().map(|value| -value).collect();
        };
        let denominator = dot(previous_gradient, previous_gradient).max(1e-12);
        let difference = subtract(gradient, previous_gradient);
        let beta = (dot(gradient, &difference) / denominator).max(0.0);
        let previous_direction = self
            .previous_direction
            .as_ref()
            .cloned()
            .unwrap_or_else(|| gradient.iter().map(|value| -value).collect());
        gradient
            .iter()
            .zip(previous_direction)
            .map(|(gradient, direction)| -gradient + beta * direction)
            .collect()
    }

    fn bfgs_update(&mut self, step: &[f64], change: &[f64]) {
        let n = self.dimension;
        let rho = 1.0 / dot(step, change);
        let mut hy = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                hy[i] += self.hessian[i * n + j] * change[j];
            }
        }
        let yhy = dot(change, &hy);
        for i in 0..n {
            for j in 0..n {
                self.hessian[i * n + j] += rho * (1.0 + yhy * rho) * step[i] * step[j]
                    - rho * (hy[i] * step[j] + step[i] * hy[j]);
            }
        }
    }
}

struct AcceptedStep {
    parameters: Vec<f64>,
    gradient: Vec<f64>,
    objective: f64,
}

#[allow(clippy::too_many_arguments)]
fn line_search(
    parameters: &[f64],
    gradient: &[f64],
    objective: f64,
    direction: &[f64],
    evaluate: &mut Evaluate<'_>,
    lower: f64,
    upper: f64,
    initial_step: f64,
) -> AcceptedStep {
    let mut step = initial_step.max(1e-9);
    let mut best = AcceptedStep {
        parameters: parameters.to_vec(),
        gradient: gradient.to_vec(),
        objective,
    };
    for _ in 0..24 {
        let candidate: Vec<f64> = parameters
            .iter()
            .zip(direction)
            .map(|(parameter, direction)| (*parameter + step * direction).clamp(lower, upper))
            .collect();
        let (candidate_objective, candidate_gradient) = evaluate(&candidate);
        if candidate_objective.is_finite() {
            let delta = subtract(&candidate, parameters);
            let decrease = -dot(gradient, &delta);
            if candidate_objective <= objective - 1e-4 * decrease.max(0.0) {
                return AcceptedStep {
                    parameters: candidate,
                    gradient: candidate_gradient,
                    objective: candidate_objective,
                };
            }
            if candidate_objective < best.objective {
                best = AcceptedStep {
                    parameters: candidate,
                    gradient: candidate_gradient,
                    objective: candidate_objective,
                };
            }
        }
        step *= 0.5;
    }
    best
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quasi_newton_methods_descend_a_quadratic() {
        for kind in [
            OptimizerKind::Lbfgs,
            OptimizerKind::Bfgs,
            OptimizerKind::ConjugateGradient,
        ] {
            let mut parameters = vec![4.0, -3.0];
            let mut optimizer = OptimizerState::new(kind, 2);
            let options = AdamOptions::default();
            for _ in 0..100 {
                let gradient = vec![2.0 * parameters[0], 2.0 * parameters[1]];
                let objective = parameters[0] * parameters[0] + parameters[1] * parameters[1];
                optimizer.update(
                    &mut parameters,
                    &gradient,
                    objective,
                    &mut |values| {
                        (
                            values[0] * values[0] + values[1] * values[1],
                            vec![2.0 * values[0], 2.0 * values[1]],
                        )
                    },
                    &options,
                    -10.0,
                    10.0,
                );
            }
            assert!(
                parameters[0].abs() < 1e-3 && parameters[1].abs() < 1e-3,
                "{kind} left {parameters:?}"
            );
        }
    }

    #[test]
    fn first_order_methods_descend_a_quadratic() {
        for kind in [
            OptimizerKind::Sgd,
            OptimizerKind::Momentum,
            OptimizerKind::Nesterov,
            OptimizerKind::RmsProp,
            OptimizerKind::Adagrad,
        ] {
            let mut parameters = vec![1.0, -1.0];
            let mut optimizer = OptimizerState::new(kind, 2);
            let options = AdamOptions {
                learning_rate: 0.05,
                ..AdamOptions::default()
            };
            for _ in 0..1_000 {
                let gradient = vec![2.0 * parameters[0], 2.0 * parameters[1]];
                optimizer.update(
                    &mut parameters,
                    &gradient,
                    0.0,
                    &mut |_| (0.0, Vec::new()),
                    &options,
                    -10.0,
                    10.0,
                );
            }
            assert!(
                parameters[0].abs() < 1e-2 && parameters[1].abs() < 1e-2,
                "{kind} left {parameters:?}"
            );
        }
    }
}
