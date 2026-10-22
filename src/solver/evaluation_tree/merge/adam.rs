//! A small deterministic Adam optimizer for the relaxed objective.

/// Hyper-parameters of Adam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamOptions {
    /// Step size.
    pub learning_rate: f64,
    /// Exponential decay of the first moment.
    pub beta1: f64,
    /// Exponential decay of the second moment.
    pub beta2: f64,
    /// Regularizer of the normalized step.
    pub epsilon: f64,
}

impl Default for AdamOptions {
    fn default() -> Self {
        AdamOptions {
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam state: the first and second moment of every coordinate.
#[derive(Debug, Clone)]
pub(super) struct Adam {
    first: Vec<f64>,
    second: Vec<f64>,
    steps: u64,
}

impl Adam {
    pub(super) fn new(dimension: usize) -> Self {
        Adam {
            first: vec![0.0; dimension],
            second: vec![0.0; dimension],
            steps: 0,
        }
    }

    /// Forget the moments, e.g. when the penalty space changes.
    pub(super) fn reset(&mut self) {
        self.first.fill(0.0);
        self.second.fill(0.0);
        self.steps = 0;
    }

    /// Apply one Adam step to `parameters` and project onto `[lower, upper]`.
    pub(super) fn update(
        &mut self,
        parameters: &mut [f64],
        gradient: &[f64],
        options: &AdamOptions,
        lower: f64,
        upper: f64,
    ) {
        self.steps += 1;
        let exponent = self.steps.min(1_000_000) as i32;
        let bias1 = 1.0 - options.beta1.powi(exponent);
        let bias2 = 1.0 - options.beta2.powi(exponent);
        for (index, parameter) in parameters.iter_mut().enumerate() {
            let derivative = gradient.get(index).copied().unwrap_or(0.0);
            self.first[index] =
                options.beta1 * self.first[index] + (1.0 - options.beta1) * derivative;
            self.second[index] = options.beta2 * self.second[index]
                + (1.0 - options.beta2) * derivative * derivative;
            let moment = self.first[index] / bias1;
            let variance = self.second[index] / bias2;
            let step = options.learning_rate * moment / (variance.sqrt() + options.epsilon);
            *parameter = (*parameter - step).clamp(lower, upper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_descends_a_quadratic() {
        let options = AdamOptions {
            learning_rate: 0.1,
            ..AdamOptions::default()
        };
        let mut parameters = vec![4.0, -3.0];
        let mut adam = Adam::new(2);
        for _ in 0..1_000 {
            let gradient = vec![2.0 * parameters[0], 2.0 * parameters[1]];
            adam.update(&mut parameters, &gradient, &options, -10.0, 10.0);
        }
        assert!(parameters[0].abs() < 1e-3, "x = {}", parameters[0]);
        assert!(parameters[1].abs() < 1e-3, "y = {}", parameters[1]);
    }

    #[test]
    fn adam_projects_onto_the_bounds() {
        let options = AdamOptions {
            learning_rate: 1.0,
            ..AdamOptions::default()
        };
        let mut parameters = vec![0.5];
        let mut adam = Adam::new(1);
        for _ in 0..50 {
            adam.update(&mut parameters, &[-5.0], &options, 0.0, 1.0);
        }
        assert_eq!(parameters, vec![1.0]);
    }

    #[test]
    fn reset_clears_the_moments() {
        let options = AdamOptions {
            learning_rate: 0.1,
            ..AdamOptions::default()
        };
        let mut adam = Adam::new(1);
        let mut parameters = vec![0.5];
        for _ in 0..100 {
            adam.update(&mut parameters, &[100.0], &options, 0.0, 1.0);
        }
        adam.reset();
        parameters[0] = 0.5;
        adam.update(&mut parameters, &[1.0], &options, 0.0, 1.0);
        assert!(
            (parameters[0] - 0.4).abs() < 1e-6,
            "after a reset Adam moves like a fresh optimizer: {}",
            parameters[0]
        );
    }
}
