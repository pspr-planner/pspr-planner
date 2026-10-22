//! Penalty spaces and smooth surrogates of the relaxed SGPlan objective.

use std::fmt;

/// The penalty function applied to a normalized violation.
///
/// The merger runs a continuation over these spaces: a linear penalty is
/// well-conditioned but never vanishes, a quadratic penalty vanishes at the
/// boundary and becomes stiff for large weights, and an exponential penalty
/// gives the strongest push near feasibility. Escalating only when a round
/// stalls keeps the early iterations cheap and the late ones accurate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum PenaltySpace {
    /// `phi(v) = v`.
    #[default]
    Linear,
    /// `phi(v) = v^2`.
    Quadratic,
    /// `phi(v) = (exp(4 v) - 1) / 4`.
    Exponential,
}

/// Steepness of [`PenaltySpace::Exponential`].
const EXPONENTIAL_SCALE: f64 = 4.0;

impl PenaltySpace {
    /// Name of the space, used in diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            PenaltySpace::Linear => "linear",
            PenaltySpace::Quadratic => "quadratic",
            PenaltySpace::Exponential => "exponential",
        }
    }

    /// `phi(v)` for a non-negative violation `v`.
    pub(super) fn value(self, violation: f64) -> f64 {
        match self {
            PenaltySpace::Linear => violation,
            PenaltySpace::Quadratic => violation * violation + violation,
            PenaltySpace::Exponential => {
                (EXPONENTIAL_SCALE * violation).exp_m1() / EXPONENTIAL_SCALE
            }
        }
    }

    /// `d phi / d v`.
    pub(super) fn slope(self, violation: f64) -> f64 {
        match self {
            PenaltySpace::Linear => 1.0,
            PenaltySpace::Quadratic => 2.0 * violation + 1.0,
            PenaltySpace::Exponential => (EXPONENTIAL_SCALE * violation).exp(),
        }
    }
}

impl fmt::Display for PenaltySpace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The logistic function `1 / (1 + exp(-value))`, computed without overflow.
pub(super) fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// The softplus function `ln(1 + exp(value))`, computed without overflow.
pub(super) fn softplus(value: f64) -> f64 {
    if value > 30.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// A log-sum-exp smooth maximum.
///
/// Returns the smooth maximum of `values` together with the softmax weights,
/// which are the gradient of the maximum with respect to the inputs.
pub(super) fn smooth_max(values: &[f64], temperature: f64) -> (f64, Vec<f64>) {
    if values.is_empty() {
        return (0.0, Vec::new());
    }
    let temperature = temperature.max(f64::MIN_POSITIVE);
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = values
        .iter()
        .map(|value| ((value - largest) / temperature).exp())
        .collect();
    let sum: f64 = scaled.iter().sum();
    let value = largest + temperature * sum.ln();
    let weights = scaled.iter().map(|weight| weight / sum).collect();
    (value, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penalty_slopes_match_finite_differences() {
        for space in [
            PenaltySpace::Linear,
            PenaltySpace::Quadratic,
            PenaltySpace::Exponential,
        ] {
            for &violation in &[0.0_f64, 0.05, 0.5, 1.0, 2.0] {
                let step = 1e-6;
                let numeric =
                    (space.value(violation + step) - space.value(violation - step)) / (2.0 * step);
                let analytic = space.slope(violation);
                assert!(
                    (numeric - analytic).abs() < 1e-4 * (1.0 + analytic.abs()),
                    "{space:?} slope at {violation}: numeric {numeric}, analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn smooth_max_dominates_the_maximum_and_normalizes() {
        let (value, weights) = smooth_max(&[0.0, 1.0, 2.0], 0.5);
        assert!(value >= 2.0, "the smooth maximum over-estimates: {value}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights[2] > weights[1] && weights[1] > weights[0]);
    }

    #[test]
    fn smooth_max_tends_to_the_maximum() {
        let (value, weights) = smooth_max(&[0.0, 5.0], 1e-3);
        assert!((value - 5.0).abs() < 1e-2, "value = {value}");
        assert!(weights[1] > 0.999);
    }

    #[test]
    fn logistic_and_softplus_are_stable() {
        assert_eq!(logistic(1e6), 1.0);
        assert_eq!(logistic(-1e6), 0.0);
        assert!((logistic(0.0) - 0.5).abs() < 1e-12);
        assert!(softplus(-1e6).is_finite());
        assert!((softplus(1e6) - 1e6).abs() < 1e-9);
        assert!((softplus(0.0) - 2.0_f64.ln()).abs() < 1e-12);
    }
}
