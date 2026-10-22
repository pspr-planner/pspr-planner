//! Distance matrices and evaluation DAGs used by the agglomeration.

use anyhow::{Result, ensure};

/// A symmetric matrix of pairwise distances between instances.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceMatrix {
    size: usize,
    entries: Vec<f64>,
}

impl DistanceMatrix {
    /// Build a distance matrix from a square, symmetric, non-negative table
    /// with a zero diagonal.
    pub fn new(rows: &[Vec<f64>]) -> Result<Self> {
        let size = rows.len();
        ensure!(size >= 1, "at least one instance is required");
        for (index, row) in rows.iter().enumerate() {
            ensure!(
                row.len() == size,
                "row {index} has {} entries, expected {size}",
                row.len()
            );
            ensure!(
                row[index] == 0.0,
                "the distance from instance {index} to itself must be 0"
            );
            for (other, &value) in row.iter().enumerate() {
                ensure!(
                    value.is_finite(),
                    "the distance ({index}, {other}) is not finite"
                );
                ensure!(value >= 0.0, "the distance ({index}, {other}) is negative");
                ensure!(
                    (value - rows[other][index]).abs() <= 1e-9,
                    "the distance matrix is not symmetric at ({index}, {other})"
                );
            }
        }
        Ok(DistanceMatrix {
            size,
            entries: rows.iter().flatten().copied().collect(),
        })
    }

    /// The number of instances.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Always `false`: a distance matrix contains at least one instance.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The distance between instances `a` and `b`.
    pub fn get(&self, a: usize, b: usize) -> f64 {
        self.entries[a * self.size + b]
    }
}

/// Additional precedence constraints between internal evaluation nodes.
///
/// Node `0` is the first internal node created when the dendrogram is built
/// (see [`super::Dendrogram`]), and there are `n - 1` such nodes for `n`
/// instances.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvaluationDag {
    pub edges: Vec<(usize, usize)>,
}

impl EvaluationDag {
    pub fn new(edges: Vec<(usize, usize)>) -> Self {
        Self { edges }
    }
}
