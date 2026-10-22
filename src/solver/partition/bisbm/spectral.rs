//! Spectral embeddings used by the BiSBM clusterer.

use nalgebra::base::{DMatrix, DVector};
use nalgebra::linalg::SymmetricEigen;

/// `Spec`/`AdaSpec`: embedding on the hollowed Gram matrix `H(A A^T)`.
///
/// Returns the `n1 x r` matrix of top-`r` eigenvectors and the estimated number
/// of communities, which is `r` itself.
pub(super) fn spec_embedding(adjacency: &DMatrix<f64>, p: f64) -> (DMatrix<f64>, usize) {
    let n = adjacency.nrows();

    let mut gram = adjacency * adjacency.transpose();
    for index in 0..n {
        gram[(index, index)] = 0.0;
    }

    let eigen = SymmetricEigen::new(gram);
    let order = descending_order(&eigen.eigenvalues);
    let eigenvalues: Vec<f64> = order
        .iter()
        .map(|&index| eigen.eigenvalues[index])
        .collect();

    // AdaSpec: largest rank whose eigenvalue gap exceeds
    // T = n1 * n2 * p^2 / log(log(n1)).
    let n1 = n as f64;
    let n2 = adjacency.ncols() as f64;
    let signal = n1 * n2 * p * p;
    let log_log = n1.ln().ln();
    let threshold = if log_log > 0.0 && log_log.is_finite() {
        signal / log_log
    } else {
        signal
    };

    let mut rank = 0;
    let mut largest_gap = (0, f64::NEG_INFINITY);
    for candidate in 1..n {
        let gap = eigenvalues[candidate - 1] - eigenvalues[candidate];
        if gap > threshold {
            rank = candidate;
        }
        if gap > largest_gap.1 {
            largest_gap = (candidate, gap);
        }
    }
    if rank == 0 {
        // Finite-sample fallback for the rank selection: keep the largest gap
        // unless the spectrum is numerically flat, in which case the group is
        // treated as a single community.
        let scale = eigenvalues.first().copied().unwrap_or(0.0).abs().max(1.0);
        if largest_gap.1 > 1e-9 * scale {
            rank = largest_gap.0;
        }
    }
    if rank == 0 {
        return (DMatrix::zeros(n, 0), 1);
    }

    let mut embedding = DMatrix::zeros(n, rank);
    for (column, &eigen_index) in order.iter().take(rank).enumerate() {
        for row in 0..n {
            embedding[(row, column)] = eigen.eigenvectors[(row, eigen_index)];
        }
    }
    (embedding, rank)
}

/// Spectral clustering on the dense normalized Laplacian of `H(A A^T)`.
///
/// Computes `L = I - D^(-1/2) S D^(-1/2)` for the similarity graph
/// `S = H(A A^T)`, embeds the type-I nodes with the eigenvectors belonging to
/// the smallest non-trivial eigenvalues and estimates the number of communities
/// from the largest gap in that part of the spectrum.
pub(super) fn laplacian_embedding(adjacency: &DMatrix<f64>) -> (DMatrix<f64>, usize) {
    let n = adjacency.nrows();

    let mut similarity = adjacency * adjacency.transpose();
    for index in 0..n {
        similarity[(index, index)] = 0.0;
    }

    let inv_sqrt_degree: Vec<f64> = (0..n)
        .map(|row| {
            let degree: f64 = (0..n).map(|column| similarity[(row, column)]).sum();
            if degree > 0.0 {
                1.0 / degree.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    let mut laplacian = DMatrix::zeros(n, n);
    for row in 0..n {
        for column in 0..n {
            laplacian[(row, column)] =
                -inv_sqrt_degree[row] * similarity[(row, column)] * inv_sqrt_degree[column];
        }
        laplacian[(row, row)] += 1.0;
    }

    let eigen = SymmetricEigen::new(laplacian);
    let order = ascending_order(&eigen.eigenvalues);
    let eigenvalues: Vec<f64> = order
        .iter()
        .map(|&index| eigen.eigenvalues[index])
        .collect();

    // Eigengap heuristic: for `k` communities the gap is `lambda_(k+1) -
    // lambda_k`, so the candidate itself is the number of communities. The
    // first eigenvalue is always zero, hence `candidate` starts at 1.
    let mut communities = 1;
    let mut best_gap = 0.0;
    for candidate in 1..n {
        let gap = eigenvalues[candidate] - eigenvalues[candidate - 1];
        if gap > best_gap {
            best_gap = gap;
            communities = candidate;
        }
    }
    if communities <= 1 {
        return (DMatrix::zeros(n, 0), 1);
    }

    let mut embedding = DMatrix::zeros(n, communities - 1);
    for (column, &eigen_index) in order.iter().skip(1).take(communities - 1).enumerate() {
        for row in 0..n {
            embedding[(row, column)] = eigen.eigenvectors[(row, eigen_index)];
        }
    }
    (embedding, communities)
}

/// Indices of `values` sorted from largest to smallest.
fn descending_order(values: &DVector<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    order
}

/// Indices of `values` sorted from smallest to largest.
fn ascending_order(values: &DVector<f64>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    order
}
