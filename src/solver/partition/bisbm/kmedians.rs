//! Deterministic `k`-medians over the spectral embedding.

use nalgebra::base::DMatrix;

/// Number of `k`-medians restarts; the run with the lowest objective wins.
const K_MEDIANS_RESTARTS: usize = 8;
/// Maximum number of Lloyd iterations per restart.
const K_MEDIANS_MAX_ITERATIONS: usize = 100;
/// Fixed seed so that clustering a graph is reproducible.
const K_MEDIANS_SEED: u64 = 0x5053_5052_2d42_4953;

/// Cluster the rows of `embedding` into `k` groups with Lloyd's `k`-medians.
///
/// Rows are assigned to their nearest centre in the Euclidean norm and centres
/// are updated to the coordinate-wise median. Several deterministic restarts
/// are performed and the run with the lowest objective is returned.
pub(super) fn k_medians(embedding: &DMatrix<f64>, k: usize) -> Vec<usize> {
    let n = embedding.nrows();
    if n == 0 {
        return Vec::new();
    }
    if k <= 1 {
        return vec![0; n];
    }
    if k >= n {
        return (0..n).collect();
    }

    let mut rng = SplitMix64::new(K_MEDIANS_SEED);
    let mut best_labels = vec![0; n];
    let mut best_cost = f64::INFINITY;
    for _ in 0..K_MEDIANS_RESTARTS {
        let mut centers = initial_centers(embedding, k, &mut rng);
        let (labels, cost) = lloyd(embedding, &mut centers);
        if cost < best_cost {
            best_cost = cost;
            best_labels = labels;
        }
    }
    best_labels
}

/// Pick `k` initial centres with `k`-medians++ (`D^2` sampling).
fn initial_centers(embedding: &DMatrix<f64>, k: usize, rng: &mut SplitMix64) -> DMatrix<f64> {
    let n = embedding.nrows();
    let dimension = embedding.ncols();
    let mut centers = DMatrix::zeros(k, dimension);
    let mut closest = vec![f64::INFINITY; n];

    let mut chosen = rng.next_index(n);
    for center in 0..k {
        if center > 0 {
            let total: f64 = closest.iter().sum();
            if total > 0.0 && total.is_finite() {
                let target = rng.next_unit() * total;
                let mut cumulative = 0.0;
                chosen = n - 1;
                for (index, &weight) in closest.iter().enumerate() {
                    cumulative += weight;
                    if cumulative >= target {
                        chosen = index;
                        break;
                    }
                }
            } else {
                chosen = center % n;
            }
        }
        for column in 0..dimension {
            centers[(center, column)] = embedding[(chosen, column)];
        }
        for (row, closest_distance) in closest.iter_mut().enumerate() {
            let distance = squared_distance(embedding, row, &centers, center);
            if distance < *closest_distance {
                *closest_distance = distance;
            }
        }
    }
    centers
}

/// Lloyd iterations for `k`-medians; returns the labels and the objective.
fn lloyd(embedding: &DMatrix<f64>, centers: &mut DMatrix<f64>) -> (Vec<usize>, f64) {
    let n = embedding.nrows();
    let k = centers.nrows();
    let dimension = embedding.ncols();

    let mut labels = vec![usize::MAX; n];
    let mut previous = vec![usize::MAX; n];
    let mut distances = vec![0.0; n];

    for _ in 0..K_MEDIANS_MAX_ITERATIONS {
        for row in 0..n {
            let mut best = 0;
            let mut best_distance = f64::INFINITY;
            for center in 0..k {
                let distance = squared_distance(embedding, row, centers, center);
                if distance < best_distance {
                    best_distance = distance;
                    best = center;
                }
            }
            labels[row] = best;
            distances[row] = best_distance;
        }

        // An empty cluster leaves its centre undefined; move the worst
        // represented point of a non-singleton cluster into it. Since `k < n`,
        // such a point always exists.
        let mut counts = vec![0usize; k];
        for &label in &labels {
            counts[label] += 1;
        }
        for center in 0..k {
            if counts[center] > 0 {
                continue;
            }
            let mut worst = usize::MAX;
            let mut worst_distance = f64::NEG_INFINITY;
            for row in 0..n {
                if counts[labels[row]] > 1 && distances[row] > worst_distance {
                    worst_distance = distances[row];
                    worst = row;
                }
            }
            if worst != usize::MAX {
                counts[labels[worst]] -= 1;
                labels[worst] = center;
                counts[center] = 1;
            }
        }

        if labels == previous {
            break;
        }
        previous.copy_from_slice(&labels);

        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (row, &label) in labels.iter().enumerate() {
            members[label].push(row);
        }
        let mut column_values = Vec::with_capacity(n);
        for center in 0..k {
            if members[center].is_empty() {
                continue;
            }
            for column in 0..dimension {
                column_values.clear();
                column_values.extend(members[center].iter().map(|&row| embedding[(row, column)]));
                column_values.sort_by(f64::total_cmp);
                let middle = column_values.len() / 2;
                centers[(center, column)] = if column_values.len() % 2 == 0 {
                    0.5 * (column_values[middle - 1] + column_values[middle])
                } else {
                    column_values[middle]
                };
            }
        }
    }

    let cost: f64 = (0..n)
        .map(|row| squared_distance(embedding, row, centers, labels[row]))
        .sum();
    (labels, cost)
}

/// Squared Euclidean distance between a row and a centre.
fn squared_distance(rows: &DMatrix<f64>, row: usize, centers: &DMatrix<f64>, center: usize) -> f64 {
    let mut sum = 0.0;
    for column in 0..rows.ncols() {
        let delta = rows[(row, column)] - centers[(center, column)];
        sum += delta * delta;
    }
    sum
}

/// Small deterministic `SplitMix64` generator.
///
/// The clustering must be reproducible across runs, so no external RNG is used.
pub(super) struct SplitMix64(u64);

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform value in `[0, 1)`.
    pub(super) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform index in `[0, bound)`.
    fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}
