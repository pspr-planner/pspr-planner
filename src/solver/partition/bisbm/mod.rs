//! Spectral clustering for bipartite graphs under the bipartite stochastic
//! block model (BiSBM).
//!
//! Implements the spectral method of Braun, *Strong Consistency Guarantees for
//! Clustering High-Dimensional Bipartite Graphs with the Spectral Method*
//! (<https://arxiv.org/abs/2304.06903>).
//!
//! # The spectral method
//!
//! Let `A` be the `n1 x n2` adjacency matrix of the bipartite graph and
//! `B = H(A A^T)` its *hollowed* Gram matrix, i.e. `A A^T` with the diagonal
//! removed. Hollowing is essential: the diagonal of `A A^T` is centred around
//! `n2 * p_max`, whereas the population diagonal `(P P^T)_ii` is of order
//! `n2 * p_max^2`, so keeping it would dominate the spectrum.
//!
//! Algorithm 1 (`Spec`) computes the top-`r` eigenvectors `U` of `B` and
//! clusters the rows of `U` with `k`-medians. Algorithm 2 (`AdaSpec`) picks the
//! embedding rank adaptively as the largest rank whose eigenvalue gap exceeds
//! the threshold `T = n1 * n2 * p_max^2 / log(log(n1))`. Exact recovery of the
//! type-I partition is guaranteed in the regime `n1 * n2 * p_max^2 >~ log(n1)`
//! (Corollaries 1 and 2 of the paper).
//!
//! # What [`cluster`] does
//!
//! 1. The requested [`Grouping`]s partition the type-I nodes into groups that
//!    are clustered independently. Type-I nodes that occur in no grouping form
//!    their own singleton groups, and a node must not occur twice.
//! 2. For every group the induced subgraph on the group's type-I nodes and
//!    *their* type-II neighbours is extracted; type-II nodes that are not
//!    connected to the group are ignored.
//! 3. The sparsity level is estimated by the density `p` of the induced
//!    subgraph. When `n1 * n2 * p^2 >= log(n1)` the group is clustered with
//!    `Spec`/`AdaSpec`. Below that threshold the hollowed Gram matrix is too
//!    noisy, so the group is clustered from the eigenvectors of the symmetric
//!    normalized Laplacian `L = I - D^(-1/2) H(A A^T) D^(-1/2)` of the dense
//!    similarity graph instead.
//!
//! `k`-medians assigns rows to their nearest centre in the Euclidean norm and
//! updates centres with the coordinate-wise median, matching the `l2 ->
//! infinity` eigenvector perturbation bound the analysis relies on.
//!
//! # Number of communities
//!
//! The paper assumes that the number of communities `k` is known, but
//! [`cluster`] does not take it as an argument. It is estimated from the
//! spectrum: `AdaSpec`'s threshold rule selects the embedding rank `r` and
//! `k = r`, and in the Laplacian regime `k` is the classic eigengap estimate on
//! the low end of the spectrum. If no gap is significant the whole group stays
//! a single cluster.
//!
//! # Cost
//!
//! Per group the dense Gram matrix costs `O(n1 * nnz(A))` and the
//! eigendecomposition `O(n1^3)`.
//!
//! # Layout
//!
//! * `spectral`: the `Spec`/`AdaSpec` embedding and the normalized Laplacian
//!   embedding.
//! * `kmedians`: the deterministic `k`-medians used on the embedding.

mod kmedians;
mod spectral;

use anyhow::{Result, bail};
use nalgebra::base::DMatrix;

use kmedians::k_medians;
use spectral::{laplacian_embedding, spec_embedding};

/// A bipartite graph with `n1` type-I and `n2` type-II nodes, stored as a
/// dense adjacency matrix. `data[(i, j)]` is `true` iff type-I node `i` is
/// adjacent to type-II node `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct BipartiteGraph {
    pub data: DMatrix<bool>,
}

/// A community: type-I nodes together with the type-II nodes they connect to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    pub n1s: Vec<usize>,
    pub n2s: Vec<usize>,
}

/// A group of type-I nodes that is clustered as one independent subproblem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grouping {
    pub n1s: Vec<usize>,
}

/// Cluster the type-I nodes of `graph` into communities.
///
/// Every grouping is clustered independently on the induced subgraph spanned by
/// its type-I nodes and the type-II nodes they connect to. Type-I nodes that
/// are not mentioned in any grouping are returned as singleton clusters so that
/// each input node appears in exactly one output cluster. The `n2s` of a
/// cluster are the type-II nodes adjacent to its type-I nodes; these may be
/// shared between clusters.
///
/// The result is deterministic: repeated calls with the same input produce the
/// same partition.
///
/// # Errors
///
/// Returns an error if a grouping refers to a type-I node that does not exist
/// or if a type-I node occurs in more than one grouping.
pub fn cluster(graph: &BipartiteGraph, required_groupings: &[Grouping]) -> Result<Vec<Cluster>> {
    let groups = divide_by_groupings(graph.data.nrows(), required_groupings)?;
    let mut clusters = Vec::new();
    for group in &groups {
        clusters.extend(cluster_group(graph, group));
    }
    Ok(clusters)
}

/// Partition the type-I nodes into the groups that are clustered independently.
///
/// Without any required grouping the whole graph forms a single group.
fn divide_by_groupings(n1: usize, required_groupings: &[Grouping]) -> Result<Vec<Vec<usize>>> {
    if required_groupings.is_empty() {
        return Ok(if n1 == 0 {
            Vec::new()
        } else {
            vec![(0..n1).collect()]
        });
    }

    let mut seen = vec![false; n1];
    let mut groups = Vec::with_capacity(required_groupings.len());
    for (index, grouping) in required_groupings.iter().enumerate() {
        let mut group = Vec::with_capacity(grouping.n1s.len());
        for &node in &grouping.n1s {
            if node >= n1 {
                bail!("grouping {index}: type-I node {node} is out of range (n1 = {n1})");
            }
            if seen[node] {
                bail!("type-I node {node} occurs in more than one grouping");
            }
            seen[node] = true;
            group.push(node);
        }
        if !group.is_empty() {
            group.sort_unstable();
            groups.push(group);
        }
    }

    // Nodes left out of every grouping are clustered on their own.
    for (node, &covered) in seen.iter().enumerate() {
        if !covered {
            groups.push(vec![node]);
        }
    }
    Ok(groups)
}

/// The induced bipartite subgraph of a group of type-I nodes.
struct InducedSubgraph {
    /// Type-I nodes of the group, ascending.
    n1s: Vec<usize>,
    /// Type-II nodes adjacent to at least one node of the group, ascending.
    n2s: Vec<usize>,
    /// Dense `|n1s| x |n2s|` adjacency matrix.
    adjacency: DMatrix<f64>,
}

/// Restrict `graph` to `group` and the type-II nodes connected to it.
fn induced_subgraph(graph: &BipartiteGraph, group: &[usize]) -> InducedSubgraph {
    let n2_total = graph.data.ncols();
    let mut connected = vec![false; n2_total];
    let mut n2s = Vec::new();
    for &node in group {
        for column in 0..n2_total {
            if graph.data[(node, column)] && !connected[column] {
                connected[column] = true;
                n2s.push(column);
            }
        }
    }

    let mut adjacency = DMatrix::zeros(group.len(), n2s.len());
    for (row, &node) in group.iter().enumerate() {
        for (column, &neighbour) in n2s.iter().enumerate() {
            if graph.data[(node, neighbour)] {
                adjacency[(row, column)] = 1.0;
            }
        }
    }

    InducedSubgraph {
        n1s: group.to_vec(),
        n2s,
        adjacency,
    }
}

/// Cluster one group of type-I nodes into communities.
fn cluster_group(graph: &BipartiteGraph, group: &[usize]) -> Vec<Cluster> {
    let induced = induced_subgraph(graph, group);

    // Type-I nodes without neighbours carry no signal; each becomes its own
    // cluster so that every input node is represented in the output.
    let mut active = Vec::new();
    let mut clusters = Vec::new();
    for (row, &node) in induced.n1s.iter().enumerate() {
        if induced.adjacency.row(row).iter().any(|&value| value != 0.0) {
            active.push(row);
        } else {
            clusters.push(Cluster {
                n1s: vec![node],
                n2s: Vec::new(),
            });
        }
    }

    if active.len() < 2 {
        clusters.extend(active.iter().map(|&row| singleton_cluster(&induced, row)));
        clusters.sort_by_key(|cluster| cluster.n1s[0]);
        return clusters;
    }

    // Drop the isolated rows before forming the similarity matrix.
    let mut adjacency = DMatrix::zeros(active.len(), induced.n2s.len());
    for (row, &source) in active.iter().enumerate() {
        for (column, &value) in induced.adjacency.row(source).iter().enumerate() {
            adjacency[(row, column)] = value;
        }
    }

    let n1 = active.len() as f64;
    let n2 = induced.n2s.len() as f64;
    let edges = adjacency.iter().filter(|&&value| value != 0.0).count() as f64;
    let p = edges / (n1 * n2);

    // `p` is the density of the induced subgraph, used as a proxy for the
    // sparsity level `p_max` of the model; constants of the `>~` are ignored.
    let (embedding, k) = if n1 * n2 * p * p >= n1.ln() {
        spec_embedding(&adjacency, p)
    } else {
        laplacian_embedding(&adjacency)
    };

    let labels = k_medians(&embedding, k);

    let mut buckets = vec![Vec::new(); k];
    for (row, &label) in labels.iter().enumerate() {
        buckets[label].push(active[row]);
    }

    // A cluster owns every type-II node adjacent to one of its type-I nodes.
    let mut adjacent = vec![false; induced.n2s.len()];
    for bucket in buckets {
        if bucket.is_empty() {
            continue;
        }
        adjacent.fill(false);
        for &row in &bucket {
            for (column, &value) in induced.adjacency.row(row).iter().enumerate() {
                if value != 0.0 {
                    adjacent[column] = true;
                }
            }
        }
        let n2s = adjacent
            .iter()
            .enumerate()
            .filter_map(|(column, &yes)| yes.then_some(induced.n2s[column]))
            .collect();
        clusters.push(Cluster { n1s: bucket, n2s });
    }

    clusters.sort_by_key(|cluster| cluster.n1s[0]);
    clusters
}

/// The cluster of a single type-I node.
fn singleton_cluster(induced: &InducedSubgraph, row: usize) -> Cluster {
    let n2s = induced
        .adjacency
        .row(row)
        .iter()
        .enumerate()
        .filter_map(|(column, &value)| (value != 0.0).then_some(induced.n2s[column]))
        .collect();
    Cluster {
        n1s: vec![induced.n1s[row]],
        n2s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample a bipartite graph with a planted co-clustering structure.
    fn planted(
        n1: usize,
        n2: usize,
        communities: usize,
        p_in: f64,
        p_out: f64,
        seed: u64,
    ) -> (BipartiteGraph, Vec<usize>) {
        let mut rng = kmedians::SplitMix64::new(seed);
        let truth: Vec<usize> = (0..n1).map(|node| node * communities / n1).collect();
        let mut data = DMatrix::from_element(n1, n2, false);
        for node in 0..n1 {
            for other in 0..n2 {
                let same_block = truth[node] == other * communities / n2;
                let probability = if same_block { p_in } else { p_out };
                data[(node, other)] = rng.next_unit() < probability;
            }
        }
        (BipartiteGraph { data }, truth)
    }

    #[test]
    fn spec_recovers_planted_communities() {
        let (graph, truth) = planted(60, 300, 3, 0.6, 0.01, 7);
        let groupings = vec![Grouping {
            n1s: (0..60).collect(),
        }];
        let clusters = cluster(&graph, &groupings).unwrap();

        assert_eq!(
            clusters.len(),
            3,
            "expected one cluster per planted community"
        );
        let mut seen = [false; 60];
        for cluster in &clusters {
            let community = truth[cluster.n1s[0]];
            for &node in &cluster.n1s {
                assert_eq!(truth[node], community, "cluster mixes planted communities");
                seen[node] = true;
            }
            assert!(!cluster.n2s.is_empty());
        }
        assert!(
            seen.iter().all(|&covered| covered),
            "not every type-I node was clustered"
        );
    }

    #[test]
    fn sparse_graphs_fall_back_to_the_laplacian() {
        let (graph, truth) = planted(200, 600, 2, 0.01, 0.0002, 11);
        let clusters = cluster(&graph, &[]).unwrap();

        assert!(
            clusters.len() >= 2,
            "the sparse branch should split the graph"
        );
        let mut seen = [false; 200];
        for cluster in &clusters {
            let community = truth[cluster.n1s[0]];
            for &node in &cluster.n1s {
                assert_eq!(truth[node], community, "cluster mixes planted communities");
                seen[node] = true;
            }
        }
        assert!(
            seen.iter().all(|&covered| covered),
            "not every type-I node was clustered"
        );
    }

    #[test]
    fn unmentioned_nodes_become_singletons() {
        let mut data = DMatrix::from_element(5, 3, false);
        data[(0, 0)] = true;
        data[(1, 1)] = true;
        data[(2, 2)] = true;
        data[(3, 0)] = true;
        data[(4, 1)] = true;
        let graph = BipartiteGraph { data };

        let groupings = vec![Grouping { n1s: vec![0, 1] }];
        let clusters = cluster(&graph, &groupings).unwrap();

        for node in 2..5 {
            assert!(
                clusters.iter().any(|cluster| cluster.n1s == vec![node]),
                "node {node} should form a singleton cluster"
            );
        }
        let mut covered: Vec<usize> = clusters.iter().flat_map(|c| c.n1s.clone()).collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn invalid_groupings_are_rejected() {
        let graph = BipartiteGraph {
            data: DMatrix::from_element(3, 2, false),
        };

        let out_of_range = vec![Grouping { n1s: vec![3] }];
        assert!(cluster(&graph, &out_of_range).is_err());

        let duplicated = vec![Grouping { n1s: vec![0] }, Grouping { n1s: vec![0, 1] }];
        assert!(cluster(&graph, &duplicated).is_err());
    }

    #[test]
    fn k_medians_separates_blobs() {
        let embedding = DMatrix::from_row_slice(6, 1, &[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]);
        let labels = k_medians(&embedding, 2);
        assert!(labels[..3].iter().all(|&label| label == labels[0]));
        assert!(labels[3..].iter().all(|&label| label == labels[3]));
        assert_ne!(labels[0], labels[3]);
    }
}
