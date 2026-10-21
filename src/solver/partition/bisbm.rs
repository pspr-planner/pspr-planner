use nalgebra::base::DMatrix;

// TODO: implement the spectral clustering algorithm from Braun et al.

/// A bipartite graph with `n1` and `n2` vertices and a flattened adjacency
/// matrix of size `n1 * n2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BipartiteGraph {
    pub data: DMatrix<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    pub n1s: Vec<usize>,
    pub n2s: Vec<usize>,
}

/// TODO: implement the spectral clustering algorithm from Braun et al.
pub fn cluster(_graph: &BipartiteGraph) -> Result<Vec<Cluster>, String> {
    Ok(Vec::new())
}
