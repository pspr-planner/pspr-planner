//! Spectral clustering of the incidence graph and the merge tree over clusters.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::Result;
use nalgebra::DMatrix;

use crate::solver::partition::agglomeration::{
    self, Agglomeration, DistanceMatrix, EvaluationDag, Options as AgglomerationOptions,
};
use crate::solver::partition::bisbm::{self, BipartiteGraph, Cluster, Grouping};

use super::EvaluationOptions;
use super::graph::{ConstraintNode, TaskConstraintGraph, TaskNode};

/// The evaluation tree of an HTN task network.
#[derive(Debug, Clone)]
pub struct EvaluationTree {
    pub tasks: Vec<TaskNode>,
    pub constraints: Vec<ConstraintNode>,
    pub clusters: Vec<Cluster>,
    /// Constraint ids adjacent to more than one cluster.
    pub shared: Vec<usize>,
    /// Local constraint ids per cluster (parallel to `clusters`).
    pub local: Vec<Vec<usize>>,
    /// Dendrogram and schedule over the clusters; absent with a single cluster.
    pub agglomeration: Option<Agglomeration>,
    /// Shared constraint ids attached to every merge node (evaluation id).
    pub merge_constraints: HashMap<usize, Vec<usize>>,
    /// Number of probabilistic formulas, the only constructor the leaf encoding
    /// cannot translate.
    pub unmodeled: usize,
}

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        DisjointSet {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, first: usize, second: usize) {
        let first = self.find(first);
        let second = self.find(second);
        if first != second {
            let (low, high) = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            self.parent[high] = low;
        }
    }
}

fn connected_groupings(graph: &TaskConstraintGraph) -> Vec<Grouping> {
    let mut sets = DisjointSet::new(graph.tasks.len());
    for constraint in 0..graph.constraints.len() {
        let mut first = None;
        for (task, row) in graph.incidence.iter().enumerate() {
            if row[constraint] {
                match first {
                    None => first = Some(task),
                    Some(anchor) => sets.union(anchor, task),
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); graph.tasks.len()];
    for task in 0..graph.tasks.len() {
        let root = sets.find(task);
        groups[root].push(task);
    }
    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .map(|n1s| Grouping { n1s })
        .collect()
}

pub(super) fn cluster_tasks(
    graph: &TaskConstraintGraph,
    options: &EvaluationOptions,
) -> Result<Vec<Cluster>> {
    if graph.tasks.is_empty() {
        return Ok(Vec::new());
    }
    let mut data = DMatrix::from_element(graph.tasks.len(), graph.constraints.len(), false);
    for (task, row) in graph.incidence.iter().enumerate() {
        for (constraint, &present) in row.iter().enumerate() {
            data[(task, constraint)] = present;
        }
    }
    let groupings = if options.groupings.is_empty() {
        connected_groupings(graph)
    } else {
        options.groupings.clone()
    };
    bisbm::cluster(&BipartiteGraph { data }, &groupings)
}

pub(super) fn build_agglomeration(
    clusters: &[Cluster],
    options: &EvaluationOptions,
) -> Result<Option<Agglomeration>> {
    if clusters.len() < 2 {
        return Ok(None);
    }
    let count = clusters.len();
    let mut rows = vec![vec![0.0; count]; count];
    for first in 0..count {
        let left: BTreeSet<usize> = clusters[first].n2s.iter().copied().collect();
        for second in first + 1..count {
            let right: BTreeSet<usize> = clusters[second].n2s.iter().copied().collect();
            let intersection = left.intersection(&right).count() as f64;
            let union = left.union(&right).count() as f64;
            let distance = if union == 0.0 {
                0.0
            } else {
                1.0 - intersection / union
            };
            rows[first][second] = distance;
            rows[second][first] = distance;
        }
    }
    let distance = DistanceMatrix::new(&rows)?;
    let levels = options.levels.unwrap_or(count - 1).max(1);
    let constraints = constraint_groups(clusters);
    let settings = AgglomerationOptions {
        levels,
        lambda: options.lambda,
        global_weight: options.global_weight,
        use_search: options.use_search,
        search_expansions: options.search_expansions,
    };
    Ok(Some(agglomeration::optimize(
        &distance,
        &EvaluationDag::default(),
        &constraints,
        &settings,
    )?))
}

/// The clusters touched by every constraint that is shared by more than one
/// cluster, in ascending constraint order.
fn constraint_groups(clusters: &[Cluster]) -> Vec<Vec<usize>> {
    let mut uses: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, cluster) in clusters.iter().enumerate() {
        for &constraint in &cluster.n2s {
            let members = uses.entry(constraint).or_default();
            if members.last() != Some(&index) {
                members.push(index);
            }
        }
    }
    uses.into_values()
        .filter(|members| members.len() > 1)
        .collect()
}

pub(super) fn attach_shared_constraints(
    agglomeration: &Agglomeration,
    clusters: &[Cluster],
    shared: &[usize],
) -> HashMap<usize, Vec<usize>> {
    let dendrogram = &agglomeration.dendrogram;
    let mut attached: HashMap<usize, Vec<usize>> = HashMap::new();
    for &constraint in shared {
        let members: Vec<usize> = clusters
            .iter()
            .enumerate()
            .filter(|(_, cluster)| cluster.n2s.contains(&constraint))
            .map(|(index, _)| index)
            .collect();
        if members.len() < 2 {
            continue;
        }
        let node = (dendrogram.leaves..dendrogram.children.len()).find(|&node| {
            members
                .iter()
                .all(|member| dendrogram.members[node].contains(member))
        });
        if let Some(node) = node {
            attached
                .entry(node - dendrogram.leaves)
                .or_default()
                .push(constraint);
        }
    }
    attached
}
