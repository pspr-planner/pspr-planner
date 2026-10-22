//! The rooted binary dendrogram induced by a complete hierarchy.

use std::collections::HashMap;

use anyhow::{Result, ensure};

use super::hierarchy::Hierarchy;

/// A rooted binary dendrogram.
///
/// Node `0..leaves` are the leaves. Internal nodes are created in
/// chronological order, so the internal node `leaves + k` has evaluation id
/// `k`. Multi-way merges at one level are binarised by folding the children in
/// ascending order.
#[derive(Debug, Clone, PartialEq)]
pub struct Dendrogram {
    pub leaves: usize,
    pub children: Vec<Vec<usize>>,
    pub members: Vec<Vec<usize>>,
    pub merge_levels: Vec<usize>,
    pub root: usize,
}

impl Dendrogram {
    /// Build the binary dendrogram induced by a complete hierarchy.
    pub fn build(hierarchy: &Hierarchy) -> Result<Self> {
        let n = hierarchy.instances();
        ensure!(
            hierarchy.is_complete(),
            "the hierarchy must assign a merge level to every pair"
        );

        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut members: Vec<Vec<usize>> = (0..n).map(|instance| vec![instance]).collect();
        let mut merge_levels = vec![0usize; n];
        let mut node_of: HashMap<Vec<usize>, usize> =
            (0..n).map(|instance| (vec![instance], instance)).collect();
        let mut previous: Vec<Vec<usize>> = (0..n).map(|instance| vec![instance]).collect();

        for level in 1..=hierarchy.levels() {
            let current = hierarchy.clusters_at(level);
            for cluster in &current {
                if cluster.len() == 1 {
                    continue;
                }
                let mut child_nodes: Vec<usize> = previous
                    .iter()
                    .filter(|previous_cluster| {
                        previous_cluster
                            .iter()
                            .all(|member| cluster.contains(member))
                    })
                    .map(|previous_cluster| node_of[previous_cluster])
                    .collect();
                child_nodes.sort_unstable();
                let mut node = *child_nodes
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("cluster {cluster:?} has no children"))?;
                for &child in &child_nodes[1..] {
                    let id = children.len();
                    let mut node_members = members[node].clone();
                    node_members.extend_from_slice(&members[child]);
                    node_members.sort_unstable();
                    children.push(vec![node, child]);
                    members.push(node_members);
                    merge_levels.push(level);
                    node = id;
                }
                node_of.insert(cluster.clone(), node);
            }
            previous = current;
        }

        let all: Vec<usize> = (0..n).collect();
        let root = *node_of
            .get(&all)
            .ok_or_else(|| anyhow::anyhow!("the hierarchy did not merge all instances"))?;
        Ok(Dendrogram {
            leaves: n,
            children,
            members,
            merge_levels,
            root,
        })
    }

    /// The number of internal (evaluation) nodes.
    pub fn internal_count(&self) -> usize {
        self.children.len() - self.leaves
    }

    /// The evaluation id of an internal node, if `node` is internal.
    pub fn evaluation_id(&self, node: usize) -> Option<usize> {
        if node >= self.leaves {
            Some(node - self.leaves)
        } else {
            None
        }
    }

    /// The direct internal predecessors of every internal node: a child must
    /// be evaluated before its parent (constraint 3).
    pub(super) fn tree_precedences(&self) -> Vec<(usize, usize)> {
        let mut precedences = Vec::new();
        for (node, children) in self.children.iter().enumerate().skip(self.leaves) {
            for &child in children {
                if let Some(child_id) = self.evaluation_id(child) {
                    precedences.push((child_id, node - self.leaves));
                }
            }
        }
        precedences
    }
}
