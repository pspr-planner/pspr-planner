//! Evaluation scheduling of the dendrogram (constraints 1-4).
//!
//! The scheduling problem is a feasibility problem, not an optimization one:
//! every internal node occupies exactly one position and every position is
//! taken, so the objective "sum of the evaluation positions" is
//! `1 + 2 + ... + n` for *every* linear extension of the precedences. The
//! minimum is therefore reached by any topological order, which is what this
//! module computes.

use std::collections::{BTreeSet, BinaryHeap, VecDeque};

use anyhow::{Result, ensure};

use super::dendrogram::Dendrogram;
use super::distance::EvaluationDag;

/// The evaluation order of the internal dendrogram nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// Internal nodes (evaluation ids) in evaluation order.
    pub order: Vec<usize>,
    /// 1-based evaluation position of every internal node.
    pub positions: Vec<usize>,
}

/// Schedule the internal dendrogram nodes subject to the tree and DAG
/// precedences.
///
/// Ties are broken towards the lowest node id, which keeps the schedule
/// deterministic.
pub fn optimal_schedule(dendrogram: &Dendrogram, dag: &EvaluationDag) -> Result<Schedule> {
    let n = dendrogram.internal_count();
    let mut precedences: BTreeSet<(usize, usize)> =
        dendrogram.tree_precedences().into_iter().collect();
    for &(before, after) in &dag.edges {
        ensure!(
            before < n && after < n,
            "DAG edge ({before}, {after}) refers to an internal node outside 0..{n}"
        );
        ensure!(
            before != after,
            "DAG edge ({before}, {after}) is a self-loop"
        );
        precedences.insert((before, after));
    }
    ensure!(
        is_acyclic(n, &precedences),
        "the precedence constraints contain a cycle"
    );
    if n == 0 {
        return Ok(Schedule {
            order: Vec::new(),
            positions: Vec::new(),
        });
    }

    let mut indegree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(before, after) in &precedences {
        successors[before].push(after);
        indegree[after] += 1;
    }
    // A min-heap over the ready nodes yields the smallest available id first.
    let mut ready: BinaryHeap<std::cmp::Reverse<usize>> = (0..n)
        .filter(|&node| indegree[node] == 0)
        .map(std::cmp::Reverse)
        .collect();

    let mut order = Vec::with_capacity(n);
    while let Some(std::cmp::Reverse(node)) = ready.pop() {
        order.push(node);
        for &successor in &successors[node] {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.push(std::cmp::Reverse(successor));
            }
        }
    }
    ensure!(
        order.len() == n,
        "the precedence constraints contain a cycle"
    );

    let mut positions = vec![0usize; n];
    for (index, &node) in order.iter().enumerate() {
        positions[node] = index + 1;
    }
    Ok(Schedule { order, positions })
}

fn is_acyclic(n: usize, edges: &BTreeSet<(usize, usize)>) -> bool {
    let mut indegree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(before, after) in edges {
        successors[before].push(after);
        indegree[after] += 1;
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&node| indegree[node] == 0).collect();
    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        for &successor in &successors[node] {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                queue.push_back(successor);
            }
        }
    }
    visited == n
}
