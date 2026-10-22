//! Joint hierarchical clustering and evaluation scheduling.
//!
//! Combines the dendrogram formulation of Gilpin et al., *Formalizing
//! Hierarchical Clustering as Integer Linear Programming*, with a custom
//! planning-style search inspired by Fast Downward.
//!
//! # Dendrogram formulation
//!
//! Let \(X\) be the set of input observations, with \(n = |X|\). The dendrogram
//! is represented using the merge-level variables of Gilpin et al. For each
//! pair of distinct observations \(i,j \in X\), let
//! \(M_{ij} \in \{1,\ldots,n-1\}\) denote the level at which \(i\) and \(j\)
//! first belong to the same cluster. The variables must satisfy the
//! dendrogram-validity constraints of the underlying formulation, including
//! symmetry, hierarchical consistency, and the conditions required to
//! represent a valid binary hierarchy.
//!
//! In particular, the merge levels satisfy
//!
//! \[
//! M_{ij} = M_{ji}
//! \qquad \forall i,j \in X,\ i \ne j.
//! \]
//!
//! The complete set of structural constraints is inherited from the chosen
//! Gilpin et al. formulation; the merge-level variables alone do not guarantee
//! that an arbitrary assignment represents a valid dendrogram.
//!
//! # Evaluation schedule
//!
//! Let \(A=(V_A,E_A)\) be the rooted binary dendrogram, with edges directed
//! from children to parents. Let \(I \subseteq V_A\) be the set of internal
//! nodes that require evaluation, and let \(N=|I|\). Leaves are assumed to
//! be available before evaluation begins.
//!
//! Introduce binary decision variables
//!
//! \[
//! x_{v,t} \in \{0,1\},
//! \]
//!
//! where \(x_{v,t}=1\) iff internal node \(v\) is evaluated at time step
//! \(t\), for \(v \in I\) and \(t \in \{1,\ldots,N\}\).
//!
//! Every internal node is evaluated exactly once:
//!
//! \[
//! \sum_{t=1}^{N} x_{v,t}=1
//! \qquad \forall v\in I.
//! \tag{1}
//! \]
//!
//! Exactly one evaluation operation occurs at each time step:
//!
//! \[
//! \sum_{v\in I}x_{v,t}=1
//! \qquad \forall t\in\{1,\ldots,N\}.
//! \tag{2}
//! \]
//!
//! Define the evaluation position of each internal node as the linear
//! expression
//!
//! \[
//! p_v=\sum_{t=1}^{N}t\,x_{v,t}.
//! \]
//!
//! For every parent-child edge \((u,v)\in E_A\) whose endpoints are both
//! internal evaluation nodes, the child must be evaluated before the parent:
//!
//! \[
//! p_u+1\le p_v
//! \qquad \forall (u,v)\in E_A,\ u,v\in I.
//! \tag{3}
//! \]
//!
//! # Additional DAG precedence constraints
//!
//! Let \(B=(V_B,E_B)\) be a directed acyclic graph whose vertices correspond
//! to internal evaluation nodes in \(A\). Let
//! \(f:V_B\to I\) map each DAG vertex to its corresponding evaluation node.
//! Every edge \((a,b)\in E_B\) requires the evaluation of \(f(a)\) to precede
//! that of \(f(b)\):
//!
//! \[
//! p_{f(a)}+1\le p_{f(b)}
//! \qquad \forall (a,b)\in E_B.
//! \tag{4}
//! \]
//!
//! The union of the precedence constraints from \(A\) and \(B\) must be
//! acyclic for a valid schedule to exist. Although \(B\) is itself a DAG,
//! its union with the tree precedence constraints may contain cycles.
//!
//! # Planning-style search
//!
//! A custom state-space search inspired by Fast Downward explores candidate
//! merge structures and evaluation schedules. Each state represents a
//! partial dendrogram and the current cluster configuration. Successor
//! generation considers legal merge and level-advance actions. The search uses
//! the dendrogram-validity constraints to reject partial structures that
//! cannot be extended to a valid hierarchy, and makes use of the constructs in
//! [`crate::solver::fast_downward`].
//!
//! # Objective
//!
//! The search jointly considers clustering quality, evaluation cost and the
//! height at which shared constraints are attached. A representative objective
//! is
//!
//! \[
//! \max_{A,\sigma}
//! \quad Q(A)-\lambda C(A,\sigma)-\gamma G(A),
//! \]
//!
//! subject to the dendrogram-validity constraints and precedence constraints
//! (1)--(4). Here \(Q(A)\) measures clustering quality, \(C(A,\sigma)\)
//! measures the cost of evaluating dendrogram \(A\) under schedule
//! \(\sigma\), and \(G(A)\) is the sum of the merge levels at which the
//! shared constraints are attached: a constraint whose clusters meet at the
//! root contributes \(L\), one handled by the first merge contributes 1.
//! \(\gamma\) is [`Options::global_weight`]. Minimising \(G(A)\) keeps
//! global constraints out of the merges high up the hierarchy, where each
//! constraint would be re-evaluated against the largest subplans.
//!
//! Unless the search strategy and heuristic satisfy additional conditions, the
//! method does not guarantee a globally optimal solution.
//!
//! # Implementation
//!
//! * `hierarchy`: the Gilpin et al. merge-level ILP (Figure 2), solved exactly
//!   with `good_lp` over SCIP. Shared constraints add one height variable each,
//!   lower-bounded by the merge level of every pair of its clusters, so the
//!   objective rewards hierarchies in which constraints are attached low.
//! * `dendrogram`: the rooted binary tree induced by the merge levels;
//!   multi-way merges are binarised in ascending order.
//! * `schedule`: the scheduling ILP (1)--(4) for a dendrogram and an additional
//!   [`EvaluationDag`].
//! * `search`: the planning-style search over partial dendrograms, built on the
//!   generic best-first search engine.
//!
//! [`optimize`] solves the ILP, optionally runs the search, and returns the
//! candidate with the better joint objective `Q(A) - lambda C(A, sigma)`.
//!
//! In the search, every complete binary hierarchy has `n - 1` internal nodes,
//! so the scheduling term is nearly constant; the average-linkage completion
//! heuristic therefore estimates states by the negated clustering objective.
//! The schedule is optimised exactly once a hierarchy has been chosen.

mod dendrogram;
mod distance;
mod hierarchy;
mod schedule;
mod search;

use anyhow::{Result, ensure};

pub use dendrogram::Dendrogram;
pub use distance::{DistanceMatrix, EvaluationDag};
pub use hierarchy::{Hierarchy, optimal_hierarchy, optimal_hierarchy_with};
pub use schedule::{Schedule, optimal_schedule};
pub use search::agglomerative_search;

/// Options for [`optimize`] and [`agglomerative_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Number of levels `L` of the hierarchy; merge levels live in `1..=L`.
    pub levels: usize,
    /// Weight of the scheduling cost in the joint objective.
    pub lambda: f64,
    /// Weight of the shared-constraint height in the joint objective.
    ///
    /// Every shared constraint contributes its merge level (the level of the
    /// lowest common ancestor of the clusters it touches), so a constraint
    /// handled at the root costs `levels` and one handled by the first merge
    /// costs 1.
    pub global_weight: f64,
    /// Also run the planning-style search (otherwise only the ILP is solved).
    pub use_search: bool,
    /// Expansion limit of the planning-style search.
    pub search_expansions: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            levels: 4,
            lambda: 0.0,
            global_weight: 1.0,
            use_search: false,
            search_expansions: 10_000,
        }
    }
}

/// A hierarchy together with its dendrogram, its optimal schedule and the
/// value of the joint objective
/// `Q(A) - lambda * C(A, sigma) - global_weight * G(A)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Agglomeration {
    pub hierarchy: Hierarchy,
    pub dendrogram: Dendrogram,
    pub schedule: Schedule,
    pub clustering_score: f64,
    pub scheduling_cost: f64,
    /// Sum of the merge levels at which the shared constraints are attached.
    pub global_cost: f64,
    pub score: f64,
}

pub(super) fn finish(
    hierarchy: Hierarchy,
    distance: &DistanceMatrix,
    dag: &EvaluationDag,
    constraints: &[Vec<usize>],
    lambda: f64,
    global_weight: f64,
) -> Result<Agglomeration> {
    let dendrogram = Dendrogram::build(&hierarchy)?;
    let schedule = optimal_schedule(&dendrogram, dag)?;
    let clustering_score = hierarchy.clustering_score(distance);
    let scheduling_cost = schedule.positions.iter().sum::<usize>() as f64;
    let global_cost = hierarchy.constraint_cost(constraints);
    Ok(Agglomeration {
        hierarchy,
        dendrogram,
        schedule,
        clustering_score,
        scheduling_cost,
        global_cost,
        score: clustering_score - lambda * scheduling_cost - global_weight * global_cost,
    })
}

/// Solve the exact ILP and, optionally, the planning-style search; return the
/// candidate with the better joint objective.
pub fn optimize(
    distance: &DistanceMatrix,
    dag: &EvaluationDag,
    constraints: &[Vec<usize>],
    options: &Options,
) -> Result<Agglomeration> {
    ensure!(options.lambda >= 0.0, "lambda must be non-negative");
    ensure!(
        options.global_weight >= 0.0,
        "the global constraint weight must be non-negative"
    );
    let hierarchy =
        optimal_hierarchy_with(distance, options.levels, constraints, options.global_weight)?;
    let mut best = finish(
        hierarchy,
        distance,
        dag,
        constraints,
        options.lambda,
        options.global_weight,
    )?;
    if options.use_search {
        let candidate = agglomerative_search(distance, dag, constraints, options)?;
        if candidate.score > best.score {
            best = candidate;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four points in two well-separated pairs on a line.
    fn paired_points() -> DistanceMatrix {
        let coordinates: [f64; 4] = [0.0, 0.1, 10.0, 10.1];
        let rows: Vec<Vec<f64>> = coordinates
            .iter()
            .map(|&first| {
                coordinates
                    .iter()
                    .map(|&second| (first - second).abs())
                    .collect()
            })
            .collect();
        DistanceMatrix::new(&rows).unwrap()
    }

    #[test]
    fn distance_matrix_is_validated() {
        assert!(DistanceMatrix::new(&[vec![0.0, 1.0], vec![1.0, 0.0]]).is_ok());
        assert!(DistanceMatrix::new(&[vec![0.0, 1.0], vec![2.0, 0.0]]).is_err());
        assert!(DistanceMatrix::new(&[vec![0.0, -1.0], vec![-1.0, 0.0]]).is_err());
        assert!(DistanceMatrix::new(&[vec![1.0]]).is_err());
        assert!(DistanceMatrix::new(&[vec![0.0, 1.0], vec![1.0]]).is_err());
    }

    #[test]
    fn ilp_merges_close_pairs_first() {
        let distance = paired_points();
        let hierarchy = optimal_hierarchy(&distance, 2).unwrap();
        assert_eq!(hierarchy.merge_level(0, 1), Some(1));
        assert_eq!(hierarchy.merge_level(2, 3), Some(1));
        assert_eq!(hierarchy.merge_level(0, 2), Some(2));
        assert_eq!(hierarchy.merge_level(0, 3), Some(2));
        assert_eq!(hierarchy.clusters_at(1), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(hierarchy.clusters_at(2), vec![vec![0, 1, 2, 3]]);
        // The score is positive because close pairs are merged first.
        assert!(hierarchy.clustering_score(&distance) > 0.0);
    }

    #[test]
    fn dendrogram_is_binary_and_has_all_leaves() {
        let distance = paired_points();
        let hierarchy = optimal_hierarchy(&distance, 2).unwrap();
        let dendrogram = Dendrogram::build(&hierarchy).unwrap();
        assert_eq!(dendrogram.leaves, 4);
        assert_eq!(dendrogram.internal_count(), 3);
        assert_eq!(dendrogram.members[dendrogram.root], vec![0, 1, 2, 3]);
        for children in dendrogram.children.iter().skip(dendrogram.leaves) {
            assert_eq!(children.len(), 2, "internal nodes are binary");
        }
    }

    #[test]
    fn schedule_respects_tree_and_dag_precedences() {
        let distance = paired_points();
        let hierarchy = optimal_hierarchy(&distance, 2).unwrap();
        let dendrogram = Dendrogram::build(&hierarchy).unwrap();
        let dag = EvaluationDag::new(vec![(0, 2)]);
        let schedule = optimal_schedule(&dendrogram, &dag).unwrap();

        assert_eq!(schedule.order.len(), 3);
        assert_eq!(schedule.positions.len(), 3);
        for (before, after) in dendrogram.tree_precedences() {
            assert!(schedule.positions[before] < schedule.positions[after]);
        }
        assert!(schedule.positions[0] < schedule.positions[2]);
        // Every position is used exactly once.
        let mut positions = schedule.positions.clone();
        positions.sort_unstable();
        assert_eq!(positions, vec![1, 2, 3]);
    }

    #[test]
    fn schedule_rejects_cycles() {
        let distance = paired_points();
        let hierarchy = optimal_hierarchy(&distance, 2).unwrap();
        let dendrogram = Dendrogram::build(&hierarchy).unwrap();
        let dag = EvaluationDag::new(vec![(0, 1), (1, 0)]);
        assert!(optimal_schedule(&dendrogram, &dag).is_err());
    }

    #[test]
    fn optimize_reports_a_consistent_objective() {
        let distance = paired_points();
        let options = Options {
            levels: 2,
            lambda: 0.5,
            global_weight: 0.5,
            use_search: false,
            search_expansions: 100,
        };
        let constraints = vec![vec![0, 2]];
        let result =
            optimize(&distance, &EvaluationDag::default(), &constraints, &options).unwrap();
        assert_eq!(
            result.dendrogram.members[result.dendrogram.root],
            vec![0, 1, 2, 3]
        );
        assert!(
            (result.score
                - (result.clustering_score
                    - 0.5 * result.scheduling_cost
                    - 0.5 * result.global_cost))
                .abs()
                < 1e-9
        );
        assert_eq!(result.schedule.positions.len(), 3);
    }

    #[test]
    fn search_finds_a_complete_hierarchy() {
        let distance = paired_points();
        let options = Options {
            levels: 2,
            lambda: 0.0,
            global_weight: 0.0,
            use_search: true,
            search_expansions: 2_000,
        };
        let result =
            agglomerative_search(&distance, &EvaluationDag::default(), &[], &options).unwrap();
        assert!(result.hierarchy.is_complete());
        assert_eq!(
            result.dendrogram.members[result.dendrogram.root],
            vec![0, 1, 2, 3]
        );
        assert_eq!(result.schedule.positions.len(), 3);
    }

    #[test]
    fn constraint_cost_counts_merge_levels() {
        let distance = paired_points();
        let hierarchy = optimal_hierarchy(&distance, 2).unwrap();
        assert_eq!(
            hierarchy.constraint_cost(&[vec![0, 1], vec![2, 3]]),
            2.0,
            "both constraints are handled by a first-level merge"
        );
        assert_eq!(
            hierarchy.constraint_cost(&[vec![0, 2]]),
            2.0,
            "0 and 2 only meet at the root"
        );
    }

    #[test]
    fn global_constraints_are_pushed_down_the_hierarchy() {
        let distance = paired_points();
        let plain = optimal_hierarchy(&distance, 2).unwrap();
        assert_eq!(plain.merge_level(0, 2), Some(2));

        let groups = vec![vec![0, 2]];
        let constrained = optimal_hierarchy_with(&distance, 2, &groups, 1_000.0).unwrap();
        assert_eq!(
            constrained.merge_level(0, 2),
            Some(1),
            "a heavy global constraint must be handled by a first-level merge"
        );
        assert_eq!(constrained.constraint_cost(&groups), 1.0);
    }

    #[test]
    fn optimize_includes_the_global_constraint_cost() {
        let distance = paired_points();
        let options = Options {
            levels: 2,
            lambda: 0.0,
            global_weight: 1_000.0,
            use_search: false,
            search_expansions: 100,
        };
        let groups = vec![vec![0, 2]];
        let result = optimize(&distance, &EvaluationDag::default(), &groups, &options).unwrap();
        assert_eq!(result.hierarchy.merge_level(0, 2), Some(1));
        assert!((result.global_cost - 1.0).abs() < 1e-9);
        assert!(
            (result.score - (result.clustering_score - 1_000.0 * result.global_cost)).abs() < 1e-9
        );
    }
}
