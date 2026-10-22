//! The planning-style search over partial dendrograms.

use crate::solver::fast_downward::{
    Action, Evaluation, SearchOptions, SearchProblem, best_first_search,
};

use super::distance::{DistanceMatrix, EvaluationDag};
use super::hierarchy::Hierarchy;
use super::{Agglomeration, Options, finish};

/// Search for a hierarchy with the planning-style best-first search.
///
/// The state model, heuristic, and objective are described in the
/// [module documentation](super).
pub fn agglomerative_search(
    distance: &DistanceMatrix,
    dag: &EvaluationDag,
    options: &Options,
) -> anyhow::Result<Agglomeration> {
    let search = MergeSearch {
        distance,
        levels: options.levels,
    };
    let search_options = SearchOptions {
        max_expansions: options.search_expansions,
        max_generated: options.search_expansions.saturating_mul(20).max(1),
        use_preferred_operators: false,
        deferred_evaluation: false,
    };
    let outcome = best_first_search(&search, &search_options).ok_or_else(|| {
        anyhow::anyhow!("the search exhausted its node limit without finding a hierarchy")
    })?;
    let hierarchy = Hierarchy::from_merge(distance.len(), options.levels, outcome.state.merge);
    finish(hierarchy, distance, dag, options.lambda)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MergeState {
    level: usize,
    merged_at_level: bool,
    merge: Vec<Vec<Option<usize>>>,
}

struct MergeSearch<'a> {
    distance: &'a DistanceMatrix,
    levels: usize,
}

impl MergeSearch<'_> {
    fn hierarchy(&self, state: &MergeState) -> Hierarchy {
        Hierarchy::from_merge(self.distance.len(), self.levels, state.merge.clone())
    }

    /// Complete a partial hierarchy with average linkage, merging everything
    /// that is still open at the current level.
    fn complete_greedily(&self, state: &MergeState) -> Hierarchy {
        let mut hierarchy = self.hierarchy(state);
        let level = state.level.clamp(1, self.levels);
        loop {
            let clusters = hierarchy.clusters_at(level);
            if clusters.len() <= 1 {
                break;
            }
            let mut best = (0usize, 1usize);
            let mut best_distance = f64::INFINITY;
            for first in 0..clusters.len() {
                for second in first + 1..clusters.len() {
                    let mut total = 0.0;
                    for &a in &clusters[first] {
                        for &b in &clusters[second] {
                            total += self.distance.get(a, b);
                        }
                    }
                    let average = total / (clusters[first].len() * clusters[second].len()) as f64;
                    if average < best_distance {
                        best_distance = average;
                        best = (first, second);
                    }
                }
            }
            for &a in &clusters[best.0] {
                for &b in &clusters[best.1] {
                    hierarchy.set_merge(a, b, level);
                }
            }
        }
        hierarchy
    }
}

impl SearchProblem for MergeSearch<'_> {
    type State = MergeState;

    fn initial_state(&self) -> MergeState {
        MergeState {
            level: 1,
            merged_at_level: false,
            merge: vec![vec![None; self.distance.len()]; self.distance.len()],
        }
    }

    fn is_goal(&self, state: &MergeState) -> bool {
        self.hierarchy(state).clusters_at(state.level).len() == 1
    }

    fn successors(&self, state: &MergeState) -> Vec<Action<MergeState>> {
        let hierarchy = self.hierarchy(state);
        let clusters = hierarchy.clusters_at(state.level);
        if clusters.len() <= 1 {
            return Vec::new();
        }
        let mut successors = Vec::new();
        for first in 0..clusters.len() {
            for second in first + 1..clusters.len() {
                let mut next = state.clone();
                for &a in &clusters[first] {
                    for &b in &clusters[second] {
                        next.merge[a.min(b)][a.max(b)] = Some(state.level);
                    }
                }
                next.merged_at_level = true;
                successors.push(Action {
                    id: successors.len(),
                    state: next,
                    cost: 1.0,
                });
            }
        }
        if state.merged_at_level && state.level < self.levels {
            let mut next = state.clone();
            next.level += 1;
            next.merged_at_level = false;
            successors.push(Action {
                id: successors.len(),
                state: next,
                cost: 0.0,
            });
        }
        successors
    }

    fn evaluate(&self, state: &MergeState) -> Evaluation {
        // The number of internal evaluations is `n - 1` for every complete
        // binary hierarchy, so the scheduling term is nearly constant during
        // search; the estimate therefore focuses on the clustering objective.
        let estimate = -self
            .complete_greedily(state)
            .clustering_score(self.distance);
        Evaluation {
            estimates: vec![estimate],
            preferred: Vec::new(),
        }
    }
}
