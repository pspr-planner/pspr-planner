//! The Gilpin et al. merge-level ILP (Figure 2) and the induced hierarchy.

use anyhow::{Result, ensure};
use good_lp::{
    Constraint, Expression, ProblemVariables, Solution, SolverModel, Variable, default_solver,
    variable,
};

use super::DistanceMatrix;

/// The outcome of the Gilpin et al. formulation: a merge level `M(a, b)` for
/// every pair of distinct instances.
#[derive(Debug, Clone, PartialEq)]
pub struct Hierarchy {
    instances: usize,
    levels: usize,
    merge: Vec<Vec<Option<usize>>>,
}

impl Hierarchy {
    pub(super) fn empty(instances: usize, levels: usize) -> Self {
        Hierarchy {
            instances,
            levels,
            merge: vec![vec![None; instances]; instances],
        }
    }

    pub(super) fn from_merge(
        instances: usize,
        levels: usize,
        merge: Vec<Vec<Option<usize>>>,
    ) -> Self {
        Hierarchy {
            instances,
            levels,
            merge,
        }
    }

    /// The level at which `a` and `b` first belong to the same cluster.
    pub fn merge_level(&self, a: usize, b: usize) -> Option<usize> {
        if a == b {
            Some(0)
        } else if a < b {
            self.merge[a][b]
        } else {
            self.merge[b][a]
        }
    }

    pub(super) fn set_merge(&mut self, a: usize, b: usize, level: usize) {
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        self.merge[low][high] = Some(level);
    }

    /// The number of levels of the hierarchy.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// The number of instances.
    pub fn instances(&self) -> usize {
        self.instances
    }

    /// Whether every pair has a merge level.
    pub fn is_complete(&self) -> bool {
        self.merge
            .iter()
            .enumerate()
            .all(|(a, row)| row.iter().skip(a + 1).all(|entry| entry.is_some()))
    }

    /// The clusters of the hierarchy at `level`, i.e. the connected components
    /// of the pairs merged at or before `level`.
    pub fn clusters_at(&self, level: usize) -> Vec<Vec<usize>> {
        let mut sets = DisjointSet::new(self.instances);
        for (a, row) in self.merge.iter().enumerate() {
            for (b, entry) in row.iter().enumerate().skip(a + 1) {
                if entry.is_some_and(|merge| merge <= level) {
                    sets.union(a, b);
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); self.instances];
        for instance in 0..self.instances {
            let root = sets.find(instance);
            groups[root].push(instance);
        }
        let mut clusters: Vec<Vec<usize>> = groups
            .into_iter()
            .filter(|group| !group.is_empty())
            .collect();
        clusters.sort_by_key(|cluster| cluster[0]);
        clusters
    }

    /// The objective of Equation (1): the sum of `D(a, c) - D(a, b)` over all
    /// triples where `a` and `b` are merged before `a` and `c`.
    pub fn clustering_score(&self, distance: &DistanceMatrix) -> f64 {
        assert!(
            self.is_complete(),
            "the clustering score requires a complete hierarchy"
        );
        let n = self.instances;
        let mut score = 0.0;
        for a in 0..n {
            for b in 0..n {
                if b == a {
                    continue;
                }
                let merged_ab = self.merge_level(a, b).expect("complete hierarchy");
                for c in 0..n {
                    if c == a || c == b {
                        continue;
                    }
                    let merged_ac = self.merge_level(a, c).expect("complete hierarchy");
                    if merged_ab < merged_ac {
                        score += distance.get(a, c) - distance.get(a, b);
                    }
                }
            }
        }
        score
    }
}

/// Solve the ILP of Gilpin et al. (Figure 2) for the given number of levels.
pub fn optimal_hierarchy(distance: &DistanceMatrix, levels: usize) -> Result<Hierarchy> {
    let n = distance.len();
    ensure!(levels >= 1, "at least one hierarchy level is required");
    ensure!(
        n >= 2,
        "hierarchical clustering needs at least two instances"
    );

    let mut variables = ProblemVariables::new();
    let mut merge: Vec<Vec<Option<Variable>>> = vec![vec![None; n]; n];
    for (a, row) in merge.iter_mut().enumerate() {
        for entry in row.iter_mut().skip(a + 1) {
            *entry = Some(variables.add(variable().integer().min(1).max(levels as f64)));
        }
    }
    let pair = |a: usize, b: usize| -> Variable {
        if a < b {
            merge[a][b].expect("pair variable")
        } else {
            merge[b][a].expect("pair variable")
        }
    };

    struct Triple {
        a: usize,
        b: usize,
        c: usize,
        o: Variable,
        z_ab: Variable,
        z_bc: Variable,
        weight: f64,
    }

    let mut triples = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if b == a {
                continue;
            }
            for c in 0..n {
                if c == a || c == b {
                    continue;
                }
                triples.push(Triple {
                    a,
                    b,
                    c,
                    o: variables.add(variable().binary()),
                    z_ab: variables.add(variable().binary()),
                    z_bc: variables.add(variable().binary()),
                    weight: distance.get(a, c) - distance.get(a, b),
                });
            }
        }
    }

    let big_m = levels as f64 + 1.0;
    let lower_bound = -(levels as f64);
    let mut constraints: Vec<Constraint> = Vec::new();
    for triple in &triples {
        let m_ab = Expression::from_other_affine(pair(triple.a, triple.b));
        let m_ac = Expression::from_other_affine(pair(triple.a, triple.c));
        let m_bc = Expression::from_other_affine(pair(triple.b, triple.c));
        let o = Expression::from_other_affine(triple.o);
        let z_ab = Expression::from_other_affine(triple.z_ab);
        let z_bc = Expression::from_other_affine(triple.z_bc);

        // -L <= M(a,c) - M(a,b) - (L+1) O <= 0
        let indicator = m_ac.clone() - m_ab.clone() - big_m * o;
        constraints.push(indicator.clone().leq(0.0));
        constraints.push(indicator.geq(lower_bound));

        // -L <= M(a,b) - M(a,c) - (L+1) Z_ab + 1 <= 0
        let left = m_ab - m_ac.clone() - big_m * z_ab.clone();
        constraints.push((left.clone() + 1.0).leq(0.0));
        constraints.push((left + 1.0).geq(lower_bound));

        // -L <= M(b,c) - M(a,c) - (L+1) Z_bc + 1 <= 0
        let right = m_bc - m_ac - big_m * z_bc.clone();
        constraints.push((right.clone() + 1.0).leq(0.0));
        constraints.push((right + 1.0).geq(lower_bound));

        // Z_ab + Z_bc >= 1
        constraints.push((z_ab + z_bc).geq(1.0));
    }

    let objective: Expression = triples
        .iter()
        .map(|triple| triple.weight * Expression::from_other_affine(triple.o))
        .sum();
    let mut model = variables.maximise(objective).using(default_solver);
    for constraint in constraints {
        model.add_constraint(constraint);
    }
    let solution = model
        .solve()
        .map_err(|error| anyhow::anyhow!("the ILP solver failed: {error}"))?;

    let mut hierarchy = Hierarchy::empty(n, levels);
    for a in 0..n {
        for b in a + 1..n {
            let value = solution.value(pair(a, b)).round() as usize;
            ensure!(
                (1..=levels).contains(&value),
                "the solver returned the merge level {value} outside 1..={levels}"
            );
            hierarchy.set_merge(a, b, value);
        }
    }
    Ok(hierarchy)
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
        let first_root = self.find(first);
        let second_root = self.find(second);
        if first_root != second_root {
            let (low, high) = if first_root < second_root {
                (first_root, second_root)
            } else {
                (second_root, first_root)
            };
            self.parent[high] = low;
        }
    }
}
