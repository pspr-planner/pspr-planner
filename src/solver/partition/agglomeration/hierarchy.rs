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

    /// The sum of the merge levels at which the given shared constraints are
    /// attached.
    ///
    /// A constraint that touches a group of instances is handled at the lowest
    /// common ancestor of those instances, whose level is the highest merge
    /// level over the pairs of the group. The cost therefore grows with the
    /// height of the merge, which is exactly the number of evaluation levels a
    /// constraint spans.
    pub fn constraint_cost(&self, constraints: &[Vec<usize>]) -> f64 {
        constraints
            .iter()
            .map(|group| {
                let mut height = 0usize;
                for (first, &a) in group.iter().enumerate() {
                    assert!(a < self.instances, "instance {a} is out of range");
                    for &b in &group[first + 1..] {
                        assert!(b < self.instances, "instance {b} is out of range");
                        if let Some(level) = self.merge_level(a, b) {
                            height = height.max(level);
                        }
                    }
                }
                height as f64
            })
            .sum()
    }
}

/// Solve the ILP of Gilpin et al. (Figure 2) without a constraint-locality
/// term.
pub fn optimal_hierarchy(distance: &DistanceMatrix, levels: usize) -> Result<Hierarchy> {
    optimal_hierarchy_with(distance, levels, &[], 0.0)
}

/// Solve the ILP of Gilpin et al. (Figure 2) with a penalty on shared
/// constraints that end up high in the hierarchy.
///
/// `groups` lists, for every shared constraint, the instances that touch it.
/// The merge level of a constraint is the highest level at which two of its
/// members join, i.e. the level of its lowest common ancestor, and the
/// objective subtracts `global_weight` times the sum of those levels. Raising
/// the weight pushes shared constraints towards the bottom of the hierarchy,
/// where small merges handle them, instead of the root merge.
pub fn optimal_hierarchy_with(
    distance: &DistanceMatrix,
    levels: usize,
    groups: &[Vec<usize>],
    global_weight: f64,
) -> Result<Hierarchy> {
    let n = distance.len();
    ensure!(levels >= 1, "at least one hierarchy level is required");
    ensure!(
        n >= 2,
        "hierarchical clustering needs at least two instances"
    );
    ensure!(
        global_weight >= 0.0,
        "the global constraint weight must be non-negative"
    );
    for group in groups {
        ensure!(
            group.iter().all(|&instance| instance < n),
            "a shared constraint refers to an instance outside the distance matrix"
        );
    }

    // The ILP below solves the same objective as the enumeration but its
    // branch-and-bound effort grows very quickly with the number of
    // instances: five instances already take tens of milliseconds and eight
    // take seconds. Small problems are therefore solved exactly by
    // enumerating the level-ordered dendrograms instead.
    if let Some(hierarchy) = enumerate_hierarchy(distance, levels, groups, global_weight) {
        return Ok(hierarchy);
    }

    solve_ilp(distance, levels, groups, global_weight)
}

/// Solve the merge-level ILP of Gilpin et al. directly.
fn solve_ilp(
    distance: &DistanceMatrix,
    levels: usize,
    groups: &[Vec<usize>],
    global_weight: f64,
) -> Result<Hierarchy> {
    let n = distance.len();
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

    struct HeightVariable {
        members: Vec<usize>,
        level: Variable,
    }

    let mut heights = Vec::new();
    if global_weight > 0.0 {
        for group in groups {
            let mut members = group.clone();
            members.sort_unstable();
            members.dedup();
            if members.len() < 2 {
                continue;
            }
            let level = variables.add(variable().min(1.0).max(levels as f64));
            heights.push(HeightVariable { members, level });
        }
    }

    for height in &heights {
        for (first, &a) in height.members.iter().enumerate() {
            for &b in &height.members[first + 1..] {
                let merge = Expression::from_other_affine(pair(a, b));
                constraints.push(Expression::from_other_affine(height.level).geq(merge));
            }
        }
    }

    let mut objective: Expression = triples
        .iter()
        .map(|triple| triple.weight * Expression::from_other_affine(triple.o))
        .sum();
    if !heights.is_empty() {
        let penalty: Expression = heights
            .iter()
            .map(|height| Expression::from_other_affine(height.level))
            .sum();
        objective -= global_weight * penalty;
    }
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

// ---------------------------------------------------------------------------
// Exact enumeration for small instances
// ---------------------------------------------------------------------------

/// Largest number of instances solved by enumeration instead of the ILP.
///
/// Every extra instance multiplies the number of coarsening steps to
/// enumerate, so the enumeration is limited to the sizes where it stays in the
/// microsecond range; larger problems keep the exact ILP.
const EXACT_INSTANCE_LIMIT: usize = 6;

/// Enumeration nodes after which the instance is handed to the ILP solver, so
/// that a pathological small instance can never be slower than before.
const EXACT_NODE_LIMIT: usize = 100_000;

/// Solve the merge-level problem by enumerating level-ordered dendrograms.
///
/// The ILP levels are an ultrametric `M(a, c) <= max(M(a, b), M(b, c))` on
/// `1..=levels`, i.e. a chain of nested partitions in which a pair shares a
/// block from its merge level onwards. Only the *order* of the distinct levels
/// enters the triple term, and the height penalty `global_weight * sum max M`
/// only shrinks when levels move down, so an optimum merges at the consecutive
/// levels `1, 2, 3, ...`. What remains is every sequence of at most `levels`
/// proper coarsenings; the search scores each one and keeps the best.
///
/// Returns `None` when the instance is too large or the search exceeds
/// [`EXACT_NODE_LIMIT`]; the caller then solves the ILP.
fn enumerate_hierarchy(
    distance: &DistanceMatrix,
    levels: usize,
    groups: &[Vec<usize>],
    global_weight: f64,
) -> Option<Hierarchy> {
    let n = distance.len();
    if n > EXACT_INSTANCE_LIMIT {
        return None;
    }
    let mut search = Enumeration {
        distance,
        levels,
        groups,
        global_weight,
        nodes: 0,
        best_score: f64::NEG_INFINITY,
        best: None,
        merge: vec![vec![0usize; n]; n],
    };
    let mut blocks: Vec<Vec<usize>> = (0..n).map(|instance| vec![instance]).collect();
    search.visit(&mut blocks, 0)?;
    let merge = search.best?;
    let mut hierarchy = Hierarchy::empty(n, levels);
    for a in 0..n {
        for b in a + 1..n {
            // A pair that no coarsening joined sits at the last level, which is
            // also how the ILP completes a partial hierarchy.
            let level = match merge[a][b] {
                0 => levels,
                value => value,
            };
            hierarchy.set_merge(a, b, level);
        }
    }
    Some(hierarchy)
}

/// Depth-first search over the coarsening chains of [`enumerate_hierarchy`].
struct Enumeration<'a> {
    distance: &'a DistanceMatrix,
    levels: usize,
    groups: &'a [Vec<usize>],
    global_weight: f64,
    /// Number of visited nodes, for [`EXACT_NODE_LIMIT`].
    nodes: usize,
    best_score: f64,
    /// Level matrix of the best chain so far (`0` marks an unmerged pair).
    best: Option<Vec<Vec<usize>>>,
    /// Level matrix of the current branch.
    merge: Vec<Vec<usize>>,
}

impl Enumeration<'_> {
    /// Evaluate the current chain and recurse into every proper coarsening.
    ///
    /// Returns `None` once the node budget is exhausted.
    fn visit(&mut self, blocks: &mut Vec<Vec<usize>>, level: usize) -> Option<()> {
        self.nodes += 1;
        if self.nodes > EXACT_NODE_LIMIT {
            return None;
        }
        self.record();

        let count = blocks.len();
        if level == self.levels || count < 2 {
            return Some(());
        }

        let mut next_blocks: Vec<Vec<usize>> = Vec::with_capacity(count);
        let mut merged: Vec<(usize, usize)> = Vec::new();
        // Every partition of the current blocks merges the blocks of each of
        // its groups at the next level; the identity partition is skipped
        // because finishing here scores the same and uses fewer levels.
        for_each_partition(count, &mut |labels, group_count| {
            if group_count == count {
                return Some(());
            }
            next_blocks.clear();
            merged.clear();
            for group in 0..group_count {
                let mut block: Vec<usize> = Vec::new();
                let mut members = 0usize;
                for (index, &label) in labels.iter().enumerate() {
                    if label == group {
                        block.extend_from_slice(&blocks[index]);
                        members += 1;
                    }
                }
                if members >= 2 {
                    for first in 0..block.len() {
                        for second in first + 1..block.len() {
                            let (low, high) = if block[first] < block[second] {
                                (block[first], block[second])
                            } else {
                                (block[second], block[first])
                            };
                            // Pairs inside one of the merged blocks already
                            // have an earlier level; only the pairs that cross
                            // two of them merge now.
                            if self.merge[low][high] == 0 {
                                merged.push((low, high));
                            }
                        }
                    }
                }
                next_blocks.push(block);
            }
            for &(low, high) in &merged {
                self.merge[low][high] = level + 1;
            }
            let outcome = self.visit(&mut next_blocks, level + 1);
            for &(low, high) in &merged {
                self.merge[low][high] = 0;
            }
            outcome
        })
    }

    /// Score the current chain and keep it if it is the best one so far.
    fn record(&mut self) {
        let score = self.score();
        if self.best.is_none() || score > self.best_score {
            self.best_score = score;
            self.best = Some(self.merge.clone());
        }
    }

    /// The objective of Equation (1) plus the height penalty, evaluated on the
    /// current chain. Pairs that never merge sit at the last level.
    fn score(&self) -> f64 {
        let n = self.distance.len();
        let levels = self.levels;
        let level = |a: usize, b: usize| -> usize {
            if a == b {
                return 0;
            }
            let (low, high) = if a < b { (a, b) } else { (b, a) };
            match self.merge[low][high] {
                0 => levels,
                value => value,
            }
        };

        let mut score = 0.0;
        for a in 0..n {
            for b in 0..n {
                if b == a {
                    continue;
                }
                let merged_ab = level(a, b);
                for c in 0..n {
                    if c == a || c == b {
                        continue;
                    }
                    if merged_ab < level(a, c) {
                        score += self.distance.get(a, c) - self.distance.get(a, b);
                    }
                }
            }
        }
        if self.global_weight > 0.0 {
            let mut members: Vec<usize> = Vec::new();
            for group in self.groups {
                members.clear();
                members.extend_from_slice(group);
                members.sort_unstable();
                members.dedup();
                if members.len() < 2 {
                    continue;
                }
                let mut height = 0usize;
                for (index, &a) in members.iter().enumerate() {
                    for &b in &members[index + 1..] {
                        height = height.max(level(a, b));
                    }
                }
                score -= self.global_weight * height as f64;
            }
        }
        score
    }
}

/// Visit every set partition of `0..count` as a restricted growth string.
///
/// `visit` receives the label of every element and the number of groups; it
/// returns `None` to stop the enumeration.
fn for_each_partition(
    count: usize,
    visit: &mut impl FnMut(&[usize], usize) -> Option<()>,
) -> Option<()> {
    fn recurse(
        count: usize,
        labels: &mut Vec<usize>,
        index: usize,
        groups: usize,
        visit: &mut impl FnMut(&[usize], usize) -> Option<()>,
    ) -> Option<()> {
        if index == count {
            return visit(labels, groups);
        }
        for label in 0..=groups {
            labels[index] = label;
            let next_groups = if label == groups { groups + 1 } else { groups };
            recurse(count, labels, index + 1, next_groups, visit)?;
        }
        Some(())
    }

    if count == 0 {
        return Some(());
    }
    let mut labels = vec![0usize; count];
    recurse(count, &mut labels, 1, 1, visit)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random distance matrix.
    fn random_distance(n: usize, seed: u64) -> DistanceMatrix {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1000) as f64 / 1000.0
        };
        let mut rows = vec![vec![0.0; n]; n];
        for a in 0..n {
            for b in a + 1..n {
                let value = next();
                rows[a][b] = value;
                rows[b][a] = value;
            }
        }
        DistanceMatrix::new(&rows).expect("the matrix is symmetric")
    }

    fn objective(
        hierarchy: &Hierarchy,
        distance: &DistanceMatrix,
        groups: &[Vec<usize>],
        weight: f64,
    ) -> f64 {
        hierarchy.clustering_score(distance) - weight * hierarchy.constraint_cost(groups)
    }

    #[test]
    fn enumeration_finds_the_ilp_optimum() {
        for n in 2..=4usize {
            for seed in 1..=3u64 {
                let distance = random_distance(n, seed * 7919 + n as u64);
                for levels in 1..=n {
                    for groups in [
                        Vec::new(),
                        vec![vec![0, n - 1]],
                        vec![vec![0, 1], vec![0, n - 1]],
                    ] {
                        let groups: Vec<Vec<usize>> = groups
                            .into_iter()
                            .filter(|group| group[0] != group[1])
                            .collect();
                        for weight in [0.0, 4.0] {
                            let enumerated =
                                enumerate_hierarchy(&distance, levels, &groups, weight)
                                    .expect("small instances are enumerated");
                            let solved = solve_ilp(&distance, levels, &groups, weight)
                                .expect("the ILP solves every small instance");
                            let enumerated_score =
                                objective(&enumerated, &distance, &groups, weight);
                            let solved_score = objective(&solved, &distance, &groups, weight);
                            assert!(
                                (enumerated_score - solved_score).abs() < 1e-9,
                                "n={n} levels={levels} groups={groups:?} weight={weight}: \
                                 enumeration {enumerated_score} != ILP {solved_score}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn enumeration_respects_the_level_order() {
        let distance = random_distance(4, 42);
        let hierarchy = enumerate_hierarchy(&distance, 4, &[], 0.0).expect("enumerated");
        assert!(hierarchy.is_complete());
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let ac = hierarchy.merge_level(a, c).expect("complete");
                    let ab = hierarchy.merge_level(a, b).expect("complete");
                    let bc = hierarchy.merge_level(b, c).expect("complete");
                    assert!(ac <= ab.max(bc), "the merge levels must be an ultrametric");
                }
            }
        }
    }
}
