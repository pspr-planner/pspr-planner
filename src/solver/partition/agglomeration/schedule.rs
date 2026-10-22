//! The evaluation-scheduling ILP (constraints 1-4).

use std::collections::{BTreeSet, VecDeque};

use anyhow::{Result, ensure};
use good_lp::{
    Constraint, Expression, ProblemVariables, Solution, SolverModel, Variable, default_solver,
    variable,
};

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

/// Solve the scheduling ILP (1)-(4) for a dendrogram and additional DAG
/// precedences, minimising the sum of the evaluation positions.
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

    let mut variables = ProblemVariables::new();
    let mut x: Vec<Vec<Variable>> = Vec::with_capacity(n);
    for _ in 0..n {
        x.push((0..n).map(|_| variables.add(variable().binary())).collect());
    }
    let position = |variable_row: &[Variable]| -> Expression {
        variable_row
            .iter()
            .enumerate()
            .map(|(time, &variable)| (time + 1) as f64 * variable)
            .sum()
    };

    let mut constraints: Vec<Constraint> = Vec::new();
    // (1) Every internal node is evaluated exactly once.
    for row in &x {
        constraints.push(
            row.iter()
                .map(|&variable| Expression::from_other_affine(variable))
                .sum::<Expression>()
                .eq(1.0),
        );
    }
    // (2) Exactly one evaluation operation per time step.
    for time in 0..n {
        constraints.push(
            x.iter()
                .map(|row| Expression::from_other_affine(row[time]))
                .sum::<Expression>()
                .eq(1.0),
        );
    }
    // (3) and (4) Precedences from the tree and the DAG.
    for &(before, after) in &precedences {
        constraints.push((position(&x[before]) - position(&x[after])).leq(-1.0));
    }

    let objective: Expression = x.iter().map(|row| position(row)).sum();
    let mut model = variables.minimise(objective).using(default_solver);
    for constraint in constraints {
        model.add_constraint(constraint);
    }
    let solution = model
        .solve()
        .map_err(|error| anyhow::anyhow!("the scheduling ILP failed: {error}"))?;

    let mut positions = vec![0usize; n];
    for (node, row) in x.iter().enumerate() {
        let mut position = 0usize;
        for (time, &variable) in row.iter().enumerate() {
            if solution.value(variable) > 0.5 {
                position = time + 1;
            }
        }
        ensure!(
            position != 0,
            "the solver did not schedule internal node {node}"
        );
        positions[node] = position;
    }
    for &(before, after) in &precedences {
        ensure!(
            positions[before] < positions[after],
            "the solver violated the precedence ({before}, {after})"
        );
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&node| (positions[node], node));
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
