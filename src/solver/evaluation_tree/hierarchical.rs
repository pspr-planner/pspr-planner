//! Hierarchical refinement of PSPR's top-level task-network plan.
//!
//! The evaluation tree plans the initial task network with compound tasks as
//! opaque operators. This module adds the two-phase refinement stage (C) and
//! the recursive evaluation stage (D) from `docs/compound-tasks.md`: it takes
//! the abstract top-level plan and recursively grounds methods and primitive
//! actions, producing a complete HTN plan.
//!
//! The refinement is intentionally conservative: it tries the method recorded
//! in the abstract plan first, then every other method, and backtracks over
//! partial-order subtasks. A recursive mode uses the existing evaluation tree
//! on each method-local subproblem, so the partitioner is reused at every
//! hierarchy level.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Result, bail, ensure};

use crate::parsing::json::{
    Constraint, Htn, InitialTaskNetwork, Predicate, Subtask, Symbol, TaskOrdering,
};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::formula::Universe;
use super::graph::{TaskNode, ground_methods};
use super::leaf::initial_atoms;
use super::verify::{GoalStatus, Step, apply_action, holds, parse_step};
use super::{EvaluationOptions, SgplanMerger, evaluate};

/// Options for hierarchical refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefinementOptions {
    /// Maximum number of method groundings considered for one task.
    pub max_groundings: usize,
    /// Maximum recursion depth through methods.
    pub max_depth: usize,
}

impl Default for RefinementOptions {
    fn default() -> Self {
        let evaluation = EvaluationOptions::default();
        RefinementOptions {
            max_groundings: evaluation.max_groundings,
            max_depth: 32,
        }
    }
}

/// One node of a complete hierarchical plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementStep {
    /// A primitive action.
    Action { task: TaskNode, step: String },
    /// A method instance and the refined plans of its subtasks.
    Method {
        task: TaskNode,
        method: String,
        bindings: BTreeMap<String, String>,
        subtasks: Vec<RefinementStep>,
    },
}

impl RefinementStep {
    /// Append the primitive action strings of this subtree to `out`.
    pub fn collect_actions(&self, out: &mut Vec<String>) {
        match self {
            RefinementStep::Action { step, .. } => out.push(step.clone()),
            RefinementStep::Method { subtasks, .. } => {
                for subtask in subtasks {
                    subtask.collect_actions(out);
                }
            }
        }
    }

    /// Number of method instances in this subtree.
    pub fn method_count(&self) -> usize {
        match self {
            RefinementStep::Action { .. } => 0,
            RefinementStep::Method { subtasks, .. } => {
                1 + subtasks
                    .iter()
                    .map(RefinementStep::method_count)
                    .sum::<usize>()
            }
        }
    }
}

/// A complete refinement of an abstract task-network plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refinement {
    /// The abstract plan produced by the evaluation tree.
    pub abstract_plan: Vec<String>,
    /// The primitive plan, in execution order.
    pub plan: Vec<String>,
    /// The decomposition tree, including method steps.
    pub steps: Vec<RefinementStep>,
    /// State after the last primitive action.
    pub final_state: BTreeSet<String>,
    /// Goal status of the final state.
    pub goal: GoalStatus,
}

impl Refinement {
    /// Number of method applications in the decomposition.
    pub fn method_count(&self) -> usize {
        self.steps.iter().map(RefinementStep::method_count).sum()
    }
}

struct RefinedTask {
    steps: Vec<RefinementStep>,
    state: BTreeSet<String>,
}

/// Refine an abstract plan with the linear (two-phase) strategy.
pub fn refine_plan(
    domain: &Domain,
    problem: &Problem,
    abstract_plan: &[String],
    options: &RefinementOptions,
) -> Result<Refinement> {
    refine_plan_with_mode(domain, problem, abstract_plan, options, false)
}

/// Refine an abstract plan, optionally using recursive evaluation for compound
/// subtasks (the recursive-tree strategy).
pub fn refine_plan_with_mode(
    domain: &Domain,
    problem: &Problem,
    abstract_plan: &[String],
    options: &RefinementOptions,
    recursive: bool,
) -> Result<Refinement> {
    let universe = Universe::new(domain, problem);
    let state = initial_atoms(problem);
    let candidates = refine_plan_steps(
        domain,
        problem,
        &universe,
        abstract_plan,
        state,
        options,
        options.max_depth,
        recursive,
    )?;
    if candidates.is_empty() {
        bail!("the abstract plan could not be refined into a complete HTN plan");
    }

    let mut fallback = None;
    for refined in candidates {
        let mut plan = Vec::new();
        for step in &refined.steps {
            step.collect_actions(&mut plan);
        }
        let goal = goal_status(domain, problem, &refined.state, options.max_groundings)?;
        let refinement = Refinement {
            abstract_plan: abstract_plan.to_vec(),
            plan,
            steps: refined.steps,
            final_state: refined.state,
            goal,
        };
        if goal != GoalStatus::Violated {
            return Ok(refinement);
        }
        fallback.get_or_insert(refinement);
    }
    Ok(fallback.expect("at least one candidate was checked"))
}

/// Build and evaluate the initial abstract plan, then refine it.
pub fn solve_hierarchical(
    domain: &Domain,
    problem: &Problem,
    options: &RefinementOptions,
    evaluation_options: &EvaluationOptions,
) -> Result<Refinement> {
    let mut merger = SgplanMerger::default();
    let evaluation = evaluate(domain, problem, &mut merger, evaluation_options)?;
    ensure!(
        evaluation.solved,
        "the evaluation tree did not solve the initial task network"
    );
    let refinement = refine_plan(domain, problem, &evaluation.plan, options)?;
    ensure!(
        problem.goal.is_none() || refinement.goal != GoalStatus::Violated,
        "the refinement does not reach the problem goal"
    );
    Ok(refinement)
}

/// Build the initial abstract plan and refine it with recursive evaluation of
/// method-local subproblems.
pub fn solve_recursive(
    domain: &Domain,
    problem: &Problem,
    options: &RefinementOptions,
    evaluation_options: &EvaluationOptions,
) -> Result<Refinement> {
    let mut merger = SgplanMerger::default();
    let evaluation = evaluate(domain, problem, &mut merger, evaluation_options)?;
    ensure!(
        evaluation.solved,
        "the evaluation tree did not solve the initial task network"
    );

    // The linear refinement is the safe path when a method's compound
    // subtasks feed effects into later subtasks, because the flat evaluation
    // of a method-local subproblem cannot see those effects. Use it first and
    // reserve the recursive subproblem evaluation for cases where it fails.
    if let Ok(refinement) = refine_plan(domain, problem, &evaluation.plan, options)
        && (problem.goal.is_none() || refinement.goal != GoalStatus::Violated)
    {
        return Ok(refinement);
    }

    let refinement = refine_plan_with_mode(domain, problem, &evaluation.plan, options, true)?;
    ensure!(
        problem.goal.is_none() || refinement.goal != GoalStatus::Violated,
        "the recursive refinement does not reach the problem goal"
    );
    Ok(refinement)
}

const MAX_REFINEMENTS: usize = 64;

fn cap(mut candidates: Vec<RefinedTask>) -> Vec<RefinedTask> {
    candidates.truncate(MAX_REFINEMENTS);
    candidates
}

#[allow(clippy::too_many_arguments)]
fn refine_plan_steps(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    abstract_plan: &[String],
    state: BTreeSet<String>,
    options: &RefinementOptions,
    depth: usize,
    recursive: bool,
) -> Result<Vec<RefinedTask>> {
    let mut candidates = vec![RefinedTask {
        steps: Vec::new(),
        state,
    }];
    for step in abstract_plan {
        let parsed = parse_step(step)?;
        let task = parsed.task().clone();
        let preferred = match &parsed {
            Step::Method { name, .. } => Some(name.as_str()),
            Step::Action(_) | Step::Compound(_) => None,
        };
        if matches!(parsed, Step::Compound(_)) {
            bail!("the abstract plan contains an opaque compound step `{step}`");
        }

        let candidate = candidates.pop().expect("one candidate before a step");
        let alternatives = refine_task(
            domain,
            problem,
            universe,
            &task,
            &candidate.state,
            options,
            depth,
            preferred,
            recursive,
        )?;
        let Some(refined) = alternatives.into_iter().next() else {
            return Ok(Vec::new());
        };
        let mut steps = candidate.steps;
        steps.extend(refined.steps);
        candidates = vec![RefinedTask {
            steps,
            state: refined.state,
        }];
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
fn refine_task(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    task: &TaskNode,
    state: &BTreeSet<String>,
    options: &RefinementOptions,
    depth: usize,
    preferred_method: Option<&str>,
    recursive: bool,
) -> Result<Vec<RefinedTask>> {
    if depth == 0 {
        return Ok(Vec::new());
    }

    // A task that names an action is tried as a primitive first. Some HDDL
    // domains use the same symbol for a task and an action; the method branch
    // below remains available if the action is not applicable.
    if domain.actions_for(&task.name).next().is_some()
        && let Ok(next) = apply_action(domain, task, state, universe, options.max_groundings)
    {
        return Ok(vec![RefinedTask {
            steps: vec![RefinementStep::Action {
                task: task.clone(),
                step: task.to_string(),
            }],
            state: next,
        }]);
    }

    let mut methods = ground_methods(domain, universe, task, options.max_groundings)?;
    methods.sort_by_key(|instance| {
        (
            preferred_method.is_some_and(|preferred| instance.method.name.name != preferred),
            instance.method.tn.subtasks.len(),
        )
    });
    for instance in methods {
        if !holds_optional(
            instance.method.precondition.as_ref(),
            &instance.bindings,
            state,
            universe,
            options.max_groundings,
        )? {
            continue;
        }

        let refined = if recursive {
            refine_method_recursive(
                domain,
                problem,
                instance.method,
                &instance.bindings,
                state,
                options,
                depth - 1,
            )?
        } else {
            refine_network(
                domain,
                problem,
                universe,
                &instance.method.tn,
                &instance.bindings,
                state,
                options,
                depth - 1,
            )?
        };

        let Some(refined) = refined.into_iter().next() else {
            continue;
        };
        return Ok(vec![RefinedTask {
            steps: vec![RefinementStep::Method {
                task: task.clone(),
                method: instance.method.name.name.clone(),
                bindings: instance.bindings.clone(),
                subtasks: refined.steps,
            }],
            state: refined.state,
        }]);
    }

    Ok(Vec::new())
}

#[allow(clippy::too_many_arguments)]
fn refine_method_recursive(
    domain: &Domain,
    problem: &Problem,
    method: &crate::parsing::json::Method,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    options: &RefinementOptions,
    depth: usize,
) -> Result<Vec<RefinedTask>> {
    let primitive_only = !method.tn.subtasks.is_empty()
        && method.tn.subtasks.iter().all(|subtask| {
            domain.actions_for(&subtask.task.name).next().is_some()
                && domain.methods_for(&subtask.task.name).next().is_none()
        });
    if !primitive_only {
        let universe = Universe::new(domain, problem);
        let mut refined = refine_network(
            domain, problem, &universe, &method.tn, bindings, state, options, depth,
        )?;
        refined.truncate(1);
        return Ok(refined);
    }

    let subproblem = method_subproblem(problem, method, bindings, state);
    let mut merger = SgplanMerger::default();
    if let Ok(evaluation) = evaluate(
        domain,
        &subproblem,
        &mut merger,
        &EvaluationOptions {
            max_groundings: options.max_groundings,
            ..EvaluationOptions::default()
        },
    ) && evaluation.solved
    {
        let sub_universe = Universe::new(domain, &subproblem);
        let refined = refine_plan_steps(
            domain,
            &subproblem,
            &sub_universe,
            &evaluation.plan,
            state.clone(),
            options,
            depth,
            true,
        )?;
        if !refined.is_empty() {
            return Ok(refined);
        }
    }

    // Some method networks contain compound subtasks whose effects are needed
    // by later subtasks. The flat evaluation of the subproblem cannot see
    // those effects, so fall back to the linear recursive refinement for this
    // method.
    let universe = Universe::new(domain, problem);
    let mut refined = refine_network(
        domain, problem, &universe, &method.tn, bindings, state, options, depth,
    )?;
    refined.truncate(1);
    Ok(refined)
}

#[allow(clippy::too_many_arguments)]
fn refine_network(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    network: &Htn,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    options: &RefinementOptions,
    depth: usize,
) -> Result<Vec<RefinedTask>> {
    let tasks: Vec<TaskNode> = network
        .subtasks
        .iter()
        .map(|subtask| ground_subtask(subtask, bindings))
        .collect();
    let ids: Vec<String> = network
        .subtasks
        .iter()
        .enumerate()
        .map(|(index, subtask)| subtask_id(subtask, index))
        .collect();

    if !constraints_hold(network.constraints.as_deref(), bindings) {
        return Ok(Vec::new());
    }

    match &network.orderings {
        TaskOrdering::Total => {
            let mut candidates = vec![RefinedTask {
                steps: Vec::new(),
                state: state.clone(),
            }];
            for task in &tasks {
                let candidate = candidates.pop().expect("one candidate before a subtask");
                let alternatives = refine_task(
                    domain,
                    problem,
                    universe,
                    task,
                    &candidate.state,
                    options,
                    depth,
                    None,
                    false,
                )?;
                let Some(refined) = alternatives.into_iter().next() else {
                    return Ok(Vec::new());
                };
                let mut steps = candidate.steps;
                steps.extend(refined.steps);
                candidates = vec![RefinedTask {
                    steps,
                    state: refined.state,
                }];
            }
            Ok(candidates)
        }
        TaskOrdering::Partial(edges) => {
            let mut predecessors = vec![HashSet::new(); tasks.len()];
            let index: HashMap<&str, usize> = ids
                .iter()
                .enumerate()
                .map(|(index, id)| (id.as_str(), index))
                .collect();
            for (before, after) in edges {
                let Some(&before) = index.get(before.as_str()) else {
                    bail!("ordering refers to unknown subtask id `{before}`");
                };
                let Some(&after) = index.get(after.as_str()) else {
                    bail!("ordering refers to unknown subtask id `{after}`");
                };
                predecessors[after].insert(before);
            }
            search_partial(
                domain,
                problem,
                universe,
                &tasks,
                &predecessors,
                state,
                options,
                depth,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn search_partial(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    tasks: &[TaskNode],
    predecessors: &[HashSet<usize>],
    state: &BTreeSet<String>,
    options: &RefinementOptions,
    depth: usize,
) -> Result<Vec<RefinedTask>> {
    let mut remaining = vec![true; tasks.len()];
    let mut results = Vec::new();
    search_partial_inner(
        domain,
        problem,
        universe,
        tasks,
        predecessors,
        &mut remaining,
        state,
        Vec::new(),
        options,
        depth,
        &mut results,
    )?;
    Ok(cap(results))
}

#[allow(clippy::too_many_arguments)]
fn search_partial_inner(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    tasks: &[TaskNode],
    predecessors: &[HashSet<usize>],
    remaining: &mut [bool],
    state: &BTreeSet<String>,
    steps: Vec<RefinementStep>,
    options: &RefinementOptions,
    depth: usize,
    results: &mut Vec<RefinedTask>,
) -> Result<()> {
    if !remaining.iter().any(|&present| present) {
        results.push(RefinedTask {
            steps,
            state: state.clone(),
        });
        return Ok(());
    }

    for index in 0..tasks.len() {
        if !remaining[index] {
            continue;
        }
        if predecessors[index].iter().any(|&before| remaining[before]) {
            continue;
        }
        let alternatives = refine_task(
            domain,
            problem,
            universe,
            &tasks[index],
            state,
            options,
            depth,
            None,
            false,
        )?;
        for refined in alternatives {
            remaining[index] = false;
            let mut next_steps = steps.clone();
            next_steps.extend(refined.steps);
            search_partial_inner(
                domain,
                problem,
                universe,
                tasks,
                predecessors,
                remaining,
                &refined.state,
                next_steps,
                options,
                depth,
                results,
            )?;
            remaining[index] = true;
            if !results.is_empty() {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn method_subproblem(
    problem: &Problem,
    method: &crate::parsing::json::Method,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
) -> Problem {
    let subtasks = method
        .tn
        .subtasks
        .iter()
        .map(|subtask| Subtask {
            id: subtask.id.clone(),
            task: subtask.task.clone(),
            terms: subtask
                .terms
                .iter()
                .map(|term| Symbol {
                    name: bindings
                        .get(&term.name)
                        .cloned()
                        .unwrap_or_else(|| term.name.clone()),
                    symbol_type: term.symbol_type.clone(),
                })
                .collect(),
        })
        .collect();
    Problem {
        name: format!("{}::{}", problem.name, method.name.name),
        domain_name: problem.domain_name.clone(),
        objects: problem.objects.clone(),
        init_state: state_to_predicates(state),
        goal: None,
        init_task_network: Some(InitialTaskNetwork {
            parameters: None,
            tn: Htn {
                subtasks,
                orderings: method.tn.orderings.clone(),
                constraints: method.tn.constraints.clone(),
            },
        }),
    }
}

fn state_to_predicates(state: &BTreeSet<String>) -> Vec<Predicate> {
    state
        .iter()
        .map(|atom| {
            if let Some(open) = atom.find('(') {
                let name = atom[..open].to_string();
                let terms = atom[open + 1..atom.len() - 1]
                    .split(',')
                    .map(str::trim)
                    .filter(|term| !term.is_empty())
                    .map(|term| Symbol {
                        name: term.to_string(),
                        symbol_type: None,
                    })
                    .collect();
                Predicate {
                    name,
                    variables: terms,
                }
            } else {
                Predicate {
                    name: atom.clone(),
                    variables: Vec::new(),
                }
            }
        })
        .collect()
}

fn ground_subtask(subtask: &Subtask, bindings: &BTreeMap<String, String>) -> TaskNode {
    TaskNode {
        name: subtask.task.name.clone(),
        terms: subtask
            .terms
            .iter()
            .map(|term| {
                bindings
                    .get(&term.name)
                    .cloned()
                    .unwrap_or_else(|| term.name.clone())
            })
            .collect(),
    }
}

fn subtask_id(subtask: &Subtask, index: usize) -> String {
    subtask
        .id
        .as_ref()
        .map(|id| id.name.clone())
        .unwrap_or_else(|| index.to_string())
}

fn constraints_hold(
    constraints: Option<&[Constraint]>,
    bindings: &BTreeMap<String, String>,
) -> bool {
    let Some(constraints) = constraints else {
        return true;
    };
    constraints.iter().all(|constraint| match constraint {
        Constraint::Equal(left, right) => resolve(left, bindings) == resolve(right, bindings),
        Constraint::NotEqual(left, right) => resolve(left, bindings) != resolve(right, bindings),
    })
}

fn resolve<'a>(term: &'a str, bindings: &'a BTreeMap<String, String>) -> &'a str {
    bindings.get(term).map(String::as_str).unwrap_or(term)
}

fn holds_optional(
    formula: Option<&crate::parsing::json::Formula>,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<bool> {
    match formula {
        Some(formula) => holds(formula, bindings, state, universe, budget),
        None => Ok(true),
    }
}

fn goal_status(
    domain: &Domain,
    problem: &Problem,
    state: &BTreeSet<String>,
    budget: usize,
) -> Result<GoalStatus> {
    let universe = Universe::new(domain, problem);
    match &problem.goal {
        None => Ok(GoalStatus::Absent),
        Some(goal) => {
            if holds(goal, &BTreeMap::new(), state, &universe, budget)? {
                Ok(GoalStatus::Satisfied)
            } else {
                Ok(GoalStatus::Violated)
            }
        }
    }
}

/// Strictly re-execute a refinement tree.
///
/// Unlike [`super::validate_plan`], this checks every method precondition and
/// every subtask recursively, and checks that the final state and goal match
/// the refinement's report.
pub fn validate_refinement(
    domain: &Domain,
    problem: &Problem,
    refinement: &Refinement,
    options: &RefinementOptions,
) -> Result<super::ValidationReport> {
    let universe = Universe::new(domain, problem);
    let state = initial_atoms(problem);
    let final_state = validate_steps(
        domain,
        &universe,
        &refinement.steps,
        state,
        options.max_groundings,
    )?;
    ensure!(
        final_state == refinement.final_state,
        "the refinement's final state does not match its execution"
    );
    let mut plan = Vec::new();
    for step in &refinement.steps {
        step.collect_actions(&mut plan);
    }
    ensure!(
        plan == refinement.plan,
        "the refinement's plan does not match its steps"
    );
    let goal = goal_status(domain, problem, &final_state, options.max_groundings)?;
    ensure!(
        goal == refinement.goal,
        "the refinement's goal status does not match its execution"
    );
    let graph = super::build_task_constraint_graph(domain, problem, options.max_groundings)?;
    Ok(super::ValidationReport {
        steps: plan.len(),
        tasks: graph.tasks.len(),
        final_state,
        goal,
    })
}

fn validate_steps(
    domain: &Domain,
    universe: &Universe<'_>,
    steps: &[RefinementStep],
    state: BTreeSet<String>,
    budget: usize,
) -> Result<BTreeSet<String>> {
    let mut state = state;
    for step in steps {
        match step {
            RefinementStep::Action { task, .. } => {
                state = apply_action(domain, task, &state, universe, budget)?;
            }
            RefinementStep::Method {
                task,
                method,
                bindings,
                subtasks,
            } => {
                let Some(schema) = domain
                    .methods_for(&task.name)
                    .filter(|schema| schema.task_terms.len() == task.terms.len())
                    .find(|schema| schema.name.name == *method)
                else {
                    bail!("no method `{method}` decomposes `{task}`");
                };
                ensure!(
                    holds_optional(
                        schema.precondition.as_ref(),
                        bindings,
                        &state,
                        universe,
                        budget,
                    )?,
                    "the precondition of method `{method}` does not hold for `{task}`"
                );
                state = validate_steps(domain, universe, subtasks, state, budget)?;
            }
        }
    }
    Ok(state)
}
