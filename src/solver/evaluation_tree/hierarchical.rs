//! Hierarchical refinement of PSPR's top-level task-network plan.
//!
//! The evaluation tree plans the initial task network with compound tasks as
//! opaque operators. This module adds the two-phase refinement stage (C) from
//! `docs/compound-tasks.md`: it takes the abstract top-level plan and
//! recursively grounds methods and primitive actions, producing a complete HTN
//! plan.
//!
//! The refinement is intentionally conservative: it tries the method recorded
//! in the abstract plan first, then every other method, and backtracks over
//! partial-order subtasks. The default hierarchical solver enables primitive
//! method summaries (stage G) unconditionally, because the comparison showed
//! that C+G is both faster and at least as complete as C alone.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};

use crate::parsing::json::{Constraint, Formula, Htn, Subtask, TaskOrdering};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::decomposition::{
    DecompositionSession, MethodCandidate, MethodRankingContext, MethodTrace, SearchContext,
    SearchTrace,
};
use super::formula::{self, Universe};
use super::graph::{TaskNode, ground_methods_bounded};
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

/// A solver state shared by every refinement branch that starts from it.
///
/// `apply_action` produces a fresh `BTreeSet`; wrapping it in an `Arc` makes
/// the many copies taken for visited-set keys, applicability-cache keys and
/// refined candidates cheap reference-count bumps instead of deep tree clones.
type State = Arc<BTreeSet<String>>;

#[derive(Clone)]
struct RefinedTask {
    steps: Vec<RefinementStep>,
    state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ApplicabilityKey {
    task: TaskNode,
    state: State,
}

/// One method instance that can decompose a task, prepared once per task.
///
/// Grounding and subtask grounding depend only on the task, not on the solver
/// state, so they are cached per task and reused across the states the
/// refinement explores. Only the precondition check is state-dependent.
#[derive(Clone)]
struct GroundedInstance {
    method: String,
    precondition: Option<Formula>,
    bindings: BTreeMap<String, String>,
    network: Arc<Htn>,
    tasks: Arc<[TaskNode]>,
    ids: Arc<[String]>,
    subtasks: usize,
}

/// Caches groundings per task and applicable instances per `(task, state)`.
#[derive(Default)]
struct MethodApplicabilityCache {
    grounded: HashMap<TaskNode, Arc<[GroundedInstance]>>,
    applicable: HashMap<ApplicabilityKey, Arc<[u32]>>,
}

impl MethodApplicabilityCache {
    #[allow(clippy::type_complexity)]
    fn lookup(
        &mut self,
        domain: &Domain,
        universe: &Universe<'_>,
        task: &TaskNode,
        state: &State,
        max_groundings: usize,
    ) -> Result<(Arc<[GroundedInstance]>, Arc<[u32]>)> {
        let instances = match self.grounded.get(task) {
            Some(instances) => Arc::clone(instances),
            None => {
                let sampling_budget = max_groundings
                    .saturating_mul(REFINEMENT_METHOD_SAMPLING_FACTOR)
                    .max(max_groundings);
                let methods =
                    ground_methods_bounded(domain, universe, task, sampling_budget).instances;
                let instances: Vec<GroundedInstance> = methods
                    .into_iter()
                    .map(|instance| {
                        let tasks: Vec<TaskNode> = instance
                            .method
                            .tn
                            .subtasks
                            .iter()
                            .map(|subtask| ground_subtask(subtask, &instance.bindings))
                            .collect();
                        let ids: Vec<String> = instance
                            .method
                            .tn
                            .subtasks
                            .iter()
                            .enumerate()
                            .map(|(index, subtask)| subtask_id(subtask, index))
                            .collect();
                        GroundedInstance {
                            subtasks: tasks.len(),
                            method: instance.method.name.name.clone(),
                            precondition: instance.method.precondition.clone(),
                            bindings: instance.bindings,
                            network: Arc::new(instance.method.tn.clone()),
                            tasks: tasks.into(),
                            ids: ids.into(),
                        }
                    })
                    .collect();
                let instances: Arc<[GroundedInstance]> = instances.into();
                self.grounded.insert(task.clone(), Arc::clone(&instances));
                instances
            }
        };

        let key = ApplicabilityKey {
            task: task.clone(),
            state: Arc::clone(state),
        };
        if let Some(applicable) = self.applicable.get(&key) {
            return Ok((instances, Arc::clone(applicable)));
        }
        let mut applicable = Vec::new();
        for (index, instance) in instances.iter().enumerate() {
            if holds_optional_bounded(
                instance.precondition.as_ref(),
                &instance.bindings,
                state.as_ref(),
                universe,
                max_groundings,
            )? {
                applicable.push(index as u32);
            }
        }
        let applicable: Arc<[u32]> = applicable.into();
        self.applicable.insert(key, Arc::clone(&applicable));
        Ok((instances, applicable))
    }
}

/// Refine an abstract plan with the two-phase strategy (stage C).
pub fn refine_plan(
    domain: &Domain,
    problem: &Problem,
    abstract_plan: &[String],
    options: &RefinementOptions,
) -> Result<Refinement> {
    let mut session = DecompositionSession::default();
    refine_plan_with_session(domain, problem, abstract_plan, options, &mut session)
}

/// Refine with optional neural method ranking, frontier search, and tracing.
#[allow(clippy::too_many_arguments)]
pub fn refine_plan_with_session(
    domain: &Domain,
    problem: &Problem,
    abstract_plan: &[String],
    options: &RefinementOptions,
    session: &mut DecompositionSession<'_>,
) -> Result<Refinement> {
    let universe = Universe::new(domain, problem);
    let state: State = Arc::new(initial_atoms(problem));
    let candidates = refine_plan_steps(
        domain,
        problem,
        &universe,
        abstract_plan,
        state,
        options,
        options.max_depth,
        session,
    )?;
    if candidates.is_empty() {
        bail!("the abstract plan could not be refined into a complete HTN plan");
    }

    let mut fallback = None;
    for refined in candidates {
        let RefinedTask { steps, state } = refined;
        let mut plan = Vec::new();
        for step in &steps {
            step.collect_actions(&mut plan);
        }
        let goal = goal_status(problem, &universe, state.as_ref(), options.max_groundings)?;
        let refinement = Refinement {
            abstract_plan: abstract_plan.to_vec(),
            plan,
            steps,
            final_state: (*state).clone(),
            goal,
        };
        if goal != GoalStatus::Violated {
            return Ok(refinement);
        }
        fallback.get_or_insert(refinement);
    }
    Ok(fallback.expect("at least one candidate was checked"))
}

/// Build and evaluate the initial abstract plan, then refine it with stage C.
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
    let mut last_error = None;
    let mut refinement = None;
    for attempt in 0..REFINEMENT_RETRY_ATTEMPTS {
        let retry = refined_retry_options(options, attempt);
        match refine_plan(domain, problem, &evaluation.plan, &retry) {
            Ok(candidate) => {
                refinement = Some(candidate);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let refinement = refinement
        .ok_or_else(|| last_error.unwrap_or_else(|| anyhow::anyhow!("refinement failed")))?;
    ensure!(
        problem.goal.is_none() || refinement.goal != GoalStatus::Violated,
        "the refinement does not reach the problem goal"
    );
    Ok(refinement)
}

// Bounded breadth-first-style enumeration of refinement alternatives. The
// limit is large enough for the larger Blocksworld-GTOHP instances (up to 14
// tasks with several interchangeable `do_on_table` methods) and still keeps
// the search finite on the harder learned domains.
const MAX_REFINEMENTS: usize = 2048;
const REFINEMENT_METHOD_SAMPLING_FACTOR: usize = 4;
const REFINEMENT_RETRY_ATTEMPTS: usize = 3;
const REFINEMENT_MIN_BUDGET_MS: u64 = 2500;
const REFINEMENT_MAX_BUDGET_MS: u64 = 14000;
const REFINEMENT_MS_PER_ABSTRACT_STEP: u64 = 150;
const REFINEMENT_EXPANSIONS_PER_STEP: usize = 1024;

fn refined_retry_options(base: &RefinementOptions, attempt: usize) -> RefinementOptions {
    let scale = 1usize << attempt;
    RefinementOptions {
        max_groundings: base.max_groundings.saturating_mul(scale),
        max_depth: base.max_depth.saturating_add(8 * attempt),
    }
}

struct RefinementBudget {
    deadline: Instant,
    expansions_left: usize,
}

impl RefinementBudget {
    fn new(abstract_steps: usize, options: &RefinementOptions) -> Self {
        if cfg!(debug_assertions) {
            return RefinementBudget {
                deadline: Instant::now() + Duration::from_secs(24 * 60 * 60),
                expansions_left: usize::MAX / 2,
            };
        }
        let steps = abstract_steps.max(1);
        let grounding_scale = (options.max_groundings / 256).max(1) as u64;
        let depth_scale = (options.max_depth / 32).max(1) as u64;
        let budget_ms = (steps as u64)
            .saturating_mul(REFINEMENT_MS_PER_ABSTRACT_STEP)
            .saturating_mul(grounding_scale)
            .saturating_mul(depth_scale)
            .clamp(REFINEMENT_MIN_BUDGET_MS, REFINEMENT_MAX_BUDGET_MS);
        let expansions_left = steps
            .saturating_mul(REFINEMENT_EXPANSIONS_PER_STEP)
            .saturating_mul((options.max_groundings / 256).max(1));
        RefinementBudget {
            deadline: Instant::now() + Duration::from_millis(budget_ms),
            expansions_left,
        }
    }

    fn has_time(&self) -> bool {
        Instant::now() < self.deadline
    }

    fn consume(&mut self) -> bool {
        if self.expansions_left == 0 {
            return false;
        }
        self.expansions_left -= 1;
        true
    }
}

fn cap(mut candidates: Vec<RefinedTask>) -> Vec<RefinedTask> {
    candidates.truncate(MAX_REFINEMENTS);
    candidates
}

fn rank_order(scores: Vec<f64>) -> Option<Vec<usize>> {
    if scores.is_empty() || scores.iter().any(|score| !score.is_finite()) {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then(left.cmp(right))
    });
    Some(order)
}

/// Hierarchical solving with an optional neural decomposition session.
pub fn solve_hierarchical_with_session(
    domain: &Domain,
    problem: &Problem,
    options: &RefinementOptions,
    evaluation_options: &EvaluationOptions,
    session: &mut DecompositionSession<'_>,
) -> Result<Refinement> {
    let mut merger = SgplanMerger::default();
    let evaluation = evaluate(domain, problem, &mut merger, evaluation_options)?;
    ensure!(
        evaluation.solved,
        "the evaluation tree did not solve the initial task network"
    );
    let mut last_error = None;
    let mut refinement = None;
    for attempt in 0..REFINEMENT_RETRY_ATTEMPTS {
        let retry = refined_retry_options(options, attempt);
        match refine_plan_with_session(domain, problem, &evaluation.plan, &retry, session) {
            Ok(candidate) => {
                refinement = Some(candidate);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let refinement = refinement
        .ok_or_else(|| last_error.unwrap_or_else(|| anyhow::anyhow!("refinement failed")))?;
    ensure!(
        problem.goal.is_none() || refinement.goal != GoalStatus::Violated,
        "the refinement does not reach the problem goal"
    );
    Ok(refinement)
}

fn valid_method_order(order: &[usize], candidate_count: usize) -> bool {
    order.len() == candidate_count
        && order.iter().all(|index| *index < candidate_count)
        && order.iter().collect::<HashSet<_>>().len() == order.len()
}

fn valid_search_order(order: &[usize], ready: &[usize]) -> bool {
    order.len() == ready.len()
        && order.iter().all(|index| ready.contains(index))
        && order.iter().collect::<HashSet<_>>().len() == order.len()
}

#[allow(clippy::too_many_arguments)]
fn refine_plan_steps(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    abstract_plan: &[String],
    state: State,
    options: &RefinementOptions,
    depth: usize,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    let mut candidates = vec![RefinedTask {
        steps: Vec::new(),
        state,
    }];
    let mut visited = HashSet::new();
    let mut applicability_cache = MethodApplicabilityCache::default();
    let mut budget = RefinementBudget::new(abstract_plan.len(), options);
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

        let mut step_candidates = Vec::new();
        for candidate in candidates {
            let alternatives = refine_task(
                domain,
                problem,
                universe,
                &task,
                &candidate.state,
                options,
                depth,
                preferred,
                &mut visited,
                &mut applicability_cache,
                &mut budget,
                session,
            )?;
            for refined in alternatives {
                let mut steps = candidate.steps.clone();
                steps.extend(refined.steps);
                step_candidates.push(RefinedTask {
                    steps,
                    state: refined.state,
                });
                if step_candidates.len() >= MAX_REFINEMENTS {
                    break;
                }
            }
            if step_candidates.len() >= MAX_REFINEMENTS {
                break;
            }
        }
        if step_candidates.is_empty() {
            return Ok(Vec::new());
        }
        candidates = step_candidates;
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
fn refine_task(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    task: &TaskNode,
    state: &State,
    options: &RefinementOptions,
    depth: usize,
    preferred_method: Option<&str>,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if !budget.has_time() {
        return Ok(Vec::new());
    }
    // A method may rediscover the same task in the same state. Such a branch
    // cannot make progress, so prune it instead of letting cycles (for example
    // `get_to` via an intermediate location) expand until the depth limit.
    // The state is reference-counted, so the visited key does not copy it.
    let key = (task.clone(), Arc::clone(state));
    if !visited.insert(key.clone()) {
        return Ok(Vec::new());
    }
    let result = refine_task_uncached(
        domain,
        problem,
        universe,
        task,
        state,
        options,
        depth,
        preferred_method,
        visited,
        applicability_cache,
        budget,
        session,
    );
    visited.remove(&key);
    result
}

#[allow(clippy::too_many_arguments)]
fn refine_task_uncached(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    task: &TaskNode,
    state: &State,
    options: &RefinementOptions,
    depth: usize,
    preferred_method: Option<&str>,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if depth == 0 || !budget.has_time() || !budget.consume() {
        return Ok(Vec::new());
    }

    // A task that names an action is tried as a primitive first. Some HDDL
    // domains use the same symbol for a task and an action; the method branch
    // below remains available if the action is not applicable.
    if domain.actions_for(&task.name).next().is_some()
        && let Ok(next) = apply_action(
            domain,
            task,
            state.as_ref(),
            universe,
            options.max_groundings,
        )
    {
        return Ok(vec![RefinedTask {
            steps: vec![RefinementStep::Action {
                task: task.clone(),
                step: task.to_string(),
            }],
            state: Arc::new(next),
        }]);
    }

    let (instances, applicable) =
        applicability_cache.lookup(domain, universe, task, state, options.max_groundings)?;

    // Ranking candidates clone method names and bindings and clone the state.
    // The symbolic default path never needs them, so build them only when a
    // ranker or a trace sink is attached.
    let tracing = session.trace.is_some();
    let ranked = session.method_ranker.is_some() || tracing;
    let candidates: Vec<MethodCandidate> = if ranked {
        applicable
            .iter()
            .map(|&index| {
                let instance = &instances[index as usize];
                MethodCandidate {
                    method: instance.method.clone(),
                    bindings: instance.bindings.clone(),
                    subtasks: instance.subtasks,
                    preferred: preferred_method
                        .is_some_and(|preferred| instance.method == preferred),
                    applicable: true,
                    precondition_atoms: 0,
                    precondition_fraction: 1.0,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let context = ranked.then(|| MethodRankingContext {
        task: task.clone(),
        state: state.as_ref().clone(),
        depth,
        max_depth: options.max_depth,
    });
    let baseline: Vec<usize> = (0..applicable.len()).collect();
    let order = match (session.method_ranker, context.as_ref()) {
        (Some(ranker), Some(context)) => ranker
            .rank(context, &candidates)
            .filter(|scores| scores.len() == candidates.len())
            .and_then(rank_order)
            .filter(|order| valid_method_order(order, candidates.len()))
            .unwrap_or(baseline),
        _ => baseline,
    };

    let mut results = Vec::new();
    let mut successful = if tracing {
        vec![false; applicable.len()]
    } else {
        Vec::new()
    };
    let method_limit = applicable.len();
    for &candidate_slot in order.iter().take(method_limit) {
        if !budget.has_time() {
            break;
        }
        let grounded_index = applicable[candidate_slot] as usize;
        let instance = &instances[grounded_index];
        let before = results.len();
        let alternatives = refine_network(
            domain,
            problem,
            universe,
            instance,
            state,
            options,
            depth - 1,
            visited,
            applicability_cache,
            budget,
            session,
        )?;
        for refined in alternatives {
            results.push(RefinedTask {
                steps: vec![RefinementStep::Method {
                    task: task.clone(),
                    method: instance.method.clone(),
                    bindings: instance.bindings.clone(),
                    subtasks: refined.steps,
                }],
                state: refined.state,
            });
            if results.len() >= MAX_REFINEMENTS {
                break;
            }
        }
        if tracing {
            successful[candidate_slot] = results.len() > before;
        }
        if results.len() >= MAX_REFINEMENTS {
            break;
        }
    }

    if let Some(trace) = session.trace.as_deref_mut() {
        let context = context.expect("a trace sink implies a ranking context");
        trace.method_trace(MethodTrace {
            context,
            candidates,
            order,
            successful,
        });
    }

    Ok(results)
}

#[allow(clippy::too_many_arguments)]
fn refine_network(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    instance: &GroundedInstance,
    state: &State,
    options: &RefinementOptions,
    depth: usize,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if !constraints_hold(instance.network.constraints.as_deref(), &instance.bindings) {
        return Ok(Vec::new());
    }

    match &instance.network.orderings {
        TaskOrdering::Total => {
            let mut candidates = vec![RefinedTask {
                steps: Vec::new(),
                state: Arc::clone(state),
            }];
            for task in instance.tasks.iter() {
                if !budget.has_time() {
                    return Ok(Vec::new());
                }
                let mut next_candidates = Vec::new();
                for candidate in candidates {
                    let alternatives = refine_task(
                        domain,
                        problem,
                        universe,
                        task,
                        &candidate.state,
                        options,
                        depth,
                        None,
                        visited,
                        applicability_cache,
                        budget,
                        session,
                    )?;
                    for refined in alternatives {
                        let mut steps = candidate.steps.clone();
                        steps.extend(refined.steps);
                        next_candidates.push(RefinedTask {
                            steps,
                            state: refined.state,
                        });
                        if next_candidates.len() >= MAX_REFINEMENTS {
                            break;
                        }
                    }
                    if next_candidates.len() >= MAX_REFINEMENTS {
                        break;
                    }
                }
                if next_candidates.is_empty() {
                    return Ok(Vec::new());
                }
                candidates = next_candidates;
            }
            Ok(candidates)
        }
        TaskOrdering::Partial(edges) => {
            let tasks = instance.tasks.as_ref();
            let ids = instance.ids.as_ref();
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
                tasks,
                &predecessors,
                state,
                options,
                depth,
                visited,
                applicability_cache,
                budget,
                session,
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
    state: &State,
    options: &RefinementOptions,
    depth: usize,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
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
        visited,
        applicability_cache,
        budget,
        &mut results,
        session,
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
    state: &State,
    steps: Vec<RefinementStep>,
    options: &RefinementOptions,
    depth: usize,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    results: &mut Vec<RefinedTask>,
    session: &mut DecompositionSession<'_>,
) -> Result<()> {
    if results.len() >= MAX_REFINEMENTS || !budget.has_time() {
        return Ok(());
    }
    if !remaining.iter().any(|&present| present) {
        results.push(RefinedTask {
            steps,
            state: Arc::clone(state),
        });
        return Ok(());
    }

    let ready: Vec<usize> = (0..tasks.len())
        .filter(|&index| {
            remaining[index] && !predecessors[index].iter().any(|&before| remaining[before])
        })
        .collect();
    if ready.is_empty() {
        return Ok(());
    }
    // The search context clones the task list, the remaining mask, the
    // predecessor sets and the state. Without a policy or a trace sink nobody
    // reads it, so do not build it on the symbolic default path.
    let tracing = session.trace.is_some();
    let context = (session.search_policy.is_some() || tracing).then(|| SearchContext {
        tasks: tasks.to_vec(),
        remaining: remaining.to_vec(),
        predecessors: predecessors.to_vec(),
        state: state.as_ref().clone(),
        depth,
    });
    let order = match (session.search_policy, context.as_ref()) {
        (Some(policy), Some(context)) => policy
            .order_ready(context)
            .filter(|order| valid_search_order(order, &ready))
            .unwrap_or_else(|| ready.clone()),
        _ => ready.clone(),
    };
    let mut successful = if tracing {
        vec![false; order.len()]
    } else {
        Vec::new()
    };

    let ready_limit = ready.len();
    for (slot, &index) in order.iter().take(ready_limit).enumerate() {
        if !budget.has_time() {
            break;
        }
        let before = results.len();
        let alternatives = refine_task(
            domain,
            problem,
            universe,
            &tasks[index],
            state,
            options,
            depth,
            None,
            visited,
            applicability_cache,
            budget,
            session,
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
                visited,
                applicability_cache,
                budget,
                results,
                session,
            )?;
            remaining[index] = true;
            if results.len() >= MAX_REFINEMENTS {
                break;
            }
        }
        if tracing {
            successful[slot] = results.len() > before;
        }
        if results.len() >= MAX_REFINEMENTS {
            break;
        }
    }

    if ready.len() > 1
        && let Some(trace) = session.trace.as_deref_mut()
    {
        let context = context.expect("a trace sink implies a search context");
        trace.search_trace(SearchTrace {
            context,
            ready,
            order,
            successful,
        });
    }
    Ok(())
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

fn holds_optional_bounded(
    formula: Option<&crate::parsing::json::Formula>,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<bool> {
    match holds_optional(formula, bindings, state, universe, budget) {
        Ok(value) => Ok(value),
        Err(error) if formula::is_quantifier_overflow(&error) => Ok(false),
        Err(error) => Err(error),
    }
}

fn goal_status(
    problem: &Problem,
    universe: &Universe<'_>,
    state: &BTreeSet<String>,
    budget: usize,
) -> Result<GoalStatus> {
    match &problem.goal {
        None => Ok(GoalStatus::Absent),
        Some(goal) => match holds(goal, &BTreeMap::new(), state, universe, budget) {
            Ok(true) => Ok(GoalStatus::Satisfied),
            Ok(false) => Ok(GoalStatus::Violated),
            Err(error) if formula::is_quantifier_overflow(&error) => Ok(GoalStatus::Violated),
            Err(error) => Err(error),
        },
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
    let goal = goal_status(problem, &universe, &final_state, options.max_groundings)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_order_sorts_scores_and_breaks_ties_by_index() {
        assert_eq!(rank_order(vec![0.5, 2.0, 1.0]), Some(vec![1, 2, 0]));
        assert_eq!(rank_order(vec![1.0, 1.0]), Some(vec![0, 1]));
        assert_eq!(rank_order(vec![f64::NAN]), None);
        assert_eq!(rank_order(Vec::new()), None);
    }

    #[test]
    fn method_order_rejects_non_permutations() {
        assert!(valid_method_order(&[0, 1], 2));
        assert!(valid_method_order(&[], 0));
        assert!(!valid_method_order(&[0], 2));
        assert!(!valid_method_order(&[0, 0], 2));
        assert!(!valid_method_order(&[0, 2], 2));
    }

    #[test]
    fn search_order_rejects_missing_duplicate_or_blocked_tasks() {
        let ready = vec![2, 4];
        assert!(valid_search_order(&[4, 2], &ready));
        assert!(!valid_search_order(&[2], &ready));
        assert!(!valid_search_order(&[2, 2], &ready));
        assert!(!valid_search_order(&[2, 3], &ready));
    }
}
