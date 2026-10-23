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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};

use crate::parsing::json::{Constraint, Formula, Htn, Subtask, TaskOrdering};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::decomposition::{
    AbstractPlanContext, DecompositionSession, MethodCandidate, MethodRankingContext, MethodTrace,
    SearchContext, SearchTrace,
};
use super::formula::{self, Universe};
use super::graph::{TaskNode, ground_methods_bounded, ground_task_network_edges};
use super::leaf::initial_atoms;
use super::verify::{
    GoalStatus, Step, apply_action, holds, parse_step, unmet_action_preconditions,
    unmet_method_preconditions,
};
use super::{EvaluationOptions, SgplanMerger, evaluate};

/// Counters describing why refinement branches were pruned or failed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefinementDiagnostics {
    /// Number of branches pruned because the global refinement time budget ran out.
    pub budget_time_exhausted: u64,
    /// Number of branches pruned because the global expansion budget was exhausted.
    pub budget_expansions_exhausted: u64,
    /// Number of branches pruned at recursion depth 0.
    pub depth_exhausted: u64,
    /// Number of recursive calls pruned by the `(task, state)` visited set.
    pub cycle_pruned: u64,
    /// Number of tasks with no applicable grounded methods in the current state.
    pub no_applicable_method: u64,
    /// Number of no-applicable dead ends per task name.
    pub no_applicable_tasks: BTreeMap<String, u64>,
    /// Number of tasks whose name matched no grounded method instance at all.
    pub no_instance_tasks: BTreeMap<String, u64>,
    /// Bounded sample of dead ends with the unmet precondition literals.
    pub dead_end_samples: Vec<String>,
    /// Number of grounded methods rejected by network equality constraints.
    pub network_constraint_rejected: u64,
    /// Number of partial-order subnetworks that reached a dead end (no ready task).
    pub partial_ready_empty: u64,
    /// Number of abstract steps that produced no refinement candidates.
    pub abstract_step_failed: u64,
}

impl RefinementDiagnostics {
    fn merge_from(&mut self, other: &Self) {
        self.budget_time_exhausted += other.budget_time_exhausted;
        self.budget_expansions_exhausted += other.budget_expansions_exhausted;
        self.depth_exhausted += other.depth_exhausted;
        self.cycle_pruned += other.cycle_pruned;
        self.no_applicable_method += other.no_applicable_method;
        for (task, count) in &other.no_applicable_tasks {
            *self.no_applicable_tasks.entry(task.clone()).or_default() += count;
        }
        for (task, count) in &other.no_instance_tasks {
            *self.no_instance_tasks.entry(task.clone()).or_default() += count;
        }
        for sample in &other.dead_end_samples {
            if self.dead_end_samples.len() >= DEAD_END_SAMPLE_LIMIT {
                break;
            }
            if !self.dead_end_samples.contains(sample) {
                self.dead_end_samples.push(sample.clone());
            }
        }
        self.network_constraint_rejected += other.network_constraint_rejected;
        self.partial_ready_empty += other.partial_ready_empty;
        self.abstract_step_failed += other.abstract_step_failed;
    }
}

/// How `refine_task` recognises recursive branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CyclePruning {
    /// Prune a task that recurs in the same state on the current path. This is
    /// the default: such a branch cannot make progress.
    #[default]
    TaskState,
    /// Do not prune recurrences. Useful to test whether the visited set rejects
    /// a refinement that would otherwise succeed, for example in domains whose
    /// recursion legitimately revisits a state with a different remainder.
    Off,
}

impl CyclePruning {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "task-state" => Some(Self::TaskState),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::TaskState => "task-state",
            Self::Off => "off",
        }
    }
}

/// Options for hierarchical refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefinementOptions {
    /// Maximum number of method groundings considered for one task.
    pub max_groundings: usize,
    /// Maximum recursion depth through methods.
    pub max_depth: usize,
    /// How recursive branches are pruned.
    pub cycle_pruning: CyclePruning,
}

impl Default for RefinementOptions {
    fn default() -> Self {
        let evaluation = EvaluationOptions::default();
        RefinementOptions {
            max_groundings: evaluation.max_groundings,
            max_depth: 32,
            cycle_pruning: std::env::var("PSPR_CYCLE_KEY")
                .ok()
                .and_then(|value| CyclePruning::parse(&value))
                .unwrap_or_default(),
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
    let mut diagnostics = RefinementDiagnostics::default();
    let candidates = refine_plan_steps(
        domain,
        problem,
        &universe,
        abstract_plan,
        state,
        options,
        options.max_depth,
        &mut diagnostics,
        session,
    )?;
    if let Some(session_diagnostics) = session.diagnostics.as_deref_mut() {
        session_diagnostics.merge_from(&diagnostics);
    }
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
    let precedence = abstract_plan_precedence(
        domain,
        problem,
        &evaluation.plan,
        evaluation_options.max_groundings,
    );
    for attempt in 0..REFINEMENT_RETRY_ATTEMPTS {
        let retry = refined_retry_options(options, attempt);
        let plans = fallback_permutation_candidates(&evaluation.plan, &precedence, attempt);
        for plan in plans {
            match refine_plan(domain, problem, &plan, &retry) {
                Ok(candidate) => {
                    refinement = Some(candidate);
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }
        if refinement.is_some() {
            break;
        }
    }
    if refinement.is_none() {
        // Every sequential order failed. Retry once with the interleaved
        // frontier search, which can mix subtasks of different top-level tasks.
        // The search has its own node bound, so it runs a single attempt
        // instead of the escalating retries above.
        match refine_plan_interleaved(domain, problem, options) {
            Ok(Some(candidate)) => refinement = Some(candidate),
            Ok(None) => {}
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

/// Default upper bound on the nodes the interleaved frontier search may visit.
///
/// The search is a completeness fallback, so it gets a budget that keeps the
/// worst case in the hundreds of milliseconds in release builds. Debug builds
/// are roughly two orders of magnitude slower per node, so they use a much
/// smaller bound to keep the test suite responsive.
/// `PSPR_INTERLEAVED_NODES` overrides both.
const INTERLEAVED_NODE_LIMIT: usize = 20_000;
const INTERLEAVED_NODE_LIMIT_DEBUG: usize = 2_000;
/// Wall-clock bound of the interleaved fallback, in milliseconds.
const INTERLEAVED_TIME_LIMIT_MS: u64 = 3_000;

fn interleaved_node_limit() -> usize {
    std::env::var("PSPR_INTERLEAVED_NODES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) {
            INTERLEAVED_NODE_LIMIT_DEBUG
        } else {
            INTERLEAVED_NODE_LIMIT
        })
}

fn interleaved_deadline() -> Instant {
    let millis = std::env::var("PSPR_INTERLEAVED_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(INTERLEAVED_TIME_LIMIT_MS);
    Instant::now() + Duration::from_millis(millis)
}

/// One task slot of the interleaved frontier, including its partial plan tree.
#[derive(Clone)]
struct FrontierTree {
    task: TaskNode,
    /// Recursion budget left for this task's own refinement.
    depth: usize,
    method: Option<(String, BTreeMap<String, String>)>,
    action: Option<String>,
    children: Vec<usize>,
}

/// A frontier node: the shared state plus every open task.
#[derive(Clone)]
struct Frontier {
    state: State,
    tree: Vec<FrontierTree>,
    /// Slots that still have to be expanded.
    pending: Vec<usize>,
    /// `(before, after)` pairs over slot ids.
    edges: Vec<(usize, usize)>,
    roots: Vec<usize>,
}

impl Frontier {
    fn is_ready(&self, slot: usize) -> bool {
        self.pending.contains(&slot)
            && !self
                .edges
                .iter()
                .any(|(before, after)| *after == slot && self.pending.contains(before))
    }
}

/// Refine the initial task network with a global interleaved frontier search.
///
/// [`refine_plan_steps`] refines every abstract step to completion before it
/// starts the next one. That is complete for a totally ordered top-level
/// network, but it rejects plans that interleave subtasks of different
/// top-level tasks. Partial-order benchmarks need that: in PCP `t1G1` leaves
/// `turnB`, so the recursive `SG1` subtask can only continue after an `SG2`
/// action switched the turn back.
///
/// This search keeps every open task in one frontier and expands whichever
/// ready task the current state allows, so such plans become reachable. It
/// returns `Ok(None)` when no interleaving was found within the node limit.
pub fn refine_plan_interleaved(
    domain: &Domain,
    problem: &Problem,
    options: &RefinementOptions,
) -> Result<Option<Refinement>> {
    let mut diagnostics = RefinementDiagnostics::default();
    refine_plan_interleaved_with_diagnostics(domain, problem, options, &mut diagnostics)
}

/// [`refine_plan_interleaved`] with an explicit diagnostics sink.
pub fn refine_plan_interleaved_with_diagnostics(
    domain: &Domain,
    problem: &Problem,
    options: &RefinementOptions,
    diagnostics: &mut RefinementDiagnostics,
) -> Result<Option<Refinement>> {
    let universe = Universe::new(domain, problem);
    let grounded = ground_task_network_edges(domain, problem, options.max_groundings)?;
    if grounded.tasks.is_empty() {
        return Ok(None);
    }
    let mut tree = Vec::with_capacity(grounded.tasks.len());
    let mut pending = Vec::with_capacity(grounded.tasks.len());
    let mut roots = Vec::with_capacity(grounded.tasks.len());
    for task in &grounded.tasks {
        tree.push(FrontierTree {
            task: task.clone(),
            depth: options.max_depth,
            method: None,
            action: None,
            children: Vec::new(),
        });
        let slot = tree.len() - 1;
        pending.push(slot);
        roots.push(slot);
    }
    let start = Frontier {
        state: Arc::new(initial_atoms(problem)),
        tree,
        pending,
        edges: grounded.edges,
        roots,
    };

    let mut cache = MethodApplicabilityCache::default();
    let mut budget = RefinementBudget::new(grounded.tasks.len(), options);
    let mut nodes_left = interleaved_node_limit();
    let deadline = interleaved_deadline();
    let Some(done) = interleaved_search(
        domain,
        problem,
        &universe,
        options,
        start,
        &mut cache,
        &mut budget,
        diagnostics,
        deadline,
        &mut nodes_left,
    )?
    else {
        return Ok(None);
    };

    let steps: Vec<RefinementStep> = done
        .roots
        .iter()
        .map(|slot| frontier_step(&done.tree, *slot))
        .collect();
    let mut plan = Vec::new();
    for step in &steps {
        step.collect_actions(&mut plan);
    }
    let goal = goal_status(problem, &universe, done.state.as_ref(), options.max_groundings)?;
    Ok(Some(Refinement {
        abstract_plan: grounded.tasks.iter().map(TaskNode::to_string).collect(),
        plan,
        steps,
        final_state: (*done.state).clone(),
        goal,
    }))
}

/// Rebuild the decomposition tree of one slot.
fn frontier_step(tree: &[FrontierTree], slot: usize) -> RefinementStep {
    let node = &tree[slot];
    if let Some(action) = &node.action {
        return RefinementStep::Action {
            task: node.task.clone(),
            step: action.clone(),
        };
    }
    match &node.method {
        Some((method, bindings)) => RefinementStep::Method {
            task: node.task.clone(),
            method: method.clone(),
            bindings: bindings.clone(),
            subtasks: node
                .children
                .iter()
                .map(|child| frontier_step(tree, *child))
                .collect(),
        },
        // Unreachable for a complete frontier: every slot is either an action
        // or a method instance once the search succeeds.
        None => RefinementStep::Action {
            task: node.task.clone(),
            step: node.task.to_string(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn interleaved_search(
    domain: &Domain,
    problem: &Problem,
    universe: &Universe<'_>,
    options: &RefinementOptions,
    node: Frontier,
    cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    diagnostics: &mut RefinementDiagnostics,
    deadline: Instant,
    nodes_left: &mut usize,
) -> Result<Option<Frontier>> {
    if *nodes_left == 0 || Instant::now() >= deadline {
        diagnostics.budget_expansions_exhausted += 1;
        return Ok(None);
    }
    *nodes_left -= 1;
    if !budget.has_time() {
        diagnostics.budget_time_exhausted += 1;
        return Ok(None);
    }
    if node.pending.is_empty() {
        return Ok(Some(node));
    }
    let ready: Vec<usize> = node
        .pending
        .iter()
        .copied()
        .filter(|slot| node.is_ready(*slot))
        .collect();
    if ready.is_empty() {
        diagnostics.partial_ready_empty += 1;
        return Ok(None);
    }

    // Execute first, decompose second. A ready task that is already a
    // primitive action in an applicable state commits the search to that
    // action instead of also branching into method decompositions, which keeps
    // the branching factor of alternating domains like PCP manageable. Only
    // when no ready action can execute does the search decompose compounds.
    let (action_slots, compound_slots): (Vec<usize>, Vec<usize>) = ready
        .into_iter()
        .partition(|slot| domain.actions_for(&node.tree[*slot].task.name).next().is_some());

    for pass in 0..2 {
        let slots = if pass == 0 {
            &action_slots
        } else {
            &compound_slots
        };
        let mut produced = false;
        for slot in slots.iter().copied() {
            let expansions = interleaved_expansions(
                domain,
                universe,
                options,
                &node,
                slot,
                cache,
                budget,
                diagnostics,
            )?;
            if !expansions.is_empty() {
                produced = true;
            }
            for next in expansions {
                if let Some(done) = interleaved_search(
                    domain,
                    problem,
                    universe,
                    options,
                    next,
                    cache,
                    budget,
                    diagnostics,
                    deadline,
                    nodes_left,
                )? {
                    return Ok(Some(done));
                }
                if *nodes_left == 0 || Instant::now() >= deadline {
                    return Ok(None);
                }
            }
        }
        if produced {
            return Ok(None);
        }
    }
    Ok(None)
}

/// Every way `slot` can be advanced by one decomposition or action step.
#[allow(clippy::too_many_arguments)]
fn interleaved_expansions(
    domain: &Domain,
    universe: &Universe<'_>,
    options: &RefinementOptions,
    node: &Frontier,
    slot: usize,
    cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    diagnostics: &mut RefinementDiagnostics,
) -> Result<Vec<Frontier>> {
    let task = node.tree[slot].task.clone();
    let depth = node.tree[slot].depth;
    let mut expansions = Vec::new();
    if depth == 0 {
        diagnostics.depth_exhausted += 1;
        return Ok(expansions);
    }
    if !budget.consume() {
        diagnostics.budget_expansions_exhausted += 1;
        return Ok(expansions);
    }

    if domain.actions_for(&task.name).next().is_some()
        && let Ok(next) = apply_action(
            domain,
            &task,
            node.state.as_ref(),
            universe,
            options.max_groundings,
        )
    {
        let mut expanded = node.clone();
        expanded.state = Arc::new(next);
        expanded.pending.retain(|pending| *pending != slot);
        expanded.tree[slot].action = Some(task.to_string());
        expansions.push(expanded);
    }

    let (instances, applicable) =
        cache.lookup(domain, universe, &task, &node.state, options.max_groundings)?;
    if applicable.is_empty() && expansions.is_empty() {
        diagnostics.no_applicable_method += 1;
        *diagnostics
            .no_applicable_tasks
            .entry(task.to_string())
            .or_default() += 1;
        if instances.is_empty() {
            *diagnostics
                .no_instance_tasks
                .entry(task.to_string())
                .or_default() += 1;
        }
    }
    // Prefer decompositions whose first subtask can execute right now: in
    // alternating domains such as PCP the state only unlocks one family at a
    // time, so leading with an executable subtask keeps the depth-first search
    // on the productive branch. Shorter decompositions break ties, which keeps
    // the search from descending into a recursive variant before its base case.
    let mut ordered: Vec<(bool, usize, u32)> = applicable
        .iter()
        .map(|index| {
            let instance = &instances[*index as usize];
            let lead_ready = instance.tasks.first().is_some_and(|first| {
                domain.actions_for(&first.name).next().is_some()
                    && apply_action(
                        domain,
                        first,
                        node.state.as_ref(),
                        universe,
                        options.max_groundings,
                    )
                    .is_ok()
            });
            (!lead_ready, instance.subtasks, *index)
        })
        .collect();
    ordered.sort_unstable();
    for (_, _, index) in ordered.iter() {
        let instance = &instances[*index as usize];
        if !constraints_hold(instance.network.constraints.as_deref(), &instance.bindings) {
            diagnostics.network_constraint_rejected += 1;
            continue;
        }
        let mut expanded = node.clone();
        expanded.pending.retain(|pending| *pending != slot);
        expanded.tree[slot].method = Some((instance.method.clone(), instance.bindings.clone()));
        let mut children = Vec::with_capacity(instance.tasks.len());
        for subtask in instance.tasks.iter() {
            expanded.tree.push(FrontierTree {
                task: subtask.clone(),
                depth: depth - 1,
                method: None,
                action: None,
                children: Vec::new(),
            });
            let child = expanded.tree.len() - 1;
            children.push(child);
            expanded.pending.push(child);
        }
        expanded.tree[slot].children = children.clone();
        match &instance.network.orderings {
            TaskOrdering::Total => {
                for pair in children.windows(2) {
                    expanded.edges.push((pair[0], pair[1]));
                }
            }
            TaskOrdering::Partial(edges) => {
                for (before, after) in edges {
                    let before = instance
                        .ids
                        .iter()
                        .position(|id| id == before)
                        .expect("ordering ids come from the method network");
                    let after = instance
                        .ids
                        .iter()
                        .position(|id| id == after)
                        .expect("ordering ids come from the method network");
                    if before != after {
                        expanded.edges.push((children[before], children[after]));
                    }
                }
            }
        }
        expansions.push(expanded);
    }
    Ok(expansions)
}


// Bounded breadth-first-style enumeration of refinement alternatives. The
// limit is large enough for the larger Blocksworld-GTOHP instances (up to 14
// tasks with several interchangeable `do_on_table` methods) and still keeps
// the search finite on the harder learned domains.
const MAX_REFINEMENTS: usize = 16384;
/// Upper bound on the retained dead-end samples in [`RefinementDiagnostics`].
const DEAD_END_SAMPLE_LIMIT: usize = 8;
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
        cycle_pruning: base.cycle_pruning,
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

fn push_unique_candidate(
    candidates: &mut Vec<RefinedTask>,
    seen_states: &mut HashSet<State>,
    candidate: RefinedTask,
) -> bool {
    if seen_states.insert(Arc::clone(&candidate.state)) {
        candidates.push(candidate);
    }
    candidates.len() >= MAX_REFINEMENTS
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

fn preferred_method_order(
    preferred_method: Option<&str>,
    instances: &[GroundedInstance],
    applicable: &[u32],
) -> Vec<usize> {
    let Some(preferred) = preferred_method else {
        return (0..applicable.len()).collect();
    };
    let mut preferred_slots = Vec::new();
    let mut fallback_slots = Vec::new();
    for slot in 0..applicable.len() {
        let index = applicable[slot] as usize;
        if instances[index].method == preferred {
            preferred_slots.push(slot);
        } else {
            fallback_slots.push(slot);
        }
    }
    if preferred_slots.is_empty() {
        (0..applicable.len()).collect()
    } else {
        preferred_slots.extend(fallback_slots);
        preferred_slots
    }
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
    let precedence = abstract_plan_precedence(
        domain,
        problem,
        &evaluation.plan,
        evaluation_options.max_groundings,
    );
    for attempt in 0..REFINEMENT_RETRY_ATTEMPTS {
        let retry = refined_retry_options(options, attempt);
        let candidates =
            permutation_candidates(&evaluation.plan, &precedence, attempt, &retry, session);
        for plan in candidates {
            match refine_plan_with_session(domain, problem, &plan, &retry, session) {
                Ok(candidate) => {
                    refinement = Some(candidate);
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }
        if refinement.is_some() {
            break;
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

fn abstract_plan_precedence(
    domain: &Domain,
    problem: &Problem,
    abstract_plan: &[String],
    max_groundings: usize,
) -> Vec<(usize, usize)> {
    let Ok(grounded) = ground_task_network_edges(domain, problem, max_groundings) else {
        return Vec::new();
    };
    if grounded.tasks.len() != abstract_plan.len() {
        return Vec::new();
    }

    let mut by_task: HashMap<TaskNode, VecDeque<usize>> = HashMap::new();
    for (index, task) in grounded.tasks.iter().enumerate() {
        by_task.entry(task.clone()).or_default().push_back(index);
    }

    let mut step_to_ground = Vec::with_capacity(abstract_plan.len());
    for step in abstract_plan {
        let Ok(parsed) = parse_step(step) else {
            return Vec::new();
        };
        let task = parsed.task();
        let Some(indices) = by_task.get_mut(task) else {
            return Vec::new();
        };
        let Some(ground_index) = indices.pop_front() else {
            return Vec::new();
        };
        step_to_ground.push(ground_index);
    }

    let mut ground_to_step = vec![usize::MAX; step_to_ground.len()];
    for (step, ground) in step_to_ground.into_iter().enumerate() {
        if ground >= ground_to_step.len() {
            return Vec::new();
        }
        ground_to_step[ground] = step;
    }

    let mut precedence = Vec::new();
    for (before, after) in grounded.edges {
        if before >= ground_to_step.len() || after >= ground_to_step.len() {
            return Vec::new();
        }
        let step_before = ground_to_step[before];
        let step_after = ground_to_step[after];
        if step_before == usize::MAX || step_after == usize::MAX {
            return Vec::new();
        }
        if step_before != step_after {
            precedence.push((step_before, step_after));
        }
    }
    precedence.sort_unstable();
    precedence.dedup();
    precedence
}

fn valid_plan_permutation(
    order: &[usize],
    step_count: usize,
    precedence: &[(usize, usize)],
) -> bool {
    if order.len() != step_count {
        return false;
    }
    let mut seen = vec![false; step_count];
    let mut position = vec![0usize; step_count];
    for (rank, index) in order.iter().copied().enumerate() {
        if index >= step_count || seen[index] {
            return false;
        }
        seen[index] = true;
        position[index] = rank;
    }
    precedence
        .iter()
        .all(|(before, after)| position[*before] < position[*after])
}

fn permutation_candidates(
    abstract_plan: &[String],
    precedence: &[(usize, usize)],
    attempt: usize,
    options: &RefinementOptions,
    session: &mut DecompositionSession<'_>,
) -> Vec<Vec<String>> {
    let mut candidates = vec![abstract_plan.to_vec()];
    if abstract_plan.len() <= 1 {
        return candidates;
    }
    let add_candidate = |order: Vec<usize>, candidates: &mut Vec<Vec<String>>| {
        if !valid_plan_permutation(&order, abstract_plan.len(), precedence) {
            return;
        }
        let plan: Vec<String> = order
            .into_iter()
            .map(|index| abstract_plan[index].clone())
            .collect();
        if !candidates.iter().any(|existing| existing == &plan) {
            candidates.push(plan);
        }
    };

    let Some(permuter) = session.plan_permuter else {
        return candidates;
    };
    let context = AbstractPlanContext {
        steps: abstract_plan.to_vec(),
        precedence: precedence.to_vec(),
        attempt,
        max_depth: options.max_depth,
    };
    let Some(mut proposals) = permuter.propose(&context) else {
        return candidates;
    };
    proposals.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then(left.order.cmp(&right.order))
    });
    for proposal in proposals {
        if !proposal.score.is_finite() {
            continue;
        }
        add_candidate(proposal.order, &mut candidates);
    }
    candidates
}

fn fallback_permutation_candidates(
    abstract_plan: &[String],
    precedence: &[(usize, usize)],
    attempt: usize,
) -> Vec<Vec<String>> {
    let mut candidates = vec![abstract_plan.to_vec()];
    if attempt == 0 || abstract_plan.len() <= 1 {
        return candidates;
    }
    let reversed: Vec<usize> = (0..abstract_plan.len()).rev().collect();
    if valid_plan_permutation(&reversed, abstract_plan.len(), precedence) {
        let plan: Vec<String> = reversed
            .into_iter()
            .map(|index| abstract_plan[index].clone())
            .collect();
        if !candidates.iter().any(|existing| existing == &plan) {
            candidates.push(plan);
        }
    }
    candidates
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
    diagnostics: &mut RefinementDiagnostics,
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
        let mut seen_step_states = HashSet::new();
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
                diagnostics,
                &mut visited,
                &mut applicability_cache,
                &mut budget,
                session,
            )?;
            for refined in alternatives {
                let mut steps = candidate.steps.clone();
                steps.extend(refined.steps);
                if push_unique_candidate(
                    &mut step_candidates,
                    &mut seen_step_states,
                    RefinedTask {
                        steps,
                        state: refined.state,
                    },
                ) {
                    break;
                }
            }
            if step_candidates.len() >= MAX_REFINEMENTS {
                break;
            }
        }
        if step_candidates.is_empty() {
            diagnostics.abstract_step_failed += 1;
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
    diagnostics: &mut RefinementDiagnostics,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if !budget.has_time() {
        diagnostics.budget_time_exhausted += 1;
        return Ok(Vec::new());
    }
    // A method may rediscover the same task in the same state. Such a branch
    // cannot make progress, so prune it instead of letting cycles (for example
    // `get_to` via an intermediate location) expand until the depth limit.
    // The state is reference-counted, so the visited key does not copy it.
    // `CyclePruning::Off` disables the check for experiments.
    let key = (task.clone(), Arc::clone(state));
    if options.cycle_pruning == CyclePruning::TaskState && !visited.insert(key.clone()) {
        diagnostics.cycle_pruned += 1;
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
        diagnostics,
        visited,
        applicability_cache,
        budget,
        session,
    );
    if options.cycle_pruning == CyclePruning::TaskState {
        visited.remove(&key);
    }
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
    diagnostics: &mut RefinementDiagnostics,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if depth == 0 {
        diagnostics.depth_exhausted += 1;
        return Ok(Vec::new());
    }
    if !budget.has_time() {
        diagnostics.budget_time_exhausted += 1;
        return Ok(Vec::new());
    }
    if !budget.consume() {
        diagnostics.budget_expansions_exhausted += 1;
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
    if applicable.is_empty() {
        diagnostics.no_applicable_method += 1;
        *diagnostics
            .no_applicable_tasks
            .entry(task.to_string())
            .or_default() += 1;
        if instances.is_empty() {
            *diagnostics
                .no_instance_tasks
                .entry(task.to_string())
                .or_default() += 1;
        } else if diagnostics.dead_end_samples.len() < DEAD_END_SAMPLE_LIMIT {
            for sample in unmet_method_preconditions(
                domain,
                universe,
                task,
                state.as_ref(),
                options.max_groundings,
                2,
            ) {
                if diagnostics.dead_end_samples.len() >= DEAD_END_SAMPLE_LIMIT {
                    break;
                }
                let entry = format!("{task}: {sample}");
                if !diagnostics.dead_end_samples.contains(&entry) {
                    diagnostics.dead_end_samples.push(entry);
                }
            }
        }
        if diagnostics.dead_end_samples.len() < DEAD_END_SAMPLE_LIMIT {
            let missing = unmet_action_preconditions(
                domain,
                task,
                state.as_ref(),
                universe,
                options.max_groundings,
            );
            if !missing.is_empty() {
                diagnostics
                    .dead_end_samples
                    .push(format!("{task} missing [{}]", missing.join(", ")));
            }
        }
        return Ok(Vec::new());
    }

    // A ranking decision needs alternatives: with fewer than two applicable
    // methods there is nothing to rank, and a decision in which no candidate
    // produced a refinement teaches the ranker nothing. Both cases are by far
    // the most common — the search visits millions of dead ends — and each
    // trace clones the whole solver state, so the candidates and the context
    // are only materialised for usable decisions.
    let decision = applicable.len() >= 2;
    let tracing = session.trace.is_some();
    let ranked = decision && (session.method_ranker.is_some() || tracing);
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
    // The baseline order is the identity, so it is only materialised when a
    // ranker or the trace needs it; iterating `0..len` below is equivalent.
    let ranked_order: Option<Vec<usize>> = match (session.method_ranker, context.as_ref()) {
        (Some(ranker), Some(context)) => ranker
            .rank(context, &candidates)
            .filter(|scores| scores.len() == candidates.len())
            .and_then(rank_order)
            .filter(|order| valid_method_order(order, candidates.len())),
        _ => None,
    };
    let method_order = ranked_order.clone().unwrap_or_else(|| {
        preferred_method_order(preferred_method, instances.as_ref(), applicable.as_ref())
    });

    let mut results = Vec::new();
    let mut seen_result_states = HashSet::new();
    // Only a ranked decision needs the per-candidate outcome vector; the
    // default path neither ranks nor records, and this runs for every visited
    // state.
    let mut successful = if ranked {
        vec![false; applicable.len()]
    } else {
        Vec::new()
    };
    for candidate_slot in method_order.iter().copied() {
        if !budget.has_time() {
            break;
        }
        let grounded_index = applicable[candidate_slot] as usize;
        let instance = &instances[grounded_index];
        let before = results.len();
        let mut produced = false;
        let alternatives = refine_network(
            domain,
            problem,
            universe,
            instance,
            state,
            options,
            depth - 1,
            diagnostics,
            visited,
            applicability_cache,
            budget,
            session,
        )?;
        for refined in alternatives {
            produced = true;
            if push_unique_candidate(
                &mut results,
                &mut seen_result_states,
                RefinedTask {
                    steps: vec![RefinementStep::Method {
                        task: task.clone(),
                        method: instance.method.clone(),
                        bindings: instance.bindings.clone(),
                        subtasks: refined.steps,
                    }],
                    state: refined.state,
                },
            ) {
                break;
            }
        }
        if ranked {
            successful[candidate_slot] = produced || results.len() > before;
        }
        if results.len() >= MAX_REFINEMENTS {
            break;
        }
    }

    if let Some(trace) = session.trace.as_deref_mut()
        && candidates.len() >= 2
        && successful.iter().any(|successful| *successful)
    {
        let context = context.expect("a trace sink implies a ranking context");
        let order = method_order;
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
    diagnostics: &mut RefinementDiagnostics,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    if !constraints_hold(instance.network.constraints.as_deref(), &instance.bindings) {
        diagnostics.network_constraint_rejected += 1;
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
                let mut seen_next_states = HashSet::new();
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
                        diagnostics,
                        visited,
                        applicability_cache,
                        budget,
                        session,
                    )?;
                    for refined in alternatives {
                        let mut steps = candidate.steps.clone();
                        steps.extend(refined.steps);
                        if push_unique_candidate(
                            &mut next_candidates,
                            &mut seen_next_states,
                            RefinedTask {
                                steps,
                                state: refined.state,
                            },
                        ) {
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
                diagnostics,
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
    diagnostics: &mut RefinementDiagnostics,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    session: &mut DecompositionSession<'_>,
) -> Result<Vec<RefinedTask>> {
    let mut remaining = vec![true; tasks.len()];
    let mut results = Vec::new();
    let mut seen_results = HashSet::new();
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
        diagnostics,
        visited,
        applicability_cache,
        budget,
        &mut results,
        &mut seen_results,
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
    diagnostics: &mut RefinementDiagnostics,
    visited: &mut HashSet<(TaskNode, State)>,
    applicability_cache: &mut MethodApplicabilityCache,
    budget: &mut RefinementBudget,
    results: &mut Vec<RefinedTask>,
    seen_results: &mut HashSet<State>,
    session: &mut DecompositionSession<'_>,
) -> Result<()> {
    if results.len() >= MAX_REFINEMENTS || !budget.has_time() {
        return Ok(());
    }
    if !remaining.iter().any(|&present| present) {
        push_unique_candidate(
            results,
            seen_results,
            RefinedTask {
                steps,
                state: Arc::clone(state),
            },
        );
        return Ok(());
    }

    let ready: Vec<usize> = (0..tasks.len())
        .filter(|&index| {
            remaining[index] && !predecessors[index].iter().any(|&before| remaining[before])
        })
        .collect();
    if ready.is_empty() {
        diagnostics.partial_ready_empty += 1;
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
            diagnostics,
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
                diagnostics,
                visited,
                applicability_cache,
                budget,
                results,
                seen_results,
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
        && successful.iter().any(|successful| *successful)
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
