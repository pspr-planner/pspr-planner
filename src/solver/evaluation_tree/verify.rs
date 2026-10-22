//! Validation of merged plans against the full HDDL semantics.
//!
//! The planner deliberately approximates the problem: leaf tasks only contain
//! local atoms, shared atoms are dropped from preconditions and effect
//! conditions, compound tasks are opaque, and the problem goal is ignored.
//! [`validate_plan`] re-executes a plan without those approximations and checks
//!
//! * ground preconditions and method preconditions,
//! * the full effect semantics (including conditional and universal effects),
//! * that every grounded task of the initial task network is executed,
//! * the ordering and equality constraints of the network, and
//! * the problem goal, when the problem has one. The goal is reported as a
//!   [`GoalStatus`] and only rejected when [`ValidationOptions::require_goal`]
//!   is set, because the planner models the task network rather than the
//!   classical goal.
//!
//! Plans are strings, so action schemas are recovered from the plan step and
//! the matching grounded task. If several action schemas share one name and
//! arity, the first applicable schema is used; such a plan is ambiguous by
//! construction.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail, ensure};

use crate::parsing::json::{Action, Formula};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::formula::Universe;
use super::graph::{
    ConstraintNode, TaskNode, bindings_from, build_task_constraint_graph, ground_atom,
};
use super::leaf::initial_atoms;

/// Options for [`validate_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationOptions {
    /// Maximum number of groundings, matching the evaluation tree that produced
    /// the plan.
    pub max_groundings: usize,
    /// Whether a violated problem goal makes validation fail.
    ///
    /// The planner models the initial task network, not the classical goal, and
    /// compound tasks are opaque, so a plan can be a correct execution of the
    /// task network without achieving the goal. Such plans report
    /// [`GoalStatus::Violated`]; set this flag to reject them.
    pub require_goal: bool,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        ValidationOptions {
            max_groundings: super::EvaluationOptions::default().max_groundings,
            require_goal: false,
        }
    }
}

/// Whether the final state satisfies the problem goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// The problem has no goal.
    Absent,
    /// The goal holds in the final state.
    Satisfied,
    /// The goal does not hold in the final state.
    Violated,
}

/// The outcome of a successful plan validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Number of plan steps.
    pub steps: usize,
    /// Number of grounded tasks of the initial task network.
    pub tasks: usize,
    /// Atoms true after the last step.
    pub final_state: BTreeSet<String>,
    /// Whether the final state satisfies the goal; see [`ValidationOptions`].
    pub goal: GoalStatus,
}

/// Re-execute `plan` against the full semantics of `domain` and `problem`.
///
/// A [`ValidationOptions::max_groundings`] value smaller than the one the
/// evaluation tree used makes the plan impossible to match against the grounded
/// task network.
pub fn validate_plan(
    domain: &Domain,
    problem: &Problem,
    plan: &[String],
    options: &ValidationOptions,
) -> Result<ValidationReport> {
    let max_groundings = options.max_groundings;
    let graph = build_task_constraint_graph(domain, problem, max_groundings)?;
    ensure!(
        !graph.tasks.is_empty(),
        "the initial task network has no grounded tasks"
    );
    let universe = Universe::new(domain, problem);
    let mut state = initial_atoms(problem);
    let mut remaining = vec![true; graph.tasks.len()];
    let mut positions: Vec<Option<usize>> = vec![None; graph.tasks.len()];

    for (position, step) in plan.iter().enumerate() {
        let parsed = parse_step(step)?;
        let task = parsed.task().clone();
        let index = graph
            .tasks
            .iter()
            .enumerate()
            .find(|(index, candidate)| remaining[*index] && **candidate == task)
            .map(|(index, _)| index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "plan step {position} (`{step}`) does not execute a remaining grounded task"
                )
            })?;
        remaining[index] = false;
        positions[index] = Some(position);

        match &parsed {
            Step::Action(task) => {
                state = apply_action(domain, task, &state, &universe, max_groundings)
                    .map_err(|error| anyhow::anyhow!("plan step {position} (`{step}`): {error}"))?;
            }
            Step::Method { name, task } => {
                check_method(domain, name, task, &state, &universe, max_groundings)
                    .map_err(|error| anyhow::anyhow!("plan step {position} (`{step}`): {error}"))?;
            }
            Step::Compound(task) => {
                check_compound(domain, task)
                    .map_err(|error| anyhow::anyhow!("plan step {position} (`{step}`): {error}"))?;
            }
        }
    }

    let missing: Vec<String> = graph
        .tasks
        .iter()
        .enumerate()
        .filter(|(index, _)| remaining[*index])
        .map(|(_, task)| task.to_string())
        .collect();
    ensure!(
        missing.is_empty(),
        "the plan does not execute {} grounded task(s): {}",
        missing.len(),
        summarize(&missing)
    );

    for constraint in &graph.constraints {
        match constraint {
            ConstraintNode::Ordering { before, after } => {
                let before_position = positions[*before].expect("executed task");
                let after_position = positions[*after].expect("executed task");
                ensure!(
                    before_position < after_position,
                    "the ordering constraint `{constraint}` is violated"
                );
            }
            ConstraintNode::Equality { left, right, equal } => {
                ensure!(
                    (left == right) == *equal,
                    "the equality constraint `{constraint}` is violated"
                );
            }
            ConstraintNode::Atom(_) => {}
        }
    }

    let goal = match &problem.goal {
        None => GoalStatus::Absent,
        Some(goal) => {
            let satisfied = holds(goal, &BTreeMap::new(), &state, &universe, max_groundings)?;
            if satisfied {
                GoalStatus::Satisfied
            } else {
                ensure!(
                    !options.require_goal,
                    "the final state does not satisfy the problem goal"
                );
                GoalStatus::Violated
            }
        }
    };

    Ok(ValidationReport {
        steps: plan.len(),
        tasks: graph.tasks.len(),
        final_state: state,
        goal,
    })
}

enum Step {
    Action(TaskNode),
    Method { name: String, task: TaskNode },
    Compound(TaskNode),
}

impl Step {
    fn task(&self) -> &TaskNode {
        match self {
            Step::Action(task) | Step::Method { task, .. } | Step::Compound(task) => task,
        }
    }
}

fn parse_step(step: &str) -> Result<Step> {
    let step = step.trim();
    if let Some(inner) = step
        .strip_prefix("task[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return Ok(Step::Compound(parse_task_reference(inner)?));
    }
    if let Some(open) = step.find('[')
        && step.ends_with(']')
    {
        let name = step[..open].trim().to_string();
        ensure!(!name.is_empty(), "malformed plan step `{step}`");
        let task = parse_task_reference(&step[open + 1..step.len() - 1])?;
        return Ok(Step::Method { name, task });
    }
    Ok(Step::Action(parse_task_reference(step)?))
}

fn parse_task_reference(text: &str) -> Result<TaskNode> {
    let text = text.trim();
    ensure!(!text.is_empty(), "empty task reference");
    if let Some(open) = text.find('(') {
        ensure!(
            text.ends_with(')'),
            "malformed task reference `{text}`: missing `)`"
        );
        let name = text[..open].trim().to_string();
        ensure!(!name.is_empty(), "malformed task reference `{text}`");
        let terms = text[open + 1..text.len() - 1]
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToString::to_string)
            .collect();
        Ok(TaskNode { name, terms })
    } else {
        Ok(TaskNode {
            name: text.to_string(),
            terms: Vec::new(),
        })
    }
}

fn summarize(items: &[String]) -> String {
    const LIMIT: usize = 5;
    if items.len() <= LIMIT {
        items.join(", ")
    } else {
        format!(
            "{}, ... ({} more)",
            items[..LIMIT].join(", "),
            items.len() - LIMIT
        )
    }
}

fn resolve<'a>(name: &'a str, bindings: &'a BTreeMap<String, String>) -> &'a str {
    bindings.get(name).map(String::as_str).unwrap_or(name)
}

fn apply_action(
    domain: &Domain,
    task: &TaskNode,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<BTreeSet<String>> {
    let mut applicable: Vec<(&Action, BTreeMap<String, String>)> = Vec::new();
    for action in domain.actions_for(&task.name) {
        if action.parameters.len() != task.terms.len() {
            continue;
        }
        if !action
            .parameters
            .iter()
            .zip(&task.terms)
            .all(|(parameter, term)| universe.compatible(parameter, term))
        {
            continue;
        }
        let bindings = bindings_from(&action.parameters, &task.terms);
        if holds_optional(
            action.preconditions.as_ref(),
            &bindings,
            state,
            universe,
            budget,
        )? {
            applicable.push((action, bindings));
        }
    }
    ensure!(
        !applicable.is_empty(),
        "no action schema for `{task}` is applicable in the current state"
    );
    // A plan step only stores the grounded operator name. If several schemas
    // share the name and arity the plan cannot distinguish them, so the first
    // applicable schema is used.
    let (action, bindings) = &applicable[0];

    let mut adds = BTreeSet::new();
    let mut deletes = BTreeSet::new();
    collect_effects(
        action.effects.as_ref(),
        bindings,
        state,
        universe,
        budget,
        &mut adds,
        &mut deletes,
    )?;
    let conflicts: Vec<&String> = adds.intersection(&deletes).collect();
    ensure!(
        conflicts.is_empty(),
        "the effects of `{task}` both add and delete {}",
        conflicts
            .iter()
            .map(|atom| format!("`{atom}`"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut next = state.clone();
    for atom in &deletes {
        next.remove(atom);
    }
    next.extend(adds);
    Ok(next)
}

fn check_method(
    domain: &Domain,
    method_name: &str,
    task: &TaskNode,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<()> {
    let Some(method) = domain
        .methods_for(&task.name)
        .filter(|method| method.task_terms.len() == task.terms.len())
        .filter(|method| {
            method
                .task_terms
                .iter()
                .zip(&task.terms)
                .all(|(parameter, term)| universe.compatible(parameter, term))
        })
        .find(|method| method.name.name == method_name)
    else {
        bail!("no method `{method_name}` decomposes `{task}`");
    };
    let bindings = bindings_from(&method.task_terms, &task.terms);
    ensure!(
        holds_optional(
            method.precondition.as_ref(),
            &bindings,
            state,
            universe,
            budget
        )?,
        "the precondition of method `{method_name}` does not hold for `{task}`"
    );
    Ok(())
}

fn check_compound(domain: &Domain, task: &TaskNode) -> Result<()> {
    let has_method = domain
        .methods_for(&task.name)
        .any(|method| method.task_terms.len() == task.terms.len());
    ensure!(
        !has_method,
        "the plan executes `{task}` without a method, but methods exist"
    );
    Ok(())
}

fn holds_optional(
    formula: Option<&Formula>,
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

fn holds(
    formula: &Formula,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<bool> {
    match formula {
        Formula::Empty => Ok(true),
        Formula::Atom(predicate) => {
            Ok(state.contains(&ground_atom(predicate, bindings).to_string()))
        }
        Formula::Not(inner) => Ok(!holds(inner, bindings, state, universe, budget)?),
        Formula::And(items) => {
            for item in items {
                if !holds(item, bindings, state, universe, budget)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Formula::Or(items) => {
            for item in items {
                if holds(item, bindings, state, universe, budget)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Formula::Xor(items) => {
            let mut satisfied = 0;
            for item in items {
                if holds(item, bindings, state, universe, budget)? {
                    satisfied += 1;
                    if satisfied > 1 {
                        return Ok(false);
                    }
                }
            }
            Ok(satisfied == 1)
        }
        Formula::Imply(left, right) => Ok(!holds(left, bindings, state, universe, budget)?
            || holds(right, bindings, state, universe, budget)?),
        Formula::Exists(variables, body) => {
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                if holds(body, &local, state, universe, budget)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Formula::ForAll(variables, body) => {
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                if !holds(body, &local, state, universe, budget)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Formula::Probabilistic(_, _) => {
            bail!("cannot validate probabilistic formulas")
        }
        Formula::Equals(left, right) => Ok(resolve(left, bindings) == resolve(right, bindings)),
    }
}

fn collect_effects(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
    state: &BTreeSet<String>,
    universe: &Universe<'_>,
    budget: usize,
    adds: &mut BTreeSet<String>,
    deletes: &mut BTreeSet<String>,
) -> Result<()> {
    let Some(formula) = formula else {
        return Ok(());
    };
    match formula {
        Formula::Empty => Ok(()),
        Formula::Atom(predicate) => {
            adds.insert(ground_atom(predicate, bindings).to_string());
            Ok(())
        }
        Formula::Not(inner) => match inner.as_ref() {
            Formula::Atom(predicate) => {
                deletes.insert(ground_atom(predicate, bindings).to_string());
                Ok(())
            }
            Formula::Empty => bail!("the effect `not ()` is unsatisfiable"),
            Formula::Not(inner) => collect_effects(
                Some(inner),
                bindings,
                state,
                universe,
                budget,
                adds,
                deletes,
            ),
            Formula::Or(items) => {
                for item in items {
                    let negated = Formula::Not(Box::new(item.clone()));
                    collect_effects(
                        Some(&negated),
                        bindings,
                        state,
                        universe,
                        budget,
                        adds,
                        deletes,
                    )?;
                }
                Ok(())
            }
            Formula::Equals(left, right) => {
                ensure!(
                    resolve(left, bindings) != resolve(right, bindings),
                    "the effect `{formula:?}` is unsatisfiable"
                );
                Ok(())
            }
            other => bail!("cannot validate the negated effect `{other:?}`"),
        },
        Formula::And(items) => {
            for item in items {
                collect_effects(Some(item), bindings, state, universe, budget, adds, deletes)?;
            }
            Ok(())
        }
        Formula::Imply(condition, effect) => {
            if holds(condition, bindings, state, universe, budget)? {
                collect_effects(
                    Some(effect),
                    bindings,
                    state,
                    universe,
                    budget,
                    adds,
                    deletes,
                )?;
            }
            Ok(())
        }
        Formula::ForAll(variables, body) => {
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                collect_effects(Some(body), &local, state, universe, budget, adds, deletes)?;
            }
            Ok(())
        }
        Formula::Or(_) | Formula::Xor(_) | Formula::Exists(_, _) => bail!(
            "cannot validate a nondeterministic effect: the plan does not record which outcome was chosen"
        ),
        Formula::Probabilistic(_, _) => bail!("cannot validate probabilistic effects"),
        Formula::Equals(left, right) => {
            ensure!(
                resolve(left, bindings) == resolve(right, bindings),
                "the effect `{formula:?}` is unsatisfiable"
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::json::{
        Action, Htn, InitialTaskNetwork, Predicate, Subtask, Symbol, TaskOrdering,
    };

    fn symbol(name: &str, symbol_type: Option<&str>) -> Symbol {
        Symbol {
            name: name.to_string(),
            symbol_type: symbol_type.map(ToString::to_string),
        }
    }

    fn atom(name: &str, terms: &[&str]) -> Predicate {
        Predicate {
            name: name.to_string(),
            variables: terms.iter().map(|term| symbol(term, None)).collect(),
        }
    }

    fn subtask(id: &str, task: &str, terms: &[&str]) -> Subtask {
        Subtask {
            id: Some(symbol(id, None)),
            task: symbol(task, None),
            terms: terms.iter().map(|term| symbol(term, None)).collect(),
        }
    }

    fn domain() -> Domain {
        Domain {
            name: "toy".to_string(),
            requirements: Vec::new(),
            types: vec![symbol("place", None)],
            constants: Vec::new(),
            predicates: vec![Predicate {
                name: "at".to_string(),
                variables: vec![symbol("?p", Some("place"))],
            }],
            compound_tasks: Vec::new(),
            methods: Vec::new(),
            actions: vec![
                Action {
                    name: "move".to_string(),
                    parameters: vec![symbol("?from", Some("place")), symbol("?to", Some("place"))],
                    preconditions: Some(Formula::Atom(atom("at", &["?from"]))),
                    effects: Some(Formula::And(vec![
                        Formula::Not(Box::new(Formula::Atom(atom("at", &["?from"])))),
                        Formula::Atom(atom("at", &["?to"])),
                    ])),
                },
                Action {
                    name: "noop".to_string(),
                    parameters: vec![symbol("?p", Some("place"))],
                    preconditions: Some(Formula::Empty),
                    effects: Some(Formula::Empty),
                },
            ],
            functions: Vec::new(),
        }
    }

    fn problem(
        subtasks: Vec<Subtask>,
        orderings: TaskOrdering,
        init_state: Vec<Predicate>,
        goal: Option<Formula>,
    ) -> Problem {
        Problem {
            name: "p".to_string(),
            domain_name: "toy".to_string(),
            objects: vec![
                symbol("a", Some("place")),
                symbol("b", Some("place")),
                symbol("c", Some("place")),
            ],
            init_state,
            goal,
            init_task_network: Some(InitialTaskNetwork {
                parameters: None,
                tn: Htn {
                    subtasks,
                    orderings,
                    constraints: None,
                },
            }),
        }
    }

    fn two_moves() -> Problem {
        problem(
            vec![
                subtask("t1", "move", &["a", "b"]),
                subtask("t2", "move", &["b", "c"]),
            ],
            TaskOrdering::Partial(vec![("t1".to_string(), "t2".to_string())]),
            vec![atom("at", &["a"])],
            Some(Formula::Atom(atom("at", &["c"]))),
        )
    }

    fn options(require_goal: bool) -> ValidationOptions {
        ValidationOptions {
            max_groundings: 64,
            require_goal,
        }
    }

    #[test]
    fn valid_plan_passes() {
        let report = validate_plan(
            &domain(),
            &two_moves(),
            &["move(a, b)".to_string(), "move(b, c)".to_string()],
            &options(true),
        )
        .unwrap();
        assert_eq!(report.steps, 2);
        assert_eq!(report.tasks, 2);
        assert_eq!(report.goal, GoalStatus::Satisfied);
        assert!(report.final_state.contains("at(c)"));
    }

    #[test]
    fn missing_task_is_reported() {
        let error = validate_plan(
            &domain(),
            &two_moves(),
            &["move(a, b)".to_string()],
            &options(false),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("does not execute"));
    }

    #[test]
    fn extra_step_is_reported() {
        let error = validate_plan(
            &domain(),
            &two_moves(),
            &[
                "move(a, b)".to_string(),
                "move(b, c)".to_string(),
                "move(a, b)".to_string(),
            ],
            &options(false),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("remaining grounded task"));
    }

    #[test]
    fn precondition_failure_is_reported() {
        let mut problem = two_moves();
        problem.init_state.clear();
        let error = validate_plan(
            &domain(),
            &problem,
            &["move(a, b)".to_string(), "move(b, c)".to_string()],
            &options(false),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("no action schema"));
    }

    #[test]
    fn ordering_violation_is_reported() {
        let noops = problem(
            vec![subtask("t1", "noop", &["a"]), subtask("t2", "noop", &["b"])],
            TaskOrdering::Total,
            Vec::new(),
            None,
        );
        let error = validate_plan(
            &domain(),
            &noops,
            &["noop(b)".to_string(), "noop(a)".to_string()],
            &options(false),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("ordering constraint"));
    }

    #[test]
    fn goal_violation_is_reported_only_when_required() {
        let problem = problem(
            vec![subtask("t1", "move", &["a", "b"])],
            TaskOrdering::Total,
            vec![atom("at", &["a"])],
            Some(Formula::Atom(atom("at", &["c"]))),
        );
        let plan = vec!["move(a, b)".to_string()];
        let report = validate_plan(&domain(), &problem, &plan, &options(false)).unwrap();
        assert_eq!(report.goal, GoalStatus::Violated);
        let error = validate_plan(&domain(), &problem, &plan, &options(true)).unwrap_err();
        assert!(format!("{error}").contains("goal"));
    }

    #[test]
    fn absent_goal_is_reported() {
        let problem = problem(
            vec![subtask("t1", "move", &["a", "b"])],
            TaskOrdering::Total,
            vec![atom("at", &["a"])],
            None,
        );
        let report = validate_plan(
            &domain(),
            &problem,
            &["move(a, b)".to_string()],
            &options(true),
        )
        .unwrap();
        assert_eq!(report.goal, GoalStatus::Absent);
    }
}
