//! Static detection of the liftability gap of the top-level projection.
//!
//! The abstraction `alpha(P)` of `docs/pspr-algorithm.tex` replaces every
//! top-level compound task `t` by operators that test the method precondition
//! and add `exec_t`. The effects of the method's subtasks never reach the
//! abstract state. This module flags the task pairs where that omission
//! matters: an earlier top-level task `t1` has a method whose decomposition
//! writes an atom `p`, and a later top-level task `t2` has a method (or a
//! primitive action) whose precondition reads `p`.
//!
//! # Detection
//!
//! 1. Ground the initial task network and compute the transitive closure of
//!    its ordering edges, so "later" is defined for partial orders too.
//! 2. Enumerate the *reachable* ground task nodes: starting from the top-level
//!    tasks, every grounded method instance contributes its subtasks, and the
//!    subtask terms are bound through the instance. The enumeration is a
//!    breadth-first search over task nodes, so a node is expanded once no
//!    matter how many parents reach it.
//! 3. Compute the *may-write* sets of every node by a monotone fixpoint: a
//!    primitive node writes the union of its effect alternatives, a compound
//!    node writes the union of all its subtasks over all its methods. The
//!    fixpoint terminates on recursive methods and yields the least solution,
//!    i.e. the atoms that a *finite* decomposition can write. Method and
//!    action preconditions are ignored, so the sets over-approximate.
//! 4. For every top-level task, compute the *may-read* literals of every
//!    grounded method (or action) instance from the DNF of its precondition.
//! 5. Report every `(t1, t2, atom)` where `t1` precedes `t2` in the closure,
//!    some method of `t1` writes `atom` with polarity `w`, and a precondition
//!    of `t2` contains `atom` with polarity `r`. The pair is a
//!    [`GapKind::HiddenSatisfier`] when `w == r` (the projection cannot see
//!    the establishment) and a [`GapKind::HiddenFalsifier`] when `w != r`
//!    (the projection keeps believing the literal).
//!
//! # Interpretation
//!
//! A reported pair is a *candidate* gap, not a proof that the instance is
//! unsolvable: the producer method may be unused, the consumer clause may be
//! satisfiable another way (`required_everywhere == false`), the atom may
//! already hold (`literal_initially_true`), or a primitive top-level task may
//! write it in the projection (`visible_producer`). Conversely, an empty
//! report is not a proof of liftability: [`LiftabilityReport::truncated`]
//! records that a bound was hit.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;

use anyhow::Result;

use crate::parsing::json::Action;
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::formula::{self, Universe};
use super::graph::{
    Atom, TaskNode, bindings_from, ground_methods_bounded, ground_task_network_edges,
};
use super::leaf::initial_atoms;

/// A ground literal: an atom together with its required polarity.
type Literal = (Atom, bool);

/// Bounds of the static analysis.
#[derive(Debug, Clone)]
pub struct LiftabilityOptions {
    /// Grounding budget handed to the task-network and method groundings.
    pub max_groundings: usize,
    /// Hard cap on the number of reachable ground task nodes.
    pub max_instances: usize,
}

impl Default for LiftabilityOptions {
    fn default() -> Self {
        LiftabilityOptions {
            max_groundings: 256,
            max_instances: 50_000,
        }
    }
}

/// Which side of the abstraction gap a reported pair belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GapKind {
    /// The earlier task makes the later task's literal true, but `alpha(P)`
    /// never writes it. The abstract operator for the later task can be
    /// disabled although a decomposition supports it.
    HiddenSatisfier,
    /// The earlier task makes the later task's literal false, but `alpha(P)`
    /// never removes or establishes it either. The abstract operator can fire
    /// although no decomposition supports it.
    HiddenFalsifier,
}

/// Where the conflicting requirement is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsumerKind {
    /// A method precondition of the later compound task.
    Method,
    /// An action precondition of the later primitive task.
    Action,
}

/// One detected `(earlier task, later task, atom)` conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftabilityGap {
    /// Index of the earlier top-level task in [`LiftabilityReport::tasks`].
    pub before: usize,
    /// Index of the later top-level task in [`LiftabilityReport::tasks`].
    pub after: usize,
    pub before_task: TaskNode,
    pub after_task: TaskNode,
    /// Method of `before` whose decomposition writes the atom.
    pub before_method: String,
    /// Method (or action) of `after` whose precondition reads the atom.
    pub after_method: String,
    pub consumer: ConsumerKind,
    pub kind: GapKind,
    pub atom: Atom,
    /// Polarity of the atom in the producer's effect.
    pub produced_positive: bool,
    /// Polarity of the atom in the consumer's precondition.
    pub required_positive: bool,
    /// Subtask occurrence of `before` that writes the atom.
    pub witness: TaskNode,
    /// The consumer literal is present in every DNF clause of its
    /// precondition, so no operator variant avoids it.
    pub required_everywhere: bool,
    /// The consumer literal already holds in the initial state.
    pub literal_initially_true: bool,
    /// A *primitive* top-level task ordered before `after` writes the same
    /// literal, so the projection does observe a producer for it.
    pub visible_producer: bool,
}

impl fmt::Display for LiftabilityGap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verb = if self.produced_positive {
            "establishes"
        } else {
            "destroys"
        };
        let required = if self.required_positive {
            format!("`{}`", self.atom)
        } else {
            format!("`not {}`", self.atom)
        };
        write!(
            formatter,
            "{} [{}] -> {} [{}]: subtask `{}` {} `{}`, but `{}` requires {}",
            self.before_task,
            self.before_method,
            self.after_task,
            self.after_method,
            self.witness,
            verb,
            self.atom,
            self.after_method,
            required,
        )?;
        match self.kind {
            GapKind::HiddenSatisfier => {
                write!(formatter, " (hidden satisfier)")?;
            }
            GapKind::HiddenFalsifier => {
                write!(formatter, " (hidden falsifier)")?;
            }
        }
        if !self.required_everywhere {
            write!(formatter, " [only some DNF clauses]")?;
        }
        if self.literal_initially_true {
            write!(formatter, " [already true initially]")?;
        }
        if self.visible_producer {
            write!(formatter, " [another primitive task writes it]")?;
        }
        Ok(())
    }
}

/// The result of [`detect_liftability_gaps`].
#[derive(Debug, Clone)]
pub struct LiftabilityReport {
    /// The grounded top-level tasks.
    pub tasks: Vec<TaskNode>,
    /// The transitive closure of the top-level ordering, as `(before, after)`.
    pub orderings: Vec<(usize, usize)>,
    pub gaps: Vec<LiftabilityGap>,
    /// A grounding or node bound was hit (or a formula exceeded the budget),
    /// so the report may be partial.
    pub truncated: bool,
    /// Number of reachable ground task nodes that were analysed.
    pub expanded: usize,
}

impl LiftabilityReport {
    /// Whether no candidate gap was found.
    pub fn is_clear(&self) -> bool {
        self.gaps.is_empty()
    }

    /// The top-level tasks that occur in at least one gap.
    pub fn involved_tasks(&self) -> BTreeSet<usize> {
        self.gaps
            .iter()
            .flat_map(|gap| [gap.before, gap.after])
            .collect()
    }
}

impl fmt::Display for LiftabilityReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.gaps.is_empty() {
            writeln!(formatter, "no gap found")?;
        } else {
            let satisfiers = self
                .gaps
                .iter()
                .filter(|gap| gap.kind == GapKind::HiddenSatisfier)
                .count();
            writeln!(
                formatter,
                "{} gap(s): {} hidden satisfier(s), {} hidden falsifier(s)",
                self.gaps.len(),
                satisfiers,
                self.gaps.len() - satisfiers
            )?;
            for gap in &self.gaps {
                writeln!(formatter, "  {gap}")?;
            }
        }
        if self.truncated {
            writeln!(
                formatter,
                "  note: the analysis hit a bound after {} expansions; \
                 the report may be incomplete",
                self.expanded
            )?;
        }
        Ok(())
    }
}

/// Detect the task pairs whose subtask effects are dropped by `alpha(P)`.
pub fn detect_liftability_gaps(
    domain: &Domain,
    problem: &Problem,
    options: &LiftabilityOptions,
) -> Result<LiftabilityReport> {
    let grounded = ground_task_network_edges(domain, problem, options.max_groundings)?;
    let analysis = analyse(
        domain,
        problem,
        &grounded.tasks,
        &grounded.edges,
        options,
        PairScope::Ordered,
    )?;
    let orderings = analysis
        .closure
        .iter()
        .enumerate()
        .flat_map(|(before, after)| after.iter().map(move |&after| (before, after)))
        .collect();
    Ok(LiftabilityReport {
        tasks: grounded.tasks,
        orderings,
        gaps: analysis.gaps,
        truncated: analysis.truncated,
        expanded: analysis.expanded,
    })
}

/// The implicit top-level orderings that [`implicit_orderings`] would add for
/// this problem, computed from a fresh grounding.
pub fn repair_orderings(
    domain: &Domain,
    problem: &Problem,
    options: &LiftabilityOptions,
) -> Result<Vec<(usize, usize)>> {
    let grounded = ground_task_network_edges(domain, problem, options.max_groundings)?;
    implicit_orderings(
        domain,
        problem,
        &grounded.tasks,
        &grounded.edges,
        &grounded.groundings,
        options,
    )
}

/// Implicit top-level orderings that repair the detected gaps.
///
/// The analysis inspects the pairs that the network leaves unordered in both
/// directions: they are the ones whose interleaving the abstract plan can get
/// wrong. A hidden satisfier makes `t1` the producer of a literal that `t2`
/// cannot do without, so `t1` must be executed (and refined) first; a hidden
/// falsifier is the mirror image, so `t2` must come first. Both rules only
/// apply when the literal is unavoidable in the consumer
/// ([`LiftabilityGap::required_everywhere`]) and neither the initial state nor
/// another primitive top-level task supplies it. Candidates that would close a
/// cycle are dropped.
pub(super) fn implicit_orderings(
    domain: &Domain,
    problem: &Problem,
    tasks: &[TaskNode],
    edges: &[(usize, usize)],
    groundings: &[(usize, usize)],
    options: &LiftabilityOptions,
) -> Result<Vec<(usize, usize)>> {
    let analysis = analyse(domain, problem, tasks, edges, options, PairScope::Unordered)?;
    let mut candidates: BTreeSet<(usize, usize)> = BTreeSet::new();
    for gap in &analysis.gaps {
        if !gap.required_everywhere || gap.visible_producer || gap.before == gap.after {
            continue;
        }
        // Groundings are independent instantiations of the same network; an
        // ordering across them would be meaningless.
        if !same_grounding(groundings, gap.before, gap.after) {
            continue;
        }
        let pair = match gap.kind {
            // The producer makes the consumer's literal true; refine it first.
            GapKind::HiddenSatisfier if !gap.literal_initially_true => (gap.before, gap.after),
            // The producer makes the consumer's literal false; refine the
            // consumer first while the literal still holds.
            GapKind::HiddenFalsifier if gap.literal_initially_true => (gap.after, gap.before),
            _ => continue,
        };
        candidates.insert(pair);
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut closure = analysis.closure;
    if (0..tasks.len()).any(|node| closure[node].contains(&node)) {
        // A cyclic task network is rejected elsewhere; do not add to it.
        return Ok(Vec::new());
    }
    let mut accepted: Vec<(usize, usize)> = Vec::new();
    for (before, after) in candidates {
        if insert_ordering(&mut closure, before, after) {
            accepted.push((before, after));
        }
    }
    Ok(accepted)
}

/// Whether two task indices belong to the same grounding.
fn same_grounding(ranges: &[(usize, usize)], before: usize, after: usize) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| (start..end).contains(&before) && (start..end).contains(&after))
}

/// Insert `before -> after` into an acyclicity-preserving closure.
///
/// Returns `false` when the edge is already implied or would close a cycle.
fn insert_ordering(closure: &mut [BTreeSet<usize>], before: usize, after: usize) -> bool {
    if before == after || closure[before].contains(&after) {
        return false;
    }
    if closure[after].contains(&before) {
        return false;
    }
    let successors = closure[after].clone();
    let predecessors: Vec<usize> = (0..closure.len())
        .filter(|&node| node == before || closure[node].contains(&before))
        .collect();
    for node in predecessors {
        closure[node].insert(after);
        closure[node].extend(successors.iter().copied());
    }
    true
}

/// Which task pairs a gap analysis inspects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairScope {
    /// Pairs that the current ordering already forces.
    Ordered,
    /// Pairs that the current ordering leaves free in both directions.
    Unordered,
}

/// The gaps of a grounded task network, together with the ordering closure.
struct Analysis {
    gaps: Vec<LiftabilityGap>,
    closure: Vec<BTreeSet<usize>>,
    truncated: bool,
    expanded: usize,
}

fn analyse(
    domain: &Domain,
    problem: &Problem,
    tasks: &[TaskNode],
    edges: &[(usize, usize)],
    options: &LiftabilityOptions,
    scope: PairScope,
) -> Result<Analysis> {
    let closure = transitive_closure(tasks.len(), edges);
    let initial = initial_atoms(problem);
    let pairs = pairs(&closure, scope);

    // Without a candidate pair there can be no gap, regardless of how the
    // subtasks decompose; report the result as complete.
    if pairs.is_empty() {
        return Ok(Analysis {
            gaps: Vec::new(),
            closure,
            truncated: false,
            expanded: 0,
        });
    }

    let mut expansion = Expansion::new(domain, problem, options);
    for task in tasks {
        expansion.intern(task.clone());
    }
    expansion.expand();
    let effects = expansion.propagate();

    // Effects of every grounded method instance per top-level task. Primitive
    // top-level tasks are not producers: their effects are part of `alpha(P)`.
    let mut producers: Vec<Vec<Producer>> = Vec::with_capacity(tasks.len());
    let mut visible: BTreeMap<Literal, BTreeSet<usize>> = BTreeMap::new();
    for (index, task) in tasks.iter().enumerate() {
        let Some(node) = expansion.index_of(task) else {
            producers.push(Vec::new());
            continue;
        };
        if let Some(primitive) = &expansion.primitive[node] {
            for atom in primitive.add.keys() {
                visible
                    .entry((atom.clone(), true))
                    .or_default()
                    .insert(index);
            }
            for atom in primitive.del.keys() {
                visible
                    .entry((atom.clone(), false))
                    .or_default()
                    .insert(index);
            }
            producers.push(Vec::new());
        } else {
            producers.push(
                expansion.methods[node]
                    .iter()
                    .map(|method| {
                        let mut method_effects = Effects::default();
                        for &child in &method.children {
                            method_effects
                                .merge_with_witness(&effects[child], &expansion.nodes[child]);
                        }
                        Producer {
                            method: method.name.clone(),
                            effects: method_effects,
                        }
                    })
                    .collect(),
            );
        }
    }

    let consumers: Vec<Vec<Consumer>> =
        tasks.iter().map(|task| expansion.consumers(task)).collect();

    let mut gaps = Vec::new();
    for (before, after) in pairs {
        for producer in &producers[before] {
            for (atom, witness, produced_positive) in producer.effects.literals() {
                for consumer in &consumers[after] {
                    let Some(usage) = consumer.uses.get(&atom) else {
                        continue;
                    };
                    for required_positive in [true, false] {
                        let occurrences = usage.count(required_positive);
                        if occurrences == 0 {
                            continue;
                        }
                        gaps.push(LiftabilityGap {
                            before,
                            after,
                            before_task: tasks[before].clone(),
                            after_task: tasks[after].clone(),
                            before_method: producer.method.clone(),
                            after_method: consumer.name.clone(),
                            consumer: consumer.kind,
                            kind: if produced_positive == required_positive {
                                GapKind::HiddenSatisfier
                            } else {
                                GapKind::HiddenFalsifier
                            },
                            atom: atom.clone(),
                            produced_positive,
                            required_positive,
                            witness: witness.clone(),
                            required_everywhere: occurrences >= consumer.clauses,
                            literal_initially_true: initial.contains(&atom.to_string())
                                == required_positive,
                            visible_producer: visible
                                .get(&(atom.clone(), required_positive))
                                .is_some_and(|writers| {
                                    writers.iter().any(|&writer| {
                                        writer != before && closure[writer].contains(&after)
                                    })
                                }),
                        });
                    }
                }
            }
        }
    }
    gaps.sort_by(|left, right| {
        (
            left.before,
            left.after,
            &left.before_method,
            &left.after_method,
            &left.atom,
            left.produced_positive,
            left.required_positive,
        )
            .cmp(&(
                right.before,
                right.after,
                &right.before_method,
                &right.after_method,
                &right.atom,
                right.produced_positive,
                right.required_positive,
            ))
    });
    gaps.dedup();

    Ok(Analysis {
        gaps,
        closure,
        truncated: expansion.truncated,
        expanded: expansion.len(),
    })
}

/// The task pairs a [`PairScope`] selects.
fn pairs(closure: &[BTreeSet<usize>], scope: PairScope) -> Vec<(usize, usize)> {
    let count = closure.len();
    match scope {
        PairScope::Ordered => (0..count)
            .flat_map(|before| closure[before].iter().map(move |&after| (before, after)))
            .collect(),
        PairScope::Unordered => (0..count)
            .flat_map(|before| {
                (0..count)
                    .filter(move |&after| {
                        before != after
                            && !closure[before].contains(&after)
                            && !closure[after].contains(&before)
                    })
                    .map(move |after| (before, after))
            })
            .collect(),
    }
}

/// The transitive closure of the top-level ordering edges.
fn transitive_closure(count: usize, edges: &[(usize, usize)]) -> Vec<BTreeSet<usize>> {
    let mut successors = vec![BTreeSet::new(); count];
    for &(before, after) in edges {
        if before < count && after < count {
            successors[before].insert(after);
        }
    }
    loop {
        let mut changed = false;
        for node in 0..count {
            let reachable: Vec<usize> = successors[node].iter().copied().collect();
            for next in reachable {
                let inherited = successors[next].clone();
                for target in inherited {
                    if successors[node].insert(target) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return successors;
        }
    }
}

/// The may-write atoms of one decomposition, with a witness subtask.
#[derive(Debug, Clone, Default)]
struct Effects {
    add: BTreeMap<Atom, TaskNode>,
    del: BTreeMap<Atom, TaskNode>,
}

impl Effects {
    /// Insert the literals of `other`, attributed to `witness`, and return the
    /// literals that were not present before.
    fn merge_with_witness(&mut self, other: &Effects, witness: &TaskNode) -> Effects {
        let mut delta = Effects::default();
        for atom in other.add.keys() {
            if !self.add.contains_key(atom) {
                self.add.insert(atom.clone(), witness.clone());
                delta.add.insert(atom.clone(), witness.clone());
            }
        }
        for atom in other.del.keys() {
            if !self.del.contains_key(atom) {
                self.del.insert(atom.clone(), witness.clone());
                delta.del.insert(atom.clone(), witness.clone());
            }
        }
        delta
    }

    fn is_empty(&self) -> bool {
        self.add.is_empty() && self.del.is_empty()
    }

    fn literals(&self) -> Vec<(Atom, TaskNode, bool)> {
        self.add
            .iter()
            .map(|(atom, witness)| (atom.clone(), witness.clone(), true))
            .chain(
                self.del
                    .iter()
                    .map(|(atom, witness)| (atom.clone(), witness.clone(), false)),
            )
            .collect()
    }
}

/// One grounded method instance of a top-level task and its may-write sets.
#[derive(Debug, Clone)]
struct Producer {
    method: String,
    effects: Effects,
}

/// One grounded precondition of a top-level task.
#[derive(Debug, Clone)]
struct Consumer {
    kind: ConsumerKind,
    name: String,
    /// Literal occurrences over the DNF clauses of the precondition.
    uses: BTreeMap<Atom, LiteralUse>,
    /// Number of DNF clauses.
    clauses: usize,
}

impl Consumer {
    fn new(kind: ConsumerKind, name: String, clauses: &[formula::Clause]) -> Option<Self> {
        if clauses.is_empty() {
            // An unsatisfiable precondition has no operator variant.
            return None;
        }
        let mut uses: BTreeMap<Atom, LiteralUse> = BTreeMap::new();
        for clause in clauses {
            for (atom, positive) in clause {
                let entry = uses.entry(atom.clone()).or_default();
                if *positive {
                    entry.positive += 1;
                } else {
                    entry.negative += 1;
                }
            }
        }
        if uses.is_empty() {
            return None;
        }
        Some(Consumer {
            kind,
            name,
            uses,
            clauses: clauses.len(),
        })
    }
}

/// How often a literal occurs in the DNF clauses of a precondition.
#[derive(Debug, Clone, Copy, Default)]
struct LiteralUse {
    positive: usize,
    negative: usize,
}

impl LiteralUse {
    fn count(self, positive: bool) -> usize {
        if positive {
            self.positive
        } else {
            self.negative
        }
    }
}

/// One grounded method instance of a reachable task node.
#[derive(Debug, Clone)]
struct MethodExpansion {
    name: String,
    children: Vec<usize>,
}

/// The reachable ground task graph and the may-write fixpoint over it.
struct Expansion<'a> {
    domain: &'a Domain,
    universe: Universe<'a>,
    options: &'a LiftabilityOptions,
    nodes: Vec<TaskNode>,
    index: HashMap<TaskNode, usize>,
    /// Primitive action effects; `None` for compound tasks.
    primitive: Vec<Option<Effects>>,
    /// Grounded method instances of a compound task.
    methods: Vec<Vec<MethodExpansion>>,
    /// Whether a node has already been expanded.
    expanded: Vec<bool>,
    /// Subtask -> parent edges for the fixpoint.
    parents: Vec<Vec<usize>>,
    queue: VecDeque<usize>,
    truncated: bool,
}

impl<'a> Expansion<'a> {
    fn new(domain: &'a Domain, problem: &'a Problem, options: &'a LiftabilityOptions) -> Self {
        Expansion {
            domain,
            universe: Universe::new(domain, problem),
            options,
            nodes: Vec::new(),
            index: HashMap::new(),
            primitive: Vec::new(),
            methods: Vec::new(),
            expanded: Vec::new(),
            parents: Vec::new(),
            queue: VecDeque::new(),
            truncated: false,
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn index_of(&self, task: &TaskNode) -> Option<usize> {
        self.index.get(task).copied()
    }

    /// Insert a task node and schedule it for expansion.
    fn intern(&mut self, task: TaskNode) -> Option<usize> {
        if let Some(&id) = self.index.get(&task) {
            return Some(id);
        }
        if self.nodes.len() >= self.options.max_instances {
            self.truncated = true;
            return None;
        }
        let id = self.nodes.len();
        self.index.insert(task.clone(), id);
        self.nodes.push(task);
        self.primitive.push(None);
        self.methods.push(Vec::new());
        self.expanded.push(false);
        self.parents.push(Vec::new());
        self.queue.push_back(id);
        Some(id)
    }

    /// Expand every queued node: a primitive node records its action effects,
    /// a compound node records the subtask nodes of every grounded method.
    fn expand(&mut self) {
        while let Some(id) = self.queue.pop_front() {
            if self.expanded[id] {
                continue;
            }
            self.expanded[id] = true;
            let task = self.nodes[id].clone();

            match self.action_effects(&task) {
                Ok(Some(effects)) => {
                    self.primitive[id] = Some(effects);
                    continue;
                }
                Ok(None) => {}
                Err(_) => {
                    // The effect formulas exceed the grounding budget. Treat
                    // the task as opaque and flag the report.
                    self.truncated = true;
                    continue;
                }
            }

            let domain = self.domain;
            let instances =
                ground_methods_bounded(domain, &self.universe, &task, self.options.max_groundings);
            if instances.truncated {
                self.truncated = true;
            }
            for instance in instances.instances {
                let mut children = Vec::with_capacity(instance.method.tn.subtasks.len());
                for subtask in &instance.method.tn.subtasks {
                    let child = TaskNode {
                        name: subtask.task.name.clone(),
                        terms: subtask
                            .terms
                            .iter()
                            .map(|term| {
                                instance
                                    .bindings
                                    .get(&term.name)
                                    .cloned()
                                    .unwrap_or_else(|| term.name.clone())
                            })
                            .collect(),
                    };
                    if let Some(child) = self.intern(child) {
                        if !self.parents[child].contains(&id) {
                            self.parents[child].push(id);
                        }
                        children.push(child);
                    }
                }
                self.methods[id].push(MethodExpansion {
                    name: instance.method.name.name.clone(),
                    children,
                });
            }
        }
    }

    /// Least fixpoint of `effects(node) = union of its subtask effects`.
    ///
    /// Primitive nodes start with their action effects; every parent then
    /// accumulates the effects of its subtasks. Monotone growth terminates
    /// even when the methods are recursive, and the least solution contains
    /// exactly the atoms that a finite decomposition can write.
    ///
    /// Only the literals that a node gained since it was last processed are
    /// propagated, so every `(atom, edge)` pair is handled once.
    fn propagate(&self) -> Vec<Effects> {
        let mut effects = vec![Effects::default(); self.nodes.len()];
        let mut pending: VecDeque<(usize, Effects)> = VecDeque::new();
        for (id, primitive) in self.primitive.iter().enumerate() {
            if let Some(primitive) = primitive {
                effects[id] = primitive.clone();
                if !primitive.is_empty() {
                    pending.push_back((id, primitive.clone()));
                }
            }
        }
        while let Some((child, delta)) = pending.pop_front() {
            for &parent in &self.parents[child] {
                let gained = effects[parent].merge_with_witness(&delta, &self.nodes[child]);
                if !gained.is_empty() {
                    pending.push_back((parent, gained));
                }
            }
        }
        effects
    }

    /// Effects of the action schemas of a primitive task, or `None` when the
    /// task names no action of the right arity.
    fn action_effects(&self, task: &TaskNode) -> Result<Option<Effects>> {
        let domain = self.domain;
        let actions: Vec<&Action> = domain
            .actions_for(&task.name)
            .filter(|action| action.parameters.len() == task.terms.len())
            .collect();
        if actions.is_empty() {
            return Ok(None);
        }
        let mut effects = Effects::default();
        for action in actions {
            let bindings = bindings_from(&action.parameters, &task.terms);
            let alternatives = formula::effect_alternatives_of(
                action.effects.as_ref(),
                &bindings,
                &self.universe,
                self.options.max_groundings,
            )?;
            for alternative in alternatives {
                for term in alternative {
                    let map = if term.value {
                        &mut effects.add
                    } else {
                        &mut effects.del
                    };
                    map.entry(term.atom).or_insert_with(|| task.clone());
                }
            }
        }
        Ok(Some(effects))
    }

    /// Preconditions of every grounded method (or action) instance of a task.
    ///
    /// A grounding that exceeds the budget leaves the task without consumers
    /// and flags the report as truncated.
    fn consumers(&mut self, task: &TaskNode) -> Vec<Consumer> {
        match self.consumers_inner(task) {
            Ok((consumers, truncated)) => {
                if truncated {
                    self.truncated = true;
                }
                consumers
            }
            Err(_) => {
                self.truncated = true;
                Vec::new()
            }
        }
    }

    fn consumers_inner(&self, task: &TaskNode) -> Result<(Vec<Consumer>, bool)> {
        let domain = self.domain;
        let actions: Vec<&Action> = domain
            .actions_for(&task.name)
            .filter(|action| action.parameters.len() == task.terms.len())
            .collect();
        let mut consumers = Vec::new();
        let mut truncated = false;
        if actions.is_empty() {
            let instances =
                ground_methods_bounded(domain, &self.universe, task, self.options.max_groundings);
            truncated = instances.truncated;
            for instance in &instances.instances {
                let clauses = formula::precondition_clauses(
                    instance.method.precondition.as_ref(),
                    &instance.bindings,
                    &self.universe,
                    self.options.max_groundings,
                )?;
                if let Some(consumer) = Consumer::new(
                    ConsumerKind::Method,
                    instance.method.name.name.clone(),
                    &clauses,
                ) {
                    consumers.push(consumer);
                }
            }
        } else {
            for action in actions {
                let bindings = bindings_from(&action.parameters, &task.terms);
                let clauses = formula::precondition_clauses(
                    action.preconditions.as_ref(),
                    &bindings,
                    &self.universe,
                    self.options.max_groundings,
                )?;
                if let Some(consumer) =
                    Consumer::new(ConsumerKind::Action, action.name.clone(), &clauses)
                {
                    consumers.push(consumer);
                }
            }
        }
        Ok((consumers, truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::json::{
        Formula, Htn, InitialTaskNetwork, Method, Predicate, Subtask, Symbol, TaskOrdering,
    };

    fn symbol(name: &str) -> Symbol {
        Symbol {
            name: name.to_string(),
            symbol_type: None,
        }
    }

    fn atom(name: &str, variables: &[&str]) -> Formula {
        Formula::Atom(Predicate {
            name: name.to_string(),
            variables: variables.iter().map(|variable| symbol(variable)).collect(),
        })
    }

    fn delete(name: &str, variables: &[&str]) -> Formula {
        Formula::Not(Box::new(atom(name, variables)))
    }

    fn action(
        name: &str,
        parameters: &[&str],
        precondition: Option<Formula>,
        effect: Option<Formula>,
    ) -> Action {
        Action {
            name: name.to_string(),
            parameters: parameters
                .iter()
                .map(|parameter| symbol(parameter))
                .collect(),
            preconditions: precondition,
            effects: effect,
        }
    }

    #[allow(clippy::type_complexity)]
    fn method(
        name: &str,
        task: &str,
        task_terms: &[&str],
        params: &[&str],
        precondition: Option<Formula>,
        subtasks: &[(&str, &[&str])],
    ) -> Method {
        Method {
            name: symbol(name),
            task: symbol(task),
            task_terms: task_terms.iter().map(|term| symbol(term)).collect(),
            params: params.iter().map(|param| symbol(param)).collect(),
            precondition,
            tn: Htn {
                subtasks: subtasks
                    .iter()
                    .map(|(task, terms)| Subtask {
                        id: None,
                        task: symbol(task),
                        terms: terms.iter().map(|term| symbol(term)).collect(),
                    })
                    .collect(),
                orderings: TaskOrdering::Total,
                constraints: None,
            },
        }
    }

    fn domain(methods: Vec<Method>, actions: Vec<Action>) -> Domain {
        Domain {
            name: "test".to_string(),
            requirements: Vec::new(),
            types: Vec::new(),
            constants: Vec::new(),
            predicates: Vec::new(),
            compound_tasks: Vec::new(),
            methods,
            actions,
            functions: Vec::new(),
        }
    }

    fn problem(tasks: &[(&str, &[&str])], objects: &[&str], init_state: Vec<Predicate>) -> Problem {
        problem_with_orderings(tasks, objects, init_state, TaskOrdering::Total)
    }

    fn problem_with_orderings(
        tasks: &[(&str, &[&str])],
        objects: &[&str],
        init_state: Vec<Predicate>,
        orderings: TaskOrdering,
    ) -> Problem {
        Problem {
            name: "p1".to_string(),
            domain_name: "test".to_string(),
            objects: objects.iter().map(|object| symbol(object)).collect(),
            init_state,
            goal: None,
            init_task_network: Some(InitialTaskNetwork {
                parameters: None,
                tn: Htn {
                    subtasks: tasks
                        .iter()
                        .map(|(task, terms)| Subtask {
                            id: None,
                            task: symbol(task),
                            terms: terms.iter().map(|term| symbol(term)).collect(),
                        })
                        .collect(),
                    orderings,
                    constraints: None,
                },
            }),
        }
    }

    fn predicate(name: &str, terms: &[&str]) -> Predicate {
        Predicate {
            name: name.to_string(),
            variables: terms.iter().map(|term| symbol(term)).collect(),
        }
    }

    #[test]
    fn detects_a_later_method_precondition_fed_by_an_earlier_subtask() {
        let domain = domain(
            vec![
                method(
                    "deliver-by-truck",
                    "deliver",
                    &["?p", "?l"],
                    &["?t"],
                    None,
                    &[("load", &["?p", "?t"])],
                ),
                method(
                    "use-loaded",
                    "use",
                    &["?p"],
                    &["?t"],
                    Some(atom("loaded", &["?p", "?t"])),
                    &[("consume", &["?p", "?t"])],
                ),
            ],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p", "?t"], None, None),
            ],
        );
        let problem = problem(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
        );

        let report =
            detect_liftability_gaps(&domain, &problem, &LiftabilityOptions::default()).unwrap();

        assert!(!report.truncated, "{report}");
        // Every object is a legal `?t`, so each of the three grounded `load`
        // instances matches the consumer's `loaded(p1, ?t)`.
        assert_eq!(report.gaps.len(), 3, "{report}");
        let gap = report
            .gaps
            .iter()
            .find(|gap| gap.atom == Atom::new("loaded", vec!["p1".into(), "t1".into()]))
            .expect("the loaded(p1, t1) gap");
        assert_eq!(gap.before, 0);
        assert_eq!(gap.after, 1);
        assert_eq!(gap.before_method, "deliver-by-truck");
        assert_eq!(gap.after_method, "use-loaded");
        assert_eq!(gap.consumer, ConsumerKind::Method);
        assert_eq!(gap.kind, GapKind::HiddenSatisfier);
        assert!(gap.produced_positive);
        assert!(gap.required_positive);
        assert_eq!(
            gap.witness,
            TaskNode {
                name: "load".to_string(),
                terms: vec!["p1".to_string(), "t1".to_string()],
            }
        );
        assert!(!gap.literal_initially_true);
        assert!(gap.required_everywhere);
        assert!(!gap.visible_producer);
        assert_eq!(report.involved_tasks(), BTreeSet::from([0, 1]));
    }

    #[test]
    fn detects_a_hidden_falsifier() {
        let domain = domain(
            vec![
                method(
                    "deliver-by-truck",
                    "deliver",
                    &["?p", "?l"],
                    &["?t"],
                    None,
                    &[("unload", &["?p", "?t"])],
                ),
                method(
                    "use-loaded",
                    "use",
                    &["?p"],
                    &["?t"],
                    Some(atom("loaded", &["?p", "?t"])),
                    &[("consume", &["?p", "?t"])],
                ),
            ],
            vec![
                action(
                    "unload",
                    &["?p", "?t"],
                    None,
                    Some(delete("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p", "?t"], None, None),
            ],
        );
        let problem = problem(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            vec![predicate("loaded", &["p1", "t1"])],
        );

        let report =
            detect_liftability_gaps(&domain, &problem, &LiftabilityOptions::default()).unwrap();

        let gap = report
            .gaps
            .iter()
            .find(|gap| gap.atom == Atom::new("loaded", vec!["p1".into(), "t1".into()]))
            .expect("the loaded(p1, t1) gap");
        assert_eq!(gap.kind, GapKind::HiddenFalsifier);
        assert!(!gap.produced_positive);
        assert!(gap.required_positive);
        assert!(gap.literal_initially_true);
    }

    #[test]
    fn reports_an_action_precondition_of_a_later_primitive_task() {
        let domain = domain(
            vec![method(
                "deliver-by-truck",
                "deliver",
                &["?p", "?l"],
                &["?t"],
                None,
                &[("load", &["?p", "?t"])],
            )],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action(
                    "consume",
                    &["?p", "?t"],
                    Some(atom("loaded", &["?p", "?t"])),
                    None,
                ),
            ],
        );
        let problem = problem(
            &[("deliver", &["p1", "l1"]), ("consume", &["p1", "t1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
        );

        let report =
            detect_liftability_gaps(&domain, &problem, &LiftabilityOptions::default()).unwrap();

        assert_eq!(report.gaps.len(), 1, "{report}");
        let gap = &report.gaps[0];
        assert_eq!(gap.consumer, ConsumerKind::Action);
        assert_eq!(gap.after_method, "consume");
        assert_eq!(gap.kind, GapKind::HiddenSatisfier);
        assert!(gap.required_everywhere);
    }

    #[test]
    fn recursive_methods_reach_a_fixpoint() {
        let domain = domain(
            vec![
                method(
                    "deliver-recursive",
                    "deliver",
                    &["?p", "?l"],
                    &["?t"],
                    None,
                    &[("deliver", &["?p", "?l"]), ("load", &["?p", "?t"])],
                ),
                method(
                    "use-loaded",
                    "use",
                    &["?p"],
                    &["?t"],
                    Some(atom("loaded", &["?p", "?t"])),
                    &[("consume", &["?p", "?t"])],
                ),
            ],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p", "?t"], None, None),
            ],
        );
        let problem = problem(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
        );

        let report =
            detect_liftability_gaps(&domain, &problem, &LiftabilityOptions::default()).unwrap();

        assert!(!report.truncated, "{report}");
        assert_eq!(report.gaps.len(), 3, "{report}");
        assert!(
            report
                .gaps
                .iter()
                .any(|gap| gap.atom == Atom::new("loaded", vec!["p1".into(), "t1".into()]))
        );
    }

    #[test]
    fn unrelated_preconditions_produce_no_gap() {
        let domain = domain(
            vec![
                method(
                    "deliver-by-truck",
                    "deliver",
                    &["?p", "?l"],
                    &["?t"],
                    None,
                    &[("load", &["?p", "?t"])],
                ),
                method(
                    "use-ready",
                    "use",
                    &["?p"],
                    &[],
                    Some(atom("ready", &["?p"])),
                    &[("consume", &["?p"])],
                ),
            ],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p"], None, None),
            ],
        );
        let problem = problem(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
        );

        let report =
            detect_liftability_gaps(&domain, &problem, &LiftabilityOptions::default()).unwrap();

        assert!(report.is_clear(), "{report}");
        assert!(!report.truncated);
        assert_eq!(report.orderings, vec![(0, 1)]);
    }

    fn deliver_and_use_domain(use_precondition: Formula) -> Domain {
        domain(
            vec![
                method(
                    "deliver-by-truck",
                    "deliver",
                    &["?p", "?l"],
                    &["?t"],
                    None,
                    &[("load", &["?p", "?t"])],
                ),
                method(
                    "use-loaded",
                    "use",
                    &["?p"],
                    &[],
                    Some(use_precondition),
                    &[("consume", &["?p"])],
                ),
            ],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p"], None, None),
            ],
        )
    }

    #[test]
    fn repairs_a_liftability_gap_with_an_implicit_ordering() {
        let domain = deliver_and_use_domain(atom("loaded", &["?p", "t1"]));
        let problem = problem_with_orderings(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
            TaskOrdering::Partial(Vec::new()),
        );

        assert_eq!(
            repair_orderings(&domain, &problem, &LiftabilityOptions::default()).unwrap(),
            vec![(0, 1)]
        );
    }

    #[test]
    fn no_repair_when_the_literal_already_holds() {
        let domain = deliver_and_use_domain(atom("loaded", &["?p", "t1"]));
        let problem = problem_with_orderings(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            vec![predicate("loaded", &["p1", "t1"])],
            TaskOrdering::Partial(Vec::new()),
        );

        assert!(
            repair_orderings(&domain, &problem, &LiftabilityOptions::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn no_repair_when_another_clause_avoids_the_literal() {
        let domain = deliver_and_use_domain(Formula::Or(vec![
            atom("loaded", &["?p", "t1"]),
            atom("ready", &["?p"]),
        ]));
        let problem = problem_with_orderings(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
            TaskOrdering::Partial(Vec::new()),
        );

        assert!(
            repair_orderings(&domain, &problem, &LiftabilityOptions::default())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn the_repair_reaches_the_evaluation_tree() {
        use crate::solver::evaluation_tree::{
            ConstraintNode, EvaluationOptions, build_evaluation_tree,
        };

        let domain = deliver_and_use_domain(atom("loaded", &["?p", "t1"]));
        let problem = problem_with_orderings(
            &[("deliver", &["p1", "l1"]), ("use", &["p1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
            TaskOrdering::Partial(Vec::new()),
        );
        let ordering = ConstraintNode::Ordering {
            before: 0,
            after: 1,
        };

        let plain =
            build_evaluation_tree(&domain, &problem, &EvaluationOptions::default()).unwrap();
        assert!(
            !plain.constraints.contains(&ordering),
            "without the repair the network has no ordering"
        );

        let repaired = build_evaluation_tree(
            &domain,
            &problem,
            &EvaluationOptions {
                implicit_orderings: true,
                ..EvaluationOptions::default()
            },
        )
        .unwrap();
        assert!(
            repaired.constraints.contains(&ordering),
            "the implicit ordering is part of the constraint graph"
        );
        // The incidence of the ordering joins both tasks, so the clusterer
        // sees a single component and the ordering is enforced downstream.
        let constraint = repaired
            .constraints
            .iter()
            .position(|constraint| *constraint == ordering)
            .unwrap();
        assert!(repaired.clusters.len() == 1);
        assert!(
            repaired
                .clusters
                .iter()
                .all(|cluster| cluster.n2s.contains(&constraint))
        );
    }

    /// A producer whose only subtask is the ground `load(p1, t1)`, so the
    /// merge can observe the shared atom without choosing a binding.
    fn ground_deliver_domain() -> Domain {
        domain(
            vec![
                method(
                    "deliver-by-truck",
                    "deliver",
                    &["?p", "?l"],
                    &[],
                    None,
                    &[("load", &["?p", "t1"])],
                ),
                method(
                    "use-loaded",
                    "use",
                    &["?p"],
                    &[],
                    Some(atom("loaded", &["?p", "t1"])),
                    &[("consume", &["?p"])],
                ),
            ],
            vec![
                action(
                    "load",
                    &["?p", "?t"],
                    None,
                    Some(atom("loaded", &["?p", "?t"])),
                ),
                action("consume", &["?p"], None, None),
            ],
        )
    }

    #[test]
    fn the_fallback_inserts_the_order_needed_to_refine() {
        use crate::solver::evaluation_tree::{
            EvaluationOptions, MethodSummary, RefinementOptions, SgplanMerger, evaluate,
            refine_plan,
        };
        use crate::solver::partition::bisbm::Grouping;
        use crate::solver::solve::solve_hierarchical_with_options;

        let domain = ground_deliver_domain();
        // `use` is the first task, so an unconstrained merge decodes it before
        // `deliver`. Only the implicit ordering makes the plan liftable.
        let problem = problem_with_orderings(
            &[("use", &["p1"]), ("deliver", &["p1", "l1"])],
            &["p1", "l1", "t1"],
            Vec::new(),
            TaskOrdering::Partial(Vec::new()),
        );
        let options = EvaluationOptions {
            method_summary: MethodSummary::PrimitiveOnly,
            groupings: vec![Grouping { n1s: vec![0] }, Grouping { n1s: vec![1] }],
            ..EvaluationOptions::default()
        };

        let mut merger = SgplanMerger::default();
        let evaluation = evaluate(&domain, &problem, &mut merger, &options).unwrap();
        assert!(
            evaluation.solved,
            "both leaves are solved and the merge holds"
        );
        assert_eq!(
            evaluation.plan.first().map(String::as_str),
            Some("use-loaded[use(p1)]"),
            "the unrepaired merge schedules the consumer first"
        );

        let refinement_options = RefinementOptions::default();
        assert!(
            refine_plan(&domain, &problem, &evaluation.plan, &refinement_options).is_err(),
            "the unaugmented plan is not liftable"
        );

        let repaired =
            solve_hierarchical_with_options(&domain, &problem, &refinement_options, &options)
                .expect("the implicit ordering repairs the refinement");
        assert!(!repaired.plan.is_empty());
    }
}
