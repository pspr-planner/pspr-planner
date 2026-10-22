//! Evaluation tree for HTN tasks.
//!
//! The pipeline follows the constraint-partitioning idea of SGPlan (Hsu et al.,
//! *Constraint partitioning in penalty formulations for solving temporal
//! planning problems*, Artificial Intelligence 170(3), 2006) and combines the
//! modules of this crate:
//!
//! 1. **Incidence graph.** The tasks of the initial task network are the
//!    type-I nodes of a bipartite graph. The type-II nodes are the constraints
//!    they touch: the ground atoms of the matching action and method schemas
//!    (preconditions and, for actions, effects) plus the ordering and equality
//!    constraints of the task network.
//! 2. **Spectral clustering.** [`bisbm::cluster`](crate::solver::partition::bisbm::cluster)
//!    partitions the tasks into communities. By default the connected
//!    components of the incidence graph are passed as required groupings, so
//!    independent subproblems are never mixed; explicit groupings can be
//!    supplied through [`EvaluationOptions::groupings`].
//! 3. **Evaluation tree.** The clusters are turned into a dendrogram with the
//!    ILP of Gilpin et al. and the scheduling ILP of
//!    [`crate::solver::partition::agglomeration`]. The leaves of the tree are
//!    the clusters; the internal nodes are merges of subproblems.
//! 4. **Leaf evaluation.** Each cluster is encoded as a multi-valued planning
//!    task containing only its *local* constraints (constraints adjacent to a
//!    single cluster) and solved with Fast Downward
//!    ([`crate::solver::fast_downward`]). Constraints shared with other
//!    clusters are dropped from the leaf models and handed to the merge nodes.
//! 5. **Merging.** Every shared constraint is attached to the lowest common
//!    ancestor of the clusters it touches. The internal nodes are evaluated in
//!    schedule order and their shared constraints, together with the child
//!    solutions, are passed to a [`MergeSolver`]. The default
//!    [`SgplanMerger`] relaxes the SGPlan penalty formulation to a smooth
//!    MINLP, optimizes the interleaving of the children with Adam over a
//!    continuation of penalty spaces, adapts the constraint weights while it
//!    runs, and validates the decoded merge exactly.
//!
//! # Layout
//!
//! * `graph`: grounding of the initial task network and the incidence graph.
//! * `clustering`: spectral clustering and the merge tree over clusters.
//! * `leaf`: encoding and solving a cluster as a planning task.
//! * `merge`: the merge-solver interface, the relaxed SGPlan MINLP, Adam, and
//!   the adaptive penalty continuation.
//!
//! # Modelling limitations
//!
//! * Only the conjunctive fragment of the formula language is encoded in the
//!   leaf planning tasks (`Empty`, `Atom`, `Not(Atom)` and conjunctions of
//!   these). Disjunctive, quantified, probabilistic and equality formulas still
//!   connect tasks to their atoms for the incidence graph, but are not imposed
//!   as leaf preconditions; they are counted in
//!   [`EvaluationTree::unmodeled`].
//! * The initial task network is grounded over the constants and objects of the
//!   problem (respecting the type hierarchy) up to
//!   [`EvaluationOptions::max_groundings`] assignments.
//! * Compound tasks are represented in a leaf by a synthetic operator that
//!   requires the method precondition and marks the task as executed. Methods
//!   are not decomposed recursively; their subtasks are not added to the task
//!   universe.
//! * Ordering constraints inside a cluster are enforced by making the
//!   `executed` variable of a task a precondition of every later task in the
//!   cluster.
//! * The merge step can only repair shared constraints by interleaving the
//!   child plans; shared atoms and equalities that no child establishes are
//!   reported as errors instead of causing replanning.

use std::collections::{BTreeSet, HashMap};

use anyhow::Result;

use crate::solver::domain::Domain;
use crate::solver::fast_downward::{self, HeuristicKind, SearchOptions};
use crate::solver::partition::bisbm::Grouping;
use crate::solver::problem::Problem;

mod clustering;
mod graph;
mod leaf;
mod merge;

pub use clustering::EvaluationTree;
pub use graph::{Atom, ConstraintNode, TaskConstraintGraph, TaskNode, build_task_constraint_graph};
pub use merge::{
    AdamOptions, MergeOutcome, MergeProblem, MergeRecord, MergeSolver, PartialSolution,
    PenaltyReport, PenaltySpace, SgplanMerger,
};

use clustering::{attach_shared_constraints, build_agglomeration, cluster_tasks};
use leaf::{build_leaf_task, initial_atoms};

/// Options for building and evaluating the tree.
#[derive(Debug, Clone)]
pub struct EvaluationOptions {
    /// Explicit task groupings for the spectral clusterer. When empty, the
    /// connected components of the incidence graph are used.
    pub groupings: Vec<Grouping>,
    /// Number of agglomeration levels; defaults to `clusters - 1`.
    pub levels: Option<usize>,
    /// Weight of the scheduling cost in the agglomeration objective.
    pub lambda: f64,
    /// Also run the ILP-style merge search of the agglomeration module.
    pub use_search: bool,
    /// Expansion limit of that search.
    pub search_expansions: usize,
    /// Maximum number of groundings of the initial task network.
    pub max_groundings: usize,
    /// Expansion limit of the Fast Downward search inside a leaf.
    pub leaf_max_expansions: usize,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        EvaluationOptions {
            groupings: Vec::new(),
            levels: None,
            lambda: 0.0,
            use_search: false,
            search_expansions: 1_000,
            max_groundings: 256,
            leaf_max_expansions: 100_000,
        }
    }
}

/// Build the evaluation tree of `problem`.
pub fn build_evaluation_tree(
    domain: &Domain,
    problem: &Problem,
    options: &EvaluationOptions,
) -> Result<EvaluationTree> {
    let graph = build_task_constraint_graph(domain, problem, options.max_groundings)?;
    let clusters = cluster_tasks(&graph, options)?;

    if clusters.is_empty() {
        return Ok(EvaluationTree {
            tasks: graph.tasks,
            constraints: graph.constraints,
            clusters,
            shared: Vec::new(),
            local: Vec::new(),
            agglomeration: None,
            merge_constraints: HashMap::new(),
            unmodeled: graph.unmodeled,
        });
    }

    let mut uses = vec![0usize; graph.constraints.len()];
    for cluster in &clusters {
        for &constraint in &cluster.n2s {
            uses[constraint] += 1;
        }
    }
    let shared: Vec<usize> = (0..graph.constraints.len())
        .filter(|&constraint| uses[constraint] > 1)
        .collect();
    let local: Vec<Vec<usize>> = clusters
        .iter()
        .map(|cluster| {
            cluster
                .n2s
                .iter()
                .copied()
                .filter(|&constraint| uses[constraint] <= 1)
                .collect()
        })
        .collect();

    let agglomeration = build_agglomeration(&clusters, options)?;
    let merge_constraints = match &agglomeration {
        Some(agglomeration) => attach_shared_constraints(agglomeration, &clusters, &shared),
        None => HashMap::new(),
    };

    Ok(EvaluationTree {
        tasks: graph.tasks,
        constraints: graph.constraints,
        clusters,
        shared,
        local,
        agglomeration,
        merge_constraints,
        unmodeled: graph.unmodeled,
    })
}

/// The result of evaluating the tree.
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub tree: EvaluationTree,
    pub leaves: Vec<PartialSolution>,
    pub merges: Vec<MergeRecord>,
    pub plan: Vec<String>,
    pub solved: bool,
}

/// Build the tree and evaluate it with the given merge solver.
pub fn evaluate(
    domain: &Domain,
    problem: &Problem,
    merger: &mut dyn MergeSolver,
    options: &EvaluationOptions,
) -> Result<Evaluation> {
    let tree = build_evaluation_tree(domain, problem, options)?;
    if tree.tasks.is_empty() {
        return Ok(Evaluation {
            tree,
            leaves: Vec::new(),
            merges: Vec::new(),
            plan: Vec::new(),
            solved: false,
        });
    }
    let atoms = initial_atoms(problem);
    let mut leaves = Vec::with_capacity(tree.clusters.len());
    for cluster_index in 0..tree.clusters.len() {
        let leaf = build_leaf_task(domain, &tree, cluster_index, &atoms)?;
        let search = fast_downward::search(
            &leaf.task,
            &[HeuristicKind::Ff],
            &SearchOptions {
                max_expansions: options.leaf_max_expansions,
                ..SearchOptions::default()
            },
        )?;
        let solution = match search {
            Some(outcome) => {
                let plan = outcome
                    .plan
                    .iter()
                    .map(|&operator| leaf.task.operators[operator].name.clone())
                    .collect();
                let plan_tasks = outcome
                    .plan
                    .iter()
                    .map(|&operator| leaf.operator_tasks[operator])
                    .collect();
                let final_atoms = leaf
                    .local_atoms
                    .iter()
                    .enumerate()
                    .filter(|(variable, _)| outcome.state[*variable] == 1)
                    .map(|(_, atom)| atom.to_string())
                    .collect();
                let mut achieved = BTreeSet::new();
                for &operator in &outcome.plan {
                    achieved.extend(leaf.effect_atoms[operator].iter().map(Atom::to_string));
                }
                PartialSolution {
                    label: format!("leaf {cluster_index}"),
                    plan,
                    plan_tasks,
                    final_atoms,
                    achieved,
                    solved: true,
                }
            }
            None => PartialSolution {
                label: format!("leaf {cluster_index}"),
                plan: Vec::new(),
                plan_tasks: Vec::new(),
                final_atoms: BTreeSet::new(),
                achieved: BTreeSet::new(),
                solved: false,
            },
        };
        leaves.push(solution);
    }

    let mut merges = Vec::new();
    let (plan, solved) = match &tree.agglomeration {
        None => {
            let plan = leaves.iter().flat_map(|leaf| leaf.plan.clone()).collect();
            let solved = leaves.iter().all(|leaf| leaf.solved);
            (plan, solved)
        }
        Some(agglomeration) => {
            let dendrogram = &agglomeration.dendrogram;
            let mut partial: HashMap<usize, PartialSolution> = HashMap::new();
            for &evaluation_id in &agglomeration.schedule.order {
                let node = dendrogram.leaves + evaluation_id;
                let mut children = Vec::new();
                for &child in &dendrogram.children[node] {
                    if child < dendrogram.leaves {
                        children.push(leaves[child].clone());
                    } else {
                        let sibling =
                            partial.get(&(child - dendrogram.leaves)).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "merge node {child} was scheduled after its parent {node}"
                                )
                            })?;
                        children.push(sibling.clone());
                    }
                }
                let shared = tree
                    .merge_constraints
                    .get(&evaluation_id)
                    .cloned()
                    .unwrap_or_default();
                let shared_constraints = shared
                    .iter()
                    .map(|&constraint| tree.constraints[constraint].clone())
                    .collect();
                let outcome = merger.merge(&MergeProblem {
                    node: evaluation_id,
                    shared_constraints,
                    children,
                    initial_atoms: atoms.clone(),
                })?;
                merges.push(MergeRecord {
                    node: evaluation_id,
                    shared,
                    plan: outcome.plan.clone(),
                    solved: outcome.solved,
                });
                partial.insert(
                    evaluation_id,
                    PartialSolution {
                        label: format!("merge {evaluation_id}"),
                        plan: outcome.plan,
                        plan_tasks: outcome.plan_tasks,
                        final_atoms: outcome.final_atoms,
                        achieved: BTreeSet::new(),
                        solved: outcome.solved,
                    },
                );
            }
            let root = dendrogram.root - dendrogram.leaves;
            let root = partial
                .get(&root)
                .ok_or_else(|| anyhow::anyhow!("the root merge node was not scheduled"))?;
            (
                root.plan.clone(),
                root.solved && leaves.iter().all(|leaf| leaf.solved),
            )
        }
    };

    Ok(Evaluation {
        tree,
        leaves,
        merges,
        plan,
        solved,
    })
}

#[cfg(test)]
mod tests {
    use crate::parsing::json::{
        Action, Formula, Htn, InitialTaskNetwork, Predicate, Subtask, Symbol, TaskOrdering,
    };

    use super::*;

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

    fn transport_domain() -> Domain {
        Domain {
            name: "transport".to_string(),
            requirements: Vec::new(),
            types: vec![symbol("place", None), symbol("truck", None)],
            constants: Vec::new(),
            predicates: vec![
                Predicate {
                    name: "at".to_string(),
                    variables: vec![symbol("?p", Some("place"))],
                },
                Predicate {
                    name: "road".to_string(),
                    variables: vec![symbol("?a", Some("place")), symbol("?b", Some("place"))],
                },
            ],
            compound_tasks: Vec::new(),
            methods: Vec::new(),
            actions: vec![Action {
                name: "drive".to_string(),
                parameters: vec![symbol("?from", Some("place")), symbol("?to", Some("place"))],
                preconditions: Some(Formula::And(vec![
                    Formula::Atom(atom("at", &["?from"])),
                    Formula::Atom(atom("road", &["?from", "?to"])),
                ])),
                effects: Some(Formula::Atom(atom("at", &["?to"]))),
            }],
            functions: Vec::new(),
        }
    }

    fn transport_problem(tasks: Vec<Subtask>, orderings: TaskOrdering) -> Problem {
        Problem {
            name: "p".to_string(),
            domain_name: "transport".to_string(),
            objects: vec![
                symbol("a", Some("place")),
                symbol("b", Some("place")),
                symbol("c", Some("place")),
            ],
            init_state: vec![
                atom("at", &["a"]),
                atom("road", &["a", "b"]),
                atom("road", &["b", "c"]),
            ],
            goal: None,
            init_task_network: Some(InitialTaskNetwork {
                parameters: None,
                tn: Htn {
                    subtasks: tasks,
                    orderings,
                    constraints: None,
                },
            }),
        }
    }

    fn two_drives() -> Problem {
        transport_problem(
            vec![
                subtask("t1", "drive", &["a", "b"]),
                subtask("t2", "drive", &["b", "c"]),
            ],
            TaskOrdering::Total,
        )
    }

    #[test]
    fn incidence_links_shared_atoms_to_both_tasks() {
        let domain = transport_domain();
        let problem = two_drives();
        let graph = build_task_constraint_graph(&domain, &problem, 10).unwrap();
        assert_eq!(graph.tasks.len(), 2);
        let at_b = graph
            .constraints
            .iter()
            .position(|constraint| {
                *constraint == ConstraintNode::Atom(Atom::new("at", vec!["b".to_string()]))
            })
            .expect("at(b) is a constraint");
        assert!(graph.incidence[0][at_b], "drive(a, b) touches at(b)");
        assert!(graph.incidence[1][at_b], "drive(b, c) touches at(b)");
        assert_eq!(graph.unmodeled, 0);
    }

    #[test]
    fn forced_groupings_expose_shared_constraints() {
        let domain = transport_domain();
        let problem = two_drives();
        let options = EvaluationOptions {
            groupings: vec![Grouping { n1s: vec![0] }, Grouping { n1s: vec![1] }],
            ..EvaluationOptions::default()
        };
        let tree = build_evaluation_tree(&domain, &problem, &options).unwrap();
        assert_eq!(tree.clusters.len(), 2);
        assert_eq!(tree.shared.len(), 2, "at(b) and the ordering are shared");
        assert!(
            tree.shared.iter().any(|&constraint| {
                tree.constraints[constraint]
                    == ConstraintNode::Atom(Atom::new("at", vec!["b".to_string()]))
            }),
            "at(b) is shared"
        );
        assert!(tree.shared.iter().any(|&constraint| {
            matches!(
                tree.constraints[constraint],
                ConstraintNode::Ordering {
                    before: 0,
                    after: 1
                }
            )
        }));
        assert!(tree.agglomeration.is_some());
        assert_eq!(tree.agglomeration.as_ref().unwrap().schedule.order.len(), 1);
    }

    #[test]
    fn pipeline_merges_shared_constraints() {
        let domain = transport_domain();
        let problem = two_drives();
        let options = EvaluationOptions {
            groupings: vec![Grouping { n1s: vec![0] }, Grouping { n1s: vec![1] }],
            ..EvaluationOptions::default()
        };
        let mut merger = SgplanMerger::default();
        let evaluation = evaluate(&domain, &problem, &mut merger, &options).unwrap();
        assert_eq!(evaluation.leaves.len(), 2);
        assert!(evaluation.leaves.iter().all(|leaf| leaf.solved));
        assert_eq!(evaluation.merges.len(), 1);
        assert_eq!(evaluation.merges[0].shared.len(), 2);
        assert!(evaluation.solved);
        assert_eq!(
            evaluation.plan,
            vec!["drive(a, b)".to_string(), "drive(b, c)".to_string()]
        );
    }

    #[test]
    fn default_tree_solves_each_component_and_respects_order() {
        let domain = transport_domain();
        let problem = two_drives();
        let mut merger = SgplanMerger::default();
        let evaluation = evaluate(
            &domain,
            &problem,
            &mut merger,
            &EvaluationOptions::default(),
        )
        .unwrap();
        assert!(evaluation.solved);
        assert_eq!(
            evaluation.plan,
            vec!["drive(a, b)".to_string(), "drive(b, c)".to_string()]
        );
    }

    #[test]
    fn unsolvable_leaf_is_reported() {
        let domain = transport_domain();
        let mut problem = two_drives();
        problem
            .init_state
            .retain(|predicate| predicate != &atom("road", &["a", "b"]));
        let mut merger = SgplanMerger::default();
        let evaluation = evaluate(
            &domain,
            &problem,
            &mut merger,
            &EvaluationOptions::default(),
        )
        .unwrap();
        assert!(!evaluation.solved);
        assert!(evaluation.plan.is_empty());
    }

    #[test]
    fn grounding_respects_types() {
        let domain = transport_domain();
        let mut problem =
            transport_problem(vec![subtask("t1", "deliver", &["?p"])], TaskOrdering::Total);
        problem.objects.push(symbol("truck", Some("truck")));
        problem.init_task_network = Some(InitialTaskNetwork {
            parameters: Some(vec![symbol("?p", Some("place"))]),
            tn: Htn {
                subtasks: vec![subtask("t1", "deliver", &["?p"])],
                orderings: TaskOrdering::Total,
                constraints: None,
            },
        });
        let graph = build_task_constraint_graph(&domain, &problem, 10).unwrap();
        let terms: Vec<_> = graph.tasks.iter().map(|task| task.terms.clone()).collect();
        assert_eq!(
            terms,
            vec![
                vec!["a".to_string()],
                vec!["b".to_string()],
                vec!["c".to_string()]
            ]
        );
        assert!(
            graph
                .tasks
                .iter()
                .all(|task| task.terms != vec!["truck".to_string()])
        );
    }
}
