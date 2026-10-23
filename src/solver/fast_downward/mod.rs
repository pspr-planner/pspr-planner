//! A compact implementation of the core of the Fast Downward planning system,
//! following Helmert, *The Fast Downward Planning System* (JAIR 26, 2006,
//! <https://jair.org/index.php/jair/article/view/10457>).
//!
//! # Scope
//!
//! The module covers the parts of the paper that are independent of PDDL
//! parsing and grounding:
//!
//! * **Multi-valued planning tasks** (Section 4): state variables with finite
//!   domains, operators with preconditions, conditional effects and costs, and
//!   layered axioms for derived variables. States are vectors with one value
//!   per variable; derived variables are re-evaluated after every operator
//!   application using the marking algorithm of Section 5.3.2.
//! * **Knowledge compilation** (Section 5): domain transition graphs,
//!   the causal graph (Definition 6), its goal-ancestor restriction and
//!   greedy cycle pruning (Section 5.2.2), and a successor generator
//!   (Definition 8) for fast applicable-operator lookup.
//! * **Heuristics** (Section 6.1 and 6.2): the FF heuristic with helpful
//!   actions, the causal graph heuristic `h_CG` (the Dijkstra-style
//!   computation of Fig. 18, evaluated bottom-up as in Fig. 17), and the
//!   `h_max` estimate.
//! * **Search** (Sections 6.3 and 6.4): greedy best-first search with a closed
//!   list, two open lists per heuristic for preferred operators, deferred
//!   heuristic evaluation, and multi-heuristic search with one open list per
//!   heuristic. The generic [`best_first_search`] engine is also used by the
//!   agglomeration search in [`crate::solver::partition::agglomeration`].
//!
//! Focused iterative-broadening search (Section 6.5) and the helped-transition
//! variant of the causal graph heuristic are not implemented; preferred
//! operators come from the FF heuristic.
//!
//! # Layout
//!
//! * `task`: the multi-valued task model and its semantics.
//! * `transitions`: domain transition graphs, the causal graph, and the cycle
//!   pruning algorithm.
//! * `successor`: the successor generator decision tree.
//! * `knowledge`: the compiled task tying the knowledge structures together.
//! * `heuristics`: `h_FF`, `h_CG`, `h_max`, and the relaxed planning graph.
//! * `search`: the generic best-first engine and the planning search.
//!
//! # Example
//!
//! ```
//! use pspr::solver::fast_downward::{search, HeuristicKind, SearchOptions, Task};
//!
//! // A robot at positions 0..3; moving right costs 1, moving left costs 2.
//! let task = Task::new(
//!     vec![("pos".into(), 3, false)],
//!     vec![0],
//!     vec![(0, 2)],
//!     vec![
//!         ("right".into(), vec![(0, 0)], vec![(vec![], 0, 1)], 1.0),
//!         ("right".into(), vec![(0, 1)], vec![(vec![], 0, 2)], 1.0),
//!     ],
//!     vec![],
//! )
//! .unwrap();
//! let outcome = search(&task, &[HeuristicKind::Ff], &SearchOptions::default())
//!     .unwrap()
//!     .unwrap();
//! assert_eq!(outcome.plan.len(), 2);
//! ```

mod heuristics;
mod knowledge;
mod search;
mod successor;
mod task;
mod transitions;

pub use heuristics::{
    HeuristicScratch, HeuristicValue, causal_graph_heuristic, causal_graph_heuristic_with,
    ff_heuristic, ff_heuristic_with, h_max, h_max_with,
};
pub use knowledge::CompiledTask;
pub use search::{
    Action, Evaluation, HeuristicKind, SearchOptions, SearchOutcome, SearchProblem,
    best_first_search, search,
};
pub use successor::SuccessorGenerator;
pub use task::{Axiom, Effect, Operator, Task, Variable};
pub use transitions::{CausalGraph, DomainTransitionGraph, PrunedCausalGraph, Transition};

/// A partial variable assignment, given as `(variable, value)` pairs.
pub type Conditions = Vec<(usize, usize)>;

/// An effect in the tuple-based [`Task::new`] constructor:
/// `(condition, variable, value)`.
pub type EffectSpec = (Conditions, usize, usize);

/// An operator in the tuple-based [`Task::new`] constructor:
/// `(name, precondition, effects, cost)`.
pub type OperatorSpec = (String, Conditions, Vec<EffectSpec>, f64);

/// An extended state: one value per variable, including derived variables.
pub type State = Vec<usize>;

/// Value assigned to derived variables before any axiom fires (the `⊥` value).
pub const UNDEFINED: usize = 0;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// A small transport task:
    ///
    /// * variable 0: truck position, domain 0..3
    /// * variable 1: package position, domain 0..3 (2 means "in the truck")
    /// * variable 2: derived `delivered`, true iff the package is at 2
    pub(crate) fn transport_task() -> Task {
        let variables = vec![
            ("truck".into(), 3, false),
            ("package".into(), 3, false),
            ("delivered".into(), 2, true),
        ];
        let operators = vec![
            // Move the truck from 0 to 1 and back.
            ("move-0-1".into(), vec![(0, 0)], vec![(vec![], 0, 1)], 1.0),
            ("move-1-0".into(), vec![(0, 1)], vec![(vec![], 0, 0)], 1.0),
            ("move-1-2".into(), vec![(0, 1)], vec![(vec![], 0, 2)], 1.0),
            ("move-2-1".into(), vec![(0, 2)], vec![(vec![], 0, 1)], 1.0),
            // Load the package at the truck's location (package position 2).
            (
                "load-0".into(),
                vec![(0, 0), (1, 0)],
                vec![(vec![], 1, 2)],
                1.0,
            ),
            (
                "load-1".into(),
                vec![(0, 1), (1, 1)],
                vec![(vec![], 1, 2)],
                1.0,
            ),
            // Unload the package at the truck's location.
            (
                "unload-1".into(),
                vec![(0, 1), (1, 2)],
                vec![(vec![], 1, 1)],
                1.0,
            ),
        ];
        let axioms = vec![Axiom {
            condition: vec![(1, 1)],
            variable: 2,
            value: 1,
            layer: 1,
        }];
        Task::new(variables, vec![0, 0, 0], vec![(2, 1)], operators, axioms).unwrap()
    }

    #[test]
    fn successor_generator_matches_brute_force() {
        let task = transport_task();
        let compiled = task.compile().unwrap();
        let mut states: Vec<State> = Vec::new();
        let mut frontier = vec![task.initial_state()];
        let mut seen: HashSet<State> = frontier.iter().cloned().collect();
        while let Some(state) = frontier.pop() {
            for operator in 0..task.operators.len() {
                if let Some(next) = task.apply(operator, &state)
                    && seen.insert(next.clone())
                {
                    frontier.push(next);
                }
            }
            states.push(state);
        }
        for state in states {
            let mut expected: Vec<usize> = (0..task.operators.len())
                .filter(|&operator| {
                    task.conditions_hold(&task.operators[operator].precondition, &state)
                })
                .collect();
            expected.sort_unstable();
            let mut actual = compiled.successor_generator.applicable(&state);
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn causal_graph_has_the_expected_arcs() {
        let task = transport_task();
        let compiled = task.compile().unwrap();
        // Loading and unloading require the truck and the package.
        assert!(compiled.causal_graph.arcs[0].contains_key(&1));
        assert!(!compiled.causal_graph.arcs[1].contains_key(&0));
        // The axiom makes `delivered` depend on the package.
        assert!(compiled.causal_graph.arcs[1].contains_key(&2));
        // The pruned graph is acyclic and agrees with the total order.
        for (source, targets) in compiled.pruned.arcs.iter().enumerate() {
            for &target in targets {
                assert!(compiled.pruned.levels[source] < compiled.pruned.levels[target]);
            }
        }
        assert!(
            compiled.pruned.relevant[0]
                && compiled.pruned.relevant[1]
                && compiled.pruned.relevant[2]
        );
    }

    #[test]
    fn axioms_are_evaluated() {
        let task = transport_task();
        let state = task.initial_state();
        assert_eq!(state[2], UNDEFINED);
        let mut moved = state.clone();
        moved[1] = 1;
        task.evaluate_axioms(&mut moved);
        assert_eq!(moved[2], 1);
    }

    #[test]
    fn ff_heuristic_counts_the_relaxed_plan() {
        let task = transport_task();
        let compiled = task.compile().unwrap();
        let initial = task.initial_state();
        let value = ff_heuristic(&compiled, &initial);
        // load-0, move-0-1, unload-1.
        assert_eq!(value.estimate, 3.0);
        assert!(!value.preferred.is_empty());
        assert!(
            value.preferred.contains(&0),
            "move-0-1 is helpful in the initial state"
        );
    }

    #[test]
    fn causal_graph_heuristic_computes_costs() {
        let task = transport_task();
        let compiled = task.compile().unwrap();
        let initial = task.initial_state();
        // The truck must move 0 -> 1 and the package must be loaded and
        // unloaded: load-0, move-0-1, unload-1.
        let estimate = causal_graph_heuristic(&compiled, &initial);
        assert!(
            (estimate - 3.0).abs() < 1e-9,
            "unexpected h_CG value {estimate}"
        );
    }

    #[test]
    fn best_first_search_solves_the_transport_task() {
        let task = transport_task();
        for heuristics in [
            vec![HeuristicKind::Ff],
            vec![HeuristicKind::CausalGraph],
            vec![HeuristicKind::Ff, HeuristicKind::CausalGraph],
        ] {
            let options = SearchOptions::default();
            let outcome = search(&task, &heuristics, &options).unwrap().unwrap();
            let final_state = task.execute(&outcome.plan).unwrap();
            assert!(task.is_goal(&final_state));
            assert_eq!(outcome.plan.len(), 3);
        }
    }

    #[test]
    fn deferred_evaluation_and_no_preferences_still_solve() {
        let task = transport_task();
        let options = SearchOptions {
            use_preferred_operators: false,
            deferred_evaluation: true,
            ..SearchOptions::default()
        };
        let outcome = search(&task, &[HeuristicKind::Ff], &options)
            .unwrap()
            .unwrap();
        assert!(task.is_goal(&task.execute(&outcome.plan).unwrap()));
    }

    #[test]
    fn invalid_tasks_are_rejected() {
        let variables = vec![("v".into(), 2, false)];
        // Initial value outside the domain.
        assert!(Task::new(variables.clone(), vec![2], vec![], vec![], vec![]).is_err());
        // Effect on a derived variable.
        let derived = vec![("v".into(), 2, true)];
        let bad_operator = vec![("o".into(), vec![], vec![(vec![], 0, 1)], 1.0)];
        assert!(Task::new(derived, vec![0], vec![], bad_operator, vec![]).is_err());
    }
}
