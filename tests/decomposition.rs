//! Integration tests for the neural-guidance seams.

use std::collections::HashSet;
use std::path::Path;

use pspr::parsing::parser::read_program_json;
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    AbstractPlanContext, AbstractPlanPermutation, AbstractPlanPermuter, DecompositionSession,
    DecompositionTraceSink, EvaluationOptions, MethodCandidate, MethodRanker,
    MethodRankingContext, MethodSummary, MethodTrace, RefinementOptions, SearchContext,
    SearchPolicy, SearchTrace, refine_plan_with_session,
    solve_hierarchical_with_session,
};
use pspr::solver::problem::load_problem;

#[derive(Debug, Default)]
struct Collector {
    methods: Vec<MethodTrace>,
    searches: Vec<SearchTrace>,
}

impl DecompositionTraceSink for Collector {
    fn method_trace(&mut self, trace: MethodTrace) {
        self.methods.push(trace);
    }

    fn search_trace(&mut self, trace: SearchTrace) {
        self.searches.push(trace);
    }
}

#[derive(Debug)]
struct ReversePolicy;

impl SearchPolicy for ReversePolicy {
    fn order_ready(&self, context: &SearchContext) -> Option<Vec<usize>> {
        let ready: Vec<usize> = (0..context.tasks.len())
            .filter(|&index| {
                context.remaining[index]
                    && !context.predecessors[index]
                        .iter()
                        .any(|before| context.remaining[*before])
            })
            .collect();
        ready
            .len()
            .gt(&1)
            .then(|| ready.into_iter().rev().collect())
    }
}

#[derive(Debug)]
struct WrongLengthRanker;

impl MethodRanker for WrongLengthRanker {
    fn rank(
        &self,
        _context: &MethodRankingContext,
        _candidates: &[MethodCandidate],
    ) -> Option<Vec<f64>> {
        Some(vec![1.0])
    }
}

fn load_fixture() -> (pspr::solver::domain::Domain, pspr::solver::problem::Problem) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/partial/program.json");
    let program = read_program_json(&path).expect("partial fixture should decode");
    let domain = process_domain(&program.domain).expect("fixture domain should process");
    let problem = load_problem(
        &domain,
        program.problem.as_ref().expect("fixture embeds a problem"),
    )
    .expect("fixture problem should load");
    (domain, problem)
}

fn load_permutation_fixture() -> (pspr::solver::domain::Domain, pspr::solver::problem::Problem) {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/partial/permutation_program.json");
    let program = read_program_json(&path).expect("permutation fixture should decode");
    let domain = process_domain(&program.domain).expect("fixture domain should process");
    let problem = load_problem(
        &domain,
        program.problem.as_ref().expect("fixture embeds a problem"),
    )
    .expect("fixture problem should load");
    (domain, problem)
}

#[derive(Debug)]
struct SwapPermuter;

impl AbstractPlanPermuter for SwapPermuter {
    fn propose(&self, context: &AbstractPlanContext) -> Option<Vec<AbstractPlanPermutation>> {
        if context.steps.len() != 2 {
            return None;
        }
        Some(vec![AbstractPlanPermutation {
            order: vec![1, 0],
            score: 1.0,
        }])
    }
}

#[test]
fn unordered_fixture_records_method_and_frontier_traces() {
    let (domain, problem) = load_fixture();
    let mut collector = Collector::default();
    let refinement = {
        let mut session = DecompositionSession {
            trace: Some(&mut collector),
            ..DecompositionSession::default()
        };
        refine_plan_with_session(
            &domain,
            &problem,
            &["process(item0)".to_string()],
            &RefinementOptions {
                max_groundings: 256,
                max_depth: 32,
                ..RefinementOptions::default()
            },
            &mut session,
        )
        .expect("the fixture should refine")
    };

    assert_eq!(refinement.plan.len(), 2);
    assert!(
        collector
            .methods
            .iter()
            .any(|trace| trace.candidates.len() == 2)
    );
    assert!(
        collector
            .searches
            .iter()
            .any(|trace| trace.ready.len() == 2)
    );
}

#[test]
fn policy_can_reverse_the_frontier_order() {
    let (domain, problem) = load_fixture();
    let mut collector = Collector::default();
    let refinement = {
        let mut session = DecompositionSession {
            search_policy: Some(&ReversePolicy),
            trace: Some(&mut collector),
            ..DecompositionSession::default()
        };
        refine_plan_with_session(
            &domain,
            &problem,
            &["process(item0)".to_string()],
            &RefinementOptions {
                max_groundings: 256,
                max_depth: 32,
                ..RefinementOptions::default()
            },
            &mut session,
        )
        .expect("the reversed policy should still refine")
    };

    assert_eq!(refinement.plan.len(), 2);
    let trace = collector
        .searches
        .iter()
        .find(|trace| trace.ready.len() == 2)
        .expect("two ready tasks");
    assert_eq!(trace.order, vec![trace.ready[1], trace.ready[0]]);
}

#[cfg(feature = "neural-base")]
#[test]
fn neural_checkpoint_guides_the_unordered_fixture() {
    use burn::module::Module;
    use burn::tensor::Device;
    use pspr::solver::evaluation_tree::{
        DECOMPOSITION_FEATURE_DIM, DECOMPOSITION_HIDDEN, DecompositionModel,
        NeuralDecompositionModel,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.burnpack");
    DecompositionModel::new(&Device::default())
        .save_file(&path)
        .unwrap();
    std::fs::write(
        path.with_extension("json"),
        format!(
            "{{\"task\":\"both\",\"hidden\":{DECOMPOSITION_HIDDEN},\"feature_dim\":{DECOMPOSITION_FEATURE_DIM}}}"
        ),
    )
    .unwrap();
    let model = NeuralDecompositionModel::from_path(&path).unwrap();

    let (domain, problem) = load_fixture();
    let mut session = DecompositionSession {
        method_ranker: Some(&model),
        search_policy: Some(&model),
        plan_permuter: Some(&model),
        trace: None,
    };
    let refinement = refine_plan_with_session(
        &domain,
        &problem,
        &["process(item0)".to_string()],
        &RefinementOptions {
            max_groundings: 256,
            max_depth: 32,
            ..RefinementOptions::default()
        },
        &mut session,
    )
    .expect("the neural checkpoint should still refine the fixture");
    assert_eq!(refinement.plan.len(), 2);
    assert_eq!(
        refinement.goal,
        pspr::solver::evaluation_tree::GoalStatus::Satisfied
    );
}

#[test]
fn malformed_ranker_output_falls_back_to_the_symbolic_order() {
    let (domain, problem) = load_fixture();
    let mut collector = Collector::default();
    let refinement = {
        let mut session = DecompositionSession {
            method_ranker: Some(&WrongLengthRanker),
            trace: Some(&mut collector),
            ..DecompositionSession::default()
        };
        refine_plan_with_session(
            &domain,
            &problem,
            &["process(item0)".to_string()],
            &RefinementOptions {
                max_groundings: 256,
                max_depth: 32,
                ..RefinementOptions::default()
            },
            &mut session,
        )
        .expect("the fallback should still refine")
    };

    assert_eq!(refinement.plan.len(), 2);
    let trace = collector
        .methods
        .iter()
        .find(|trace| trace.candidates.len() == 2)
        .expect("two method candidates");
    assert_eq!(trace.order, vec![0, 1]);
    assert_eq!(trace.order.iter().collect::<HashSet<_>>().len(), 2);
}

#[test]
fn permutation_retry_recovers_when_identity_order_fails() {
    let (domain, problem) = load_permutation_fixture();
    let evaluation_options = EvaluationOptions {
        method_summary: MethodSummary::Off,
        implicit_orderings: false,
        ..EvaluationOptions::default()
    };
    let refinement_options = RefinementOptions {
        max_groundings: 256,
        max_depth: 32,
        ..RefinementOptions::default()
    };

    // Baseline session (no permuter) should fail to refine this instance.
    let baseline = solve_hierarchical_with_session(
        &domain,
        &problem,
        &refinement_options,
        &evaluation_options,
        &mut DecompositionSession::default(),
    );
    assert!(baseline.is_err(), "baseline should fail without permutation retry");

    let mut session = DecompositionSession {
        plan_permuter: Some(&SwapPermuter),
        ..DecompositionSession::default()
    };
    let refinement = solve_hierarchical_with_session(
        &domain,
        &problem,
        &refinement_options,
        &evaluation_options,
        &mut session,
    )
    .expect("permutation retry should recover");
    assert!(
        refinement
            .abstract_plan
            .first()
            .is_some_and(|step| step.contains("second")),
        "the recovered abstract plan should execute second before first"
    );
    assert_eq!(refinement.goal, pspr::solver::evaluation_tree::GoalStatus::Absent);
}
