//! Temporary diagnostic: report the size of the collected decomposition traces.
use std::path::PathBuf;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    DecompositionSession, DecompositionTraceSink, EvaluationOptions, MethodSummary, MethodTrace,
    RefinementOptions, SearchTrace, SgplanMerger, evaluate, refine_plan_with_session,
};
use pspr::solver::problem::load_problem;

#[derive(Debug, Default)]
struct Sizes {
    methods: Vec<MethodTrace>,
    searches: Vec<SearchTrace>,
}

impl DecompositionTraceSink for Sizes {
    fn method_trace(&mut self, trace: MethodTrace) {
        self.methods.push(trace);
    }
    fn search_trace(&mut self, trace: SearchTrace) {
        self.searches.push(trace);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (domain_path, problem_path) = match args.as_slice() {
        [domain, problem] => (domain.clone(), problem.clone()),
        _ => {
            eprintln!("usage: trace_sizes <domain.hddl> <problem.hddl>");
            std::process::exit(2);
        }
    };
    let parser = ParserConfig {
        executable: PathBuf::from(
            std::env::var_os("HDDL_PARSER").unwrap_or_else(|| "hddl_analyzer".into()),
        ),
    };
    let program = parse_program(
        &parser,
        &PathBuf::from(&domain_path),
        Some(&PathBuf::from(&problem_path)),
    )
    .expect("parse");
    let domain = process_domain(&program.domain).expect("domain");
    let problem = load_problem(&domain, program.problem.as_ref().expect("problem")).expect("load");

    let evaluation_options = EvaluationOptions {
        max_groundings: 32_768,
        leaf_max_expansions: 100_000,
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };
    let mut merger = SgplanMerger::default().without_initializer();
    let evaluation = evaluate(&domain, &problem, &mut merger, &evaluation_options).expect("eval");
    let options = RefinementOptions {
        max_groundings: 32_768,
        max_depth: 64,
    };

    let mut sizes = Sizes::default();
    let mut session = DecompositionSession {
        trace: Some(&mut sizes),
        ..DecompositionSession::default()
    };
    let refinement = refine_plan_with_session(
        &domain,
        &problem,
        &evaluation.plan,
        &options,
        &mut session,
    );

    let mut counts: Vec<usize> = sizes
        .methods
        .iter()
        .map(|trace| trace.candidates.len())
        .collect();
    counts.sort_unstable();
    let method_count = counts.len();
    let candidate_sum: usize = counts.iter().sum();
    let candidate_max = counts.last().copied().unwrap_or(0);
    let mut search_tasks: Vec<usize> = sizes
        .searches
        .iter()
        .map(|trace| trace.context.tasks.len())
        .collect();
    search_tasks.sort_unstable();
    let search_states: usize = sizes
        .searches
        .iter()
        .map(|trace| trace.context.state.len())
        .sum();

    println!("plan steps: {}", evaluation.plan.len());
    println!(
        "refinement: {:?}",
        refinement.as_ref().map(|refinement| refinement.plan.len())
    );
    println!(
        "method traces: {method_count}, candidates total {candidate_sum}, max {candidate_max}, \
         median {:?}",
        counts.get(method_count / 2)
    );
    // Dense adjacency of one method graph: (candidates + 2)^2 * 4 bytes.
    let dense = |count: usize| (count + 2) * (count + 2) * 4;
    println!(
        "dense adjacency: total {} MiB, max {} MiB",
        counts.iter().map(|&count| dense(count)).sum::<usize>() / (1024 * 1024),
        dense(candidate_max) / (1024 * 1024)
    );
    println!(
        "search traces: {}, in-cluster tasks max {:?}, state atoms total {search_states}",
        sizes.searches.len(),
        search_tasks.last()
    );
}
