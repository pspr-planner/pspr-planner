//! Phase-level benchmark of the hierarchical solver.
//!
//! Usage:
//!     profile_phases <domain.hddl> <problem.hddl> [repeats]
//!
//! Parses the input once, then reports the median wall time of
//! `build_evaluation_tree`, `evaluate`, `refine_plan` and
//! `validate_refinement`. `HDDL_PARSER` selects the parser executable.

use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    EvaluationOptions, MethodSummary, RefinementOptions, SgplanMerger, build_evaluation_tree,
    evaluate, refine_plan, validate_refinement,
};
use pspr::solver::problem::load_problem;

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let (domain_path, problem_path, repeats) = match arguments.as_slice() {
        [domain, problem] => (domain, problem, 3),
        [domain, problem, repeats] => (
            domain,
            problem,
            repeats.parse().expect("repeats must be an integer"),
        ),
        _ => {
            eprintln!("usage: profile_phases <domain.hddl> <problem.hddl> [repeats]");
            std::process::exit(2);
        }
    };

    let parser = env::var_os("HDDL_PARSER").unwrap_or_else(|| "hddl_analyzer".into());
    let config = ParserConfig {
        executable: PathBuf::from(parser),
    };
    let program = parse_program(
        &config,
        &PathBuf::from(domain_path),
        Some(&PathBuf::from(problem_path)),
    )
    .expect("the parser failed");
    let domain = process_domain(&program.domain).expect("domain translation failed");
    let problem = load_problem(
        &domain,
        program.problem.as_ref().expect("no problem in the input"),
    )
    .expect("problem translation failed");

    let max_groundings = env::var("PSPR_MAX_GROUNDINGS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let evaluation_options = EvaluationOptions {
        method_summary: MethodSummary::PrimitiveOnly,
        max_groundings,
        ..EvaluationOptions::default()
    };
    let refinement_options = RefinementOptions::default();

    let mut tree_times = Vec::new();
    let mut evaluate_times = Vec::new();
    let mut refine_times = Vec::new();
    let mut validate_times = Vec::new();
    let mut shape = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut refine_failed = false;
    let mut validate_failed = false;

    for _ in 0..repeats {
        let started = Instant::now();
        let tree = build_evaluation_tree(&domain, &problem, &evaluation_options)
            .expect("tree construction failed");
        tree_times.push(started.elapsed().as_secs_f64() * 1e3);

        let started = Instant::now();
        let mut merger = SgplanMerger::default();
        let evaluation = evaluate(&domain, &problem, &mut merger, &evaluation_options)
            .expect("evaluation failed");
        evaluate_times.push(started.elapsed().as_secs_f64() * 1e3);
        shape.0 = tree.tasks.len();
        shape.1 = tree.constraints.len();
        shape.2 = tree.clusters.len();
        shape.3 = evaluation.plan.len();

        let started = Instant::now();
        let refinement = refine_plan(&domain, &problem, &evaluation.plan, &refinement_options);
        refine_times.push(started.elapsed().as_secs_f64() * 1e3);
        let Ok(refinement) = refinement else {
            refine_failed = true;
            break;
        };

        let started = Instant::now();
        let report = validate_refinement(&domain, &problem, &refinement, &refinement_options);
        validate_times.push(started.elapsed().as_secs_f64() * 1e3);
        let steps = match &report {
            Ok(report) => report.steps,
            Err(_) => {
                validate_failed = true;
                0
            }
        };
        shape.4 = steps;
        black_box((&tree, &evaluation, &refinement, &report));
    }

    let refined = if refine_failed {
        "refine failed".to_string()
    } else if validate_failed {
        format!(
            "refine {:.3} ms, validate {:.3} ms (invalid)",
            median(&mut refine_times),
            median(&mut validate_times)
        )
    } else {
        format!(
            "refine {:.3} ms, validate {:.3} ms",
            median(&mut refine_times),
            median(&mut validate_times)
        )
    };
    println!(
        "{}: tasks={} constraints={} clusters={} abstract_steps={} plan_steps={} | \
         tree {:.3} ms, evaluate {:.3} ms, {}",
        problem.name,
        shape.0,
        shape.1,
        shape.2,
        shape.3,
        shape.4,
        median(&mut tree_times),
        median(&mut evaluate_times),
        refined,
    );
}
