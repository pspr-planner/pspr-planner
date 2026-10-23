//! Diagnose hierarchical refinement under multiple options for one instance.
//!
//! Usage:
//!   diagnose_refine_modes <domain.hddl> <problem.hddl>

use std::path::PathBuf;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    DecompositionSession, EvaluationOptions, MethodSummary, RefinementDiagnostics,
    RefinementOptions,
};
use pspr::solver::problem::load_problem;

fn run_mode(
    label: &str,
    domain: &pspr::solver::domain::Domain,
    problem: &pspr::solver::problem::Problem,
    method_summary: MethodSummary,
    implicit_orderings: bool,
    max_groundings: usize,
    max_depth: usize,
) {
    let evaluation = EvaluationOptions {
        method_summary,
        max_groundings,
        implicit_orderings,
        ..EvaluationOptions::default()
    };
    let refinement = RefinementOptions {
        max_groundings,
        max_depth,
        ..RefinementOptions::default()
    };
    let start = std::time::Instant::now();
    let mut diagnostics = RefinementDiagnostics::default();
    let result = {
        let mut session = DecompositionSession {
            diagnostics: Some(&mut diagnostics),
            ..DecompositionSession::default()
        };
        pspr::solver::evaluation_tree::solve_hierarchical_with_session(
            domain,
            problem,
            &refinement,
            &evaluation,
            &mut session,
        )
        .map_err(|error| format!("{error:#}"))
    };
    match result {
        Ok(r) => println!(
            "{label}: solved goal={:?} actions={} methods={} abstract_steps={} in {:?}",
            r.goal,
            r.plan.len(),
            r.method_count(),
            r.abstract_plan.len(),
            start.elapsed()
        ),
        Err(error) => println!("{label}: failed in {:?}: {error}", start.elapsed()),
    }
    println!(
        "  stats: time_budget={} expansion_budget={} depth={} cycle={} no_applicable={} constraint_reject={} ready_empty={} abstract_step_failed={}",
        diagnostics.budget_time_exhausted,
        diagnostics.budget_expansions_exhausted,
        diagnostics.depth_exhausted,
        diagnostics.cycle_pruned,
        diagnostics.no_applicable_method,
        diagnostics.network_constraint_rejected,
        diagnostics.partial_ready_empty,
        diagnostics.abstract_step_failed,
    );
    let mut dead_ends: Vec<(&String, &u64)> = diagnostics.no_applicable_tasks.iter().collect();
    dead_ends.sort_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
    let summary: Vec<String> = dead_ends
        .iter()
        .take(6)
        .map(|(task, count)| format!("{task}={count}"))
        .collect();
    if !summary.is_empty() {
        println!("  dead-end tasks: {}", summary.join(" "));
    }
    for sample in &diagnostics.dead_end_samples {
        println!("  dead-end sample: {sample}");
    }
    if !diagnostics.no_instance_tasks.is_empty() {
        let mut items: Vec<(&String, &u64)> = diagnostics.no_instance_tasks.iter().collect();
        items.sort_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
        let text: Vec<String> = items
            .iter()
            .take(6)
            .map(|(task, count)| format!("{task}={count}"))
            .collect();
        println!("  no-instance tasks: {}", text.join(" "));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [domain_path, problem_path] = args.as_slice() else {
        eprintln!("usage: diagnose_refine_modes <domain> <problem>");
        std::process::exit(2);
    };

    let parser = std::env::var_os("HDDL_PARSER").unwrap_or_else(|| "hddl_analyzer".into());
    let config = ParserConfig {
        executable: PathBuf::from(parser),
    };
    let program = parse_program(
        &config,
        &PathBuf::from(domain_path),
        Some(&PathBuf::from(problem_path)),
    )
    .unwrap_or_else(|error| panic!("parse failed: {error}"));

    let domain = process_domain(&program.domain).unwrap_or_else(|error| panic!("domain: {error}"));
    let problem = load_problem(
        &domain,
        program
            .problem
            .as_ref()
            .unwrap_or_else(|| panic!("missing problem")),
    )
    .unwrap_or_else(|error| panic!("problem: {error}"));

    let budget = std::env::var("PSPR_MAX_GROUNDINGS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);
    let max_depth = std::env::var("PSPR_MAX_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);

    println!(
        "instance={} budget={} depth={}",
        problem.name, budget, max_depth
    );
    {
        let refinement = RefinementOptions {
            max_groundings: budget,
            max_depth,
            ..RefinementOptions::default()
        };
        let mut diagnostics = RefinementDiagnostics::default();
        let start = std::time::Instant::now();
        match pspr::solver::evaluation_tree::refine_plan_interleaved_with_diagnostics(
            &domain,
            &problem,
            &refinement,
            &mut diagnostics,
        ) {
            Ok(Some(r)) => println!(
                "interleaved: solved goal={:?} actions={} methods={} in {:?}",
                r.goal,
                r.plan.len(),
                r.method_count(),
                start.elapsed()
            ),
            Ok(None) => println!("interleaved: no solution in {:?}", start.elapsed()),
            Err(error) => println!("interleaved: error in {:?}: {error:#}", start.elapsed()),
        }
        println!(
            "  interleaved stats: time_budget={} expansion_budget={} depth={} cycle={} no_applicable={} constraint_reject={} ready_empty={}",
            diagnostics.budget_time_exhausted,
            diagnostics.budget_expansions_exhausted,
            diagnostics.depth_exhausted,
            diagnostics.cycle_pruned,
            diagnostics.no_applicable_method,
            diagnostics.network_constraint_rejected,
            diagnostics.partial_ready_empty,
        );
        let mut dead_ends: Vec<(&String, &u64)> = diagnostics.no_applicable_tasks.iter().collect();
        dead_ends.sort_by(|left, right| right.1.cmp(left.1).then(left.0.cmp(right.0)));
        let summary: Vec<String> = dead_ends
            .iter()
            .take(6)
            .map(|(task, count)| format!("{task}={count}"))
            .collect();
        if !summary.is_empty() {
            println!("  interleaved dead-end tasks: {}", summary.join(" "));
        }
    }
    run_mode(
        "summary+repair",
        &domain,
        &problem,
        MethodSummary::PrimitiveOnly,
        true,
        budget,
        max_depth,
    );
    run_mode(
        "summary+no-repair",
        &domain,
        &problem,
        MethodSummary::PrimitiveOnly,
        false,
        budget,
        max_depth,
    );
    run_mode(
        "no-summary+repair",
        &domain,
        &problem,
        MethodSummary::Off,
        true,
        budget,
        max_depth,
    );
    run_mode(
        "no-summary+no-repair",
        &domain,
        &problem,
        MethodSummary::Off,
        false,
        budget,
        max_depth,
    );
}
