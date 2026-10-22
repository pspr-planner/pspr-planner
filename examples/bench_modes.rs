//! Machine-readable benchmark of PSPR's compound-task techniques.
//!
//! Usage:
//!     bench_modes [mode] <domain.hddl> <problem.hddl>
//!
//! Modes are `abstract` (default), `summary`, and `hierarchical`. The
//! hierarchical mode refines with primitive method summaries enabled (C+G).
//!
//! The output is a single line of `key=value` pairs:
//! `mode`, `solved`, `goal`, `actions`, `methods`, `abstract_steps`, `tasks`
//! and optionally `error`.
//!
//! `HDDL_PARSER` selects the parser executable (default `hddl_analyzer`);
//! `PSPR_MAX_GROUNDINGS` overrides the method and formula grounding budget
//! (default 256). The comparison script sets it to 4096.

use std::env;
use std::path::PathBuf;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    EvaluationOptions, GoalStatus, MethodSummary, RefinementOptions, ValidationOptions,
    validate_plan, validate_refinement,
};
use pspr::solver::problem::load_problem;
use pspr::solver::solve::{solve_hierarchical_with_options, solve_with_options};

fn goal_name(status: GoalStatus) -> &'static str {
    match status {
        GoalStatus::Absent => "absent",
        GoalStatus::Satisfied => "satisfied",
        GoalStatus::Violated => "violated",
    }
}

fn report(
    mode: &str,
    solved: bool,
    goal: &str,
    actions: usize,
    methods: usize,
    abstract_steps: usize,
    tasks: usize,
) {
    println!(
        "mode={mode} solved={solved} goal={goal} actions={actions} methods={methods} \
         abstract_steps={abstract_steps} tasks={tasks}"
    );
}

fn fail(mode: &str, error: &str) {
    println!(
        "mode={mode} solved=false goal=invalid actions=0 methods=0 abstract_steps=0 tasks=0 \
         error={error:?}"
    );
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let (mode, domain_path, problem_path) = match arguments.as_slice() {
        [domain, problem] => ("abstract", domain, problem),
        [mode, domain, problem] => (mode.as_str(), domain, problem),
        _ => {
            eprintln!("usage: bench_modes [mode] <domain.hddl> <problem.hddl>");
            std::process::exit(2);
        }
    };

    let parser = env::var_os("HDDL_PARSER").unwrap_or_else(|| "hddl_analyzer".into());
    let config = ParserConfig {
        executable: PathBuf::from(parser),
    };
    let max_groundings = env::var("PSPR_MAX_GROUNDINGS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);

    let program = match parse_program(
        &config,
        &PathBuf::from(domain_path),
        Some(&PathBuf::from(problem_path)),
    ) {
        Ok(program) => program,
        Err(error) => {
            fail(mode, &format!("{error:?}"));
            std::process::exit(1);
        }
    };
    let domain = match process_domain(&program.domain) {
        Ok(domain) => domain,
        Err(error) => {
            fail(mode, &format!("{error:?}"));
            std::process::exit(1);
        }
    };
    let Some(problem_ast) = program.problem.as_ref() else {
        fail(mode, "no problem");
        std::process::exit(1);
    };
    let problem = match load_problem(&domain, problem_ast) {
        Ok(problem) => problem,
        Err(error) => {
            fail(mode, &format!("{error:?}"));
            std::process::exit(1);
        }
    };

    let evaluation_options = EvaluationOptions {
        method_summary: if mode == "summary" || mode == "hierarchical" {
            MethodSummary::PrimitiveOnly
        } else {
            MethodSummary::Off
        },
        max_groundings,
        ..EvaluationOptions::default()
    };
    let refinement_options = RefinementOptions {
        max_groundings,
        ..RefinementOptions::default()
    };

    match mode {
        "abstract" | "summary" => {
            let result = match solve_with_options(&domain, &problem, &evaluation_options) {
                Ok(result) => result,
                Err(error) => {
                    fail(mode, &error);
                    std::process::exit(1);
                }
            };
            let report = match validate_plan(
                &domain,
                &problem,
                &result.plan,
                &ValidationOptions::default(),
            ) {
                Ok(report) => report,
                Err(error) => {
                    fail(mode, &format!("{error:#}"));
                    std::process::exit(1);
                }
            };
            report_line(
                mode,
                result.solved,
                report.goal,
                result.plan.len(),
                0,
                result.plan.len(),
                report.tasks,
            );
        }
        "hierarchical" => {
            let refinement = solve_hierarchical_with_options(
                &domain,
                &problem,
                &refinement_options,
                &evaluation_options,
            );
            let refinement = match refinement {
                Ok(refinement) => refinement,
                Err(error) => {
                    fail(mode, &error);
                    std::process::exit(1);
                }
            };
            let validation =
                match validate_refinement(&domain, &problem, &refinement, &refinement_options) {
                    Ok(validation) => validation,
                    Err(error) => {
                        fail(mode, &format!("{error:#}"));
                        std::process::exit(1);
                    }
                };
            report_line(
                mode,
                true,
                refinement.goal,
                refinement.plan.len(),
                refinement.method_count(),
                refinement.abstract_plan.len(),
                validation.tasks,
            );
        }
        other => {
            fail(other, "unknown mode");
            std::process::exit(2);
        }
    }
}

fn report_line(
    mode: &str,
    solved: bool,
    goal: GoalStatus,
    actions: usize,
    methods: usize,
    abstract_steps: usize,
    tasks: usize,
) {
    report(
        mode,
        solved,
        goal_name(goal),
        actions,
        methods,
        abstract_steps,
        tasks,
    );
}
