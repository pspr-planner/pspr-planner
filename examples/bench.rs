//! Print one machine-readable result line for a planning problem.
//!
//! Usage:
//!     HDDL_PARSER=/path/to/hddl_analyzer \
//!         cargo run --release --example bench -- domain.hddl problem.hddl
//!
//! The output is `key=value` pairs: `solved`, `goal`, `steps`, `tasks` and
//! optionally `error`. It is used by `scripts/compare_panda.sh`.

use std::env;
use std::path::PathBuf;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{GoalStatus, ValidationOptions, validate_plan};
use pspr::solver::problem::load_problem;
use pspr::solver::solve::solve;

fn goal_name(status: GoalStatus) -> &'static str {
    match status {
        GoalStatus::Absent => "absent",
        GoalStatus::Satisfied => "satisfied",
        GoalStatus::Violated => "violated",
    }
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 3 {
        eprintln!("usage: bench <domain.hddl> <problem.hddl>");
        std::process::exit(2);
    }
    let parser = env::var_os("HDDL_PARSER").unwrap_or_else(|| "hddl_analyzer".into());
    let config = ParserConfig {
        executable: PathBuf::from(parser),
    };

    let program = match parse_program(
        &config,
        &PathBuf::from(&arguments[1]),
        Some(&PathBuf::from(&arguments[2])),
    ) {
        Ok(program) => program,
        Err(error) => {
            println!("solved=false goal=invalid steps=0 tasks=0 error={error:?}");
            std::process::exit(1);
        }
    };
    let domain = match process_domain(&program.domain) {
        Ok(domain) => domain,
        Err(error) => {
            println!("solved=false goal=invalid steps=0 tasks=0 error={error:?}");
            std::process::exit(1);
        }
    };
    let Some(problem_ast) = program.problem.as_ref() else {
        println!("solved=false goal=invalid steps=0 tasks=0 error=\"no problem\"");
        std::process::exit(1);
    };
    let problem = match load_problem(&domain, problem_ast) {
        Ok(problem) => problem,
        Err(error) => {
            println!("solved=false goal=invalid steps=0 tasks=0 error={error:?}");
            std::process::exit(1);
        }
    };
    let result = match solve(&domain, &problem) {
        Ok(result) => result,
        Err(error) => {
            println!(
                "solved=false goal=invalid steps=0 tasks=0 error={:?}",
                format!("{error:#}")
            );
            std::process::exit(1);
        }
    };

    match validate_plan(
        &domain,
        &problem,
        &result.plan,
        &ValidationOptions::default(),
    ) {
        Ok(report) => println!(
            "solved={} goal={} steps={} tasks={}",
            result.solved,
            goal_name(report.goal),
            result.plan.len(),
            report.tasks
        ),
        Err(error) => println!(
            "solved={} goal=invalid steps={} tasks=0 error={:?}",
            result.solved,
            result.plan.len(),
            format!("{error:#}")
        ),
    }
}
