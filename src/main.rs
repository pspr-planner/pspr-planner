use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use pspr::parsing::discovery::{discover_sibling_problems, expand_problem_arg};
use pspr::parsing::json::{ProblemAst, Program};
use pspr::parsing::parser::{
    ParserConfig, is_json_file, parse_program, read_program_json, resolve_parser_config,
};
use pspr::solver::domain::{Domain, process_domain};
use pspr::solver::problem::{Problem, load_problem};
use pspr::solver::solve::solve;

#[derive(Debug, Parser)]
#[command(
    name = "pspr",
    version,
    about = "Load an HDDL domain through the koala-planner HDDL-Parser JSON export, \
             load the problems that belong to it, and dispatch each to solve",
    long_about = None
)]
struct Options {
    /// Domain file (.hddl, or a .json export of a whole program)
    #[arg(value_name = "DOMAIN")]
    domain_path: PathBuf,

    /// Problem .hddl files or directories; defaults to the domain's sibling problem files
    #[arg(value_name = "PROBLEM")]
    problem_paths: Vec<PathBuf>,

    /// Path to the hddl_analyzer executable (default: $HDDL_PARSER or hddl_analyzer on PATH)
    #[arg(long, value_name = "PATH")]
    parser: Option<PathBuf>,
}

fn main() -> ExitCode {
    let options = Options::parse();
    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) if message.is_empty() => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("pspr: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(options: &Options) -> Result<(), String> {
    if is_json_file(&options.domain_path) {
        run_json_input(options)
    } else {
        run_hddl_input(options)
    }
}

/// A JSON export already contains the domain and at most one problem.
fn run_json_input(options: &Options) -> Result<(), String> {
    let program = read_program_json(&options.domain_path)?;
    if !options.problem_paths.is_empty() {
        return Err("a .json input already contains its problem; \
                    use an .hddl domain to pair it with problem files"
            .to_string());
    }

    let domain = process_domain(&program.domain)?;
    print_domain(&domain);

    match &program.problem {
        None => println!("No problem is embedded in the JSON input."),
        Some(problem_ast) => handle_problem(&domain, problem_ast)?,
    }
    Ok(())
}

/// An HDDL domain is parsed once; each problem is parsed against the same
/// domain file, then loaded and solved.
fn run_hddl_input(options: &Options) -> Result<(), String> {
    let config = resolve_parser_config(options.parser.as_deref())?;
    let program = parse_program(&config, &options.domain_path, None)?;
    let domain = process_domain(&program.domain)?;
    print_domain(&domain);

    if program.problem.is_some() {
        eprintln!("pspr: warning: ignoring the problem embedded in the domain input");
    }

    let problem_paths = resolve_problem_paths(options)?;
    if problem_paths.is_empty() {
        println!("No problem files found; nothing to solve.");
        return Ok(());
    }

    println!("Loading {} problem file(s).", problem_paths.len());
    let mut failed = false;
    for path in &problem_paths {
        if let Err(message) = solve_problem_file(&config, &domain, &options.domain_path, path) {
            eprintln!("pspr: {message}");
            failed = true;
        }
    }

    if failed {
        // Errors have been reported individually.
        Err(String::new())
    } else {
        Ok(())
    }
}

fn resolve_problem_paths(options: &Options) -> Result<Vec<PathBuf>, String> {
    if options.problem_paths.is_empty() {
        return discover_sibling_problems(&options.domain_path)
            .map_err(|err| format!("cannot list problem files: {err}"));
    }

    let mut paths = Vec::new();
    for argument in &options.problem_paths {
        let expanded = expand_problem_arg(argument)
            .map_err(|err| format!("cannot read {}: {err}", argument.display()))?;
        paths.extend(expanded);
    }
    Ok(paths)
}

fn solve_problem_file(
    config: &ParserConfig,
    domain: &Domain,
    domain_path: &std::path::Path,
    problem_path: &std::path::Path,
) -> Result<(), String> {
    let program: Program = parse_program(config, domain_path, Some(problem_path))?;
    let problem_ast: ProblemAst = program.problem.ok_or_else(|| {
        format!(
            "the parser produced no problem for {}",
            problem_path.display()
        )
    })?;
    handle_problem(domain, &problem_ast)
}

/// Stub boundary: problem AST -> [`Problem`] -> [`solve`].
fn handle_problem(domain: &Domain, problem_ast: &ProblemAst) -> Result<(), String> {
    let problem: Problem = load_problem(domain, problem_ast)?;
    println!(
        "  problem {} ({} objects)",
        problem.name,
        problem.objects.len()
    );

    let result = solve(domain, &problem)?;
    println!(
        "    solve {}: stub, plan length {}",
        result.problem,
        result.plan.len()
    );
    Ok(())
}

fn print_domain(domain: &Domain) {
    println!("Domain {}", domain.name);
    println!(
        "  {} types, {} constants, {} predicates",
        domain.types.len(),
        domain.constants.len(),
        domain.predicates.len()
    );
    println!(
        "  {} actions, {} compound tasks, {} methods",
        domain.actions.len(),
        domain.compound_tasks.len(),
        domain.methods.len()
    );
}
