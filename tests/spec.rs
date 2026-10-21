//! Port of the Haskell `test/Spec.hs` suite.

use std::fs;
use std::path::{Path, PathBuf};

use pspr::parsing::discovery::discover_sibling_problems;
use pspr::parsing::json::{
    Constraint, DomainAst, Formula, NumberType, ProblemAst, Program, TaskOrdering, decode_program,
};
use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::problem::{Problem, load_problem};
use pspr::solver::solve::{SolveResult, solve};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_path(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn decode_fixture(name: &str) -> Program {
    let path = fixture_path(name);
    let bytes =
        fs::read(&path).unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    decode_program(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn program_fixture() -> Program {
    decode_fixture("program.json")
}

#[test]
fn program_fixture_decodes() {
    let program = program_fixture();
    let domain: &DomainAst = &program.domain;

    assert_eq!(domain.name, "transport");
    assert_eq!(domain.actions.len(), 2);
    assert_eq!(domain.methods.len(), 2);
    assert_eq!(domain.types.as_ref().map(Vec::len), Some(3));

    let first_method = &domain.methods[0];
    assert_eq!(
        first_method.tn.orderings,
        TaskOrdering::Partial(vec![("t1".to_string(), "t2".to_string())])
    );
    assert_eq!(
        first_method.tn.constraints,
        Some(vec![
            Constraint::Equal("?p".to_string(), "depot".to_string()),
            Constraint::NotEqual("?p".to_string(), "other".to_string()),
        ])
    );
    assert!(
        matches!(first_method.precondition, Some(Formula::And(_))),
        "method precondition is a conjunction"
    );

    let second_method = &domain.methods[1];
    assert_eq!(second_method.tn.orderings, TaskOrdering::Total);

    let problem: &ProblemAst = program
        .problem
        .as_ref()
        .expect("expected an embedded problem");
    assert_eq!(problem.name, "p1");
    assert_eq!(problem.domain_name, "transport");

    let init_tn = problem
        .init_tn
        .as_ref()
        .expect("expected an initial task network");
    assert_eq!(init_tn.tn.orderings, TaskOrdering::Total);
}

#[test]
fn minimal_fixture_decodes() {
    let program = decode_fixture("minimal.json");
    assert_eq!(program.domain.types, None);
    assert_eq!(program.domain.constants, None);
    assert_eq!(program.problem, None);
}

#[test]
fn formula_fixture_decodes() {
    let program = decode_fixture("formulas.json");
    let actions = &program.domain.actions;
    assert_eq!(actions.len(), 1, "expected one action");

    let action = &actions[0];
    match &action.preconditions {
        Some(Formula::And(terms)) => {
            assert_eq!(terms.len(), 9, "formula conjunct count");
            match terms.as_slice() {
                [
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    Formula::Probabilistic(NumberType::Real(real), _),
                    Formula::Probabilistic(NumberType::Integer(integer), last),
                ] => {
                    assert_eq!(*real, 0.25);
                    assert_eq!(*integer, 2);
                    assert_eq!(**last, Formula::Empty);
                }
                other => panic!("unexpected formula shapes: {other:?}"),
            }
        }
        other => panic!("expected a conjunction, got {other:?}"),
    }

    assert_eq!(action.effects, Some(Formula::Empty));
}

#[test]
fn domain_problem_solve() {
    let program = program_fixture();
    let domain = process_domain(&program.domain).expect("processDomain");
    assert_eq!(domain.name, "transport");

    let problem_ast = program
        .problem
        .as_ref()
        .expect("expected an embedded problem");
    let problem: Problem = load_problem(&domain, problem_ast).expect("loadProblem");
    assert_eq!(problem.name, "p1");

    let result: SolveResult = solve(&domain, &problem).expect("solve");
    assert!(result.plan.is_empty(), "stub plan");
}

#[test]
fn parser_dispatch() {
    let directory = tempfile::tempdir().expect("temp dir");
    let parser_path = directory.path().join("hddl_analyzer");
    let fixture = fixture_path("program.json");

    let script = format!("#!/bin/sh\nprintf '[Ok]\\n'\ncat '{}'\n", fixture.display());
    fs::write(&parser_path, script).expect("write fake parser");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&parser_path, fs::Permissions::from_mode(0o755))
            .expect("chmod fake parser");
    }

    let config = ParserConfig {
        executable: parser_path,
    };
    let program = parse_program(
        &config,
        Path::new("domain.hddl"),
        Some(Path::new("problem.hddl")),
    )
    .unwrap_or_else(|err| panic!("fake parser dispatch failed: {err}"));

    assert_eq!(program.domain.name, "transport");
}

#[test]
fn discovers_sibling_problems() {
    let directory = tempfile::tempdir().expect("temp dir");
    let domain = directory.path().join("domain.hddl");
    fs::write(&domain, "").expect("write domain");
    fs::write(directory.path().join("p01.hddl"), "").expect("write problem");
    fs::write(directory.path().join("notes.txt"), "").expect("write notes");

    let problems = discover_sibling_problems(&domain).expect("discover");
    assert_eq!(problems, vec![directory.path().join("p01.hddl")]);
}
