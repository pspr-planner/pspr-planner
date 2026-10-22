//! Integration tests on the IPC HTN benchmark set.
//!
//! The tests are skipped when the benchmark checkout or the external HDDL
//! parser is unavailable, so `cargo test` still works on a bare checkout.
//!
//! The parser is resolved from `$HDDL_PARSER`, then from `hddl_analyzer` on
//! `PATH`, then from `$HOME/.cargo/bin/hddl_analyzer`, and finally from
//! `../hddl/target/release/hddl_analyzer` relative to the crate root. The
//! benchmark directory defaults to `../hddl/test_data/ipc` and can be
//! overridden with `$PSPR_IPC_DIR`.
//!
//! Every case asserts the problem name decoded from the HDDL file and the
//! length of the deterministic plan the evaluation-tree pipeline returns.

use std::path::PathBuf;
use std::sync::OnceLock;

use pspr::parsing::parser::{ParserConfig, parse_program, resolve_parser_config};
use pspr::solver::domain::{Domain, process_domain};
use pspr::solver::evaluation_tree::{
    EvaluationOptions, GoalStatus, ValidationOptions, build_evaluation_tree, validate_plan,
};
use pspr::solver::problem::{Problem, load_problem};
use pspr::solver::solve::solve;

struct Ipc {
    root: PathBuf,
    parser: ParserConfig,
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn discover() -> Option<Ipc> {
    let root = std::env::var_os("PSPR_IPC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("../hddl/test_data/ipc"));
    if !root.is_dir() {
        eprintln!(
            "skipping IPC tests: {} is not a directory (set PSPR_IPC_DIR)",
            root.display()
        );
        return None;
    }
    let parser = parser_config()?;
    Some(Ipc { root, parser })
}

fn parser_config() -> Option<ParserConfig> {
    if let Ok(config) = resolve_parser_config(None) {
        return Some(config);
    }
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".cargo/bin/hddl_analyzer"));
    }
    candidates.push(manifest_dir().join("../hddl/target/release/hddl_analyzer"));
    for candidate in candidates {
        if let Ok(config) = resolve_parser_config(Some(&candidate)) {
            return Some(config);
        }
    }
    eprintln!(
        "skipping IPC tests: hddl_analyzer was not found (set HDDL_PARSER or build it \
         at ../hddl/target/release/hddl_analyzer)"
    );
    None
}

fn ipc() -> Option<&'static Ipc> {
    static IPC: OnceLock<Option<Ipc>> = OnceLock::new();
    IPC.get_or_init(discover).as_ref()
}

fn load_ipc(ipc: &Ipc, domain_name: &str, problem_name: &str) -> (Domain, Problem) {
    let domain_path = ipc.root.join(domain_name).join("domain.hddl");
    let problem_path = ipc.root.join(domain_name).join(problem_name);
    assert!(domain_path.is_file(), "missing {}", domain_path.display());
    assert!(problem_path.is_file(), "missing {}", problem_path.display());

    let program = parse_program(&ipc.parser, &domain_path, Some(&problem_path))
        .unwrap_or_else(|error| panic!("{domain_name}/{problem_name}: {error}"));
    let domain = process_domain(&program.domain)
        .unwrap_or_else(|error| panic!("{domain_name}/{problem_name}: {error}"));
    let ast = program
        .problem
        .as_ref()
        .unwrap_or_else(|| panic!("{domain_name}/{problem_name}: the parser returned no problem"));
    let problem = load_problem(&domain, ast)
        .unwrap_or_else(|error| panic!("{domain_name}/{problem_name}: {error}"));
    (domain, problem)
}

/// Solve one IPC problem, check the decoded problem name and plan length, and
/// re-execute the plan against the full HDDL semantics.
fn check(
    domain: &str,
    problem: &str,
    expected_name: &str,
    expected_length: usize,
    require_goal: bool,
) {
    let Some(ipc) = ipc() else {
        return;
    };
    let (domain_model, problem_model) = load_ipc(ipc, domain, problem);
    let result = solve(&domain_model, &problem_model)
        .unwrap_or_else(|error| panic!("{domain}/{problem}: {error}"));
    assert_eq!(
        result.problem, expected_name,
        "{domain}/{problem}: problem name"
    );
    assert_eq!(
        result.plan.len(),
        expected_length,
        "{domain}/{problem}: plan {:?}",
        result.plan
    );
    assert!(
        result.plan.iter().all(|step| !step.is_empty()),
        "{domain}/{problem}: empty operator name"
    );
    assert!(result.solved, "{domain}/{problem}: the tree is not solved");
    let report = validate_plan(
        &domain_model,
        &problem_model,
        &result.plan,
        &ValidationOptions {
            require_goal,
            ..ValidationOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("{domain}/{problem}: invalid plan: {error}"));
    assert_eq!(report.steps, result.plan.len(), "{domain}/{problem}");
    assert!(report.tasks > 0, "{domain}/{problem}: no grounded tasks");
    if require_goal {
        assert!(
            matches!(report.goal, GoalStatus::Absent | GoalStatus::Satisfied),
            "{domain}/{problem}: goal status {:?}",
            report.goal
        );
    }
}

#[test]
fn ipc_blocksworld_gtoph() {
    for (problem, name, length) in [
        ("p01.hddl", "BW-rand-5", 3),
        ("p02.hddl", "BW-rand-7", 6),
        ("p03.hddl", "BW-rand-9", 5),
        ("p04.hddl", "BW-rand-11", 14),
        ("p05.hddl", "BW-rand-13", 12),
    ] {
        check("Blocksworld-GTOHP", problem, name, length, false);
    }
}

#[test]
fn ipc_depots() {
    for (problem, name, length) in [
        ("p01.hddl", "depotprob1818", 2),
        ("p02.hddl", "depotprob7512", 4),
        ("p03.hddl", "depotprob1935", 6),
    ] {
        check("Depots", problem, name, length, false);
    }
}

#[test]
fn ipc_satellite_gtoph() {
    for (problem, length) in [("p01.hddl", 3), ("p02.hddl", 5)] {
        check("Satellite-GTOHP", problem, "strips-sat-x-1", length, false);
    }
}

/// The leaf search currently fails on two tasks of Satellite-GTOHP p03, so the
/// evaluation tree returns a partial plan. The validator rejects it because
/// grounded tasks are missing.
#[test]
fn partial_plans_are_rejected_by_the_validator() {
    let Some(ipc) = ipc() else {
        return;
    };
    let (domain, problem) = load_ipc(ipc, "Satellite-GTOHP", "p03.hddl");
    let result = solve(&domain, &problem).expect("solve");
    assert!(!result.solved, "the tree is currently unsolved");
    let error = validate_plan(
        &domain,
        &problem,
        &result.plan,
        &ValidationOptions::default(),
    )
    .expect_err("the partial plan must not validate");
    assert!(
        format!("{error}").contains("does not execute"),
        "unexpected validation error: {error}"
    );
}

#[test]
fn ipc_transport() {
    for (problem, name, length) in [
        ("pfile01.hddl", "pfile01", 2),
        ("pfile02.hddl", "pfile02", 3),
        ("pfile03.hddl", "pfile03", 3),
    ] {
        check("Transport", problem, name, length, true);
    }
}

#[test]
fn ipc_partial_order_transport() {
    check("PO_Transport", "pfile01.hddl", "p", 2, true);
}

#[test]
fn ipc_partial_order_satellite() {
    check(
        "PO_Satellite",
        "1obs-1sat-1mod.hddl",
        "p1obs_1sat_1mod",
        1,
        true,
    );
}

#[test]
fn ipc_freecell() {
    check(
        "Freecell-Learned-ECAI-16",
        "probfreecell-02-1.hddl",
        "p",
        4,
        true,
    );
}

#[test]
fn ipc_logistics() {
    for (problem, length) in [
        ("probLOGISTICS-04-0.hddl", 4),
        ("probLOGISTICS-04-1.hddl", 4),
        ("probLOGISTICS-04-2.hddl", 4),
    ] {
        check("Logistics-Learned-ECAI-16", problem, "p", length, true);
    }
}

#[test]
fn ipc_assembly_hierarchical() {
    check(
        "AssemblyHierarchical",
        "genericLinearProblem_depth01.hddl",
        "generischesLinearesVerkabelungsproblemTiefe1",
        1,
        false,
    );
}

#[test]
fn ipc_barman_bdi() {
    for (problem, name, length) in [
        ("pfile01.hddl", "p-1-2-2", 1),
        ("pfile02.hddl", "p-2-3-3", 2),
    ] {
        check("Barman-BDI", problem, name, length, true);
    }
}

#[test]
fn ipc_robot() {
    check("Robot", "pfile_01_001.hddl", "pfile_01_001", 1, true);
}

#[test]
fn ipc_minecraft_player() {
    check(
        "Minecraft-Player",
        "p-003-003-003-003.hddl",
        "house",
        1,
        true,
    );
}

#[test]
fn ipc_monroe() {
    check(
        "Monroe_PO_1",
        "pfile01-p-0014-fix-power-line-4.hddl",
        "someProblem",
        1,
        false,
    );
}

#[test]
fn ipc_pcp() {
    check("PCP_1", "p-pcp01.hddl", "someProblem", 2, true);
}

/// Blocksworld-GTOHP, Depots, Satellite-GTOHP, AssemblyHierarchical and
/// Monroe_PO_1 have an initial task network of compound tasks with methods, so
/// they cannot achieve their classical goal: the evaluation tree models
/// compound tasks as opaque operators, so no subtask effects are applied. The
/// validator reports the goal as violated instead of rejecting the execution.
#[test]
fn opaque_compound_tasks_report_a_violated_goal() {
    let Some(ipc) = ipc() else {
        return;
    };
    let (domain, problem) = load_ipc(ipc, "Blocksworld-GTOHP", "p01.hddl");
    let result = solve(&domain, &problem).expect("solve");
    let report = validate_plan(
        &domain,
        &problem,
        &result.plan,
        &ValidationOptions::default(),
    )
    .expect("the plan executes the task network");
    assert_eq!(report.goal, GoalStatus::Violated);

    let error = validate_plan(
        &domain,
        &problem,
        &result.plan,
        &ValidationOptions {
            require_goal: true,
            ..ValidationOptions::default()
        },
    )
    .expect_err("the goal is not achieved");
    assert!(format!("{error}").contains("goal"));
}

/// The selected IPC domains only use formula constructors the leaf encoding can
/// translate, so the evaluation tree must not report any unmodelled formula.
#[test]
fn ipc_formulas_are_fully_supported() {
    let Some(ipc) = ipc() else {
        return;
    };
    for (domain_name, problem_name) in [
        ("Blocksworld-GTOHP", "p01.hddl"),
        ("Depots", "p01.hddl"),
        ("Satellite-GTOHP", "p01.hddl"),
        ("Transport", "pfile01.hddl"),
        ("PO_Transport", "pfile01.hddl"),
        ("PO_Satellite", "1obs-1sat-1mod.hddl"),
        ("Freecell-Learned-ECAI-16", "probfreecell-02-1.hddl"),
        ("Logistics-Learned-ECAI-16", "probLOGISTICS-04-0.hddl"),
        ("AssemblyHierarchical", "genericLinearProblem_depth01.hddl"),
        ("Barman-BDI", "pfile01.hddl"),
        ("Robot", "pfile_01_001.hddl"),
        ("Minecraft-Player", "p-003-003-003-003.hddl"),
        ("Monroe_PO_1", "pfile01-p-0014-fix-power-line-4.hddl"),
        ("PCP_1", "p-pcp01.hddl"),
    ] {
        let (domain, problem) = load_ipc(ipc, domain_name, problem_name);
        let tree = build_evaluation_tree(&domain, &problem, &EvaluationOptions::default())
            .unwrap_or_else(|error| panic!("{domain_name}/{problem_name}: {error}"));
        assert_eq!(tree.unmodeled, 0, "{domain_name}/{problem_name}");
        assert!(
            !tree.tasks.is_empty(),
            "{domain_name}/{problem_name}: no grounded tasks"
        );
    }
}
