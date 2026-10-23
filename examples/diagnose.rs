//! Temporary diagnostic helper: evaluates a hierarchical instance and prints
//! leaf, merge and refinement details.
use std::path::PathBuf;

use pspr::parsing::parser::{ParserConfig, parse_program};
use pspr::solver::domain::process_domain;
use pspr::solver::evaluation_tree::{
    DecompositionSession, DecompositionTraceSink, EvaluationOptions, MethodSummary, MethodTrace,
    RefinementOptions, SearchTrace, SgplanMerger, build_evaluation_tree, evaluate,
    refine_plan_with_session,
};
use pspr::solver::problem::load_problem;

#[derive(Debug, Default)]
struct Printer;

impl DecompositionTraceSink for Printer {
    fn method_trace(&mut self, trace: MethodTrace) {
        let applicable = trace.candidates.iter().filter(|c| c.applicable).count();
        let successful = trace.successful.iter().filter(|s| **s).count();
        println!(
            "[method] task={} depth={}/{} candidates={} applicable={} successful={} state={:?}",
            trace.context.task,
            trace.context.depth,
            trace.context.max_depth,
            trace.candidates.len(),
            applicable,
            successful,
            trace.context.state,
        );
        for (candidate, ok) in trace.candidates.iter().zip(&trace.successful) {
            println!(
                "    method={} bindings={:?} applicable={} subtasks={} ok={ok}",
                candidate.method, candidate.bindings, candidate.applicable, candidate.subtasks
            );
        }
    }

    fn search_trace(&mut self, _trace: SearchTrace) {}
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [domain_path, problem_path] = args.as_slice() else {
        eprintln!("usage: diagnose <domain> <problem>");
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
    .unwrap();
    let domain = process_domain(&program.domain).unwrap();
    let problem = load_problem(&domain, program.problem.as_ref().unwrap()).unwrap();

    let options = EvaluationOptions {
        method_summary: MethodSummary::PrimitiveOnly,
        implicit_orderings: true,
        ..EvaluationOptions::default()
    };
    let mut merger = SgplanMerger::default();
    let started = std::time::Instant::now();
    let tree = build_evaluation_tree(&domain, &problem, &options).unwrap();
    println!(
        "BUILD {:?}: tasks={} constraints={} clusters={}",
        started.elapsed(),
        tree.tasks.len(),
        tree.constraints.len(),
        tree.clusters.len()
    );
    let started = std::time::Instant::now();
    let evaluation = evaluate(&domain, &problem, &mut merger, &options).unwrap();
    println!("EVAL {:?}", started.elapsed());
    println!(
        "TREE solved={} plan={:?}",
        evaluation.solved, evaluation.plan
    );
    println!("  tasks={:?}", evaluation.tree.tasks);
    let unbound: Vec<_> = evaluation
        .tree
        .constraints
        .iter()
        .filter(|constraint| format!("{constraint}").contains('?'))
        .collect();
    println!("  unbound constraints ({}): {unbound:?}", unbound.len());
    for leaf in &evaluation.leaves {
        println!(
            "  leaf {} solved={} plan={:?} final={:?}",
            leaf.label, leaf.solved, leaf.plan, leaf.final_atoms
        );
    }
    for merge in &evaluation.merges {
        println!(
            "  merge node={} solved={} plan={:?}",
            merge.node, merge.solved, merge.plan
        );
    }

    let mut printer = Printer;
    let mut session = DecompositionSession {
        method_ranker: None,
        search_policy: None,
        plan_permuter: None,
        trace: Some(&mut printer),
    };
    let refinement = refine_plan_with_session(
        &domain,
        &problem,
        &evaluation.plan,
        &RefinementOptions::default(),
        &mut session,
    );
    match refinement {
        Ok(refinement) => println!(
            "REFINE ok steps={} goal={:?}",
            refinement.plan.len(),
            refinement.goal
        ),
        Err(error) => println!("REFINE error: {error:#}"),
    }
}
