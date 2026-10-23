//! Compare optimization backends on the merge nodes reached by the C+G pipeline.
//!
//! The example first collects `MergeProblem`s with the normal SGPlan merger,
//! then reruns every selected optimizer on the same problems and reports
//! feasibility, exact violation, objective, Adam rounds, and wall-clock time.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use pspr::parsing::discovery::discover_sibling_problems;
use pspr::parsing::parser::{ParserConfig, parse_program, resolve_parser_config};
use pspr::solver::domain::{Domain, process_domain};
use pspr::solver::evaluation_tree::{
    AdamOptions, ConstraintNode, EvaluationOptions, MergeOutcome, MergeProblem, MergeSolver,
    MethodSummary, OptimizerKind, PartialSolution, SgplanMerger, evaluate,
};
use pspr::solver::problem::{Problem, load_problem};

const CASES: &[(&str, &str)] = &[
    ("Satellite-GTOHP", "p01.hddl"),
    ("Satellite-GTOHP", "p02.hddl"),
    ("Satellite-GTOHP", "p03.hddl"),
    ("Depots", "p01.hddl"),
    ("Depots", "p02.hddl"),
    ("Depots", "p03.hddl"),
    ("Transport", "pfile01.hddl"),
    ("Transport", "pfile02.hddl"),
    ("Transport", "pfile03.hddl"),
    ("PO_Transport", "pfile01.hddl"),
    ("PO_Satellite", "1obs-1sat-1mod.hddl"),
    ("Logistics-Learned-ECAI-16", "probLOGISTICS-04-0.hddl"),
    ("Logistics-Learned-ECAI-16", "probLOGISTICS-04-1.hddl"),
    ("Logistics-Learned-ECAI-16", "probLOGISTICS-04-2.hddl"),
    ("AssemblyHierarchical", "genericLinearProblem_depth01.hddl"),
    ("Blocksworld-GTOHP", "p01.hddl"),
    ("Blocksworld-GTOHP", "p03.hddl"),
    ("Blocksworld-GTOHP", "p04.hddl"),
];

#[derive(Debug, Parser)]
#[command(
    name = "compare_merge_optimizers",
    about = "Compare optimizer backends on the SGPlan relaxed merge problem"
)]
struct Args {
    /// Root of the IPC tree.
    #[arg(long, default_value = "../hddl/test_data/ipc")]
    ipc_dir: PathBuf,
    /// Explicit domain paths; without this option, the built-in case list is used.
    #[arg(long)]
    domain: Vec<PathBuf>,
    /// Problems paired with every `--domain`; if omitted, siblings are used.
    #[arg(long)]
    problem: Vec<PathBuf>,
    /// Comma-separated optimizer names; default is all backends.
    #[arg(long, value_delimiter = ',')]
    optimizer: Vec<String>,
    /// Grounding limit used while collecting merge problems.
    #[arg(long, default_value_t = 32_768)]
    max_groundings: usize,
    /// Fast Downward expansion limit inside each leaf.
    #[arg(long, default_value_t = 100_000)]
    leaf_max_expansions: usize,
    /// Repeats per problem and optimizer; timings report the median.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Print one row per problem and optimizer.
    #[arg(long)]
    verbose: bool,
    /// Use the synthetic reverse-order stress suite instead of IPC cases.
    #[arg(long)]
    synthetic: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.repeats == 0 {
        bail!("--repeats must be at least 1");
    }
    let parser = resolve_parser_config(None).map_err(anyhow::Error::msg)?;
    let optimizers = select_optimizers(&args)?;
    let pairs = input_pairs(&args)?;
    let options = EvaluationOptions {
        max_groundings: args.max_groundings,
        leaf_max_expansions: args.leaf_max_expansions,
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };

    let problems = if args.synthetic {
        synthetic_problems()
    } else {
        let mut problems = Vec::new();
        for (domain, problem) in &pairs {
            match collect_problems(&parser, domain, problem, &options) {
                Ok(mut found) => {
                    if args.verbose {
                        eprintln!(
                            "{} + {}: {} merge problem(s)",
                            domain.display(),
                            problem.display(),
                            found.len()
                        );
                    }
                    problems.append(&mut found);
                }
                Err(error) => eprintln!(
                    "skipping {} + {}: {error:#}",
                    domain.display(),
                    problem.display()
                ),
            }
        }
        problems
    };
    if problems.is_empty() {
        bail!("the selected cases produced no merge problems");
    }
    println!("collected {} merge problem(s)", problems.len());

    let mut summaries = Vec::new();
    for optimizer in optimizers {
        let mut runs = Vec::new();
        for (problem_index, problem) in problems.iter().enumerate() {
            for repeat in 0..args.repeats {
                let mut merger = SgplanMerger::default()
                    .with_optimizer(optimizer)
                    .without_initializer();
                let started = Instant::now();
                let outcome = merger.merge(problem);
                let elapsed = started.elapsed();
                let report = merger.report.clone();
                let feasible = report.as_ref().is_some_and(|report| report.feasible);
                let violation = report
                    .as_ref()
                    .map(|report| report.violations.iter().sum::<f64>())
                    .unwrap_or(f64::INFINITY);
                let objective = report
                    .as_ref()
                    .map(|report| report.objective)
                    .unwrap_or(f64::NAN);
                let rounds = report.as_ref().map(|report| report.rounds).unwrap_or(0);
                let plan_len = outcome.as_ref().ok().map(|outcome| outcome.plan.len());
                if args.verbose {
                    println!(
                        "{optimizer} problem={} repeat={} outcome={} feasible={} violation={violation:.6e} objective={objective:.6e} rounds={rounds} plan_len={:?} ms={:.3}",
                        problem_index,
                        repeat,
                        if outcome.is_ok() { "ok" } else { "error" },
                        feasible,
                        plan_len,
                        elapsed.as_secs_f64() * 1e3
                    );
                }
                runs.push(Run {
                    feasible,
                    violation,
                    objective,
                    rounds,
                    plan_len,
                    ms: elapsed.as_secs_f64() * 1e3,
                });
            }
        }
        summaries.push((optimizer, summarize(runs)));
    }

    println!();
    println!(
        "| Optimizer | Runs | Feasible | Mean violation | Mean relaxed objective | Mean plan | Mean rounds | Median ms | Mean ms | Max ms |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (optimizer, summary) in summaries {
        println!(
            "| {} | {} | {}/{} | {:.3e} | {:.3e} | {:.2} | {:.1} | {:.3} | {:.3} | {:.3} |",
            optimizer,
            summary.runs,
            summary.feasible,
            summary.runs,
            summary.mean_violation,
            summary.mean_objective,
            summary.mean_plan_len,
            summary.mean_rounds,
            summary.median_ms,
            summary.mean_ms,
            summary.max_ms
        );
    }
    println!();
    println!("Adam options: {:?}", AdamOptions::default());
    Ok(())
}

fn select_optimizers(args: &Args) -> Result<Vec<OptimizerKind>> {
    if args.optimizer.is_empty() {
        return Ok(OptimizerKind::ALL.to_vec());
    }
    args.optimizer
        .iter()
        .map(|name| {
            OptimizerKind::parse(name).ok_or_else(|| anyhow::anyhow!("unknown optimizer `{name}`"))
        })
        .collect()
}

fn synthetic_problems() -> Vec<MergeProblem> {
    let cases = [
        (4usize, 0usize),
        (8, 0),
        (12, 0),
        (16, 0),
        (24, 0),
        (8, 1),
        (12, 1),
        (16, 1),
        (24, 1),
    ];
    cases
        .into_iter()
        .enumerate()
        .map(|(node, (children, variant))| {
            let mut child_solutions = Vec::new();
            for child in 0..children {
                let length = 1 + (child * 7 + variant * 3) % 4;
                let mut plan = Vec::new();
                let mut plan_tasks = Vec::new();
                for step in 0..length {
                    plan.push(format!("op{child}_{step}"));
                    plan_tasks.push(child * 16 + step);
                }
                child_solutions.push(PartialSolution {
                    label: format!("synthetic {child}"),
                    plan,
                    plan_tasks,
                    final_atoms: BTreeSet::new(),
                    achieved: BTreeSet::new(),
                    solved: true,
                });
            }
            let shared_constraints = (1..children)
                .map(|child| ConstraintNode::Ordering {
                    before: child * 16,
                    after: (child - 1) * 16,
                })
                .collect();
            MergeProblem {
                node,
                shared_constraints,
                children: child_solutions,
                initial_atoms: BTreeSet::new(),
            }
        })
        .collect()
}

fn input_pairs(args: &Args) -> Result<Vec<(PathBuf, PathBuf)>> {
    if args.domain.is_empty() {
        return Ok(CASES
            .iter()
            .map(|(domain, problem)| {
                (
                    args.ipc_dir.join(domain).join("domain.hddl"),
                    args.ipc_dir.join(domain).join(problem),
                )
            })
            .collect());
    }
    let mut pairs = Vec::new();
    for domain in &args.domain {
        let problems = if args.problem.is_empty() {
            discover_sibling_problems(domain)
                .with_context(|| format!("cannot list problems next to {}", domain.display()))?
        } else {
            args.problem.clone()
        };
        for problem in problems {
            pairs.push((domain.clone(), problem));
        }
    }
    Ok(pairs)
}

fn collect_problems(
    parser: &ParserConfig,
    domain_path: &Path,
    problem_path: &Path,
    options: &EvaluationOptions,
) -> Result<Vec<MergeProblem>> {
    let program =
        parse_program(parser, domain_path, Some(problem_path)).map_err(anyhow::Error::msg)?;
    let domain: Domain = process_domain(&program.domain).map_err(anyhow::Error::msg)?;
    let problem_ast = program
        .problem
        .ok_or_else(|| anyhow::anyhow!("the parser returned no problem"))?;
    let problem: Problem = load_problem(&domain, &problem_ast).map_err(anyhow::Error::msg)?;
    let mut recorder = RecordingMerger {
        inner: SgplanMerger::default().without_initializer(),
        problems: Vec::new(),
    };
    if let Err(error) = evaluate(&domain, &problem, &mut recorder, options) {
        eprintln!("evaluation of {} failed: {error:#}", problem_path.display());
    }
    Ok(recorder.problems)
}

#[derive(Debug, Clone)]
struct Run {
    feasible: bool,
    violation: f64,
    objective: f64,
    rounds: usize,
    plan_len: Option<usize>,
    ms: f64,
}

#[derive(Debug, Clone)]
struct Summary {
    runs: usize,
    feasible: usize,
    mean_violation: f64,
    mean_objective: f64,
    mean_plan_len: f64,
    mean_rounds: f64,
    median_ms: f64,
    mean_ms: f64,
    max_ms: f64,
}

fn summarize(runs: Vec<Run>) -> Summary {
    let count = runs.len().max(1) as f64;
    let feasible = runs.iter().filter(|run| run.feasible).count();
    let mut times: Vec<f64> = runs.iter().map(|run| run.ms).collect();
    times.sort_by(f64::total_cmp);
    let median_ms = if times.is_empty() {
        0.0
    } else if times.len().is_multiple_of(2) {
        (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
    } else {
        times[times.len() / 2]
    };
    let finite_objectives: Vec<f64> = runs
        .iter()
        .filter(|run| run.feasible)
        .map(|run| run.objective)
        .filter(|value| value.is_finite())
        .collect();
    let feasible_plan_lengths: Vec<f64> = runs
        .iter()
        .filter(|run| run.feasible)
        .filter_map(|run| run.plan_len.map(|length| length as f64))
        .collect();
    Summary {
        runs: runs.len(),
        feasible,
        mean_violation: runs.iter().map(|run| run.violation).sum::<f64>() / count,
        mean_objective: if finite_objectives.is_empty() {
            f64::NAN
        } else {
            finite_objectives.iter().sum::<f64>() / finite_objectives.len() as f64
        },
        mean_plan_len: if feasible_plan_lengths.is_empty() {
            f64::NAN
        } else {
            feasible_plan_lengths.iter().sum::<f64>() / feasible_plan_lengths.len() as f64
        },
        mean_rounds: runs.iter().map(|run| run.rounds as f64).sum::<f64>() / count,
        median_ms,
        mean_ms: runs.iter().map(|run| run.ms).sum::<f64>() / count,
        max_ms: times.last().copied().unwrap_or(0.0),
    }
}

#[derive(Debug)]
struct RecordingMerger {
    inner: SgplanMerger,
    problems: Vec<MergeProblem>,
}

impl MergeSolver for RecordingMerger {
    fn merge(&mut self, problem: &MergeProblem) -> Result<MergeOutcome> {
        self.problems.push(problem.clone());
        self.inner.merge(problem)
    }
}
