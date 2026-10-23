//! Train the unified neural decomposition model.
//!
//! The binary collects method-ranking and frontier-search traces from the
//! symbolic C+G pipeline, then trains the shared graph encoder with one of
//! three objectives:
//!
//! * `method`: listwise ranking of grounded method candidates;
//! * `policy`: listwise ordering of ready tasks plus a value regression;
//! * `both`: a single optimizer step stream containing both losses.
//!
//! The symbolic refiner remains authoritative. The trained model only
//! reorders applicable methods and ready tasks.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
#[cfg(feature = "neural-gpu")]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, Int, Tensor, TensorData, activation};
use clap::{Parser, ValueEnum};
use pspr::parsing::discovery::discover_sibling_problems;
use pspr::parsing::parser::{ParserConfig, parse_program, resolve_parser_config};
use pspr::solver::domain::{Domain, process_domain};
use pspr::solver::evaluation_tree::{
    DECOMPOSITION_FEATURE_DIM, DECOMPOSITION_HIDDEN, DecompositionGraph, DecompositionModel,
    DecompositionSession, DecompositionTraceSink, EvaluationOptions, GraphTensors, GuidanceKind,
    MethodSummary, MethodTrace, RefinementOptions, SearchTrace, SgplanMerger, evaluate,
    refine_plan_with_session,
};
use pspr::solver::problem::{Problem, load_problem};
#[cfg(any(feature = "neural", feature = "neural-gpu"))]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TrainTask {
    /// Train only the method-ranking head.
    Method,
    /// Train only the frontier policy and value heads.
    Policy,
    /// Train both heads in one training stream.
    Both,
}

#[derive(Debug, Parser)]
#[command(
    name = "train_decomposition_model",
    about = "Train the unified method ranker and frontier policy/value model"
)]
struct Args {
    /// Which decomposition head(s) to train.
    #[arg(long, value_enum, default_value_t = TrainTask::Both)]
    task: TrainTask,
    /// Where to write the trained Burn module.
    #[arg(long, default_value = "target/decomposition-model.burnpack")]
    output: PathBuf,
    /// Domains to train on. Without this option, `--ipc-dir` is scanned.
    #[arg(long)]
    domain: Vec<PathBuf>,
    /// Problems paired with every `--domain`; if omitted, siblings are used.
    #[arg(long)]
    problem: Vec<PathBuf>,
    /// Root of an IPC-style directory tree when no `--domain` is given.
    #[arg(long, default_value = "../hddl/test_data/ipc")]
    ipc_dir: PathBuf,
    /// Adam epochs over the collected decision traces.
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    /// Adam learning rate.
    #[arg(long, default_value_t = 1.0e-3)]
    learning_rate: f64,
    /// Maximum number of training decisions; `--task both` splits this
    /// budget between the two heads.
    #[arg(long, default_value_t = 4_096)]
    max_samples: usize,
    /// Number of Rayon worker threads for trace collection and example
    /// construction. Zero uses Rayon's default or `RAYON_NUM_THREADS`.
    #[cfg(any(feature = "neural", feature = "neural-gpu"))]
    #[arg(long, default_value_t = 0, value_name = "N")]
    jobs: usize,
    /// Grounding limit while collecting traces.
    #[arg(long, default_value_t = 32_768)]
    max_groundings: usize,
    /// Fast Downward expansion limit inside each leaf.
    #[arg(long, default_value_t = 100_000)]
    leaf_max_expansions: usize,
    /// Explicit HDDL parser path; otherwise `HDDL_PARSER`/`PATH` is used.
    #[arg(long)]
    parser: Option<PathBuf>,
    /// Print one line per collected problem.
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug)]
struct MethodExample {
    graph: DecompositionGraph,
    target: Vec<f32>,
    label: String,
}

#[derive(Debug)]
struct PolicyExample {
    graph: DecompositionGraph,
    target: Vec<f32>,
    value_target: f32,
    label: String,
}

#[derive(Debug)]
enum Example {
    Method(MethodExample),
    Policy(PolicyExample),
}

impl Example {
    fn task(&self) -> TrainTask {
        match self {
            Example::Method(_) => TrainTask::Method,
            Example::Policy(_) => TrainTask::Policy,
        }
    }
}

#[derive(Debug, Default)]
struct TraceCollector {
    methods: Vec<MethodTrace>,
    searches: Vec<SearchTrace>,
}

struct CollectedProblem {
    methods: Vec<MethodTrace>,
    searches: Vec<SearchTrace>,
    refined: Option<(usize, usize)>,
    refinement_error: Option<String>,
}

struct LoadedProblem {
    domain: Domain,
    problem: Problem,
}

struct EvaluatedProblem {
    domain_path: PathBuf,
    problem_path: PathBuf,
    domain: Domain,
    problem: Problem,
    plan: Vec<String>,
}

struct PreparedExample {
    graph: GraphTensors,
    indices: Tensor<1, Int>,
    target: Tensor<1>,
    value_target: Option<Tensor<1>>,
    task: TrainTask,
    label: String,
}

impl PreparedExample {
    fn task(&self) -> TrainTask {
        self.task
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl DecompositionTraceSink for TraceCollector {
    fn method_trace(&mut self, trace: MethodTrace) {
        self.methods.push(trace);
    }

    fn search_trace(&mut self, trace: SearchTrace) {
        self.searches.push(trace);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        bail!("--epochs must be at least 1");
    }
    if args.max_samples == 0 {
        bail!("--max-samples must be at least 1");
    }
    #[cfg(any(feature = "neural", feature = "neural-gpu"))]
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|error| anyhow::anyhow!("cannot configure Rayon: {error}"))?;
    }
    let run_started = Instant::now();
    let parser = resolve_parser_config(args.parser.as_deref()).map_err(anyhow::Error::msg)?;
    let pairs = input_pairs(&args)?;
    if pairs.is_empty() {
        bail!("no domain/problem pairs found");
    }

    let (method_traces, search_traces) = collect_traces(&args, &parser, &pairs)?;
    let examples = build_examples(args.task, &method_traces, &search_traces, args.max_samples)?;
    if examples.is_empty() {
        bail!(
            "no usable {} decision trace found",
            match args.task {
                TrainTask::Method => "method-ranking",
                TrainTask::Policy => "frontier-policy",
                TrainTask::Both => "decomposition",
            }
        );
    }
    let method_count = examples
        .iter()
        .filter(|example| example.task() == TrainTask::Method)
        .count();
    let policy_count = examples.len() - method_count;
    let trained_kind = match args.task {
        TrainTask::Method if method_count == 0 => {
            bail!("no usable method-ranking trace was found")
        }
        TrainTask::Policy if policy_count == 0 => {
            bail!(
                "no usable frontier-policy trace was found; the domains need an unordered method network"
            )
        }
        TrainTask::Both if method_count == 0 && policy_count == 0 => {
            bail!("no usable decomposition decision trace found")
        }
        TrainTask::Both if method_count == 0 => {
            eprintln!("warning: no method-ranking trace was found; training the policy head only");
            GuidanceKind::Policy
        }
        TrainTask::Both if policy_count == 0 => {
            eprintln!("warning: no frontier-policy trace was found; training the method head only");
            GuidanceKind::Method
        }
        TrainTask::Method => GuidanceKind::Method,
        TrainTask::Policy => GuidanceKind::Policy,
        TrainTask::Both => GuidanceKind::Both,
    };
    println!(
        "collected {} trace(s), {} training example(s) ({} method, {} policy) in {}",
        method_traces.len() + search_traces.len(),
        examples.len(),
        method_count,
        policy_count,
        format_duration(run_started.elapsed().as_secs_f64())
    );

    train(&args, &examples, trained_kind, run_started)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

fn input_pairs(args: &Args) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut domains = args.domain.clone();
    if domains.is_empty() {
        find_domains(&args.ipc_dir, &mut domains)?;
    }
    let mut pairs = Vec::new();
    for domain in domains {
        let problems = if args.problem.is_empty() {
            discover_sibling_problems(&domain)
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

fn find_domains(root: &Path, domains: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        bail!("IPC directory does not exist: {}", root.display());
    }
    let mut entries: Vec<_> = fs::read_dir(root)
        .with_context(|| format!("cannot read {}", root.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            find_domains(&path, domains)?;
        } else if path
            .file_name()
            .map(|name| name.eq_ignore_ascii_case("domain.hddl"))
            .unwrap_or(false)
        {
            domains.push(path);
        }
    }
    Ok(())
}

fn collect_traces(
    args: &Args,
    parser: &ParserConfig,
    pairs: &[(PathBuf, PathBuf)],
) -> Result<(Vec<MethodTrace>, Vec<SearchTrace>)> {
    let evaluation_options = EvaluationOptions {
        max_groundings: args.max_groundings,
        leaf_max_expansions: args.leaf_max_expansions,
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };
    let refinement_options = RefinementOptions {
        max_groundings: args.max_groundings,
        max_depth: 64,
    };
    let mut failures: BTreeMap<String, (usize, PathBuf)> = BTreeMap::new();

    // Parse and load inputs in parallel. This stage does not touch SCIP.
    #[cfg(any(feature = "neural", feature = "neural-gpu"))]
    let loaded: Vec<Result<LoadedProblem>> = pairs
        .par_iter()
        .map(|(domain_path, problem_path)| load_one(parser, domain_path, problem_path))
        .collect();
    #[cfg(not(any(feature = "neural", feature = "neural-gpu")))]
    let loaded: Vec<Result<LoadedProblem>> = pairs
        .iter()
        .map(|(domain_path, problem_path)| load_one(parser, domain_path, problem_path))
        .collect();

    // `russcip` creates one model per call and its own test suite exercises
    // concurrent models. If a downstream SCIP build is not thread-safe, run
    // with `--jobs 1`; the parallel iterator then degenerates to serial work.
    #[cfg(any(feature = "neural", feature = "neural-gpu"))]
    let evaluated_with_paths: Vec<(PathBuf, PathBuf, Result<EvaluatedProblem>)> = pairs
        .par_iter()
        .zip(loaded.into_par_iter())
        .map(|((domain_path, problem_path), loaded)| {
            let result = loaded.and_then(|loaded| {
                let (domain, problem, plan) = evaluate_one(loaded, &evaluation_options)?;
                Ok(EvaluatedProblem {
                    domain_path: domain_path.clone(),
                    problem_path: problem_path.clone(),
                    domain,
                    problem,
                    plan,
                })
            });
            (domain_path.clone(), problem_path.clone(), result)
        })
        .collect();
    #[cfg(not(any(feature = "neural", feature = "neural-gpu")))]
    let evaluated_with_paths: Vec<(PathBuf, PathBuf, Result<EvaluatedProblem>)> = pairs
        .iter()
        .zip(loaded)
        .map(|((domain_path, problem_path), loaded)| {
            let result = loaded.and_then(|loaded| {
                let (domain, problem, plan) = evaluate_one(loaded, &evaluation_options)?;
                Ok(EvaluatedProblem {
                    domain_path: domain_path.clone(),
                    problem_path: problem_path.clone(),
                    domain,
                    problem,
                    plan,
                })
            });
            (domain_path.clone(), problem_path.clone(), result)
        })
        .collect();

    let mut evaluated = Vec::new();
    for (domain_path, problem_path, result) in evaluated_with_paths {
        match result {
            Ok(evaluated_problem) => evaluated.push(evaluated_problem),
            Err(error) => {
                let message = format!("{error:#}");
                if args.verbose {
                    eprintln!(
                        "skipping {} + {}: {message}",
                        domain_path.display(),
                        problem_path.display()
                    );
                }
                let entry = failures
                    .entry(message)
                    .or_insert_with(|| (0, problem_path.clone()));
                entry.0 += 1;
            }
        }
    }

    #[cfg(any(feature = "neural", feature = "neural-gpu"))]
    let results: Vec<CollectedProblem> = evaluated
        .par_iter()
        .map(|evaluated| refine_one(evaluated, &refinement_options))
        .collect();
    #[cfg(not(any(feature = "neural", feature = "neural-gpu")))]
    let results: Vec<CollectedProblem> = evaluated
        .iter()
        .map(|evaluated| refine_one(evaluated, &refinement_options))
        .collect();

    let mut methods = Vec::new();
    let mut searches = Vec::new();
    for (evaluated, mut collected) in evaluated.into_iter().zip(results) {
        if args.verbose {
            if let Some((actions, methods)) = collected.refined {
                println!("  refined {actions} action(s), {methods} method(s)");
            }
            if let Some(error) = &collected.refinement_error {
                eprintln!("  refinement ended early: {error}");
            }
            println!(
                "{} + {}: {} method trace(s), {} frontier trace(s)",
                evaluated.domain_path.display(),
                evaluated.problem_path.display(),
                collected.methods.len(),
                collected.searches.len()
            );
        }
        methods.append(&mut collected.methods);
        searches.append(&mut collected.searches);
    }
    for (reason, (count, example)) in failures {
        eprintln!(
            "skipped {count} problem(s), e.g. {}: {reason}",
            example.display()
        );
    }
    Ok((methods, searches))
}

fn load_one(
    parser: &ParserConfig,
    domain_path: &Path,
    problem_path: &Path,
) -> Result<LoadedProblem> {
    let program =
        parse_program(parser, domain_path, Some(problem_path)).map_err(anyhow::Error::msg)?;
    let domain: Domain = process_domain(&program.domain).map_err(anyhow::Error::msg)?;
    let problem_ast = program
        .problem
        .ok_or_else(|| anyhow::anyhow!("the parser returned no problem"))?;
    let problem: Problem = load_problem(&domain, &problem_ast).map_err(anyhow::Error::msg)?;
    Ok(LoadedProblem { domain, problem })
}

fn evaluate_one(
    loaded: LoadedProblem,
    evaluation_options: &EvaluationOptions,
) -> Result<(Domain, Problem, Vec<String>)> {
    let LoadedProblem { domain, problem } = loaded;
    let mut merger = SgplanMerger::default().without_initializer();
    let evaluation = evaluate(&domain, &problem, &mut merger, evaluation_options)
        .with_context(|| "evaluation failed")?;
    Ok((domain, problem, evaluation.plan))
}

fn refine_one(evaluated: &EvaluatedProblem, options: &RefinementOptions) -> CollectedProblem {
    if evaluated.plan.is_empty() {
        return CollectedProblem {
            methods: Vec::new(),
            searches: Vec::new(),
            refined: None,
            refinement_error: None,
        };
    }

    let mut collector = TraceCollector::default();
    let refinement = {
        let mut session = DecompositionSession {
            trace: Some(&mut collector),
            ..DecompositionSession::default()
        };
        refine_plan_with_session(
            &evaluated.domain,
            &evaluated.problem,
            &evaluated.plan,
            options,
            &mut session,
        )
    };
    let (refined, refinement_error) = match refinement {
        Ok(refinement) => (
            Some((refinement.plan.len(), refinement.method_count())),
            None,
        ),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    CollectedProblem {
        methods: collector.methods,
        searches: collector.searches,
        refined,
        refinement_error,
    }
}

fn build_examples(
    task: TrainTask,
    method_traces: &[MethodTrace],
    search_traces: &[SearchTrace],
    max_samples: usize,
) -> Result<Vec<Example>> {
    // Give each head a budget of its own. If one stream is sparse, the other
    // head receives the unused budget instead of leaving it idle.
    let (method_budget, policy_budget) = if task == TrainTask::Both {
        let method_available = method_traces
            .iter()
            .filter(|trace| usable_method(trace))
            .count();
        let policy_available = search_traces
            .iter()
            .filter(|trace| usable_policy(trace))
            .count();
        let half = max_samples.div_ceil(2);
        let method_budget = method_available.min(max_samples);
        let policy_budget = policy_available.min(max_samples);
        if method_budget + policy_budget <= max_samples {
            (method_budget, policy_budget)
        } else {
            let methods = method_budget.min(max_samples - policy_budget.min(half));
            let policies = policy_budget.min(max_samples - methods);
            (methods, policies)
        }
    } else if task == TrainTask::Method {
        (max_samples, 0)
    } else {
        (0, max_samples)
    };
    let method_examples = build_method_examples(method_traces, method_budget);
    let policy_examples = build_policy_examples(search_traces, policy_budget);
    if !matches!(task, TrainTask::Both) {
        return Ok(method_examples.into_iter().chain(policy_examples).collect());
    }

    // Alternate the two heads so that a `--task both` run sees both losses
    // interleaved instead of training one head for the whole first half.
    let mut examples = Vec::with_capacity(method_examples.len() + policy_examples.len());
    let mut methods = method_examples.into_iter();
    let mut policies = policy_examples.into_iter();
    loop {
        let mut pushed = false;
        if let Some(example) = methods.next() {
            examples.push(example);
            pushed = true;
        }
        if let Some(example) = policies.next() {
            examples.push(example);
            pushed = true;
        }
        if !pushed {
            break;
        }
    }
    Ok(examples)
}

fn usable_method(trace: &MethodTrace) -> bool {
    trace.candidates.len() >= 2
        && trace.successful.len() == trace.candidates.len()
        && trace.successful.iter().any(|successful| *successful)
}

fn usable_policy(trace: &SearchTrace) -> bool {
    trace.ready.len() >= 2
        && trace.order.len() == trace.ready.len()
        && trace.successful.len() == trace.order.len()
        && trace.successful.iter().any(|successful| *successful)
}

#[cfg(any(feature = "neural", feature = "neural-gpu"))]
const EXAMPLE_CHUNK: usize = 512;

#[cfg(any(feature = "neural", feature = "neural-gpu"))]
fn build_method_examples(traces: &[MethodTrace], budget: usize) -> Vec<Example> {
    if budget == 0 {
        return Vec::new();
    }
    let mut examples = Vec::with_capacity(budget);
    let chunk_size = EXAMPLE_CHUNK.min(budget).max(1);
    for chunk in traces.chunks(chunk_size) {
        let mut batch: Vec<Example> = chunk
            .par_iter()
            .filter_map(method_example)
            .map(Example::Method)
            .collect();
        let remaining = budget - examples.len();
        if batch.len() > remaining {
            batch.truncate(remaining);
        }
        examples.extend(batch);
        if examples.len() >= budget {
            break;
        }
    }
    examples
}

#[cfg(not(any(feature = "neural", feature = "neural-gpu")))]
fn build_method_examples(traces: &[MethodTrace], budget: usize) -> Vec<Example> {
    traces
        .iter()
        .filter_map(method_example)
        .take(budget)
        .map(Example::Method)
        .collect()
}

#[cfg(any(feature = "neural", feature = "neural-gpu"))]
fn build_policy_examples(traces: &[SearchTrace], budget: usize) -> Vec<Example> {
    if budget == 0 {
        return Vec::new();
    }
    let mut examples = Vec::with_capacity(budget);
    let chunk_size = EXAMPLE_CHUNK.min(budget).max(1);
    for chunk in traces.chunks(chunk_size) {
        let mut batch: Vec<Example> = chunk
            .par_iter()
            .filter_map(policy_example)
            .map(Example::Policy)
            .collect();
        let remaining = budget - examples.len();
        if batch.len() > remaining {
            batch.truncate(remaining);
        }
        examples.extend(batch);
        if examples.len() >= budget {
            break;
        }
    }
    examples
}

#[cfg(not(any(feature = "neural", feature = "neural-gpu")))]
fn build_policy_examples(traces: &[SearchTrace], budget: usize) -> Vec<Example> {
    traces
        .iter()
        .filter_map(policy_example)
        .take(budget)
        .map(Example::Policy)
        .collect()
}

fn method_example(trace: &MethodTrace) -> Option<MethodExample> {
    if trace.candidates.len() < 2 || trace.successful.len() != trace.candidates.len() {
        return None;
    }
    let positives = trace.successful.iter().filter(|value| **value).count();
    if positives == 0 {
        return None;
    }
    let graph = DecompositionGraph::method(&trace.context, &trace.candidates);
    if graph.candidate_indices.len() != trace.candidates.len() {
        return None;
    }
    let target = normalized_labels(&trace.successful);
    Some(MethodExample {
        graph,
        target,
        label: trace.context.task.name.clone(),
    })
}

fn policy_example(trace: &SearchTrace) -> Option<PolicyExample> {
    if trace.ready.len() < 2
        || trace.order.len() != trace.ready.len()
        || trace.successful.len() != trace.order.len()
    {
        return None;
    }
    let positives = trace.successful.iter().filter(|value| **value).count();
    if positives == 0 {
        return None;
    }
    let graph = DecompositionGraph::frontier(&trace.context);
    let mut labels = vec![false; graph.ready_indices.len()];
    for (slot, action) in trace.order.iter().enumerate() {
        if let Some(position) = graph.ready_indices.iter().position(|index| index == action) {
            labels[position] = trace.successful[slot];
        }
    }
    if !labels.iter().any(|value| *value) {
        return None;
    }
    let remaining = trace
        .context
        .remaining
        .iter()
        .filter(|value| **value)
        .count();
    let value_target = if trace.context.tasks.is_empty() {
        0.0
    } else {
        remaining as f32 / trace.context.tasks.len() as f32
    };
    Some(PolicyExample {
        graph,
        target: normalized_labels(&labels),
        value_target,
        label: format!("frontier depth {}", trace.context.depth),
    })
}

fn normalized_labels(labels: &[bool]) -> Vec<f32> {
    let positives = labels.iter().filter(|value| **value).count().max(1);
    labels
        .iter()
        .map(|value| if *value { 1.0 / positives as f32 } else { 0.0 })
        .collect()
}

#[cfg(feature = "neural-gpu")]
fn training_device() -> Device {
    println!("training device: wgpu GPU backend");
    Device::wgpu(DeviceKind::DefaultDevice).autodiff()
}

#[cfg(not(feature = "neural-gpu"))]
fn training_device() -> Device {
    println!("training device: default CPU backend (enable neural-gpu for GPU)");
    Device::default().autodiff()
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "?".to_string();
    }
    if seconds < 10.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    let hours = total / 3_600;
    let minutes = (total % 3_600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn train(
    args: &Args,
    examples: &[Example],
    trained_kind: GuidanceKind,
    run_started: Instant,
) -> Result<()> {
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let training_started = Instant::now();
    let device = training_device();
    let mut model = DecompositionModel::new(&device);
    let mut optimizer = AdamConfig::new().init();
    let prepared: Vec<PreparedExample> = examples
        .iter()
        .filter_map(|example| prepare_example(example, &device))
        .collect();
    if prepared.len() != examples.len() {
        bail!(
            "{} training example(s) could not be prepared for the selected device",
            examples.len() - prepared.len()
        );
    }
    let total_steps = args.epochs.saturating_mul(prepared.len());
    let mut best_loss = f64::INFINITY;
    let mut overall_total = 0.0f64;
    let mut overall_count = 0usize;

    println!(
        "training {} example(s), {} epoch(s), feature_dim={DECOMPOSITION_FEATURE_DIM}, hidden={DECOMPOSITION_HIDDEN}",
        prepared.len(),
        args.epochs
    );
    for epoch in 0..args.epochs {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for offset in 0..prepared.len() {
            let index = (offset * 7919 + epoch * 104_729) % prepared.len();
            let example = &prepared[index];
            let loss = example_loss(&model, example);
            let step_loss = loss
                .clone()
                .and_then(|loss| loss.into_data().try_to_vec::<f32>().ok())
                .and_then(|values| values.first().copied());
            if let Some(loss) = loss {
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(args.learning_rate, model, gradients);
            }
            if let Some(value) = step_loss.filter(|value| value.is_finite()) {
                total += value;
                count += 1;
                overall_total += f64::from(value);
                overall_count += 1;
            }

            let global_step = epoch * prepared.len() + offset + 1;
            let percent = if total_steps == 0 {
                100.0
            } else {
                global_step as f64 / total_steps as f64 * 100.0
            };
            let elapsed = training_started.elapsed().as_secs_f64();
            let average_step = elapsed / global_step as f64;
            let remaining = average_step * total_steps.saturating_sub(global_step) as f64;
            println!(
                "progress {percent:6.2}% ({global_step}/{total_steps}) | epoch {}/{} | {} {} | loss {} | elapsed {} | eta {}",
                epoch + 1,
                args.epochs,
                match example.task() {
                    TrainTask::Method => "method",
                    TrainTask::Policy => "policy",
                    TrainTask::Both => "both",
                },
                example.label(),
                step_loss
                    .filter(|value| value.is_finite())
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                format_duration(elapsed),
                format_duration(remaining)
            );
        }
        if count == 0 {
            bail!("no finite training loss was produced");
        }
        let mean = f64::from(total / count as f32);
        best_loss = best_loss.min(mean);
        let overall_mean = if overall_count == 0 {
            0.0
        } else {
            overall_total / overall_count as f64
        };
        let completed_steps = (epoch + 1) * prepared.len();
        let elapsed = training_started.elapsed().as_secs_f64();
        let average_step = elapsed / completed_steps as f64;
        let remaining =
            average_step * args.epochs.saturating_sub(epoch + 1) as f64 * prepared.len() as f64;
        println!(
            "epoch {}/{} complete: loss {mean:.6e} best {best_loss:.6e} overall {overall_mean:.6e} | elapsed {} | eta {}",
            epoch + 1,
            args.epochs,
            format_duration(elapsed),
            format_duration(remaining)
        );
    }

    let elapsed = training_started.elapsed().as_secs_f64();
    println!(
        "training complete: {} epoch(s), {} example(s), best epoch loss {best_loss:.6e} | training time {} | total run time {}",
        args.epochs,
        examples.len(),
        format_duration(elapsed),
        format_duration(run_started.elapsed().as_secs_f64())
    );

    model
        .clone()
        .save_file(&args.output)
        .with_context(|| format!("cannot save {}", args.output.display()))?;
    let metadata = args.output.with_extension("json");
    let method_count = examples
        .iter()
        .filter(|example| example.task() == TrainTask::Method)
        .count();
    let policy_count = examples.len() - method_count;
    fs::write(
        &metadata,
        format!(
            "{{\"task\":\"{}\",\"hidden\":{},\"feature_dim\":{},\"examples\":{},\"method_examples\":{},\"policy_examples\":{}}}\n",
            trained_kind.name(),
            DECOMPOSITION_HIDDEN,
            DECOMPOSITION_FEATURE_DIM,
            examples.len(),
            method_count,
            policy_count
        ),
    )
    .with_context(|| format!("cannot write {}", metadata.display()))?;
    println!("wrote {}", metadata.display());
    Ok(())
}

fn prepare_example(example: &Example, device: &Device) -> Option<PreparedExample> {
    match example {
        Example::Method(example) => {
            if example.graph.candidate_indices.len() < 2
                || example.target.len() != example.graph.candidate_indices.len()
            {
                return None;
            }
            Some(PreparedExample {
                graph: example.graph.tensors(device),
                indices: index_tensor(&example.graph.candidate_indices, device),
                target: tensor(&example.target, device),
                value_target: None,
                task: TrainTask::Method,
                label: example.label.clone(),
            })
        }
        Example::Policy(example) => {
            if example.graph.ready_indices.len() < 2
                || example.target.len() != example.graph.ready_indices.len()
            {
                return None;
            }
            Some(PreparedExample {
                graph: example.graph.tensors(device),
                indices: index_tensor(&example.graph.ready_indices, device),
                target: tensor(&example.target, device),
                value_target: Some(tensor(&[example.value_target], device)),
                task: TrainTask::Policy,
                label: example.label.clone(),
            })
        }
    }
}

fn example_loss(model: &DecompositionModel, example: &PreparedExample) -> Option<Tensor<1>> {
    let (method_scores, policy_scores, value) = model.forward(&example.graph);
    match example.task {
        TrainTask::Method => {
            let scores = method_scores.select(0, example.indices.clone());
            Some(cross_entropy(scores, example.target.clone()))
        }
        TrainTask::Policy => {
            let scores = policy_scores.select(0, example.indices.clone());
            let policy_loss = cross_entropy(scores, example.target.clone());
            let value_target = example.value_target.clone()?;
            let value_loss = (value - value_target).square().mean().mul_scalar(0.1);
            Some(policy_loss + value_loss)
        }
        TrainTask::Both => None,
    }
}

fn cross_entropy(scores: Tensor<1>, target: Tensor<1>) -> Tensor<1> {
    let log_probs = activation::softmax(scores, 0).log();
    (target * log_probs).sum().mul_scalar(-1.0)
}

fn index_tensor(indices: &[usize], device: &Device) -> Tensor<1, Int> {
    Tensor::<1, Int>::from_data(
        TensorData::new(
            indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
            [indices.len()],
        ),
        device,
    )
}

fn tensor(values: &[f32], device: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(values.to_vec(), [values.len()]), device)
}
