//! Train the Design 1 heterogeneous GAT warm start for the merge solver.
//!
//! The binary collects the merge problems reached by the C+G pipeline, runs
//! the existing SGPlan merge optimizer on each of them to obtain teacher
//! positions and weights, and then minimizes a two-head regression loss with
//! Adam. The saved Burn module is loaded by `SgplanMerger` when
//! `PSPR_MERGE_MODEL` points at it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn::module::Module;
#[cfg(not(feature = "neural-gpu"))]
use burn::module::{ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams};
#[cfg(feature = "neural-gpu")]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, Tensor, TensorData};
use clap::Parser;
use pspr::parsing::discovery::discover_sibling_problems;
use pspr::parsing::parser::{ParserConfig, parse_program, resolve_parser_config};
use pspr::solver::domain::{Domain, process_domain};
use pspr::solver::evaluation_tree::{
    DEFAULT_HEADS, EvaluationOptions, FEATURE_DIM, GatModel, MergeGraph, MergeOutcome,
    MergeProblem, MergeSolver, MethodSummary, SgplanMerger, TrainingSample, evaluate,
};
use pspr::solver::problem::{Problem, load_problem};
#[cfg(not(feature = "neural-gpu"))]
use rayon::prelude::*;

#[derive(Debug, Parser)]
#[command(
    name = "train_merge_initializer",
    about = "Train the GAT warm start for the SGPlan merge solver"
)]
struct Args {
    /// Where to write the trained Burn module.
    #[arg(long, default_value = "target/merge-initializer.burnpack")]
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
    /// Adam epochs over the collected merge problems.
    #[arg(long, default_value_t = 40)]
    epochs: usize,
    /// Adam learning rate.
    #[arg(long, default_value_t = 1.0e-3)]
    learning_rate: f64,
    /// Hidden width of the attention layers.
    #[arg(long, default_value_t = 96)]
    hidden: usize,
    /// Number of attention heads.
    #[arg(long, default_value_t = DEFAULT_HEADS)]
    heads: usize,
    /// Number of samples processed per parallel CPU optimizer step.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    /// Maximum number of teacher-labelled merge problems.
    #[arg(long, default_value_t = 512)]
    max_samples: usize,
    /// Grounding limit while collecting merge problems. The training default
    /// is larger than the planner default because the IPC Assembly instances
    /// need about 28k groundings for a single compound task.
    #[arg(long, default_value_t = 32_768)]
    max_groundings: usize,
    /// Fast Downward expansion limit inside each leaf.
    #[arg(long, default_value_t = 100_000)]
    leaf_max_expansions: usize,
    /// Explicit HDDL parser path; otherwise `HDDL_PARSER`/`PATH` is used.
    #[arg(long)]
    parser: Option<PathBuf>,
    /// Print one line per collected problem instead of only per epoch.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let run_started = Instant::now();
    let parser = resolve_parser_config(args.parser.as_deref()).map_err(anyhow::Error::msg)?;
    let pairs = input_pairs(&args)?;
    if pairs.is_empty() {
        bail!("no domain/problem pairs found");
    }

    let problems = collect_problems(&args, &parser, &pairs)?;
    if problems.is_empty() {
        bail!("the C+G pipeline reached no merge nodes");
    }
    let samples = teacher_samples(&problems, args.max_samples, args.verbose)?;
    if samples.is_empty() {
        bail!("no merge problem produced a usable teacher target");
    }
    println!(
        "collected {} merge problem(s), {} labelled sample(s) in {}",
        problems.len(),
        samples.len(),
        format_duration(run_started.elapsed().as_secs_f64())
    );

    train(&args, &samples, run_started)?;
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

fn collect_problems(
    args: &Args,
    parser: &ParserConfig,
    pairs: &[(PathBuf, PathBuf)],
) -> Result<Vec<MergeProblem>> {
    let options = EvaluationOptions {
        max_groundings: args.max_groundings,
        leaf_max_expansions: args.leaf_max_expansions,
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };
    let mut problems = Vec::new();
    let mut failures: BTreeMap<String, (usize, PathBuf)> = BTreeMap::new();
    for (domain_path, problem_path) in pairs {
        let result = collect_one(parser, domain_path, problem_path, &options);
        match result {
            Ok((mut found, warning)) => {
                if let Some(warning) = warning {
                    if args.verbose {
                        eprintln!(
                            "{} + {}: {warning}",
                            domain_path.display(),
                            problem_path.display()
                        );
                    }
                    let entry = failures
                        .entry(warning)
                        .or_insert_with(|| (0, problem_path.clone()));
                    entry.0 += 1;
                }
                if args.verbose {
                    println!(
                        "{} + {}: {} merge problem(s)",
                        domain_path.display(),
                        problem_path.display(),
                        found.len()
                    );
                }
                problems.append(&mut found);
            }
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
    for (reason, (count, example)) in failures {
        eprintln!(
            "skipped {count} problem(s), e.g. {}: {reason}",
            example.display()
        );
    }
    Ok(problems)
}

fn collect_one(
    parser: &ParserConfig,
    domain_path: &Path,
    problem_path: &Path,
    options: &EvaluationOptions,
) -> Result<(Vec<MergeProblem>, Option<String>)> {
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
    let warning = evaluate(&domain, &problem, &mut recorder, options)
        .err()
        .map(|error| format!("evaluation failed: {error:#}"));
    Ok((recorder.problems, warning))
}

fn teacher_samples(
    problems: &[MergeProblem],
    max_samples: usize,
    verbose: bool,
) -> Result<Vec<TrainingSample>> {
    let teacher = SgplanMerger::default().without_initializer();
    let mut samples = Vec::new();
    for problem in problems {
        if samples.len() >= max_samples {
            break;
        }
        let graph = MergeGraph::from_problem(problem);
        if graph.child_count == 0 || graph.constraint_count == 0 || problem.children.is_empty() {
            continue;
        }
        let target = teacher.teacher_initialization(problem);
        if target.positions.len() != graph.child_count
            || target.weights.len() != graph.constraint_count
            || target.positions.iter().any(|value| !value.is_finite())
            || target.weights.iter().any(|value| !value.is_finite())
        {
            continue;
        }
        if verbose {
            println!(
                "sample node={} children={} constraints={}",
                problem.node, graph.child_count, graph.constraint_count
            );
        }
        samples.push(TrainingSample {
            graph,
            positions: target.positions,
            weights: target.weights,
        });
    }
    Ok(samples)
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

fn train(args: &Args, samples: &[TrainingSample], run_started: Instant) -> Result<()> {
    #[cfg(feature = "neural-gpu")]
    {
        train_sequential(args, samples, run_started)
    }
    #[cfg(not(feature = "neural-gpu"))]
    {
        train_parallel_cpu(args, samples, run_started)
    }
}

#[cfg(feature = "neural-gpu")]
fn train_sequential(args: &Args, samples: &[TrainingSample], run_started: Instant) -> Result<()> {
    let _batch_size = args.batch_size;
    if args.epochs == 0 {
        bail!("--epochs must be at least 1");
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let training_started = Instant::now();
    let device = training_device();
    let mut model = GatModel::new(FEATURE_DIM, args.hidden, args.heads, &device);
    let mut optimizer = AdamConfig::new().init();
    let mut best_loss = f64::INFINITY;
    let total_steps = args.epochs.saturating_mul(samples.len());
    let mut overall_total = 0.0f64;
    let mut overall_count = 0usize;

    for epoch in 0..args.epochs {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for offset in 0..samples.len() {
            let index = (offset * 7919 + epoch * 104_729) % samples.len();
            let sample = &samples[index];
            let graph = sample.graph.tensors(&device);
            let (positions, weights) = model.forward(&graph);
            let mut step_loss = None;
            if positions.dims()[0] != 0 && weights.dims()[0] != 0 {
                let position_target = Tensor::<1>::from_data(
                    TensorData::new(
                        sample
                            .positions
                            .iter()
                            .map(|value| *value as f32)
                            .collect::<Vec<_>>(),
                        [sample.positions.len()],
                    ),
                    &device,
                );
                let weight_target = Tensor::<1>::from_data(
                    TensorData::new(
                        sample
                            .weights
                            .iter()
                            .map(|value| *value as f32)
                            .collect::<Vec<_>>(),
                        [sample.weights.len()],
                    ),
                    &device,
                );
                let position_loss = (positions - position_target).square().mean();
                let weight_loss = (weights.log() - weight_target.log())
                    .square()
                    .mean()
                    .mul_scalar(0.5);
                let loss = position_loss + weight_loss;
                step_loss = loss
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .ok()
                    .and_then(|values| values.first().copied());
                if let Some(value) = step_loss {
                    total += value;
                    count += 1;
                    overall_total += f64::from(value);
                    overall_count += 1;
                }
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(args.learning_rate, model, gradients);
            }

            let global_step = epoch * samples.len() + offset + 1;
            let percent = if total_steps == 0 {
                100.0
            } else {
                global_step as f64 / total_steps as f64 * 100.0
            };
            let elapsed = training_started.elapsed().as_secs_f64();
            let average_step = elapsed / global_step as f64;
            let remaining = average_step * total_steps.saturating_sub(global_step) as f64;
            println!(
                "progress {percent:6.2}% ({global_step}/{total_steps}) | epoch {}/{} | sample {}/{} | loss {:.3e} | elapsed {} | eta {}",
                epoch + 1,
                args.epochs,
                offset + 1,
                samples.len(),
                step_loss.unwrap_or(f32::NAN),
                format_duration(elapsed),
                format_duration(remaining)
            );
        }
        if count == 0 {
            bail!("no non-empty graph during training");
        }
        let mean = f64::from(total / count as f32);
        best_loss = best_loss.min(mean);
        let overall_mean = if overall_count == 0 {
            0.0
        } else {
            overall_total / overall_count as f64
        };
        let completed_steps = (epoch + 1) * samples.len();
        let elapsed = training_started.elapsed().as_secs_f64();
        let average_step = elapsed / completed_steps as f64;
        let remaining =
            average_step * args.epochs.saturating_sub(epoch + 1) as f64 * samples.len() as f64;
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
        "training complete: {} epoch(s), {} sample(s), best epoch loss {best_loss:.6e} | training time {} | total run time {}",
        args.epochs,
        samples.len(),
        format_duration(elapsed),
        format_duration(run_started.elapsed().as_secs_f64())
    );

    model
        .clone()
        .save_file(&args.output)
        .with_context(|| format!("cannot save {}", args.output.display()))?;
    let metadata = args.output.with_extension("json");
    fs::write(
        &metadata,
        format!(
            "{{\"hidden\":{},\"heads\":{},\"feature_dim\":{},\"samples\":{}}}\n",
            args.hidden,
            args.heads,
            FEATURE_DIM,
            samples.len()
        ),
    )
    .with_context(|| format!("cannot write {}", metadata.display()))?;
    println!("wrote {}", metadata.display());
    Ok(())
}

#[cfg(not(feature = "neural-gpu"))]
fn compute_step(
    model: &GatModel,
    sample: &TrainingSample,
    device: &Device,
) -> (Option<GradientsParams>, Option<f32>) {
    let graph = sample.graph.tensors(device);
    let (positions, weights) = model.forward(&graph);
    if positions.dims()[0] == 0 || weights.dims()[0] == 0 {
        return (None, None);
    }
    let position_target = Tensor::<1>::from_data(
        TensorData::new(
            sample
                .positions
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            [sample.positions.len()],
        ),
        device,
    );
    let weight_target = Tensor::<1>::from_data(
        TensorData::new(
            sample
                .weights
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            [sample.weights.len()],
        ),
        device,
    );
    let position_loss = (positions - position_target).square().mean();
    let weight_loss = (weights.log() - weight_target.log())
        .square()
        .mean()
        .mul_scalar(0.5);
    let loss = position_loss + weight_loss;
    let value = loss
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied());
    let gradients = GradientsParams::from_grads(loss.backward(), model);
    (Some(gradients), value)
}

#[cfg(not(feature = "neural-gpu"))]
fn compute_batch(
    model: &GatModel,
    batch: &[&TrainingSample],
    device: &Device,
) -> Vec<(Option<GradientsParams>, Option<f32>)> {
    // Each rayon task gets its own module clone so forward/backward graphs are
    // independent. The parameter ids are shared, which lets the gradients be
    // averaged against the same model afterwards.
    let replicas: Vec<GatModel> = batch.iter().map(|_| model.clone()).collect();
    batch
        .par_iter()
        .zip(replicas.into_par_iter())
        .map(|(sample, replica)| compute_step(&replica, sample, device))
        .collect()
}

#[cfg(not(feature = "neural-gpu"))]
fn average_gradients(model: &GatModel, parts: &[GradientsParams]) -> GradientsParams {
    struct Average<'a> {
        parts: &'a [GradientsParams],
        out: GradientsParams,
    }

    impl ModuleVisitor for Average<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let mut sum: Option<Tensor<D>> = None;
            for part in self.parts {
                if let Some(value) = part.get::<D>(param.id) {
                    sum = Some(match sum {
                        Some(accumulated) => accumulated + value,
                        None => value,
                    });
                }
            }
            if let Some(sum) = sum {
                self.out
                    .register(param.id, sum / self.parts.len().max(1) as f32);
            }
        }
    }

    let mut visitor = Average {
        parts,
        out: GradientsParams::new(),
    };
    model.visit(&mut visitor);
    visitor.out
}

#[cfg(not(feature = "neural-gpu"))]
fn train_parallel_cpu(args: &Args, samples: &[TrainingSample], run_started: Instant) -> Result<()> {
    if args.epochs == 0 {
        bail!("--epochs must be at least 1");
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let training_started = Instant::now();
    let device = training_device();
    let threads = rayon::current_num_threads();
    println!("rayon threads: {threads}");
    let mut model = GatModel::new(FEATURE_DIM, args.hidden, args.heads, &device);
    let mut optimizer = AdamConfig::new().init();
    let mut best_loss = f64::INFINITY;
    let total_steps = args.epochs.saturating_mul(samples.len());
    let batch_size = args.batch_size.max(1);
    let mut overall_total = 0.0f64;
    let mut overall_count = 0usize;

    for epoch in 0..args.epochs {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for batch_start in (0..samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples.len());
            let batch: Vec<&TrainingSample> = (batch_start..batch_end)
                .map(|offset| {
                    let index = (offset * 7919 + epoch * 104_729) % samples.len();
                    &samples[index]
                })
                .collect();
            let results = compute_batch(&model, &batch, &device);
            let mut parts = Vec::new();
            for (local_index, (gradients, loss)) in results.into_iter().enumerate() {
                if let Some(value) = loss {
                    total += value;
                    count += 1;
                    overall_total += f64::from(value);
                    overall_count += 1;
                }
                if let Some(gradients) = gradients {
                    parts.push(gradients);
                }

                let global_step = epoch * samples.len() + batch_start + local_index + 1;
                let percent = if total_steps == 0 {
                    100.0
                } else {
                    global_step as f64 / total_steps as f64 * 100.0
                };
                let elapsed = training_started.elapsed().as_secs_f64();
                let average_step = elapsed / global_step as f64;
                let remaining = average_step * total_steps.saturating_sub(global_step) as f64;
                println!(
                    "progress {percent:6.2}% ({global_step}/{total_steps}) | epoch {}/{} | sample {}/{} | loss {:.3e} | elapsed {} | eta {}",
                    epoch + 1,
                    args.epochs,
                    batch_start + local_index + 1,
                    samples.len(),
                    loss.unwrap_or(f32::NAN),
                    format_duration(elapsed),
                    format_duration(remaining)
                );
            }
            if parts.is_empty() {
                continue;
            }
            let gradients = average_gradients(&model, &parts);
            model = optimizer.step(args.learning_rate, model, gradients);
        }
        if count == 0 {
            bail!("no non-empty graph during training");
        }
        let mean = f64::from(total / count as f32);
        best_loss = best_loss.min(mean);
        let overall_mean = if overall_count == 0 {
            0.0
        } else {
            overall_total / overall_count as f64
        };
        let completed_steps = (epoch + 1) * samples.len();
        let elapsed = training_started.elapsed().as_secs_f64();
        let average_step = elapsed / completed_steps as f64;
        let remaining =
            average_step * args.epochs.saturating_sub(epoch + 1) as f64 * samples.len() as f64;
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
        "training complete: {} epoch(s), {} sample(s), {} batch size, {} rayon thread(s), best epoch loss {best_loss:.6e} | training time {} | total run time {}",
        args.epochs,
        samples.len(),
        batch_size,
        threads,
        format_duration(elapsed),
        format_duration(run_started.elapsed().as_secs_f64())
    );

    model
        .clone()
        .save_file(&args.output)
        .with_context(|| format!("cannot save {}", args.output.display()))?;
    let metadata = args.output.with_extension("json");
    fs::write(
        &metadata,
        format!(
            "{{\"hidden\":{},\"heads\":{},\"feature_dim\":{},\"samples\":{},\"batch_size\":{}}}\n",
            args.hidden,
            args.heads,
            FEATURE_DIM,
            samples.len(),
            batch_size
        ),
    )
    .with_context(|| format!("cannot write {}", metadata.display()))?;
    println!("wrote {}", metadata.display());
    Ok(())
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
