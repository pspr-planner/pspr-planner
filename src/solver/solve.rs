use crate::solver::domain::Domain;
use crate::solver::evaluation_tree::{
    self, EvaluationOptions, MethodSummary, Refinement, RefinementOptions, SgplanMerger,
};
#[cfg(feature = "neural-base")]
use crate::solver::evaluation_tree::{
    DecompositionSession, GuidanceKind, MethodRanker, NeuralDecompositionModel, SearchPolicy,
};
use crate::solver::problem::Problem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveResult {
    pub problem: String,
    pub plan: Vec<String>,
    /// Whether every leaf of the evaluation tree was solved and every merge
    /// satisfied its shared constraints.
    pub solved: bool,
}

/// Build the constraint-partitioned evaluation tree and evaluate it with the
/// SGPlan-style merger.
///
/// The returned plan is empty when no leaf of the tree can be solved (or when
/// the problem has no initial task network). A partially solved tree returns
/// the steps that were found and sets [`SolveResult::solved`] to `false`.
pub fn solve(domain: &Domain, problem: &Problem) -> Result<SolveResult, String> {
    solve_with_options(domain, problem, &EvaluationOptions::default())
}

/// Solve with explicit evaluation options (for example enabling primitive
/// method summaries).
pub fn solve_with_options(
    domain: &Domain,
    problem: &Problem,
    options: &EvaluationOptions,
) -> Result<SolveResult, String> {
    let mut merger = SgplanMerger::default();
    let evaluation = evaluation_tree::evaluate(domain, problem, &mut merger, options)
        .map_err(|error| format!("{error:#}"))?;
    Ok(SolveResult {
        problem: problem.name.clone(),
        plan: evaluation.plan,
        solved: evaluation.solved,
    })
}

/// Solve the initial task network and refine its abstract plan into a complete
/// HTN plan with primitive method summaries (C+G).
pub fn solve_hierarchical(domain: &Domain, problem: &Problem) -> Result<Refinement, String> {
    let evaluation = EvaluationOptions {
        method_summary: MethodSummary::PrimitiveOnly,
        ..EvaluationOptions::default()
    };
    solve_hierarchical_with_options(domain, problem, &RefinementOptions::default(), &evaluation)
}

/// Hierarchical refinement with explicit options.
pub fn solve_hierarchical_with_options(
    domain: &Domain,
    problem: &Problem,
    refinement: &RefinementOptions,
    evaluation: &EvaluationOptions,
) -> Result<Refinement, String> {
    #[cfg(feature = "neural-base")]
    {
        let method_model = model_from_env("PSPR_METHOD_RANKER", Some(GuidanceKind::Method));
        let policy_model = model_from_env("PSPR_SEARCH_POLICY", Some(GuidanceKind::Policy));
        if method_model.is_some() || policy_model.is_some() {
            let mut session = DecompositionSession {
                method_ranker: method_model
                    .as_ref()
                    .map(|model| model as &dyn MethodRanker),
                search_policy: policy_model
                    .as_ref()
                    .map(|model| model as &dyn SearchPolicy),
                trace: None,
            };
            return evaluation_tree::solve_hierarchical_with_session(
                domain,
                problem,
                refinement,
                evaluation,
                &mut session,
            )
            .map_err(|error| format!("{error:#}"));
        }
        if let Some(model) = model_from_env("PSPR_DECOMPOSITION_MODEL", None) {
            let kind = model.guidance_kind();
            let mut session = DecompositionSession {
                method_ranker: kind.uses_method().then_some(&model as &dyn MethodRanker),
                search_policy: kind.uses_policy().then_some(&model as &dyn SearchPolicy),
                trace: None,
            };
            return evaluation_tree::solve_hierarchical_with_session(
                domain,
                problem,
                refinement,
                evaluation,
                &mut session,
            )
            .map_err(|error| format!("{error:#}"));
        }
    }
    evaluation_tree::solve_hierarchical(domain, problem, refinement, evaluation)
        .map_err(|error| format!("{error:#}"))
}

/// Load an optional decomposition checkpoint. Missing or malformed
/// checkpoints are reported and then ignored so the symbolic path stays
/// available.
#[cfg(feature = "neural-base")]
fn model_from_env(
    variable: &str,
    required: Option<GuidanceKind>,
) -> Option<NeuralDecompositionModel> {
    let path = std::env::var_os(variable)?;
    let path = std::path::PathBuf::from(path);
    match NeuralDecompositionModel::from_path(&path) {
        Ok(model) => {
            let kind = model.guidance_kind();
            let supported = match required {
                None => true,
                Some(GuidanceKind::Method) => kind.uses_method(),
                Some(GuidanceKind::Policy) => kind.uses_policy(),
                Some(GuidanceKind::Both) => kind == GuidanceKind::Both,
            };
            if !supported {
                eprintln!(
                    "warning: ignoring {variable}={}: checkpoint was trained for {} guidance, not {}",
                    path.display(),
                    kind.name(),
                    required.map(GuidanceKind::name).unwrap_or("any")
                );
                return None;
            }
            Some(model)
        }
        Err(error) => {
            eprintln!("warning: ignoring {variable}={}: {error:#}", path.display());
            None
        }
    }
}
