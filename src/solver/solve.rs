use crate::solver::domain::Domain;
use crate::solver::evaluation_tree::{
    self, EvaluationOptions, MethodSummary, Refinement, RefinementOptions,
};
#[cfg(feature = "neural-base")]
use crate::solver::evaluation_tree::{
    DecompositionSession, GuidanceKind, MethodRanker, NeuralDecompositionModel, SearchPolicy,
};
use crate::solver::problem::Problem;

/// Solve the initial task network and refine its abstract plan into a complete
/// HTN plan with primitive method summaries (C+G).
pub fn solve(domain: &Domain, problem: &Problem) -> Result<Refinement, String> {
    solve_hierarchical(domain, problem)
}

/// Solve the initial task network and refine its abstract plan into a complete
/// HTN plan with primitive method summaries (C+G).
pub fn solve_hierarchical(domain: &Domain, problem: &Problem) -> Result<Refinement, String> {
    let evaluation = EvaluationOptions {
        method_summary: MethodSummary::PrimitiveOnly,
        implicit_orderings: true,
        ..EvaluationOptions::default()
    };
    solve_hierarchical_with_options(domain, problem, &RefinementOptions::default(), &evaluation)
}

/// Hierarchical refinement with explicit options.
///
/// When the first attempt fails and the liftability analysis finds an implicit
/// ordering that could repair the projection, the solve is retried with those
/// orderings inserted into the task network
/// ([`EvaluationOptions::implicit_orderings`]).
pub fn solve_hierarchical_with_options(
    domain: &Domain,
    problem: &Problem,
    refinement: &RefinementOptions,
    evaluation: &EvaluationOptions,
) -> Result<Refinement, String> {
    match solve_hierarchical_once(domain, problem, refinement, evaluation) {
        Ok(refinement) => Ok(refinement),
        Err(first_error) if !evaluation.implicit_orderings => {
            let liftability = evaluation_tree::LiftabilityOptions {
                max_groundings: evaluation.max_groundings,
                ..evaluation_tree::LiftabilityOptions::default()
            };
            let repairable = evaluation_tree::repair_orderings(domain, problem, &liftability)
                .map(|orderings| !orderings.is_empty())
                .unwrap_or(false);
            if !repairable {
                return Err(first_error);
            }
            let repaired = EvaluationOptions {
                implicit_orderings: true,
                ..evaluation.clone()
            };
            solve_hierarchical_once(domain, problem, refinement, &repaired).map_err(|_| first_error)
        }
        Err(error) => Err(error),
    }
}

fn solve_hierarchical_once(
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
