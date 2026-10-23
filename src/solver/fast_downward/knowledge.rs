//! Compiled knowledge for a planning task (Section 5 of Helmert, JAIR 2006).

use anyhow::Result;

use super::Task;
use super::successor::SuccessorGenerator;
use super::transitions::{
    CausalGraph, DomainTransitionGraph, PrunedCausalGraph, build_domain_transition_graphs,
    prune_and_simplify_dtgs,
};

/// A task together with its compiled knowledge.
#[derive(Debug, Clone)]
pub struct CompiledTask<'a> {
    pub task: &'a Task,
    pub dtgs: Vec<DomainTransitionGraph>,
    pub causal_graph: CausalGraph,
    pub pruned: PrunedCausalGraph,
    pub successor_generator: SuccessorGenerator,
    /// Identifier of `(variable, 0)` for every variable.
    ///
    /// The heuristics index facts densely by `fact_offsets[v] + value`, and the
    /// offsets only depend on the variable domains, so they are computed once
    /// here instead of on every state evaluation.
    pub fact_offsets: Vec<usize>,
    /// Number of facts, i.e. the size of the dense fact arrays.
    pub fact_count: usize,
}

impl<'a> CompiledTask<'a> {
    /// Dense identifier of the fact `(variable, value)`.
    #[inline]
    pub fn fact_id(&self, variable: usize, value: usize) -> usize {
        self.fact_offsets[variable] + value
    }
}

impl Task {
    /// Compile the task (Section 5).
    pub fn compile(&self) -> Result<CompiledTask<'_>> {
        self.validate()?;
        let dtgs = build_domain_transition_graphs(self);
        let causal_graph = CausalGraph::build(self, &dtgs);
        let pruned = causal_graph.prune(self);
        let dtgs = prune_and_simplify_dtgs(dtgs, &pruned);
        let successor_generator = SuccessorGenerator::build(self);
        let mut fact_offsets = Vec::with_capacity(self.variables.len());
        let mut fact_count = 0;
        for variable in &self.variables {
            fact_offsets.push(fact_count);
            fact_count += variable.size;
        }
        Ok(CompiledTask {
            task: self,
            dtgs,
            causal_graph,
            pruned,
            successor_generator,
            fact_offsets,
            fact_count,
        })
    }
}
