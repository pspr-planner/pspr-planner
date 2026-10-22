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
        Ok(CompiledTask {
            task: self,
            dtgs,
            causal_graph,
            pruned,
            successor_generator,
        })
    }
}
