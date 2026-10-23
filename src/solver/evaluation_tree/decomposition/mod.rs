//! Neural guidance for compound-task decomposition.
//!
//! This module implements the two selected decomposition models:
//!
//! * a method ranker that scores grounded method instances for `refine_task`;
//! * a frontier policy/value model that orders ready tasks in
//!   `search_partial_inner`.
//!
//! Both use the same typed graph representation and the same Burn encoder.
//! The symbolic refiner remains authoritative: the ranker only reorders
//! applicable methods, the policy only reorders ready tasks, and the exact
//! validator still checks the final plan.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::graph::TaskNode;

#[cfg(feature = "neural-base")]
use burn::module::Module;
#[cfg(feature = "neural-base")]
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
#[cfg(feature = "neural-base")]
use burn::tensor::{Device, Tensor, TensorData, activation, s};

/// Feature width of a decomposition graph node.
///
/// The slots are shared by the three node kinds:
/// 0/1/2 node-type flags; 3 ready; 4 preferred; 5 method-name hash;
/// 6 subtasks; 7 binding count; 8 depth; 9 applicable; 10 precondition
/// fraction; 11 task-name hash; 12 arity; 13 state size; 14 candidate count;
/// 15 ready count; 16/17 predecessor/successor counts; 18 remaining; 19 term
/// or binding hash; 20 full task/binding overlap; 22 precondition atoms;
/// 23 remaining count; 24 state hash.
pub const DECOMPOSITION_FEATURE_DIM: usize = 32;
/// Hidden width used by the unified decomposition model.
#[cfg(feature = "neural-base")]
pub const DECOMPOSITION_HIDDEN: usize = 64;

/// Which decomposition head(s) a saved model was trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuidanceKind {
    /// Only the method-ranking head is trained.
    Method,
    /// Only the frontier policy/value heads are trained.
    Policy,
    /// Both heads were trained together.
    #[default]
    Both,
}

impl GuidanceKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "method" => Some(Self::Method),
            "policy" => Some(Self::Policy),
            "both" => Some(Self::Both),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Method => "method",
            Self::Policy => "policy",
            Self::Both => "both",
        }
    }

    pub fn uses_method(self) -> bool {
        matches!(self, Self::Method | Self::Both)
    }

    pub fn uses_policy(self) -> bool {
        matches!(self, Self::Policy | Self::Both)
    }
}

/// Width of one node feature vector.
const FEATURE_DIM: usize = DECOMPOSITION_FEATURE_DIM;

/// A grounded method candidate presented to the ranker.
#[derive(Debug, Clone)]
pub struct MethodCandidate {
    pub method: String,
    pub bindings: BTreeMap<String, String>,
    pub subtasks: usize,
    pub preferred: bool,
    pub applicable: bool,
    pub precondition_atoms: usize,
    pub precondition_fraction: f64,
}

/// Context of one `refine_task` call.
#[derive(Debug, Clone)]
pub struct MethodRankingContext {
    pub task: TaskNode,
    pub state: BTreeSet<String>,
    pub depth: usize,
    pub max_depth: usize,
}

/// Context of one partial-order search node.
#[derive(Debug, Clone)]
pub struct SearchContext {
    pub tasks: Vec<TaskNode>,
    pub remaining: Vec<bool>,
    pub predecessors: Vec<HashSet<usize>>,
    pub state: BTreeSet<String>,
    pub depth: usize,
}

/// Scores method candidates for a compound task.
pub trait MethodRanker: std::fmt::Debug + Send + Sync {
    /// Return one score per candidate in `candidates`, or `None` to use the
    /// symbolic baseline order.
    fn rank(
        &self,
        context: &MethodRankingContext,
        candidates: &[MethodCandidate],
    ) -> Option<Vec<f64>>;
}

/// Orders ready tasks and optionally estimates the remaining cost.
pub trait SearchPolicy: std::fmt::Debug + Send + Sync {
    /// Return a permutation of the ready task indices, or `None` to use
    /// ascending index order.
    fn order_ready(&self, context: &SearchContext) -> Option<Vec<usize>>;

    /// Estimate the cost-to-go at this frontier. The first implementation only
    /// uses this for diagnostics and branch ordering.
    fn value(&self, _context: &SearchContext) -> Option<f64> {
        None
    }
}

/// One recorded method-ranking decision.
#[derive(Debug, Clone)]
pub struct MethodTrace {
    pub context: MethodRankingContext,
    pub candidates: Vec<MethodCandidate>,
    /// Candidate indices in the order the refiner tried them.
    pub order: Vec<usize>,
    /// `successful[i]` is true when candidate `i` produced at least one
    /// recursive refinement from this state.
    pub successful: Vec<bool>,
}

/// One recorded partial-order search decision.
#[derive(Debug, Clone)]
pub struct SearchTrace {
    pub context: SearchContext,
    pub ready: Vec<usize>,
    pub order: Vec<usize>,
    /// Parallel to `order`: whether that action eventually produced a result.
    pub successful: Vec<bool>,
}

/// Optional trace sink used by the training binary.
pub trait DecompositionTraceSink: std::fmt::Debug {
    fn method_trace(&mut self, trace: MethodTrace);
    fn search_trace(&mut self, trace: SearchTrace);
}

/// Optional neural components passed through the symbolic refiner.
#[derive(Default)]
pub struct DecompositionSession<'a> {
    pub method_ranker: Option<&'a dyn MethodRanker>,
    pub search_policy: Option<&'a dyn SearchPolicy>,
    pub trace: Option<&'a mut dyn DecompositionTraceSink>,
}

/// A graph consumed by the unified decomposition model.
#[derive(Debug, Clone)]
pub struct DecompositionGraph {
    pub node_count: usize,
    pub features: Vec<f32>,
    pub adjacency: Vec<f32>,
    /// Node indices of method candidates, in candidate order.
    pub candidate_indices: Vec<usize>,
    /// Node indices of ready tasks, in task order.
    pub ready_indices: Vec<usize>,
    pub global_index: usize,
}

impl DecompositionGraph {
    /// Build a graph for a method-ranking decision.
    pub fn method(context: &MethodRankingContext, candidates: &[MethodCandidate]) -> Self {
        let candidate_count = candidates.len();
        let node_count = candidate_count + 2;
        let global_index = node_count - 1;
        let mut features = vec![0.0; node_count * FEATURE_DIM];
        let mut adjacency = vec![0.0; node_count * node_count];
        let mut candidate_indices = Vec::with_capacity(candidate_count);

        let task_base = 0;
        features[task_base] = 1.0;
        features[task_base + 11] = hash_bucket(&context.task.name);
        features[task_base + 12] = context.task.terms.len() as f32 / 8.0;
        features[task_base + 14] = candidate_count as f32 / 32.0;
        features[task_base + 8] = normalized(context.depth, context.max_depth);
        features[task_base + 13] = (context.state.len() as f32 / 64.0).min(1.0);
        features[task_base + 19] = terms_hash(&context.task.terms);
        features[task_base + 24] = state_hash(&context.state);

        let task_terms: HashSet<&str> = context.task.terms.iter().map(String::as_str).collect();
        for (index, candidate) in candidates.iter().enumerate() {
            let node = index + 1;
            candidate_indices.push(node);
            let base = node * FEATURE_DIM;
            features[base + 1] = 1.0;
            features[base + 4] = if candidate.preferred { 1.0 } else { 0.0 };
            features[base + 5] = hash_bucket(&candidate.method);
            features[base + 6] = candidate.subtasks as f32 / 16.0;
            features[base + 7] = candidate.bindings.len() as f32 / 16.0;
            features[base + 9] = if candidate.applicable { 1.0 } else { 0.0 };
            features[base + 10] = candidate.precondition_fraction.clamp(0.0, 1.0) as f32;
            features[base + 19] = binding_hash(&candidate.bindings);
            features[base + 20] = if candidate.bindings.is_empty() {
                0.0
            } else {
                candidate
                    .bindings
                    .values()
                    .filter(|value| task_terms.contains(value.as_str()))
                    .count() as f32
                    / candidate.bindings.len() as f32
            };
            features[base + 22] = candidate.precondition_atoms as f32 / 16.0;
            features[base + 14] = candidate_count as f32 / 32.0;
            add_edge(&mut adjacency, node_count, 0, node);
            add_edge(&mut adjacency, node_count, node, 0);
            add_edge(&mut adjacency, node_count, global_index, node);
            add_edge(&mut adjacency, node_count, node, global_index);
        }

        let base = global_index * FEATURE_DIM;
        features[base + 2] = 1.0;
        features[base + 14] = candidate_count as f32 / 32.0;
        features[base + 8] = normalized(context.depth, context.max_depth);
        features[base + 13] = (context.state.len() as f32 / 64.0).min(1.0);
        features[base + 24] = state_hash(&context.state);

        connect_all(&mut adjacency, node_count);
        DecompositionGraph {
            node_count,
            features,
            adjacency,
            candidate_indices,
            ready_indices: Vec::new(),
            global_index,
        }
    }

    /// Build a graph for a partial-order frontier.
    pub fn frontier(context: &SearchContext) -> Self {
        let task_count = context.tasks.len();
        let node_count = task_count + 1;
        let global_index = node_count - 1;
        let mut features = vec![0.0; node_count * FEATURE_DIM];
        let mut adjacency = vec![0.0; node_count * node_count];
        let mut ready_indices = Vec::new();
        let remaining_count = context.remaining.iter().filter(|value| **value).count();

        for (index, task) in context.tasks.iter().enumerate() {
            let base = index * FEATURE_DIM;
            let ready = context.remaining[index]
                && !context.predecessors[index]
                    .iter()
                    .any(|before| context.remaining[*before]);
            if ready {
                ready_indices.push(index);
            }
            let successors = context
                .predecessors
                .iter()
                .filter(|set| set.contains(&index))
                .count();
            features[base] = 1.0;
            features[base + 3] = if ready { 1.0 } else { 0.0 };
            features[base + 11] = hash_bucket(&task.name);
            features[base + 12] = task.terms.len() as f32 / 8.0;
            features[base + 16] = context.predecessors[index].len() as f32 / 16.0;
            features[base + 17] = successors as f32 / 16.0;
            features[base + 19] = terms_hash(&task.terms);
            features[base + 20] = hash_bucket(&task.to_string());
            features[base + 23] = remaining_count as f32 / 32.0;
            features[base + 8] = normalized(context.depth, 32);
            features[base + 13] = (context.state.len() as f32 / 64.0).min(1.0);
            features[base + 18] = if context.remaining[index] { 1.0 } else { 0.0 };
            features[base + 24] = state_hash(&context.state);
            add_edge(&mut adjacency, node_count, global_index, index);
            add_edge(&mut adjacency, node_count, index, global_index);
        }
        for (after, predecessors) in context.predecessors.iter().enumerate() {
            for &before in predecessors {
                if before < task_count {
                    add_edge(&mut adjacency, node_count, before, after);
                    add_edge(&mut adjacency, node_count, after, before);
                }
            }
        }

        let base = global_index * FEATURE_DIM;
        features[base + 2] = 1.0;
        features[base + 15] = ready_indices.len() as f32 / 32.0;
        features[base + 23] = remaining_count as f32 / 32.0;
        features[base + 8] = normalized(context.depth, 32);
        features[base + 13] = (context.state.len() as f32 / 64.0).min(1.0);
        features[base + 24] = state_hash(&context.state);

        connect_all(&mut adjacency, node_count);
        DecompositionGraph {
            node_count,
            features,
            adjacency,
            candidate_indices: Vec::new(),
            ready_indices,
            global_index,
        }
    }

    /// Materialize the graph as Burn tensors.
    #[cfg(feature = "neural-base")]
    pub fn tensors(&self, device: &Device) -> GraphTensors {
        GraphTensors {
            features: Tensor::<2>::from_data(
                TensorData::new(self.features.clone(), [self.node_count, FEATURE_DIM]),
                device,
            ),
            adjacency: Tensor::<2>::from_data(
                TensorData::new(self.adjacency.clone(), [self.node_count, self.node_count]),
                device,
            ),
            node_count: self.node_count,
            global_index: self.global_index,
            candidate_indices: self.candidate_indices.clone(),
            ready_indices: self.ready_indices.clone(),
        }
    }
}

/// Burn tensors of one [`DecompositionGraph`].
#[cfg(feature = "neural-base")]
#[derive(Debug)]
pub struct GraphTensors {
    pub features: Tensor<2>,
    pub adjacency: Tensor<2>,
    pub node_count: usize,
    pub global_index: usize,
    pub candidate_indices: Vec<usize>,
    pub ready_indices: Vec<usize>,
}

/// One graph-attention layer for the unified decomposition model.
#[cfg(feature = "neural-base")]
#[derive(Module, Debug)]
struct GraphLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    norm: LayerNorm,
}

#[cfg(feature = "neural-base")]
impl GraphLayer {
    fn new(hidden: usize, device: &Device) -> Self {
        Self {
            query: LinearConfig::new(hidden, hidden).init(device),
            key: LinearConfig::new(hidden, hidden).init(device),
            value: LinearConfig::new(hidden, hidden).init(device),
            output: LinearConfig::new(hidden, hidden).init(device),
            norm: LayerNormConfig::new(hidden).init(device),
        }
    }

    fn forward(&self, input: Tensor<2>, adjacency: Tensor<2>) -> Tensor<2> {
        let hidden = input.dims()[1];
        let query = self.query.forward(input.clone());
        let key = self.key.forward(input.clone());
        let value = self.value.forward(input.clone());
        let bias = adjacency.sub_scalar(1.0).mul_scalar(1.0e9);
        let scores = query
            .matmul(key.transpose())
            .div_scalar((hidden as f32).sqrt())
            + bias;
        let attention = activation::softmax(scores, 1);
        let context = attention.matmul(value);
        let update = self.output.forward(context);
        activation::leaky_relu(self.norm.forward(input + update), 0.2)
    }
}

/// Unified method-ranking and frontier policy/value model.
#[cfg(feature = "neural-base")]
#[derive(Module, Debug)]
pub struct DecompositionModel {
    input: Linear,
    layer_one: GraphLayer,
    layer_two: GraphLayer,
    layer_three: GraphLayer,
    method_head: Linear,
    policy_head: Linear,
    value_head: Linear,
}

#[cfg(feature = "neural-base")]
impl DecompositionModel {
    pub fn new(device: &Device) -> Self {
        let hidden = DECOMPOSITION_HIDDEN;
        Self {
            input: LinearConfig::new(FEATURE_DIM, hidden).init(device),
            layer_one: GraphLayer::new(hidden, device),
            layer_two: GraphLayer::new(hidden, device),
            layer_three: GraphLayer::new(hidden, device),
            method_head: LinearConfig::new(hidden, 1).init(device),
            policy_head: LinearConfig::new(hidden, 1).init(device),
            value_head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// Return method-candidate scores, ready-task scores, and a scalar value.
    pub fn forward(&self, graph: &GraphTensors) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
        let hidden = self.input.forward(graph.features.clone());
        let hidden = self.layer_one.forward(hidden, graph.adjacency.clone());
        let hidden = self.layer_two.forward(hidden, graph.adjacency.clone());
        let hidden = self.layer_three.forward(hidden, graph.adjacency.clone());
        let method_scores = self
            .method_head
            .forward(hidden.clone())
            .reshape([graph.node_count]);
        let policy_scores = self
            .policy_head
            .forward(hidden.clone())
            .reshape([graph.node_count]);
        let global = hidden.slice(s![
            graph.global_index..graph.global_index + 1,
            0..DECOMPOSITION_HIDDEN
        ]);
        let value = self.value_head.forward(global).reshape([1]);
        (method_scores, policy_scores, value)
    }
}

/// Inference wrapper that implements both decomposition traits.
#[cfg(feature = "neural-base")]
#[derive(Debug)]
pub struct NeuralDecompositionModel {
    model: DecompositionModel,
    device: Device,
    kind: GuidanceKind,
}

#[cfg(feature = "neural-base")]
impl NeuralDecompositionModel {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("decomposition model does not exist: {}", path.display());
        }
        let kind = read_guidance_metadata(path)?;
        let device = Device::default();
        let model = DecompositionModel::new(&device).load_file(path);
        Ok(Self {
            model,
            device,
            kind,
        })
    }

    /// The heads that this checkpoint was trained for.
    pub fn guidance_kind(&self) -> GuidanceKind {
        self.kind
    }

    fn scores(&self, graph: &DecompositionGraph) -> Option<(Vec<f64>, Vec<f64>, f64)> {
        let tensors = graph.tensors(&self.device);
        let (method, policy, value) = self.model.forward(&tensors);
        let method = method
            .into_data()
            .try_to_vec::<f32>()
            .ok()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let policy = policy
            .into_data()
            .try_to_vec::<f32>()
            .ok()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let value = value
            .into_data()
            .try_to_vec::<f32>()
            .ok()?
            .first()
            .copied()?;
        Some((method, policy, f64::from(value)))
    }
}

#[cfg(feature = "neural-base")]
impl MethodRanker for NeuralDecompositionModel {
    fn rank(
        &self,
        context: &MethodRankingContext,
        candidates: &[MethodCandidate],
    ) -> Option<Vec<f64>> {
        if candidates.len() <= 1 {
            return None;
        }
        let graph = DecompositionGraph::method(context, candidates);
        let (method, _, _) = self.scores(&graph)?;
        if method.len() != graph.node_count {
            return None;
        }
        let scores: Vec<f64> = graph
            .candidate_indices
            .iter()
            .filter_map(|index| method.get(*index).copied())
            .collect();
        (scores.len() == candidates.len() && scores.iter().all(|score| score.is_finite()))
            .then_some(scores)
    }
}

#[cfg(feature = "neural-base")]
impl SearchPolicy for NeuralDecompositionModel {
    fn order_ready(&self, context: &SearchContext) -> Option<Vec<usize>> {
        let graph = DecompositionGraph::frontier(context);
        if graph.ready_indices.len() <= 1 {
            return None;
        }
        let (_, policy, _) = self.scores(&graph)?;
        if policy.len() != graph.node_count {
            return None;
        }
        let mut scored: Vec<(usize, f64)> = graph
            .ready_indices
            .iter()
            .filter_map(|index| policy.get(*index).copied().map(|score| (*index, score)))
            .filter(|(_, score)| score.is_finite())
            .collect();
        if scored.len() != graph.ready_indices.len() {
            return None;
        }
        scored.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        Some(scored.into_iter().map(|(index, _)| index).collect())
    }

    fn value(&self, context: &SearchContext) -> Option<f64> {
        let graph = DecompositionGraph::frontier(context);
        let (_, _, value) = self.scores(&graph)?;
        value.is_finite().then_some(value)
    }
}

#[cfg(feature = "neural-base")]
fn read_guidance_metadata(path: &std::path::Path) -> anyhow::Result<GuidanceKind> {
    let metadata_path = path.with_extension("json");
    if !metadata_path.exists() {
        return Ok(GuidanceKind::Both);
    }
    let text = std::fs::read_to_string(&metadata_path)
        .map_err(|error| anyhow::anyhow!("cannot read {}: {error}", metadata_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| anyhow::anyhow!("cannot parse {}: {error}", metadata_path.display()))?;
    if let Some(hidden) = value.get("hidden").and_then(serde_json::Value::as_u64)
        && hidden as usize != DECOMPOSITION_HIDDEN
    {
        anyhow::bail!(
            "{} was trained with hidden={hidden}, expected {DECOMPOSITION_HIDDEN}",
            metadata_path.display()
        );
    }
    if let Some(feature_dim) = value.get("feature_dim").and_then(serde_json::Value::as_u64)
        && feature_dim as usize != DECOMPOSITION_FEATURE_DIM
    {
        anyhow::bail!(
            "{} was trained with feature_dim={feature_dim}, expected {DECOMPOSITION_FEATURE_DIM}",
            metadata_path.display()
        );
    }
    match value.get("task").and_then(serde_json::Value::as_str) {
        None => Ok(GuidanceKind::Both),
        Some(task) => GuidanceKind::parse(task).ok_or_else(|| {
            anyhow::anyhow!(
                "{} has unknown task={task:?}; expected method, policy, or both",
                metadata_path.display()
            )
        }),
    }
}

fn normalized(value: usize, maximum: usize) -> f32 {
    value as f32 / maximum.max(1) as f32
}

fn hash_bucket(value: &str) -> f32 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash % 1024) as f32 / 1024.0
}

fn terms_hash(terms: &[String]) -> f32 {
    hash_bucket(&terms.join("|"))
}

fn binding_hash(bindings: &BTreeMap<String, String>) -> f32 {
    let mut key = String::new();
    for (parameter, value) in bindings {
        key.push_str(parameter);
        key.push('=');
        key.push_str(value);
        key.push(';');
    }
    hash_bucket(&key)
}

fn state_hash(state: &BTreeSet<String>) -> f32 {
    let mut key = String::new();
    for atom in state {
        key.push_str(atom);
        key.push(';');
    }
    hash_bucket(&key)
}

fn add_edge(adjacency: &mut [f32], node_count: usize, from: usize, to: usize) {
    if from < node_count && to < node_count {
        adjacency[from * node_count + to] = 1.0;
    }
}

fn connect_all(adjacency: &mut [f32], node_count: usize) {
    for node in 0..node_count {
        add_edge(adjacency, node_count, node, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_graph_connects_candidates_to_task_and_global() {
        let context = MethodRankingContext {
            task: TaskNode {
                name: "do".to_string(),
                terms: vec!["a".to_string()],
            },
            state: BTreeSet::new(),
            depth: 1,
            max_depth: 8,
        };
        let candidates = vec![
            MethodCandidate {
                method: "m1".to_string(),
                bindings: BTreeMap::new(),
                subtasks: 2,
                preferred: true,
                applicable: true,
                precondition_atoms: 1,
                precondition_fraction: 1.0,
            },
            MethodCandidate {
                method: "m2".to_string(),
                bindings: BTreeMap::new(),
                subtasks: 1,
                preferred: false,
                applicable: true,
                precondition_atoms: 0,
                precondition_fraction: 1.0,
            },
        ];
        let graph = DecompositionGraph::method(&context, &candidates);
        assert_eq!(graph.node_count, 4);
        assert_eq!(graph.candidate_indices, vec![1, 2]);
        assert_eq!(graph.adjacency[1], 1.0);
        assert_eq!(graph.adjacency[3 * 4 + 2], 1.0);
    }

    #[test]
    fn method_graph_distinguishes_bindings() {
        let context = MethodRankingContext {
            task: TaskNode {
                name: "move".to_string(),
                terms: vec!["a".to_string(), "b".to_string()],
            },
            state: BTreeSet::new(),
            depth: 1,
            max_depth: 8,
        };
        let candidate = |value: &str| MethodCandidate {
            method: "m".to_string(),
            bindings: BTreeMap::from([("?x".to_string(), value.to_string())]),
            subtasks: 1,
            preferred: false,
            applicable: true,
            precondition_atoms: 0,
            precondition_fraction: 1.0,
        };
        let graph = DecompositionGraph::method(&context, &[candidate("a"), candidate("b")]);
        let first = &graph.features[FEATURE_DIM..2 * FEATURE_DIM];
        let second = &graph.features[2 * FEATURE_DIM..3 * FEATURE_DIM];
        assert_ne!(first[19], second[19]);
        assert_eq!(first[20], 1.0);
        assert_eq!(second[20], 1.0);
    }

    #[test]
    fn frontier_graph_marks_ready_tasks() {
        let context = SearchContext {
            tasks: vec![
                TaskNode {
                    name: "a".to_string(),
                    terms: Vec::new(),
                },
                TaskNode {
                    name: "b".to_string(),
                    terms: Vec::new(),
                },
            ],
            remaining: vec![true, true],
            predecessors: vec![HashSet::new(), HashSet::from([0])],
            state: BTreeSet::new(),
            depth: 0,
        };
        let graph = DecompositionGraph::frontier(&context);
        assert_eq!(graph.ready_indices, vec![0]);
        assert_eq!(graph.node_count, 3);
        assert_eq!(graph.adjacency[1], 1.0);
        // The global node must be readable from every task node.
        assert_eq!(graph.adjacency[2], 1.0);
        assert_eq!(graph.adjacency[2 * 3], 1.0);
    }

    #[test]
    fn guidance_kind_parses_names() {
        assert_eq!(GuidanceKind::parse("method"), Some(GuidanceKind::Method));
        assert_eq!(GuidanceKind::parse("policy"), Some(GuidanceKind::Policy));
        assert_eq!(GuidanceKind::parse("both"), Some(GuidanceKind::Both));
        assert_eq!(GuidanceKind::parse("other"), None);
        assert!(GuidanceKind::Method.uses_method());
        assert!(!GuidanceKind::Method.uses_policy());
        assert!(GuidanceKind::Both.uses_method());
        assert!(GuidanceKind::Both.uses_policy());
    }

    #[cfg(feature = "neural-base")]
    #[test]
    fn model_outputs_finite_scores() {
        use burn::module::Module;

        let device = Device::default();
        let model = DecompositionModel::new(&device);
        let context = MethodRankingContext {
            task: TaskNode {
                name: "do".to_string(),
                terms: vec!["a".to_string()],
            },
            state: BTreeSet::new(),
            depth: 1,
            max_depth: 8,
        };
        let candidates = vec![
            MethodCandidate {
                method: "m1".to_string(),
                bindings: BTreeMap::from([("?x".to_string(), "a".to_string())]),
                subtasks: 1,
                preferred: true,
                applicable: true,
                precondition_atoms: 0,
                precondition_fraction: 1.0,
            },
            MethodCandidate {
                method: "m2".to_string(),
                bindings: BTreeMap::from([("?x".to_string(), "b".to_string())]),
                subtasks: 2,
                preferred: false,
                applicable: true,
                precondition_atoms: 0,
                precondition_fraction: 1.0,
            },
        ];
        let graph = DecompositionGraph::method(&context, &candidates);
        let (method, policy, value) = model.forward(&graph.tensors(&device));
        assert_eq!(method.dims(), [graph.node_count]);
        assert_eq!(policy.dims(), [graph.node_count]);
        assert_eq!(value.dims(), [1]);
        for tensor in [method, policy, value] {
            let values = tensor.into_data().try_to_vec::<f32>().unwrap();
            assert!(values.iter().all(|value| value.is_finite()));
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.burnpack");
        model.clone().save_file(&path).unwrap();
        let context = SearchContext {
            tasks: vec![TaskNode {
                name: "a".to_string(),
                terms: Vec::new(),
            }],
            remaining: vec![true],
            predecessors: vec![HashSet::new()],
            state: BTreeSet::new(),
            depth: 0,
        };
        let (_, _, value) = model.forward(&DecompositionGraph::frontier(&context).tensors(&device));
        assert!(value.into_data().try_to_vec::<f32>().unwrap()[0].is_finite());
    }

    #[cfg(feature = "neural-base")]
    #[test]
    fn metadata_selects_trained_head() {
        use burn::module::Module;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.burnpack");
        DecompositionModel::new(&Device::default())
            .save_file(&path)
            .unwrap();
        std::fs::write(
            path.with_extension("json"),
            format!(
                "{{\"task\":\"method\",\"hidden\":{DECOMPOSITION_HIDDEN},\"feature_dim\":{DECOMPOSITION_FEATURE_DIM}}}"
            ),
        )
        .unwrap();
        let model = NeuralDecompositionModel::from_path(&path).unwrap();
        assert_eq!(model.guidance_kind(), GuidanceKind::Method);
    }
}
