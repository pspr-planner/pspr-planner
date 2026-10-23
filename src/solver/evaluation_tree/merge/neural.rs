//! Design 1: a heterogeneous graph-attention warm start for the merge solver.
//!
//! The graph has one node per child plan, one node per shared constraint, and
//! one global node. Child nodes carry length, solved status, incidence, and
//! atom-overlap features; constraint nodes carry the relation type, the
//! normalized ordering gap, unresolved/polarity flags, and atom/term hashes.
//! Three normalized GAT layers exchange information along a dense incidence
//! adjacency, and two heads predict the normalized start positions `q` and the
//! positive penalty weights `w`.
//!
//! The module is only compiled with `--features neural`; the default build
//! therefore has no Burn dependency at run time.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Result;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData, activation, s};

use super::super::graph::ConstraintNode;
use super::{MergeInitialization, MergeInitializer, MergeProblem};

/// Number of input features per graph node.
pub const FEATURE_DIM: usize = 32;
/// Hidden width used by the training and inference defaults.
pub const DEFAULT_HIDDEN: usize = 96;
/// Number of attention heads in the default model.
pub const DEFAULT_HEADS: usize = 4;

/// A graph representation of one merge problem.
#[derive(Debug, Clone)]
pub struct MergeGraph {
    pub child_count: usize,
    pub constraint_count: usize,
    /// Row-major `(child_count + constraint_count + 1) x FEATURE_DIM` features.
    pub features: Vec<f32>,
    /// Row-major dense adjacency, including self loops.
    pub adjacency: Vec<f32>,
    /// Normalized plan length of every child.
    pub lengths: Vec<f32>,
}

impl MergeGraph {
    /// Extract the heterogeneous graph of a merge problem.
    pub fn from_problem(problem: &MergeProblem) -> Self {
        let child_count = problem.children.len();
        let constraint_count = problem.shared_constraints.len();
        let node_count = child_count + constraint_count + 1;
        let global = node_count - 1;
        let mut features = vec![0.0; node_count * FEATURE_DIM];
        let mut adjacency = vec![0.0; node_count * node_count];
        let mut lengths = vec![0.0; child_count];

        let total_steps: usize = problem
            .children
            .iter()
            .map(|child| child.plan.len())
            .sum::<usize>()
            .max(1);
        let max_steps = problem
            .children
            .iter()
            .map(|child| child.plan.len())
            .max()
            .unwrap_or(0)
            .max(1);
        for (child_index, child) in problem.children.iter().enumerate() {
            lengths[child_index] = child.plan.len() as f32 / total_steps as f32;
        }

        let mut locations = HashMap::new();
        for (child_index, child) in problem.children.iter().enumerate() {
            for (step_index, task) in child.plan_tasks.iter().enumerate() {
                locations.insert(*task, (child_index, step_index));
            }
        }

        let mut degree = vec![0usize; child_count + constraint_count];
        for (constraint_index, constraint) in problem.shared_constraints.iter().enumerate() {
            match constraint {
                ConstraintNode::Ordering { before, after } => {
                    if let Some((before_child, _)) = locations.get(before) {
                        degree[*before_child] += 1;
                        degree[child_count + constraint_index] += 1;
                    }
                    if let Some((after_child, _)) = locations.get(after) {
                        degree[*after_child] += 1;
                        degree[child_count + constraint_index] += 1;
                    }
                }
                ConstraintNode::Atom(_) | ConstraintNode::Equality { .. } => {
                    for value in degree.iter_mut().take(child_count) {
                        *value += 1;
                    }
                    degree[child_count + constraint_index] = child_count;
                }
            }
        }

        for (child_index, child) in problem.children.iter().enumerate() {
            let base = child_index * FEATURE_DIM;
            let length = lengths[child_index];
            features[base] = 1.0;
            features[base + 1] = length;
            features[base + 2] = ((child.plan.len() as f32) + 1.0).ln() / 10.0;
            features[base + 3] = if child.solved { 1.0 } else { 0.0 };
            features[base + 4] = child.plan.len() as f32 / max_steps as f32;
            features[base + 5] = (child.final_atoms.len() as f32 / 20.0).min(1.0);
            features[base + 6] = (child.achieved.len() as f32 / 50.0).min(1.0);
            features[base + 7] = degree[child_index] as f32 / (constraint_count + 1) as f32;
            if let Some(first) = child.plan.first() {
                features[base + 8] = hash_bucket(first);
            }
            if let Some(last) = child.plan.last() {
                features[base + 9] = hash_bucket(last);
            }
            for (slot, atom) in child.final_atoms.iter().take(4).enumerate() {
                features[base + 10 + slot] = hash_bucket(atom);
            }
            for (slot, atom) in child.achieved.iter().take(4).enumerate() {
                features[base + 14 + slot] = hash_bucket(atom);
            }
        }

        for (constraint_index, constraint) in problem.shared_constraints.iter().enumerate() {
            let node = child_count + constraint_index;
            let base = node * FEATURE_DIM;
            features[base] = 0.0;
            features[base + 1] = 1.0;
            features[base + 2] = 0.0;
            features[base + 14] =
                degree[child_count + constraint_index] as f32 / (child_count + 1).max(1) as f32;
            match constraint {
                ConstraintNode::Ordering { before, after } => {
                    features[base + 3] = 1.0;
                    let before = locations.get(before).copied();
                    let after = locations.get(after).copied();
                    if let (Some((before_child, before_step)), Some((after_child, after_step))) =
                        (before, after)
                    {
                        let denominator = problem.children[before_child].plan.len().max(1) as f32;
                        features[base + 6] = (after_step as f32 - before_step as f32) / denominator;
                        features[base + 7] = 1.0;
                        if before_child != after_child {
                            add_edge(&mut adjacency, node_count, before_child, after_child);
                            add_edge(&mut adjacency, node_count, after_child, before_child);
                        }
                        add_edge(&mut adjacency, node_count, node, before_child);
                        add_edge(&mut adjacency, node_count, before_child, node);
                        add_edge(&mut adjacency, node_count, node, after_child);
                        add_edge(&mut adjacency, node_count, after_child, node);
                    }
                }
                ConstraintNode::Atom(atom) => {
                    features[base + 4] = 1.0;
                    let text = atom.to_string();
                    for slot in 0..4 {
                        features[base + 9 + slot] = hash_bucket_at(&text, slot);
                    }
                    for (child_index, child) in problem.children.iter().enumerate() {
                        if child.final_atoms.contains(&text)
                            || child.achieved.contains(&text)
                            || child
                                .final_atoms
                                .iter()
                                .any(|item| item.contains(&atom.predicate))
                            || child
                                .achieved
                                .iter()
                                .any(|item| item.contains(&atom.predicate))
                        {
                            add_edge(&mut adjacency, node_count, node, child_index);
                            add_edge(&mut adjacency, node_count, child_index, node);
                        }
                    }
                }
                ConstraintNode::Equality { left, right, equal } => {
                    features[base + 5] = 1.0;
                    features[base + 8] = if *equal { 1.0 } else { 0.0 };
                    for slot in 0..4 {
                        features[base + 9 + slot] =
                            hash_bucket_at(&format!("{left}|{right}"), slot);
                    }
                    for child in 0..child_count {
                        add_edge(&mut adjacency, node_count, node, child);
                        add_edge(&mut adjacency, node_count, child, node);
                    }
                }
            }
        }

        let base = global * FEATURE_DIM;
        features[base + 2] = 1.0;
        features[base + 15] = child_count as f32 / (child_count + 1) as f32;
        features[base + 16] = constraint_count as f32 / (child_count + constraint_count + 1) as f32;
        features[base + 17] = (total_steps as f32 / 100.0).min(1.0);
        features[base + 18] = total_steps as f32 / child_count.max(1) as f32 / max_steps as f32;
        features[base + 19] = if child_count == 0 {
            0.0
        } else {
            problem.children.iter().filter(|child| child.solved).count() as f32 / child_count as f32
        };

        for node in 0..node_count {
            add_edge(&mut adjacency, node_count, node, node);
            add_edge(&mut adjacency, node_count, node, global);
            add_edge(&mut adjacency, node_count, global, node);
        }

        MergeGraph {
            child_count,
            constraint_count,
            features,
            adjacency,
            lengths,
        }
    }

    pub fn node_count(&self) -> usize {
        self.child_count + self.constraint_count + 1
    }

    /// Materialize the graph as Burn tensors on `device`.
    pub fn tensors(&self, device: &Device) -> GraphTensors {
        let node_count = self.node_count();
        GraphTensors {
            features: Tensor::<2>::from_data(
                TensorData::new(self.features.clone(), [node_count, FEATURE_DIM]),
                device,
            ),
            adjacency: Tensor::<2>::from_data(
                TensorData::new(self.adjacency.clone(), [node_count, node_count]),
                device,
            ),
            lengths: Tensor::<1>::from_data(
                TensorData::new(self.lengths.clone(), [self.child_count]),
                device,
            ),
            child_count: self.child_count,
            constraint_count: self.constraint_count,
        }
    }
}

/// Burn tensors of one [`MergeGraph`].
#[derive(Debug)]
pub struct GraphTensors {
    pub features: Tensor<2>,
    pub adjacency: Tensor<2>,
    pub lengths: Tensor<1>,
    pub child_count: usize,
    pub constraint_count: usize,
}

/// One teacher-labelled merge problem.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub graph: MergeGraph,
    pub positions: Vec<f64>,
    pub weights: Vec<f64>,
}

/// One attention head of a graph-attention layer.
#[derive(Module, Debug)]
struct GatHead {
    query: Linear,
    key: Linear,
    value: Linear,
}

impl GatHead {
    fn new(hidden: usize, head_dim: usize, device: &Device) -> Self {
        Self {
            query: LinearConfig::new(hidden, head_dim).init(device),
            key: LinearConfig::new(hidden, head_dim).init(device),
            value: LinearConfig::new(hidden, head_dim).init(device),
        }
    }

    fn forward(&self, input: Tensor<2>, bias: &Tensor<2>, head_dim: usize) -> Tensor<2> {
        let query = self.query.forward(input.clone());
        let key = self.key.forward(input.clone());
        let value = self.value.forward(input.clone());
        let scores = query
            .matmul(key.transpose())
            .div_scalar((head_dim as f32).sqrt())
            + bias.clone();
        let attention = activation::softmax(scores, 1);
        attention.matmul(value)
    }
}

/// Multi-head graph-attention layer used by [`GatModel`].
#[derive(Module, Debug)]
struct GatLayer {
    heads: Vec<GatHead>,
    output: Linear,
    norm: LayerNorm,
}

impl GatLayer {
    fn new(hidden: usize, heads: usize, device: &Device) -> Self {
        let heads = heads.max(1);
        let head_dim = hidden.div_ceil(heads);
        Self {
            heads: (0..heads)
                .map(|_| GatHead::new(hidden, head_dim, device))
                .collect(),
            output: LinearConfig::new(head_dim * heads, hidden).init(device),
            norm: LayerNormConfig::new(hidden).init(device),
        }
    }

    fn forward(&self, input: Tensor<2>, adjacency: Tensor<2>) -> Tensor<2> {
        let head_dim = input.dims()[1].div_ceil(self.heads.len().max(1));
        let bias = adjacency.sub_scalar(1.0).mul_scalar(1.0e9);
        let contexts: Vec<Tensor<2>> = self
            .heads
            .iter()
            .map(|head| head.forward(input.clone(), &bias, head_dim))
            .collect();
        let context = Tensor::cat(contexts, 1);
        let update = self.output.forward(context);
        activation::leaky_relu(self.norm.forward(input + update), 0.2)
    }
}

/// Design 1 heterogeneous GAT.
#[derive(Module, Debug)]
pub struct GatModel {
    input: Linear,
    layer_one: GatLayer,
    layer_two: GatLayer,
    layer_three: GatLayer,
    position_head: Linear,
    weight_head: Linear,
}

impl GatModel {
    pub fn new(input_dim: usize, hidden: usize, heads: usize, device: &Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, hidden).init(device),
            layer_one: GatLayer::new(hidden, heads, device),
            layer_two: GatLayer::new(hidden, heads, device),
            layer_three: GatLayer::new(hidden, heads, device),
            position_head: LinearConfig::new(hidden, 1).init(device),
            weight_head: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// Predict normalized start positions `[child_count]` and positive
    /// weights `[constraint_count]`.
    pub fn forward(&self, graph: &GraphTensors) -> (Tensor<1>, Tensor<1>) {
        let hidden = self.input.forward(graph.features.clone());
        let hidden = self.layer_one.forward(hidden, graph.adjacency.clone());
        let hidden = self.layer_two.forward(hidden, graph.adjacency.clone());
        let hidden = self.layer_three.forward(hidden, graph.adjacency.clone());

        let position_logits = self.position_head.forward(hidden.clone());
        let raw_positions = position_logits
            .slice(s![0..graph.child_count, 0..1])
            .reshape([graph.child_count]);
        let upper = graph.lengths.clone().mul_scalar(-1.0).add_scalar(1.0);
        let positions = activation::sigmoid(raw_positions) * upper;

        let weight_logits = self.weight_head.forward(hidden);
        let raw_weights = weight_logits
            .slice(s![
                graph.child_count..graph.child_count + graph.constraint_count,
                0..1
            ])
            .reshape([graph.constraint_count]);
        let weights = activation::softplus(raw_weights, 1.0) + 1.0e-3;
        (positions, weights)
    }
}

/// Inference wrapper that loads a saved [`GatModel`] and implements
/// [`MergeInitializer`].
#[derive(Debug)]
pub struct NeuralMergeInitializer {
    model: GatModel,
    device: Device,
}

impl NeuralMergeInitializer {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::read_to_string(path.with_extension("json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
        let hidden = metadata
            .as_ref()
            .and_then(|value| value.get("hidden").and_then(|value| value.as_u64()))
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_HIDDEN);
        let heads = metadata
            .as_ref()
            .and_then(|value| value.get("heads").and_then(|value| value.as_u64()))
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_HEADS);
        Self::from_path_with_config(path, hidden, heads)
    }

    pub fn from_path_with_hidden(path: impl AsRef<Path>, hidden: usize) -> Result<Self> {
        Self::from_path_with_config(path, hidden, DEFAULT_HEADS)
    }

    pub fn from_path_with_config(
        path: impl AsRef<Path>,
        hidden: usize,
        heads: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("merge initializer model does not exist: {}", path.display());
        }
        let device = Device::default();
        let model = GatModel::new(FEATURE_DIM, hidden, heads, &device).load_file(path);
        Ok(Self { model, device })
    }
}

impl MergeInitializer for NeuralMergeInitializer {
    fn initialize(&self, problem: &MergeProblem) -> Option<MergeInitialization> {
        if problem.children.is_empty() || problem.shared_constraints.is_empty() {
            return None;
        }
        let graph = MergeGraph::from_problem(problem);
        let tensors = graph.tensors(&self.device);
        let (positions, weights) = self.model.forward(&tensors);
        let positions = positions.into_data().try_to_vec::<f32>().ok()?;
        let weights = weights.into_data().try_to_vec::<f32>().ok()?;
        if positions.len() != graph.child_count
            || weights.len() != graph.constraint_count
            || positions.iter().any(|value| !value.is_finite())
            || weights.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        Some(MergeInitialization {
            positions: positions
                .into_iter()
                .map(|value| f64::from(value).clamp(0.0, 1.0))
                .collect(),
            weights: weights
                .into_iter()
                .map(|value| f64::from(value).max(1.0e-6))
                .collect(),
            space: None,
        })
    }
}

fn add_edge(adjacency: &mut [f32], node_count: usize, from: usize, to: usize) {
    if from < node_count && to < node_count {
        adjacency[from * node_count + to] = 1.0;
    }
}

fn hash_bucket(value: &str) -> f32 {
    hash_bucket_at(value, 0)
}

fn hash_bucket_at(value: &str, slot: usize) -> f32 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= slot as u64;
    (hash % 1024) as f32 / 1024.0
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::solver::evaluation_tree::ConstraintNode;
    use crate::solver::evaluation_tree::merge::PartialSolution;

    fn problem() -> MergeProblem {
        let child =
            |label: &str, steps: &[&str], tasks: &[usize], final_atom: &str| PartialSolution {
                label: label.to_string(),
                plan: steps.iter().map(|step| step.to_string()).collect(),
                plan_tasks: tasks.to_vec(),
                final_atoms: BTreeSet::from([final_atom.to_string()]),
                achieved: BTreeSet::from([final_atom.to_string()]),
                solved: true,
            };
        MergeProblem {
            node: 0,
            shared_constraints: vec![ConstraintNode::Ordering {
                before: 0,
                after: 1,
            }],
            children: vec![
                child("a", &["one", "two"], &[0, 1], "at(a)"),
                child("b", &["three"], &[2], "at(b)"),
            ],
            initial_atoms: BTreeSet::new(),
        }
    }

    #[test]
    fn graph_has_child_constraint_and_global_nodes() {
        let graph = MergeGraph::from_problem(&problem());
        assert_eq!(graph.child_count, 2);
        assert_eq!(graph.constraint_count, 1);
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.features.len(), 4 * FEATURE_DIM);
        assert_eq!(graph.adjacency.len(), 16);
        assert_eq!(graph.lengths, vec![2.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(graph.adjacency[0], 1.0);
        assert_eq!(graph.adjacency[2], 1.0);
        assert_eq!(graph.adjacency[8], 1.0);
    }

    #[test]
    fn gat_outputs_are_bounded_and_positive() {
        let graph = MergeGraph::from_problem(&problem()).tensors(&Device::default());
        let model = GatModel::new(FEATURE_DIM, 16, DEFAULT_HEADS, &Device::default());
        let (positions, weights) = model.forward(&graph);
        let positions = positions.into_data().try_to_vec::<f32>().unwrap();
        let weights = weights.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(weights.len(), 1);
        assert!(positions.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!(weights.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn a_missing_model_is_an_error_not_a_panic() {
        let result = NeuralMergeInitializer::from_path("/definitely/not/a/model.burnpack");
        assert!(result.is_err());
    }
}
