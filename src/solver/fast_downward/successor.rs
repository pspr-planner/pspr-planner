//! Successor generator (Definition 8, Section 5.3.1 of Helmert, JAIR 2006).

use super::{State, Task};

#[derive(Debug, Clone, PartialEq)]
enum GeneratorNode {
    Selector {
        variable: usize,
        children: Vec<Option<Box<GeneratorNode>>>,
        dont_care: Option<Box<GeneratorNode>>,
    },
    Generator(Vec<usize>),
}

/// A decision tree over operator preconditions that returns all applicable
/// operators for a state.
#[derive(Debug, Clone, PartialEq)]
pub struct SuccessorGenerator {
    root: GeneratorNode,
}

impl SuccessorGenerator {
    pub(super) fn build(task: &Task) -> Self {
        let operators: Vec<usize> = (0..task.operators.len()).collect();
        let variables: Vec<usize> = (0..task.variables.len()).collect();
        SuccessorGenerator {
            root: build_generator(task, operators, &variables),
        }
    }

    /// All operators whose preconditions hold in the extended state `state`.
    pub fn applicable(&self, state: &State) -> Vec<usize> {
        let mut result = Vec::new();
        collect_applicable(&self.root, state, &mut result);
        result
    }
}

fn build_generator(task: &Task, operators: Vec<usize>, variables: &[usize]) -> GeneratorNode {
    if operators.is_empty() || variables.is_empty() {
        return GeneratorNode::Generator(operators);
    }
    let variable = variables[0];
    let remainder = &variables[1..];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); task.variables[variable].size];
    let mut dont_care = Vec::new();
    let mut any_specific = false;
    for &operator in &operators {
        match task.operators[operator]
            .precondition
            .iter()
            .find(|&&(index, _)| index == variable)
        {
            Some(&(_, value)) => {
                children[value].push(operator);
                any_specific = true;
            }
            None => dont_care.push(operator),
        }
    }
    if !any_specific {
        // This variable does not constrain the branch; skip it.
        return build_generator(task, operators, remainder);
    }
    GeneratorNode::Selector {
        variable,
        children: children
            .into_iter()
            .map(|subset| {
                if subset.is_empty() {
                    None
                } else {
                    Some(Box::new(build_generator(task, subset, remainder)))
                }
            })
            .collect(),
        dont_care: if dont_care.is_empty() {
            None
        } else {
            Some(Box::new(build_generator(task, dont_care, remainder)))
        },
    }
}

fn collect_applicable(node: &GeneratorNode, state: &State, result: &mut Vec<usize>) {
    match node {
        GeneratorNode::Generator(operators) => result.extend_from_slice(operators),
        GeneratorNode::Selector {
            variable,
            children,
            dont_care,
        } => {
            if let Some(child) = &children[state[*variable]] {
                collect_applicable(child, state, result);
            }
            if let Some(child) = dont_care {
                collect_applicable(child, state, result);
            }
        }
    }
}
