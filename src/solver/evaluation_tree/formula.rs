//! Translation of parser formulas into the conjunctive fragment of the leaf
//! planning tasks.
//!
//! The translation is a small compiler:
//!
//! 1. quantifiers are expanded over the typed constants and objects of the
//!    problem;
//! 2. `Imply`, `Xor` and composite `Not` are rewritten, so the result is in
//!    negation normal form over literals;
//! 3. `Equals` is evaluated against the current bindings;
//! 4. preconditions are converted to disjunctive normal form (DNF), one clause
//!    per operator variant;
//! 5. effects are compiled into conditional effects; disjunctive and
//!    existential effects become operator variants, an angelic choice made by
//!    the search.
//!
//! `Probabilistic` is the only constructor that cannot be translated: the leaf
//! planning tasks and the search are deterministic.

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};

use crate::parsing::json::{Formula, Symbol};
use crate::solver::domain::Domain;
use crate::solver::problem::Problem;

use super::graph::{Atom, ground_atom};

/// Upper bound on the number of DNF clauses or effect alternatives produced for
/// one formula. Keeps pathological formulas from exhausting memory.
const MAX_ALTERNATIVES: usize = 4096;

/// A literal of a compiled formula: a ground atom and its polarity.
pub(super) type Literal = (Atom, bool);

/// A conjunction of literals.
pub(super) type Clause = Vec<Literal>;

/// A conditional effect `condition -> atom := value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EffectTerm {
    pub(super) condition: Clause,
    pub(super) atom: Atom,
    pub(super) value: bool,
}

/// One deterministic outcome of an effect formula.
pub(super) type EffectAlternative = Vec<EffectTerm>;

/// The constants and objects a quantifier can range over.
pub(super) struct Universe<'a> {
    domain: &'a Domain,
    symbols: BTreeMap<String, Option<String>>,
}

impl<'a> Universe<'a> {
    pub(super) fn new(domain: &'a Domain, problem: &'a Problem) -> Self {
        let mut symbols = BTreeMap::new();
        for symbol in domain.constants.iter().chain(problem.objects.iter()) {
            symbols
                .entry(symbol.name.clone())
                .or_insert_with(|| symbol.symbol_type.clone());
        }
        Universe { domain, symbols }
    }

    /// Names compatible with the declared type of `variable`.
    pub(super) fn candidates(&self, variable: &Symbol) -> Vec<String> {
        self.symbols
            .keys()
            .filter(|name| self.compatible(variable, name))
            .cloned()
            .collect()
    }

    /// Whether `name` is an object or constant compatible with the declared
    /// type of `variable`.
    pub(super) fn compatible(&self, variable: &Symbol, name: &str) -> bool {
        let Some(symbol_type) = self.symbols.get(name) else {
            return false;
        };
        match (&variable.symbol_type, symbol_type) {
            (Some(required), Some(actual)) => self.domain.is_subtype_of(actual, required),
            (Some(_), None) => true,
            (None, _) => true,
        }
    }

    /// Every assignment of the quantified `variables`, at most `budget`.
    pub(super) fn assignments(
        &self,
        variables: &[Symbol],
        budget: usize,
    ) -> Result<Vec<BTreeMap<String, String>>> {
        let mut assignments: Vec<BTreeMap<String, String>> = vec![BTreeMap::new()];
        for variable in variables {
            let candidates = self.candidates(variable);
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            let mut extended = Vec::new();
            for assignment in &assignments {
                for candidate in &candidates {
                    let mut copy = assignment.clone();
                    copy.insert(variable.name.clone(), candidate.clone());
                    extended.push(copy);
                    ensure!(
                        extended.len() <= budget,
                        "the quantified formula expands to more than {budget} bindings; \
                         raise EvaluationOptions::max_groundings or simplify the formula"
                    );
                }
            }
            assignments = extended;
        }
        Ok(assignments)
    }

    /// A prefix of the quantified assignments, capped at `budget`.
    ///
    /// This never errors on combinatorial blow-up and is used by bounded
    /// approximations (for example incidence extraction) where partial
    /// exploration is preferable to aborting the whole solve.
    pub(super) fn assignments_capped(
        &self,
        variables: &[Symbol],
        budget: usize,
    ) -> Vec<BTreeMap<String, String>> {
        if budget == 0 {
            return Vec::new();
        }
        let mut assignments: Vec<BTreeMap<String, String>> = vec![BTreeMap::new()];
        for variable in variables {
            let candidates = self.candidates(variable);
            if candidates.is_empty() {
                return Vec::new();
            }
            let mut extended = Vec::new();
            for assignment in &assignments {
                for candidate in &candidates {
                    if extended.len() >= budget {
                        break;
                    }
                    let mut copy = assignment.clone();
                    copy.insert(variable.name.clone(), candidate.clone());
                    extended.push(copy);
                }
                if extended.len() >= budget {
                    break;
                }
            }
            assignments = extended;
        }
        assignments
    }
}

pub(super) fn is_quantifier_overflow(error: &anyhow::Error) -> bool {
    error
        .to_string()
        .contains("the quantified formula expands to more than")
}

/// A quantifier-free formula in negation normal form.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Normal {
    True,
    False,
    Literal(Atom, bool),
    And(Vec<Normal>),
    Or(Vec<Normal>),
}

fn negate(formula: &Normal) -> Normal {
    match formula {
        Normal::True => Normal::False,
        Normal::False => Normal::True,
        Normal::Literal(atom, positive) => Normal::Literal(atom.clone(), !positive),
        Normal::And(items) => Normal::Or(items.iter().map(negate).collect()),
        Normal::Or(items) => Normal::And(items.iter().map(negate).collect()),
    }
}

fn compile(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<Normal> {
    let Some(formula) = formula else {
        return Ok(Normal::True);
    };
    match formula {
        Formula::Empty => Ok(Normal::True),
        Formula::Atom(predicate) => Ok(Normal::Literal(ground_atom(predicate, bindings), true)),
        Formula::Not(inner) => Ok(negate(&compile(Some(inner), bindings, universe, budget)?)),
        Formula::And(items) => Ok(Normal::And(
            items
                .iter()
                .map(|item| compile(Some(item), bindings, universe, budget))
                .collect::<Result<_>>()?,
        )),
        Formula::Or(items) => Ok(Normal::Or(
            items
                .iter()
                .map(|item| compile(Some(item), bindings, universe, budget))
                .collect::<Result<_>>()?,
        )),
        Formula::Xor(items) => {
            let compiled = items
                .iter()
                .map(|item| compile(Some(item), bindings, universe, budget))
                .collect::<Result<Vec<_>>>()?;
            let mut alternatives = Vec::with_capacity(compiled.len());
            for (index, item) in compiled.iter().enumerate() {
                let mut conjunction = vec![item.clone()];
                for (other, alternative) in compiled.iter().enumerate() {
                    if other != index {
                        conjunction.push(negate(alternative));
                    }
                }
                alternatives.push(Normal::And(conjunction));
            }
            Ok(Normal::Or(alternatives))
        }
        Formula::Imply(left, right) => Ok(Normal::Or(vec![
            negate(&compile(Some(left), bindings, universe, budget)?),
            compile(Some(right), bindings, universe, budget)?,
        ])),
        Formula::Exists(variables, body) => {
            let mut alternatives = Vec::new();
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                alternatives.push(compile(Some(body), &local, universe, budget)?);
            }
            Ok(Normal::Or(alternatives))
        }
        Formula::ForAll(variables, body) => {
            let mut conjunction = Vec::new();
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                conjunction.push(compile(Some(body), &local, universe, budget)?);
            }
            Ok(Normal::And(conjunction))
        }
        Formula::Probabilistic(_, _) => bail!(
            "probabilistic formulas are not supported: the leaf planning tasks and the search are deterministic"
        ),
        Formula::Equals(left, right) => {
            let left = bindings.get(left).unwrap_or(left);
            let right = bindings.get(right).unwrap_or(right);
            Ok(if left == right {
                Normal::True
            } else {
                Normal::False
            })
        }
    }
}

/// Drop contradictory literals and duplicate clauses.
fn normalize(clauses: Vec<Clause>) -> Vec<Clause> {
    let mut normalized: Vec<Clause> = Vec::new();
    for clause in clauses {
        let mut literals: Vec<Literal> = Vec::new();
        let mut consistent = true;
        for literal in clause {
            if literals
                .iter()
                .any(|existing| existing.0 == literal.0 && existing.1 != literal.1)
            {
                consistent = false;
                break;
            }
            if !literals.contains(&literal) {
                literals.push(literal);
            }
        }
        if consistent && !normalized.contains(&literals) {
            normalized.push(literals);
        }
    }
    normalized
}

/// Disjunctive normal form of a compiled precondition.
fn clauses(normal: &Normal) -> Result<Vec<Clause>> {
    let clauses = match normal {
        Normal::True => vec![Clause::new()],
        Normal::False => Vec::new(),
        Normal::Literal(atom, positive) => vec![vec![(atom.clone(), *positive)]],
        Normal::Or(items) => {
            let mut alternatives = Vec::new();
            for item in items {
                alternatives.extend(clauses(item)?);
                ensure!(
                    alternatives.len() <= MAX_ALTERNATIVES,
                    "the formula expands to more than {MAX_ALTERNATIVES} alternatives; \
                     simplify it or split the action"
                );
            }
            alternatives
        }
        Normal::And(items) => {
            let mut product: Vec<Clause> = vec![Clause::new()];
            for item in items {
                let item_clauses = clauses(item)?;
                if item_clauses.is_empty() {
                    return Ok(Vec::new());
                }
                let mut next = Vec::new();
                for left in &product {
                    for right in &item_clauses {
                        let mut clause = left.clone();
                        clause.extend(right.iter().cloned());
                        next.push(clause);
                        ensure!(
                            next.len() <= MAX_ALTERNATIVES,
                            "the formula expands to more than {MAX_ALTERNATIVES} clauses; \
                             simplify it or split the action"
                        );
                    }
                }
                product = next;
            }
            product
        }
    };
    Ok(normalize(clauses))
}

/// Convert a precondition to DNF.
///
/// `None` is the empty conjunction (one empty clause); an unsatisfiable
/// precondition yields no clauses and the caller must drop the operator.
pub(super) fn precondition_clauses(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<Vec<Clause>> {
    clauses(&compile(formula, bindings, universe, budget)?)
}

/// An effect formula before it is turned into concrete effect terms.
#[derive(Debug, Clone)]
enum Effect {
    /// No effect.
    None,
    /// An effect formula that can never be applied.
    False,
    Literal(Atom, bool),
    And(Vec<Effect>),
    /// A nondeterministic choice between alternatives.
    Or(Vec<Effect>),
    /// `condition -> effect`; the condition is a compiled formula.
    Conditional(Normal, Box<Effect>),
}

fn negated(item: &Formula) -> Formula {
    Formula::Not(Box::new(item.clone()))
}

fn evaluate_equality(left: &str, right: &str, bindings: &BTreeMap<String, String>) -> bool {
    let left = bindings.get(left).map(String::as_str).unwrap_or(left);
    let right = bindings.get(right).map(String::as_str).unwrap_or(right);
    left == right
}

fn compile_effect(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<Effect> {
    let Some(formula) = formula else {
        return Ok(Effect::None);
    };
    match formula {
        Formula::Empty => Ok(Effect::None),
        Formula::Atom(predicate) => Ok(Effect::Literal(ground_atom(predicate, bindings), true)),
        Formula::Not(inner) => match inner.as_ref() {
            Formula::Atom(predicate) => {
                Ok(Effect::Literal(ground_atom(predicate, bindings), false))
            }
            Formula::Empty => Ok(Effect::False),
            Formula::Not(inner) => compile_effect(Some(inner), bindings, universe, budget),
            Formula::And(items) => Ok(Effect::Or(
                items
                    .iter()
                    .map(|item| compile_effect(Some(&negated(item)), bindings, universe, budget))
                    .collect::<Result<_>>()?,
            )),
            Formula::Or(items) => Ok(Effect::And(
                items
                    .iter()
                    .map(|item| compile_effect(Some(&negated(item)), bindings, universe, budget))
                    .collect::<Result<_>>()?,
            )),
            Formula::Equals(left, right) => Ok(if evaluate_equality(left, right, bindings) {
                Effect::False
            } else {
                Effect::None
            }),
            other => bail!("the negated effect `{other:?}` is not supported"),
        },
        Formula::And(items) => Ok(Effect::And(
            items
                .iter()
                .map(|item| compile_effect(Some(item), bindings, universe, budget))
                .collect::<Result<_>>()?,
        )),
        // An exclusive choice is still a choice of one outcome for the planner.
        Formula::Or(items) | Formula::Xor(items) => Ok(Effect::Or(
            items
                .iter()
                .map(|item| compile_effect(Some(item), bindings, universe, budget))
                .collect::<Result<_>>()?,
        )),
        Formula::Imply(condition, effect) => Ok(Effect::Conditional(
            compile(Some(condition), bindings, universe, budget)?,
            Box::new(compile_effect(Some(effect), bindings, universe, budget)?),
        )),
        Formula::Exists(variables, body) => {
            let mut alternatives = Vec::new();
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                alternatives.push(compile_effect(Some(body), &local, universe, budget)?);
            }
            Ok(Effect::Or(alternatives))
        }
        Formula::ForAll(variables, body) => {
            let mut conjunction = Vec::new();
            for assignment in universe.assignments(variables, budget)? {
                let mut local = bindings.clone();
                local.extend(assignment);
                conjunction.push(compile_effect(Some(body), &local, universe, budget)?);
            }
            Ok(Effect::And(conjunction))
        }
        Formula::Probabilistic(_, _) => bail!(
            "probabilistic effects are not supported: the leaf planning tasks and the search are deterministic"
        ),
        Formula::Equals(left, right) => Ok(if evaluate_equality(left, right, bindings) {
            Effect::None
        } else {
            Effect::False
        }),
    }
}

/// Turn an effect formula into alternative sets of conditional effect terms.
///
/// Each returned alternative is one deterministic outcome; the caller turns
/// every alternative into an operator variant. An empty result means that no
/// outcome exists and the operator must be dropped.
fn effect_alternatives(effect: &Effect) -> Result<Vec<EffectAlternative>> {
    match effect {
        Effect::None => Ok(vec![EffectAlternative::new()]),
        Effect::False => Ok(Vec::new()),
        Effect::Literal(atom, value) => Ok(vec![vec![EffectTerm {
            condition: Clause::new(),
            atom: atom.clone(),
            value: *value,
        }]]),
        Effect::Or(items) => {
            let mut alternatives = Vec::new();
            for item in items {
                alternatives.extend(effect_alternatives(item)?);
                ensure!(
                    alternatives.len() <= MAX_ALTERNATIVES,
                    "the effect expands to more than {MAX_ALTERNATIVES} alternatives; \
                     simplify it or split the action"
                );
            }
            Ok(alternatives)
        }
        Effect::And(items) => {
            let mut product: Vec<EffectAlternative> = vec![EffectAlternative::new()];
            for item in items {
                let item_alternatives = effect_alternatives(item)?;
                if item_alternatives.is_empty() {
                    return Ok(Vec::new());
                }
                let mut next = Vec::new();
                for left in &product {
                    for right in &item_alternatives {
                        let mut alternative = left.clone();
                        alternative.extend(right.iter().cloned());
                        next.push(alternative);
                        ensure!(
                            next.len() <= MAX_ALTERNATIVES,
                            "the effect expands to more than {MAX_ALTERNATIVES} alternatives; \
                             simplify it or split the action"
                        );
                    }
                }
                product = next;
            }
            Ok(product)
        }
        Effect::Conditional(condition, inner) => {
            let condition_clauses = clauses(condition)?;
            if condition_clauses.is_empty() {
                // An unsatisfiable condition makes the implication vacuous.
                return Ok(vec![EffectAlternative::new()]);
            }
            let mut alternatives = Vec::new();
            for alternative in effect_alternatives(inner)? {
                let mut terms = Vec::new();
                for term in alternative {
                    for clause in &condition_clauses {
                        let mut condition = term.condition.clone();
                        condition.extend(clause.iter().cloned());
                        terms.push(EffectTerm {
                            condition,
                            atom: term.atom.clone(),
                            value: term.value,
                        });
                    }
                }
                alternatives.push(terms);
            }
            Ok(alternatives)
        }
    }
}

/// Compile an effect formula into operator alternatives.
pub(super) fn effect_alternatives_of(
    formula: Option<&Formula>,
    bindings: &BTreeMap<String, String>,
    universe: &Universe<'_>,
    budget: usize,
) -> Result<Vec<EffectAlternative>> {
    effect_alternatives(&compile_effect(formula, bindings, universe, budget)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(name: &str, terms: &[&str]) -> Atom {
        Atom::new(name, terms.iter().map(|term| term.to_string()).collect())
    }

    #[test]
    fn normalize_drops_contradictory_and_duplicate_clauses() {
        let a = atom("a", &[]);
        let b = atom("b", &[]);
        let normalized = normalize(vec![
            vec![(a.clone(), true), (a.clone(), false)],
            vec![(b.clone(), true)],
            vec![(b.clone(), true)],
        ]);
        assert_eq!(normalized, vec![vec![(b, true)]]);
    }

    #[test]
    fn dnf_distributes_conjunctions_over_disjunctions() {
        let a = atom("a", &[]);
        let b = atom("b", &[]);
        let c = atom("c", &[]);
        let normal = Normal::And(vec![
            Normal::Or(vec![
                Normal::Literal(a.clone(), true),
                Normal::Literal(b.clone(), true),
            ]),
            Normal::Literal(c.clone(), true),
        ]);
        let clauses = clauses(&normal).unwrap();
        assert_eq!(
            clauses,
            vec![
                vec![(a, true), (c.clone(), true)],
                vec![(b, true), (c, true)]
            ]
        );
    }

    #[test]
    fn dnf_of_false_is_empty() {
        assert!(clauses(&Normal::False).unwrap().is_empty());
    }

    #[test]
    fn conditional_effects_duplicate_terms_per_condition_clause() {
        let a = atom("a", &[]);
        let b = atom("b", &[]);
        let effect = Effect::Conditional(
            Normal::Or(vec![
                Normal::Literal(a.clone(), true),
                Normal::Literal(b.clone(), true),
            ]),
            Box::new(Effect::Literal(atom("c", &[]), true)),
        );
        let alternatives = effect_alternatives(&effect).unwrap();
        assert_eq!(alternatives.len(), 1);
        let conditions: Vec<Vec<Literal>> = alternatives[0]
            .iter()
            .map(|term| term.condition.clone())
            .collect();
        assert_eq!(conditions, vec![vec![(a, true)], vec![(b, true)]]);
    }
}
