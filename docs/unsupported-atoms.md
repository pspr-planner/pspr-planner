# Formula translation

The parser (`src/parsing/json.rs`) exposes the full HDDL formula language. The
solver now translates all deterministic constructors into the multi-valued
planning tasks of each leaf; this document describes that translation and the
remaining gaps. The filename is kept from the earlier gap list, when only the
conjunctive fragment was supported.

The translation lives in
[`src/solver/evaluation_tree/formula.rs`](../src/solver/evaluation_tree/formula.rs)
and is used by

- the incidence graph (`graph.rs`), which grounds the same atoms the leaf
  encoding will use, and
- the leaf encoding (`leaf.rs`), which turns the translated formulas into
  operators and conditional effects.

## Pipeline

1. **Quantifier expansion.** `Exists(vars, body)` and `ForAll(vars, body)`
   enumerate the constants and objects whose type is compatible with each
   quantified variable (`Universe::assignments`, using
   `Domain::is_subtype_of`). Nested quantifiers expand recursively. The number
   of bindings per quantifier is capped by
   `EvaluationOptions::max_groundings`.
2. **Negation normal form.** `Imply(a, b)` becomes `Or(Not(a), b)`, `Xor` is
   rewritten to "exactly one" disjunctions of its items, and `Not` is pushed to
   the literals. The result contains only literals, `And` and `Or`.
3. **Equality.** `Equals(a, b)` is evaluated against the current bindings
   (action parameters and quantifier witnesses). A true equality becomes the
   empty formula and a false one becomes unsatisfiable.
4. **DNF for preconditions.** The normal form is distributed to disjunctive
   normal form. Every clause becomes one operator variant; a clause that
   contains both polarities of an atom is dropped by normalization.
5. **Effects.** Effect formulas are compiled separately, because effects are
   updates rather than constraints:
   - `And` composes effects,
   - `Imply(condition, effect)` becomes a conditional effect; a disjunctive
     condition is split into one conditional effect per DNF clause,
   - `Or`, `Xor` and `Exists` become operator variants (one outcome per
     alternative),
   - `ForAll` expands to a conjunction of effects,
   - `Not(Atom)` becomes a delete effect.
6. **Leaf model.** Each operator variant contains one precondition clause, one
   effect outcome, and the synthetic `executed:task` marker. Only local atoms
   become leaf variables; literals on shared atoms are dropped from preconditions
   and effect conditions, while shared effect atoms are still recorded for merge
   provisioning.

DNF and alternative lists are capped at 4096 entries per formula
(`MAX_ALTERNATIVES`); exceeding the cap is an error rather than an
out-of-memory failure.

## Constructor status

| Constructor | Precondition | Effect | `unmodeled` |
| --- | --- | --- | --- |
| `Empty` | true | no effect | 0 |
| `Atom` | positive literal | add effect | 0 |
| `Not(Atom)` | negative literal | delete effect | 0 |
| `And(...)` | conjunction of the items | composition | 0 |
| `Or(...)` | DNF, one operator per clause | one operator variant per outcome | 0 |
| `Xor(...)` | exactly-one DNF | choice of one outcome | 0 |
| `Not(composite)` | negated and pushed inwards | negated and pushed inwards | 0 |
| `Imply(a, b)` | rewritten to `Or(Not(a), b)` | conditional effect | 0 |
| `Exists(vars, f)` | disjunction over witnesses | one variant per witness | 0 |
| `ForAll(vars, f)` | conjunction over witnesses | conjunction over witnesses | 0 |
| `Equals(a, b)` | evaluated during grounding | constant effect or impossible outcome | 0 |
| `Probabilistic(p, f)` | rejected | rejected | 1 per occurrence |

`unmodeled` now counts only probabilistic constructors. It is accumulated per
matching action or method schema per grounded task, so a schema reused by
several tasks is counted several times. The count is surfaced through
`TaskConstraintGraph::unmodeled` / `EvaluationTree::unmodeled`.

## Constructor details

### `Atom`, `Not(Atom)`, `Empty`, `And`

These are encoded directly. Negative literals are supported regardless of the
`:negative-preconditions` requirement; `Domain::requirements` is stored but not
consulted.

### `Or`

A precondition `Or(a, b)` becomes two clauses, `a` and `b`, hence two operator
variants in the leaf task. At runtime at least one variant is applicable when
the disjunction holds.

A disjunctive **effect** is a nondeterministic update. The translator produces
one operator variant per outcome and lets the search choose, i.e. it treats the
effect as an angelic choice. This is sound when the modelled action may pick an
outcome, and optimistic when the outcome is genuinely environment-controlled.
A faithful treatment of the latter would require conformant planning.

### `Xor`

For preconditions, `Xor(a, b, ...)` is rewritten as the disjunction of
"item `i` holds and every other item is false". For effects, an exclusive
choice is only a choice of one outcome, so it is translated like `Or`.

### `Not` of a composite formula

Negation is pushed inwards with De Morgan's laws by `negate`, so
`Not(And(a, b))` becomes `Or(Not(a), Not(b))`. Because the normal form is
quantifier-free at that point, quantifier duality is not needed.

### `Imply`

In a precondition, `Imply(a, b)` is `Or(Not(a), b)` and is handled by the DNF
step.

In an effect, the implication is a genuine conditional effect. The condition is
compiled to DNF and every effect term is copied once per condition clause, so
`(c1 ∨ c2) → e` becomes `c1 → e` and `c2 → e`. The planning task model supports
conditional effects natively (`EffectSpec` carries a condition), so no auxiliary
variables are needed.

### `Exists` and `ForAll`

Both expand over the typed universe. A quantified variable is bound to every
compatible constant or object; the type check uses the same subtype relation as
the initial task network grounding.

- `Exists` is a disjunction over witnesses in a precondition and one operator
  variant per witness in an effect.
- `ForAll` is a conjunction over witnesses in both positions.
- An empty witness set is false for `Exists` (no clauses, no outcomes) and true
  for `ForAll` (an empty conjunction, a no-op effect).

The witness enumeration is deterministic (`BTreeMap` order), so plans are
reproducible.

### `Equals`

`Equals(a, b)` compares the two names after resolving bindings. This is a
compile-time evaluation:

- true: the formula imposes no constraint (`Empty`); in an effect it is a no-op;
- false: the precondition is unsatisfiable, so the operator variant is dropped;
  in an effect the outcome is impossible.

This is distinct from the task-network constraints `Constraint::Equal` and
`Constraint::NotEqual`, which become `ConstraintNode::Equality` and are checked
by the merge solver.

## Remaining limitations

- **`Probabilistic`** cannot be translated. The leaf planning task and the best
  first search are deterministic, and the merge constraints are hard
  constraints. Leaf construction fails with an explicit error, and the
  constructor is counted in `unmodeled`. Supporting it requires an MDP solver
  and a different leaf result (policy and probability rather than a plan).
- **Exotic negated effects.** `Not(Imply(...))`, `Not(Xor(...))` and other
  negations of effects that are not `And`/`Or` of effects are rejected with
  "the negated effect ... is not supported". They have no natural update
  semantics; De Morgan covers the meaningful cases.
- **DNF blow-up.** The 4096-entry cap bounds the expansion. Large conjunctions
  of disjunctions, or quantifiers over many objects, can hit it; the error
  suggests simplifying the formula or splitting the action.
- **Angelic disjunctive effects.** As described under `Or`, an effect choice is
  made by the planner. This is not sound under adversarial nondeterminism.
- **Shared atoms.** Literals on shared atoms are dropped from leaf
  preconditions and effect conditions, exactly as before. Shared effect atoms
  are still collected so the merge can validate them. Conditions on shared
  atoms are therefore over-approximated inside a leaf.
- **Grounding budget.** Quantifier expansion is charged against
  `EvaluationOptions::max_groundings` in addition to the grounding of the
  initial task network. A quantified formula whose expansion exceeds the budget
  is an error.

## Related gaps outside the formula language

- **Problem goal.** `Problem::goal` is decoded and stored but never used by the
  evaluation tree; leaf goals are "all tasks executed".
- **Method decomposition.** Compound tasks are represented by a synthetic
  operator that requires the method precondition and marks the task executed.
  Method subtasks are not added to the task universe.
- **Numeric fluents.** `Domain::functions` is parsed but not modelled, and
  `Predicate` terms are plain objects, so numeric preconditions are not
  representable in the AST.
- **Domain axioms.** `RequirementType::DomainAxioms` is stored, but the parser
  AST has no derived-predicate field, so axioms cannot currently enter the
  solver. The planning task model itself supports layered axioms.

## Tests

The translation is covered by:

- unit tests in `formula.rs` for normalization, DNF distribution and
  conditional-effect splitting;
- end-to-end tests in `evaluation_tree/mod.rs` for `Or`, `Xor`, composite
  `Not`, `Imply`, `Exists`, `ForAll`, `Equals`, conditional effects,
  disjunctive effects and probabilistic rejection;
- the existing transport-domain tests, which pin the unchanged behaviour of
  the conjunctive fragment.
