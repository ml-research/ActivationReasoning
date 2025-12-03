from dataclasses import dataclass
from typing import List, Tuple, Union, Set


# Define a base class for all logical components.
# This will be the parent for both the operators and the generated components.
class LogicalComponent:
    """Base class for all logical components."""

    pass


# --- Internal Dataclasses for holding the final logical structure ---


@dataclass(frozen=True)
class _NotComponent(LogicalComponent):
    keyword: str
    component: LogicalComponent | str

    def __repr__(self):
        return f"¬{self.component}"


@dataclass(frozen=True)
class _AndComponent(LogicalComponent):
    keyword: str
    component1: LogicalComponent | str
    component2: LogicalComponent | str

    def __repr__(self):
        return f"({self.component1} ∧ {self.component2})"


@dataclass(frozen=True)
class _OrComponent(LogicalComponent):
    keyword: str
    component1: LogicalComponent | str
    component2: LogicalComponent | str

    def __repr__(self):
        return f"({self.component1} ∨ {self.component2})"


@dataclass(frozen=True)
class _ImpliesComponent(LogicalComponent):
    keyword: str
    component1: LogicalComponent | str
    component2: LogicalComponent | str

    def __repr__(self):
        return f"({self.component1} → {self.component2})"


@dataclass(frozen=True)
class _XorComponent(
    LogicalComponent
):  # Kept for potential future use, though not directly used by parser
    keyword: str
    component1: LogicalComponent | str
    component2: LogicalComponent | str

    def __repr__(self):
        return f"({self.component1} ⊕ {self.component2})"


# --- Operator Classes that can be initialized and then called ---


class Not:
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(
        self, component: LogicalComponent | str
    ) -> Union[LogicalComponent, str]:
        # If the component is ALREADY a NotComponent, just return its inner component.
        # This resolves ¬(¬A) to A during the parsing step.
        if isinstance(component, _NotComponent):
            return component.component

        return _NotComponent(self.keyword, component)


class And:
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(
        self, c1: LogicalComponent | str, c2: LogicalComponent | str
    ) -> _AndComponent:
        return _AndComponent(self.keyword, c1, c2)


class Or:
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(
        self, c1: LogicalComponent | str, c2: LogicalComponent | str
    ) -> _OrComponent:
        return _OrComponent(self.keyword, c1, c2)


class Implies:
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(
        self, c1: LogicalComponent | str, c2: LogicalComponent | str
    ) -> _ImpliesComponent:
        return _ImpliesComponent(self.keyword, c1, c2)


class Xor:
    def __init__(self, keyword: str):
        self.keyword = keyword


# --- LogicalParser class to encapsulate parsing logic and keywords ---


class LogicalParser:
    """
    A class to parse logical sentences based on a predefined set of keywords.
    """

    def __init__(
        self,
        implies_keyword: str,
        or_keyword: str,
        and_keyword: str,
        not_keyword: str,
        xor_keyword: str,
        universal_quantifier: str = "x",
        implies_aliases: List[str] = None,
    ):
        self.implies = Implies(implies_keyword)
        self.or_op = Or(or_keyword)
        self.and_op = And(and_keyword)
        self.not_op = Not(not_keyword)
        self.xor_op = Xor(xor_keyword)
        self.universal_quantifier = universal_quantifier

        if implies_aliases:
            self.implies_aliases = set(implies_aliases)
        else:
            self.implies_aliases = set()

        self.all_keywords = {
            implies_keyword,
            or_keyword,
            and_keyword,
            not_keyword,
            xor_keyword,
        }
        self.subjects = set()
        # The rest of the state is handled by reset_rules()
        self.reset_rules()

    def reset_rules(self):
        # UNCHANGED from original
        self.parsed_rules = []
        self.universal_rules = []
        self.consequent_map = {}
        self.antecedents = set()
        self.parsed_rule_to_original = {}
        self.facts = set()
        self.subjects = set()

    def _translate_query_aliases(self, query: List[str]) -> List[str]:
        """Replaces any implication aliases in a query with the canonical keyword."""
        if not self.implies_aliases:
            return query

        canonical_keyword = self.implies.keyword
        return [
            canonical_keyword if token in self.implies_aliases else token
            for token in query
        ]

    # def _collect_subjects(self, component: LogicalComponent | str):
    #     """Recursively finds all subjects (non-keywords) in a component and adds to self.subjects."""
    #     if isinstance(component, str):
    #         if (
    #             component not in self.all_keywords
    #             and component != self.universal_quantifier
    #         ):
    #             self.subjects.add(component)
    #     elif isinstance(component, _NotComponent):
    #         self._collect_subjects(component.component)
    #     elif isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
    #         self._collect_subjects(component.component1)
    #         self._collect_subjects(component.component2)
    def _collect_subjects(self, component: LogicalComponent | str):
        """Recursively finds all subjects (non-keywords) in a component and adds to self.subjects."""
        if isinstance(component, str):
            is_potential_subject = (
                component not in self.all_keywords
                and component != self.universal_quantifier
                and " " not in component
                and "(" not in component
            )
            if is_potential_subject:
                if component.istitle() or component in self.subjects:
                    self.subjects.add(component)
        elif isinstance(component, _NotComponent):
            self._collect_subjects(component.component)
        elif isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
            self._collect_subjects(component.component1)
            self._collect_subjects(component.component2)

    def _split_into_subqueries(self, query: list[str]) -> list[list[str]]:
        # UNCHANGED from original
        is_indices = [
            i for i, token in enumerate(query) if token == self.implies.keyword
        ]
        if not is_indices:
            return [query]
        subqueries = []
        start_index = 0
        for is_idx in is_indices:
            end_index = is_idx + 2
            subquery_slice = query[start_index:end_index]
            subqueries.append(subquery_slice)
            start_index = is_idx + 2
        if start_index < len(query):
            if subqueries:
                subqueries[-1].extend(query[start_index:])
            else:
                subqueries.append(query[start_index:])
        return subqueries

    @classmethod
    def parse_statement(cls, tokens: tuple[str]) -> LogicalComponent | str:
        # UPDATED to include new keywords
        temp_parser = cls("is", "or", "and", "not", "xor", ["do", "have"])
        return temp_parser.parse(list(tokens))

    def parse(self, tokens: list[str]) -> LogicalComponent | str:
        # UPGRADED to handle XOR and nested implications
        if not tokens:
            return ""
        if len(tokens) == 1 and tokens[0] not in self.all_keywords:
            return tokens[0]
        if self.implies.keyword in tokens:
            split_index = max(
                i for i, token in enumerate(tokens) if token == self.implies.keyword
            )
            if split_index > 0:
                ante = self.parse(tokens[:split_index])
                cons = (
                    self.parse(tokens[split_index + 1 :])
                    if split_index < len(tokens) - 1
                    else ""
                )
                return self.implies(ante, cons)
            return ""
        if self.xor_op.keyword in tokens:
            split_index = tokens.index(self.xor_op.keyword)
            if split_index > 0 and split_index < len(tokens) - 1:
                c1 = self.parse(tokens[:split_index])
                c2 = self.parse(tokens[split_index + 1 :])
                return self.or_op(
                    self.and_op(c1, self.not_op(c2)), self.and_op(self.not_op(c1), c2)
                )
        if self.or_op.keyword in tokens:
            split_index = tokens.index(self.or_op.keyword)
            if split_index > 0 and split_index < len(tokens) - 1:
                return self.or_op(
                    self.parse(tokens[:split_index]),
                    self.parse(tokens[split_index + 1 :]),
                )
        if self.and_op.keyword in tokens:
            split_index = tokens.index(self.and_op.keyword)
            if split_index > 0 and split_index < len(tokens) - 1:
                return self.and_op(
                    self.parse(tokens[:split_index]),
                    self.parse(tokens[split_index + 1 :]),
                )
        if len(tokens) > 1 and all(token not in self.all_keywords for token in tokens):
            return self.and_op(tokens[0], self.parse(tokens[1:]))
        if tokens[0] == self.not_op.keyword and len(tokens) > 1:
            return self.not_op(self.parse(tokens[1:]))
        return " ".join(tokens)

    def _get_propositions(self, component: LogicalComponent | str, propositions: set):
        # UNCHANGED from original (but now used by new add_rules)
        if isinstance(component, str):
            if component:
                propositions.add(component)
        elif isinstance(component, _NotComponent):
            self._get_propositions(component.component, propositions)
        elif isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
            self._get_propositions(component.component1, propositions)
            self._get_propositions(component.component2, propositions)

    def _evaluate(self, component: LogicalComponent | str, assignments: dict) -> bool:
        # UNCHANGED from original
        if isinstance(component, str):
            return assignments.get(component, False)
        elif isinstance(component, _NotComponent):
            return not self._evaluate(component.component, assignments)
        elif isinstance(component, _AndComponent):
            return self._evaluate(component.component1, assignments) and self._evaluate(
                component.component2, assignments
            )
        elif isinstance(component, _OrComponent):
            return self._evaluate(component.component1, assignments) or self._evaluate(
                component.component2, assignments
            )
        elif isinstance(component, _ImpliesComponent):
            return not self._evaluate(
                component.component1, assignments
            ) or self._evaluate(component.component2, assignments)

    # --- NEW HELPER METHODS for the solver ---
    def _add_consequents_to_map(
        self, component: LogicalComponent | str, rule: _ImpliesComponent
    ):
        if isinstance(component, str):
            if component not in self.consequent_map:
                self.consequent_map[component] = []
            self.consequent_map[component].append(rule)
        elif isinstance(component, _NotComponent):
            if component not in self.consequent_map:
                self.consequent_map[component] = []
            self.consequent_map[component].append(rule)
            self._add_consequents_to_map(component.component, rule)
        elif isinstance(component, (_AndComponent, _OrComponent)):
            self._add_consequents_to_map(component.component1, rule)
            self._add_consequents_to_map(component.component2, rule)

    def _get_subject(self, component: LogicalComponent | str) -> str | None:
        if isinstance(component, str):
            if (
                component not in self.all_keywords
                and component != self.universal_quantifier
            ):
                return component
            return None
        elif isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
            return self._get_subject(component.component1)
        elif isinstance(component, _NotComponent):
            return self._get_subject(component.component)
        return None

    def _instantiate_component(
        self, component: LogicalComponent | str, subject: str
    ) -> LogicalComponent | str:
        if isinstance(component, str):
            return subject if component == self.universal_quantifier else component
        if isinstance(component, _NotComponent):
            return self.not_op(
                self._instantiate_component(component.component, subject)
            )
        if isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
            c1 = self._instantiate_component(component.component1, subject)
            c2 = self._instantiate_component(component.component2, subject)
            return type(component)(component.keyword, c1, c2)
        return component

    def _is_goal_in_consequent(
        self, goal: LogicalComponent | str, consequent: LogicalComponent | str
    ) -> bool:
        if goal == consequent:
            return True
        if isinstance(consequent, (_AndComponent, _OrComponent)):
            return self._is_goal_in_consequent(
                goal, consequent.component1
            ) or self._is_goal_in_consequent(goal, consequent.component2)
        if isinstance(consequent, _NotComponent):
            return self._is_goal_in_consequent(goal, consequent.component)
        return False

    # --- END NEW HELPER METHODS ---

    def add_rules(self, rules: List[Tuple[str, ...]]):
        self.reset_rules()

        specific_rule_tokens = []
        universal_rule_tokens = []

        # First pass: separate universal rules from specific ones
        for rule_tokens in rules:
            if self.universal_quantifier in rule_tokens:
                universal_rule_tokens.append(rule_tokens)
            else:
                specific_rule_tokens.append(rule_tokens)

        # Second pass: process all specific rules to find every known subject
        for rule_tokens in specific_rule_tokens:
            temp_parsed_rule = self.parse(list(rule_tokens))
            self._collect_subjects(temp_parsed_rule)

        # Third pass: create specific instances of universal rules for each subject
        all_specific_rules_to_process = list(specific_rule_tokens)
        for uni_rule_tokens in universal_rule_tokens:
            for subject in self.subjects:
                # Create a new, specific rule by replacing 'x' with the subject
                new_rule_tokens = tuple(
                    subject if token == self.universal_quantifier else token
                    for token in uni_rule_tokens
                )
                all_specific_rules_to_process.append(new_rule_tokens)

        # Final pass: parse and store all specific rules (original + expanded)
        for rule_tokens in all_specific_rules_to_process:
            rule = self.parse(list(rule_tokens))
            if rule not in self.parsed_rules:  # Avoid duplicates
                self.parsed_rules.append(rule)
                self.parsed_rule_to_original[rule] = list(rule_tokens)
                if isinstance(rule, _ImpliesComponent):
                    self.facts.add(rule)
                    self._get_propositions(rule.component1, self.antecedents)
                    self._add_consequents_to_map(rule.component2, rule)

    def _negate_component(
        self, component: Union[LogicalComponent, str]
    ) -> Union[LogicalComponent, str]:
        """Takes a component and returns its logical negation."""
        # This reuses the logic from your Not class to handle ¬(¬A) => A
        return self.not_op(component)

    def _backward_chain(
        self,
        goal: LogicalComponent | str,
        visited: set,
        memo: dict,
        knowledge_base: Union[Set[LogicalComponent | str], LogicalComponent | str],
    ) -> list[LogicalComponent] | None:
        if not isinstance(knowledge_base, set):
            knowledge_base = {knowledge_base}

        if goal in memo:
            return memo[goal]
        if goal in visited:
            return None

        negated_goal = self._negate_component(goal)
        if negated_goal in memo and memo[negated_goal] is not None:
            memo[goal] = None  # Memoize this failure
            return None

        # Check if the goal is already in our set of known facts for this proof.
        if goal in knowledge_base:
            return []  # Success, it's a known fact.

        # The rest of the function uses the original 'premise' variable name for simplicity.
        premise = knowledge_base

        # --- Base cases for compound goals ---
        if isinstance(goal, _AndComponent):
            p1 = self._backward_chain(
                goal.component1, visited | {goal}, memo, knowledge_base
            )
            if p1 is None:
                return None
            p2 = self._backward_chain(
                goal.component2, visited | {goal}, memo, knowledge_base
            )
            if p2 is None:
                return None
            return list(dict.fromkeys(p1 + p2))

        if isinstance(goal, _OrComponent):
            p1 = self._backward_chain(
                goal.component1, visited | {goal}, memo, knowledge_base
            )
            if p1 is not None:
                return p1
            return self._backward_chain(
                goal.component2, visited | {goal}, memo, knowledge_base
            )

        if isinstance(goal, _ImpliesComponent):
            # This logic is for handling nested implications if they become a goal.
            # A more advanced solver might use the deduction theorem here.
            return self._backward_chain(
                goal.component2, visited | {goal}, memo, knowledge_base
            )

        # --- Find and apply relevant rules ---
        possible_rules = list(self.consequent_map.get(goal, []))
        subject = self._get_subject(goal) or self._get_subject(premise)
        if subject:
            for uni_rule in self.universal_rules:
                inst_rule = self._instantiate_component(uni_rule, subject)
                consequent = (
                    inst_rule.component2
                    if isinstance(inst_rule, _ImpliesComponent)
                    else inst_rule
                )
                if self._is_goal_in_consequent(goal, consequent):
                    if isinstance(inst_rule, _ImpliesComponent):
                        possible_rules.append(inst_rule)
                    else:
                        possible_rules.append(
                            _ImpliesComponent("is", inst_rule, inst_rule)
                        )

        min_path = None
        for rule in possible_rules:
            # This helper determines the next subgoal based on the rule being used.
            def build_new_goal(
                current_goal, rule_consequent, rule_antecedent, in_or_branch=False
            ):
                if current_goal == rule_consequent:
                    return rule_antecedent
                if isinstance(rule_consequent, _AndComponent):
                    other_part = None
                    if current_goal == rule_consequent.component1:
                        other_part = rule_consequent.component2
                    elif current_goal == rule_consequent.component2:
                        other_part = rule_consequent.component1
                    if other_part:
                        # If inside an OR, the new goal needs both the antecedent
                        # and the other part of the AND.
                        if in_or_branch:
                            return self.and_op(rule_antecedent, other_part)
                        # Otherwise, the new goal is just the antecedent.
                        else:
                            return rule_antecedent
                if isinstance(rule_consequent, _OrComponent):
                    # Try the first branch of the OR; if it fails, try the second.
                    return build_new_goal(
                        current_goal, rule_consequent.component1, rule_antecedent, True
                    ) or build_new_goal(
                        current_goal, rule_consequent.component2, rule_antecedent, True
                    )
                return None

            new_goal = build_new_goal(goal, rule.component2, rule.component1)
            if new_goal is None:
                continue

            sub_path = self._backward_chain(
                new_goal, visited | {goal}, memo, knowledge_base
            )

            if sub_path is not None:
                current_path = sub_path + [rule]
                if min_path is None or len(current_path) < len(min_path):
                    min_path = current_path

        memo[goal] = min_path
        return min_path

    # Add this helper method to your LogicalParser class
    def _get_atomic_propositions(
        self, component: Union[LogicalComponent, str], atoms: Set[str]
    ):
        """Recursively finds all atomic string propositions in a component."""
        if isinstance(component, str):
            # We only care about base facts, not keywords or quantifiers
            if (
                component
                and component not in self.all_keywords
                and component != self.universal_quantifier
            ):
                atoms.add(component)
        elif isinstance(component, _NotComponent):
            self._get_atomic_propositions(component.component, atoms)
        elif isinstance(component, (_AndComponent, _OrComponent, _ImpliesComponent)):
            self._get_atomic_propositions(component.component1, atoms)
            self._get_atomic_propositions(component.component2, atoms)

    def find_proof(
        self, query: list[str], fuzzy: bool = False, threshold: float = 0.3
    ) -> list[LogicalComponent] | None:
        # UNCHANGED from original
        subqueries = self._split_into_subqueries(query)
        proofs = []
        for subquery in subqueries:
            parsed_query = self.parse(subquery)
            if not isinstance(parsed_query, _ImpliesComponent):
                return proofs
            query_antecedent = parsed_query.component1
            query_consequent = parsed_query.component2
            starting_premise = None
            if fuzzy:
                query_propositions = set()
                self._get_propositions(query_antecedent, query_propositions)
                completed_rule, consequent = self._find_best_match_for_query(
                    query_propositions, threshold
                )
                if completed_rule is None:
                    return proofs
                parsed_antecedent = self.parse(list(completed_rule))
                if not isinstance(parsed_antecedent, _ImpliesComponent):
                    starting_premise = parsed_antecedent
                else:
                    starting_premise = parsed_antecedent.component1
            else:
                starting_premise = query_antecedent
                propositions_in_premise = set()
                self._get_propositions(starting_premise, propositions_in_premise)
                if not propositions_in_premise.issubset(self.antecedents):
                    premise_proof = self._backward_chain(
                        starting_premise, set(), {}, "start"
                    )
                    if premise_proof is None:
                        return proofs
                    else:
                        return proofs
            memo = {}
            proof = self._backward_chain(
                query_consequent, set(), memo, starting_premise
            )
            if proof is None:
                return proofs
            else:
                proofs.extend(proof)
        return proofs

    def solve(
        self, query: list[str], fuzzy: bool = False, threshold: float = 0.3
    ) -> bool:
        # UNCHANGED from original
        proofs = self.find_proof(query, fuzzy, threshold)
        if proofs is not None and proofs != []:
            return True
        else:
            return False

    def solve_three_valued(self, query: list[str]) -> str:
        """
        Solves a query using strict three-valued logic. If any atomic part
        of the query is unknown, the whole query is uncertain.
        """
        query = self._translate_query_aliases(query)
        parsed_query = self.parse(query)

        if not isinstance(parsed_query, _ImpliesComponent):
            # Fallback for simple fact queries
            if self.find_proof(query) is not None:
                return "true"
            return "uncertain"

        premise = parsed_query.component1
        consequent = parsed_query.component2
        knowledge_base = self.facts | {premise}
        memo = {}

        # 1. NEW: Strict evaluation pre-check for unknown atoms
        atomic_propositions = set()
        self._get_atomic_propositions(consequent, atomic_propositions)

        for atom in atomic_propositions:
            memo.clear()
            atom_is_provable = (
                self._backward_chain(atom, set(), memo, knowledge_base) is not None
            )

            memo.clear()
            negated_atom = self._negate_component(atom)
            negated_atom_is_provable = (
                self._backward_chain(negated_atom, set(), memo, knowledge_base)
                is not None
            )

            if not atom_is_provable and not negated_atom_is_provable:
                # Found an atom that is neither provably true nor provably false.
                # Under strict evaluation, the entire query is uncertain.
                return "uncertain"

        # 2. If all atoms are known, proceed with normal evaluation
        memo.clear()
        can_prove_true = (
            self._backward_chain(consequent, set(), memo, knowledge_base) is not None
        )

        memo.clear()
        negated_consequent = self._negate_component(consequent)
        can_prove_false = (
            self._backward_chain(negated_consequent, set(), memo, knowledge_base)
            is not None
        )

        # 3. Decide based on the results
        if can_prove_true and can_prove_false:
            return "uncertain"
        elif can_prove_true:
            return "true"
        elif can_prove_false:
            return "false"
        else:
            return "uncertain"

    def _find_best_match_for_query(
        self, query_propositions: Set[str], threshold: float
    ) -> tuple[list[str], str] | tuple[None, None]:
        # UNCHANGED from original
        best_match = None
        best_score = -1.0
        for rule in self.parsed_rules:
            if isinstance(rule, _ImpliesComponent):
                rule_propositions = set()
                self._get_propositions(rule.component1, rule_propositions)
                if not rule_propositions or not query_propositions:
                    score = 0
                else:
                    intersection = len(
                        query_propositions.intersection(rule_propositions)
                    )
                    union = len(query_propositions.union(rule_propositions))
                    score = intersection / union
                if score > best_score and score >= threshold:
                    best_score = score
                    original_rule_tokens = self.parsed_rule_to_original.get(rule)
                    consequent = rule.component2
                    best_match = (original_rule_tokens, str(consequent))
        if best_match:
            return best_match
        return None, None

    def complete_statement(
        self, query: list[str], fuzzy: bool = False, threshold: float = 0.3
    ) -> tuple[list[str], str] | tuple[None, None]:
        # UNCHANGED from original
        parsed_query = self.parse(query)
        query_propositions = set()
        self._get_propositions(parsed_query, query_propositions)
        if not fuzzy:
            for rule in self.parsed_rules:
                if isinstance(rule, _ImpliesComponent):
                    rule_propositions = set()
                    self._get_propositions(rule.component1, rule_propositions)
                    if query_propositions == rule_propositions:
                        original_rule_tokens = self.parsed_rule_to_original.get(rule)
                        consequent = rule.component2
                        return original_rule_tokens, str(consequent)
            return None, None
        else:
            return self._find_best_match_for_query(query_propositions, threshold)

    def chain_consequents(
        self, query: list[str], fuzzy: bool = False, threshold: float = 0.3
    ) -> str | None:
        # UNCHANGED from original
        completed_rule, current_consequent = self.complete_statement(
            query, fuzzy, threshold
        )
        if current_consequent is None:
            return None
        visited_consequents = {current_consequent}
        while True:
            found_next_link = False
            next_consequent = None
            for rule in self.parsed_rules:
                if isinstance(rule, _ImpliesComponent) and isinstance(
                    rule.component1, str
                ):
                    if rule.component1 == current_consequent:
                        if str(rule.component2) not in visited_consequents:
                            next_consequent = rule.component2
                            found_next_link = True
                            break
            if found_next_link:
                current_consequent = str(next_consequent)
                visited_consequents.add(current_consequent)
            else:
                return current_consequent

    def get_rule_index(
        self, rule_to_find: Union[Tuple[str, ...], LogicalComponent]
    ) -> int | None:
        # UNCHANGED from original
        parsed_rule_to_find = rule_to_find
        if isinstance(rule_to_find, tuple):
            parsed_rule_to_find = self.parse(list(rule_to_find))
        for i, rule in enumerate(self.parsed_rules):
            if rule == parsed_rule_to_find:
                return i
        return None


# --- Examples of usage ---
if __name__ == "__main__":
    prontoQA_examples = False
    proverQA_examples = True

    if prontoQA_examples:
        # Create an instance of the LogicalParser, passing keywords directly
        parser = LogicalParser(
            implies_keyword="is",
            or_keyword="or",
            and_keyword="and",
            not_keyword="not",
            xor_keyword="xor",
        )

        # Example 1: Simple statement
        tokens1 = ["Apples", "and", "Oranges"]
        parsed1 = parser.parse(tokens1)
        print("Example 1: Simple statement")
        print(parsed1)
        print("-" * 20)

        # Example 2: Nested statement
        tokens2 = ["not", "Apples", "and", "Oranges"]
        parsed2 = parser.parse(tokens2)
        print("Example 2: Nested statement")
        print(parsed2)
        print("-" * 20)

        # Example 3: Complex nested statement with multiple operators
        tokens3 = ["A", "and", "B", "or", "C", "is", "D"]
        parsed3 = parser.parse(tokens3)
        print("Example 3: Another complex nested statement")
        print(parsed3)
        print("-" * 20)

        # Example 4: A more complex statement
        tokens4 = ["not", "A", "and", "B", "is", "C", "or", "D"]
        parsed4 = parser.parse(tokens4)
        print("Example 4: A more complex statement")
        print(parsed4)
        print("-" * 20)

        # Example 5: Logical consequence solving and finding a proof
        rules = [("A", "is", "B"), ("B", "is", "C")]
        query = ["A", "is", "C"]
        parser.reset_rules()
        parser.add_rules(rules)
        is_consequence = parser.solve(query)
        proof_path = parser.find_proof(query)

        print(f"Example 5: Solving logical consequence and finding a proof")
        print("Rules:")
        for rule_tokens in rules:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query)}")
        print(f"Is the query a consequence of the rules? {is_consequence}")
        print(f"Proof Path: {proof_path}")
        print("-" * 20)

        # Example 6: Logical consequence solving (should be False)
        rules2 = [
            (
                "A",
                "is",
                "B",
            )
        ]
        query2 = ["B", "is", "A"]
        parser.reset_rules()
        parser.add_rules(rules2)
        is_consequence2 = parser.solve(query2)
        proof_path2 = parser.find_proof(query2)

        print(f"Example 6: Solving logical consequence (False case)")
        print("Rules:")
        for rule_tokens in rules2:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query2)}")
        print(f"Is the query a consequence of the rules? {is_consequence2}")
        print(f"Proof Path: {proof_path2}")
        print("-" * 20)

        # Example 7: Demonstrating the new shortest path logic
        # Rule B implies A, but there is also a longer path from C to A
        rules3 = [
            ("B", "is", "A"),
            ("C", "is", "B"),
            ("D", "is", "C"),
            ("D", "is", "A"),
        ]
        query3 = ["D", "is", "A"]
        parser.reset_rules()
        parser.add_rules(rules3)
        is_consequence3 = parser.solve(query3)
        proof_path3 = parser.find_proof(query3)

        print(f"Example 7: Demonstrating the new shortest path logic")
        print("Rules:")
        for rule_tokens in rules3:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query3)}")
        print(f"Is the query a consequence of the rules? {is_consequence3}")
        print(f"Proof Path: {proof_path3}")
        print("-" * 20)

        # User's specific example
        rules4 = [
            ("numpus", "is", "impus"),
            ("numpus", "is", "not", "wooden"),
            ("numpus", "is", "sterpus"),
            ("vumpus", "is", "brimpus"),
            ("vumpus", "is", "lempus"),
            ("vumpus", "is", "not", "discordant"),
            ("rompus", "is", "not", "dull"),
            ("lorpus", "is", "not", "mean"),
            ("lorpus", "is", "wumpus"),
            ("gorpus", "is", "moderate"),
            ("zumpus", "is", "not", "spicy"),
            ("shumpus", "is", "gorpus"),
            ("shumpus", "is", "numpus"),
            ("shumpus", "is", "windy", "and", "green"),
            ("brimpus", "is", "grimpus"),
            ("brimpus", "is", "large"),
            ("brimpus", "is", "zumpus"),
            ("grimpus", "is", "orange"),
            ("grimpus", "is", "rompus"),
            ("grimpus", "is", "shumpus"),
            ("impus", "is", "not", "opaque"),
            ("lempus", "is", "hot"),
            ("max", "is", "vumpus"),
            ("vumpus", "is", "lorpus"),
            ("hans", "is", "red"),
        ]
        parser.reset_rules()
        parser.add_rules(rules4)
        query = ["shumpus", "is", "windy"]
        print("User's specific example:")
        print("Rules:")
        for rule_tokens in rules4:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query)}")
        print(f"Proof: {parser.find_proof(query)}")
        print(f"Solution: {parser.solve(query)}")
        print("-" * 20)

        query = ["max", "is", "not", "mean", "and", "windy"]
        print("User's specific example2:")
        print("Rules:")
        for rule_tokens in rules4:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query)}")
        print(f"Proof: {parser.find_proof(query)}")
        print(f"Solution: {parser.solve(query)}")
        print("-" * 20)

        query = ["vumpus", "is", "hot"]
        print("User's specific example3:")
        print("Rules:")
        for rule_tokens in rules4:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query)}")
        print(f"Proof: {parser.find_proof(query)}")
        print(f"Solution: {parser.solve(query)}")
        print("-" * 20)

        query = ["max", "is", "orange"]
        print("User's specific example4:")
        print("Rules:")
        for rule_tokens in rules4:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query)}")
        print(f"Proof: {parser.find_proof(query)}")
        print(f"Solution: {parser.solve(query)}")
        print("-" * 20)

        # Example 8: Direct statement completion
        parser.reset_rules()
        rules5 = [
            ("black", "and", "red", "and", "gold", "is", "german"),
            ("blue", "and", "white", "and", "red", "is", "french"),
        ]
        parser.add_rules(rules5)
        query_to_complete = ["black", "and", "red", "and", "gold"]
        completed_rule, consequent = parser.complete_statement(query_to_complete)

        print(f"Example 8: Direct statement completion")
        print("Rules:")
        for rule_tokens in rules5:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query_to_complete)}")
        print(f"Completed Rule: {parser.parse(completed_rule)}")
        print(f"Consequent: {consequent}")
        print("-" * 20)

        # Example 9: Chaining consequents
        parser.reset_rules()
        rules6 = [
            ("black", "and", "red", "and", "gold", "is", "german"),
            ("blue", "and", "white", "and", "red", "is", "french"),
            ("german", "is", "cool"),
        ]
        parser.add_rules(rules6)
        query_to_chain = ["black", "and", "red", "and", "gold"]
        chained_consequent = parser.chain_consequents(query_to_chain)

        print(f"Example 9: Chaining consequents")
        print("Rules:")
        for rule_tokens in rules6:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(query_to_chain)}")
        print(f"Final Chained Consequent: {chained_consequent}")
        print("-" * 20)

        # Example 10: Implicit 'and' parsing
        parser.reset_rules()
        parser.add_rules(rules6)
        implicit_and_query = ["black", "red", "gold"]
        completed_rule_implicit, consequent_implicit = parser.complete_statement(
            implicit_and_query
        )

        print(f"Example 10: Implicit 'and' parsing")
        print("Rules:")
        for rule_tokens in rules6:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query (Implicit 'and'): {implicit_and_query}")
        print(f"Parsed Query: {parser.parse(implicit_and_query)}")
        print(f"Completed Rule: {parser.parse(completed_rule_implicit)}")
        print(f"Consequent: {consequent_implicit}")
        print("-" * 20)

        # Example 11: Using the new class method
        print("Example 11: Using the new class method")
        tokens_to_parse = ("A", "and", "B", "or", "C")
        parsed_with_classmethod = LogicalParser.parse_statement(tokens_to_parse)

        # Let's verify it with an instance method call
        temp_parser = LogicalParser(
            implies_keyword="is",
            or_keyword="or",
            and_keyword="and",
            not_keyword="not",
            xor_keyword="xor",
        )
        parsed_with_instance = temp_parser.parse(list(tokens_to_parse))

        print(f"Tokens to parse: {tokens_to_parse}")
        print(f"Parsed via class method: {parsed_with_classmethod}")
        print(f"Parsed via instance method: {parsed_with_instance}")
        print(
            f"Results are identical: {parsed_with_classmethod == parsed_with_instance}"
        )
        print("-" * 20)

        # Example 12: Getting the index of a rule
        parser.reset_rules()
        rules7 = [
            ("apple", "and", "banana", "is", "fruit"),
            ("fruit", "is", "healthy"),
            ("healthy", "and", "tasty", "is", "great_snack"),
        ]
        parser.add_rules(rules7)

        print("Example 12: Getting the index of a rule")
        print("Rules:")
        for rule_tokens in rules7:
            print(f"  {parser.parse(list(rule_tokens))}")

        # Case 1: Find rule by original tuple
        rule_to_find_tuple = ("fruit", "is", "healthy")
        index_from_tuple = parser.get_rule_index(rule_to_find_tuple)
        print(f"Index of {rule_to_find_tuple} is: {index_from_tuple}")

        # Case 2: Find rule by parsed LogicalComponent
        parsed_rule_to_find = parser.parse(list(rules7[2]))
        index_from_component = parser.get_rule_index(parsed_rule_to_find)
        print(f"Index of {parsed_rule_to_find} is: {index_from_component}")

        # Case 3: Rule not found
        non_existent_rule = ("not", "apple", "is", "not_a_fruit")
        index_not_found = parser.get_rule_index(non_existent_rule)
        print(f"Index of {non_existent_rule} is: {index_not_found}")
        print("-" * 20)

        # Example 13: Fuzzy matching with `complete_statement`
        print("Example 13: Fuzzy matching with `complete_statement`")
        parser.reset_rules()
        rules8 = [
            ("black", "and", "red", "and", "gold", "is", "german"),
            ("blue", "and", "white", "and", "red", "is", "french"),
        ]
        parser.add_rules(rules8)

        # A query with an extra word "brown"
        fuzzy_query1 = ["brown", "black", "red", "gold"]
        completed_rule1, consequent1 = parser.complete_statement(
            fuzzy_query1, fuzzy=True, threshold=0.7
        )

        print(f"Rules:")
        for rule_tokens in rules8:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"\nQuery: {fuzzy_query1}")
        print(
            f"Completed Rule: {parser.parse(completed_rule1) if completed_rule1 else None}"
        )
        print(f"Consequent: {consequent1}")

        # A query with an extra, unrelated word "pink"
        fuzzy_query2 = ["brown", "black", "red", "pink", "gold"]
        completed_rule2, consequent2 = parser.complete_statement(
            fuzzy_query2, fuzzy=True, threshold=0.5
        )

        print(f"\nQuery: {fuzzy_query2}")
        print(
            f"Completed Rule: {parser.parse(completed_rule2) if completed_rule2 else None}"
        )
        print(f"Consequent: {consequent2}")
        print("-" * 20)

        # Example 14: Chaining with fuzzy matching
        print("Example 14: Chaining with fuzzy matching")
        parser.reset_rules()
        rules9 = [
            ("black", "and", "red", "and", "gold", "is", "german"),
            ("german", "is", "cool"),
        ]
        parser.add_rules(rules9)

        fuzzy_query_chain = ["brown", "black", "red", "gold"]
        chained_consequent_fuzzy = parser.chain_consequents(
            fuzzy_query_chain, fuzzy=True
        )

        print(f"Rules:")
        for rule_tokens in rules9:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"\nQuery: {fuzzy_query_chain}")
        print(f"Final Chained Consequent: {chained_consequent_fuzzy}")
        print("-" * 20)

        # Example 15: New fuzzy matching for `find_proof` and `solve`
        print("Example 15: Using fuzzy matching for `find_proof` and `solve`")
        parser.reset_rules()
        rules10 = [
            ("apple", "and", "banana", "is", "fruit"),
            ("fruit", "is", "healthy"),
        ]
        parser.add_rules(rules10)

        # This query has an extra word "green" in the antecedent.
        fuzzy_proof_query = ["apple", "banana", "green", "is", "healthy"]

        # A strict solve will fail.
        strict_solve = parser.solve(fuzzy_proof_query)
        strict_proof = parser.find_proof(fuzzy_proof_query)

        # A fuzzy solve will succeed.
        fuzzy_solve = parser.solve(fuzzy_proof_query, fuzzy=True)
        fuzzy_proof = parser.find_proof(fuzzy_proof_query, fuzzy=True, threshold=0.5)

        print(f"Rules:")
        for rule_tokens in rules10:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"Query: {parser.parse(fuzzy_proof_query)}")
        print(f"\nAttempting strict solve:")
        print(f"Solution: {strict_solve}")
        print(f"Proof: {strict_proof}")
        print(f"\nAttempting fuzzy solve:")
        print(f"Solution: {fuzzy_solve}")
        print(f"Proof: {fuzzy_proof}")
        print("-" * 20)

        # Example 16: FIX for the user's reported bug.
        # The query "apple banana is healthy" should be correctly parsed and solved.
        print("Example 16: FIX for implicit 'and' in the query")
        parser.reset_rules()
        rules11 = [
            ("apple", "and", "banana", "is", "fruit"),
            ("fruit", "is", "healthy"),
        ]
        parser.add_rules(rules11)

        # User's query with implicit 'and'
        bug_query = ["apple", "banana", "is", "healthy"]

        # This should now correctly return True.
        bug_solve = parser.solve(bug_query)
        bug_proof = parser.find_proof(bug_query)

        print(f"Rules:")
        for rule_tokens in rules11:
            print(f"  {parser.parse(list(rule_tokens))}")
        print(f"\nQuery: {parser.parse(bug_query)}")
        print(f"Solution: {bug_solve}")
        print(f"Proof: {bug_proof}")
        print("-" * 20)

        # Example 17: New fix for incomplete statements.
        print("Example 17: Handling incomplete statements gracefully")
        parser.reset_rules()

        # Test cases that previously caused an IndexError
        incomplete_is = ["wumpus", "is"]
        incomplete_and = ["max", "and"]
        incomplete_not = ["not"]
        empty_list = []

        print(f"Parsing '{incomplete_is}': {parser.parse(incomplete_is)}")
        print(f"Parsing '{incomplete_and}': {parser.parse(incomplete_and)}")
        print(f"Parsing '{incomplete_not}': {parser.parse(incomplete_not)}")
        print(f"Parsing '{empty_list}': {parser.parse(empty_list)}")
        print("-" * 20)

        # Example 18: FIX for the user's reported bug with multiple 'is' keywords
        print("Example 18: FIX for the user's reported bug with multiple 'is' keywords")
        parser.reset_rules()
        parser.add_rules(
            [
                ("max", "is", "vumpus"),
                ("vumpus", "is", "lorpus"),
                ("lorpus", "is", "wumpus"),
            ]
        )
        # The query is now separated into two sub-queries.
        compound_query = ["max", "is", "wumpus", "lorpus", "is", "wumpus"]
        subqueries = parser._split_into_subqueries(compound_query)
        compound_solve = parser.solve(compound_query)
        compound_proof = parser.find_proof(compound_query)
        print(f"Original Query: {compound_query}")
        print(f"Subqueries: {subqueries}")
        print(f"Solution: {compound_solve}")
        print(f"Proof: {compound_proof}")
        print("-" * 20)

        # Example 19: The new user query
        print(
            "Example 19: Handling the user's original query from Example 18 correctly"
        )
        parser.reset_rules()
        parser.add_rules(rules4)
        # This is the actual query from the user's initial report.
        new_query = ["max", "is", "orange", "wumpus", "is"]
        subqueries = parser._split_into_subqueries(new_query)

        new_solve = parser.solve(new_query)
        new_proof = parser.find_proof(new_query)

        print(f"Original Query: {new_query}")
        print(f"Subqueries: {subqueries}")
        print(f"\nSolution: {new_solve}")
        print(f"Proof: {new_proof}")
        print("-" * 20)

        # Example 19_2: The new user query
        print(
            "Example 19_2: Handling the user's original query from Example 18 correctly"
        )
        parser.reset_rules()
        parser.add_rules(rules4)
        # This is the actual query from the user's initial report.
        new_query = ["true", "false", "max", "is", "orange"]
        subqueries = parser._split_into_subqueries(new_query)

        new_solve = parser.solve(new_query, fuzzy=True)
        new_proof = parser.find_proof(new_query, fuzzy=True, threshold=0.3)

        print(f"Original Query: {new_query}")
        print(f"Subqueries: {subqueries}")
        print(f"\nSolution: {new_solve}")
        print(f"Proof: {new_proof}")
        print("-" * 20)

        # Example 20: A simple `max is orange` query that should pass.
        print("Example 20: A simple 'max is orange' query that should pass.")
        parser.reset_rules()
        parser.add_rules(rules4)
        simple_query = ["max", "is", "orange"]
        simple_solve = parser.solve(simple_query)
        simple_proof = parser.find_proof(simple_query)
        print(f"Query: {simple_query}")
        print(f"Solution: {simple_solve}")
        print(f"Proof: {simple_proof}")
        print("-" * 20)

        # Example 21: Proving a compound consequent (the user's original goal)
        print("Example 21: Proving 'max is orange and wumpus'")
        parser.reset_rules()
        rules5_and_wumpus = rules4 + [
            ("vumpus", "is", "lorpus"),
            ("lorpus", "is", "wumpus"),
        ]
        parser.add_rules(rules5_and_wumpus)
        compound_query_max = ["max", "is", "orange", "and", "wumpus"]
        compound_solve_max = parser.solve(compound_query_max)
        compound_proof_max = parser.find_proof(compound_query_max)

        print(f"Query: {compound_query_max}")
        print(f"Solution: {compound_solve_max}")
        print(f"Proof: {compound_proof_max}")
        print("-" * 20)

    # --- Example Usage ---
    if proverQA_examples:
        # 1. Initialize the parser with the new 'xor' keyword
        parser = LogicalParser(
            implies_keyword="is",
            or_keyword="or",
            and_keyword="and",
            not_keyword="not",
            xor_keyword="xor",
        )

        # 2. Let's see how it parses an XOR statement
        xor_statement = ("light-on", "xor", "door-closed")
        parsed_xor = parser.parse(list(xor_statement))

        print(f"Original: {' '.join(xor_statement)}")
        print(f"Parsed: {parsed_xor}")
        # Expected Output: Parsed: ((light-on ∧ ¬door-closed) ∨ (¬light-on ∧ door-closed))

        # 3. Use it in the solver with rules
        rules = [
            ("power-on", "is", "light-on", "xor", "fan-on"),
            ("light-on", "is", "room-bright"),
            ("fan-on", "is", "room-cool"),
        ]

        parser.add_rules(rules)

        # Let's test the logic. If power is on and the fan is NOT on, the light must be on.
        # From the rule: power-on -> (light-on XOR fan-on)
        # This is equivalent to: power-on -> ((light-on AND NOT fan-on) OR (NOT light-on AND fan-on))
        # So, if we know (power-on AND NOT fan-on), we can deduce light-on.

        query1 = ("power-on", "and", "not", "fan-on", "is", "light-on")
        can_be_solved1 = parser.solve(list(query1))
        print(
            f"\nCan we prove 'power-on and not fan-on is light-on'? {can_be_solved1}"
        )  # Should be True

        # If power is on and the light is on, the fan must be off.
        query2 = ("power-on", "and", "light-on", "is", "not", "fan-on")
        can_be_solved2 = parser.solve(list(query2))
        print(
            f"Can we prove 'power-on and light-on is not fan-on'? {can_be_solved2}"
        )  # Should be True
