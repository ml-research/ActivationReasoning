import torch
from ar.config import LogicConfig
from typing import Optional
from ar.model.logic import LogicalParser
from ar.utils import remove_consecutive_duplicates, replace_or_with_xor, implicit_is


class ALReasoner:
    def __init__(
        self,
        rules: dict,
        config: Optional[LogicConfig] = None,
        verbose: bool = False,
    ):
        self.rules = rules
        self.active_rules = torch.tensor([])  # shape is (batch, num_rules)
        self.rule_checking = (
            config.reasoner_rules_checking if config is not None else "simple"
        )
        self.verbose = True
        self.parser = LogicalParser(
            implies_keyword="is",
            or_keyword="or",
            and_keyword="and",
            not_keyword="not",
            xor_keyword="xor",
            universal_quantifier="x",
            implies_aliases=["do", "have"],
        )
        self.parser.add_rules(self.rules.keys())

        if len(self.rules) > 0 and verbose:
            print("-" * 20 + f"AL-Reasoner Component:" + "-" * 20)
            print(f"The reasoner uses '{self.rule_checking}' config for rule checking.")
            for rule_tokens in self.rules.keys():
                print(f"  {self.parser.parse(rule_tokens)}")
            print("-" * 20 + "End of AL-Reasoner Component" + "-" * 20)

    def _get_currently_active_rules(self, active_concepts):
        """
        Given some active concepts, determine the rules that are activated
        Args:
            List of str: Currently active concepts
        Returns:
            torch.Tensor: Currently active rules mask (shape: (num_rules))
        """

        def check_if_rule_active(active_concepts, rule_concepts):
            """
            Check if a rule is active based on the currently active concepts.
            The rule is active if the colors are in the active concepts list in order.
            Args:
                List of str: Currently active concepts
                List of str: Rule concepts for a specific rule (e.g. ['green', 'white', 'red'])
            Returns:
                bool: Whether the rule is active
            """
            idx = 0
            for rule_concept in rule_concepts:
                if rule_concept not in active_concepts[idx:]:
                    return False
                idx = active_concepts.index(rule_concept) + 1
            return True

        if self.rule_checking == "legacy":
            active_rules = torch.zeros(
                len(self.rules), dtype=torch.int
            )  # shape is (num_rules)
            for rule_id, (rule_concepts, rule) in enumerate(self.rules.items()):
                if check_if_rule_active(active_concepts, rule_concepts):
                    active_rules[rule_id] = 1
            return active_rules
        elif self.rule_checking == "simple":
            # TODO: What if more than one rule activates? -> Do something with permutate?
            active_rules = torch.zeros(len(self.rules), dtype=torch.int)
            print(active_concepts)
            completed_rule_implicit, consequent_implicit = (
                self.parser.complete_statement(
                    active_concepts, fuzzy=True, threshold=0.5
                )
            )
            if completed_rule_implicit:
                completed_rule_implicit = tuple(completed_rule_implicit)
                rule_id = list(self.rules.keys()).index(completed_rule_implicit)
                active_rules[rule_id] = 1
            return active_rules
        elif self.rule_checking == "complex":
            active_rules = torch.zeros(len(self.rules), dtype=torch.int)
            rule_path = self.parser.find_proof(
                remove_consecutive_duplicates(active_concepts),
                fuzzy=True,
                threshold=0.3,
            )
            if rule_path:
                rule_id = self.parser.get_rule_index(rule_path[-1])
                active_rules[rule_id] = 1
            return active_rules
        elif self.rule_checking == "open_world":
            active_rules = torch.zeros(len(self.rules), dtype=torch.int)
            # print(
            #     implicit_is(
            #         replace_or_with_xor(remove_consecutive_duplicates(active_concepts))
            #     ),
            # )

            answer = self.parser.solve_three_valued(
                implicit_is(
                    replace_or_with_xor(remove_consecutive_duplicates(active_concepts))
                ),
            )
            # print(answer)
            for rule_id, (rule_concepts, rule) in enumerate(self.rules.items()):
                if (answer,) == rule_concepts:
                    active_rules[rule_id] = 1
            return active_rules
        else:
            raise ValueError(
                f"'{self.rule_checking}' is not a valid rule checking method."
            )

    def add_concepts(
        self, activate_concept_mask, concept_names
    ) -> tuple[torch.Tensor, list]:
        """
        Activate the rules based on the currently active concepts.

        Args:
            torch.Tensor: Mask of activated concepts (shape: (batch, seq_len, num_concepts))
            List of str: List of concept names in the same order as the mask
        Returns:
            torch.Tensor: Tensor activated rules (shape: (batch, num_rules))
            List of str: List of rule names in the same order as the active rules tensor
        """
        # TODO: currently this is done for each token, we can optimize this by only checking the new tokens
        # TODO: _get_currently_active_rules uses concepts, we can use concept ids instead

        # if there are no active rules, we initialize the active rules tensor
        if not self.active_rules.numel():
            self.active_rules = torch.zeros(
                (activate_concept_mask.shape[0], len(self.rules)), dtype=torch.int
            )  # shape is (batch, num_rules)

        extracted_concepts = torch.nonzero(
            activate_concept_mask, as_tuple=False
        )  # shape is (total number of activated concepts, 3) where 3 is the [batch_id, seq_id, concept_id]

        if len(extracted_concepts) == 0:
            return self.active_rules, list(self.rules.values())
        for batch_id in range(activate_concept_mask.shape[0]):
            batch_concepts = extracted_concepts[
                extracted_concepts[:, 0] == batch_id
            ]  # get the concepts for the current batch

            batch_concepts = batch_concepts[
                batch_concepts[:, 1].argsort()
            ]  # order batch concepts by sequence id
            b_activated_concepts = [
                concept_names[concept_id] for _, _, concept_id in batch_concepts
            ]  # get the concept names

            # check the active rules and the active latent features
            b_active_rules = self._get_currently_active_rules(
                b_activated_concepts
            )  # shape is (num_rules)

            # update the active rules tensor, we take the max because we want to keep the rules that are active in the previous tokens
            self.active_rules[batch_id] = torch.max(
                self.active_rules[batch_id], b_active_rules
            )  # update the active rules tensor

        return self.active_rules, list(self.rules.values())

    def reset(self):
        """
        Reset the active rules tensor.
        """
        self.active_rules = torch.tensor([])

    def get_active_rules(self) -> list:
        """
        Get the currently activated rules.
        Returns:
            batched_rules: List of lists of rules for each batch
        """
        b_size = self.active_rules.shape[0]
        rule_names = list(self.rules.values())

        # get the batch ids and rule ids of the active rules
        batch_ids, rule_ids = torch.nonzero(self.active_rules, as_tuple=True)
        # convert rule ids to rule names
        active_rules = [rule_names[rule_id] for rule_id in rule_ids]

        # convert to list of rules for each batch
        batched_rules = [[] for _ in range(b_size)]
        for batch_id, rule in zip(batch_ids, active_rules):
            batched_rules[batch_id].append(rule)
        return batched_rules
