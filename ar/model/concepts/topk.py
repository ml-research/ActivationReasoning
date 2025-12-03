from __future__ import annotations

import os
from typing import Dict, List

import torch

from ar.config import LogicConfig
from ar.utils import COLORS, RESET, safe_load_tensor, get_weight_vector


class TopKConcepts:
    """
    Handle concept preparation for the ``top_k`` search strategy.

    This class loads the cached latent activations produced during concept search
    and extracts the most informative SAE features per concept.
    """

    def __init__(
        self,
        concepts: List[str],
        cache_dir: str,
        config: LogicConfig,
        verbose: bool = True,
    ) -> None:
        if config.concept_dictionary != "top_k":
            raise ValueError(
                "TopKConcepts can only be used with concept_dictionary='top_k'. "
                f"Received '{config.concept_dictionary}'."
            )

        self.concepts = concepts
        self.cache_dir = cache_dir
        self.config = config
        self.verbose = verbose

        self.search_top_k = config.search_top_k
        self.concept_dictionary_top_k_order = config.concept_dictionary_top_k_order
        self.concept_dictionary_top_k_concepts = (
            config.concept_dictionary_top_k_concepts
        )
        self.detection_top_k_output = config.detection_top_k_output

        self._concept_dict: Dict[str, Dict[str, List[int]]] = {}

        self._build_concepts()

    def __len__(self) -> int:
        return len(self.concepts)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def forward(
        self, latent_activations: torch.Tensor, concept_names: List[str]
    ) -> torch.Tensor:
        """
        Identifies and adds new concepts from model outputs. And activates rules based on the new concepts.

        Args:
            latent_activations (torch.Tensor): Latent activations from the SAE model (batch, sequence_length, latent_dim)
            attention_mask (torch.Tensor): Attention mask for the input sequence (batch, sequence_length)
        Returns:
            torch.Tensor: Mask of activated concepts (shape: (batch, sequence_length, num_concepts))
        """
        concept_tensor, _ = (
            self.get_concept_tensor()
        )  # shape is (num_concepts, steering_top_k_rule), indices of the concepts in the latent space
        if self.concept_dictionary_top_k_concepts > concept_tensor.shape[1]:
            raise ValueError(
                f"Top k concepts ({self.concept_dictionary_top_k_concepts}) for is greater than the number of indices available ({concept_tensor.shape[1]}). Please check the concept indices."
            )

        top_activations, top_output_indices = torch.topk(
            latent_activations, self.detection_top_k_output, dim=-1
        )  # top_activations shape is (batch, sequence_length, detection_top_k_output)

        # shape is (1, 1, 1, num_concepts, concept_dictionary_top_k_concepts)
        concept_dictionary_top_k_concepts_indices = (
            concept_tensor[:, : self.concept_dictionary_top_k_concepts]
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .cuda()
        )

        # Shape (batch, sequence_length, detection_top_k_output, 1, 1)
        reshaped_output_indices = top_output_indices.unsqueeze(-1).unsqueeze(-1).cuda()

        # Shape: (batch, sequence_length, detection_top_k_output, 1, 1)
        reshaped_activation = top_activations.unsqueeze(-1).unsqueeze(-1).cuda()

        # Check which output indices match with which concept indices
        # Result shape: (batch, sequence_length, detection_top_k_output, num_concepts, concept_dictionary_top_k_concepts)
        matches = reshaped_output_indices == concept_dictionary_top_k_concepts_indices

        # shape is (batch, sequence_length, detection_top_k_output, num_concepts, concept_dictionary_top_k_concepts)
        concept_activations = reshaped_activation * matches.float()

        # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
        activations = concept_activations.sum(dim=2)

        # add weighted activations based on the index position
        weights = get_weight_vector(
            self.concept_dictionary_top_k_concepts,
            steering_weighting_function="log_decay",
            mean=1,
            std=None,
        ).to(activations.device)
        activations = (
            activations * weights
        )  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
        return activations

    def get_concept_names(self) -> List[str]:
        return self.concepts

    def get_concept_dict(self) -> Dict[str, Dict[str, List[int]]]:
        return self._concept_dict

    def update_concept_dict_entry(
        self, concept_name: str, concept_dict: Dict[str, List[int]]
    ) -> None:
        self._concept_dict[concept_name] = concept_dict

    def get_concept_tensor(self) -> tuple[torch.Tensor, List[str]]:
        """Return indices as a tensor with shape (num_concepts, top_k) and concept names."""
        indices: List[List[int]] = []
        for concept in self.concepts:
            if concept not in self._concept_dict:
                raise ValueError(
                    f"Concept {concept} not found in cached activations. Did you run search()?"
                )
            indices.append(self._concept_dict[concept]["indices"])
        return torch.tensor(indices), self.concepts

    def get_concept_indices(self, concept: str, top_k: int | None = None) -> List[int]:
        if concept not in self._concept_dict:
            raise ValueError(f"Concept '{concept}' not found.")
        indices = self._concept_dict[concept]["indices"]
        return indices[:top_k] if top_k is not None else indices

    def get_concept_weights(
        self, concept: str, top_k: int | None = None
    ) -> List[float]:
        if concept not in self._concept_dict:
            raise ValueError(f"Concept '{concept}' not found.")
        weights = self._concept_dict[concept]["weights"]
        return weights[:top_k] if top_k is not None else weights

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _build_concepts(self) -> None:
        activations = self._load_latent_activations()
        self._concept_dict = compute_top_k_activations(
            activations,
            k=self.search_top_k,
            concept_dictionary_top_k_order=self.concept_dictionary_top_k_order,
            verbose=self.verbose,
        )

    def _load_latent_activations(self) -> Dict[str, Dict[str, torch.Tensor]]:
        sae_latent_activations: Dict[str, Dict[str, torch.Tensor]] = {
            "positive": {},
            "negative": {},
        }

        for concept in self.concepts:
            try:
                sae_latent_activations["positive"][concept] = safe_load_tensor(
                    os.path.join(self.cache_dir, "concepts", f"{concept}_positive.pt")
                )
                sae_latent_activations["negative"][concept] = safe_load_tensor(
                    os.path.join(self.cache_dir, "concepts", f"{concept}_negative.pt")
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Could not find SAE indices for concept '{concept}' in {self.cache_dir}. "
                    "Please run search() first."
                ) from exc

        return sae_latent_activations


# ------------------------------------------------------------------------- #
# Functional helpers (ported from the original Concepts implementation)     #
# ------------------------------------------------------------------------- #


def compute_top_k_activations(
    activations: Dict[str, Dict[str, torch.Tensor]],
    k: int = 10,
    concept_dictionary_top_k_order: str = "unique_first",
    verbose: bool = False,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Compute the top-k SAE indices per concept using cached activations.
    """
    txt = (
        "-" * 20
        + "AL Gready Concepts"
        + "-" * 20
        + "\n"
        + f"Concept Overview (strategy: {concept_dictionary_top_k_order}):\n"
    )

    if not activations:
        return {}

    activations_pos = activations["positive"]
    activations_neg = activations["negative"]
    concept_names = list(activations_pos.keys())
    sae_latent_dimension = activations_pos[concept_names[0]].shape[-1]

    mean_activations = torch.empty((0, sae_latent_dimension))

    for concept in concept_names:
        positive_mean = activations_pos[concept].mean(dim=0)
        negative_mean = activations_neg[concept].mean(dim=0)
        concept_activation = positive_mean - negative_mean
        mean_activations = (
            torch.cat((mean_activations, concept_activation.unsqueeze(0)), dim=0)
            if mean_activations.numel()
            else concept_activation.unsqueeze(0)
        )

    top_values, top_indices = torch.topk(mean_activations, k, dim=1)

    top_indices_reordered = torch.zeros_like(top_indices)
    top_values_reordered = torch.zeros_like(top_values)

    all_indices = top_indices.flatten()
    unique_indices, counts = torch.unique(all_indices, return_counts=True)
    index_to_freq = {
        idx.item(): count.item() for idx, count in zip(unique_indices, counts)
    }

    for i in range(top_indices.shape[0]):
        concept_indices = top_indices[i].clone()
        concept_values = top_values[i].clone()

        if concept_indices.numel() == 0:
            continue

        frequencies = torch.tensor(
            [index_to_freq.get(idx.item(), 0) for idx in concept_indices],
            device=concept_indices.device,
        )

        _, sorted_idx = torch.sort(frequencies)
        top_indices_reordered[i] = concept_indices[sorted_idx]
        top_values_reordered[i] = concept_values[sorted_idx]

    results: Dict[str, Dict[str, List[int]]] = {
        concept: {"indices": [], "weights": []} for concept in activations_pos
    }
    con_width = max(len(c) for c in activations_pos) + 5

    for i, concept in enumerate(activations_pos):
        org_indices = top_indices[i].cpu().tolist()
        if concept_dictionary_top_k_order == "unique_first":
            indices = top_indices_reordered[i].cpu().tolist()
            weights = top_values_reordered[i].cpu().tolist()
        elif concept_dictionary_top_k_order == "unique_only":
            indices = [
                idx if index_to_freq[idx] == 1 else 0
                for idx in top_indices_reordered[i].cpu().tolist()
            ]
            weights = [
                value if index_to_freq[idx] == 1 else 0
                for value, idx in zip(
                    top_values_reordered[i].cpu().tolist(),
                    top_indices_reordered[i].cpu().tolist(),
                )
            ]
        elif concept_dictionary_top_k_order == "original_order":
            indices = top_indices[i].cpu().tolist()
            weights = top_values[i].cpu().tolist()
        else:
            raise ValueError(
                f"Indexing strategy '{concept_dictionary_top_k_order}' is not supported."
            )

        concept_txt = f"{concept:<{con_width}} "
        max_width = len(str(sae_latent_dimension))
        colored_indices_org: List[str] = []
        colored_indices_new: List[str] = []

        for new_idx, orig_idx in zip(indices, org_indices):
            for idx, container in (
                (new_idx, colored_indices_new),
                (orig_idx, colored_indices_org),
            ):
                if idx != 0 and index_to_freq[idx] == 1:
                    container.append(f"{COLORS['GREEN']}{idx:>{max_width}}{RESET}")
                else:
                    container.append(f"{COLORS['ORANGE']}{idx:>{max_width}}{RESET}")

        if colored_indices_org != colored_indices_new:
            concept_txt += (
                ", ".join(colored_indices_new)
                + "  <-  "
                + ", ".join(colored_indices_org)
            )
        else:
            concept_txt += ", ".join(colored_indices_org)

        txt += concept_txt + "\n"
        results[concept] = {"indices": indices, "weights": weights}

    txt += f"Color legend: {COLORS['GREEN']}■{RESET} unique, {COLORS['ORANGE']}■{RESET} duplicate\n"
    if verbose:
        print(concept_dictionary_top_k_order)
        print(txt.strip())

    return results


__all__ = ["TopKConcepts", "compute_top_k_activations"]
