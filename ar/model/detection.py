import os

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from ar.config import LogicConfig
from ar.model.concepts import Concepts
from ar.model.tresholding import DetectionThresholds, ThresholdManager
from ar.utils import COLORS, RESET, get_weight_vector


class ALConceptDetector:
    """
    Class to manage activation logic rules and steering for language models.

    This component handles:
    1. Tracking active concepts
    2. Applying rules based on concept activation
    3.

    Attributes:
        rules (dict): Dictionary of rules and their associated concepts
        activate_concept_mask (torch.Tensor): Mask of activated concepts (shape: (batch, sequence_length, num_concepts))
        activate_concept_confidence (torch.Tensor): Confidence for each concept activation (shape: (batch, sequence_length, num_concepts))
        verbose (bool): Flag to control verbosity of logging
    """

    def __init__(
        self,
        verbose: bool = True,
    ):
        """
        Initialize the logic component.

        Args:
            concepts (List[str]): List of concepts to be activated
            verbose (bool): Flag to control verbosity of logging
        """

        # Input logging
        self.verbose = verbose

        # Concepts
        self.concepts = None  # type: Optional[Concepts]
        self.concept_names = []  # List of concept names

        # State variables
        self.activate_concept_mask = torch.empty(
            0
        )  # shape is (batch, sequence_length, num_concepts), a mask that can be mapped on the input
        self.activate_concept_confidence = torch.empty(
            0
        )  # shape is (batch, sequence_length, num_concepts), the confidence for each concept activation
        self.attention_mask = torch.empty(0)  # shape is (batch, sequence_length)
        self.local_thresholds = torch.zeros(0)
        self.global_thresholds = torch.zeros(0)
        self.threshold_manager: Optional[ThresholdManager] = None

    def set_up(self, concepts: Concepts, config: LogicConfig, cache_dir: str, verbose: bool = False):
        """
        Set up the detector component with the indices and weights of the concepts. search_top_k are the indices of the concepts in the latent space.
        This method is called after the logic component is initialized.

        Args:
            concepts (Concepts): Concepts object containing the concept indices and weights
            verbose (bool): Flag to control verbosity of logging
        """
        self.verbose = verbose if verbose is not None else self.verbose
        self.concepts = concepts
        self.cache_dir = cache_dir
        self.config = config if config is not None else LogicConfig()
        self.concept_names = self.concepts.get_concept_names()  # List of concept names
        
        self.threshold_manager = ThresholdManager(
            cache_dir=self.cache_dir,
            config=self.config,
            concept_names=self.concept_names,
            verbose=self.verbose,
        )

        thresholds: DetectionThresholds = self.threshold_manager.initialize_thresholds(
            detection_threshold_config=self.config.detection_threshold,
            detector=self,
            reset_cache=False,
        )
        self.local_thresholds, self.global_thresholds = thresholds.as_tuple()

        if self.verbose:
            print(f"Concept Detector set up with {len(self.concept_names)} concepts and detection threshold={self.config.detection_threshold}")
    
    
    def reset(self):
        """
        Resets state variables to empty tensors.
        """
        self.activate_concept_mask = torch.empty(0)
        self.activate_concept_confidence = torch.empty(0)
        self.attention_mask = torch.empty(0)

    def module_ready(self):
        """
        Check if the logic component is ready for use.

        Returns:
            bool: True if the component is ready, False otherwise
        """
        # if indices are empty, the module is not ready and should be set up
        return self.concepts is not None

    def detect(
        self,
        latent_activations,
        attention_mask,
    ):
        """
        Identifies and adds new concepts from model outputs. And activates rules based on the new concepts.

        Args:
            latent_activations (torch.Tensor): Latent activations from the SAE model (batch, sequence_length, latent_dim)
            attention_mask (torch.Tensor): Attention mask for the input sequence (batch, sequence_length)
        Returns:
            torch.Tensor: Mask of activated concepts (shape: (batch, sequence_length, num_concepts))
        """

        if self.concepts is None:
            raise ValueError(
                "Concepts have not been set up. Please call 'set_up()' with a Concepts object before using this method."
            )
        latent_activations = latent_activations.cpu()

        # only keep the top k activations
        top_activations, top_output_indices = torch.topk(latent_activations, self.config.detection_top_k_output, dim=-1)  # top_activations shape is (batch, sequence_length, detection_top_k_output), (batch, sequence_length, detection_top_k_output)
        latent_activations = torch.zeros_like(latent_activations, dtype=torch.float).scatter(-1, top_output_indices, top_activations)  # shape is (batch, sequence_length, latent_dim)

        # if self.concepts.search_concept_type == 'sentence':
        #     sum_activations = attention_mask.sum(dim=1).to(latent_activations.device)  # (batch, sequence_length), sum of attention mask across the sequence length
        #     seq_latent_activations = latent_activations.sum(dim=1)  # (batch, latent_dim)
        #     latent_activations = seq_latent_activations / sum_activations.clamp(min=1e-6).unsqueeze(-1)  # (batch, latent_dim), normalize by the number of tokens in the sequence to avoid division by zero
        # else:
        # we try to just take the mean
        # latent_activations = latent_activations.view(
        #     b_size * seq_len, latent_dim
        # )  # Reshape to (batch * sequence_length, latent_dim)

        # In your detection function:
        concept_predictions = self.concepts.forward(latent_activations, self.concept_names)  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)        
        
        # apply attention mask to the concept probabilities
        activate_concept_prob = concept_predictions * attention_mask.unsqueeze(-1).unsqueeze(-1).to(concept_predictions.device)  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)

        # convert probabilities to binary mask
        activate_concept_mask = (activate_concept_prob.sum(dim=-1) > 0).int()  # shape is (batch, sequence_length, num_concepts)

        if (
            not self.config.detection_allow_multi
            and activate_concept_prob.any()
            and activate_concept_mask.shape[2] > 1
        ):
            activate_concept_mask, activate_concept_prob = _remove_multi_activations(
                activate_concept_mask, activate_concept_prob
            )

        if self.activate_concept_mask.numel() == 0:
            # First call, just use the current tensors directly
            self.activate_concept_mask = activate_concept_mask
            self.activate_concept_confidence = concept_predictions
            self.attention_mask = attention_mask
        else:
            # Check if batch size has changed
            if self.activate_concept_mask.shape[0] != activate_concept_mask.shape[0]:
                raise ValueError(
                    f"Batch size has changed from {self.activate_concept_mask.shape[0]} to {activate_concept_mask.shape[0]}. Please use same batch size for all calls to detect_concepts. or reset the module."
                )
            else:
                # Normal case, concatenate along sequence dimension
                self.activate_concept_mask = torch.cat(
                    (self.activate_concept_mask, activate_concept_mask), dim=1
                )
                self.activate_concept_confidence = torch.cat(
                    (self.activate_concept_confidence, concept_predictions), dim=1
                )
                self.attention_mask = torch.cat(
                    (self.attention_mask, attention_mask), dim=1
                )
                
    # def detect_concepts(
    #     self,
    #     latent_activations,
    #     attention_mask,
    # ):
    #     """
    #     Identifies and adds new concepts from model outputs. And activates rules based on the new concepts.

    #     Args:
    #         latent_activations (torch.Tensor): Latent activations from the SAE model (batch, sequence_length, latent_dim)
    #         attention_mask (torch.Tensor): Attention mask for the input sequence (batch, sequence_length)
    #     Returns:
    #         torch.Tensor: Mask of activated concepts (shape: (batch, sequence_length, num_concepts))
    #         list[str]: List of concept names
    #     """
    #     if self.concepts is None:
    #         raise ValueError(
    #             "Concepts have not been set up. Please call 'set_up()' with a Concepts object before using this method."
    #         )
    #     concept_tensor, _ = (
    #         self.concepts.get_concept_tensor()
    #     )  # shape is (num_concepts, steering_top_k_rule), indices of the concepts in the latent space
    #     if self.config.concept_dictionary_top_k_concepts > concept_tensor.shape[1]:
    #         raise ValueError(
    #             f"Top k concepts ({self.config.concept_dictionary_top_k_concepts}) for is greater than the number of indices available ({concept_tensor.shape[1]}). Please check the concept indices."
    #         )

    #     latent_activations = (
    #         latent_activations.cpu()
    #     )  # shape is (batch, sequence_length, latent_dim)
    #     top_activations, top_output_indices = torch.topk(
    #         latent_activations, self.config.detection_top_k_output, dim=-1
    #     )  # top_activations shape is (batch, sequence_length, detection_top_k_output)

    #     # shape is (1, 1, 1, num_concepts, concept_dictionary_top_k_concepts)
    #     concept_dictionary_top_k_concepts_indices = (
    #         concept_tensor[:, :self.config.concept_dictionary_top_k_concepts]
    #         .unsqueeze(0)
    #         .unsqueeze(0)
    #         .unsqueeze(0)
    #         .cuda()
    #     )

    #     # Shape (batch, sequence_length, detection_top_k_output, 1, 1)
    #     reshaped_output_indices = top_output_indices.unsqueeze(-1).unsqueeze(-1).cuda()

    #     # Shape: (batch, sequence_length, detection_top_k_output, 1, 1)
    #     reshaped_activation = top_activations.unsqueeze(-1).unsqueeze(-1).cuda()

    #     # Check which output indices match with which concept indices
    #     # Result shape: (batch, sequence_length, detection_top_k_output, num_concepts, concept_dictionary_top_k_concepts)
    #     matches = reshaped_output_indices == concept_dictionary_top_k_concepts_indices

    #     # shape is (batch, sequence_length, detection_top_k_output, num_concepts, concept_dictionary_top_k_concepts)
    #     concept_activations = reshaped_activation * matches.float()

    #     # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
    #     activations = concept_activations.sum(dim=2)

    #     # add weighted activations based on the index position
    #     weights = get_weight_vector(
    #         self.config.concept_dictionary_top_k_concepts,
    #         steering_weighting_function="log_decay",
    #         mean=1,
    #         std=None,
    #     ).to(activations.device)
    #     activations = (
    #         activations * weights
    #     )  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)

    #     # apply the detection threshold
    #     # activations[activations < detection_threshold] = 0.0      # Set values below threshold to 0

    #     # apply the activation threshold
    #     activate_concept_mask = (activations.sum(dim=-1) > 0).int()  # shape is (batch, sequence_length, num_concepts)

    #     # if multi activation is not allowed, we only consider the concept with the highest activation strength
    #     if (
    #         not self.config.detection_allow_multi
    #         and activations.any()
    #         and activate_concept_mask.shape[2] > 1
    #     ):
    #         activate_concept_mask, activations = _remove_multi_activations(
    #             activate_concept_mask, activations
    #         )

    #     if self.activate_concept_mask.numel() == 0:
    #         # First call, just use the current tensors directly
    #         self.activate_concept_mask = activate_concept_mask
    #         self.activate_concept_confidence = activations
    #         self.attention_mask = attention_mask
    #     else:
    #         # Check if batch size has changed
    #         if self.activate_concept_mask.shape[0] != activate_concept_mask.shape[0]:
    #             raise ValueError(
    #                 f"Batch size has changed from {self.activate_concept_mask.shape[0]} to {activate_concept_mask.shape[0]}. Please use same batch size for all calls to detect_concepts. or reset the module."
    #             )
    #         else:
    #             # Normal case, concatenate along sequence dimension
    #             # print(f'Concatenating activate_concept_mask and activate_concept_confidence with shapes {self.activate_concept_mask.shape} and {activate_concept_mask.shape}')
    #             self.activate_concept_mask = torch.cat(
    #                 (self.activate_concept_mask, activate_concept_mask), dim=1
    #             )
    #             self.activate_concept_confidence = torch.cat(
    #                 (self.activate_concept_confidence, activations), dim=1
    #             )
    #             self.attention_mask = torch.cat(
    #                 (self.attention_mask, attention_mask), dim=1
    #             )

    def get_concept_mask(self):
        return self.activate_concept_mask

    def get_concept_names(self):
        return self.concept_names

    def get_local_concepts(self) -> torch.Tensor:
        """
        We get the probabilities of the concepts based on the following:
        1. The strength of the activation of the concept indices in the latent space (confidence of activation)
        2. The position of the concept indice (represnting the importance of the indice for the concept). We can us different weighting functions to weight the indices.

        Args:
            weighting (str): Weighting function to use for the concept indices. Options are 'linear_decay', 'exponential_decay', 'uniform', 'linear', 'exponential'.
        Returns:
            torch.Tensor: Probabilities of the concepts (shape: (batch, sequence_length, num_concepts))
        """
        # activate_concept_confidence -- shape (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
        top_k = self.activate_concept_confidence.shape[-1]
        weight_vector = get_weight_vector(
            top_k, steering_weighting_function="log_decay", mean=1, std=None
        ).to(self.activate_concept_confidence.device)  # Shape: (top_k)
        weighted = (
            self.activate_concept_confidence @ weight_vector
        )  # shape is (batch, sequence_length, num_concepts)
        # apply thresholding
        weighted = torch.clamp(weighted - self.local_thresholds.to(self.activate_concept_confidence.device), min=0.0)
        return weighted # shape is (batch, sequence_length, num_concepts)
    
    def get_global_concepts(self) -> torch.Tensor:
        """
        Get the aggregated concept activations across the sequence length.

        Returns:
            torch.Tensor: Aggregated concept activations (shape: (batch, num_concepts))
        """
        top_k = self.activate_concept_confidence.shape[-1]
        weight_vector = get_weight_vector(
            top_k, steering_weighting_function="uniform", mean=1, std=None
        ).to(self.activate_concept_confidence.device)  # Shape: (top_k)
        weighted = (
            self.activate_concept_confidence @ weight_vector
        )  # shape is (batch, sequence_length, num_concepts)

        aggregated = weighted.sum(dim=1)  # shape is (batch, num_concepts)

        # normalize by length of sequence using the attention mask
        num_tokens = self.attention_mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1).to(self.activate_concept_confidence.device)  # shape is (batch, 1)
        aggregated = aggregated / num_tokens  # shape is (batch, num_concepts)
        # apply thresholding
        aggregated = torch.clamp(aggregated - self.global_thresholds.to(self.activate_concept_confidence.device), min=0.0)
        return aggregated # shape is (batch, num_concepts)

    def extract_concepts(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        Get the currently activated concepts and their positions in the sequence, and their batch ids.
        Returns:
            batch_ids (torch.Tensor): Batch ids (shape total number of activated concepts)
            seq_ids (torch.Tensor): Sequence ids (shape total number of activated concepts)
            concept_ids (torch.Tensor): Concept ids (shape total number of activated concepts)
            concept_names (List of str): Currently active concepts (shape total number of activated concepts)
            confidences (torch.Tensor): Confidence for each concept activation (shape total number of activated concepts)
        """
        if self.concepts is None:
            raise ValueError(
                "Concepts have not been set up. Please call 'set_up()' with a Concepts object before using this method."
            )
        activate_concept_confidence = self.get_local_concepts().unsqueeze(
            -1
        )  # shape is (batch, sequence_length, num_concepts)
        concept_mask = (
            self.activate_concept_mask
        )  # shape is (batch, sequence_length, num_concepts)
        batch_ids, seq_ids, concept_ids = torch.nonzero(concept_mask, as_tuple=True)
        confidences = activate_concept_confidence[
            batch_ids, seq_ids, concept_ids, :
        ].sum(-1)  # shape is (total number of activated concepts)
        # convert concept ids to concept
        concept_names = [self.concept_names[concept_id] for concept_id in concept_ids]
        # print(batch_ids, seq_ids, concept_ids, concept_names, confidences)
        return batch_ids, seq_ids, concept_ids, concept_names, confidences

    def get_activations(self) -> torch.Tensor:
        """
        Get the current activation matrix for the concepts.
        Returns:
            torch.Tensor: Activation matrix (shape: (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts))
        """
        return self.activate_concept_confidence

    def auto_thresholding(
        self,
        label_path: str,
        reset_cache: bool = False,
    ) -> None:
        """
        Recompute thresholds via the threshold manager and update local/global tensors.
        """
        if self.threshold_manager is None:
            raise ValueError("Threshold manager not initialised. Call set_up first.")

        thresholds = self.threshold_manager.auto_thresholding(
            detector=self,
            label_path=label_path,
            reset_cache=reset_cache,
        )
        self.local_thresholds, self.global_thresholds = thresholds.as_tuple()






def _remove_multi_activations(activate_concept_mask, activations):
    """
    Remove multi activations from the concept mask and activations.

    Args:
        activate_concept_mask (torch.Tensor): Concept activation mask (shape: (batch, sequence_length, num_concepts))
        activations (torch.Tensor): Concept activations (shape: (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts))
    Returns:
        torch.Tensor: Updated concept activation mask (shape: (batch, sequence_length, num_concepts))
        torch.Tensor: Updated concept activations (shape: (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts))
    """

    c_activations = activations.sum(
        dim=-1
    )  # shape is (batch, sequence_length, num_concepts)
    # print(f'removing multi activations')
    active_concepts = torch.argmax(
        c_activations, dim=-1
    )  # shape is (batch, sequence_length)
    # convert indices into a one-hot encoded sparse matrix
    _acm = F.one_hot(
        active_concepts, num_classes=activate_concept_mask.shape[2]
    ).float()  # shape is (batch, sequence_length, num_concepts)
    # we need to apply the mask to the active concept mask to avoid argmaxing on 0 activations
    activate_concept_mask = (
        _acm * activate_concept_mask
    )  # shape is (batch, sequence_length, num_concepts)
    # update the activations
    activations = (
        activations * activate_concept_mask.unsqueeze(-1)
    )  # shape is (batch, sequence_length, num_concepts, concept_dictionary_top_k_concepts)
    return activate_concept_mask, activations