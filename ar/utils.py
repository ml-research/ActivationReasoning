import os
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    classification_report,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

def separate_duplicate_indices(
    top_indices_pos,
    top_indices_neg,
    remove_negatives=True,
    mutually_exclusive_classes=True,
):
    """
    Separate indices into three clear categories: unique, cross-concept duplicates, and negative overlaps.

    Args:
        top_indices_pos (torch.Tensor): Tensor of top indices for each concept (shape: (num_concepts, top_k))
        top_indices_neg (torch.Tensor): Tensor of negative indices for each concept (shape: (num_concepts, top_k))

    Returns:
        tuple: (unique_mask, cross_concept_mask, negative_overlap_mask)
            All masks have shape (num_concepts, top_k)
    """
    # Initialize masks
    negative_overlap_mask = torch.zeros_like(top_indices_pos, dtype=torch.bool)
    cross_concept_duplicate_mask = torch.zeros_like(top_indices_pos, dtype=torch.bool)

    if remove_negatives:
        # Step 1: Create mask for indices that overlap with negatives
        negative_overlap_mask = []
        for i in range(top_indices_pos.shape[0]):
            pos_indices = top_indices_pos[i]
            neg_indices = top_indices_neg[i]
            overlap_mask = torch.isin(pos_indices, neg_indices)
            negative_overlap_mask.append(overlap_mask)
        negative_overlap_mask = torch.stack(
            negative_overlap_mask, dim=0
        )  # shape: (num_concepts, top_k)

    if mutually_exclusive_classes:
        # Step 2: Find cross-concept duplicates among clean indices
        cross_concept_duplicate_mask = []
        for i in range(top_indices_pos.shape[0]):
            current_row = top_indices_pos[i]
            # Get all other rows
            other_rows = torch.cat(
                [top_indices_pos[:i], top_indices_pos[i + 1 :]], dim=0
            )

            # For each element in current row, check if it appears in other rows
            # Ignore -1 values (they represent filtered out indices)
            duplicate_mask = torch.zeros_like(current_row, dtype=torch.bool)
            for j, idx in enumerate(current_row):
                if idx != -1:  # Only check non-filtered indices
                    duplicate_mask[j] = torch.any(other_rows == idx)

            cross_concept_duplicate_mask.append(duplicate_mask)

        cross_concept_duplicate_mask = torch.stack(
            cross_concept_duplicate_mask, dim=0
        )  # shape: (num_concepts, top_k)

    # Step 4: Create the three final masks
    # Unique: not in negatives AND not shared across concepts
    unique_mask = ~negative_overlap_mask & ~cross_concept_duplicate_mask

    # Cross-concept shared: not in negatives BUT shared across concepts
    cross_concept_mask = ~negative_overlap_mask & cross_concept_duplicate_mask

    # Negative overlap: already computed
    # negative_overlap_mask = negative_overlap_mask (already computed)

    return unique_mask, cross_concept_mask, negative_overlap_mask


def intersect_indices_for_colors(activated_indices: dict):
    """
    Find common indices for each color in the input
    activated_indices: dict for each color with indices in a tensor of shape (n, 192), where n is the number of occurences of the color in the inputs and 192 is top_indices
    """
    common_indices = defaultdict(lambda: np.array([], dtype=np.int64))
    for color, indices in activated_indices.items():
        # iterate over batch
        for i in range(indices.shape[0]):
            if i == 0:
                common_indices[color] = indices[i].numpy()
            else:
                common_indices[color] = np.intersect1d(
                    common_indices[color], indices[i].numpy()
                )
    return common_indices


COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "ORANGE": "\033[38;5;208m",
    "PURPLE": "\033[38;5;165m",
    "LIME": "\033[38;5;118m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "OKBLUE": "\033[94m",
}


BACKGROUND_COLORS = {
    "RED": "\033[41m",
    "GREEN": "\033[42m",
    "YELLOW": "\033[43m",
    "BLUE": "\033[44m",
    "MAGENTA": "\033[45m",
}


BOLD = "\033[1m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"


def get_weight_vector(
    dim_size, steering_weighting_function="uniform", mean=1.0, std=None
):
    tensor_indices = torch.arange(dim_size)  # Shape: (dim_size)
    # Calculate weights based on the current weighting function
    if steering_weighting_function == "linear_decay":
        weights = 1 - (tensor_indices.float() / max(dim_size - 1, 1))

    elif steering_weighting_function == "exponential_decay":
        decay_rate = 1.0
        weights = torch.exp(-tensor_indices.float() / decay_rate)

    elif steering_weighting_function == "inverse_position":
        weights = 1.0 / (tensor_indices.float() + 1.0)

    elif steering_weighting_function == "softmax_based":
        temperature = 2.0
        position_scores = -tensor_indices.float() / temperature
        weights = torch.softmax(position_scores, dim=0)

    elif steering_weighting_function == "sigmoid_decay":
        steepness = 2.0
        mid_point = dim_size / 2.0
        weights = 1.0 / (
            1.0 + torch.exp((tensor_indices.float() - mid_point) * steepness)
        )

    elif steering_weighting_function == "power_law_decay":
        exponent = 2.0
        weights = 1.0 / ((tensor_indices.float() + 1.0) ** exponent)

    elif steering_weighting_function == "cosine_decay":
        weights = torch.cos(tensor_indices.float() * torch.pi / (2.0 * dim_size))

    elif steering_weighting_function == "log_decay":
        weights = 1.0 - (
            torch.log(tensor_indices.float() + 1.0)
            / torch.log(torch.tensor(dim_size + 1.0))
        )

    elif steering_weighting_function == "uniform":
        return torch.ones_like(tensor_indices, dtype=torch.float) * mean
    else:
        raise ValueError(f"Invalid weighting function: {steering_weighting_function}")

    # Rescale the weights to have mean and std
    # if weights.numel() == 1 or weights.std() == 0:
    #     # Avoid division by zero: just set to mean
    #     weights = torch.ones_like(weights) * mean
    # else:
    #     if std is None:
    #         std = weights.std()
    #     weights = (weights - weights.mean()) / weights.std() * std + mean
    if std is not None:
        # Scale to desired std but preserve non-zero pattern
        weights = weights / weights.mean() * mean  # Scale to desired mean

        if weights.std() > 0:  # Only adjust std if non-constant
            current_std = weights.std()
            weights = mean + (weights - mean) * (std / current_std)
    else:
        # Just scale to the desired mean
        weights = weights / weights.mean() * mean

    # Clip the weights to be non-negative
    weights = torch.clamp(weights, min=0.0)

    return weights


def weighting(steering_direction, steering_weighting_function="uniform"):
    """
    Generate weights for each position in the dimension based on the weighting function.
        1. The weight decay based on the position in the dimension.
        2. Weights are normalized to sum to 1 along the dimension.
    Args:
        steering_direction (torch.Tensor): The steering direction tensor (Shape: (batch, num_rules, steering_top_k_rule, llm_hidden_dim))
        steering_weighting_function (str): Type of weighting function to use. Options include:
    Returns:
        torch.Tensor: Weights for each position in the dimension (Shape: (batch, num_rules, steering_top_k_rule, 1))
    """
    dim_size = steering_direction.shape[2]
    tensor_indices = torch.arange(
        dim_size, device=steering_direction.device
    )  # Shape: (dim_size)
    # Normalize the steering weights

    activation_strength = steering_direction.norm(
        p=2, dim=-1, keepdim=True
    )  # Shape: (batch, num_rules, steering_top_k_rule, 1)
    # Calculate sum per rule
    rule_sums = activation_strength.sum(
        dim=-2, keepdim=True
    )  # Shape: (batch, num_rules, 1, 1)
    # Create mask for non-zero sums to avoid division by zero
    mask = rule_sums > 0  # Shape: (batch, num_rules, 1, 1)
    mask = mask.expand_as(
        activation_strength
    )  # Shape: (batch, num_rules, steering_top_k_rule, 1)

    # Calculate weights based on the current weighting function
    if steering_weighting_function == "activations":
        # Normalize only where sum is non-zero
        normalized_strengths = torch.zeros_like(
            activation_strength, device=steering_direction.device
        )
        normalized_strengths = torch.where(
            mask, activation_strength / rule_sums.clamp(min=1e-8), normalized_strengths
        )
        # Reshape to match expected output shape
        return normalized_strengths  # Shape: (batch, num_rules, steering_top_k_rule, 1)

    # Get the weight vector based on the weighting function
    weights = get_weight_vector(
        dim_size, steering_weighting_function, mean=1 / dim_size
    )

    weights = weights.cuda()
    # normalize the weights
    weights = weights.view(1, 1, dim_size, 1)  # Shape: (1, 1, dim_size, 1)
    # expand batch and num_rules dimensions
    weights = weights.expand(
        steering_direction.shape[0], steering_direction.shape[1], dim_size, 1
    )  # Shape: (batch, num_rules, dim_size, 1)
    # apply the mask to the weights
    weights = torch.where(
        mask, weights, torch.zeros_like(weights)
    )  # Shape: (batch, num_rules, dim_size, 1)

    return weights


def safety_plot(y_true, meta):
    y_true = np.array([a.item() for a in y_true])
    y_true = y_true.reshape(-1)

    y_scores = np.array([m["score"] for m in meta])
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    all_acc = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = (y_true == y_pred).mean()
        all_acc.append(accuracy)
    auprc = auc(recall, precision)

    # Find best threshold for F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(all_acc)
    best_threshold = (
        thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    )
    best_f1 = f1_scores[best_threshold_idx]
    best_precision = precision[best_threshold_idx]
    best_recall = recall[best_threshold_idx]
    best_accuracy = (y_true == (y_scores >= best_threshold).astype(int)).mean()
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_scores[i] >= best_threshold:
            TP += 1
        elif y_true[i] == 0 and y_scores[i] < best_threshold:
            TN += 1
        elif y_true[i] == 0 and y_scores[i] >= best_threshold:
            FP += 1
        elif y_true[i] == 1 and y_scores[i] < best_threshold:
            FN += 1
    # Calculate binary predictions using the best threshold
    y_pred = (y_scores >= best_threshold).astype(int)

    # Plot the precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.scatter(
        recall[best_threshold_idx],
        precision[best_threshold_idx],
        color="red",
        marker="o",
        label=f"Best threshold = {best_threshold:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print metrics
    print(
        f"AUPRC: {auprc:.3f}, Average Precision Score: {average_precision_score(y_true, y_scores):.3f}"
    )
    print(
        f"Best Threshold: {best_threshold:.3f} (Accuracy: {best_accuracy:.3f}, F1: {best_f1:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f})"
    )
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print("\nClassification Report at Best Threshold:")
    print(classification_report(y_true, y_pred))


def safe_load_tensor(path: str) -> torch.Tensor:
    """Modified method to safely load potentially sparse activations"""
    # Load the tensor - could be sparse or dense
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SAE latents not found at {path}. Please run find_concept_indicies() to extract them."
        )
    tensor_path = os.path.join(path)
    activation = torch.load(tensor_path)

    # Check if it's a sparse tensor and convert to dense for compatibility
    if activation.is_sparse:
        activation = activation.to_dense()

    return activation


def remove_duplicates_and_preserve_order(input_list):
    """
    Removes duplicate elements from a list while maintaining the original order.

    This function iterates through the input list and uses a set to keep track of
    elements that have already been encountered. It builds a new list containing
    only the unique elements in the order they first appeared.

    Args:
        input_list: The list from which to remove duplicates.

    Returns:
        A new list with duplicates removed, preserving the original order.
    """

    # Use a set for efficient O(1) average time complexity for lookups.
    # This set will store the elements we have already seen.
    seen = set()

    # This will be the new list containing only unique elements.
    result_list = []

    # Iterate through each element in the original list.
    for item in input_list:
        # Check if the current element is not in our set of seen elements.
        if item not in seen:
            # If it's a new element, add it to the seen set and to our result list.
            seen.add(item)
            result_list.append(item)

    return result_list


def remove_consecutive_duplicates(input_list):
    """
    Removes consecutive duplicate elements from a list while preserving the order.

    This function iterates through the list, appending an element to the result
    only if it is different from the previous element that was appended.

    Args:
        input_list: The list from which to remove consecutive duplicates.

    Returns:
        A new list with consecutive duplicates removed.
    """
    # Handle the edge case of an empty list
    if not input_list:
        return []

    # Start with the first element, as it's never a consecutive duplicate of a non-existent item.
    result_list = [input_list[0]]

    # Iterate through the list starting from the second element.
    for i in range(1, len(input_list)):
        # Compare the current element with the last element in our result list.
        if (
            input_list[i] != result_list[-1]
            and input_list[i] != "Answer: True"
            and input_list[i] != "Answer: False"
            and input_list[i] != "Answer: Uncertain"
        ):
            result_list.append(input_list[i])

    return result_list


def replace_or_with_xor(tokens):
    """
    Finds occurrences of "but not both", changes the preceding "or" to "xor",
    and removes the "but not both" token.

    This function iterates through a list of tokens. For each instance of the
    string "but not both", it searches backwards to find the nearest preceding
    "or" and replaces it with "xor". It then removes the "but not both" token.

    Args:
        tokens: A list of strings.

    Returns:
        A new list with the appropriate "or" tokens replaced by "xor" and
        "but not both" tokens removed.
    """
    # Create a copy of the list to avoid modifying the original
    new_tokens = list(tokens)

    # Find the indices of all occurrences of "but not both"
    bnb_indices = [i for i, token in enumerate(new_tokens) if token == "but not both"]

    # Iterate backwards through the indices to safely remove items
    for index in sorted(bnb_indices, reverse=True):
        # Search backwards from the current index to find the closest "or"
        # The range goes from (index - 1) down to 0.
        for i in range(index - 1, -1, -1):
            if new_tokens[i] == "or":
                # Replace the found "or" with "xor"
                new_tokens[i] = "xor"
                # Break the inner loop once the closest "or" is found and replaced
                break

        # Remove the "but not both" token
        del new_tokens[index]

    return new_tokens


def implicit_is(tokens: list) -> list:
    if (
        len(tokens) >= 2
        and tokens[0][0].isupper()
        and tokens[1] not in {"is", "do", "have"}
    ):
        return [tokens[0], "is"] + tokens[1:]
    else:
        return tokens
